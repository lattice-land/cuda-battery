// Copyright 2022 Pierre Talbot

#include <vector>
#include "battery/bitset.hpp"
#include "battery/memory.hpp"
#include "battery/allocator.hpp"
#include "battery/utility.hpp"

template<class T>
CUDA const char* name_of_type() {
  if constexpr(std::is_same_v<T, char>) { return "char"; }
  else if constexpr(std::is_same_v<T, short>) { return "short"; }
  else if constexpr(std::is_same_v<T, int>) { return "int"; }
  else if constexpr(std::is_same_v<T, long int>) { return "long int"; }
  else if constexpr(std::is_same_v<T, long long int>) { return "long long int"; }
  else if constexpr(std::is_same_v<T, unsigned char>) { return "unsigned char"; }
  else if constexpr(std::is_same_v<T, unsigned short>) { return "unsigned short"; }
  else if constexpr(std::is_same_v<T, unsigned int>) { return "unsigned int"; }
  else if constexpr(std::is_same_v<T, unsigned long int>) { return "unsigned long int"; }
  else if constexpr(std::is_same_v<T, unsigned long long int>) { return "unsigned long long int"; }
  else if constexpr(std::is_same_v<T, float>) { return "float"; }
  else if constexpr(std::is_same_v<T, double>) { return "double"; }
  else {
    static_assert(std::is_same_v<T, char>, "Type unsupported in name_of_type.");
    return "unknown";
  }
}

enum UtilityOperation {
  RU_CAST,
  RD_CAST,
  POPCOUNT,
  COUNTL_ZERO,
  COUNTL_ONE,
  COUNTR_ZERO,
  COUNTR_ONE
};

CUDA const char* name_of_utility_op(UtilityOperation op) {
  switch(op) {
    case RU_CAST: return "ru_cast";
    case RD_CAST: return "rd_cast";
    case POPCOUNT: return "popcount";
    case COUNTL_ZERO: return "countl_zero";
    case COUNTL_ONE: return "countl_one";
    case COUNTR_ZERO: return "countr_zero";
    case COUNTR_ONE: return "countr_one";
    default: assert(0); return "unknown";
  }
}

template<class R, class T>
CUDA R run_utility_op(T input, UtilityOperation op) {
  if constexpr(std::is_unsigned_v<T>) {
    switch(op) {
      case POPCOUNT: return battery::popcount(input);
      case COUNTL_ZERO: return battery::countl_zero(input);
      case COUNTL_ONE: return battery::countl_one(input);
      case COUNTR_ZERO: return battery::countr_zero(input);
      case COUNTR_ONE: return battery::countr_one(input);
      default: break;
    }
  }
  switch(op) {
    case RU_CAST: return battery::ru_cast<R>(input);
    case RD_CAST: return battery::rd_cast<R>(input);
    default: assert(0); return R{};
  }
}

template<class R, class T>
CUDA_GLOBAL void run_utility_op_gpu(R* res, T input, UtilityOperation op) {
  *res = run_utility_op<R>(input, op);
}

template<class T, class R>
void analyse_test_result(T input, R expect, R cpu_result, R gpu_result, UtilityOperation op) {
  if(cpu_result != expect || gpu_result != expect) {
    const char* op_name = name_of_utility_op(op);
    printf("%s %s(%s)\n", name_of_type<R>(), op_name, name_of_type<T>());
    printf("%s(", op_name);
    ::battery::print(input);
    printf(") == ");
    ::battery::print(expect);
    printf("\n%s(", op_name);
    ::battery::print(input);
    printf(")__CPU == ");
    ::battery::print(cpu_result);
    printf("\n%s(", op_name);
    ::battery::print(input);
    printf(")__GPU == ");
    ::battery::print(gpu_result);
    printf("\n");
    exit(EXIT_FAILURE);
  }
}

template<class T, class R>
void test_utility_ops(std::vector<T> inputs, std::vector<R> outputs, UtilityOperation op) {
  assert(inputs.size() == outputs.size());
  // Add testing for bottom and top elements for the casting operations.
  if(op == RU_CAST || op == RD_CAST) {
    if constexpr(std::is_signed_v<T>) {
      inputs.push_back(battery::limits<T>::bot());
      outputs.push_back(battery::limits<R>::bot());
    }
    inputs.push_back(battery::limits<T>::top());
    outputs.push_back(battery::limits<R>::top());
  }
  battery::managed_allocator managed_allocator;
  for(int i = 0; i < inputs.size(); ++i) {
    R cpu_result = run_utility_op<R>(inputs[i], op);
    R* gpu_result = new(managed_allocator) R();
    run_utility_op_gpu<R><<<1, 1>>>(gpu_result, inputs[i], op);
    CUDAEX(cudaDeviceSynchronize());
    analyse_test_result(inputs[i], outputs[i], cpu_result, *gpu_result, op);
  }
}

template<class From>
void limits_tests_round_cast(UtilityOperation op) {
  test_utility_ops<From, float>({0, 1, 100}, {0, 1, 100}, op);
  test_utility_ops<From, double>({0, 1, 100}, {0, 1, 100}, op);
  test_utility_ops<From, char>({0, 1, 100}, {0, 1, 100}, op);
  test_utility_ops<From, unsigned char>({0, 1, 100}, {0, 1, 100}, op);
  test_utility_ops<From, short>({0, 1, 100}, {0, 1, 100}, op);
  test_utility_ops<From, unsigned short>({0, 1, 100}, {0, 1, 100}, op);
  test_utility_ops<From, int>({0, 1, 100}, {0, 1, 100}, op);
  test_utility_ops<From, long long int>({0, 1, 100}, {0, 1, 100}, op);
  test_utility_ops<From, unsigned int>({0, 1, 100}, {0, 1, 100}, op);
  test_utility_ops<From, unsigned long long int>({0, 1, 100}, {0, 1, 100}, op);
}

template<class From>
void limits_tests_round_cast() {
  limits_tests_round_cast<From>(RU_CAST);
  limits_tests_round_cast<From>(RD_CAST);
}

void limits_tests_round_cast_all() {
  limits_tests_round_cast<float>();
  limits_tests_round_cast<double>();
  limits_tests_round_cast<char>();
  limits_tests_round_cast<unsigned char>();
  limits_tests_round_cast<short>();
  limits_tests_round_cast<unsigned short>();
  limits_tests_round_cast<int>();
  limits_tests_round_cast<long long int>();
  limits_tests_round_cast<unsigned int>();
  limits_tests_round_cast<unsigned long long int>();
}

constexpr int int_not_float = 100000001;
constexpr long long int long_long_not_double = 100000000000000001;

template<class I>
void cast_I_to_float_double() {
  static_assert(sizeof(I) >= sizeof(int));
  test_utility_ops<I, float>({int_not_float}, {100000008.0f}, RU_CAST);
  test_utility_ops<I, float>({int_not_float}, {100000000.0f}, RD_CAST);

  if constexpr(std::is_signed_v<I>) {
    test_utility_ops<I, float>({-int_not_float}, {-100000000.0f}, RU_CAST);
    test_utility_ops<I, float>({-int_not_float}, {-100000008.0f}, RD_CAST);
  }

  if constexpr(sizeof(I) >= sizeof(long long int)) {
    test_utility_ops<I, float>({long_long_not_double}, {100000007020609536.0f}, RU_CAST);
    test_utility_ops<I, float>({long_long_not_double}, {99999998430674944.0f}, RD_CAST);
    test_utility_ops<I, double>({int_not_float, long_long_not_double}, {100000001.0, 100000000000000016.0}, RU_CAST);
    test_utility_ops<I, double>({int_not_float, long_long_not_double}, {100000001.0, 100000000000000000.0}, RD_CAST);

    if constexpr(std::is_signed_v<I>) {
      test_utility_ops<I, float>({-long_long_not_double}, {-99999998430674944.0f}, RU_CAST);
      test_utility_ops<I, float>({-long_long_not_double}, {-100000007020609536.0f}, RD_CAST);
      test_utility_ops<I, double>({-int_not_float, -long_long_not_double}, {-100000001.0, -100000000000000000.0}, RU_CAST);
      test_utility_ops<I, double>({-int_not_float, -long_long_not_double}, {-100000001.0, -100000000000000016.0}, RD_CAST);
    }
  }
}

template<class I>
void cast_F_to_integers() {
  if constexpr(std::is_signed_v<I>) {
    test_utility_ops<float, I>({-10.0f, -10.5f}, {-10, -10}, RU_CAST);
    test_utility_ops<double, I>({-10.0, -10.5}, {-10, -10}, RU_CAST);
    test_utility_ops<float, I>({-10.0f, -10.5f}, {-10, -11}, RD_CAST);
    test_utility_ops<double, I>({-10.0, -10.5}, {-10, -11}, RD_CAST);
  }
  test_utility_ops<float, I>({0.f, 10.f, 10.5f}, {0, 10, 11}, RU_CAST);
  test_utility_ops<double, I>({0., 10., 10.5}, {0, 10, 11}, RU_CAST);
  test_utility_ops<float, I>({0.f, 10.f, 10.5f}, {0, 10, 10}, RD_CAST);
  test_utility_ops<double, I>({0., 10., 10.5}, {0, 10, 10}, RD_CAST);
}

void test_all_casts() {
  limits_tests_round_cast_all();

  cast_I_to_float_double<int>();
  cast_I_to_float_double<unsigned int>();
  cast_I_to_float_double<long long int>();
  cast_I_to_float_double<unsigned long long int>();

  cast_F_to_integers<char>();
  cast_F_to_integers<unsigned char>();
  cast_F_to_integers<short>();
  cast_F_to_integers<unsigned short>();
  cast_F_to_integers<int>();
  cast_F_to_integers<unsigned int>();
  cast_F_to_integers<long long int>();
  cast_F_to_integers<unsigned long long int>();

  test_utility_ops<double, float>({100000000000000016.0}, {100000007020609536.0f}, RU_CAST);
  test_utility_ops<double, float>({100000000000000016.0}, {99999998430674944.0f}, RD_CAST);
}

template<class I>
void test_bitwise_operations() {
  constexpr I bits = sizeof(I) * CHAR_BIT;
  test_utility_ops<I, int>(
    {0, 1, ((I)1) << (bits-1), (I)1 | ((I)1 << (bits-1)),  battery::limits<I>::top()},
    {0, 1, 1,             2,                    bits},
    POPCOUNT);
  test_utility_ops<I, int>(
    {0,    1,      (I)1 << 2, (I)1 << (bits-1), (I)1 | ((I)1 << (bits-1)), battery::limits<I>::top()},
    {bits, bits-1, bits-3, 0,             0,                   0},
    COUNTL_ZERO);
  test_utility_ops<I, int>(
    {0,    1, (I)1 << 2, (I)1 << (bits-1), (I)1 | ((I)1 << (bits-1)), battery::limits<I>::top()},
    {bits, 0, 2,      bits-1,        0,                   0},
    COUNTR_ZERO);
  test_utility_ops<I, int>(
    {0, 1, (I)1 << 2, (I)1 << (bits-1), (I)1 | ((I)1 << (bits-1)), battery::limits<I>::top(), (I)1 << (bits-1) | (I)1 << (bits-2)},
    {0, 0, 0,      1,             1,                   bits,                      2},
    COUNTL_ONE);
  test_utility_ops<I, int>(
    {0, 1, (I)1 << 2, (I)1 << (bits-1), (I)1 | ((I)1 << (bits-1)), battery::limits<I>::top(), 1 | 2},
    {0, 1, 0,      0,             1,                   bits,                      2},
    COUNTR_ONE);
}

int main() {
  test_all_casts();
  test_bitwise_operations<unsigned char>();
  test_bitwise_operations<unsigned short>();
  test_bitwise_operations<unsigned int>();
  test_bitwise_operations<unsigned long long int>();
  printf("utility_test_cpu_gpu complete.\n");
  return 0;
}
