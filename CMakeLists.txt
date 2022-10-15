cmake_minimum_required(VERSION 3.20)

# Avoid warning about DOWNLOAD_EXTRACT_TIMESTAMP in CMake 3.24:
# Fix found at https://github.com/ethereum/solidity/pull/13429
if (CMAKE_VERSION VERSION_GREATER_EQUAL "3.24.0")
  cmake_policy(SET CMP0135 NEW)
endif()

include(FetchContent)
include(CMakePrintHelpers)

project(cuda-battery LANGUAGES CUDA CXX)
option(GPU "GPU" ON)
option(BATTERY_BUILD_TESTS "BATTERY_BUILD_TESTS" OFF)

add_link_options ($<DEVICE_LINK:-arch=sm_75>)

if(NOT DEFINED CMAKE_CXX_STANDARD)
    set(CMAKE_CXX_STANDARD 20)
    set(CMAKE_CXX_STANDARD_REQUIRED ON)
endif()

if(NOT DEFINED CMAKE_CUDA_STANDARD)
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
endif()

# Preparing the library

add_library(cuda_battery INTERFACE)
target_include_directories(cuda_battery INTERFACE include)

if(GPU)
  set_property(TARGET cuda_battery PROPERTY CUDA_ARCHITECTURES 75)
  set_property(TARGET cuda_battery PROPERTY CUDA_SEPARABLE_COMPILATION ON)
endif()

# --frounding-math is necessary with GCC because `#pragma STDC FENV_ACCESS ON` does not work.
# This option will help in making the compiler take into account rounding modes, but bugs persist so floating-point arithmetic on CPUs might not always be sound right now...
# C.f. https://gcc.gnu.org/bugzilla/show_bug.cgi?id=34678
IF((CMAKE_CXX_COMPILER_ID STREQUAL "GNU") OR (CMAKE_CXX_COMPILER_ID STREQUAL "Clang"))
  add_compile_options(-frounding-math)
ENDIF()

target_compile_options(cuda_battery INTERFACE
    $<$<COMPILE_LANGUAGE:CUDA>:
        --expt-relaxed-constexpr
        -rdc=true
        --compiler-options -frounding-math>
    $<$<CONFIG:Debug>:-g -DDEBUG>
    $<$<CONFIG:LDebug>:-g -DTRACE>
)

if(BATTERY_BUILD_TESTS)

# Google Test dependency

FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/609281088cfefc76f9d0ce82e1ff6c30cc3591e5.zip
)
# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

enable_testing()
include(GoogleTest)

add_executable(utility_test_cpu tests/utility_test_cpu.cpp)
target_link_libraries(utility_test_cpu cuda_battery gtest_main)
gtest_discover_tests(utility_test_cpu)

add_executable(string_test tests/string_test.cpp)
target_link_libraries(string_test cuda_battery gtest_main)
gtest_discover_tests(string_test)

add_executable(variant_test tests/variant_test.cpp)
target_link_libraries(variant_test cuda_battery gtest_main)
gtest_discover_tests(variant_test)

add_executable(vector_test tests/vector_test.cpp)
target_link_libraries(vector_test cuda_battery gtest_main)
gtest_discover_tests(vector_test)

add_executable(unique_ptr_test tests/unique_ptr_test.cpp)
target_link_libraries(unique_ptr_test cuda_battery gtest_main)
gtest_discover_tests(unique_ptr_test)

add_executable(shared_ptr_test tests/shared_ptr_test.cpp)
target_link_libraries(shared_ptr_test cuda_battery gtest_main)
gtest_discover_tests(shared_ptr_test)

add_executable(bitset_test tests/bitset_test.cpp)
target_link_libraries(bitset_test cuda_battery gtest_main)
gtest_discover_tests(bitset_test)

if(GPU)
  set_source_files_properties(tests/utility_test_cpu_gpu.cpp PROPERTIES LANGUAGE CUDA)
  add_executable(utility_test_cpu_gpu tests/utility_test_cpu_gpu.cpp)
  set_property(TARGET utility_test_cpu_gpu PROPERTY CUDA_SEPARABLE_COMPILATION ON)
  set_property(TARGET utility_test_cpu_gpu PROPERTY CUDA_ARCHITECTURES 75)
  target_compile_options(utility_test_cpu_gpu PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>: --expt-relaxed-constexpr -rdc=true --compiler-options -frounding-math>
    $<$<CONFIG:Debug>:-g -DDEBUG>
    $<$<CONFIG:LDebug>:-g -DTRACE>)
  target_link_libraries(utility_test_cpu_gpu cuda_battery)
  add_test(NAME utility_test_cpu_gpu COMMAND compute-sanitizer --error-exitcode 1 $<TARGET_FILE:utility_test_cpu_gpu>)
endif()

endif()

# Documentation

find_package(Doxygen REQUIRED doxygen)

# Options based on those of Halide: https://github.com/halide/Halide/blob/feb81a2a0f7bc39cfa499e4e1a7dfedbc5250aaa/doc/CMakeLists.txt
set(DOXYGEN_PROJECT_NAME "Cuda battery library")
set(DOXYGEN_BUILTIN_STL_SUPPORT YES)
set(DOXYGEN_CASE_SENSE_NAMES NO)
set(DOXYGEN_CLASS_DIAGRAMS NO)
set(DOXYGEN_DISTRIBUTE_GROUP_DOC YES)
set(DOXYGEN_EXTRACT_ALL YES)
set(DOXYGEN_FILE_PATTERNS *.hpp)
set(DOXYGEN_GENERATE_TREEVIEW YES)
set(DOXYGEN_HIDE_IN_BODY_DOCS YES)
set(DOXYGEN_QUIET YES)
set(DOXYGEN_RECURSIVE YES)
set(DOXYGEN_REFERENCED_BY_RELATION YES)
set(DOXYGEN_REFERENCES_RELATION YES)
set(DOXYGEN_SORT_BY_SCOPE_NAME YES)
set(DOXYGEN_SORT_MEMBER_DOCS NO)
set(DOXYGEN_SOURCE_BROWSER NO)
set(DOXYGEN_STRIP_CODE_COMMENTS NO)
set(DOXYGEN_USE_MATHJAX YES)
set(DOXYGEN_EXTRA_PACKAGES stmaryrd)

doxygen_add_docs(doc_cuda_battery
                 "include/"
                 ALL
                 COMMENT "Generate HTML documentation")
