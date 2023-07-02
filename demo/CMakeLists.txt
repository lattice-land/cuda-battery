cmake_minimum_required(VERSION 3.20)

# Avoid warning about DOWNLOAD_EXTRACT_TIMESTAMP in CMake 3.24:
# Fix found at https://github.com/ethereum/solidity/pull/13429
if (CMAKE_VERSION VERSION_GREATER_EQUAL "3.24.0")
  cmake_policy(SET CMP0135 NEW)
endif()

include(FetchContent)
include(CMakePrintHelpers)

# Name of the project (you can change "demo")
project(demo LANGUAGES CUDA CXX)

# The tests are activated by default.
option(DEMO_BUILD_TESTS "DEMO_BUILD_TESTS" ON)

if(NOT DEFINED CMAKE_CXX_STANDARD)
    set(CMAKE_CXX_STANDARD 20)
    set(CMAKE_CXX_STANDARD_REQUIRED ON)
endif()

if(NOT DEFINED CMAKE_CUDA_STANDARD)
    set(CMAKE_CUDA_STANDARD 20)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
endif()

# Automatically download a dependency from Github in the project.
FetchContent_Declare(
  cuda_battery
  GIT_REPOSITORY "https://github.com/lattice-land/cuda-battery.git"
  GIT_TAG        develop
)
FetchContent_MakeAvailable(cuda_battery)

# Add any `.cpp` files that contains CUDA code here.
# If you want to keep `.cu` files, simply delete this command.
set_source_files_properties(
  src/demo.cpp
  PROPERTIES LANGUAGE CUDA)

# This is the main executable, add any additional `.cpp` files required to compile the project here, for instance:
# add_executable(demo src/demo.cpp src/config.cpp)
add_executable(demo src/demo.cpp)

set_property(TARGET demo PROPERTY CUDA_ARCHITECTURES native)

# Cross-compiling?
# If you want to compile for a particular architecture, you can use the following line instead:
# set_property(TARGET demo PROPERTY CUDA_ARCHITECTURES 75)

# Multiple CUDA files?
# If you have several `.cpp` files with CUDA code, you must activate separable compilation:
# set_property(TARGET demo APPEND PROPERTY CUDA_SEPARABLE_COMPILATION ON)

# We make the .hpp files of the library `cuda_battery` available to the project.
target_include_directories(demo PRIVATE include ${cuda_battery_SOURCE_DIR}/include)

# We add several compilation options to the executable.
target_compile_options(demo PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:
    --expt-relaxed-constexpr
    -diag-suppress 20011,20014 # Calling __host__ from __host__ __device__ kind of warnings. (generated due to templated functions working on both CPU/GPU).
    --compiler-options -frounding-math
    $<$<CONFIG:Debug>:-G>>
  $<$<CONFIG:Debug>:-g -DDEBUG>
)

if(DEMO_BUILD_TESTS)

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

set_source_files_properties(
  tests/demo_test_cpu.cpp
  tests/demo_test_gpu.cpp
  PROPERTIES LANGUAGE CUDA)

# For the tests relying on the CPU only, we can use the Google Test framework.
add_executable(demo_test_cpu tests/demo_test_cpu.cpp)
set_property(TARGET demo_test_cpu PROPERTY CUDA_ARCHITECTURES native)
target_compile_options(demo_test_cpu PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:
    --expt-relaxed-constexpr
    -diag-suppress 20011,20014
    --compiler-options -frounding-math
    $<$<CONFIG:Debug>:-G>>
  $<$<CONFIG:Debug>:-g -DDEBUG>)
target_include_directories(demo_test_cpu PRIVATE include ${cuda_battery_SOURCE_DIR}/include)
target_link_libraries(demo_test_cpu gtest_main)
gtest_discover_tests(demo_test_cpu)

# For the tests relying on the GPU, we have no testing framework, and we must test "by hand".
# All the steps done for creating the main executable must be repeated here.
set_source_files_properties(tests/demo_test_gpu.cpp PROPERTIES LANGUAGE CUDA)
add_executable(demo_test_gpu tests/demo_test_gpu.cpp)
set_property(TARGET demo_test_gpu PROPERTY CUDA_ARCHITECTURES native)
target_compile_options(demo_test_gpu PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:
    --expt-relaxed-constexpr
    -diag-suppress 20011,20014
    --compiler-options -frounding-math
    $<$<CONFIG:Debug>:-G>>
  $<$<CONFIG:Debug>:-g -DDEBUG>)
target_include_directories(demo_test_gpu PRIVATE include ${cuda_battery_SOURCE_DIR}/include)
# We run the tests using compute-sanitizer to check for memory leaks.
add_test(NAME demo_test_gpu COMMAND compute-sanitizer --error-exitcode 1 $<TARGET_FILE:demo_test_gpu>)

endif()

# Documentation

find_package(Doxygen REQUIRED doxygen)

# Options based on those of Halide: https://github.com/halide/Halide/blob/feb81a2a0f7bc39cfa499e4e1a7dfedbc5250aaa/doc/CMakeLists.txt
set(DOXYGEN_PROJECT_NAME "CUDA battery demo")
set(DOXYGEN_BUILTIN_STL_SUPPORT YES)
set(DOXYGEN_CASE_SENSE_NAMES NO)
set(DOXYGEN_CLASS_DIAGRAMS NO)
set(DOXYGEN_DISTRIBUTE_GROUP_DOC YES)
set(DOXYGEN_EXTRACT_ALL YES)
set(DOXYGEN_EXTRACT_PRIVATE NO)
set(DOXYGEN_FILE_PATTERNS *.hpp)
set(DOXYGEN_GENERATE_TREEVIEW YES)
set(DOXYGEN_HIDE_IN_BODY_DOCS YES)
set(DOXYGEN_QUIET YES)
set(DOXYGEN_RECURSIVE YES)
set(DOXYGEN_SORT_BY_SCOPE_NAME YES)
set(DOXYGEN_SORT_MEMBER_DOCS NO)
set(DOXYGEN_SOURCE_BROWSER NO)
set(DOXYGEN_STRIP_CODE_COMMENTS NO)
set(DOXYGEN_USE_MATHJAX YES)
set(DOXYGEN_EXTRA_PACKAGES stmaryrd)
# We suppose that everything in the namespace `impl` is not documented.
set(DOXYGEN_EXCLUDE_SYMBOLS = impl)
set(DOXYGEN_ENABLE_PREPROCESSING YES)
# We run Doxygen with __NVCC__ activated so it generates the documentation of GPU functions as well.
set(DOXYGEN_PREDEFINED __NVCC__)
set(DOXYGEN_REFERENCED_BY_RELATION NO)
set(DOXYGEN_REFERENCES_RELATION NO)

doxygen_add_docs(doc_cuda_battery_demo
                 "include/"
                 ALL
                 COMMENT "Generate HTML documentation")
