cmake_minimum_required(VERSION 3.24)
cmake_policy(SET CMP0135 NEW)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES native)
endif()

project(cuda-battery
  VERSION 1.0.0
  HOMEPAGE_URL "https://github.com/lattice-land/cuda-battery"
  LANGUAGES CUDA CXX)
option(GPU "GPU" ON)
option(REDUCE_PTX_SIZE "REDUCE_PTX_SIZE" OFF)
option(CUDA_BATTERY_BUILD_TESTS "CUDA_BATTERY_BUILD_TESTS" OFF)

# Preparing the library

add_library(cuda_battery INTERFACE)
target_compile_features(cuda_battery INTERFACE cxx_std_20)
target_compile_features(cuda_battery INTERFACE cuda_std_20)
target_include_directories(cuda_battery INTERFACE include)

target_compile_options(cuda_battery INTERFACE
    "$<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr;$<$<CONFIG:Debug>:-G>>"
    # -frounding-math is necessary with GCC because `#pragma STDC FENV_ACCESS ON` does not work.
    # This option will help in making the compiler take into account rounding modes, but bugs persist so floating-point arithmetic on CPUs might not always be correct right now...
    # C.f. https://gcc.gnu.org/bugzilla/show_bug.cgi?id=34678
    "$<$<CXX_COMPILER_ID:GNU,Clang>:SHELL:$<$<COMPILE_LANGUAGE:CUDA>:--compiler-options> -frounding-math>"
)

if(REDUCE_PTX_SIZE)
  target_compile_definitions(cuda_battery INTERFACE "$<$<COMPILE_LANGUAGE:CUDA>:REDUCE_PTX_SIZE>")
endif()

if(CUDA_BATTERY_BUILD_TESTS)

# Google Test dependency

include(FetchContent)

FetchContent_Declare(
  googletest
  GIT_REPOSITORY "https://github.com/google/googletest"
  GIT_TAG v1.13.0
)
# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

enable_testing()
include(GoogleTest)

# I. CPU Tests (ending with "_test.cpp")
file(GLOB cpu_test_files tests/*_test.cpp)
foreach(file ${cpu_test_files})
  cmake_path(GET file STEM test_name)
  add_executable(${test_name} ${file})
  target_link_libraries(${test_name} cuda_battery gtest_main)
  gtest_discover_tests(${test_name})
endforeach()

# II. GPU Tests (ending with "_gpu.cpp")
if(GPU)
  file(GLOB gpu_test_files tests/*_gpu.cpp)
  foreach(file ${gpu_test_files})
    cmake_path(GET file STEM test_name)
    set_source_files_properties(${file} PROPERTIES LANGUAGE CUDA)
    add_executable(${test_name} ${file})
    target_link_libraries(${test_name} cuda_battery)
    add_test(NAME ${test_name} COMMAND compute-sanitizer --error-exitcode 1 $<TARGET_FILE:${test_name}>)
  endforeach()
endif()

endif()

# Documentation

find_package(Doxygen REQUIRED doxygen)

set(DOXYGEN_PROJECT_NAME "Cuda battery library")
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
set(DOXYGEN_EXCLUDE_SYMBOLS = impl,MAP_LIMITS,FLOAT_ARITHMETIC_CUDA_IMPL,FLOAT_ARITHMETIC_CPP_IMPL)
set(DOXYGEN_ENABLE_PREPROCESSING YES)
set(DOXYGEN_PREDEFINED __NVCC__)
set(DOXYGEN_REFERENCED_BY_RELATION NO)
set(DOXYGEN_REFERENCES_RELATION NO)

doxygen_add_docs(doc_cuda_battery
                 "include/"
                 ALL
                 COMMENT "Generate HTML documentation")
