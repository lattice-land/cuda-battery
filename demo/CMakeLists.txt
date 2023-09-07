cmake_minimum_required(VERSION 3.24)

cmake_policy(SET CMP0135 NEW)

# I. Project main description and options.

# We compile the project for the native architecture, unless the user has specified a different architecture.
# An architecture specification can be given at configuration time using `-DCMAKE_CUDA_ARCHITECTURES=75`.
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES native)
endif()

# Name and information about this project.
project(cuda-battery-demo
  HOMEPAGE_URL "https://github.com/lattice-land/cuda-battery"
  LANGUAGES CUDA CXX)

# An option to build the project with the tests activated or not (tests are built by default here).
option(CUDA_BATTERY_DEMO_BUILD_TESTS "CUDA_BATTERY_DEMO_BUILD_TESTS" ON)

# II. Add dependencies.

include(FetchContent)

# This is how you add a dependency to a CMake project, of course the dependency must have a CMakeLists.txt file in its repository for CMake to know how to build it.
# Here, we automatically download the cuda-battery library from Github.
FetchContent_Declare(
  cuda_battery
  GIT_REPOSITORY "https://github.com/lattice-land/cuda-battery.git"
  GIT_TAG        v1.0.0
)
FetchContent_MakeAvailable(cuda_battery)

# III. Build the main executable.

# Add any `.cpp` files that contains CUDA code here.
# If you want to keep `.cu` files, simply delete this command.
set_source_files_properties(
  src/demo.cpp
  src/inkernel_allocation.cpp
  src/simple.cpp
  PROPERTIES LANGUAGE CUDA)

# Multiple CUDA files?
# If you have several `.cpp` files with CUDA code, you must activate separable compilation:
# set_property(TARGET demo APPEND PROPERTY CUDA_SEPARABLE_COMPILATION ON)

# This is the main executable, add any additional `.cpp` files required to compile the project here, for instance:
# add_executable(demo src/demo.cpp src/config.cpp)
add_executable(demo src/demo.cpp)

# Add the include directory.
target_include_directories(demo PRIVATE include)

# We include the library `cuda_battery` (it is not really linked since it is just a header-only library, that's just the CMake terminology), note that all the compile options of cuda_battery are inherited by the executable.
target_link_libraries(demo cuda_battery)

# The executable inherits the compiler options of cuda_battery.
# We also delete warnings telling us that calling __host__ from __host__ __device__ is forbidden (they are generated due to templated functions working on both CPU/GPU).
target_compile_options(demo PRIVATE
  "$<$<COMPILE_LANGUAGE:CUDA>:SHELL:-diag-suppress 20011,20014>"
)

# Describe the second executable.
add_executable(inkernel_allocation src/inkernel_allocation.cpp)
target_link_libraries(inkernel_allocation cuda_battery)

# Describe the third executable.
add_executable(simple src/simple.cpp)
target_link_libraries(simple cuda_battery)

# IV. Build and execute the CPU and GPU tests.
# You can simply add new CPU tests by adding a new file in `tests/`, and GPU tests by terminating the filename with "gpu".
# They will be automatically taken into account.

if(CUDA_BATTERY_DEMO_BUILD_TESTS)

# Google Test dependency

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

file(GLOB test_files tests/*.cpp)
foreach(file ${test_files})
  cmake_path(GET file STEM test_name)
  set_source_files_properties(${file} PROPERTIES LANGUAGE CUDA)
  add_executable(${test_name} ${file})
  target_include_directories(${test_name} PRIVATE include)
  target_compile_options(${test_name} PRIVATE
    "$<$<COMPILE_LANGUAGE:CUDA>:SHELL:-diag-suppress 20011,20014>"
  )
  if(${test_name} MATCHES ".*gpu$")
    target_link_libraries(${test_name} cuda_battery)
    # We run the tests using compute-sanitizer to check for memory leaks.
    add_test(NAME ${test_name} COMMAND compute-sanitizer --error-exitcode 1 $<TARGET_FILE:${test_name}>)
  else()
    target_link_libraries(${test_name} cuda_battery gtest_main)
    gtest_discover_tests(${test_name})
  endif()
endforeach()

endif()

# V. Documentation

find_package(Doxygen REQUIRED doxygen)

set(DOXYGEN_PROJECT_NAME "DEMO of the cuda-battery library")
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
