cmake_minimum_required(VERSION 3.20)

include(FetchContent)
include(CMakePrintHelpers)

project(cuda-battery LANGUAGES CUDA CXX)
option(GPU "GPU" ON)

add_link_options ($<DEVICE_LINK:-arch=sm_75>)

if(NOT DEFINED CMAKE_CXX_STANDARD)
    set(CMAKE_CXX_STANDARD 20)
    set(CMAKE_CXX_STANDARD_REQUIRED ON)
endif()

if(NOT DEFINED CMAKE_CUDA_STANDARD)
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
endif()

# Google Test dependency

FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/609281088cfefc76f9d0ce82e1ff6c30cc3591e5.zip
)
# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

# Preparing the library

if(GPU)
    set_source_files_properties(
        src/allocator.cpp
        src/utility.cpp
        PROPERTIES LANGUAGE CUDA)
endif()

add_library(cuda_battery src/allocator.cpp src/utility.cpp)
target_include_directories(cuda_battery PUBLIC include)

if(GPU)
    set_property(TARGET cuda_battery PROPERTY CUDA_ARCHITECTURES 75)
    set_property(TARGET cuda_battery PROPERTY CUDA_SEPARABLE_COMPILATION ON)
endif()

target_compile_options(cuda_battery PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:
        --expt-relaxed-constexpr
        -rdc=true>
    $<$<CONFIG:Debug>:-g -DDEBUG>
    $<$<CONFIG:LDebug>:-g -DTRACE>
)

enable_testing()
include(GoogleTest)

add_executable(utility_test tests/utility_test.cpp)
target_link_libraries(utility_test cuda_battery gtest_main)
gtest_discover_tests(utility_test)

add_executable(darray_test tests/darray_test.cpp)
target_link_libraries(darray_test cuda_battery gtest_main)
gtest_discover_tests(darray_test)

add_executable(string_test tests/string_test.cpp)
target_link_libraries(string_test cuda_battery gtest_main)
gtest_discover_tests(string_test)

add_executable(variant_test tests/variant_test.cpp)
target_link_libraries(variant_test cuda_battery gtest_main)
gtest_discover_tests(variant_test)

add_executable(vector_test tests/vector_test.cpp)
target_link_libraries(vector_test cuda_battery gtest_main)
gtest_discover_tests(vector_test)

if(GPU)
  set_source_files_properties(tests/polymorphic_darray_test.cpp PROPERTIES LANGUAGE CUDA)
  add_executable(polymorphic_darray_test tests/polymorphic_darray_test.cpp)
  set_property(TARGET polymorphic_darray_test PROPERTY CUDA_SEPARABLE_COMPILATION ON)
  set_property(TARGET polymorphic_darray_test PROPERTY CUDA_ARCHITECTURES 75)
  target_compile_options(polymorphic_darray_test PRIVATE --expt-relaxed-constexpr -rdc=true
    $<$<CONFIG:Debug>:-g -DDEBUG>
    $<$<CONFIG:LDebug>:-g -DTRACE>)
  target_link_libraries(polymorphic_darray_test cuda_battery)
  add_test(NAME polymorphic_darray_test COMMAND cuda-memcheck --error-exitcode 1 $<TARGET_FILE:polymorphic_darray_test>)
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
