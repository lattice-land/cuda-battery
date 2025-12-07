# HIP Tests CMakeLists.txt
# This file builds and configures the hipified test files

# Google Test dependency
include(FetchContent)

FetchContent_Declare(
  googletest
  GIT_REPOSITORY "https://github.com/google/googletest"
  GIT_TAG v1.15.2
  GIT_SHALLOW 1
)
# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

enable_testing()
include(GoogleTest)

# Get all test files first
file(GLOB all_test_files *.cpp)

# I. CPU Tests (ending with "_test.cpp", but NOT "_cpu_gpu.cpp" or "_gpu.cpp")
set(hip_cpu_test_files)
foreach(file ${all_test_files})
  get_filename_component(filename ${file} NAME)
  if(filename MATCHES ".*_test\\.cpp$" AND NOT filename MATCHES ".*_cpu_gpu\\.cpp$" AND NOT filename MATCHES ".*_gpu\\.cpp$")
    list(APPEND hip_cpu_test_files ${file})
  endif()
endforeach()

foreach(file ${hip_cpu_test_files})
  cmake_path(GET file STEM test_name)
  set(hip_test_name "${test_name}_hip")
  add_executable(${hip_test_name} ${file})
  target_link_libraries(${hip_test_name} hip_battery gtest_main)
  gtest_discover_tests(${hip_test_name})
endforeach()

# II. GPU Tests (ending with "_gpu.cpp", but NOT "_cpu_gpu.cpp") - only if GPU is enabled
if(GPU)
  set(hip_gpu_test_files)
  foreach(file ${all_test_files})
    get_filename_component(filename ${file} NAME)
    if(filename MATCHES ".*_gpu\\.cpp$" AND NOT filename MATCHES ".*_cpu_gpu\\.cpp$")
      list(APPEND hip_gpu_test_files ${file})
    endif()
  endforeach()
  
  foreach(file ${hip_gpu_test_files})
    cmake_path(GET file STEM test_name)
    set(hip_test_name "${test_name}_hip")
    set_source_files_properties(${file} PROPERTIES LANGUAGE HIP)
    add_executable(${hip_test_name} ${file})
    target_link_libraries(${hip_test_name} hip_battery gtest_main)
    # For NVIDIA backend, link CUDA driver library for complete HIP support
    if(CUDAToolkit_FOUND AND HIP_PLATFORM STREQUAL "nvidia")
      target_link_libraries(${hip_test_name} CUDA::cuda_driver)
    endif()
    target_link_options(${hip_test_name} PRIVATE $<$<CXX_COMPILER_ID:MSVC>:/NODEFAULTLIB:LIBCMT>)
    gtest_discover_tests(${hip_test_name})
  endforeach()

  # III. CPU+GPU Tests (ending with "_cpu_gpu.cpp") - only if GPU is enabled
  set(hip_cpu_gpu_test_files)
  foreach(file ${all_test_files})
    get_filename_component(filename ${file} NAME)
    if(filename MATCHES ".*_cpu_gpu\\.cpp$")
      list(APPEND hip_cpu_gpu_test_files ${file})
    endif()
  endforeach()
  
  foreach(file ${hip_cpu_gpu_test_files})
    cmake_path(GET file STEM test_name)
    set(hip_test_name "${test_name}_hip")
    set_source_files_properties(${file} PROPERTIES LANGUAGE HIP)
    add_executable(${hip_test_name} ${file})
    target_link_libraries(${hip_test_name} hip_battery gtest_main)
    # For NVIDIA backend, link CUDA driver library for complete HIP support
    if(CUDAToolkit_FOUND AND HIP_PLATFORM STREQUAL "nvidia")
      target_link_libraries(${hip_test_name} CUDA::cuda_driver)
    endif()
    target_link_options(${hip_test_name} PRIVATE $<$<CXX_COMPILER_ID:MSVC>:/NODEFAULTLIB:LIBCMT>)
    
    # Skip test discovery for CPU+GPU tests to avoid timeout during GPU initialization
    # Add tests manually instead
    add_test(NAME ${hip_test_name} COMMAND ${hip_test_name})
  endforeach()
endif()
