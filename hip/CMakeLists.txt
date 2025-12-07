cmake_minimum_required(VERSION 3.24)
cmake_policy(SET CMP0135 NEW)

# Check if we should build CPU-only or with GPU support
option(GPU "Enable GPU support" ON)

# HIP Platform selection
set(HIP_PLATFORM "nvidia" CACHE STRING "HIP Platform: nvidia or amd")
set_property(CACHE HIP_PLATFORM PROPERTY STRINGS "nvidia" "amd")

if(GPU)
  # Set HIP_PLATFORM environment variable for hipcc
  set(ENV{HIP_PLATFORM} ${HIP_PLATFORM})
  message(STATUS "HIP Platform: ${HIP_PLATFORM}")
  
  # Platform-specific environment setup
  if(HIP_PLATFORM STREQUAL "nvidia")
    message(STATUS "Configuring for NVIDIA GPU platform")
    if(NOT DEFINED ENV{CUDA_PATH})
      set(ENV{CUDA_PATH} "/usr/local/cuda")
    endif()
    message(STATUS "CUDA_PATH: $ENV{CUDA_PATH}")
  elseif(HIP_PLATFORM STREQUAL "amd")
    message(STATUS "Configuring for AMD GPU platform")
    if(NOT DEFINED ENV{ROCM_PATH})
      set(ENV{ROCM_PATH} "/opt/rocm")
    endif()
    message(STATUS "ROCM_PATH: $ENV{ROCM_PATH}")
  else()
    message(FATAL_ERROR "Invalid HIP_PLATFORM: ${HIP_PLATFORM}. Must be 'nvidia' or 'amd'")
  endif()
  
  # Optional: Enable hipcc verbose output for debugging
  if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(ENV{HIPCC_VERBOSE} "1")
    message(STATUS "HIPCC_VERBOSE enabled for debug builds")
  endif()
  # Let CMake automatically detect the HIP compiler
  # For NVIDIA: it will use nvcc or clang with CUDA support
  # For AMD: it will use ROCm clang
  message(STATUS "Letting CMake automatically detect HIP compiler for platform: ${HIP_PLATFORM}")
  
  # Set appropriate GPU architectures based on platform
  if(NOT DEFINED CMAKE_HIP_ARCHITECTURES)
    if(HIP_PLATFORM STREQUAL "nvidia")
      # NVIDIA GPU architectures
      set(CMAKE_HIP_ARCHITECTURES "50;52;60;61;70;75;80;86;89;90" CACHE STRING "Targeted NVIDIA GPU architectures")
      message(STATUS "Using NVIDIA GPU architectures: ${CMAKE_HIP_ARCHITECTURES}")
    elseif(HIP_PLATFORM STREQUAL "amd")
      # AMD GPU architectures  
      set(CMAKE_HIP_ARCHITECTURES "gfx906;gfx908;gfx90a;gfx1030;gfx1100" CACHE STRING "Targeted AMD GPU architectures")
      message(STATUS "Using AMD GPU architectures: ${CMAKE_HIP_ARCHITECTURES}")
    endif()
  endif()
  
  # Enable HIP language support
  enable_language(HIP)
  
  project(hip-battery
    HOMEPAGE_URL "https://github.com/lattice-land/cuda-battery"
    LANGUAGES HIP CXX)
  
  # Find HIP - this works for both NVIDIA and AMD backends
  find_package(hip REQUIRED)
  
  # Add HIP include directories (same as what hipcc adds automatically)
  if(HIP_PLATFORM STREQUAL "nvidia")
    # Add both CUDA and ROCm include paths for NVIDIA HIP backend
    find_path(CUDA_INCLUDE_DIR cuda_runtime.h PATHS /usr/local/cuda*/include /usr/local/cuda/include ENV CUDA_PATH)
    find_path(ROCM_INCLUDE_DIR hip/hip_runtime.h PATHS /opt/rocm*/include /opt/rocm/include ENV ROCM_PATH)
    if(CUDA_INCLUDE_DIR)
      include_directories(SYSTEM ${CUDA_INCLUDE_DIR})
      message(STATUS "Added CUDA include directory: ${CUDA_INCLUDE_DIR}")
    endif()
    if(ROCM_INCLUDE_DIR)
      include_directories(SYSTEM ${ROCM_INCLUDE_DIR})
      message(STATUS "Added ROCm include directory: ${ROCM_INCLUDE_DIR}")
    endif()
  elseif(HIP_PLATFORM STREQUAL "amd")
    # Add ROCm include paths for AMD HIP backend
    find_path(ROCM_INCLUDE_DIR hip/hip_runtime.h PATHS /opt/rocm*/include /opt/rocm/include ENV ROCM_PATH)
    if(ROCM_INCLUDE_DIR)
      include_directories(SYSTEM ${ROCM_INCLUDE_DIR})
      message(STATUS "Added ROCm include directory: ${ROCM_INCLUDE_DIR}")
    endif()
  endif()
  
  # Configure platform-specific dependencies and backend macros
  if(HIP_PLATFORM STREQUAL "nvidia")
    message(STATUS "HIP: Using NVIDIA CUDA backend")
    # Define backend-specific macro for conditional compilation
    add_compile_definitions(__HIP_BACKEND_NVIDIA__)
    # For NVIDIA, find CUDA toolkit for additional libraries
    find_package(CUDAToolkit QUIET)
    if(CUDAToolkit_FOUND)
      message(STATUS "CUDA Toolkit found: ${CUDAToolkit_VERSION}")
    endif()
  elseif(HIP_PLATFORM STREQUAL "amd")
    message(STATUS "HIP: Using AMD ROCm backend")
    # Define backend-specific macro for conditional compilation
    add_compile_definitions(__HIP_BACKEND_AMD__)
    # For AMD, find rocPRIM and other ROCm libraries
    find_package(rocprim QUIET)
    if(rocprim_FOUND)
      message(STATUS "rocPRIM found")
    endif()
  endif()
  
else()
  project(hip-battery
    HOMEPAGE_URL "https://github.com/lattice-land/cuda-battery"
    LANGUAGES CXX)
endif()

# Options
option(REDUCE_PTX_SIZE "REDUCE_PTX_SIZE" OFF)
option(HIP_BATTERY_BUILD_TESTS "HIP_BATTERY_BUILD_TESTS" OFF)
option(HIP_BATTERY_BUILD_DOC "Build documentation" OFF)

# Preparing the library
add_library(hip_battery INTERFACE)
target_compile_features(hip_battery INTERFACE cxx_std_20)
# Ensure HIP headers are found first and exclusively
target_include_directories(hip_battery INTERFACE 
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)

# Link libraries based on GPU support
if(GPU)
  # Link HIP libraries - this works for both NVIDIA and AMD
  target_link_libraries(hip_battery INTERFACE hip::host)
  
  # Platform-specific libraries
  if(HIP_PLATFORM STREQUAL "nvidia")
    # For NVIDIA backend, link CUDA libraries if available
    if(CUDAToolkit_FOUND)
      target_link_libraries(hip_battery INTERFACE CUDA::cudart)
    endif()
  elseif(HIP_PLATFORM STREQUAL "amd")
    # For AMD backend, link rocPRIM if available
    if(rocprim_FOUND)
      target_link_libraries(hip_battery INTERFACE roc::rocprim)
    endif()
  endif()
endif()

# Compile options
target_compile_options(hip_battery INTERFACE
    # HIP compiler options (only when GPU is enabled)
    "$<$<AND:$<BOOL:${GPU}>,$<COMPILE_LANGUAGE:HIP>,$<CONFIG:Debug>>:-g;-O0>"
    "$<$<AND:$<BOOL:${GPU}>,$<COMPILE_LANGUAGE:HIP>,$<CONFIG:Release>>:-O3>"
    # NVIDIA-specific options for constexpr compatibility
    "$<$<AND:$<BOOL:${GPU}>,$<STREQUAL:${HIP_PLATFORM},nvidia>,$<COMPILE_LANGUAGE:HIP>>:--expt-relaxed-constexpr>"
    # CPU compiler options for floating point
    "$<$<CXX_COMPILER_ID:GNU,Clang>:-frounding-math>"
    "$<$AND:$<COMPILE_LANGUAGE:CXX>,$<CXX_COMPILER_ID:MSVC>>:/fp:strict>"
)

# Conditional definitions
if(GPU)
  # HIP-specific definitions that work for both backends
  target_compile_definitions(hip_battery INTERFACE __HIP__)
  
  # Platform-specific definitions
  if(HIP_PLATFORM STREQUAL "nvidia")
    target_compile_definitions(hip_battery INTERFACE __HIP_PLATFORM_NVIDIA__)
  elseif(HIP_PLATFORM STREQUAL "amd") 
    target_compile_definitions(hip_battery INTERFACE __HIP_PLATFORM_AMD__)
  endif()
endif()

if(REDUCE_PTX_SIZE)
  target_compile_definitions(hip_battery INTERFACE "$<$<COMPILE_LANGUAGE:HIP>:REDUCE_PTX_SIZE>")
endif()

if(HIP_BATTERY_BUILD_TESTS)
  enable_testing()
  add_subdirectory(tests)
endif()

# Documentation
if(HIP_BATTERY_BUILD_DOC)

find_package(Doxygen REQUIRED doxygen)

set(DOXYGEN_PROJECT_NAME "HIP battery library")
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
set(DOXYGEN_EXCLUDE_SYMBOLS = impl,MAP_LIMITS,FLOAT_ARITHMETIC_HIP_IMPL,FLOAT_ARITHMETIC_CPP_IMPL)
set(DOXYGEN_ENABLE_PREPROCESSING YES)
set(DOXYGEN_PREDEFINED __HIP__ __HIP_PLATFORM_AMD__)
set(DOXYGEN_REFERENCED_BY_RELATION NO)
set(DOXYGEN_REFERENCES_RELATION NO)

doxygen_add_docs(doc_hip_battery
                 "include/"
                 ALL
                 COMMENT "Generate HTML documentation for HIP")

endif()