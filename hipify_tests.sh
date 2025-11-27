#!/bin/bash

# Define source and destination directories
SRC_DIR="tests"
HIP_DIR="hip/tests"

# File extensions to hipify (adjust as needed)
FILE_EXTENSIONS=("*.cpp" "*.cu" "*.h" "*.hpp")

# 1. Create the destination directory if it doesn't exist
mkdir -p "$HIP_DIR"

echo "Starting HIPIFY process for tests from '$SRC_DIR' to '$HIP_DIR'..."

# 2. Use 'find' to locate relevant files and process them in a loop
# The loop handles file names with spaces correctly
find "$SRC_DIR" -type f \( -name "*.cpp" -o -name "*.cu" -o -name "*.h" -o -name "*.hpp" \) | while IFS= read -r file; do
    # Calculate the relative path of the file starting from the base SRC_DIR
    relative_path="${file#$SRC_DIR/}"

    # Determine the full path of the output file
    output_file="$HIP_DIR/$relative_path"

    # Ensure the output directory structure exists for the current file
    output_dir=$(dirname "$output_file")
    mkdir -p "$output_dir"

    # Run hipify-clang for the specific file with additional options to handle conflicts
    echo "Converting: $file -> $output_file"
    
    # First try with comprehensive hipify options for test files
    if hipify-clang "$file" -o "$output_file" \
        --cuda-path=/usr/local/cuda \
        --clang-resource-directory=/usr/lib/llvm-20/lib/clang/20 \
        --no-cuda-wrapper-headers \
        --skip-excluded-preprocessor-conditional-blocks \
        --print-stats \
        -I"include" \
        -I"hip/include" \
        -D__HIP_PLATFORM_AMD__ \
        -D__HIP__ \
        2>/dev/null; then
        
        echo "  Successfully converted with hipify-clang"
        
    else
        echo "  Standard hipify failed, trying with preprocessing..."
        
        # If that fails, try preprocessing the file first to resolve conditionals
        temp_file=$(mktemp --suffix=.${file##*.})
        
        # Preprocess with HIP definitions and include paths for tests
        clang -E -D__HIP_PLATFORM_AMD__ -D__HIP__ -DHIP_VERSION_MAJOR=5 \
              -I/opt/rocm/include -I"$SRC_DIR" -I"include" -I"hip/include" \
              -I"build/gpu-debug/_deps/googletest-src/googletest/include" \
              "$file" -o "$temp_file" 2>/dev/null
        
        # Try hipify on preprocessed file
        if hipify-clang "$temp_file" -o "$output_file" \
            --cuda-path=/usr/local/cuda \
            --clang-resource-directory=/usr/lib/llvm-20/lib/clang/20 \
            --no-cuda-wrapper-headers \
            2>/dev/null; then
            
            echo "  Successfully converted preprocessed file"
            
        else
            echo "  Hipify failed, copying and doing manual substitutions..."
            
            # Fall back to manual copying and basic substitutions
            cp "$file" "$output_file"
            
            # Basic CUDA to HIP substitutions using sed
            sed -i 's/#include <cuda_runtime\.h>/#include <hip\/hip_runtime.h>/g' "$output_file"
            sed -i 's/#include <cuda\.h>/#include <hip\/hip_runtime_api.h>/g' "$output_file"
            sed -i 's/#include <device_launch_parameters\.h>//g' "$output_file"
            
            # CUDA API to HIP API conversions
            sed -i 's/cudaError_t/hipError_t/g' "$output_file"
            sed -i 's/cudaSuccess/hipSuccess/g' "$output_file"
            sed -i 's/cudaMalloc/hipMalloc/g' "$output_file"
            sed -i 's/cudaFree/hipFree/g' "$output_file"
            sed -i 's/cudaMemcpy/hipMemcpy/g' "$output_file"
            sed -i 's/cudaMemcpyHostToDevice/hipMemcpyHostToDevice/g' "$output_file"
            sed -i 's/cudaMemcpyDeviceToHost/hipMemcpyDeviceToHost/g' "$output_file"
            sed -i 's/cudaDeviceSynchronize/hipDeviceSynchronize/g' "$output_file"
            sed -i 's/cudaGetLastError/hipGetLastError/g' "$output_file"
            sed -i 's/cudaGetErrorString/hipGetErrorString/g' "$output_file"
            
            # CUDA built-ins (keep as-is, HIP supports these)
            # sed -i 's/__global__/__global__/g' "$output_file"
            # sed -i 's/__device__/__device__/g' "$output_file"
            # sed -i 's/__host__/__host__/g' "$output_file"
            
            # Grid/Block dimension variables
            sed -i 's/blockIdx\.x/hipBlockIdx_x/g' "$output_file"
            sed -i 's/blockIdx\.y/hipBlockIdx_y/g' "$output_file"
            sed -i 's/blockIdx\.z/hipBlockIdx_z/g' "$output_file"
            sed -i 's/threadIdx\.x/hipThreadIdx_x/g' "$output_file"
            sed -i 's/threadIdx\.y/hipThreadIdx_y/g' "$output_file"
            sed -i 's/threadIdx\.z/hipThreadIdx_z/g' "$output_file"
            sed -i 's/blockDim\.x/hipBlockDim_x/g' "$output_file"
            sed -i 's/blockDim\.y/hipBlockDim_y/g' "$output_file"
            sed -i 's/blockDim\.z/hipBlockDim_z/g' "$output_file"
            sed -i 's/gridDim\.x/hipGridDim_x/g' "$output_file"
            sed -i 's/gridDim\.y/hipGridDim_y/g' "$output_file"
            sed -i 's/gridDim\.z/hipGridDim_z/g' "$output_file"
            
            # Update include paths for battery headers to use HIP versions
            sed -i 's/#include <battery\/\([^>]*\)>/#include <battery\/\1>/g' "$output_file"
            sed -i 's/#include "battery\/\([^"]*\)"/#include "battery\/\1"/g' "$output_file"
            
            echo "  Manual conversion completed"
        fi
        
        # Clean up temp file
        rm -f "$temp_file"
    fi
    
    # Post-process the converted test file
    echo "  Post-processing $output_file..."
    
    # Fix potential std namespace conflicts in test files
    if [[ -f "$output_file" ]]; then
        # Wrap problematic using declarations in guards
        sed -i '/using std::/i #ifndef __HIP_DEVICE_COMPILE__' "$output_file"
        sed -i '/using std::/a #endif' "$output_file"
        
        # Make sure we include HIP versions of battery headers
        sed -i 's|#include <battery/|#include <battery/|g' "$output_file"
        sed -i 's|#include "battery/|#include "battery/|g' "$output_file"
        
        # Update any compute-sanitizer specific code for ROCm equivalents
        sed -i 's/compute-sanitizer/rocm-debug-agent/g' "$output_file"
        
        # Handle test-specific CUDA calls that might need HIP equivalents
        sed -i 's/CUDA_CALL/HIP_CALL/g' "$output_file"
        sed -i 's/CHECK_CUDA/CHECK_HIP/g' "$output_file"
    fi
    
done

echo "HIPIFY process for tests complete."

# Create a simple CMakeLists.txt for HIP tests if it doesn't exist
if [[ ! -f "$HIP_DIR/CMakeLists.txt" ]]; then
    echo "Creating CMakeLists.txt for HIP tests..."
    cat > "$HIP_DIR/CMakeLists.txt" << 'EOF'
# HIP Tests CMakeLists.txt
# This file lists the hipified test files

# Add HIP test executables here
# Example:
# hip_add_executable(vector_test_hip vector_test.cpp)
# target_link_libraries(vector_test_hip gtest gtest_main)

EOF
fi

echo "All test hipification processing complete."
echo "HIP test files are now available in: $HIP_DIR"