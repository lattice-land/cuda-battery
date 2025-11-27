#!/bin/bash

# Define source and destination directories
SRC_DIR="include/battery"
HIP_DIR="hip/include/battery"

# File extensions to hipify (adjust as needed)
FILE_EXTENSIONS=("*.hpp" "*.cu" "*.cpp" "*.h")

# 1. Create the destination directory if it doesn't exist
mkdir -p "$HIP_DIR"

echo "Starting HIPIFY process from '$SRC_DIR' to '$HIP_DIR'..."

# 2. Use 'find' to locate relevant files and process them in a loop
# The loop handles file names with spaces correctly
find "$SRC_DIR" -type f \( -name "*.hpp" -o -name "*.cu" -o -name "*.cpp" -o -name "*.h" \) | while IFS= read -r file; do
    # Calculate the relative path of the file starting from the base SRC_DIR
    relative_path="${file#$SRC_DIR/}"

    # Determine the full path of the output file
    output_file="$HIP_DIR/$relative_path"

    # Ensure the output directory structure exists for the current file
    output_dir=$(dirname "$output_file")
    mkdir -p "$output_dir"

    # Run hipify-clang for the specific file with additional options to handle conflicts
    echo "Converting: $file -> $output_file"
    
    # First try with --no-cuda-wrapper-headers to avoid conflicts
    if ! hipify-clang "$file" -o "$output_file" \
        --cuda-path=/usr/local/cuda \
        --clang-resource-directory=/usr/lib/llvm-20/lib/clang/20 \
        --no-cuda-wrapper-headers \
        --safe-math \
        --skip-excluded-preprocessor-conditional-blocks \
        --print-stats \
        2>/dev/null; then
        
        echo "  Standard hipify failed, trying with preprocessing..."
        
        # If that fails, try preprocessing the file first to resolve conditionals
        temp_file=$(mktemp --suffix=.${file##*.})
        
        # Preprocess with HIP definitions
        clang -E -D__HIP_PLATFORM_AMD__ -D__HIP__ -DHIP_VERSION_MAJOR=5 \
              -I/opt/rocm/include -I"$SRC_DIR" -I"include" \
              "$file" -o "$temp_file" 2>/dev/null
        
        # Try hipify on preprocessed file
        if ! hipify-clang "$temp_file" -o "$output_file" \
            --cuda-path=/usr/local/cuda \
            --clang-resource-directory=/usr/lib/llvm-20/lib/clang/20 \
            --no-cuda-wrapper-headers \
            2>/dev/null; then
            
            echo "  Hipify failed, copying and doing manual substitutions..."
            
            # Fall back to manual copying and basic substitutions
            cp "$file" "$output_file"
            
            # Basic CUDA to HIP substitutions using sed
            sed -i 's/#include <cuda_runtime\.h>/#include <hip\/hip_runtime.h>/g' "$output_file"
            sed -i 's/#include <cuda\.h>/#include <hip\/hip_runtime_api.h>/g' "$output_file"
            sed -i 's/cudaError_t/hipError_t/g' "$output_file"
            sed -i 's/cudaSuccess/hipSuccess/g' "$output_file"
            sed -i 's/cudaMalloc/hipMalloc/g' "$output_file"
            sed -i 's/cudaFree/hipFree/g' "$output_file"
            sed -i 's/cudaMemcpy/hipMemcpy/g' "$output_file"
            sed -i 's/cudaDeviceSynchronize/hipDeviceSynchronize/g' "$output_file"
            sed -i 's/__global__/__global__/g' "$output_file"
            sed -i 's/__device__/__device__/g' "$output_file"
            sed -i 's/__host__/__host__/g' "$output_file"
            sed -i 's/blockIdx/hipBlockIdx_x/g' "$output_file"
            sed -i 's/threadIdx/hipThreadIdx_x/g' "$output_file"
            sed -i 's/blockDim/hipBlockDim_x/g' "$output_file"
            
            echo "  Manual conversion completed with warnings"
        fi
        
        # Clean up temp file
        rm -f "$temp_file"
    fi
done

echo "HIPIFY process complete."

# Post-process to fix common issues
echo "Post-processing files to fix namespace conflicts..."
find "$HIP_DIR" -name "*.hpp" -o -name "*.h" | while read -r file; do
    # Fix std namespace conflicts by wrapping in HIP-specific guards
    sed -i '/using std::/i #ifndef __HIP_DEVICE_COMPILE__' "$file"
    sed -i '/using std::/a #endif' "$file"
done

echo "All processing complete."