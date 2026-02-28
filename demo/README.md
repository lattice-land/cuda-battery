# Battery for CUDA/HIP programming

Companion code for the [CUDA battery tutorial](https://lattice-land.github.io/CUDA-Battery.html).
It can be copied to start your own CUDA/HIP/CMake project.

## CUDA (NVIDIA)

To compile in release mode:
```
cmake -DCMAKE_BUILD_TYPE=Release -DCUDA_BATTERY_DEMO_BUILD_TESTS=ON -Bbuild/gpu-release &&
cmake --build build/gpu-release
```

And in debug mode:
```
cmake -DCMAKE_BUILD_TYPE=Debug -DCUDA_BATTERY_DEMO_BUILD_TESTS=ON -Bbuild/gpu-debug &&
cmake --build build/gpu-debug
```

To run the tests:
```
cd build/gpu-debug && ctest --output-on-failure
```

To run the executables:
```
./build/gpu-debug/demo gpu 10000 256
./build/gpu-debug/simple
./build/gpu-debug/inkernel_allocation
```

## HIP (AMD ROCm)

Requires ROCm installed (tested with ROCm 6.2+).

To compile in release mode:
```
cmake -DHIP=ON -DGPU=OFF -DCMAKE_BUILD_TYPE=Release -DCUDA_BATTERY_DEMO_BUILD_TESTS=ON -Bbuild/hip-release &&
cmake --build build/hip-release
```

And in debug mode:
```
cmake -DHIP=ON -DGPU=OFF -DCMAKE_BUILD_TYPE=Debug -DCUDA_BATTERY_DEMO_BUILD_TESTS=ON -Bbuild/hip-debug &&
cmake --build build/hip-debug
```

To run the tests:
```
cd build/hip-debug && ctest --output-on-failure
```

To run the executables:
```
./build/hip-debug/demo_hip gpu 10000 256
./build/hip-debug/simple_hip
./build/hip-debug/inkernel_allocation_hip
```

### HIP on NVIDIA (temporary, for development only)

If you are developing on an NVIDIA machine and want to verify the HIP build
infrastructure without AMD hardware, set `HIP_PLATFORM=nvidia`:

```
HIP_PLATFORM=nvidia cmake -DHIP=ON -DGPU=OFF -DCMAKE_BUILD_TYPE=Debug \
  -DCUDA_BATTERY_DEMO_BUILD_TESTS=ON -Bbuild/hip-nvidia-debug &&
cmake --build build/hip-nvidia-debug
```

> **Note:** With `HIP_PLATFORM=nvidia`, CMake selects nvcc as the HIP compiler,
> so `BATTERY_CUDA_BACKEND` is active rather than `BATTERY_HIP_BACKEND`.
> This verifies that the HIP-language build system works but does **not** exercise
> the HIP API code paths. Those are only exercised on AMD hardware with the
> standard HIP build above. The `hip-nvidia` build variant will be removed once
> AMD testing is established.
