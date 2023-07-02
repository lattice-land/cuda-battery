# Battery for CUDA programming

Companion code for the [CUDA battery tutorial](https://lattice-land.github.io/CUDA-Battery.html).
It can be copied to start your own CUDA/CMake project.

To compile in release mode:
```
mkdir -p build/gpu-release
cmake -DCMAKE_BUILD_TYPE=Release -DDEMO_BUILD_TESTS=ON -Bbuild/gpu-release &&
cmake --build build/gpu-release
```

And in debug mode:
```
mkdir -p build/gpu-debug
cmake -DCMAKE_BUILD_TYPE=Debug -DDEMO_BUILD_TESTS=ON -Bbuild/gpu-debug &&
cmake --build build/gpu-debug
```
