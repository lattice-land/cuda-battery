#!/bin/sh

mkdir -p build/cpu-debug
cmake -DCMAKE_BUILD_TYPE=Debug -DGPU=OFF -DBATTERY_BUILD_TESTS=ON -Bbuild/cpu-debug &&
cmake --build build/cpu-debug &&
(cd build/cpu-debug;
ctest;
cd ../..)

mkdir -p build/gpu-debug
cmake -DCMAKE_BUILD_TYPE=Debug -DGPU=ON -DBATTERY_BUILD_TESTS=ON -Bbuild/gpu-debug &&
cmake --build build/gpu-debug &&
(cd build/gpu-debug;
ctest;
cd ../..)
