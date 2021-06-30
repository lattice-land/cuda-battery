# Battery for CUDA programming

[![Build Status](https://travis-ci.com/lattice-land/cuda-battery.svg?branch=main)](https://travis-ci.com/lattice-land/cuda-battery)

This library provides a number of data structures to ease the programming in CUDA.
Its goal is to support the development of various lattice data structures in our project "Lattice Land".
One specific design choice is that most structure will be semi-immutable.
For instance, we do not provide an array data structure that can arbitrarily grow (e.g., vector) but instead a fixed size array, but still allocated at runtime with a dynamic size (in contrast to `std::array`).
This choice is made because
