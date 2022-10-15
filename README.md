# Battery for CUDA programming

[![Build Status](https://travis-ci.com/lattice-land/cuda-battery.svg?branch=main)](https://travis-ci.com/lattice-land/cuda-battery)

This library provides a number of data structures to ease the programming in CUDA.
Our goal is to support the development of various lattice data structures in our project "Lattice Land".

Everything is under the namespace `battery::`.

## Allocator

`allocator.hpp` provides several allocators for CPU and GPU:

1. *Standard allocator*: Allocate in CPU's memory from host side using standard `malloc` and `free` functions (`StandardAllocator`).
2. *Global allocator*: Allocate in GPU's global memory from host side (`GlobalAllocatorCPU`) and device side (`GlobalAllocatorGPU`).
3. *Managed allocator*: Allocate in managed memory from host side (`ManagedAllocator`).
4. *Shared allocator*: Allocate in a pool of memory from host or device side where memory is never freed (`PoolAllocator`), especially useful for dealing with shared memory in GPU.

## Memory

`memory.hpp` provides abstractions over various kinds of memory in terms of loads and stores, a memory usually carries an allocator too:

1. *Local sequential memory*: Accessed by only one thread and managed automatically (e.g., stored on the stack) (`LocalMemory`).
2. *Unprotected memory*: Memory with unprotected load and store accesses, suppose to be accessed by only one thread and managed by an allocator (`Memory<Allocator>`).
3. *Read-only memory*: Memory with only a load operation, managed by an allocator and can be accessed by several threads since the values are not supposed to change (`ReadOnlyMemory<Allocator>`).
4. *Shared atomic memory*: Memory with atomic load and store operations, managed by an allocator and can be accessed by several threads (`AtomicMemory<Allocator, memory_order>`).
    The `memory_order` defaults to the weakest memory consistency (relaxed) which is enough if you stay within the framework of the Lattice-Land project.
5. *Shared GPU atomic memory*: Same than shared atomic memory, but with an additional scope of the memory:
    * `AtomicMemoryBlock`: Memory can only be accessed within a block (shared memory).
    * `AtomicMemoryDevice`: Memory can only be accessed in a single device (global memory).
    * `AtomicMemorySystem`: Memory can be accessed from several devices.

## STL data structures

We provide various data structures that can be used on both the CPU and GPU, with the same code.
It means that you can write code once that run on both the CPU and GPU.

Your code *will not be parallelized or optimized for GPU*.
Basically, if you ask for a vector, it will be the same on CPU and GPU, the parallelism and memory accesses protection are left to you (unless explicitely mentioned).

* Vector `vector<T, Allocator>`.
* Shared pointer `shared_ptr<T, Allocator>`.
* Unique pointer `unique_ptr<T, Allocator>`.
* String `string<Allocator>` (non-resizable and null-terminated).
* Tuple `tuple<T1, ..., Tn>`, depending on the compiler used (e.g., nvcc or gcc), it includes `cuda::std::tuple` or `std::tuple`.
* Variant `variant<T1, ..., Tn>`.

Moreover, we provide a variant of `bitset` parametrized by a memory:

* `bitset<N, Mem, T>` is a bitset of size `N * sizeof(T)` where the elements of the underlying array are stored depending on the `Mem` policy (e.g., either as atomics or not).

## Utility

Provide unified functions that either forward the call to the standard library if called on host, or to the CUDA primitives when available if called on device.
Some functions are reimplemented from scratch to provide them on GPU.

* *Utility/arithmetic functions*: `swap`, `min`, `max`, `signum`, `ipow`.
* *String functions*: `strlen`, `strcmp`.
* *Floating-point functions*: `isnan`, `nextafter`, `add_up`, `add_down`, `sub_up`, `sub_down`, `mul_up`, `mul_down`, `div_up`, `div_down`.
* *Bits-level functions*: `popcount`, `countl_zero`, `countl_one`, `countr_zero`, `countr_one`.

### Infinities

For the purposes of Lattice-Land, we often need to represent infinities of primitive types, these are given by `Limits<T>::bot()` for the smallest element and `Limits<T>::top()` for the largest element.
For floating-point numbers, it returns infinities, and for integers it returns the smallest/largest element representable.
When applicable, we usually consider that `MIN_INT` and `MAX_INT` (or equivalent) are special elements modeling infinities for integer types.
This is especially important for the casting rules below.

### Casting functions

We provide functions to cast primitive types to other types according to a rounding mode.

* `ru_cast<To, From>`: cast the value of type `To` to a value of type `From`, it is guaranteed that the casted value is greater or equal to the input value.
In other terms, when the casted value is not exactly representable (e.g., `float` to `int`) we round towards infinity.
* `rd_cast<To, From>`: Same as `ru_cast` but rounds towards negative infinity.

Some things to know about these functions:

* *Infinities* are preserved, e.g., `MAX_INT` is mapped to the floating-point infinity value.
* *Overflow and underflow*: nothing special is done.
* *Rounding mode*: on CPU, we use the rounding mode to perform conversion with floating-point numbers. /!\ It is not reinitialized to the rounding mode before the call was made.
* `utility_test_cpu_gpu` tests whether the rounding functions behaves the same on CPU and GPU.
