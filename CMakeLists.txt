cmake_minimum_required(VERSION 3.14 FATAL_ERROR)
cmake_policy(SET CMP0048 NEW)
project(cuzk LANGUAGES CUDA CXX)

find_package(CUDA 10.0 REQUIRED)
find_package(gflags COMPONENTS nothreads_static)

# add_definitions(-DALT_BN128)
# add_definitions(-DBLS12_381)
add_definitions(-DMNT4753)

set(CMAKE_CUDA_ARCHITECTURES "60")

add_subdirectory(gpu-groth16-prover-3x)

add_subdirectory(test)