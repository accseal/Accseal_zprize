include_directories(BEFORE ..)
cmake_minimum_required(VERSION 3.14 FATAL_ERROR)
cmake_policy(SET CMP0048 NEW)
project(gzkp-demo-msm LANGUAGES CUDA CXX)

find_package(CUDA 11.4 REQUIRED)

# V100
set(CMAKE_CUDA_ARCHITECTURES "70")

set(CMAKE_BUILD_TYPE Release)
set(CMAKE_EXPORT_COMPILE_COMMANDS NO)
set(CMAKE_CUDA_COMPILER /usr/local/cuda-11.4/bin/nvcc)

macro(cuda_target_compile_options TARGET VERBOSE)
  if (CMAKE_CUDA_COMPILER_ID STREQUAL "NVIDIA")
    if (${VERBOSE})
      target_compile_options(${TARGET} PRIVATE --ptxas-options=-v;-lineinfo)
    endif ()
  elseif(CMAKE_CUDA_COMPILER_ID STREQUAL "Clang")
  else()
    message(FATAL_ERROR "Unknown CMAKE_CUDA_COMPILER_ID ${CMAKE_CUDA_COMPILER_ID}")
  endif()
endmacro()

add_library(rust_gzkp_msm_lib STATIC gzkp-msm.cu)
cuda_target_compile_options(rust_gzkp_msm_lib OFF)
install(TARGETS rust_gzkp_msm_lib DESTINATION .)
