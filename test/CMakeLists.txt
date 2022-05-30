include_directories(BEFORE ..)
# set(CUDA_BUILD_CUBIN ON)

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


# add_executable(test_ntt test_ntt.cu)
# cuda_target_compile_options(test_ntt OFF)
# target_link_libraries(test_ntt PRIVATE prover_reference_functions stdc++)

add_executable(test_msm test_multiscalar.cu)
cuda_target_compile_options(test_msm ON)
target_link_libraries(test_msm PRIVATE prover_reference_functions stdc++)

