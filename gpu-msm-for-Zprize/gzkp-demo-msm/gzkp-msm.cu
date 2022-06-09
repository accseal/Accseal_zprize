#include <map>
#include <string>
#include <chrono>
#include <iostream>
#include <assert.h>

#include "src/msm.h"


static constexpr int ScalarBit = 253;

typedef G1_type G1;


struct Point
{
    uint64_t *point;
};

struct Scalar
{
    uint64_t *scalar;
};

extern "C"
var* multiexp_cuda_c(const Scalar *scalar, const Point* point, uint64_t length)
{

    auto beginning = now();
    auto t = beginning;
    
    int C = (int)log2(length) / 2;
    int C2 = (C + 1) / 2;
    int windows_num = (ScalarBit + C - 1) / C;

    size_t *points = nullptr;
    size_t *scalars = nullptr;
    typedef typename G1::field_type FF;
    size_t aff_pt_bytes = (2 * FF::DEGREE * FF::ELT_LIMBS) * sizeof(uint64_t);
    size_t total_aff_bytes = length * aff_pt_bytes;

    cudaMalloc(&scalars, (length) * ELT_LIMBS_R * sizeof(uint64_t));
    if (cudaPeekAtLastError() != cudaSuccess) {
        printf("Cuda Error 0: %s\n", cudaGetErrorString(cudaPeekAtLastError()));
    }
    cudaMalloc(&points, total_aff_bytes);
    if (cudaPeekAtLastError() != cudaSuccess) {
        printf("Cuda Error 1: %s\n", cudaGetErrorString(cudaPeekAtLastError()));
    }
    cudaDeviceSynchronize();
    // print_time(t, "allocate gpu memory");

    cudaMemcpy(scalars, scalar->scalar, length * ELT_LIMBS_R * sizeof(uint64_t), cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();
    // print_time(t, "transfer scalar vector to gpu");
    if (cudaPeekAtLastError() != cudaSuccess) {
        printf("Cuda Error 2: %s\n", cudaGetErrorString(cudaPeekAtLastError()));
    }
    
    cudaMemcpy(points, point->point, total_aff_bytes, cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();
    if (cudaPeekAtLastError() != cudaSuccess) {
        printf("Cuda Error 3: %s\n", cudaGetErrorString(cudaPeekAtLastError()));
    }
    // print_time(t, "transfer point vector to gpu");

    auto msm = now();
    cudaStream_t sA;
    cudaStreamCreate(&sA);
    auto result = gzkp::msm::msm_pippenger<G1, ScalarBit>(sA, points, scalars, length, C, C2, windows_num);
    cudaStreamSynchronize(sA);
    cudaDeviceSynchronize();
    if (cudaPeekAtLastError() != cudaSuccess) {
        printf("Cuda Error 6: %s\n", cudaGetErrorString(cudaPeekAtLastError()));
    }

    size_t JAC_POINT_LIMBS_ECp_MNT4 = 3 * G1::field_type::DEGREE * ELT_BYTES_Q;
    auto result_ = (var *)malloc(JAC_POINT_LIMBS_ECp_MNT4);
    cudaMemcpy(result_, result, JAC_POINT_LIMBS_ECp_MNT4, cudaMemcpyDeviceToHost);

    cudaFree(scalars);
    cudaFree(points);
    // print_time(msm, "MSM");

    // print_time(beginning, "e2e");

    // std::cout << std::flush;

    return result_;
}