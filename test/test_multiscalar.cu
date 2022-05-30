#include <iostream>
#include <string>
#include <chrono>
#include <assert.h>

#include "src/common.h"
#include "src/file.h"
#include "src/msm.h"
#include "gpu-groth16-prover-3x/libsnark/prover_reference_include/prover_reference_functions.hpp"

#ifdef MNT4753
static constexpr int ScalarBit = 753;
#endif
#ifdef ALT_BN128
static constexpr int ScalarBit = 254;
#endif
#ifdef BLS12_381
static constexpr int ScalarBit = 255;
#endif

typedef G1_type G1;
typedef G2_type G2;


template <typename B>
void test_MSM(
  int size,
  int window_size,
  int reduce_size,
  var *scalars,
  const var *points
) {
  B::init_public_params();

  cudaStream_t sA;
  cudaStreamCreate(&sA);
  gzkp::msm::FromMonty<Fr_type>(sA, scalars, size);
  int win_nums = (ScalarBit + window_size - 1) / window_size; 
  auto result = gzkp::msm::msm_pippenger<G1, ScalarBit>(sA, points, scalars, size, window_size, reduce_size, win_nums);

  size_t JAC_POINT_LIMBS_ECp_MNT4 = 3 * G1::field_type::DEGREE * G1::field_type::ELT_LIMBS * sizeof(var);
  auto out_ = (var *)malloc(JAC_POINT_LIMBS_ECp_MNT4);
  cudaMemcpy(out_, result, JAC_POINT_LIMBS_ECp_MNT4, cudaMemcpyDeviceToHost);
  // B::G1 *evaluation = B::read_pt_ECp(out_);
  printf("evaluation_At:\n");
  B::print_G1(B::read_pt_ECp(out_));
}

int main(int argc, char *argv[]) {
  setbuf(stdout, NULL);
  std::string curve(argv[1]);
  const char *parameters_file_path = argv[2];
  const char *inputs_file_path = argv[3];

  gzkp::File file(parameters_file_path, inputs_file_path);
  auto &parameters = file.parameters;
  auto &inputs = file.inputs;
  size_t size = parameters.m+1;
  int window_size = atoi(argv[4]);
  int reduce_size = atoi(argv[5]);
  
  var *scalars = nullptr;
  cudaMalloc(&scalars, (size) * ELT_LIMBS_R * sizeof(uint64_t));
  cudaMemcpy(scalars, inputs.w, (size) * ELT_LIMBS_R * sizeof(uint64_t), cudaMemcpyHostToDevice);
  // load points for pippenger
  auto points = gzkp::msm::load_points_affine<G1>(size, parameters.a);

  if (curve == "MNT4753") {
    test_MSM<mnt4753_libsnark>(size, window_size, reduce_size, scalars, points);
  } else if (curve == "BLS12_381") {
    test_MSM<bls12_381_libsnark>(size, window_size, reduce_size, scalars, points);
  } else if (curve == "ALT_BN128") {
    test_MSM<alt_bn128_libsnark>(size, window_size, reduce_size, scalars, points);    
  }
  return 0;
}