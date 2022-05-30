#include <string>
#include <chrono>
#include <iostream>

#define NDEBUG 1

#define USE_UNIFIED_MEMORY
// #define SERIAL_MULTI_EXP

#include <prover_reference_functions.hpp>

#include "multiexp/reduce.cu"

// This is where all the FFTs happen

// template over the bundle of types and functions.
// Overwrites ca!
template <typename B>
typename B::vector_Fr *compute_H(size_t d, typename B::vector_Fr *ca,
                                 typename B::vector_Fr *cb,
                                 typename B::vector_Fr *cc) {
  auto domain = B::get_evaluation_domain(d + 1);

  B::domain_iFFT(domain, ca);
  B::domain_iFFT(domain, cb);

  B::domain_cosetFFT(domain, ca);
  B::domain_cosetFFT(domain, cb);

  // Use ca to store H
  auto H_tmp = ca;

  size_t m = B::domain_get_m(domain);
  // for i in 0 to m: H_tmp[i] *= cb[i]
  B::vector_Fr_muleq(H_tmp, cb, m);

  B::domain_iFFT(domain, cc);
  B::domain_cosetFFT(domain, cc);

  m = B::domain_get_m(domain);

  // for i in 0 to m: H_tmp[i] -= cc[i]
  B::vector_Fr_subeq(H_tmp, cc, m);

  B::domain_divide_by_Z_on_coset(domain, H_tmp);

  B::domain_icosetFFT(domain, H_tmp);

  m = B::domain_get_m(domain);
  typename B::vector_Fr *H_res = B::vector_Fr_zeros(m + 1);
  B::vector_Fr_copy_into(H_tmp, H_res, m);
  return H_res;
}

static size_t read_size_t(FILE* input) {
  size_t n;
  fread((void *) &n, sizeof(size_t), 1, input);
  return n;
}

template< typename B >
struct ec_type;

template<>
struct ec_type<mnt4753_libsnark> {
    typedef ECp_MNT4 ECp;
    typedef ECp2_MNT4 ECpe;
};

template<>
struct ec_type<mnt6753_libsnark> {
    typedef ECp_MNT6 ECp;
    typedef ECp3_MNT6 ECpe;
};


void
check_trailing(FILE *f, const char *name) {
    long bytes_remaining = 0;
    while (fgetc(f) != EOF)
        ++bytes_remaining;
    if (bytes_remaining > 0)
        fprintf(stderr, "!! Trailing characters in \"%s\": %ld\n", name, bytes_remaining);
}


static inline auto now() -> decltype(std::chrono::high_resolution_clock::now()) {
    return std::chrono::high_resolution_clock::now();
}

template<typename T>
void
print_time(T &t1, const char *str) {
    auto t2 = std::chrono::high_resolution_clock::now();
    auto tim = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count();
    printf("%s: %ld ms\n", str, tim);
    t1 = t2;
}

void debug_dump_data(var *data, size_t N) {
    uint64_t* buffer;
    uint64_t total_bytes = sizeof(uint64_t) * (12 * N);
    buffer = (uint64_t*)malloc(total_bytes);
    cudaMemcpy(buffer, data, total_bytes, cudaMemcpyDeviceToHost);
    for (int i = 0; i < N; ++i) {
      for (int j = 0; j < 12; ++j) {
        std::cout << buffer[i * 12 + j] << " \n"[j + 1 == 12];
      }
    }
    std::cout << std::flush;
}

template <typename B>
void run_prover(
        const char *params_path,
        const char *input_path,
        const char *output_path,
        const char *preprocessed_path)
{
    B::init_public_params();

    size_t primary_input_size = 1;

    auto beginning = now();
    auto t = beginning;

    FILE *params_file = fopen(params_path, "r");
    size_t d = read_size_t(params_file);
    size_t m = read_size_t(params_file);
    rewind(params_file);

    printf("d = %zu, m = %zu\n", d, m);

    typedef typename ec_type<B>::ECp ECp;
    typedef typename ec_type<B>::ECpe ECpe;

    typedef typename B::G1 G1;
    typedef typename B::G2 G2;
    // static constexpr int R = 32;
    static constexpr int R = 256;
    static constexpr int C = 5;
#ifdef SERIAL_MULTI_EXP
    auto t_main = now();
    t = t_main;
    FILE *inputs_file = fopen(input_path, "r");
    auto w_ = load_scalars(m + 1, inputs_file);
    rewind(inputs_file);
    auto inputs = B::read_input(inputs_file, d, m);
    fclose(inputs_file);
    const var *w = w_.get();
    print_time(t, "load inputs");

    auto t_gpu = now();
    auto coefficients_for_H = compute_H<B>(d, B::input_ca(inputs), B::input_cb(inputs), B::input_cc(inputs));
    print_time(t, "cpu H");
    auto H_ = convert_scalars_format(coefficients_for_H, d);
    const var *H = H_.get();
    print_time(t, "convert H");

    FILE *preprocessed_file = fopen(preprocessed_path, "r");

    size_t space = ((m + 1) + R - 1) / R;
    size_t space_H = (d + R - 1) / R;
    cudaStream_t sA;

    auto A_mults = load_points_affine<ECp>(((1U << C) - 1)*(m + 1), preprocessed_file);
    print_time(t, "load A mults");
    auto out_A = allocate_memory(space * ECpe::NELTS * ELT_BYTES);
    ec_reduce_straus<ECp, C, R>(sA, out_A.get(), A_mults.get(), w, m + 1);
    cudaStreamSynchronize(sA);
    print_time(t, "multi-scalar multiplication A done");
    cudaFree(A_mults.get());
    t=now();

    auto B1_mults = load_points_affine<ECp>(((1U << C) - 1)*(m + 1), preprocessed_file);
    print_time(t, "load B1 mults");
    auto out_B1 = allocate_memory(space * ECpe::NELTS * ELT_BYTES);
    ec_reduce_straus<ECp, C, R>(sA, out_B1.get(), B1_mults.get(), w, m + 1);
    cudaStreamSynchronize(sA);
    print_time(t, "multi-scalar multiplication B1 done");
    cudaFree(B1_mults.get());
    t=now();

    auto B2_mults = load_points_affine<ECpe>(((1U << C) - 1)*(m + 1), preprocessed_file);
    print_time(t, "load B2 mults");
    auto out_B2 = allocate_memory(space * ECpe::NELTS * ELT_BYTES);
    ec_reduce_straus<ECpe, C, 2*R>(sA, out_B2.get(), B2_mults.get(), w, m + 1);
    cudaStreamSynchronize(sA);
    print_time(t, "multi-scalar multiplication B2 done");
    cudaFree(B2_mults.get());
    t=now();

    auto L_mults = load_points_affine<ECp>(((1U << C) - 1)*(m - 1), preprocessed_file);
    print_time(t, "load L mults");
    auto out_L = allocate_memory(space * ECpe::NELTS * ELT_BYTES);
    ec_reduce_straus<ECp, C, R>(sA, out_L.get(), L_mults.get(), w + (primary_input_size + 1) * ELT_LIMBS, m - 1);
    cudaStreamSynchronize(sA);
    print_time(t, "multi-scalar multiplication L done");
    cudaFree(L_mults.get());
    t=now();

    auto H_mults = load_points_affine<ECp_MNT4>(((1U << C) - 1) * (d), preprocessed_file);
    print_time(t, "load H mults");
    auto out_H = allocate_memory(space_H * ECpe::NELTS * ELT_BYTES);
    ec_reduce_straus<ECp, C, R>(sA, out_H.get(), H_mults.get(), H, d);
    cudaDeviceSynchronize();
    print_time(t, "multi-scalar multiplication H done");
    cudaFree(H_mults.get());
    fclose(preprocessed_file);
    


#else
    FILE *preprocessed_file = fopen(preprocessed_path, "r");

    size_t space = ((m + 1) + R - 1) / R;
    size_t space_H = (d + R - 1) / R;

    auto A_mults = load_points_affine<ECp>(((1U << C) - 1)*(m + 1), preprocessed_file);
    auto out_A = allocate_memory(space * ECpe::NELTS * ELT_BYTES);

    auto B1_mults = load_points_affine<ECp>(((1U << C) - 1)*(m + 1), preprocessed_file);
    auto out_B1 = allocate_memory(space * ECpe::NELTS * ELT_BYTES);

    auto B2_mults = load_points_affine<ECpe>(((1U << C) - 1)*(m + 1), preprocessed_file);
    auto out_B2 = allocate_memory(space * ECpe::NELTS * ELT_BYTES);

    auto L_mults = load_points_affine<ECp>(((1U << C) - 1)*(m - 1), preprocessed_file);
    auto out_L = allocate_memory(space * ECpe::NELTS * ELT_BYTES);

    auto H_mults = load_points_affine<ECp_MNT4>(((1U << C) - 1) * (d), preprocessed_file);
    auto out_H = allocate_memory(space_H * ECpe::NELTS * ELT_BYTES);

    fclose(preprocessed_file);

    print_time(t, "load preprocessing");

    auto t_main = t;

    FILE *inputs_file = fopen(input_path, "r");
    auto w_ = load_scalars(m + 1, inputs_file);
    rewind(inputs_file);
    auto inputs = B::read_input(inputs_file, d, m);
    fclose(inputs_file);
    const var *w = w_.get();
    print_time(t, "load inputs");


    auto t_gpu = t;
    auto coefficients_for_H = compute_H<B>(d, B::input_ca(inputs), B::input_cb(inputs), B::input_cc(inputs));
    print_time(t, "cpu H");
    auto H_ = convert_scalars_format(coefficients_for_H, d);
    const var *H = H_.get();
    print_time(t, "convert H");

    cudaStream_t sA, sB1, sB2, sL, sH;
    auto t_mul = t;
    ec_reduce_straus<ECp, C, R>(sA, out_A.get(), A_mults.get(), w, m + 1);
    cudaStreamSynchronize(sA);
    print_time(t_mul, "multi-scalar multiplication A done");
    ec_reduce_straus<ECpe, C, 2*R>(sB2, out_B2.get(), B2_mults.get(), w, m + 1);
    cudaStreamSynchronize(sB2);
    print_time(t_mul, "multi-scalar multiplication B2 done");
    ec_reduce_straus<ECp, C, R>(sB1, out_B1.get(), B1_mults.get(), w, m + 1);
    ec_reduce_straus<ECp, C, R>(sL, out_L.get(), L_mults.get(), w + (primary_input_size + 1) * ELT_LIMBS, m - 1);
    ec_reduce_straus<ECp, C, R>(sH, out_H.get(), H_mults.get(), H, d);
    // print_time(t, "gpu launch");
    
    cudaDeviceSynchronize();
    print_time(t, "multi-scalar multiplication done");
#endif

#ifdef USE_UNIFIED_MEMORY
    cudaStreamSynchronize(sA);
    G1 *evaluation_At = B::read_pt_ECp(out_A.get());
    cudaStreamSynchronize(sB1);
    G1 *evaluation_Bt1 = B::read_pt_ECp(out_B1.get());
    cudaStreamSynchronize(sB2);
    G2 *evaluation_Bt2 = B::read_pt_ECpe(out_B2.get());
    cudaStreamSynchronize(sL);
    G1 *evaluation_Lt = B::read_pt_ECp(out_L.get());
    cudaStreamSynchronize(sH);
    G1 *evaluation_Ht = B::read_pt_ECp(out_H.get());
#else
    size_t JAC_POINT_LIMBS_ECp_MNT4 = 3 * ECp_MNT4::field_type::DEGREE * ELT_BYTES;
    size_t JAC_POINT_LIMBS_ECp2_MNT4 = 3 * ECp2_MNT4::field_type::DEGREE * ELT_BYTES;
    auto out_A_ = (var *)malloc(JAC_POINT_LIMBS_ECp_MNT4);
    auto out_B1_ = (var *)malloc(JAC_POINT_LIMBS_ECp_MNT4);
    auto out_L_ = (var *)malloc(JAC_POINT_LIMBS_ECp_MNT4);
    auto out_H_ = (var *)malloc(JAC_POINT_LIMBS_ECp_MNT4);
    auto out_B2_ = (var *)malloc(JAC_POINT_LIMBS_ECp2_MNT4);
    cudaMemcpy(out_A_, out_A.get(), JAC_POINT_LIMBS_ECp_MNT4, cudaMemcpyDeviceToHost);
    cudaMemcpy(out_B1_, out_B1.get(), JAC_POINT_LIMBS_ECp_MNT4, cudaMemcpyDeviceToHost);
    cudaMemcpy(out_L_, out_L.get(), JAC_POINT_LIMBS_ECp_MNT4, cudaMemcpyDeviceToHost);
    cudaMemcpy(out_H_, out_H.get(), JAC_POINT_LIMBS_ECp_MNT4, cudaMemcpyDeviceToHost);
    cudaMemcpy(out_B2_, out_B2.get(), JAC_POINT_LIMBS_ECp2_MNT4, cudaMemcpyDeviceToHost);

    G1 *evaluation_At = B::read_pt_ECp(out_A_);
    G1 *evaluation_Bt1 = B::read_pt_ECp(out_B1_);
    G2 *evaluation_Bt2 = B::read_pt_ECpe(out_B2_);
    G1 *evaluation_Lt = B::read_pt_ECp(out_L_);
    G1 *evaluation_Ht = B::read_pt_ECp(out_H_);
#endif    

    printf("evaluation_At:\n");
    B::print_G1(evaluation_At);
    printf("evaluation_Bt1:\n");
    B::print_G1(evaluation_Bt1);
    printf("evaluation_Bt2:\n");
    B::print_G2(evaluation_Bt2);
    printf("evaluation_Ht:\n");
    B::print_G1(evaluation_Ht);
    printf("evaluation_Lt:\n");
    B::print_G1(evaluation_Lt);
    print_time(t_gpu, "gpu e2e");
    t = t_gpu;
    auto scaled_Bt1 = B::G1_scale(B::input_r(inputs), evaluation_Bt1);
    // printf("scaled_Bt1:\n");
    // B::print_G1(scaled_Bt1);
    auto Lt1_plus_scaled_Bt1 = B::G1_add(evaluation_Lt, scaled_Bt1);
    // printf("Lt1_plus_scaled_Bt1:\n");
    // B::print_G1(Lt1_plus_scaled_Bt1);
    auto final_C = B::G1_add(evaluation_Ht, Lt1_plus_scaled_Bt1);

    // printf("final_C:\n");
    // B::print_G1(final_C);
    
    print_time(t, "cpu 2");

    B::groth16_output_write(evaluation_At, evaluation_Bt2, final_C, output_path);

    print_time(t, "store");

    print_time(t_main, "Total time from input to output: ");

#ifdef SERIAL_MULTI_EXP
    cudaStreamDestroy(sA);
#else 
    cudaStreamDestroy(sA);
    cudaStreamDestroy(sB1);
    cudaStreamDestroy(sB2);
    cudaStreamDestroy(sL);
    cudaStreamDestroy(sH);
#endif

    B::delete_G1(evaluation_At);
    B::delete_G1(evaluation_Bt1);
    B::delete_G2(evaluation_Bt2);
    B::delete_G1(evaluation_Ht);
    B::delete_G1(evaluation_Lt);
    B::delete_G1(scaled_Bt1);
    B::delete_G1(Lt1_plus_scaled_Bt1);
    B::delete_vector_Fr(coefficients_for_H);
    B::delete_groth16_input(inputs);

    print_time(t, "cleanup");
    print_time(beginning, "Total runtime (incl. file reads)");
}

int main(int argc, char **argv) {
  setbuf(stdout, NULL); 
  std::string curve(argv[1]);
  std::string mode(argv[2]);

  const char *params_path = argv[3];

  if (mode == "compute") {
      const char *input_path = argv[4];
      const char *output_path = argv[5];
      const char *preprocessed_path = argv[6];

      if (curve == "MNT4753") {
          run_prover<mnt4753_libsnark>(params_path, input_path, output_path, preprocessed_path);
      } else if (curve == "MNT6753") {
          run_prover<mnt6753_libsnark>(params_path, input_path, output_path, preprocessed_path);
      }
  } else if (mode == "preprocess") {
#if 0
      if (curve == "MNT4753") {
          run_preprocess<mnt4753_libsnark>(params_path);
      } else if (curve == "MNT6753") {
          run_preprocess<mnt4753_libsnark>(params_path);
      }
#endif
  }

  return 0;
}
