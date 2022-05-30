#include "src/gsl/assert.h"

#include "src/ntt.h"
#include "src/ntt_field.h"

#include <array>
#include <iostream>
#include <memory>
#include <vector>

#include "src/file.h"
#include "src/testlib.h"

namespace cuzk {

template <int n_size> class Main {
public:
  Main() {
    gzkp::ntt::initialize_constants<n_size>();
    cudaMalloc(&powers_, BYTES << n1);
    cudaMalloc(&twiddles_, BYTES << n);
    cudaMalloc(&a, BYTES << n);
    cudaMalloc(&b, BYTES << n);
    cudaMalloc(&c, BYTES << n);
    cudaMalloc(&out, BYTES << n);
  }

  ~Main() {
    cudaFree(out);
    cudaFree(c);
    cudaFree(b);
    cudaFree(a);
    cudaFree(twiddles_);
    cudaFree(powers_);
  }

  static void debug_dump_data(uint64_t *d_data) {
    static uint64_t buffer[LIMBS << n];
    copy_to_host(buffer, d_data);
    for (int i = 0; i < 1 << n; ++i) {
      for (int j = 0; j < LIMBS; ++j) {
        std::cout << buffer[i * LIMBS + j] << " \n"[j + 1 == LIMBS];
      }
    }
    std::cout << std::flush;
  }

  void debug_dump_out() {
    debug_dump_data(out);
  }

  void run(const Inputs &input) {

    cudaStream_t stream0, stream1;
    cudaStreamCreate(&stream0);
    cudaStreamCreate(&stream1);

    gzkp::ntt::prepare_powers<n_size>(powers_, stream0);
    cudaMemcpyAsync(out, input.a, BYTES << n, cudaMemcpyHostToDevice, stream1);
    gzkp::ntt::prepare_dif_twiddles<n>(powers_, twiddles_, stream0);
    cudaStreamSynchronize(stream0);

    cudaStreamSynchronize(stream1);
    gzkp::ntt::reshape_input<n>(out, a, stream0);
    cudaStreamSynchronize(stream0);
    cudaMemcpyAsync(out, input.b, BYTES << n, cudaMemcpyHostToDevice, stream1);

    gzkp::ntt::decimation_in_freq<n>(twiddles_, a, stream0);
    
    cudaStreamSynchronize(stream1);
    gzkp::ntt::reshape_input<n>(out, b, stream0);
    cudaStreamSynchronize(stream0);
    cudaMemcpyAsync(out, input.c, BYTES << n, cudaMemcpyHostToDevice, stream1);
    gzkp::ntt::decimation_in_freq<n>(twiddles_, b, stream0);
    
    cudaStreamSynchronize(stream1);
    gzkp::ntt::reshape_input<n>(out, c, stream0);
    cudaStreamSynchronize(stream0);
    gzkp::ntt::decimation_in_freq<n>(twiddles_, c, stream0);

    cudaStreamSynchronize(stream0);

    gzkp::ntt::prepare_dit_twiddles<n>(powers_, twiddles_, stream0);
    cudaStreamSynchronize(stream0);

    gzkp::ntt::decimation_in_time<n>(twiddles_, a, stream0);
    gzkp::ntt::decimation_in_time<n>(twiddles_, b, stream0);
    gzkp::ntt::decimation_in_time<n>(twiddles_, c, stream0);

    cudaStreamSynchronize(stream0);

    gzkp::ntt::dot_product<n>(out, a, b, c, stream0);

    gzkp::ntt::prepare_inv_dif_twiddles<n>(powers_, twiddles_, stream0);
    gzkp::ntt::decimation_in_freq<n>(twiddles_, out, stream0);

    gzkp::ntt::general_bit_reverse<n>(out, stream0);

    gzkp::ntt::p_reduce<n>(out, stream0);

    cudaStreamSynchronize(stream0);
  }


  uint64_t *a, *b, *c, *out;

private:
  static const int n = n_size;
  static const int n1 = n + 1;

  static void copy_to_device(uint64_t *d, const uint64_t *h) {
    cudaMemcpy(d, h, BYTES << n, cudaMemcpyHostToDevice);
  }

  static void copy_to_host(uint64_t *h, const uint64_t *d) {
    // 24由于栈空间的影响无法分配,改成malloc
    uint64_t* buffer;
    buffer = (uint64_t*)malloc(sizeof(uint64_t) * (LIMBS << n));
    cudaMemcpy(buffer, d, BYTES << n, cudaMemcpyDeviceToHost);
    for (int batch = 0; batch < 3; ++batch) {
      for (int i = 0; i < 1 << n; ++i) {
        for (int lane = 0; lane < 4; ++lane) {
          h[i * 12 + lane * 3 + batch] =
              buffer[batch << (n + 2) | i << 2 | lane];
        }
      }
    }
  }

  uint64_t *powers_, *twiddles_;
};

} // namespace cuzk

static void print_usage(const char *argv[]) {
  std::cerr << "Usage: " << argv[0] << " parameter_file input_file"
            << std::endl;
}

int main(int argc, const char *argv[]) {
  using namespace cuzk;

  if (2 >= argc) {
    print_usage(argv);
    return -1;
  }

  int *cuda_init;
  cudaMalloc(&cuda_init, sizeof(int));

  const char *parameters_file_path = argv[1];
  const char *inputs_file_path = argv[2];
  // const char *output_file_path = argv[3];

  File file(parameters_file_path, inputs_file_path);
  // auto &parameters = file.parameters;
  auto &inputs = file.inputs;

  // size_t d = parameter.d;
  // size_t m = parameters.m;

  static const int n = 24;

  // Expects(m == n);
  // std::cerr << "DEBUG|d=" << d << std::endl;

  // for (int i = 0; i <= d; ++i) {
  //   for (int j = 0; j < LIMBS; ++j) {
  //     std::cout << input.a[i * LIMBS + j] << " \n"[j + 1 == LIMBS];
  //   }
  // }
  // std::cout << std::endl;

  auto t0 = std::chrono::high_resolution_clock::now();
  Main<n> main;
  // uint64_t *out = new uint64_t[LIMBS << (n + n)];
  main.run(inputs);
  cudaDeviceSynchronize();

  auto t1 = std::chrono::high_resolution_clock::now();

  main.debug_dump_out();

  std::cout << "BREAKDOWN|COMPUTE_H="
            << std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0)
                       .count() /
                   1000000.0
            << std::endl;

  inputs.free();


  // FILE *output_file = fopen(output_file_path, "w");
  // Ensures(fwrite(out, sizeof(uint64_t), (d + 1) * LIMBS, output_file) == (d +
  // 1) * LIMBS); fclose(output_file);

  // std::cerr << "OK" << std::endl;
}
