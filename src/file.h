#pragma once

#include "common.h"

#include <cstdint>
#include <cstdio>
#include <vector>
#include <cuda.h>
#include <cuda_runtime.h>
#include "gsl/assert.h"

namespace gzkp {

template <int degree>
static inline uint64_t* read_uint64s(FILE *f, size_t count) {
  // std::vector<uint64_t> buffer(count * degree * LIMBS);
  uint64_t *buffer;
  uint64_t buffer_size = count * degree * LIMBS;
  cudaError_t cudaStatus = cudaHostAlloc((void **)&buffer, buffer_size * sizeof(uint64_t), cudaHostAllocDefault);
  if (cudaStatus != cudaSuccess)
	{
    printf("CUDA Error: %s\n", cudaGetErrorString(cudaPeekAtLastError()));
		exit(-1);
	}
  Ensures(fread(buffer, sizeof(uint64_t), buffer_size, f) ==
          buffer_size);
  return buffer;
}

struct Parameters {
  static Parameters load(FILE *f) {
    rewind(f);

    size_t d, m;

    Ensures(fread(&d, sizeof(size_t), 1, f) == 1);
    Ensures(fread(&m, sizeof(size_t), 1, f) == 1);
    auto a = read_uint64s<2>(f, m + 1);
    auto b1 = read_uint64s<2>(f, m + 1);
    auto b2 = read_uint64s<4>(f, m + 1);
    auto l = read_uint64s<2>(f, m - 1);
    auto h = read_uint64s<2>(f, d);
    return Parameters{d, m, a, b1, b2, l, h};
  }
  void free() {
    cudaFreeHost(a);
    cudaFreeHost(b1);
    cudaFreeHost(b2);
    cudaFreeHost(l);
    cudaFreeHost(h);
  }
  size_t d, m;
  // std::vector<uint64_t> a, b1, b2, l, h;
  uint64_t *a, *b1, *b2, *l, *h;
};

struct Inputs {
  static Inputs load(const Parameters &p, FILE *f) {
    rewind(f);
    auto w = read_uint64s<1>(f, p.m + 1);
    auto a = read_uint64s<1>(f, p.m);
    auto b = read_uint64s<1>(f, p.m);
    auto c = read_uint64s<1>(f, p.m);
    auto r = read_uint64s<1>(f, 1);
    return Inputs{w, a, b, c, r};
  }

  void free() {
    cudaFreeHost(w);
    cudaFreeHost(aA);
    cudaFreeHost(aB);
    cudaFreeHost(aC);
    cudaFreeHost(r);
  }
  // std::vector<uint64_t> w, a, b, c;
  uint64_t *w, *aA, *aB, *aC, *r;
};

#ifdef BLS12_381
template <int degree, int LIMBS>
static inline uint64_t* read_uint64s_(FILE *f, size_t count) {
  // std::vector<uint64_t> buffer(count * degree * LIMBS);
  uint64_t *buffer;
  uint64_t buffer_size = count * degree * LIMBS;
  cudaError_t cudaStatus = cudaHostAlloc((void **)&buffer, buffer_size * sizeof(uint64_t), cudaHostAllocDefault);
  if (cudaStatus != cudaSuccess)
	{
		printf("host alloc fail!\n");
		exit(-1);
	}
  Ensures(fread(buffer, sizeof(uint64_t), buffer_size, f) ==
          buffer_size);
  return buffer;
}

struct Parameters_for_app {

  static Parameters_for_app load(FILE *f) {
    size_t aa, bb, ll, hh;
    Ensures(fread((void *) &aa, sizeof(size_t), 1, f) == 1);
    Ensures(fread((void *) &bb, sizeof(size_t), 1, f) == 1);
    Ensures(fread((void *) &ll, sizeof(size_t), 1, f) == 1);
    Ensures(fread((void *) &hh, sizeof(size_t), 1, f) == 1);

    auto a = read_uint64s_<2, 6>(f, aa);
    auto b1 = read_uint64s_<2, 6>(f, bb);
    auto b2 = read_uint64s_<4, 6>(f, bb);
    auto l = read_uint64s_<2, 6>(f, ll);
    auto h = read_uint64s_<2, 6>(f, hh);
    return Parameters_for_app{aa, bb, ll, hh, a, b1, b2, l, h};
  }

  void free() {
    cudaFreeHost(a);
    cudaFreeHost(b1);
    cudaFreeHost(b2);
    cudaFreeHost(l);
    cudaFreeHost(h);
  }
  
  size_t a_length, b_length, l_length, h_length;
  uint64_t *a, *b1, *b2, *l, *h;
};

struct Inputs_for_app {
  static Inputs_for_app load(const Parameters_for_app &p, FILE *f) {
    rewind(f);
    auto wa = read_uint64s_<1, 4>(f, p.a_length);
    auto wb = read_uint64s_<1, 4>(f, p.b_length);
    auto wl = read_uint64s_<1, 4>(f, p.l_length);
    auto wh = read_uint64s_<1, 4>(f, p.h_length);
    // auto r = read_uint64s_<1, 4>(f, 1);
    // auto a = read_uint64s_<1, 4>(f, p.h_length + 1);
    // auto b = read_uint64s_<1, 4>(f, p.h_length + 1);
    // auto c = read_uint64s_<1, 4>(f, p.h_length + 1);
    return Inputs_for_app{wa, wb, wl, wh, nullptr, nullptr, nullptr, nullptr};
  }

  void free() {
    cudaFreeHost(sa);
    cudaFreeHost(sb);
    cudaFreeHost(sl);
    cudaFreeHost(sh);
    cudaFreeHost(aA);
    cudaFreeHost(aB);
    cudaFreeHost(aC);
    cudaFreeHost(sr);
  }

  uint64_t *sa, *sb, *sl, *sh, *sr;
  uint64_t *aA, *aB, *aC;
};
#else
struct Parameters_for_app {
  static Parameters_for_app load(FILE *f) {
    rewind(f);
    size_t a_length, b_length, l_length, h_length;
    Ensures(fread(&a_length, sizeof(size_t), 1, f) == 1);
    Ensures(fread(&b_length, sizeof(size_t), 1, f) == 1);
    Ensures(fread(&l_length, sizeof(size_t), 1, f) == 1);
    Ensures(fread(&h_length, sizeof(size_t), 1, f) == 1);
    // Ensures(h_length == m-1);
    auto a = read_uint64s<2>(f, a_length);
    auto b1 = read_uint64s<2>(f, b_length);
    auto b2 = read_uint64s<4>(f, b_length);
    auto l = read_uint64s<2>(f, l_length);
    auto h = read_uint64s<2>(f, h_length);
    return Parameters_for_app{a_length, b_length, l_length, h_length, a, b1, b2, l, h};
  }

  void free() {
    cudaFreeHost(a);
    cudaFreeHost(b1);
    cudaFreeHost(b2);
    cudaFreeHost(l);
    cudaFreeHost(h);
  }

  size_t a_length, b_length, l_length, h_length;
  uint64_t *a, *b1, *b2, *l, *h;
};

struct Inputs_for_app {
  static Inputs_for_app load(const Parameters_for_app &p, FILE *f) {
    rewind(f);
    // Ensures((h_length == m-1) && (aA_length == m));
    auto a = read_uint64s<1>(f, p.a_length);
    auto b = read_uint64s<1>(f, p.b_length);
    auto l = read_uint64s<1>(f, p.l_length);
    auto h = read_uint64s<1>(f, p.h_length);
    auto r = read_uint64s<1>(f, 1);
    auto aA = read_uint64s<1>(f, p.h_length + 1);
    auto aB = read_uint64s<1>(f, p.h_length + 1);
    auto aC = read_uint64s<1>(f, p.h_length + 1);
    return Inputs_for_app{a, b, l, h, r, aA, aB, aC};
  }

  void free() {
    cudaFreeHost(sa);
    cudaFreeHost(sb);
    cudaFreeHost(sl);
    cudaFreeHost(sh);
    cudaFreeHost(sr);
    cudaFreeHost(aA);
    cudaFreeHost(aB);
    cudaFreeHost(aC);
  }
  
  uint64_t *sa, *sb, *sl, *sh, *sr;
  uint64_t *aA, *aB, *aC;
};
#endif

struct File {
  File(const char *parameters_file_path, const char *inputs_file_path) {
    FILE *parameters_file = fopen(parameters_file_path, "r");
    Ensures(parameters_file);
    parameters = Parameters::load(parameters_file);
    fclose(parameters_file);
    FILE *inputs_file = fopen(inputs_file_path, "r");
    Ensures(inputs_file);
    inputs = Inputs::load(parameters, inputs_file);
    fclose(inputs_file);
  }

  File(const char *parameters_file_path, const char *inputs_file_path, uint64_t m) {
    FILE *parameters_file = fopen(parameters_file_path, "r");
    Ensures(parameters_file);
    parameters_for_app = Parameters_for_app::load(parameters_file);
    fclose(parameters_file);
    FILE *inputs_file = fopen(inputs_file_path, "r");
    Ensures(inputs_file);
    inputs_for_app =  Inputs_for_app::load(parameters_for_app, inputs_file);
    fclose(inputs_file);
  }

  Parameters parameters;
  Inputs inputs;
  Inputs_for_app inputs_for_app;
  Parameters_for_app parameters_for_app;
};

} // namespace gzkp
