#pragma once

#include <chrono>
#include <cooperative_groups.h>
#include <cstdint>
#include <cstring>
#include <inttypes.h>
#include <memory>
#include <tuple>
#include <vector>

#include <cuda.h>
#include <cuda_runtime.h>

#include "ff/curves.cuh"


static inline auto now() -> decltype(std::chrono::high_resolution_clock::now()) {
    return std::chrono::high_resolution_clock::now();
}

template<typename T>
std::string
print_time(T &t1, const char *str) {
    auto t2 = std::chrono::high_resolution_clock::now();
    auto tim = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count();
    printf("%s: %ld ms\n", str, tim);
    t1 = t2;
    return std::to_string(tim);
}

template <int ELT_LIMBS>
void debug_dump_data_1(int *data, size_t N, int bytes) {
  size_t total_bytes = N * ELT_LIMBS * bytes;

  int * buffer = (int *)malloc(total_bytes);
  cudaMemcpy(buffer, data, total_bytes, cudaMemcpyDeviceToHost);
  for (int i = 0; i < N; ++i) {
      for (int j = 0; j < ELT_LIMBS; ++j) {
          std::cout << buffer[i * ELT_LIMBS + j] << " \n"[j + 1 == ELT_LIMBS];
      }
  }
  std::cout << std::flush;
}

namespace gzkp {
namespace msm {

template <typename Fr, int ScalarBit>
__global__ void PreprocessScalars1(const var *scalars, int *count, size_t N, int level, int C, int win_num) {
  const int T = threadIdx.x, B = blockIdx.x, D = blockDim.x;
  const int total_u64 = N * Fr::ELT_LIMBS;
  const int batch_size = total_u64 / level; 
  var C_MASK = (1U << C) - 1U;
  int n, b, end, addr_start, addr_end, i, r, bottom_bits, W;
  var s, s_plus;

  end = (B + 1) * batch_size;
  if(B == level - 1)
    end = total_u64; 
  
  for(int j = B * batch_size; j < end; j += D){
    n = j + T;
    b = n % Fr::ELT_LIMBS; 
    if(n < end){
      addr_start = b * digit::BITS; 
      addr_end = (b + 1) * digit::BITS - 1;

      s = *(scalars + n);

      if(b == (ScalarBit - 1) / 64) 
        addr_end = C * ((ScalarBit + C - 1) / C) - 1;
      i = addr_end - (addr_end % C);

      while(i >= addr_start){
        r = i % digit::BITS;
        var win = (s >> r) & C_MASK;
        // Handle case where C doesn't divide digit::BITS
        bottom_bits = digit::BITS - r;
        // detect when window overlaps digit boundary
        if (bottom_bits < C && (b != (ScalarBit - 1) / 64)) {
          s_plus = *(scalars + n + 1); 
          win |= (s_plus << bottom_bits) & C_MASK;
        }
        if(win != 0){
          W = win_num - 1 - i / C;
          atomicAdd(&count[W * (1 << C) + win], 1);
        }
        i -= C;
      }
    }
  }
}

template <int ScalarBit>
__global__ void PreprocessScalars2(int *wins_start, int *wins_end, int *count, int *pos, size_t N, int C, int win_num) {
  const int T = threadIdx.x, B = blockIdx.x;
  int temp;

  if(B < win_num && T == 0){
    wins_start[B * (1 << C) + 0] = 0;
    pos[B * (1 << C) + 0] = 0;
    for(int i = 1; i < (1 << C); i++){
      temp = wins_start[B * (1 << C) + i - 1] + count[B * (1 << C) + i - 1];
      wins_start[B * (1 << C) + i] = temp;
      pos[B * (1 << C) + i] = temp;
      wins_end[B * (1 << C) + i - 1] = temp;
    }
    wins_end[B * (1 << C) + (1 << C) - 1] = wins_start[B * (1 << C) + (1 << C) - 1] + count[B * (1 << C) + (1 << C) - 1];
  }
}

template <typename Fr, int ScalarBit>
__global__ void PreprocessScalars3(const var *scalars, int *wins, int *pos, size_t N, int level, int C, int win_num) {
  const int T = threadIdx.x, B = blockIdx.x, D = blockDim.x;
  const int total_u64 = N * Fr::ELT_LIMBS;
  const int batch_size = total_u64 / level; 
  var C_MASK = (1U << C) - 1U;
  int n, b, end, addr_start, addr_end, i, r, bottom_bits, pos_old, W;
  var s, s_plus;

  end = (B + 1) * batch_size;
  if(B == level - 1)
    end = total_u64; 
  
  for(int j = B * batch_size; j < end; j += D){
    n = j + T;
    b = n % Fr::ELT_LIMBS; 
    if(n < end){
      addr_start = b * digit::BITS; 
      addr_end = (b + 1) * digit::BITS - 1;

      s = *(scalars + n);

      if(b == (ScalarBit - 1) / 64) 
        addr_end = C * ((ScalarBit + C - 1) / C) - 1;
      i = addr_end - (addr_end % C);

      while(i >= addr_start){
        r = i % digit::BITS;
        var win = (s >> r) & C_MASK;
        // Handle case where C doesn't divide digit::BITS
        bottom_bits = digit::BITS - r;
        // detect when window overlaps digit boundary
        if (bottom_bits < C && (b != (ScalarBit - 1) / 64)) {
          s_plus = *(scalars + n + 1); 
          win |= (s_plus << bottom_bits) & C_MASK;
        }
        if(win != 0){
          W = win_num - 1 - i / C;
          pos_old = atomicAdd(&pos[W * (1 << C) + win], 1);
          wins[W * N + pos_old] = n / Fr::ELT_LIMBS;
        }
        i -= C;
      }
    }
    __syncthreads();
  }
}

template<typename EC, int ScalarBit>
void PreprocessScalars(cudaStream_t &strm, const var *scalars, int *wins, int *wins_start, int *wins_end, size_t N, int C, int win_num) {
  typedef typename EC::group_type Fr;
  
  int *pos, *count;
  cudaMalloc(&pos, (1 << C) * win_num * sizeof(int));
  cudaMemset(pos, 0, (1 << C) * win_num * sizeof(int));
  cudaMalloc(&count, (1 << C) * win_num * sizeof(int)); 
  cudaMemset(count, 0, (1 << C) * win_num * sizeof(int));

  int process_parallel_level = 80;
  PreprocessScalars1<Fr, ScalarBit>
      <<<process_parallel_level, 512, 0, strm>>>(scalars, count, N, process_parallel_level, C, win_num);
  cudaStreamSynchronize(strm);
  
  PreprocessScalars2<ScalarBit>
      <<<win_num, 1, 0, strm>>>(wins_start, wins_end, count, pos, N, C, win_num);
  cudaStreamSynchronize(strm);

  PreprocessScalars3<Fr, ScalarBit>
      <<<process_parallel_level, 512, 0, strm>>>(scalars, wins, pos, N, process_parallel_level, C, win_num);
  cudaStreamSynchronize(strm);

  cudaFree(pos);
  cudaFree(count);
}

template <typename EC, int win_num = -1, bool except_one = false>
__global__ __launch_bounds__(1024 / EC::field_type::DEGREE, 1) void ProcessAllWinsBuckets(var *out, const var *bases_, int *wins, int *wins_start,
                                         int *wins_end, size_t N, size_t level, int C) {
  typedef typename EC::field_type FF;
  int T = threadIdx.x, B = blockIdx.x, D = blockDim.x;
  int tiles_per_block = D / FF::BIG_WIDTH; // tiles_per_block = 32
  int tileIdx = T / FF::BIG_WIDTH;
  int W = B / level; //level blocks handle one window
  int I_offset = (B % level) * ((1 << C) / level);

  int JAC_POINT_LIMBS = 3 * FF::DEGREE * FF::ELT_LIMBS;
  int AFF_POINT_LIMBS = 2 * FF::DEGREE * FF::ELT_LIMBS;
  wins = wins + W * N;

  __shared__ EC tile_bucket[128 / FF::DEGREE][FF::BIG_WIDTH];

  EC x;
  for (int I = I_offset; I < I_offset + ((1 << C) / level); I++) {

    if(except_one){
      if((I == 1) && (W == win_num - 1)){
        continue;
      }
    }

    int out_off = (W * (1<<C) + I) * JAC_POINT_LIMBS;
    auto g = fixnum::layout<FF::BIG_WIDTH>();
    EC::set_zero(tile_bucket[tileIdx][g.thread_rank()]);
    EC::set_zero(x);
    if(I) {
      for(int i = wins_start[W * (1 << C) + I]; i < wins_end[W * (1 << C) + I]; i += tiles_per_block){
        if (i + tileIdx < wins_end[W * (1 << C) + I]) {
          EC::load_affine(x, bases_ + wins[i + tileIdx] * AFF_POINT_LIMBS);
          EC::mixed_add(tile_bucket[tileIdx][g.thread_rank()], tile_bucket[tileIdx][g.thread_rank()], x);
        }
        __syncthreads();
      }
      __syncthreads();

      for (int m = tiles_per_block / 2; m > 0; m /= 2) {
        for (int j = 0; j < m; j += tiles_per_block) {
          if (j + tileIdx < m) {
            EC::add(tile_bucket[j + tileIdx][g.thread_rank()],
                    tile_bucket[j + tileIdx][g.thread_rank()],
                    tile_bucket[j + tileIdx + m][g.thread_rank()]); 
          }
        }
        __syncthreads();
      }
      __syncthreads();
    }
    if (tileIdx == 0) {
      EC::store_jac(out + out_off, tile_bucket[0][g.thread_rank()]);
    }
    __syncthreads();
  }
}

template <typename EC>
__global__ void MergeAllWinsToOne(var *out, int level, int C, int win_num) {
  typedef typename EC::field_type FF;
  int T = threadIdx.x, B = blockIdx.x; // Block0 processes Bucket1; Block(2^C-2) processes Bucket(2^C-1)
  int tileIdx = T / FF::BIG_WIDTH;
  int bucket_number = B * level + tileIdx;

  int JAC_POINT_LIMBS = 3 * FF::DEGREE * FF::ELT_LIMBS;

  auto g = fixnum::layout();

  __shared__ EC Q[128 / FF::DEGREE][FF::BIG_WIDTH];

  EC::set_zero(Q[tileIdx][g.thread_rank()]);

  EC x;
  // TODO:need to speed up
  for(int i = 0; i < win_num; i++){
    // multiply Q[i] by 2^C
    EC::mul_2exp(C, Q[tileIdx][g.thread_rank()], Q[tileIdx][g.thread_rank()]);
    EC::load_jac(x, out + (((1 << C)) * i + bucket_number) * JAC_POINT_LIMBS);
    EC::add(Q[tileIdx][g.thread_rank()], Q[tileIdx][g.thread_rank()], x);
    __syncthreads();
  }

  int out_off = bucket_number * JAC_POINT_LIMBS;  
  EC::store_jac(out + out_off, Q[tileIdx][g.thread_rank()]);
}


template <typename EC>
__global__ void PippengerReduce1(const var *out1, var *out2, int C, int C2) {
  typedef typename EC::group_type Fr;
  typedef typename EC::field_type FF;
  const int T = threadIdx.x, B = blockIdx.x, D = blockDim.x;
  const int tiles_per_block = D / FF::BIG_WIDTH;
  const int tileIdx = T / FF::BIG_WIDTH;

  const int JAC_POINT_LIMBS = 3 * FF::DEGREE * FF::ELT_LIMBS;

  __shared__ EC tile_bucket[4][FF::BIG_WIDTH];
  __shared__ EC x[4][FF::BIG_WIDTH];

  Fr A;
  auto g = fixnum::layout();
  EC::set_zero(tile_bucket[tileIdx][g.thread_rank()]);

  for(int j = 0; j < 1 << C2; j += tiles_per_block){
    if(j + tileIdx < 1 << C2){
      EC::load_jac(x[tileIdx][g.thread_rank()], out1 + (B * (1 << C2) + j + tileIdx) * JAC_POINT_LIMBS);
      EC::add(tile_bucket[tileIdx][g.thread_rank()], tile_bucket[tileIdx][g.thread_rank()], x[tileIdx][g.thread_rank()]);
    }
    __syncthreads();
  }
  __syncthreads();

  for (int m = tiles_per_block / 2; m > 0; m /= 2) {
    for (int j = 0; j < m; j += tiles_per_block) {
      if (j + tileIdx < m) {
        EC::add(tile_bucket[j + tileIdx][g.thread_rank()],
          tile_bucket[j + tileIdx][g.thread_rank()],
          tile_bucket[j + tileIdx + m][g.thread_rank()]); 
      }
    }
    __syncthreads();
  }
  __syncthreads();
  
  if(tileIdx == 0){
    if (fixnum::layout().thread_rank() == 0) {
      A.a = (1 << C2) * B;
    } else {
      A.a = 0;
    }
    EC::mul(tile_bucket[0][g.thread_rank()], A.a, tile_bucket[0][g.thread_rank()]);
    int out_off = B * JAC_POINT_LIMBS;
    EC::store_jac(out2 + out_off, tile_bucket[0][g.thread_rank()]);
  }
}

template <typename EC>
__global__ void PippengerReduce2(const var *out1, var *out2, int C, int C2) {
  typedef typename EC::group_type Fr;
  typedef typename EC::field_type FF;
  const int T = threadIdx.x, B = blockIdx.x, D = blockDim.x;
  const int tiles_per_block = D / FF::BIG_WIDTH;
  const int tileIdx = T / FF::BIG_WIDTH;

  const int JAC_POINT_LIMBS = 3 * FF::DEGREE * FF::ELT_LIMBS;

  __shared__ EC tile_bucket[4][FF::BIG_WIDTH];
  __shared__ EC x[4][FF::BIG_WIDTH];

  Fr A;
  auto g = fixnum::layout();
  EC::set_zero(tile_bucket[tileIdx][g.thread_rank()]);

  for(int j = 0; j < 1 << (C - C2); j += tiles_per_block){
    if(j + tileIdx < 1 << (C - C2)){
      EC::load_jac(x[tileIdx][g.thread_rank()], out1 + ((j + tileIdx) * (1 << C2) + B) * JAC_POINT_LIMBS);
      EC::add(tile_bucket[tileIdx][g.thread_rank()], tile_bucket[tileIdx][g.thread_rank()], x[tileIdx][g.thread_rank()]);
    }
    __syncthreads();
  }
  __syncthreads();

  for (int m = tiles_per_block / 2; m > 0; m /= 2) {
    for (int j = 0; j < m; j += tiles_per_block) {
      if (j + tileIdx < m) {
        EC::add(tile_bucket[j + tileIdx][g.thread_rank()],
          tile_bucket[j + tileIdx][g.thread_rank()],
          tile_bucket[j + tileIdx + m][g.thread_rank()]); 
      }
    }
    __syncthreads();
  }
  __syncthreads();

  if(tileIdx == 0){
    if (fixnum::layout().thread_rank() == 0) {
      A.a = B;
    } else {
      A.a = 0;
    }
    EC::mul(tile_bucket[0][g.thread_rank()], A.a, tile_bucket[0][g.thread_rank()]);
    int out_off = ((1 << (C - C2)) + B) * JAC_POINT_LIMBS;
    EC::store_jac(out2 + out_off, tile_bucket[0][g.thread_rank()]);
  }
}


template <typename EC>
__global__ __launch_bounds__(512, 2) void ec_sum_all(var *X, const var *Y, size_t n) {
  typedef typename EC::field_type FF;
  int T = threadIdx.x, B = blockIdx.x, D = blockDim.x;
  int elts_per_block = D / FF::BIG_WIDTH;
  int tileIdx = T / FF::BIG_WIDTH;

  int idx = elts_per_block * B + tileIdx;

  if (idx < n) {
    EC z, x, y;
    int off = idx * EC::NELTS * FF::ELT_LIMBS;

    EC::load_jac(x, X + off);
    EC::load_jac(y, Y + off);

    EC::add(z, x, y);

    EC::store_jac(X + off, z);
  }
}


template <typename EC>
void ReducetheLastWinOpt(cudaStream_t &strm, var *out1, var *out2, int C, int C2) {
  typedef typename EC::field_type FF;

  size_t pt_limbs = EC::NELTS * FF::ELT_LIMBS;

  // 简易版pippenger
  PippengerReduce1<EC><<<1 << (C - C2), 32, 0, strm>>>(out1, out2, C, C2);
  PippengerReduce2<EC><<<1 << C2, 32, 0, strm>>>(out1, out2, C, C2);
  cudaStreamSynchronize(strm);

  int threads_per_block = 512;
  if(EC::field_type::DEGREE == 2){
    threads_per_block = 256;
  }

  size_t n = (1 << (C - C2)) + (1 << C2);
  size_t r = n & 1, m = n / 2;
  for (; m != 0; r = m & 1, m >>= 1) {
    size_t nblocks =
        (m * FF::BIG_WIDTH + threads_per_block - 1) / threads_per_block;

    ec_sum_all<EC>
        <<<nblocks, threads_per_block, 0, strm>>>(out2, out2 + m * pt_limbs, m);
    if (r)
      ec_sum_all<EC>
          <<<1, threads_per_block, 0, strm>>>(out2, out2 + 2 * m * pt_limbs, 1);
  }
  cudaStreamSynchronize(strm);
}

template <typename Fp>
__global__ void from_monty(var *scalar, size_t n) {
  int T = threadIdx.x, B = blockIdx.x, D = blockDim.x;
  int elts_per_block = D / Fp::BIG_WIDTH;
  int tileIdx = T / Fp::BIG_WIDTH;

  int idx = elts_per_block * B + tileIdx;

  if (idx < n) {
    // typedef Fr_type Fr;
    Fp w;
    int w_off = idx * Fp::ELT_LIMBS;

    Fp::load(w, scalar + w_off);

    Fp::from_monty(w, w);

    Fp::store_a(scalar + w_off, w.a);
  }
}

template <typename Fp>
void FromMonty(cudaStream_t &strm, var *w, size_t n) {
  const int threads_per_block = 1024;
  size_t nblocks = (n * Fp::BIG_WIDTH + threads_per_block - 1) / threads_per_block;
  from_monty<Fp><<<nblocks, threads_per_block, 0, strm>>>(w, n);
}

static inline double as_mebibytes(size_t n) {
    return n / (long double)(1UL << 20);
}

void print_meminfo(size_t allocated) {
    size_t free_mem, dev_mem;
    cudaMemGetInfo(&free_mem, &dev_mem);
    fprintf(stderr, "Allocated %zu bytes; device has %.1f MiB free (%.1f%%).\n",
            allocated, as_mebibytes(free_mem), 100.0 * free_mem / dev_mem);
}

var *allocate_memory(size_t nbytes, int dbg = 0) {
  var *mem = nullptr;
#ifdef USE_UNIFIED_MEMORY
  cudaMallocManaged(&mem, nbytes);
#else
  cudaMalloc(&mem, nbytes);
#endif
  if (mem == nullptr) {
    fprintf(stderr, "Failed to allocate enough device memory %d\n", nbytes);
    size_t free_mem, dev_mem;
    cudaMemGetInfo(&free_mem, &dev_mem);
    fprintf(stderr, "Allocated %zu bytes; device has %.1f MiB free (%.1f%%).\n",
          nbytes, as_mebibytes(free_mem), 100.0 * free_mem / dev_mem);
    abort();
  }
  if (dbg)
    print_meminfo(nbytes);
  return mem;
}

template <typename EC> 
var* load_points_affine(size_t n, const var *inputs) {
    typedef typename EC::field_type FF;

    static constexpr size_t coord_bytes = FF::DEGREE * (FF::ELT_LIMBS * sizeof(var));
    static constexpr size_t aff_pt_bytes = 2 * coord_bytes;

    size_t total_aff_bytes = n * aff_pt_bytes;

    auto mem = allocate_memory(total_aff_bytes);
#ifdef USE_UNIFIED_MEMORY
    memcpy(mem, inputs, total_aff_bytes);
#else
    auto err = cudaMemcpy(mem, inputs, total_aff_bytes, cudaMemcpyHostToDevice);
#endif
    return mem;
}

template<typename EC, int ScalarBit>
var *msm_pippenger(cudaStream_t &strm, const var *bases, const var *scalars, size_t N, int C, int C2, int win_num) {
  size_t nblocks_add = win_num * (1 << C); 
  size_t nblocks_reduce = (1 << C);       

  int nthreads_add = 1024; // for BLS12
  if(EC::field_type::DEGREE == 2){
    nthreads_add = 512;
  }

  // auto t = now();
  int *wins, *wins_start, *wins_end;
  cudaMalloc(&wins, N * win_num * sizeof(int)); 
  cudaMalloc(&wins_start, (1 << C) * win_num * sizeof(int)); 
  cudaMalloc(&wins_end, (1 << C) * win_num * sizeof(int)); 
  auto out = allocate_memory(nblocks_add * EC::NELTS * (EC::field_type::ELT_LIMBS * sizeof(var)));
  auto out2 = allocate_memory(((1 << (C - C2)) + (1 << C2)) * EC::NELTS * (EC::field_type::ELT_LIMBS * sizeof(var)));
  cudaStreamSynchronize(strm);
  // print_time(t, "Initialize memory space");
  // preprocess all scalars
  PreprocessScalars<EC, ScalarBit>(strm, scalars, wins, wins_start, wins_end, N, C, win_num);
  if (cudaPeekAtLastError() != cudaSuccess) {
    printf("Preprocess All Scalars Error: %s\n", cudaGetErrorString(cudaPeekAtLastError()));
  }
  cudaStreamSynchronize(strm);
  // print_time(t, "Preprocess All Scalars");
  // cudaFree(scalar);

  int process_parallel_level = 1 << (C);  // control parallel granularity, per windows needs "process_parallel_level" blocks to process  
  ProcessAllWinsBuckets<EC>
      <<<win_num * process_parallel_level, nthreads_add, 0, strm>>>(out, bases, wins, wins_start, wins_end, N, process_parallel_level, C);
  if (cudaPeekAtLastError() != cudaSuccess) {
    printf("Process Buckets Error: %s\n", cudaGetErrorString(cudaPeekAtLastError()));
  }
  cudaStreamSynchronize(strm);
  // print_time(t, "ProcessAllWinsBuckets");


  //per block process "reduce_parallel_level" buckets
  int reduce_parallel_level = 1 << 5; 
  MergeAllWinsToOne<EC>
      <<<nblocks_reduce / reduce_parallel_level, reduce_parallel_level * EC::field_type::BIG_WIDTH, 0, strm>>>(out, reduce_parallel_level, C, win_num);
  if (cudaPeekAtLastError() != cudaSuccess) {
    printf("Merge Error: %s\n", cudaGetErrorString(cudaPeekAtLastError()));
  }
  cudaStreamSynchronize(strm);
  // print_time(t, "MergeAllWinsToOne");
  
  // ReducetheLastWin<EC, C>(strm, out.get(), nblocks_reduce-1);
  ReducetheLastWinOpt<EC>(strm, out, out2, C, C2);
  if (cudaPeekAtLastError() != cudaSuccess) {
    printf("Reduce Error: %s\n", cudaGetErrorString(cudaPeekAtLastError()));
  }
  cudaStreamSynchronize(strm);
  // print_time(t, "ReducetheLastWin");
  return out2;
}

} // namespace msm
} // namespace gzkp