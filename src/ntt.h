#pragma once

#include "common.h"
#include "gsl/assert.h"
#include "ff/arith.cuh"

#ifdef MNT4753
#include "ntt_const/mnt4753_const.h"
#endif
#ifdef ALT_BN128
#include "ntt_const/alt_bn128_const.h"
#endif
#ifdef BLS12_381
#include "ntt_const/bls12_381_const.h"
#endif

namespace gzkp {
namespace ntt {
static void debug_dump_data(int n, uint64_t *d_data) {
  uint64_t *h_data = new uint64_t[LIMBS << n];
  cudaMemcpy(h_data, d_data, BYTES << n, cudaMemcpyDeviceToHost);
  for (int i = 0; i < 1 << n; ++i) {
    for (int j = 0; j < LIMBS; ++j) {
      printf("%lu%c", h_data[(j % 3) << (n + 2) | i << 2 | (j / 3)],
             " \n"[j + 1 == LIMBS]);
    }
  }
  delete[] h_data;
}

// ｜min  |median｜max｜
// ｜0...8|0...8 |0...8|
// 对于n_size>9的情况下, level从0到8的twiddle位置(level从0到8)
// sid的值从0到2^9-1
template <int n_size> struct MinIdTrait {
  __device__ __forceinline__ static int get_twiddle_id(int level, int bid, int sid) {
    static_assert(n_size >= MAX_STEPS_16, "");
    return sid & ((1 << level) - 1);
  }
};

//对于n_size>18的情况下, level从0到8的twiddle位置(level从9到17)
// sid的值从0到2^9-1 bid从0到2^(n-9)(当n>18的时候,会有重复的值)
template <int n_size> struct MedianIdTrait {
  __device__ __forceinline__ static int get_twiddle_id(int level, int bid, int sid) {
    // static_assert(n_size >= 18, "");
    // 先bid%2^9找到对应的block 然后每个block中用到1<<level的个twiddle
    return  ((bid & ((1 << MAX_STEPS_16) - 1))<< level) | (sid & ((1 << level) - 1));
  }
};

template <int n_size> struct MedianIdTrait_1 {
  __device__ __forceinline__ static int get_twiddle_id(int level, int bid, int sid) {
    // static_assert(n_size >= 18, "");
    // 先bid%2^9找到对应的block 然后每个block中用到1<<level的个twiddle
    return  ((bid & ((1 << (2*MAX_STEPS_16)) - 1))<< level) | (sid & ((1 << level) - 1));
  }
};

template <int n_size> struct MedianIdTrait_2 {
  __device__ __forceinline__ static int get_twiddle_id(int level, int bid, int sid) {
    // static_assert(n_size >= 18, "");
    // 先bid%2^9找到对应的block 然后每个block中用到1<<level的个twiddle
    return  ((bid & ((1 << (3*MAX_STEPS_16)) - 1))<< level) | (sid & ((1 << level) - 1));
  }
};

//对于n_size-9*(0,1)<9的情况下, level从0到n_size-9*(0,1)的twiddle位置(level从9到16,或者从18到25)
// sid的值从0到2^8-1 bid从0到2^(n-9)(当n>18的时候,会有重复的值)
template <int n_size> struct MaxIdTrait {
  __device__ __forceinline__ static int get_twiddle_id(int level, int bid, int sid) {
    // 一�`��大的block中放了多少小的block
    int mini_block_num = 1 << (MAX_STEPS_16 - (n_size % MAX_STEPS_16));
    // 每个小的block中sid的数目
    int mini_block_sid_num = 1 << ((n_size % MAX_STEPS_16) - 1);
    //每个min block中用到1<<level的个twiddle
    return (bid * (mini_block_num * (1 << level))) 
            //找到sid属于哪个小的block      //找到这个小的block的开始位置
          + ((sid / mini_block_sid_num) << level)
            //找到sid在这个小的block中对应的twiddle的位置
          + ((sid % mini_block_sid_num) & ((1<<level) - 1));
  }
};

template <int n> struct ColumnMajored {
  __host__ __device__ __forceinline__ static uint64_t load(const uint64_t *a,
                                                           int i, int j) {
    return a[j << n | i];
  }

  __host__ __device__ __forceinline__ static void store(uint64_t *a, int i,
                                                        int j, uint64_t v) {
    a[j << n | i] = v;
  }
};

template <int n1, int inversed> struct IndexInversion {};

template <int n1> struct IndexInversion<n1, 0> {
  __device__ __forceinline__ static int inverse(int i) { return 0; }
};

template <int n1> struct IndexInversion<n1, 1> {
  __device__ __forceinline__ static int inverse(int i) { return i; }
};

template <int n1> struct IndexInversion<n1, -1> {
  __device__ __forceinline__ static int inverse(int i) { return (1 << n1) - i; }
};


// 最大能支持24
__device__ __constant__ uint16_t BIT_REVERSE_TABLE[1 << 12];
__device__ __constant__ uint64_t NTT_NORMALIZER[LIMBS],
    VANISH_COEFFICIENT[LIMBS];

template <int n_size> static void initialize_constants() {
  {
    const int nsqrt = n_size >> 1;

    uint16_t* h_table;
    cudaHostAlloc((void**)(&h_table), sizeof(uint16_t)*(1<<nsqrt), cudaHostAllocDefault);
    for (int mask = 0; mask < 1 << nsqrt; ++mask) {
      int rmask = 0;
      for (int i = 0; i < nsqrt; ++i) {
        rmask = rmask << 1 | (mask >> i & 1);
      }
      h_table[mask] = rmask;
    }
    cudaMemcpyToSymbol(BIT_REVERSE_TABLE, h_table, sizeof(uint16_t) << nsqrt);
    cudaFreeHost(h_table);
  }

  // NTT_NORMALIZER : -2^{-n_size}
  static uint64_t buffer[LIMBS];
  memcpy(buffer, ntt::const_u64::R, BYTES);
  for (int i = 0; i < n_size; ++i) {
    static uint64_t new_buffer[LIMBS];
    ntt::host_multiply_2p(new_buffer, ntt::const_u64::TWO_INV, buffer);
    memcpy(buffer, new_buffer, BYTES);
  }
  {
    static uint64_t new_buffer[LIMBS];
    ntt::host_multiply_2p(new_buffer, ntt::const_u64::R_NEGATE, buffer);
    cudaMemcpyToSymbol(NTT_NORMALIZER, new_buffer, BYTES);
  }
  // VANISH_COEFFICIENT : 2^{-(2n + 1)}
  for (int i = 0; i < n_size + 1; ++i) {
    static uint64_t new_buffer[LIMBS];
    ntt::host_multiply_2p(new_buffer, ntt::const_u64::TWO_INV, buffer);
    memcpy(buffer, new_buffer, BYTES);
  }
  cudaMemcpyToSymbol(VANISH_COEFFICIENT, buffer, BYTES);
}

template <int n_size> class BasicBlockNTT {
public:
  static const int n = n_size;
  static const int n1 = n + 1;
  static const int nsqrt = n_size >> 1;

  // static_assert(ntt::NTTField::LIMBS == 12, "");

  static const int BATCH = ntt::NTTField::BATCH;
  static const int SLOT_WIDTH = ntt::NTTField::SLOT_WIDTH;

  __device__ __forceinline__ static void PreparePowersInit(uint64_t *powers) {
    const int tid = threadIdx.x;
    ColumnMajored<n1>::store(powers, 0, tid, ntt::const_u64::R[tid]);
  }

  template <int noffset, int nstep>
  __device__ __forceinline__ static void PreparePowers(uint64_t *powers) {
    using namespace ntt::const_u64;

    const int bid = blockIdx.x;
    const int tid = threadIdx.x;
    const int sid = fixnum::slot_id<BIG_WIDTH_R>();

    __shared__ uint64_t spowers[LIMBS][1 << nstep];
    if (tid == 0) {
#pragma unroll
      for (int j = 0; j < LIMBS; ++j) {
        spowers[j][0] =
            ColumnMajored<n1>::load(powers, bid << (n1 - noffset), j);
      }
    }
    __syncthreads();
#pragma unroll
    for (int level = 0; level < nstep; ++level) {
      if (sid < (1 << level)) {
        const int id0 = sid << (nstep - level);
        const int id1 = id0 | 1 << (nstep - level - 1);
        Fr_MNT4 b_r;
        Fr_MNT4::load_stride<ELT_LIMBS_R, BIG_WIDTH_R>(b_r, &spowers[0][id0], 1 << nstep);
        Fr_MNT4 out_r;
        Fr_MNT4 omega;
        Fr_MNT4::load<ELT_LIMBS_R, BIG_WIDTH_R>(omega, M_OMEGAS[ORDER - noffset - 2 - level]);
        Fr_MNT4::mul<ELT_LIMBS_R, BIG_WIDTH_R>(out_r, omega, b_r);
        Fr_MNT4::store_stride<ELT_LIMBS_R, BIG_WIDTH_R>(&spowers[0][id1], out_r, 1 << nstep);
      }
      __syncthreads();
    }
    if (tid < (1 << nstep)) {
#pragma unroll
      for (int j = 0; j < LIMBS; ++j) {
        const uint64_t v = spowers[j][tid];
        ColumnMajored<n1>::store(
            powers, (bid << nstep | tid) << (n1 - noffset - nstep), j, v);
      }
    }
  }

  template <int noffset, int nstep, int uroot, int coset>
  __device__ __forceinline__ static void PrepareTwiddles(const uint64_t *__restrict__ powers,
    uint64_t *__restrict__ twiddles) {
      const int bid = blockIdx.x;
      const int tid = threadIdx.x;

      const int ori_loc = ((bid << nstep) | tid);
      for (int level = 0; level < n; level ++) {
        const int uroot_index = (ori_loc << (n - level));
        int loc = ori_loc;
        int batch = level / nstep;
        if (ori_loc < (1 << level)) {
          if (batch == 0) {
            loc = ori_loc;
          }
          else if (batch < 1) {
            loc = bid + (tid << (level - nstep));
          }
          else {
            if (bid < (1<<((batch - 1) * nstep))) {
              loc = (bid << (level - (batch-1)*nstep)) + (tid << (level - batch*nstep));
            }
            else {
              int mini_block = bid & ((1<<((batch - 1) * nstep)) - 1);
              loc = (mini_block << (level - (batch - 1)*nstep)) + (tid << (level - batch*nstep)) + (bid >> ((batch-1)*nstep));
            }
          }
          for (int j = 0; j < LIMBS; ++j) {
            const uint64_t v = ColumnMajored<n1>::load(
              powers,
              twiddle_index<uroot, coset>(uroot_index, 1 << (n - 1 - level)),
              j);
            // twiddles[(1 << level) * LIMBS + loc + (j << level)] = v; 
            twiddles[(1 << level) * LIMBS + loc*LIMBS + j] = v;
          }
        }
      }
      // 对于level <= 6, 每层只有 1<<level 个twiddle值, 每个block只会用到一个twiddle值
      // const int ori_loc = ((bid << nstep) | tid);
      // // 简化前6轮的判断分支,前6轮中只会用到bid = 0
      // if (bid == 0) {
      //   for (int level = 0; level < nstep; level ++) {
      //     if (ori_loc < (1 << level)) {
      //       //间隔的值
      //       const int uroot_index = (ori_loc << (n - level));
      //       // 对于level <= 6, 每层只有 1<<level 个twiddle值, 每个block只会用到一个twiddle值
      //       int loc = ori_loc;
      //       for (int j = 0; j < LIMBS; ++j) {
      //         const uint64_t v = ColumnMajored<n1>::load(powers,
      //                                                   twiddle_index<uroot, coset>(uroot_index, 1 << (n - 1 - level)),
      //                                                   j);
      //         // twiddles[(1 << level) * LIMBS + loc + (j << level)] = v; 
      //         twiddles[(1 << level) * LIMBS + loc*LIMBS + j] = v; 
      //       }
      //     }
      //   } 
      // }

      // for (int level = nstep; level < n; level ++) {
      //   if (ori_loc < (1 << level)) {
      //     //间隔的值
      //     const int uroot_index = (ori_loc << (n - level));
      //     int loc = ori_loc;
      //     // 对于 9 < level < 18
      //     if (level > nstep && level < 2*nstep) {
      //       loc = bid + (tid << (level - nstep));
      //     }
      //     // 对于 18 <= level < 27
      //     else if (level >= 2*nstep && level < 3*nstep) {
      //       if (bid < (1<<nstep)) {
      //         loc = (bid << (level - nstep)) + (tid << (level - 2*nstep));
      //       }
      //       else {
      //         int mini_block = bid & ((1<<nstep) - 1);
      //         loc = (mini_block << (level - nstep)) + (tid << (level - 2*nstep)) + (bid >> nstep);
      //       }
      //     }
      //     else if (level >= 3*nstep && level < 4*nstep) {
      //       if (bid < (1<<(2*nstep))) {
      //         loc = (bid << (level - 2*nstep)) + (tid << (level - 3*nstep));
      //       }
      //       else {
      //         int mini_block = bid & ((1<<(2*nstep)) - 1);
      //         loc = (mini_block << (level - 2*nstep)) + (tid << (level - 3*nstep)) + (bid >> (2*nstep));
      //       }
      //     }
      //     else if (level >= 4*nstep && level < 5*nstep) {
      //       if (bid < (1<<(3*nstep))) {
      //         loc = (bid << (level - 3*nstep)) + (tid << (level - 4*nstep));
      //       }
      //       else {
      //         int mini_block = bid & ((1<<(3*nstep)) - 1);
      //         loc = (mini_block << (level - 3*nstep)) + (tid << (level - 4*nstep)) + (bid >> (3*nstep));
      //       }
      //     }
      //     for (int j = 0; j < LIMBS; ++j) {
      //       const uint64_t v = ColumnMajored<n1>::load(
      //           powers,
      //           twiddle_index<uroot, coset>(uroot_index, 1 << (n - 1 - level)),
      //           j);
      //       // twiddles[(1 << level) * LIMBS + loc + (j << level)] = v; 
      //       twiddles[(1 << level) * LIMBS + loc*LIMBS + j] = v;
      //     }
      //   }
      // }
  }

// #define G(batch, slot, lane)                                                  \
//   gdata[(batch) << (n + 2) | (slot) << 2 | lane]

  template <template <int> class IdTrait>
  __device__ __forceinline__ static void
  DecimationInTime(const uint64_t *__restrict__ gtwiddles,
                   uint64_t *__restrict__ gdata,
                   const int level_offset, 
                   int start, int end) {
    // start < end, start=0,1,...,n
    // read operators
    const int bid = blockIdx.x;
    const int tid = threadIdx.x;
    const int tileIdx = tid / BIG_WIDTH_R;
    auto g = fixnum::layout<BIG_WIDTH_R>();
    int steps = end - start;

    // share memory is 48KB
    // __shared__ uint64_t data[3][4 << MAX_SHAREMEM];
    __shared__ uint64_t data[BIG_WIDTH_R << MAX_SHAREMEM_16];
    // handle butterfly step is MAX_STEPS
    if (steps == MAX_STEPS_16) {
      uint64_t offset1 = 1 << start; // 第end-1轮蝴蝶操作时，操作数之间的offset
      uint64_t offset2 = offset1 << 1; // 相邻线程间读取的操作数之间的偏移
      uint64_t offset_block = (((bid / offset1) * offset1) << MAX_STEPS_16) + (bid % offset1);     
      uint64_t i0 = offset_block + tileIdx*offset2; 
      uint64_t i1 = i0 + offset1;
      data[tileIdx*2*BIG_WIDTH_R+g.thread_rank()]=0;
      data[tileIdx*2*BIG_WIDTH_R+BIG_WIDTH_R+g.thread_rank()]=0;
      if (g.thread_rank() < LIMBS) {
        data[tileIdx*2*BIG_WIDTH_R+g.thread_rank()] = gdata[i0*LIMBS+g.thread_rank()];
        data[tileIdx*2*BIG_WIDTH_R+BIG_WIDTH_R+g.thread_rank()] = gdata[i1*LIMBS+g.thread_rank()];
      }
    }
    // handle butterfly step less than MAX_STEPS
    else {
      uint64_t offset1 = 1 << start;
      uint64_t offset2 = offset1 << 1;
      uint64_t offset_block = bid << (MAX_STEPS_16-steps);
      uint64_t i0 = offset_block + (tileIdx >> (steps-1)) + (tileIdx & ((1 << (steps-1))-1)) * offset2;
      uint64_t i1 = i0 + offset1;
      data[tileIdx*2*BIG_WIDTH_R+g.thread_rank()]=0;
      data[tileIdx*2*BIG_WIDTH_R+BIG_WIDTH_R+g.thread_rank()]=0;
      if (g.thread_rank() < LIMBS) {
        data[tileIdx*2*BIG_WIDTH_R+g.thread_rank()] = gdata[i0*LIMBS+g.thread_rank()];
        data[tileIdx*2*BIG_WIDTH_R+BIG_WIDTH_R+g.thread_rank()] = gdata[i1*LIMBS+g.thread_rank()];
      }
    }
    
    __syncthreads();
  
#define BUTTERFLY(level)                                                       \
  do {                                                                         \
    DecimationInTimeButterfly<IdTrait>(                   \
        level_offset, level, steps, gtwiddles, reinterpret_cast<uint64_t *>(data));                        \
  } while (false)
    BUTTERFLY(0);
    BUTTERFLY(1);
    BUTTERFLY(2);
    BUTTERFLY(3);
    BUTTERFLY(4);
    BUTTERFLY(5);
    BUTTERFLY(6);
    BUTTERFLY(7);
#undef BUTTERFLY

    if (steps == MAX_STEPS_16) {
      uint64_t offset1 = 1 << start; // 第end-1轮蝴蝶操作时，操作数之间的offset
      uint64_t offset2 = offset1 << 1; // 相邻线程间读取的操作数之间的偏移
      uint64_t offset_block = (((bid / offset1) * offset1) << MAX_STEPS_16) + (bid % offset1);     
      uint64_t i0 = offset_block + tileIdx * offset2;
      uint64_t i1 = i0 + offset1;
      if (g.thread_rank() < LIMBS) {
        gdata[i0*LIMBS+g.thread_rank()] = data[tileIdx*2*BIG_WIDTH_R+g.thread_rank()];
        gdata[i1*LIMBS+g.thread_rank()] = data[tileIdx*2*BIG_WIDTH_R+BIG_WIDTH_R+g.thread_rank()];
      }
    }
    else {
      uint64_t offset1 = 1 << start;
      uint64_t offset2 = offset1 << 1;
      uint64_t offset_block = bid << (MAX_STEPS_16-steps);
      uint64_t i0 = offset_block + (tileIdx / (1 << (steps - 1))) + (tileIdx % (1 << (steps - 1))) * offset2;
      uint64_t i1 = i0 + offset1;
      if (g.thread_rank() < LIMBS) {
        gdata[i0*LIMBS+g.thread_rank()] = data[tileIdx*2*BIG_WIDTH_R+g.thread_rank()];
        gdata[i1*LIMBS+g.thread_rank()] = data[tileIdx*2*BIG_WIDTH_R+BIG_WIDTH_R+g.thread_rank()];
      }
    }
  }

  template <template <int> class IdTrait, bool DEBUG=false>
  __device__ __forceinline__ static void
  DecimationInFreq(const uint64_t *__restrict__ gtwiddles, 
                   uint64_t *__restrict__ gdata,
                   const int level_offset, 
                   int start, int end) {
    // end > start, start:0,1,2,3...
    // read operators

    const int bid = blockIdx.x;
    const int tid = threadIdx.x;
    const int tileIdx = tid / BIG_WIDTH_R;
    auto g = fixnum::layout<BIG_WIDTH_R>();
    int steps = end - start;

    __shared__ uint64_t data[BIG_WIDTH_R << MAX_SHAREMEM_16];
    if (steps == MAX_STEPS_16) {
      uint64_t offset1 = 1 << (n - end); // 第end-1轮蝴蝶操作时，操作数之间的offset
      uint64_t offset2 = offset1 << 1; // 相邻线程间读取的操作数之间的偏移
      uint64_t offset_block = (((bid / offset1) * offset1) << MAX_STEPS_16) + (bid % offset1);     
      uint64_t i0 = offset_block + tileIdx*offset2; 
      uint64_t i1 = i0 + offset1;
      data[tileIdx*2*BIG_WIDTH_R+g.thread_rank()]=0;
      data[tileIdx*2*BIG_WIDTH_R+BIG_WIDTH_R+g.thread_rank()]=0;
      if (g.thread_rank() < LIMBS) {
        data[tileIdx*2*BIG_WIDTH_R+g.thread_rank()] = gdata[i0*LIMBS+g.thread_rank()];
        data[tileIdx*2*BIG_WIDTH_R+BIG_WIDTH_R+g.thread_rank()] = gdata[i1*LIMBS+g.thread_rank()];
      }
    }
    else {
      uint64_t offset1 = 1 << (n - end);
      uint64_t offset2 = offset1 << 1;
      uint64_t offset_block = bid << (MAX_STEPS_16-steps);
      uint64_t i0 = offset_block + (tileIdx >> (steps-1)) + (tileIdx & ((1 << (steps-1))-1)) * offset2;
      uint64_t i1 = i0 + offset1;
      data[tileIdx*2*BIG_WIDTH_R+g.thread_rank()]=0;
      data[tileIdx*2*BIG_WIDTH_R+BIG_WIDTH_R+g.thread_rank()]=0;
      if (g.thread_rank() < LIMBS) {
        data[tileIdx*2*BIG_WIDTH_R+g.thread_rank()] = gdata[i0*LIMBS+g.thread_rank()];
        data[tileIdx*2*BIG_WIDTH_R+BIG_WIDTH_R+g.thread_rank()] = gdata[i1*LIMBS+g.thread_rank()];
      }
    }
    __syncthreads();

#define BUTTERFLY(level)                                                        \
  do {                                                                          \
    DecimationInFreqButterfly<IdTrait, DEBUG>(                                  \
        level_offset, level, steps, gtwiddles, reinterpret_cast<uint64_t *>(data));    \
  } while (false)
    BUTTERFLY(7);
    BUTTERFLY(6);
    BUTTERFLY(5);
    BUTTERFLY(4);
    BUTTERFLY(3);
    BUTTERFLY(2);
    BUTTERFLY(1);
    BUTTERFLY(0);
#undef BUTTERFLY

    if (steps == MAX_STEPS_16) {
      uint64_t offset1 = 1 << (n - end); // 第end-1轮蝴蝶操作时，操作数之间的offset
      uint64_t offset2 = offset1 << 1; // 相邻线程间读取的操作数之间的偏移
      uint64_t offset_block = (((bid / offset1) * offset1) << MAX_STEPS_16) + (bid % offset1);     
      uint64_t i0 = offset_block + tileIdx*offset2; 
      uint64_t i1 = i0 + offset1;
      if (g.thread_rank() < LIMBS) {
        gdata[i0*LIMBS+g.thread_rank()] = data[tileIdx*2*BIG_WIDTH_R+g.thread_rank()] ;
        gdata[i1*LIMBS+g.thread_rank()] = data[tileIdx*2*BIG_WIDTH_R+BIG_WIDTH_R+g.thread_rank()];
      }
    }
    else {
      int steps = end - start;
      uint64_t offset1 = 1 << (n - end);
      uint64_t offset2 = offset1 << 1;
      uint64_t offset_block = bid << (MAX_STEPS_16-steps);
      uint64_t i0 = offset_block + (tileIdx >> (steps-1)) + (tileIdx & ((1 << (steps-1))-1)) * offset2;
      uint64_t i1 = i0 + offset1;
      if (g.thread_rank() < LIMBS) {
        gdata[i0*LIMBS+g.thread_rank()] = data[tileIdx*2*BIG_WIDTH_R+g.thread_rank()];
        gdata[i1*LIMBS+g.thread_rank()] = data[tileIdx*2*BIG_WIDTH_R+BIG_WIDTH_R+g.thread_rank()];
      }
    }
  }
// #undef G

  template <template <int> class IdTrait>
  __device__ __forceinline__ static void
  DecimationInTimeButterfly(const int level_offset, 
                            const int level,
                            int butterfly_total_level,
                            const uint64_t *__restrict__ gtwiddles,
                            uint64_t *__restrict__ data) {
    if (level >= butterfly_total_level)
      return ;

    // const int bid = blockIdx.x;
    const int tid = threadIdx.x;
    const int tileIdx = tid / BIG_WIDTH_R;

    uint32_t i0 = tileIdx << 1;
    const int offset = tileIdx & ((1 << level) - 1);
    asm("bfi.b32 %0, %1, %2, 0, %3;"
        : "=r"(i0)
        : "r"(offset), "r"(i0), "r"(level + 1));
    const uint32_t i1 = i0 | 1 << level;

    const uint64_t *twiddle =
        gtwiddles + (LIMBS << (level_offset + level)) +
        IdTrait<n_size>::get_twiddle_id(level, blockIdx.x, tileIdx)*LIMBS;

    Fr_MNT4 t_r, b_r, twiddle_r;
    Fr_MNT4::load<ELT_LIMBS_R, BIG_WIDTH_R>(b_r, data + i1*BIG_WIDTH_R);
    Fr_MNT4::load<ELT_LIMBS_R, BIG_WIDTH_R>(twiddle_r, twiddle);
    Fr_MNT4::mul<ELT_LIMBS_R, BIG_WIDTH_R>(t_r, twiddle_r, b_r);
    Fr_MNT4 a_r;
    Fr_MNT4::load<ELT_LIMBS_R, BIG_WIDTH_R>(a_r, data + i0*BIG_WIDTH_R);

    Fr_MNT4::sub<ELT_LIMBS_R, BIG_WIDTH_R>(b_r, a_r, t_r);
    Fr_MNT4::store<ELT_LIMBS_R, BIG_WIDTH_R>(data + i1*BIG_WIDTH_R, b_r);
    Fr_MNT4::add<ELT_LIMBS_R, BIG_WIDTH_R>(a_r, a_r, t_r);
    Fr_MNT4::store<ELT_LIMBS_R, BIG_WIDTH_R>(data + i0*BIG_WIDTH_R, a_r);
    __syncthreads();
  }

  template <template <int> class IdTrait, bool DEBUG=false>
  __device__ __forceinline__ static void
  DecimationInFreqButterfly(const int level_offset, 
                            const int level,
                            int butterfly_total_level,
                            const uint64_t *__restrict__ gtwiddles,
                            uint64_t *__restrict__ data) {
    if (level >= butterfly_total_level)
      return ;

    // const int bid = blockIdx.x;
    const int tid = threadIdx.x;
    const int tileIdx = tid / BIG_WIDTH_R;
    
    uint32_t i0 = tileIdx << 1;
    const int offset = tileIdx & ((1 << level) - 1);
    asm("bfi.b32 %0, %1, %2, 0, %3;"
        : "=r"(i0)
        : "r"(offset), "r"(i0), "r"(level + 1));
    const uint32_t i1 = i0 | 1 << level;

    const uint64_t *twiddle =
        gtwiddles + (LIMBS << (level_offset + level)) +
        IdTrait<n_size>::get_twiddle_id(level, blockIdx.x, tileIdx)*LIMBS;

    Fr_MNT4 t_r, b_r, a_r, twiddle_r;
    Fr_MNT4::load<ELT_LIMBS_R, BIG_WIDTH_R>(b_r, data + i1*BIG_WIDTH_R);
    Fr_MNT4::load<ELT_LIMBS_R, BIG_WIDTH_R>(a_r, data + i0*BIG_WIDTH_R);
    Fr_MNT4::add<ELT_LIMBS_R, BIG_WIDTH_R>(t_r, b_r, a_r);
    Fr_MNT4::store<ELT_LIMBS_R, BIG_WIDTH_R>(data + i0*BIG_WIDTH_R, t_r);
    Fr_MNT4::sub<ELT_LIMBS_R, BIG_WIDTH_R>(a_r, a_r, b_r);
    Fr_MNT4::load<ELT_LIMBS_R, BIG_WIDTH_R>(twiddle_r, twiddle);
    Fr_MNT4::mul<ELT_LIMBS_R, BIG_WIDTH_R>(t_r, twiddle_r, a_r);
    Fr_MNT4::store<ELT_LIMBS_R, BIG_WIDTH_R>(data + i1*BIG_WIDTH_R, t_r);

    __syncthreads();
  }

  __device__ __forceinline__ static void
  DotProduct(uint64_t *__restrict__ out, const uint64_t *__restrict__ a,
             const uint64_t *__restrict__ b, const uint64_t *__restrict__ c) {
    const int bid = blockIdx.x;
    const int tid = threadIdx.x;
    // const int D = blockDim.x;
    const int tileIdx = tid / BIG_WIDTH_R;
    int loc = (bid << MAX_STEPS_16) + tileIdx;
    Fr_MNT4 t_r, b_r, c_r;
    Fr_MNT4 m_r;
    Fr_MNT4::load<ELT_LIMBS_R, BIG_WIDTH_R>(b_r, b + loc*LIMBS);
    Fr_MNT4::load<ELT_LIMBS_R, BIG_WIDTH_R>(m_r, a + loc*LIMBS);
    Fr_MNT4::mul<ELT_LIMBS_R, BIG_WIDTH_R>(t_r, m_r, b_r);
    Fr_MNT4::load<ELT_LIMBS_R, BIG_WIDTH_R>(m_r, NTT_NORMALIZER);
    Fr_MNT4::mul<ELT_LIMBS_R, BIG_WIDTH_R>(b_r, m_r, t_r);
    Fr_MNT4::load<ELT_LIMBS_R, BIG_WIDTH_R>(c_r, c + loc*LIMBS);
    Fr_MNT4::add<ELT_LIMBS_R, BIG_WIDTH_R>(t_r, b_r, c_r);
    Fr_MNT4::load<ELT_LIMBS_R, BIG_WIDTH_R>(m_r, VANISH_COEFFICIENT);
    Fr_MNT4::mul<ELT_LIMBS_R, BIG_WIDTH_R>(b_r, m_r, t_r);
    Fr_MNT4::store<ELT_LIMBS_R, BIG_WIDTH_R>(out + loc*LIMBS, b_r);
  }

  // mode P
  // TODO: recode this function
  __device__ __forceinline__ static void
  PReduce(uint64_t *__restrict__ data) {
    const int sid = blockIdx.x << (MAX_STEPS - 1) | threadIdx.x >> 2;
    uint64_t a_r[BATCH];
    // ntt::NTTField::load<ntt::Stride<4 << n>>(a_r, data + (sid << 2));
    // ntt::NTTField::p_reduce(a_r);
    // ntt::NTTField::store<ntt::Stride<4 << n>>(data + (sid << 2), a_r);
  }

  __device__ __forceinline__ static void
  ReshapeInput(uint64_t *__restrict__ in,
              uint64_t *__restrict__ out) {
    const int bid = blockIdx.x;
    const int tid = threadIdx.x;
    int loc = (bid << MAX_STEPS) | tid;
    for (int batch = 0; batch < BATCH; ++batch) {
#pragma unroll
      for (int offset = 0; offset < 4; ++offset) {
        out[batch << (n_size + 2) | loc << 2 | offset] =
            in[loc * 12 + offset * 3 + batch];
      }
    }
  }

  __device__ __forceinline__ static void
  ReshapeBack(uint64_t *__restrict__ out,
              uint64_t *__restrict__ in) {
    const int bid = blockIdx.x;
    const int tid = threadIdx.x;
    int loc = (bid << MAX_STEPS) | tid;
    for (int batch = 0; batch < BATCH; ++batch) {
#pragma unroll
      for (int offset = 0; offset < 4; ++offset) {
        out[loc * 12 + offset * 3 + batch] =
            in[batch << (n_size + 2) | loc << 2 | offset];
      }
    }
  }

#define G(batch, row, column, offset, nsqrt)                                          \
  gdata[(batch) << (n + 2) | (row) << (nsqrt + 2) | (column) << 2 | (offset)]

  //only for even
  __device__ __forceinline__ static void
  BitReverse(uint64_t *__restrict__ gdata, int reduce_scale) {
    const int column = threadIdx.x + ((blockIdx.x % (1 << reduce_scale)) * (1 << 9));
    const int row = blockIdx.x / (1 << reduce_scale);
    const int rrow = index_bit_reverse(row);
    const int rcolumn = index_bit_reverse(column);
    if (row < rcolumn || row == rcolumn && column < rrow) {
#pragma unroll
      for (int batch = 0; batch < BATCH; ++batch) {
#pragma unroll
        for (int offset = 0; offset < 4; ++offset) {
          const uint64_t src = G(batch, row, column, offset, nsqrt);
          const uint64_t dst = G(batch, rcolumn, rrow, offset, nsqrt);
          G(batch, row, column, offset, nsqrt) = dst;
          G(batch, rcolumn, rrow, offset, nsqrt) = src;
        }
      }
    }
  }
#undef G

  __device__ __forceinline__ static void General_BitReverse(uint64_t *__restrict__ gdata) {
    const int bid = blockIdx.x;
    const int sid = fixnum::slot_id<BIG_WIDTH_R>();
    const int k = (bid << MAX_STEPS_16) | sid;
    const size_t rk = bitreverse(k, n);
    if (k < rk) {
      Fr_MNT4 src;
      Fr_MNT4::load<ELT_LIMBS_R, BIG_WIDTH_R>(src, gdata+k*LIMBS);
      Fr_MNT4 dst;
      Fr_MNT4::load<ELT_LIMBS_R, BIG_WIDTH_R>(dst, gdata+rk*LIMBS);
      Fr_MNT4::store<ELT_LIMBS_R, BIG_WIDTH_R>(gdata+k*LIMBS, dst);
      Fr_MNT4::store<ELT_LIMBS_R, BIG_WIDTH_R>(gdata+rk*LIMBS, src);
    }
  }

private:
  template <int uroot, int coset>
  __device__ __forceinline__ static int twiddle_index(int u, int c) {
    return (IndexInversion<n1, uroot>::inverse(u) +
            IndexInversion<n1, coset>::inverse(c)) &
           ((1 << n1) - 1);
  }

  __device__ __forceinline__ static int index_bit_reverse(int index) {
    return BIT_REVERSE_TABLE[index];
  }

  __device__ __forceinline__ static size_t bitreverse(size_t n, const size_t l)
  {
    size_t r = 0;
    for (size_t k = 0; k < l; ++k)
    {
        r = (r << 1) | (n & 1);
        n >>= 1;
    }
    return r;
  }

}; // namespace ntt

template <int n_size>
__global__ __launch_bounds__(MAX_THREADS,
                             2) void PreparePowersInit(uint64_t *powers) {
  BasicBlockNTT<n_size>::PreparePowersInit(powers);
}

template <int n_size, int noffset, int nstep>
__global__ __launch_bounds__(MAX_THREADS,
                             2) void PreparePowers(uint64_t *powers) {
  BasicBlockNTT<n_size>::template PreparePowers<noffset, nstep>(powers);
}

template <int n_size> void prepare_powers(uint64_t *powers, cudaStream_t stream) {
  PreparePowersInit<n_size><<<1, LIMBS>>>(powers);
#define PREPARE(offset, step)                                                  \
  PreparePowers<n_size, (offset), (step)>                                       \
      <<<1 << (offset), 1 << ((step) + LOG_WIDTH - 1), 0, stream>>>(powers)

  const int n1 = n_size + 1;
  if (n1 % MAX_STEPS_16 != 0) {
    PREPARE(0, n1%MAX_STEPS_16);
  }
  if (n1 >= 2* MAX_STEPS_16) {
    PREPARE(n1%MAX_STEPS_16, MAX_STEPS_16);
  }
  if (n1 >= 3* MAX_STEPS_16) {
    PREPARE(n1%MAX_STEPS_16 + MAX_STEPS_16, MAX_STEPS_16);
  }
  if (n1 > MAX_STEPS_16) {
    PREPARE(n1 - MAX_STEPS_16, MAX_STEPS_16);
  }

#undef PREPARE
#ifdef DEBUG_DUMP_PREPARED_POWERS
  static const int n1 = n_size + 1;
  uint64_t *h_powers = new uint64_t[LIMBS << n1];
  cudaMemcpy(h_powers, powers, BYTES << n1, cudaMemcpyDeviceToHost);
  for (int i = 0; i < 1 << n1; ++i) {
    for (int j = 0; j < LIMBS; ++j) {
      printf("%lu%c", ColumnMajored<n1>::load(h_powers, i, j),
             " \n"[j + 1 == LIMBS]);
    }
  }
  delete[] h_powers;
#endif
}


template <int n_size, int noffset, int nstep>
__global__ __launch_bounds__(MAX_THREADS, 2) void PrepareDITTwiddles(
    const uint64_t *__restrict__ powers, uint64_t *__restrict__ twiddles) {
  BasicBlockNTT<n_size>::template PrepareTwiddles<noffset, nstep, -1, 0>(powers, twiddles);
}

template <int n_size, int noffset, int nstep>
__global__ __launch_bounds__(MAX_THREADS, 2) void PrepareDIFTwiddles(
  const uint64_t *__restrict__ powers, uint64_t *__restrict__ twiddles) {
  BasicBlockNTT<n_size>::template PrepareTwiddles<noffset, nstep, 1, 1>(powers, twiddles);
}

template <int n_size, int noffset, int nstep>
__global__ __launch_bounds__(MAX_THREADS, 2) void PrepareINVDIFTwiddles(
  const uint64_t *__restrict__ powers, uint64_t *__restrict__ twiddles) {
  BasicBlockNTT<n_size>::template PrepareTwiddles<noffset, nstep, 1, -1>(powers, twiddles);
}

template <int n_size>
__global__ __launch_bounds__(MAX_THREADS, 2) void ReshapeInput(
  uint64_t *__restrict__ in, uint64_t *__restrict__ out) {
  BasicBlockNTT<n_size>::ReshapeInput(in, out);
}

template <int n_size>
__global__ __launch_bounds__(MAX_THREADS, 2) void ReshapeBack(
  uint64_t *__restrict__ out, uint64_t *__restrict__ in) {
  BasicBlockNTT<n_size>::ReshapeBack(out, in);
}

template <int n_size>
void prepare_dit_twiddles(const uint64_t *__restrict__ powers,
                  uint64_t *__restrict__ twiddles, cudaStream_t stream) {
#define PREPARE(offset, step)                                                  \
  PrepareDITTwiddles<n_size, (offset), (step)>                                       \
      <<<1 << (offset), 1 << (step), 0, stream>>>(powers, twiddles)
  PREPARE(n_size - MAX_STEPS_16 - 1, MAX_STEPS_16);
#undef PREPARE
}

template <int n_size>
void prepare_dif_twiddles(const uint64_t *__restrict__ powers,
                  uint64_t *__restrict__ twiddles, cudaStream_t stream) {
#define PREPARE(offset, step)                                                  \
  PrepareDIFTwiddles<n_size, (offset), (step)>                                       \
      <<<1 << (offset), 1 << (step), 0, stream>>>(powers, twiddles)
  PREPARE(n_size - MAX_STEPS_16 - 1, MAX_STEPS_16);
#undef PREPARE
}

template <int n_size>
void prepare_inv_dif_twiddles(const uint64_t *__restrict__ powers,
                  uint64_t *__restrict__ twiddles, cudaStream_t stream) {
#define PREPARE(offset, step)                                                  \
  PrepareINVDIFTwiddles<n_size, (offset), (step)>                                       \
      <<<1 << (offset), 1 << (step), 0, stream>>>(powers, twiddles)
  PREPARE(n_size - MAX_STEPS_16 - 1, MAX_STEPS_16);
#undef PREPARE
}

template <template <int> class IdTrait, int level_offset, int n_size>
__global__ __launch_bounds__(MAX_THREADS, 1) void DecimationInTimeByBlock(
    const uint64_t *__restrict__ twiddles, uint64_t *__restrict__ data,
    int start, int end) {
  BasicBlockNTT<n_size>::template DecimationInTime<IdTrait>(
    twiddles, data, level_offset, start, end);
}

template <template <int> class IdTrait, int level_offset, int n_size, bool DEBUG=false>
__global__ __launch_bounds__(MAX_THREADS, 1) void DecimationInFreqByBlock(
    const uint64_t *__restrict__ twiddles, uint64_t *__restrict__ data, 
    int start, int end) {
  BasicBlockNTT<n_size>::template DecimationInFreq<IdTrait>(
    twiddles, data, level_offset, start, end);
}

template <int n_size, int nsqrt>
__global__ __launch_bounds__(1024,
                             2) void BitReverse(uint64_t *__restrict__ data,
                                                int reduce_scale) {
  BasicBlockNTT<n_size>::BitReverse(data, reduce_scale);
}

template <int n_size>
__global__ __launch_bounds__(1024,
                             2) void General_BitReverse(uint64_t *__restrict__ data) {
  BasicBlockNTT<n_size>::General_BitReverse(data);
}

template <int n_size>
__global__ __launch_bounds__(MAX_THREADS, 2) void DotProduct(
    uint64_t *__restrict__ out, const uint64_t *__restrict__ a,
    const uint64_t *__restrict__ b, const uint64_t *__restrict__ c) {
  BasicBlockNTT<n_size>::DotProduct(out, a, b, c);
}

template <int n_size>
__global__ __launch_bounds__(MAX_THREADS, 2) void PReduce(
    uint64_t *__restrict__ data) {
  BasicBlockNTT<n_size>::PReduce(data);
}

template <int n_size>
void decimation_in_time(const uint64_t *__restrict__ twiddles,
                      uint64_t *__restrict__ data, cudaStream_t stream) {
  // NOTE: can only handle n_size < 27
  int blockNums = 1 << (n_size-MAX_STEPS_16);
  if (n_size >= MAX_STEPS_16) {
    DecimationInTimeByBlock<MinIdTrait, 0, n_size>
      <<< blockNums,  1 << (MAX_STEPS_16+LOG_WIDTH-1), 0, stream>>>(twiddles, data, 0, MAX_STEPS_16);
  }
  if (n_size >=  2*MAX_STEPS_16) {
    DecimationInTimeByBlock<MedianIdTrait, MAX_STEPS_16, n_size>
      <<< blockNums,  1 << (MAX_STEPS_16+LOG_WIDTH-1), 0, stream>>>(twiddles, data, MAX_STEPS_16, 2*MAX_STEPS_16);
  }
  if (n_size >=  3*MAX_STEPS_16) {
    DecimationInTimeByBlock<MedianIdTrait_1, 2*MAX_STEPS_16, n_size>
      <<< blockNums,  1 << (MAX_STEPS_16+LOG_WIDTH-1), 0, stream>>>(twiddles, data, 2*MAX_STEPS_16, 3*MAX_STEPS_16);
  }
  if (n_size >=  4*MAX_STEPS_16) {
    DecimationInTimeByBlock<MedianIdTrait_2, 3*MAX_STEPS_16, n_size>
      <<< blockNums,  1 << (MAX_STEPS_16+LOG_WIDTH-1), 0, stream>>>(twiddles, data, 3*MAX_STEPS_16, 4*MAX_STEPS_16);
  }
  if ((n_size % MAX_STEPS_16) != 0) {
    DecimationInTimeByBlock<MaxIdTrait, n_size-(n_size % MAX_STEPS_16), n_size>
      <<< blockNums,  1 << (MAX_STEPS_16+LOG_WIDTH-1), 0, stream>>>(twiddles, data, n_size-(n_size % MAX_STEPS_16), n_size);
  }
}

template <int n_size, bool DEBUG = false>
void decimation_in_freq(const uint64_t *__restrict__ twiddles,
                        uint64_t *__restrict__ data, cudaStream_t stream) {
  int blockNums = 1 << (n_size - MAX_STEPS_16);
  if ((n_size % MAX_STEPS_16) != 0) {
    DecimationInFreqByBlock<MaxIdTrait, n_size-(n_size % MAX_STEPS_16), n_size>
      <<< blockNums,  1 << (MAX_STEPS_16+LOG_WIDTH-1), 0, stream>>>(twiddles, data, 0, n_size % MAX_STEPS_16);
  }
  if (n_size >=  4*MAX_STEPS_16) {
    DecimationInFreqByBlock<MedianIdTrait_2, 3*MAX_STEPS_16, n_size>
      <<< blockNums,  1 << (MAX_STEPS_16+LOG_WIDTH-1), 0, stream>>>(twiddles, data, n_size % MAX_STEPS_16, n_size - 3*MAX_STEPS_16);
  }
  if (n_size >=  3*MAX_STEPS_16) {
    DecimationInFreqByBlock<MedianIdTrait_1, 2*MAX_STEPS_16, n_size>
      <<< blockNums,  1 << (MAX_STEPS_16+LOG_WIDTH-1), 0, stream>>>(twiddles, data, n_size - 3*MAX_STEPS_16, n_size - 2*MAX_STEPS_16);
  }
  if (n_size >=  2*MAX_STEPS_16) {
    DecimationInFreqByBlock<MedianIdTrait, MAX_STEPS_16, n_size>
      <<< blockNums,  1 << (MAX_STEPS_16+LOG_WIDTH-1), 0, stream>>>(twiddles, data, n_size - 2*MAX_STEPS_16, n_size - MAX_STEPS_16);
  }
  if (n_size >= MAX_STEPS_16) {
    DecimationInFreqByBlock<MinIdTrait, 0, n_size>
      <<< blockNums,  1 << (MAX_STEPS_16+LOG_WIDTH-1), 0, stream>>>(twiddles, data, n_size - MAX_STEPS_16, n_size);
  }
}

template <int n_size>
void dot_product(uint64_t *__restrict__ out, const uint64_t *__restrict__ a,
                 const uint64_t *__restrict__ b,
                 const uint64_t *__restrict__ c, cudaStream_t stream) {
  if (n_size >= MAX_STEPS_16) {
    // n_size >= 9
    int blockNums = 1 << (n_size - MAX_STEPS_16);
    DotProduct<n_size><<<blockNums, MAX_THREADS, 0, stream>>>(out, a, b, c);
  }
  else {
    // n_size < 9
    DotProduct<n_size><<<1, 1 << (n_size + 1), 0, stream>>>(out, a, b, c);  
  }
}

template <int n_size>
void p_reduce(uint64_t *__restrict__ data, cudaStream_t stream) {
  if (n_size >= MAX_STEPS) {
    //n_size > 9
    int blockNums = 1 << (n_size - MAX_STEPS + 1);
    PReduce<n_size><<<blockNums, 1 << (MAX_STEPS+1), 0, stream>>>(data);
  }
  else {
    //n_size < 9
    PReduce<n_size><<<1, 1 << (n_size + 1), 0, stream>>>(data);  
  }
}

template <int n_size>
void reshape_input(uint64_t *__restrict__ in,
                  uint64_t *__restrict__ out, cudaStream_t stream) {
  ReshapeInput<n_size>                                       \
    <<<1 << (n_size - MAX_STEPS), 1 << (MAX_STEPS), 0, stream>>>(in, out);
}

template <int n_size>
void reshape_back(uint64_t *__restrict__ out,
                  uint64_t *__restrict__ in, cudaStream_t stream) {
  ReshapeBack<n_size>                                       \
    <<<1 << (n_size - MAX_STEPS), 1 << (MAX_STEPS), 0, stream>>>(out, in);
}

// can only handle n_size is even
template <int n_size, int nsqrt> void bit_reverse(uint64_t *__restrict__ data,
                                                   cudaStream_t stream) {
  int reduce_scale = (nsqrt > MAX_STEPS) * (nsqrt - MAX_STEPS);
  BitReverse<n_size, nsqrt><<<1 << (nsqrt + reduce_scale), 1 << (nsqrt - reduce_scale), 0, stream>>>(data, reduce_scale);  
}

template <int n_size> void general_bit_reverse(uint64_t *__restrict__ data,
                                                   cudaStream_t stream) {
  General_BitReverse<n_size><<< 1<<(n_size - MAX_STEPS_16 + 1), MAX_THREADS, 0, stream>>>(data);
}

} // namespace ntt

} // namespace gzkp

