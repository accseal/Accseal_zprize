#pragma once

#include <cstdint>

namespace gzkp {

#ifdef MNT4753
static const size_t LIMBS = 12;
// ShareMemory的大小为BYTES*2^MAX_SHAREMEM
static const int MAX_SHAREMEM_16 = 6; //1024/16=1<<6
// 最多一次性能够处理的蝶形操作的轮数,由ShareMemory决定
static const int MAX_STEPS_16 = MAX_SHAREMEM_16;
static const int LOG_WIDTH = 4;
#endif
#ifdef ALT_BN128
static const size_t LIMBS = 4;
// ShareMemory的大小为BYTES*2^MAX_SHAREMEM
static const int MAX_SHAREMEM_16 = 8; //1024/4=1<<8
// 最多一次性能够处理的蝶形操作的轮数,由ShareMemory决定
static const int MAX_STEPS_16 = MAX_SHAREMEM_16; // =8
static const int LOG_WIDTH = 2;
#endif
#ifdef BLS12_381
static const size_t LIMBS = 4;
// ShareMemory的大小为BYTES*2^MAX_SHAREMEM
static const int MAX_SHAREMEM_16 = 8; //1024/4=1<<8
// 最多一次性能够处理的蝶形操作的轮数,由ShareMemory决定
static const int MAX_STEPS_16 = MAX_SHAREMEM_16; // =8
static const int LOG_WIDTH = 2;
#endif
static const size_t BYTES = LIMBS * sizeof(uint64_t);

static const int SM_NUM = 80;
static const int WARP_SIZE = 32;
static const int MAX_THREADS = 1024;

// ShareMemory的大小为BYTES*2^MAX_SHAREMEM
static const int MAX_SHAREMEM = 9;
// 最多一次性能够处理的蝶形操作的轮数,由ShareMemory决定
static const int MAX_STEPS = MAX_SHAREMEM;

} // namespace gzkp