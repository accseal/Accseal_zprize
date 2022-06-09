#pragma once

#include <cooperative_groups.h>

#include "primitives.cuh"

/*
 * var is the basic register type that we deal with. The
 * interpretation of (one or more) such registers is determined by the
 * struct used, e.g. digit, fixnum, etc.
 */
typedef std::uint64_t var;

static constexpr size_t ELT_LIMBS_Q = 6;
static constexpr size_t ELT_BYTES_Q = ELT_LIMBS_Q * sizeof(var);
static constexpr size_t BIG_WIDTH_Q = ELT_LIMBS_Q + 2; // = 8
static constexpr size_t ELT_LIMBS_R = 4;
static constexpr size_t ELT_BYTES_R = ELT_LIMBS_R * sizeof(var);
static constexpr size_t BIG_WIDTH_R = ELT_LIMBS_R; 

struct digit {
  static constexpr int BYTES = sizeof(var);
  static constexpr int BITS = BYTES * 8;

  __device__ __forceinline__ static void add(var &s, var a, var b) {
    s = a + b;
  }

  __device__ __forceinline__ static void add_cy(var &s, int &cy, var a, var b) {
    s = a + b;
    cy = s < a;
  }

  __device__ __forceinline__ static void sub(var &d, var a, var b) {
    d = a - b;
  }

  __device__ __forceinline__ static void sub_br(var &d, int &br, var a, var b) {
    d = a - b;
    br = d > a;
  }

  __device__ __forceinline__ static var zero() { return 0ULL; }

  __device__ __forceinline__ static int is_max(var a) { return a == ~0ULL; }

  __device__ __forceinline__ static int is_min(var a) { return a == 0ULL; }

  __device__ __forceinline__ static int is_zero(var a) { return a == zero(); }

  __device__ __forceinline__ static void mul_lo(var &lo, var a, var b) {
    lo = a * b;
  }

  // lo = a * b + c (mod 2^64)
  __device__ __forceinline__ static void mad_lo(var &lo, var a, var b, var c) {
    internal::mad_lo(lo, a, b, c);
  }

  // as above but increment cy by the mad carry
  __device__ __forceinline__ static void mad_lo_cy(var &lo, int &cy, var a,
                                                   var b, var c) {
    internal::mad_lo_cc(lo, a, b, c);
    internal::addc(cy, cy, 0);
  }

  __device__ __forceinline__ static void mad_hi(var &hi, var a, var b, var c) {
    internal::mad_hi(hi, a, b, c);
  }

  // as above but increment cy by the mad carry
  __device__ __forceinline__ static void mad_hi_cy(var &hi, int &cy, var a,
                                                   var b, var c) {
    internal::mad_hi_cc(hi, a, b, c);
    internal::addc(cy, cy, 0);
  }
};

struct fixnum {
  // TODO: Previous versiona allowed 'auto' return type here instead
  // of this mess
  template <int BIG_WIDTH_ = BIG_WIDTH_Q>
  __device__ static cooperative_groups::thread_block_tile<BIG_WIDTH_> layout() {
    return cooperative_groups::tiled_partition<BIG_WIDTH_>(
        cooperative_groups::this_thread_block());
  }

  template <int BIG_WIDTH_ = BIG_WIDTH_Q>
  __device__ __forceinline__ static int lane_id() {
    return layout<BIG_WIDTH_>().thread_rank();
  }

  template <int BIG_WIDTH_ = BIG_WIDTH_Q>
  __device__ __forceinline__ static int slot_id() {
    return threadIdx.x / BIG_WIDTH_;
  }

  template <int BIG_WIDTH_ = BIG_WIDTH_Q>
  __device__ __forceinline__ static var zero() { return digit::zero(); }

  template <int BIG_WIDTH_ = BIG_WIDTH_Q>
  __device__ __forceinline__ static var one() {
    auto t = layout<BIG_WIDTH_>().thread_rank();
    return (var)(t == 0);
  }

  template <int BIG_WIDTH_ = BIG_WIDTH_Q>
  __device__ static void add_cy(var &r, int &cy_hi, const var &a,
                                const var &b) {
    int cy;
    digit::add_cy(r, cy, a, b);
    // r propagates carries iff r = FIXNUM_MAX
    var r_cy = effective_carries<BIG_WIDTH_>(cy_hi, digit::is_max(r), cy);
    digit::add(r, r, r_cy);
  }

  template <int BIG_WIDTH_ = BIG_WIDTH_Q>
  __device__ static void add(var &r, const var &a, const var &b) {
    int cy_hi;
    add_cy<BIG_WIDTH_>(r, cy_hi, a, b);
  }

  template <int BIG_WIDTH_ = BIG_WIDTH_Q>
  __device__ static void sub_br(var &r, int &br_lo, const var &a,
                                const var &b) {
    int br;
    digit::sub_br(r, br, a, b);
    // r propagates borrows iff r = FIXNUM_MIN
    var r_br = effective_carries<BIG_WIDTH_>(br_lo, digit::is_min(r), br);
    digit::sub(r, r, r_br);
  }

  template <int BIG_WIDTH_ = BIG_WIDTH_Q>
  __device__ static void sub(var &r, const var &a, const var &b) {
    int br_lo;
    sub_br<BIG_WIDTH_>(r, br_lo, a, b);
  }

  template <int BIG_WIDTH_ = BIG_WIDTH_Q>
  __device__ static uint32_t nonzero_mask(var r) {
    return fixnum::layout<BIG_WIDTH_>().ballot(!digit::is_zero(r));
  }

  template <int BIG_WIDTH_ = BIG_WIDTH_Q>
  __device__ static int is_zero(var r) { return nonzero_mask<BIG_WIDTH_>(r) == 0U; }

  template <int BIG_WIDTH_ = BIG_WIDTH_Q>
  __device__ static int most_sig_dig(var x) {
    enum { UINT32_BITS = 8 * sizeof(uint32_t) };

    uint32_t a = nonzero_mask<BIG_WIDTH_>(x);
    return UINT32_BITS - (internal::clz(a) + 1);
  }

  template <int BIG_WIDTH_ = BIG_WIDTH_Q>
  __device__ static int cmp(var x, var y) {
    var r;
    int br;
    sub_br<BIG_WIDTH_>(r, br, x, y);
    // r != 0 iff x != y. If x != y, then br != 0 => x < y.
    return nonzero_mask<BIG_WIDTH_>(r) ? (br ? -1 : 1) : 0;
  }

  template <int BIG_WIDTH_ = BIG_WIDTH_Q>
  __device__ static var effective_carries(int &cy_hi, int propagate, int cy) {
    uint32_t allcarries, p, g;
    auto grp = fixnum::layout<BIG_WIDTH_>();

    g = grp.ballot(cy);                       // carry generate
    p = grp.ballot(propagate);                // carry propagate
    allcarries = (p | g) + g;                 // propagate all carries
    cy_hi = (allcarries >> grp.size()) & 1;   // detect hi overflow
    allcarries = (allcarries ^ p) | (g << 1); // get effective carries
    return (allcarries >> grp.thread_rank()) & 1;
  }
};


// Apparently we still can't do partial specialisation of function
// templates in C++, so we do it in a class instead. Woot.
template <int n> struct mul_ {
  template <typename G> __device__ static void x(G &z, const G &x);
};

template <> template <typename G> __device__ void mul_<2>::x(G &z, const G &x) {
  // TODO: Shift by one bit
  G::add(z, x, x);
}

template <> template <typename G> __device__ void mul_<4>::x(G &z, const G &x) {
  // TODO: Shift by two bits
  mul_<2>::x(z, x); // z = 2x
  mul_<2>::x(z, z); // z = 4x
}

template <> template <typename G> __device__ void mul_<8>::x(G &z, const G &x) {
  // TODO: Shift by three bits
  mul_<4>::x(z, x); // z = 4x
  mul_<2>::x(z, z); // z = 8x
}

template <>
template <typename G>
__device__ void mul_<16>::x(G &z, const G &x) {
  // TODO: Shift by four bits
  mul_<8>::x(z, x); // z = 8x
  mul_<2>::x(z, z); // z = 16x
}

template <>
template <typename G>
__device__ void mul_<32>::x(G &z, const G &x) {
  // TODO: Shift by five bits
  mul_<16>::x(z, x); // z = 16x
  mul_<2>::x(z, z);  // z = 32x
}

template <>
template <typename G>
__device__ void mul_<64>::x(G &z, const G &x) {
  // TODO: Shift by six bits
  mul_<32>::x(z, x); // z = 32x
  mul_<2>::x(z, z);  // z = 64x
}

template <> template <typename G> __device__ void mul_<3>::x(G &z, const G &x) {
  G t;
  mul_<2>::x(t, x);
  G::add(z, t, x);
}

template <>
template <typename G>
__device__ void mul_<11>::x(G &z, const G &x) {
  // TODO: Do this without carry/overflow checks
  // TODO: Check that this is indeed optimal
  // 11 = 8 + 2 + 1
  G t;
  mul_<2>::x(t, x); // t = 2x
  G::add(z, t, x);  // z = 3x
  mul_<4>::x(t, t); // t = 8x
  G::add(z, z, t);  // z = 11x
}

template <>
template <typename G>
__device__ void mul_<13>::x(G &z, const G &x) {
  // 13 = 8 + 4 + 1
  G t;
  mul_<4>::x(t, x); // t = 4x
  G::add(z, t, x);  // z = 5x
  mul_<2>::x(t, t); // t = 8x
  G::add(z, z, t);  // z = 13x
}

template <>
template <typename G>
__device__ void mul_<26>::x(G &z, const G &x) {
  // 26 = 16 + 8 + 2
  G t;
  mul_<2>::x(z, x); // z = 2x
  mul_<4>::x(t, z); // t = 8x
  G::add(z, z, t);  // z = 10x
  mul_<2>::x(t, t); // t = 16x
  G::add(z, z, t);  // z = 26x
}

template <>
template <typename G>
__device__ void mul_<121>::x(G &z, const G &x) {
  // 121 = 64 + 32 + 16 + 8 + 1
  G t;
  mul_<8>::x(t, x); // t = 8x
  G::add(z, t, x);  // z = 9x
  mul_<2>::x(t, t); // t = 16x
  G::add(z, z, t);  // z = 25x
  mul_<2>::x(t, t); // t = 32x
  G::add(z, z, t);  // z = 57x
  mul_<2>::x(t, t); // t = 64x
  G::add(z, z, t);  // z = 121x
}

// TODO: Bleughk! This is obviously specific to MNT6 curve over Fp3.
template <>
template <typename Fp3>
__device__ void mul_<-1>::x(Fp3 &z, const Fp3 &x) {
  // multiply by (0, 0, 11) = 11 x^2  (where x^3 = alpha)
  static constexpr int CRV_A = 11;
  static constexpr int ALPHA = 11;
  Fp3 y = x;
  mul_<CRV_A * ALPHA>::x(z.a0, y.a1);
  mul_<CRV_A * ALPHA>::x(z.a1, y.a2);
  mul_<CRV_A>::x(z.a2, y.a0);
}