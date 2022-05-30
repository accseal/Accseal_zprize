#pragma once

#include "fixnum.cuh"

#ifdef MNT4753
__device__ __constant__ const var MOD_Q[BIG_WIDTH_Q] = {
    0x5e9063de245e8001ULL,
    0xe39d54522cdd119fULL,
    0x638810719ac425f0ULL,
    0x685acce9767254a4ULL,
    0xb80f0da5cb537e38ULL,
    0xb117e776f218059dULL,
    0x99d124d9a15af79dULL,
    0x7fdb925e8a0ed8dULL,
    0x5eb7e8f96c97d873ULL,
    0xb7f997505b8fafedULL,
    0x10229022eee2cdadULL,
    0x1c4c62d92c411ULL

    ,
    0x0ULL,
    0x0ULL,
    0x0ULL,
    0x0ULL // just to make an even 16
};

// -Q^{-1} (mod 2^64)
static constexpr var Q_NINV_MOD = 0xf2044cfbe45e7fffULL;

// 2^768 mod Q
__device__ __constant__ const var X_MOD_Q[BIG_WIDTH_Q] = {
    0x98a8ecabd9dc6f42ULL,
    0x91cd31c65a034686ULL,
    0x97c3e4a0cd14572eULL,
    0x79589819c788b601ULL,
    0xed269c942108976fULL,
    0x1e0f4d8acf031d68ULL,
    0x320c3bb713338559ULL,
    0x598b4302d2f00a62ULL,
    0x4074c9cbfd8ca621ULL,
    0xfa47edb3865e88cULL,
    0x95455fb31ff9a195ULL,
    0x7b479ec8e242ULL

    ,
    0x0ULL,
    0x0ULL,
    0x0ULL,
    0x0ULL // just to make an even 16
};

__device__ __constant__ const var X_MOD_Q_SQUARE[BIG_WIDTH_Q] = {
    9543532818279272648, 14400822270957206538, 11694698513092523271, 12148079935867669629, 
    9269646469758926303, 3406836361927163112, 12162090941207000471, 16157923474893434631, 
    2667779087013515845, 10251887603103569699, 12718712199237708975, 46402434282629,
    0x0ULL, 0x0ULL, 0x0ULL, 0x0ULL // just to make an even 16
};

__device__ __constant__ const var MOD_R[BIG_WIDTH_Q] = {
    0xd90776e240000001ULL,
    0x4ea099170fa13a4fULL,
    0xd6c381bc3f005797ULL,
    0xb9dff97634993aa4ULL,
    0x3eebca9429212636ULL,
    0xb26c5c28c859a99bULL,
    0x99d124d9a15af79dULL,
    0x7fdb925e8a0ed8dULL,
    0x5eb7e8f96c97d873ULL,
    0xb7f997505b8fafedULL,
    0x10229022eee2cdadULL,
    0x1c4c62d92c411ULL

    ,
    0x0ULL,
    0x0ULL,
    0x0ULL,
    0x0ULL // just to make an even 16
};

// -R^{-1} (mod 2^64)
const var R_NINV_MOD = 0xc90776e23fffffffULL;

// 2^768 mod R
__device__ __constant__ const var X_MOD_R[BIG_WIDTH_Q] = {
    0xb99680147fff6f42ULL,
    0x4eb16817b589cea8ULL,
    0xa1ebd2d90c79e179ULL,
    0xf725caec549c0daULL,
    0xab0c4ee6d3e6dad4ULL,
    0x9fbca908de0ccb62ULL,
    0x320c3bb713338498ULL,
    0x598b4302d2f00a62ULL,
    0x4074c9cbfd8ca621ULL,
    0xfa47edb3865e88cULL,
    0x95455fb31ff9a195ULL,
    0x7b479ec8e242ULL

    ,
    0x0ULL,
    0x0ULL,
    0x0ULL,
    0x0ULL // just to make an even 16
};

#endif

#ifdef ALT_BN128
// alt_bn128_Fq2::non_residue.mont_repr.print_hex();
// 2259d6b14729c0fa51e1a247090812318d087f6872aabf4f68c3488912edefaa
__device__ __constant__ const var ALPHA_VALUE[BIG_WIDTH_Q] = {
    // 0x3c208c16d87cfd46ULL, 0x97816a916871ca8dULL,
    // 0xb85045b68181585dULL, 0x30644e72e131a029ULL
    0x68c3488912edefaaULL, 0x8d087f6872aabf4fULL,
    0x51e1a24709081231ULL, 0x2259d6b14729c0faULL
};

__device__ __constant__ const var X_MOD_Q_SQUARE[BIG_WIDTH_Q] = {
  17522657719365597833ULL, 13107472804851548667ULL, 5164255478447964150ULL, 493319470278259999ULL
};

__device__ __constant__ const var MOD_Q[BIG_WIDTH_Q] = {
  0x3c208c16d87cfd47, 0x97816a916871ca8d, 0xb85045b68181585d, 0x30644e72e131a029
};

static constexpr var Q_NINV_MOD = 0x87d20782e4866389;

__device__ __constant__ const var X_MOD_Q[BIG_WIDTH_Q] = {
    0xd35d438dc58f0d9d, 0x0a78eb28f5c70b3d, 0x666ea36f7879462c, 0xe0a77c19a07df2f
};

__device__ __constant__ const var MOD_R[BIG_WIDTH_Q] = {
  0x43e1f593f0000001, 0x2833e84879b97091, 0xb85045b68181585d, 0x30644e72e131a029
};
// -R^{-1} (mod 2^64)
const var R_NINV_MOD = 0xc2e1f593efffffff;

// 2^768 mod R
__device__ __constant__ const var X_MOD_R[BIG_WIDTH_Q] = {
  0xac96341c4ffffffb, 0x36fc76959f60cd29, 0x666ea36f7879462e, 0xe0a77c19a07df2f
};
#endif

#ifdef BLS12_381
// not this bls12_381_Fq2::non_residue.mont_repr.print_hex();
// not this 40ab3263eff0206ef148d1ea0f4c069eca8f3318332bb7a07e83a49a2e99d6932b7fff2ed47fffd43f5fffffffcaaae
__device__ __constant__ const var ALPHA_VALUE[BIG_WIDTH_Q] = {
  0x43f5fffffffcaaae, 0x32b7fff2ed47fffd,
  0x07e83a49a2e99d69, 0xeca8f3318332bb7a,
  0xef148d1ea0f4c069, 0x40ab3263eff0206,
  0x0, 0x0
};

// 1a0111ea397fe69a4b1ba7b6434bacd764774b84f38512bf6730d2a0f6b0f6241eabfffeb153ffff b9feffffffffaaab
__device__ __constant__ const var MOD_Q[BIG_WIDTH_Q] = {
  13402431016077863595ULL, 2210141511517208575ULL, 7435674573564081700ULL, 7239337960414712511ULL, 
  5412103778470702295ULL, 1873798617647539866ULL, 0ULL, 0ULL
};

static constexpr var Q_NINV_MOD = 9940570264628428797ULL;

// 15f65ec3fa80e4935c071a97a256ec6d77ce5853705257455f48985753c758baebf4000bc40c0002760900000002fffd
__device__ __constant__ const var X_MOD_Q[BIG_WIDTH_Q] = {
  8505329371266088957ULL, 17002214543764226050ULL, 6865905132761471162ULL, 8632934651105793861ULL, 
  6631298214892334189ULL, 1582556514881692819ULL, 0ULL, 0ULL
};

__device__ __constant__ const var X_MOD_Q_SQUARE[BIG_WIDTH_Q] = {
  17644856173732828998ULL, 754043588434789617ULL, 10224657059481499349ULL, 7488229067341005760ULL, 
  11130996698012816685ULL, 1267921511277847466ULL, 0ULL, 0ULL
};

// 73eda753299d7d483339d80809a1d80553bda402fffe5bfeffffffff00000001
__device__ __constant__ const var MOD_R[] = {
  18446744069414584321, 6034159408538082302, 3691218898639771653, 8353516859464449352,
  0, 0, 0, 0
};
// -R^{-1} (mod 2^64)
const var R_NINV_MOD = 18446744069414584319;

// 2^768 mod R
// 1824b159acc5056f998c4fefecbc4ff55884b7fa0003480200000001fffffffe
__device__ __constant__ const var X_MOD_R[] = {
  8589934590, 6378425256633387010, 11064306276430008309, 1739710354780652911,
  0, 0, 0, 0
};
#endif

struct MOD_Q_ {
  template <int BIG_WIDTH_ = BIG_WIDTH_Q>
  __device__ __forceinline__ static int lane() {
    return fixnum::layout<BIG_WIDTH_>().thread_rank();
  }

  template <int BIG_WIDTH_ = BIG_WIDTH_Q>
  __device__ __forceinline__ static var mod(int lane) { return MOD_Q[lane]; }

  static constexpr var ninv_mod = Q_NINV_MOD;

  template <int BIG_WIDTH_ = BIG_WIDTH_Q>
  __device__ __forceinline__ static var monty_one(int lane) { return X_MOD_Q[lane]; }

  template <int BIG_WIDTH_ = BIG_WIDTH_Q>
  __device__ __forceinline__ static var monty_rsquare(int lane) { return X_MOD_Q_SQUARE[lane]; }
};

struct MOD_R_ {
  template <int BIG_WIDTH_ = BIG_WIDTH_R>
  __device__ __forceinline__ static int lane() {
    return fixnum::layout<BIG_WIDTH_>().thread_rank();
  }

  template <int BIG_WIDTH_ = BIG_WIDTH_R>
  __device__ __forceinline__ static var mod(int lane) { return MOD_R[lane]; }

  static constexpr var ninv_mod = R_NINV_MOD;

  template <int BIG_WIDTH_ = BIG_WIDTH_R>
  __device__ __forceinline__ static var monty_one(int lane) { return X_MOD_R[lane]; }
};

template <typename _modulus_info_, int ELT_LIMBS_, int BIG_WIDTH_> struct Fp_model {
  // typedef Fp_model PrimeField;
  typedef Fp_model Fp;

  using modulus_info = _modulus_info_;

  var a;

  static constexpr int DEGREE = 1;
  static constexpr int ELT_LIMBS = ELT_LIMBS_;
  static constexpr int BIG_WIDTH = BIG_WIDTH_;

  __device__ static void load(Fp &x, const var *mem) {
    int t = fixnum::layout<BIG_WIDTH_>().thread_rank();
    x.a = (t < ELT_LIMBS_) ? mem[t] : 0UL;
  }

  __device__ static void store(var *mem, const Fp &x) {
    int t = fixnum::layout<BIG_WIDTH_>().thread_rank();
    if (t < ELT_LIMBS_)
      mem[t] = x.a;
  }

  __device__ static void store_a(var *mem, const var a) {
    int t = fixnum::layout<BIG_WIDTH_>().thread_rank();
    if (t < ELT_LIMBS_)
      mem[t] = a;
  }

  __device__ static void load_stride(Fp &x, const var *mem, size_t stride) {
    int t = fixnum::layout<BIG_WIDTH_>().thread_rank();
    x.a = (t < ELT_LIMBS_) ? mem[t*stride] : 0UL;
  }

  __device__ static void store_stride(var *mem, const Fp &x, size_t stride) {
    int t = fixnum::layout<BIG_WIDTH_>().thread_rank();
    if (t < ELT_LIMBS_)
      mem[t*stride] = x.a;
  }

  __device__ static void load_alpha(Fp &x) {
#if (defined BLS12_381) || (defined ALT_BN128)
    int t = fixnum::layout<BIG_WIDTH_>().thread_rank();
    x.a = (t < ELT_LIMBS_) ? ALPHA_VALUE[t] : 0UL;
#endif
  }

  __device__ static int are_equal(const Fp &x, const Fp &y) {
    return fixnum::cmp<BIG_WIDTH_>(x.a, y.a) == 0;
  }

  __device__ static void set_zero(Fp &x) { x.a = fixnum::zero<BIG_WIDTH_>(); }

  __device__ static int is_zero(const Fp &x) { return fixnum::is_zero<BIG_WIDTH_>(x.a); }

  __device__ static void set_one(Fp &x) { x.a = modulus_info::monty_one<BIG_WIDTH_>(fixnum::layout<BIG_WIDTH_>().thread_rank()); }

  __device__ static void add(Fp &zz, const Fp &xx, const Fp &yy) {
    int br;
    var x = xx.a, y = yy.a, z, r;
    var mod = modulus_info::mod<BIG_WIDTH_>(fixnum::layout<BIG_WIDTH_>().thread_rank());
    fixnum::add<BIG_WIDTH_>(z, x, y);
    fixnum::sub_br<BIG_WIDTH_>(r, br, z, mod);
    zz.a = br ? z : r;
  }

  __device__ static void neg(Fp &z, const Fp &x) {
    var mod = modulus_info::mod<BIG_WIDTH_>(fixnum::layout<BIG_WIDTH_>().thread_rank());
    fixnum::sub<BIG_WIDTH_>(z.a, mod, x.a);
  }

  __device__ static void sub(Fp &z, const Fp &x, const Fp &y) {
    int br;
    var r, mod = modulus_info::mod<BIG_WIDTH_>(fixnum::layout<BIG_WIDTH_>().thread_rank());
    fixnum::sub_br<BIG_WIDTH_>(r, br, x.a, y.a);
    if (br)
      fixnum::add<BIG_WIDTH_>(r, r, mod);
    z.a = r;
  }

  __device__ __noinline__ static void mul(Fp &zz, const Fp &xx, const Fp &yy) {
    auto grp = fixnum::layout<BIG_WIDTH_>();
    int L = grp.thread_rank();
    var mod = modulus_info::mod<BIG_WIDTH_>(fixnum::layout<BIG_WIDTH_>().thread_rank());

    var x = xx.a, y = yy.a, z = digit::zero();
    var tmp;
    digit::mul_lo(tmp, x, modulus_info::ninv_mod);
    digit::mul_lo(tmp, tmp, grp.shfl(y, 0));
    int cy = 0;

    for (int i = 0; i < ELT_LIMBS_; ++i) {
      var u;
      var xi = grp.shfl(x, i);
      var z0 = grp.shfl(z, 0);
      var tmpi = grp.shfl(tmp, i);

      digit::mad_lo(u, z0, modulus_info::ninv_mod, tmpi);
      digit::mad_lo_cy(z, cy, mod, u, z);
      digit::mad_lo_cy(z, cy, y, xi, z);

      assert(L || z == 0);     // z[0] must be 0
      z = grp.shfl_down(z, 1); // Shift right one word
      z = (L >= ELT_LIMBS_ - 1) ? 0 : z;

      digit::add_cy(z, cy, z, cy);
      digit::mad_hi_cy(z, cy, mod, u, z);
      digit::mad_hi_cy(z, cy, y, xi, z);
    }
    // Resolve carries
    int msb = grp.shfl(cy, ELT_LIMBS_ - 1);
    cy = grp.shfl_up(cy, 1); // left shift by 1
    cy = (L == 0) ? 0 : cy;

    fixnum::add_cy<BIG_WIDTH_>(z, cy, z, cy);
    msb += cy;
    assert(msb == !!msb); // msb = 0 or 1.

    // br = 0 ==> z >= mod
    var r;
    int br;
    fixnum::sub_br<BIG_WIDTH_>(r, br, z, mod);
    if (msb || br == 0) {
      // If the msb was set, then we must have had to borrow.
      assert(!msb || msb == br);
      z = r;
    }
    zz.a = z;
  }

  __device__ static void sqr(Fp &z, const Fp &x) {
    // TODO: Find a faster way to do this. Actually only option
    // might be full squaring with REDC.
    mul(z, x, x);
  }

#if 0
    __device__
    static void
    inv(Fp &z, const Fp &x) {
        // FIXME: Implement!  See HEHCC Algorithm 11.12.
        z = x;
    }
#endif

  __device__ static void from_monty(Fp &y, const Fp &x) {
    Fp one;
    one.a = fixnum::one<BIG_WIDTH_>();
    mul(y, x, one);
  }

  __device__ static void to_monty(Fp &y, const Fp &x) {
    Fp r_square;
    r_square.a = modulus_info::monty_rsquare<BIG_WIDTH_>(fixnum::layout<BIG_WIDTH_>().thread_rank());
    mul(y, x, r_square);
  }
};

// Reference for multiplication and squaring methods below:
// https://pdfs.semanticscholar.org/3e01/de88d7428076b2547b60072088507d881bf1.pdf

template <typename Fp, int ALPHA> struct Fp2_model {
  typedef Fp2_model Fp2;

  // TODO: Use __builtin_align__(8) or whatever they use for the
  // builtin vector types.
  Fp a0, a1;

  static constexpr size_t ELT_LIMBS = Fp::ELT_LIMBS;

  static constexpr int DEGREE = 2;

  __device__ static void load(Fp2 &x, const var *mem) {
    Fp::load(x.a0, mem);
    Fp::load(x.a1, mem + ELT_LIMBS);
  }

  __device__ static void store(var *mem, const Fp2 &x) {
    Fp::store(mem, x.a0);
    Fp::store(mem + ELT_LIMBS, x.a1);
  }

  __device__ static int are_equal(const Fp2 &x, const Fp2 &y) {
    return Fp::are_equal(x.a0, y.a0) && Fp::are_equal(x.a1, y.a1);
  }

  __device__ static void set_zero(Fp2 &x) {
    Fp::set_zero(x.a0);
    Fp::set_zero(x.a1);
  }

  __device__ static int is_zero(const Fp2 &x) {
    return Fp::is_zero(x.a0) && Fp::is_zero(x.a1);
  }

  __device__ static void set_one(Fp2 &x) {
    Fp::set_one(x.a0);
    Fp::set_zero(x.a1);
  }

  __device__ static void add(Fp2 &s, const Fp2 &x, const Fp2 &y) {
    Fp::add(s.a0, x.a0, y.a0);
    Fp::add(s.a1, x.a1, y.a1);
  }

  __device__ static void sub(Fp2 &s, const Fp2 &x, const Fp2 &y) {
    Fp::sub(s.a0, x.a0, y.a0);
    Fp::sub(s.a1, x.a1, y.a1);
  }

  __device__ static void mul(Fp2 &p, const Fp2 &a, const Fp2 &b) {
#ifdef MNT4753
    Fp a0_b0, a1_b1, a0_plus_a1, b0_plus_b1, c, t0, t1;

    Fp::mul(a0_b0, a.a0, b.a0);
    Fp::mul(a1_b1, a.a1, b.a1);

    Fp::add(a0_plus_a1, a.a0, a.a1);
    Fp::add(b0_plus_b1, b.a0, b.a1);
    Fp::mul(c, a0_plus_a1, b0_plus_b1);

    mul_<ALPHA>::x(t0, a1_b1);
    Fp::sub(t1, c, a0_b0);

    Fp::add(p.a0, a0_b0, t0);
    Fp::sub(p.a1, t1, a1_b1);
#endif
#ifdef ALT_BN128
    Fp a0_b0, a1_b1, a0_plus_a1, b0_plus_b1, c, t0, t1;
    Fp xx;

    Fp::mul(a0_b0, a.a0, b.a0);
    Fp::mul(a1_b1, a.a1, b.a1);

    Fp::add(a0_plus_a1, a.a0, a.a1);
    Fp::add(b0_plus_b1, b.a0, b.a1);
    Fp::mul(c, a0_plus_a1, b0_plus_b1);

    //mul_<ALPHA>::x(t0, a1_b1);
    Fp::load_alpha(xx);
    Fp::mul(t0, xx, a1_b1);
    Fp::sub(t1, c, a0_b0);

    Fp::add(p.a0, a0_b0, t0);
    Fp::sub(p.a1, t1, a1_b1);
#endif
#ifdef BLS12_381
    Fp a0_b0, a1_b1, a0_plus_a1, b0_plus_b1, c, t0, t1;
    Fp xx;

    Fp::mul(a0_b0, a.a0, b.a0);
    Fp::mul(a1_b1, a.a1, b.a1);

    Fp::add(a0_plus_a1, a.a0, a.a1);
    Fp::add(b0_plus_b1, b.a0, b.a1);
    Fp::mul(c, a0_plus_a1, b0_plus_b1);

    //mul_<ALPHA>::x(t0, a1_b1);
    Fp::load_alpha(xx);
    Fp::mul(t0, xx, a1_b1);
    Fp::sub(t1, c, a0_b0);

    Fp::add(p.a0, a0_b0, t0);
    Fp::sub(p.a1, t1, a1_b1);
#endif
  }

  __device__ static void sqr(Fp2 &s, const Fp2 &a) {
#ifdef MNT4753
    Fp a0_a1, a0_plus_a1, a0_plus_13_a1, t0, t1, t2;

    Fp::mul(a0_a1, a.a0, a.a1);
    Fp::add(a0_plus_a1, a.a0, a.a1);
    mul_<ALPHA>::x(t0, a.a1);
    Fp::add(a0_plus_13_a1, a.a0, t0);
    Fp::mul(t0, a0_plus_a1, a0_plus_13_a1);
    // TODO: Could do mul_14 to save a sub?
    Fp::sub(t1, t0, a0_a1);
    mul_<ALPHA>::x(t2, a0_a1);
    Fp::sub(s.a0, t1, t2);
    mul_<2>::x(s.a1, a0_a1);
#endif
#ifdef ALT_BN128
    Fp a0_a1, a0_plus_a1, a0_plus_13_a1, t0, t1, t2;
    Fp xx;

    Fp::mul(a0_a1, a.a0, a.a1);
    Fp::add(a0_plus_a1, a.a0, a.a1);
    //mul_<ALPHA>::x(t0, a.a1);
    Fp::load_alpha(xx);
    Fp::mul(t0, xx, a.a1);
    Fp::add(a0_plus_13_a1, a.a0, t0);
    Fp::mul(t0, a0_plus_a1, a0_plus_13_a1);
    // TODO: Could do mul_14 to save a sub?
    Fp::sub(t1, t0, a0_a1);
    //mul_<ALPHA>::x(t2, a0_a1);
    Fp::load_alpha(xx);
    Fp::mul(t2, xx, a0_a1);
    Fp::sub(s.a0, t1, t2);
    mul_<2>::x(s.a1, a0_a1);
#endif
#ifdef BLS12_381
    Fp a0_a1, a0_plus_a1, a0_plus_13_a1, t0, t1, t2;
    Fp xx;

    Fp::mul(a0_a1, a.a0, a.a1);
    Fp::add(a0_plus_a1, a.a0, a.a1);
    //mul_<ALPHA>::x(t0, a.a1);
    Fp::load_alpha(xx);
    Fp::mul(t0, xx, a.a1);
    Fp::add(a0_plus_13_a1, a.a0, t0);
    Fp::mul(t0, a0_plus_a1, a0_plus_13_a1);
    // TODO: Could do mul_14 to save a sub?
    Fp::sub(t1, t0, a0_a1);
    //mul_<ALPHA>::x(t2, a0_a1);
    Fp::load_alpha(xx);
    Fp::mul(t2, xx, a0_a1);
    Fp::sub(s.a0, t1, t2);
    mul_<2>::x(s.a1, a0_a1);
#endif
  }

  __device__ static void from_monty(Fp2 &s, const Fp2 &x) {
    Fp::from_monty(s.a0, x.a0);
    Fp::from_monty(s.a1, x.a1);
  }

  __device__ static void to_monty(Fp2 &s, const Fp2 &x) {
    Fp::to_monty(s.a0, x.a0);
    Fp::to_monty(s.a1, x.a1);
  }
};

template <typename Fp, int ALPHA> struct Fp3_model {
  // typedef Fp PrimeField;
  typedef Fp3_model Fp3;

  // TODO: Use __builtin_align__(8) or whatever they use for the
  // builtin vector types.
  Fp a0, a1, a2;
  
  static constexpr size_t ELT_LIMBS = Fp::ELT_LIMBS;

  static constexpr int DEGREE = 3;

  __device__ static void load(Fp3 &x, const var *mem) {
    Fp::load(x.a0, mem);
    Fp::load(x.a1, mem + ELT_LIMBS);
    Fp::load(x.a2, mem + 2 * ELT_LIMBS);
  }

  __device__ static void store(var *mem, const Fp3 &x) {
    Fp::store(mem, x.a0);
    Fp::store(mem + ELT_LIMBS, x.a1);
    Fp::store(mem + 2 * ELT_LIMBS, x.a2);
  }

  __device__ static int are_equal(const Fp3 &x, const Fp3 &y) {
    return Fp::are_equal(x.a0, y.a0) && Fp::are_equal(x.a1, y.a1) &&
           Fp::are_equal(x.a2, y.a2);
  }

  __device__ static void set_zero(Fp3 &x) {
    Fp::set_zero(x.a0);
    Fp::set_zero(x.a1);
    Fp::set_zero(x.a2);
  }

  __device__ static int is_zero(const Fp3 &x) {
    return Fp::is_zero(x.a0) && Fp::is_zero(x.a1) && Fp::is_zero(x.a2);
  }

  __device__ static void set_one(Fp3 &x) {
    Fp::set_one(x.a0);
    Fp::set_zero(x.a1);
    Fp::set_zero(x.a2);
  }

  __device__ static void add(Fp3 &s, const Fp3 &x, const Fp3 &y) {
    Fp::add(s.a0, x.a0, y.a0);
    Fp::add(s.a1, x.a1, y.a1);
    Fp::add(s.a2, x.a2, y.a2);
  }

  __device__ static void sub(Fp3 &s, const Fp3 &x, const Fp3 &y) {
    Fp::sub(s.a0, x.a0, y.a0);
    Fp::sub(s.a1, x.a1, y.a1);
    Fp::sub(s.a2, x.a2, y.a2);
  }

  __device__ static void mul(Fp3 &p, const Fp3 &a, const Fp3 &b) {
    Fp a0_b0, a1_b1, a2_b2;
    Fp a0_plus_a1, a1_plus_a2, a0_plus_a2, b0_plus_b1, b1_plus_b2, b0_plus_b2;
    Fp t0, t1, t2;

    Fp::mul(a0_b0, a.a0, b.a0);
    Fp::mul(a1_b1, a.a1, b.a1);
    Fp::mul(a2_b2, a.a2, b.a2);

    // TODO: Consider interspersing these additions among the
    // multiplications above.
    Fp::add(a0_plus_a1, a.a0, a.a1);
    Fp::add(a1_plus_a2, a.a1, a.a2);
    Fp::add(a0_plus_a2, a.a0, a.a2);

    Fp::add(b0_plus_b1, b.a0, b.a1);
    Fp::add(b1_plus_b2, b.a1, b.a2);
    Fp::add(b0_plus_b2, b.a0, b.a2);

    Fp::mul(t0, a1_plus_a2, b1_plus_b2);
    Fp::add(t1, a1_b1, a2_b2);
    Fp::sub(t0, t0, t1);
    mul_<ALPHA>::x(t0, t0);
    Fp::add(p.a0, a0_b0, t0);

    Fp::mul(t0, a0_plus_a1, b0_plus_b1);
    Fp::add(t1, a0_b0, a1_b1);
    mul_<ALPHA>::x(t2, a2_b2);
    Fp::sub(t2, t2, t1);
    Fp::add(p.a1, t0, t2);

    Fp::mul(t0, a0_plus_a2, b0_plus_b2);
    Fp::sub(t1, a1_b1, a0_b0);
    Fp::sub(t1, t1, a2_b2);
    Fp::add(p.a2, t0, t1);
  }

  __device__ static void sqr(Fp3 &s, const Fp3 &a) {
    Fp a0a0, a1a1, a2a2;
    Fp a0_plus_a1, a1_plus_a2, a0_plus_a2;
    Fp t0, t1;

    Fp::sqr(a0a0, a.a0);
    Fp::sqr(a1a1, a.a1);
    Fp::sqr(a2a2, a.a2);

    // TODO: Consider interspersing these additions among the
    // squarings above.
    Fp::add(a0_plus_a1, a.a0, a.a1);
    Fp::add(a1_plus_a2, a.a1, a.a2);
    Fp::add(a0_plus_a2, a.a0, a.a2);

    Fp::sqr(t0, a1_plus_a2);
    // TODO: Remove sequential data dependencies (here and elsewhere)
    Fp::sub(t0, t0, a1a1);
    Fp::sub(t0, t0, a2a2);
    mul_<ALPHA>::x(t0, t0);
    Fp::add(s.a0, a0a0, t0);

    Fp::sqr(t0, a0_plus_a1);
    Fp::sub(t0, t0, a0a0);
    Fp::sub(t0, t0, a1a1);
    mul_<ALPHA>::x(t1, a2a2);
    Fp::add(s.a1, t0, t1);

    Fp::sqr(t0, a0_plus_a2);
    Fp::sub(t0, t0, a0a0);
    Fp::add(t0, t0, a1a1);
    Fp::sub(s.a2, t0, a2a2);
  }
};


// template <typename _modulus_info_> struct Fr_model {
//   typedef Fr PrimeField;

//   using modulus_info = _modulus_info_;

//   // static constexpr size_t ELT_LIMBS = _limbs_;

//   var a;

//   static constexpr int DEGREE = 1;

//   __device__ static void load(Fr &x, const var *mem) {
//     int t = fixnum::layout().thread_rank();
//     x.a = (t < ELT_LIMBS_R) ? mem[t] : 0UL;
//   }

//   __device__ static void store(var *mem, const Fr &x) {
//     int t = fixnum::layout().thread_rank();
//     if (t < ELT_LIMBS_R)
//       mem[t] = x.a;
//   }

//   __device__ static void store_a(var *mem, const var a) {
//     int t = fixnum::layout().thread_rank();
//     if (t < ELT_LIMBS_R)
//       mem[t] = a;
//   }

//   __device__ static void load_stride(Fr &x, const var *mem, size_t stride) {
//     int t = fixnum::layout().thread_rank();
//     x.a = (t < ELT_LIMBS_R) ? mem[t*stride] : 0UL;
//   }

//   __device__ static void store_stride(var *mem, const Fr &x, size_t stride) {
//     int t = fixnum::layout().thread_rank();
//     if (t < ELT_LIMBS_R)
//       mem[t*stride] = x.a;
//   }


//   __device__ static void load_alpha(Fr &x) {
// #if (defined BLS12_381) || (defined ALT_BN128)
//     int t = fixnum::layout().thread_rank();
//     x.a = (t < ELT_LIMBS_R) ? ALPHA_VALUE[t] : 0UL;
// #endif
//   }

//   __device__ static int are_equal(const Fr &x, const Fr &y) {
//     return fixnum::cmp(x.a, y.a) == 0;
//   }

//   __device__ static void set_zero(Fr &x) { x.a = fixnum::zero(); }

//   __device__ static int is_zero(const Fr &x) { return fixnum::is_zero(x.a); }

//   __device__ static void set_one(Fr &x) { x.a = modulus_info::monty_one(fixnum::layout().thread_rank()); }

//   __device__ static void add(Fr &zz, const Fr &xx, const Fr &yy) {
//     int br;
//     var x = xx.a, y = yy.a, z, r;
//     var mod = modulus_info::mod(fixnum::layout().thread_rank());
//     fixnum::add(z, x, y);
//     fixnum::sub_br(r, br, z, mod);
//     zz.a = br ? z : r;
//   }

//   __device__ static void neg(Fr &z, const Fr &x) {
//     var mod = modulus_info::mod(fixnum::layout().thread_rank());
//     fixnum::sub(z.a, mod, x.a);
//   }

//   __device__ static void sub(Fr &z, const Fr &x, const Fr &y) {
//     int br;
//     var r, mod = modulus_info::mod(fixnum::layout().thread_rank());
//     fixnum::sub_br(r, br, x.a, y.a);
//     if (br)
//       fixnum::add(r, r, mod);
//     z.a = r;
//   }

//   __device__ __noinline__ static void mul(Fr &zz, const Fr &xx, const Fr &yy) {
//     auto grp = fixnum::layout();
//     int L = grp.thread_rank();
//     var mod = modulus_info::mod(fixnum::layout().thread_rank());

//     var x = xx.a, y = yy.a, z = digit::zero();
//     var tmp;
//     digit::mul_lo(tmp, x, modulus_info::ninv_mod);
//     digit::mul_lo(tmp, tmp, grp.shfl(y, 0));
//     int cy = 0;

//     for (int i = 0; i < ELT_LIMBS_R; ++i) {
//       var u;
//       var xi = grp.shfl(x, i);
//       var z0 = grp.shfl(z, 0);
//       var tmpi = grp.shfl(tmp, i);

//       digit::mad_lo(u, z0, modulus_info::ninv_mod, tmpi);
//       digit::mad_lo_cy(z, cy, mod, u, z);
//       digit::mad_lo_cy(z, cy, y, xi, z);

//       assert(L || z == 0);     // z[0] must be 0
//       z = grp.shfl_down(z, 1); // Shift right one word
//       z = (L >= ELT_LIMBS_R - 1) ? 0 : z;

//       digit::add_cy(z, cy, z, cy);
//       digit::mad_hi_cy(z, cy, mod, u, z);
//       digit::mad_hi_cy(z, cy, y, xi, z);
//     }
//     // Resolve carries
//     int msb = grp.shfl(cy, ELT_LIMBS_R - 1);
//     cy = grp.shfl_up(cy, 1); // left shift by 1
//     cy = (L == 0) ? 0 : cy;

//     fixnum::add_cy(z, cy, z, cy);
//     msb += cy;
//     assert(msb == !!msb); // msb = 0 or 1.

//     // br = 0 ==> z >= mod
//     var r;
//     int br;
//     fixnum::sub_br(r, br, z, mod);
//     if (msb || br == 0) {
//       // If the msb was set, then we must have had to borrow.
//       assert(!msb || msb == br);
//       z = r;
//     }
//     zz.a = z;
//   }

//   __device__ static void sqr(Fr &z, const Fr &x) {
//     // TODO: Find a faster way to do this. Actually only option
//     // might be full squaring with REDC.
//     mul(z, x, x);
//   }

// #if 0
//     __device__
//     static void
//     inv(Fr &z, const Fr &x) {
//         // FIXME: Implement!  See HEHCC Algorithm 11.12.
//         z = x;
//     }
// #endif

//   __device__ static void from_monty(Fr &y, const Fr &x) {
//     Fr one;
//     one.a = fixnum::one();
//     mul(y, x, one);
//   }
// };


#ifdef MNT4753
// typedef Fp_model<MOD_R_> Fr_type_;
typedef Fp_model<MOD_R_, ELT_LIMBS_R, BIG_WIDTH_R> Fr_type;
typedef Fp_model<MOD_Q_, ELT_LIMBS_Q, BIG_WIDTH_Q> Fq_type;
typedef Fp2_model<Fq_type, 13> Fq2_type;

// typedef Fp_model<MOD_Q_> Fr_MNT6;
// typedef Fp_model<MOD_R_> Fp_MNT6;
// typedef Fp3_model<Fp_MNT6, 11> Fp3_MNT6;
#endif

#ifdef ALT_BN128
// typedef Fp_model<MOD_R_> Fr_type_;
typedef Fp_model<MOD_R_, ELT_LIMBS_R, BIG_WIDTH_R> Fr_type;
typedef Fp_model<MOD_Q_, ELT_LIMBS_Q, BIG_WIDTH_Q> Fq_type;
typedef Fp2_model<Fq_type, 13> Fq2_type;
#endif

#ifdef BLS12_381
// typedef Fr<MOD_R_> Fr_type_;
typedef Fp_model<MOD_R_, ELT_LIMBS_R, BIG_WIDTH_R> Fr_type;
typedef Fp_model<MOD_Q_, ELT_LIMBS_Q, BIG_WIDTH_Q> Fq_type;
typedef Fp2_model<Fq_type, 2> Fq2_type;
#endif