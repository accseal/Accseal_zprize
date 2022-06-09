#pragma once

#include "fixnum.cuh"

// const NONRESIDUE: Fq
__device__ __constant__ const var ALPHA_VALUE[BIG_WIDTH_Q] = {
  0xfc0b8000000002fa, 0x97d39cf6e000018b, 0x2072420fbfa05044, 0xcbbcbd50d97c3802,
  0xbaf1ec35813f9eb, 0x9974a2c0945ad2, 0x0, 0x0
};

// const MODULUS: BigInteger
__device__ __constant__ const var MOD_Q[BIG_WIDTH_Q] = {
  0x8508c00000000001, 0x170b5d4430000000, 0x1ef3622fba094800, 0x1a22d9f300f5138f,
  0xc63b05c06ca1493b, 0x1ae3a4617c510ea, 0ULL, 0ULL
};

static constexpr var Q_NINV_MOD = 9586122913090633727;

// const R: BigInteger
__device__ __constant__ const var X_MOD_Q[BIG_WIDTH_Q] = {
  202099033278250856, 5854854902718660529, 11492539364873682930, 8885205928937022213,
  5545221690922665192, 39800542322357402, 0ULL, 0ULL
};

// const R2: BigInteger
__device__ __constant__ const var X_MOD_Q_SQUARE[BIG_WIDTH_Q] = {
  0xb786686c9400cd22, 0x329fcaab00431b1, 0x22a5f11162d6b46d, 0xbfdf7d03827dc3ac,
  0x837e92f041790bf9, 0x6dfccb1e914b88, 0ULL, 0ULL
};

// const MODULUS: BigInteger
__device__ __constant__ const var MOD_R[] = {
  725501752471715841,
  6461107452199829505,
  6968279316240510977,
  1345280370688173398,
  0, 0, 0, 0
};
// -R^{-1} (mod 2^64)
const var R_NINV_MOD = 725501752471715839;

// R: BigInteger
__device__ __constant__ const var X_MOD_R[] = {
  9015221291577245683,
  8239323489949974514,
  1646089257421115374,
  958099254763297437,
  0, 0, 0, 0
};

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
  template <int BIG_WIDTH_ = BIG_WIDTH_Q>
  __device__ __forceinline__ static int lane() {
    return fixnum::layout<BIG_WIDTH_>().thread_rank();
  }

  template <int BIG_WIDTH_ = BIG_WIDTH_Q>
  __device__ __forceinline__ static var mod(int lane) { return MOD_R[lane]; }

  static constexpr var ninv_mod = R_NINV_MOD;

  template <int BIG_WIDTH_ = BIG_WIDTH_Q>
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
    int t = fixnum::layout<BIG_WIDTH_>().thread_rank();
    x.a = (t < ELT_LIMBS_) ? ALPHA_VALUE[t] : 0UL;
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
  }

  __device__ static void sqr(Fp2 &s, const Fp2 &a) {
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


template <typename _modulus_info_, int ELT_LIMBS_, int BIG_WIDTH_> struct Fr_model {
  // typedef Fr PrimeField;

  // using modulus_info = _modulus_info_;

  // // static constexpr size_t ELT_LIMBS = _limbs_;

  // var a;

  // static constexpr int DEGREE = 1;
  typedef Fr_model Fr;

  using modulus_info = _modulus_info_;

  var a;

  static constexpr int DEGREE = 1;
  static constexpr int ELT_LIMBS = ELT_LIMBS_;
  static constexpr int BIG_WIDTH = BIG_WIDTH_;

  __device__ static void load(Fr &x, const var *mem) {
    int t = fixnum::layout().thread_rank();
    x.a = (t < ELT_LIMBS) ? mem[t] : 0UL;
  }

  __device__ static void store(var *mem, const Fr &x) {
    int t = fixnum::layout().thread_rank();
    if (t < ELT_LIMBS)
      mem[t] = x.a;
  }

  __device__ static void store_a(var *mem, const var a) {
    int t = fixnum::layout().thread_rank();
    if (t < ELT_LIMBS)
      mem[t] = a;
  }

  __device__ static void load_stride(Fr &x, const var *mem, size_t stride) {
    int t = fixnum::layout().thread_rank();
    x.a = (t < ELT_LIMBS) ? mem[t*stride] : 0UL;
  }

  __device__ static void store_stride(var *mem, const Fr &x, size_t stride) {
    int t = fixnum::layout().thread_rank();
    if (t < ELT_LIMBS)
      mem[t*stride] = x.a;
  }


  __device__ static void load_alpha(Fr &x) {
#if (defined BLS12_381) || (defined ALT_BN128)
    int t = fixnum::layout().thread_rank();
    x.a = (t < ELT_LIMBS) ? ALPHA_VALUE[t] : 0UL;
#endif
  }

  __device__ static int are_equal(const Fr &x, const Fr &y) {
    return fixnum::cmp(x.a, y.a) == 0;
  }

  __device__ static void set_zero(Fr &x) { x.a = fixnum::zero(); }

  __device__ static int is_zero(const Fr &x) { return fixnum::is_zero(x.a); }

  __device__ static void set_one(Fr &x) { x.a = modulus_info::monty_one(fixnum::layout().thread_rank()); }

  __device__ static void add(Fr &zz, const Fr &xx, const Fr &yy) {
    int br;
    var x = xx.a, y = yy.a, z, r;
    var mod = modulus_info::mod(fixnum::layout().thread_rank());
    fixnum::add(z, x, y);
    fixnum::sub_br(r, br, z, mod);
    zz.a = br ? z : r;
  }

  __device__ static void neg(Fr &z, const Fr &x) {
    var mod = modulus_info::mod(fixnum::layout().thread_rank());
    fixnum::sub(z.a, mod, x.a);
  }

  __device__ static void sub(Fr &z, const Fr &x, const Fr &y) {
    int br;
    var r, mod = modulus_info::mod(fixnum::layout().thread_rank());
    fixnum::sub_br(r, br, x.a, y.a);
    if (br)
      fixnum::add(r, r, mod);
    z.a = r;
  }

  __device__ __noinline__ static void mul(Fr &zz, const Fr &xx, const Fr &yy) {
    auto grp = fixnum::layout();
    int L = grp.thread_rank();
    var mod = modulus_info::mod(fixnum::layout().thread_rank());

    var x = xx.a, y = yy.a, z = digit::zero();
    var tmp;
    digit::mul_lo(tmp, x, modulus_info::ninv_mod);
    digit::mul_lo(tmp, tmp, grp.shfl(y, 0));
    int cy = 0;

    for (int i = 0; i < ELT_LIMBS; ++i) {
      var u;
      var xi = grp.shfl(x, i);
      var z0 = grp.shfl(z, 0);
      var tmpi = grp.shfl(tmp, i);

      digit::mad_lo(u, z0, modulus_info::ninv_mod, tmpi);
      digit::mad_lo_cy(z, cy, mod, u, z);
      digit::mad_lo_cy(z, cy, y, xi, z);

      assert(L || z == 0);     // z[0] must be 0
      z = grp.shfl_down(z, 1); // Shift right one word
      z = (L >= ELT_LIMBS - 1) ? 0 : z;

      digit::add_cy(z, cy, z, cy);
      digit::mad_hi_cy(z, cy, mod, u, z);
      digit::mad_hi_cy(z, cy, y, xi, z);
    }
    // Resolve carries
    int msb = grp.shfl(cy, ELT_LIMBS - 1);
    cy = grp.shfl_up(cy, 1); // left shift by 1
    cy = (L == 0) ? 0 : cy;

    fixnum::add_cy(z, cy, z, cy);
    msb += cy;
    assert(msb == !!msb); // msb = 0 or 1.

    // br = 0 ==> z >= mod
    var r;
    int br;
    fixnum::sub_br(r, br, z, mod);
    if (msb || br == 0) {
      // If the msb was set, then we must have had to borrow.
      assert(!msb || msb == br);
      z = r;
    }
    zz.a = z;
  }

  __device__ static void sqr(Fr &z, const Fr &x) {
    // TODO: Find a faster way to do this. Actually only option
    // might be full squaring with REDC.
    mul(z, x, x);
  }

#if 0
    __device__
    static void
    inv(Fr &z, const Fr &x) {
        // FIXME: Implement!  See HEHCC Algorithm 11.12.
        z = x;
    }
#endif

  __device__ static void from_monty(Fr &y, const Fr &x) {
    Fr one;
    one.a = fixnum::one();
    mul(y, x, one);
  }
};

typedef Fp_model<MOD_R_, ELT_LIMBS_R, BIG_WIDTH_R> Fr_type;
typedef Fp_model<MOD_Q_, ELT_LIMBS_Q, BIG_WIDTH_Q> Fq_type;
typedef Fp2_model<Fq_type, 2> Fq2_type;
