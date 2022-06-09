
#[derive(Debug, Clone)]
#[repr(C)]
pub struct FFITraitObject {
    ptr: usize,
}

pub struct Result {
    x: usize,
    y: usize,
    z: usize,
}



pub fn multi_scalar_mul<G1: AffineCurve>(
    bases: &[G1],
    scalars: &[<G1::ScalarField as PrimeField>::BigInteger],
) -> G1::Projective {
    println!("MSM scale len: {}", scalars.len());
    println!("MSM bases len: {}", bases.len());

    unsafe{
        let mut coeffs_ori = vec![];
        let mut bases_ori = vec![];
        let mut out_A = vec![0u8; 6 * 8 * 3];

        // push ptr of scalars and bases into vec
        coeffs_ori.push(mem::transmute(&scalars[0]));
        bases_ori.push(mem::transmute(&bases[0]));

        let mut output = multiexp_cuda_c(coeffs_ori.as_ptr(), bases_ori.as_ptr(), scalars.len());
        
        // Cast to the G1Projective object defined by the test interface
        let mut output = unsafe {*(output as *const Result as * const G1::Projective)};

        output
    }
}

#[link(name="rust_gzkp_msm_lib", kind="static")]
extern {
    pub fn multiexp_cuda_c(
        scalars: *const FFITraitObject, 
        bases: *const FFITraitObject, 
        length: usize) ->
        * mut Result
}