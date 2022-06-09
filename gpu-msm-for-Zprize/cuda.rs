
#[derive(Debug, Clone)]
#[repr(C)]
pub struct FFITraitObject {
    ptr: usize,
}

pub struct cuda_G1Projective {
    x: usize,
    y: usize,
    z: usize,
}

#[link(name="rust_gzkp_msm_lib", kind="static")]
extern {
    pub fn multiexp_cuda_c(
        scalars: *const FFITraitObject, 
        bases: *const FFITraitObject, 
        length: usize) ->
        * mut cuda_G1Projective
}