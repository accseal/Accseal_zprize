use cmake;

fn main() {
    let dst = cmake::Config::new("gzkp-demo-msm").build();

    println!("cargo:rustc-link-search=native={}", dst.display());
    println!("cargo:rustc-link-lib=static=rust_gzkp_msm_lib");
    
    println!("cargo:rustc-link-lib=dylib=stdc++"); // link to stdc++ lib
    println!("cargo:rustc-link-lib=dylib=procps");
    println!("cargo:rustc-link-lib=dylib=gmp");
    
    let lib_path = env!("LD_LIBRARY_PATH");
    let cuda_lib_path: Vec<_> = lib_path.split(':').into_iter().filter(|path| path.contains("/cuda")).collect(); // 查找 cuda 库

    if cuda_lib_path.is_empty() {
        panic!("Ensure cuda installed on your environment");
    } else {
        println!("cargo:rustc-link-search=native={}", cuda_lib_path[0]);
        println!("cargo:rustc-link-lib=cudart"); // cuda run-time lib
    }
}