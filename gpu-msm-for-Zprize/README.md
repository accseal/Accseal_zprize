# GZKP-demo-msm
This is a GPU-accelerated MSM library that supports BLS12-377 curve.

# Dependency
This library relies on the followings:
- C++ build environment
- CMake build infrastructure (version 3.14)
- Nvidia CUDA Toolkit (version 11.4)
- rust library
    - libc = "0.2"
    - cuda-driver-sys = "0.3"
    - cmake = "0.1"
    - dunce = "1.0.0"

# How to use
Please refer to the functions defined in the "example-rust.rs" file to use the ***gzkp_msm_lib*** library

# Build
Put the "build.rs" file and the files in the *gzkp-demo-msm* directory into the test harness.

