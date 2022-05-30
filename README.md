# Accseal_zprize
This project is used for Zprize

# GZKP-demo
Zero-knowledge proof is a cryptographic protocol with computational integrity and privacy. It allows one party to convince the other that a certain assertion is correct without providing any useful information. Zero-knowledge proof is widely used in privacy-related applications, such as digital currency, verifiable machine learning, verifiable database outsourcing, etc. The computation process is mainly composed of three modules: parameter generator, prover, and verifier.

The computational bottleneck of the zero-knowledge proof system is the prover module. Slow proof generation speed significantly limits the system throughput. We develop the GZKP-demo, A GPU system that supports various ZKP applications. GZKP-demo supports multiple elliptic curves, such as MNT4753, BLS12_381, and ALT_BN128.

Copyright (C) 2022,  http://www.accseal.com.


## Dependency
 - gpu-groth16-prover-3x is based on commit [ec048](https://github.com/CodaProtocol/gpu-groth16-prover-3x/tree/ec0480380b897deaff77c49d9696115c2a2fd80c)

 - install dependency library:
```
sudo apt-get install -y build-essential \
    cmake \
    git \
    libomp-dev \
    libgmp3-dev \
    libprocps-dev \
    python-markdown \
    libboost-all-dev \
    libssl-dev \
    pkg-config \
    nvidia-cuda-toolkit
```

## Requirement
 - [NVIDIA CUDA](https://developer.nvidia.com/cuda-downloads) 10.2 or above

## Build
```
./build.sh
```

## Genearte test data
```
./build/Release/gpu-groth16-prover-3x/libsnark/generate_parameters $scale $params_path $input_path
```

## Benckmark
```
./build/Release/test/test_msm $curve $params_path $input_path $window_size $reduce_size # for msm
```

## Support or Contact
GZKP is developed by Accseal.

If you have any questions, please contact Maurice Mao(xingzhong.mao@accseal.com). We welcome you to commit your modification to support our project.