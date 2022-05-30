#!/bin/bash -x
set -o errexit

cd "$(dirname "${BASH_SOURCE[0]}")"

{
# if [[ ! -d build/Debug ]]; then
#   cmake -B build/Debug -DCURVE=MNT4 -DCMAKE_BUILD_TYPE=Debug -DCMAKE_EXPORT_COMPILE_COMMANDS=YES -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ -DCMAKE_CUDA_COMPILER=nvcc -G Ninja || (rm -rf build/Debug && exit 1)
# fi
# if ! cmake --build build/Debug; then
#   exit 1
# fi

# if [[ ! -d build/NVCCDebug ]]; then
#   cmake -B build/NVCCDebug -DCURVE=MNT4 -DCMAKE_BUILD_TYPE=Debug -DCMAKE_C_COMPILER=gcc -DCMAKE_CXX_COMPILER=g++ -DCMAKE_CUDA_COMPILER=nvcc -G Ninja || (rm -rf build/NVCCDebug && exit 1)
# fi
# cmake --build build/NVCCDebug

if [[ ! -d build/Release ]]; then
  cmake -B build/Release -DCMAKE_BUILD_TYPE=Release -DCMAKE_EXPORT_COMPILE_COMMANDS=NO -DCMAKE_CUDA_COMPILER=nvcc -G Ninja || (rm -rf build/Release && exit 1)
  # cmake -B build/Release -DCMAKE_C_COMPILER:FILEPATH=/usr/local/gcc-7.3.0/bin/gcc -DCMAKE_CXX_COMPILER:FILEPATH=/usr/local/gcc-7.3.0/bin/g++ -DCURVE=MNT4 -DCMAKE_BUILD_TYPE=Release -DCMAKE_EXPORT_COMPILE_COMMANDS=NO -DCMAKE_CUDA_COMPILER=nvcc -G Ninja || (rm -rf build/Release && exit 1)
  # cmake -DMULTICORE=ON -B build/Release -DCMAKE_C_COMPILER:FILEPATH=/usr/local/gcc-7.3.0/bin/gcc -DCMAKE_CXX_COMPILER:FILEPATH=/usr/local/gcc-7.3.0/bin/g++ -DCURVE=MNT4 -DCMAKE_BUILD_TYPE=Release -DCMAKE_EXPORT_COMPILE_COMMANDS=NO -DCMAKE_CUDA_COMPILER=nvcc -G Ninja || (rm -rf build/Release && exit 1)
fi
cmake --build build/Release

printf "\nCOMPILED\n\n"
} >&2

if [[ ! -d output ]]; then
    mkdir output
fi

if [[ "$2" == "run" ]]; then
  mkdir -p run
  n=$4
  parameters_file="$5/run/$1/parameters.$n.dat"
  preprocessed_file="$5/run/$1/preprocessed.$n.dat"
  preprocessed_file_all="$5/run/$1/preprocessed.$n.dat.all"
  preprocessed_file_new="$5/run/$1/preprocessed.$n.C.$6.dat.new" # $6 参数指的是窗口的大小
  inputs_file="$5/run/$1/inputs.$n.dat"
  app_parameters_file="$4/test_parameter.out"
  app_inputs_file="$4/test_input.out"
  if [[ "$3" == "zcash" ]]; then
    if [[ $4 == 21 ]]; then
      inputs_file_for_zcash="$4/zcash/753/createjoinsplit_1_bin_753.dat"
    elif [[ $4 == 17 ]]; then
      inputs_file_for_zcash="$4/zcash/753/createsaplingspend_1_bin_753.dat"
    elif [[ $4 == 13 ]]; then
      inputs_file_for_zcash="$4/zcash/753/createsaplingoutput_1_bin_753.dat"
    else
      echo "n should be 21, 17 or 13"
      exit 1
    fi
  fi
  # inputs_file_for_zcash="$4/zcash/createjoinsplit_1_bin_753.dat"
  # if [[ ! -f "$parameters_file" ]]; then
  #   (./build/Release/gpu-groth16-prover-3x/libsnark/generate_parameters $n $parameters_file $inputs_file && ./build/Release/gpu-groth16-prover-3x/libsnark/main MNT4753 preprocess $parameters_file $preprocessed_file_all 5) || (rm -rf $parameters_file $inputs_file $preprocessed_file_all && exit 1)
  # fi

  case "$3" in
    test_ntt_cpu)
      exec ./build/Release/thirdparty/gpu-groth16-prover-3x/test_ntt_ref $parameters_file $inputs_file
    ;;
    test_ntt_gpu)
      exec ./build/Release/test/test_ntt $parameters_file $inputs_file && rm -rf core.* build
    ;;
    test_H)
      exec ./build/Release/test/test_H $parameters_file $inputs_file && rm -rf core.* build
    ;;
    # cuda_prover)
    #   exec ./build/Release/test/cuda_prover MNT4753 compute $parameters_file $inputs_file output/evaluate_$3_cuda.dat $preprocessed_file_all
    # ;;
    cuda_prover)
      exec ./build/Release/test/cuda_prover_general $1 compute $parameters_file $inputs_file output/evaluate_$3_cuda.dat $preprocessed_file_all
    ;;
    app)
      exec ./build/Release/test/cuda_prover_general $1 app $app_parameters_file $app_inputs_file output/evaluate_$3_app.dat
    ;;
    cuda_prover_by_preprocess)
      exec ./build/Release/test/cuda_prover MNT4753 compute $parameters_file $inputs_file output/evaluate_$3_cuda_preprocess.dat $preprocessed_file_new
    ;;
    zcash)
      exec ./build/Release/test/cuda_prover MNT4753 zcash $parameters_file $inputs_file_for_zcash $preprocessed_file_all
    ;;
    application)
      exec ./build/Release/test/cuda_prover MNT4753 application $parameters_file_for_application $inputs_file_for_application $preprocessed_file_for_application
    ;;
    3x)
      exec ./build/Release/gpu-groth16-prover-3x/cuda_prover_piecewise MNT4753 compute $parameters_file $inputs_file output/evaluate_$3_3x.dat $preprocessed_file
    ;;
    3x-application)
      exec ./build/Release/gpu-groth16-prover-3x/cuda_prover_piecewise MNT4753 application $parameters_file_for_application $inputs_file_for_application
    ;;
    straus)
      exec ./build/Release/gpu-groth16-prover-3x/cuda_prover_piecewise_straus MNT4753 compute $parameters_file $inputs_file output/evaluate_$3_straus.dat $preprocessed_file_all
    ;;
    main)
      exec ./build/Release/gpu-groth16-prover-3x/libsnark/main MNT4753 compute $parameters_file $inputs_file output/evaluate_$3_cpu.dat
    ;;
    test_monty)
      exec ./build/Release/test/test_monty
    ;;
    preprocess_for_pippenger)
    # $5 参数指的是窗口的大小
      exec ./build/Release/gpu-groth16-prover-3x/libsnark/main MNT4753 preprocess_for_pippenger $parameters_file $preprocessed_file_new $5
    ;;
  esac
elif [[ "$1" == "msm" ]]; then
  case "$2" in
    test_multiexp)
      exec ./build/Release/src/multiexp/test_multiexp
    ;;
    bench_multiexp)
      exec ./build/Release/src/multiexp/bench_multiexp
    ;;
  esac
fi
