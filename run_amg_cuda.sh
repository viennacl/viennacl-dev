#!/bin/bash

# CUDA builds:

nvcc ../examples/tutorial/amg.cu -I.. -O3 -DNDEBUG -DVIENNACL_WITH_CUDA -arch=sm_35 -o amg_cuda

./amg_cuda matrices/poisson2d_3969.mtx   > log/cuda-3969.txt
./amg_cuda matrices/poisson2d_16129.mtx  > log/cuda-16129.txt
./amg_cuda matrices/poisson2d_65025.mtx  > log/cuda-65025.txt
./amg_cuda matrices/poisson2d_261121.mtx > log/cuda-261121.txt
./amg_cuda matrices/poisson2d_1046529.mtx > log/cuda-1046529.txt
./amg_cuda matrices/poisson2d_4190209.mtx > log/cuda-4190209.txt

