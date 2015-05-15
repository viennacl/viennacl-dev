#!/bin/bash

# CUDA builds:

nvcc ../examples/tutorial/amg.cu -I.. -O3 -DNDEBUG -DVIENNACL_WITH_CUDA -arch=sm_35 -o amg_cuda

./amg_cuda matrices/poisson2d_3969.mtx   > log/cuda-2d-3969.txt
./amg_cuda matrices/poisson2d_16129.mtx  > log/cuda-2d-16129.txt
./amg_cuda matrices/poisson2d_65025.mtx  > log/cuda-2d-65025.txt
./amg_cuda matrices/poisson2d_261121.mtx > log/cuda-2d-261121.txt
./amg_cuda matrices/poisson2d_1046529.mtx > log/cuda-2d-1046529.txt
./amg_cuda matrices/poisson2d_4190209.mtx > log/cuda-2d-4190209.txt

./amg_cuda matrices/poisson3d_3825.mtx   > log/cuda-3d-3825.txt
./amg_cuda matrices/poisson3d_31713.mtx  > log/cuda-3d-31713.txt
./amg_cuda matrices/poisson3d_257985.mtx > log/cuda-3d-257985.txt

