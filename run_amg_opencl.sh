#!/bin/bash

# OpenCL builds:

g++ ../examples/tutorial/amg.cpp -I.. -DVIENNACL_WITH_OPENMP -fopenmp -O3 -DNDEBUG -DVIENNACL_WITH_OPENCL -lOpenCL -o amg_opencl

./amg_opencl matrices/poisson2d_3969.mtx   > log/opencl-3969.txt
./amg_opencl matrices/poisson2d_16129.mtx  > log/opencl-16129.txt
./amg_opencl matrices/poisson2d_65025.mtx  > log/opencl-65025.txt
./amg_opencl matrices/poisson2d_261121.mtx > log/opencl-261121.txt
./amg_opencl matrices/poisson2d_1046529.mtx > log/opencl-1046529.txt


