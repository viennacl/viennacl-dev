#!/bin/bash

# OpenMP builds:
g++ ../examples/tutorial/amg.cpp -I.. -DVIENNACL_WITH_OPENMP -fopenmp -O3 -DNDEBUG -o amg_openmp


./amg_openmp matrices/poisson2d_3969.mtx   > log/openmp-3969.txt
./amg_openmp matrices/poisson2d_16129.mtx  > log/openmp-16129.txt 
./amg_openmp matrices/poisson2d_65025.mtx  > log/openmp-65025.txt
./amg_openmp matrices/poisson2d_261121.mtx > log/openmp-261121.txt
./amg_openmp matrices/poisson2d_1046529.mtx > log/openmp-1046529.txt 



