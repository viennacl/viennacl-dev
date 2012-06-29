#!/bin/bash

g++ generate-blas3-solve-align1.cpp -o generate-blas3-solve-align1
g++ generate-blas3-prod-align1.cpp -o generate-blas3-prod-align1

./generate-blas3.sh

g++ converter.cpp -o converter -lboost_filesystem-mt
./converter
