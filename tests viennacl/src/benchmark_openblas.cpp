
/* =========================================================================
   Copyright (c) 2010-2015, Institute for Microelectronics,
   Institute for Analysis and Scientific Computing,
   TU Wien.
   Portions of this software are copyright by UChicago Argonne, LLC.

   -----------------
   ViennaCL - The Vienna Computing Library
   -----------------

   Project Head:    Karl Rupp                   rupp@iue.tuwien.ac.at

   (A list of authors and contributors can be found in the PDF manual)

   License:         MIT (X11), see file LICENSE in the base directory
   ============================================================================= */

/* compile with g++ ../benchmark_1_openblas.cpp -o bench_1_openblas -Wall -pedantic -O3 -fopenmp -std=c++11 -I /usr/local/openblas/include/ -L/usr/local/openblas/lib -lopenblas -lpthread
 */

#include "viennacl/matrix.hpp"
#include "viennacl/matrix_proxy.hpp"
#include "viennacl/vector.hpp"
#include "viennacl/vector_proxy.hpp"

#include "viennacl/linalg/inner_prod.hpp"
#include "viennacl/linalg/prod.hpp"
#include "viennacl/linalg/lu.hpp"
#include "viennacl/tools/timer.hpp"

#include <iomanip>
#include <stdlib.h>
#include <stdio.h>
#include <cblas.h>

/* matrix transpose for float matrices in cloumn major storage order */ 
void matrix_transpose(double *A, double *B, int m, int n)
{
  for(int j = 0; j < n; ++j)
    for(int i = 0; i < m; ++i)
    {
      B[j*m + i] = A[i*m + j];
    }
}

void init_random(double *A, int m, int n)
{
  for (int j = 0; j < n; ++j)
    for (int i = 0; i < m; ++i)
      A[j*m + i] = ((double)rand()) / ((double)RAND_MAX);
}

void bench(size_t BLAS3_N)
{
  using viennacl::linalg::inner_prod;
  using viennacl::linalg::prod;
  using viennacl::linalg::lu_factorize;
  using viennacl::trans;

  viennacl::tools::timer timer;
  double time_previous, time_spent;
  size_t Nruns;
  double time_per_benchmark = 1;
  int i = 0;

#define BENCHMARK_OP(OPERATION, PERF)           \
  OPERATION;                                    \
  timer.start();                                \
  Nruns = 0;                                    \
  time_spent = 0;                               \
  i = 0;                                        \
  while (i < 5)                                 \
  {                                             \
    if(time_spent >= time_per_benchmark)        \
      break;                                    \
    time_previous = timer.get();                \
    OPERATION;                                  \
    time_spent += timer.get() - time_previous;  \
    Nruns+=1;                                   \
    i++;                                        \
  }                                             \
  time_spent/=(double)Nruns;                    \
  std::cout << PERF << " ";

  //BLAS3
  {
    /* doubleoperations */
    double*A = (double*)malloc(sizeof(double)*BLAS3_N*BLAS3_N);
    double*B = (double*)malloc(sizeof(double)*BLAS3_N*BLAS3_N);
    double*C = (double*)malloc(sizeof(double)*BLAS3_N*BLAS3_N);
  
    init_random(A, BLAS3_N, BLAS3_N);
    init_random(B, BLAS3_N, BLAS3_N);
  
    double*AT = (double*)malloc(sizeof(double)*BLAS3_N*BLAS3_N);
    matrix_transpose(A,AT,BLAS3_N,BLAS3_N);
    double*BT = (double*)malloc(sizeof(double)*BLAS3_N*BLAS3_N);
    matrix_transpose(B,BT,BLAS3_N,BLAS3_N);
  
  

    BENCHMARK_OP(
      cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, BLAS3_N, BLAS3_N, BLAS3_N, 1.0, A, BLAS3_N, B, BLAS3_N, 0.0, C, BLAS3_N),
      double(2*BLAS3_N*BLAS3_N*BLAS3_N)/time_spent*1e-9
      );
    BENCHMARK_OP(
      cblas_dgemm(CblasColMajor, CblasNoTrans, CblasTrans, BLAS3_N, BLAS3_N, BLAS3_N, 1.0, A, BLAS3_N, B, BLAS3_N, 0.0, C, BLAS3_N),
      double(2*BLAS3_N*BLAS3_N*BLAS3_N)/time_spent*1e-9
      );
    BENCHMARK_OP(
      cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans, BLAS3_N, BLAS3_N, BLAS3_N, 1.0, A, BLAS3_N, B, BLAS3_N, 0.0, C, BLAS3_N),
      double(2*BLAS3_N*BLAS3_N*BLAS3_N)/time_spent*1e-9
      );
    BENCHMARK_OP(
      cblas_dgemm(CblasColMajor, CblasTrans, CblasTrans, BLAS3_N, BLAS3_N, BLAS3_N, 1.0, A, BLAS3_N, B, BLAS3_N, 0.0, C, BLAS3_N),
      double(2*BLAS3_N*BLAS3_N*BLAS3_N)/time_spent*1e-9
      );
    
    free(A);
    free(B);
    free(C);
    free(BT);
    free(AT);
  }


}


int main(int argc, char *argv[])
{
  if(argc != 2)
    std::cout << "usage: bench_blas3 matrix-size" << std::endl;
 
  std::size_t BLAS3_N = std::stoi(argv[1]);

  bench(BLAS3_N);
  std::cerr << "size " << BLAS3_N << ": OpenBLAS done!" << std::endl;
}


