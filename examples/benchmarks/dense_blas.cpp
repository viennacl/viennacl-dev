/* =========================================================================
   Copyright (c) 2010-2014, Institute for Microelectronics,
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

#include "benchmark-utils.hpp"

#include "viennacl/matrix.hpp"
#include "viennacl/matrix_proxy.hpp"
#include "viennacl/vector.hpp"
#include "viennacl/vector_proxy.hpp"

#include "viennacl/linalg/inner_prod.hpp"
#include "viennacl/linalg/prod.hpp"
#include "viennacl/linalg/lu.hpp"

#include <iomanip>
#include <stdlib.h>

template<class T, class F>
void init_random(viennacl::matrix<T, F> & M)
{
  std::vector<T> cM(M.internal_size());
  for (unsigned int i = 0; i < M.size1(); ++i)
    for (unsigned int j = 0; j < M.size2(); ++j)
      cM[F::mem_index(i, j, M.internal_size1(), M.internal_size2())] = (T)(rand())/RAND_MAX;
  viennacl::fast_copy(&cM[0],&cM[0] + cM.size(),M);
}

template<class T>
void init_random(viennacl::vector<T> & x)
{
  std::vector<T> cx(x.internal_size());
  for (unsigned int i = 0; i < cx.size(); ++i)
    cx[i] = (T)(rand())/RAND_MAX;
  viennacl::fast_copy(&cx[0], &cx[0] + cx.size(), x.begin());
}

template<class T>
void bench(size_t M, size_t N, size_t K, size_t vecN, std::string const & prefix)
{
  using viennacl::linalg::inner_prod;
  using viennacl::linalg::prod;
  using viennacl::linalg::lu_factorize;
  using viennacl::trans;

  Timer timer;
  double time_previous, time_spent;
  size_t Nruns;
  double time_per_benchmark = 1e-1;

#define BENCHMARK_OP(OPERATION, NAME, PERF, INDEX) \
  OPERATION; \
  viennacl::backend::finish();\
  timer.start(); \
  Nruns = 0; \
  time_spent = 0; \
  while (time_spent < time_per_benchmark) \
  { \
    time_previous = timer.get(); \
    OPERATION; \
    viennacl::backend::finish(); \
    time_spent += timer.get() - time_previous; \
    Nruns+=1; \
  } \
  time_spent/=(double)Nruns; \
  std::cout << prefix << NAME " : " << PERF << " " INDEX << std::endl; \

  //BLAS1
  {
    viennacl::scalar<T> s(0);
    T alpha = (T)2.4;
    viennacl::vector<T> x(vecN);
    viennacl::vector<T> y(vecN);
    viennacl::vector<T> z(vecN);

    init_random(x);
    init_random(y);
    init_random(z);

    BENCHMARK_OP(x = y,                "COPY", std::setprecision(3) << 2*vecN*sizeof(T)/time_spent * 1e-9, "GB/s")
    BENCHMARK_OP(x = y + alpha*x,      "AXPY", std::setprecision(3) << 3*vecN*sizeof(T)/time_spent * 1e-9, "GB/s")
    BENCHMARK_OP(s = inner_prod(x, y), "DOT",  std::setprecision(3) << 2*vecN*sizeof(T)/time_spent * 1e-9, "GB/s")
  }


  //BLAS2
  {
    viennacl::matrix<T> A(M, N);
    viennacl::vector<T> x(N);
    viennacl::vector<T> y(M);
    init_random(A);
    init_random(x);
    init_random(y);

    BENCHMARK_OP(y = prod(A, x),        "GEMV-N", std::setprecision(3) << (M + N + M*N)*sizeof(T)/time_spent * 1e-9, "GB/s")
    BENCHMARK_OP(x = prod(trans(A), y), "GEMV-T", std::setprecision(3) << (M + N + M*N)*sizeof(T)/time_spent * 1e-9, "GB/s")
  }

  //BLAS3
  {
    viennacl::matrix<T> C(M, N);
    viennacl::matrix<T> A(M, K);
    viennacl::matrix<T> B(K, N);
    viennacl::matrix<T> AT = trans(A);
    viennacl::matrix<T> BT = trans(B);
    init_random(A);
    init_random(B);

    BENCHMARK_OP(C = prod(A, B),                 "GEMM-NN",      int(2*M*N*K/time_spent*1e-9), "GFLOPs/s");
    BENCHMARK_OP(C = prod(A, trans(BT)),         "GEMM-NT",      int(2*M*N*K/time_spent*1e-9), "GFLOPs/s");
    BENCHMARK_OP(C = prod(trans(AT), B),         "GEMM-TN",      int(2*M*N*K/time_spent*1e-9), "GFLOPs/s");
    BENCHMARK_OP(C = prod(trans(AT), trans(BT)), "GEMM-TT",      int(2*M*N*K/time_spent*1e-9), "GFLOPs/s");
    BENCHMARK_OP(lu_factorize(A),                "LU-FACTORIZE", int(2*M*K*K/time_spent*1e-9), "GFLOPs/s");
  }


}

int main()
{
  std::size_t M = 2432;
  std::size_t N = 2432;
  std::size_t K = 2432;

  std::size_t xN = 5000000;
  std::cout << "Benchmark : BLAS" << std::endl;
  std::cout << "----------------" << std::endl;
  bench<float>(M, N, K, xN, "s");
  std::cout << "----" << std::endl;
  bench<double>(M, N, K, xN, "d");
}
