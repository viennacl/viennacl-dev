/* =========================================================================
   Copyright (c) 2010-2016, Institute for Microelectronics,
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

template<class T, class F>
void init_random(viennacl::matrix<T, F> & M)
{
  std::vector<T> cM(M.internal_size());
  for (std::size_t i = 0; i < M.size1(); ++i)
    for (std::size_t j = 0; j < M.size2(); ++j)
      cM[F::mem_index(i, j, M.internal_size1(), M.internal_size2())] = T(rand())/T(RAND_MAX);
  viennacl::fast_copy(&cM[0],&cM[0] + cM.size(),M);
}

template<class T>
void init_random(viennacl::vector<T> & x)
{
  std::vector<T> cx(x.internal_size());
  for (std::size_t i = 0; i < cx.size(); ++i)
    cx[i] = T(rand())/T(RAND_MAX);
  viennacl::fast_copy(&cx[0], &cx[0] + cx.size(), x.begin());
}

template<class T>
void bench(size_t BLAS1_N, size_t BLAS2_M, size_t BLAS2_N, size_t BLAS3_M, size_t BLAS3_N, size_t BLAS3_K, std::string const & prefix)
{
  using viennacl::linalg::inner_prod;
  using viennacl::linalg::prod;
  using viennacl::linalg::lu_factorize;
  using viennacl::trans;

  viennacl::tools::timer timer;
  double time_previous, time_spent;
  size_t Nruns;
  double time_per_benchmark = 1;

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
    viennacl::vector<T> x(BLAS1_N);
    viennacl::vector<T> y(BLAS1_N);
    viennacl::vector<T> z(BLAS1_N);

    init_random(x);
    init_random(y);
    init_random(z);

    BENCHMARK_OP(x = y,                "COPY", std::setprecision(3) << double(2*BLAS1_N*sizeof(T))/time_spent * 1e-9, "GB/s")
    BENCHMARK_OP(x = y + alpha*x,      "AXPY", std::setprecision(3) << double(3*BLAS1_N*sizeof(T))/time_spent * 1e-9, "GB/s")
    BENCHMARK_OP(s = inner_prod(x, y), "DOT",  std::setprecision(3) << double(2*BLAS1_N*sizeof(T))/time_spent * 1e-9, "GB/s")
  }


  //BLAS2
  {
    viennacl::matrix<T,viennacl::column_major> A(BLAS2_M, BLAS2_N);
    viennacl::vector<T> x(BLAS2_N);
    viennacl::vector<T> y(BLAS2_M);
    init_random(A);
    init_random(x);
    init_random(y);

    BENCHMARK_OP(y = prod(A, x),        "GEMV-N", std::setprecision(3) << double((BLAS2_M + BLAS2_N + BLAS2_M*BLAS2_N)*sizeof(T))/time_spent * 1e-9, "GB/s")
    BENCHMARK_OP(x = prod(trans(A), y), "GEMV-T", std::setprecision(3) << double((BLAS2_M + BLAS2_N + BLAS2_M*BLAS2_N)*sizeof(T))/time_spent * 1e-9, "GB/s")
  }

  //BLAS3
  {
    viennacl::matrix<T,viennacl::column_major> C(BLAS3_M, BLAS3_N);
    viennacl::matrix<T,viennacl::column_major> A(BLAS3_M, BLAS3_K);
    viennacl::matrix<T,viennacl::column_major> B(BLAS3_K, BLAS3_N);
    viennacl::matrix<T,viennacl::column_major> AT = trans(A);
    viennacl::matrix<T,viennacl::column_major> BT = trans(B);
    init_random(A);
    init_random(B);

    BENCHMARK_OP(C = prod(A, B),                 "GEMM-NN",      double(2*BLAS3_M*BLAS3_N*BLAS3_K)/time_spent*1e-9, "GFLOPs/s");
    BENCHMARK_OP(C = prod(A, trans(BT)),         "GEMM-NT",      double(2*BLAS3_M*BLAS3_N*BLAS3_K)/time_spent*1e-9, "GFLOPs/s");
    BENCHMARK_OP(C = prod(trans(AT), B),         "GEMM-TN",      double(2*BLAS3_M*BLAS3_N*BLAS3_K)/time_spent*1e-9, "GFLOPs/s");
    BENCHMARK_OP(C = prod(trans(AT), trans(BT)), "GEMM-TT",      double(2*BLAS3_M*BLAS3_N*BLAS3_K)/time_spent*1e-9, "GFLOPs/s");
    //BENCHMARK_OP(lu_factorize(A),                "LU-FACTORIZE", double(2*BLAS3_M*BLAS3_K*BLAS3_K)/time_spent*1e-9, "GFLOPs/s");
  }


}

int main()
{
#ifdef VIENNACL_WITH_OPENCL
  std::cout << std::endl;
  std::cout << "----------------------------------------------" << std::endl;
  std::cout << "               Device Info" << std::endl;
  std::cout << "----------------------------------------------" << std::endl;
  std::cout << std::endl;
  std::cout << viennacl::ocl::current_device().info() << std::endl;
  std::cout << std::endl;
#endif

  std::size_t BLAS1_N = 10000000;

  std::size_t BLAS2_M = 3840;
  std::size_t BLAS2_N = 3840;

  std::size_t BLAS3_M = 1976;
  std::size_t BLAS3_N = 1976;
  std::size_t BLAS3_K = 1976;

  std::cout << "Benchmark : BLAS" << std::endl;
  std::cout << "----------------" << std::endl;
  bench<float>(BLAS1_N, BLAS2_M, BLAS2_N, BLAS3_M, BLAS3_N, BLAS3_K, "s");
  std::cout << "----" << std::endl;
#ifdef VIENNACL_WITH_OPENCL
  if ( viennacl::ocl::current_device().double_support() )
#endif
  bench<double>(BLAS1_N, BLAS2_M, BLAS2_N, BLAS3_M, BLAS3_N, BLAS3_K, "d");
}
