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
#include "../../viennacl/matrix.hpp"
#include "../../viennacl/matrix_proxy.hpp"
#include "../../viennacl/linalg/prod.hpp"
#include "../../viennacl/tools/timer.hpp"

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
void bench(size_t BLAS3_N, bool fast)
{
  using viennacl::linalg::prod;
  using viennacl::trans;

  viennacl::tools::timer timer;
  double time_previous, time_spent;
  size_t Nruns;
  double time_per_benchmark = 1;
  int i;

#define BENCHMARK_OP(OPERATION, PERF)                   \
  OPERATION;                                            \
  timer.start();                                        \
  Nruns = 0;                                            \
  time_spent = 0;                                       \
  i = 0;                                                \
  while (i < 5)                                         \
  {                                                     \
    if(time_spent >= time_per_benchmark)                \
      break;                                            \
    time_previous = timer.get();                        \
    OPERATION;                                          \
    time_spent += timer.get() - time_previous;          \
    Nruns+=1;                                           \
    i++;                                                \
  }                                                     \
  time_spent/=(double)Nruns;                            \
  std::cout << PERF << " ";

  //BLAS3
  {
  viennacl::matrix<T,viennacl::column_major> C(BLAS3_N, BLAS3_N);
  viennacl::matrix<T,viennacl::column_major> A(BLAS3_N, BLAS3_N);
  viennacl::matrix<T,viennacl::column_major> B(BLAS3_N, BLAS3_N);
  init_random(A);
  init_random(B);
  viennacl::matrix<T,viennacl::column_major> AT = trans(A);
  viennacl::matrix<T,viennacl::column_major> BT = trans(B);
  

  BENCHMARK_OP(C = prod(A, B),                 double(2*BLAS3_N*BLAS3_N*BLAS3_N)/time_spent*1e-9);
  if (!fast)
  {
    BENCHMARK_OP(C = prod(A, trans(BT)),         double(2*BLAS3_N*BLAS3_N*BLAS3_N)/time_spent*1e-9);
    BENCHMARK_OP(C = prod(trans(AT), B),         double(2*BLAS3_N*BLAS3_N*BLAS3_N)/time_spent*1e-9);
    BENCHMARK_OP(C = prod(trans(AT), trans(BT)), double(2*BLAS3_N*BLAS3_N*BLAS3_N)/time_spent*1e-9);
  }
}


}

int main(int argc, char *argv[])
{
  std::size_t BLAS3_N;
  bool        fast    = false;         
  std::string type    = "double";
  std::string mode    = "not fast";

  if (argc >= 2)
  {
    BLAS3_N = std::stoi(argv[1]);
  }
  if (argc >= 3)
  {
    type = argv[2];

    if ((type != "float") && (type != "double"))
    {
      std::cout << "unsupported entry type!" << std::endl;
      return -1;
    }

  }
  if (argc >= 4)
  {
    mode = argv[3];

    /* skips transposed cases */
    if (mode == "fast")
      fast = true;
    else 
      fast = false;
  }
  if ((argc == 1) || (argc >= 5))
  {
    std::cout << "usage: bench_viennacl_avx matrix-size [entry-type] [mode]" << std::endl;
    return -1;
  }

  /* bench with specified datatype */
  if (type == "float")
    bench<float>(BLAS3_N, fast);
  else if(type == "double")
    bench<double>(BLAS3_N, fast);
  
#ifdef VIENNACL_WITH_AVX
#  ifdef AVX_KERNEL2
  std::cerr << type << ", size " << BLAS3_N << ": ViennaCL with AVX, Kernel2 done!" << std::endl;
#  else
  std::cerr << type << "size " << BLAS3_N << ": ViennaCL with AVX done!" << std::endl;
#  endif
#else
  std::cerr << type << "size " << BLAS3_N << ": ViennaCL done!" << std::endl;
#endif
}
