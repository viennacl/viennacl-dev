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

#define DOUBLE (1)
#define FLOAT  (0)

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

void usage()
{
  std::cout << "usage: bench_viennacl_avx matrix-size [double|float] [mode]" << std::endl;
  exit(-1);
}

void handle_args (int argc, char *argv[], bool &fast, bool &type, std::size_t &BLAS3_N)
{ 
  /* fall-through intended! */
  switch (argc)
  {
  case 4: 
    if (strncmp(argv[3],"fast", 4) == 0)
      fast = true;

  case 3:
    if ( strncmp(argv[2],"double", 6) == 0 )
      type = DOUBLE;
    else if (strncmp(argv[2],"float", 5) == 0)
      type = FLOAT;
    else 
      usage();
   
  case 2: 
    BLAS3_N = std::stoi(argv[1]);
    break;

  default: 
    usage();
  }
}

int main(int argc, char *argv[])
{
  std::size_t BLAS3_N;

  bool fast    = false;         
  bool type    = DOUBLE;

  handle_args(argc, argv, fast, type, BLAS3_N);

  std::string version;

# ifdef VIENNACL_WITH_AVX
#  ifdef AVX_KERNEL2
    version = "with AVX kernel 2 ";
#  else
    version = "with AVX ";
#  endif
# else
   version = "";
# endif

  /* bench with specified datatype */
  if (type == FLOAT)
  {
    bench<float>(BLAS3_N, fast);
    std::cerr << "ViennaCL " << version <<  "done! type: float, size: " << BLAS3_N  << std::endl;
  }
  else if(type == DOUBLE)
  {
    bench<double>(BLAS3_N, fast);
    std::cerr << "ViennaCL " << version <<  "done! type: double, size: " << BLAS3_N  << std::endl;
  }
}
