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

//compile with g++ ../benchmark_1_eigen.cpp -o bench_1_eigen -Wall -pedantic -O3 -fopenmp -std=c++11


#include "Eigen/Dense"
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

#define DOUBLE (1)
#define FLOAT  (0)

/* benchmark macro: runs multiplication with prior warmup and measures and prints performance */
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

void bench_double(size_t BLAS3_N)
{

  viennacl::tools::timer timer;
  double time_previous, time_spent;
  size_t Nruns;
  double time_per_benchmark = 1;
  int i;
   
  /* double matrices */
  Eigen::MatrixXd Cd = Eigen::MatrixXd::Random(BLAS3_N, BLAS3_N);
  Eigen::MatrixXd Ad = Eigen::MatrixXd::Random(BLAS3_N, BLAS3_N);
  Eigen::MatrixXd Bd = Eigen::MatrixXd::Random(BLAS3_N, BLAS3_N);
  Eigen::MatrixXd ATd = Ad.transpose();
  Eigen::MatrixXd BTd = Bd.transpose();

  /* double operations */
  BENCHMARK_OP(Cd = Ad*Bd,                 double(2*BLAS3_N*BLAS3_N*BLAS3_N)/time_spent*1e-9);
  BENCHMARK_OP(Cd = Ad*BTd.transpose(),         double(2*BLAS3_N*BLAS3_N*BLAS3_N)/time_spent*1e-9);
  BENCHMARK_OP(Cd = ATd.transpose()*Bd,         double(2*BLAS3_N*BLAS3_N*BLAS3_N)/time_spent*1e-9);
  BENCHMARK_OP(Cd = ATd.transpose()*BTd.transpose(), double(2*BLAS3_N*BLAS3_N*BLAS3_N)/time_spent*1e-9);
}

void bench_float(size_t BLAS3_N)
{
  viennacl::tools::timer timer;
  double time_previous, time_spent;
  size_t Nruns;
  double time_per_benchmark = 1;
  int i;

  /*float matrices */
  Eigen::MatrixXf Cf = Eigen::MatrixXf::Random(BLAS3_N, BLAS3_N);
  Eigen::MatrixXf Af = Eigen::MatrixXf::Random(BLAS3_N, BLAS3_N);
  Eigen::MatrixXf Bf = Eigen::MatrixXf::Random(BLAS3_N, BLAS3_N);
  Eigen::MatrixXf ATf = Af.transpose();
  Eigen::MatrixXf BTf = Bf.transpose();

  /* float operations */
  BENCHMARK_OP(Cf = Af*Bf,                      double(2*BLAS3_N*BLAS3_N*BLAS3_N)/time_spent*1e-9);
  BENCHMARK_OP(Cf = Af*BTf.transpose(),         double(2*BLAS3_N*BLAS3_N*BLAS3_N)/time_spent*1e-9);
  BENCHMARK_OP(Cf = ATf.transpose()*Bf,         double(2*BLAS3_N*BLAS3_N*BLAS3_N)/time_spent*1e-9);
  BENCHMARK_OP(Cf = ATf.transpose()*BTf.transpose(), double(2*BLAS3_N*BLAS3_N*BLAS3_N)/time_spent*1e-9);
    
}

void usage()
{
  std::cout << "usage: bench_eigen matrix-size [double|float] [mode]" << std::endl;
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
    if (strncmp(argv[2],"double", 6))
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
  
  bool fast = false;
  bool type = DOUBLE;

  handle_args(argc, argv, fast, type, BLAS3_N);

 /* bench with specified datatype */
  if (type == FLOAT)
  {
    bench_float(BLAS3_N);
    std::cerr << "Eigen done! type: float, size: " << BLAS3_N << std::endl;
  }
  else if(type == DOUBLE)
  {
    bench_double(BLAS3_N);
    std::cerr << "Eigen done! type: double, size: " << BLAS3_N << std::endl;
  }
}
