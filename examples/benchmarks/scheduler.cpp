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

/*
*
*   Benchmark:   Vector operations (vector.cpp and vector.cu are identical, the latter being required for compilation using CUDA nvcc)
*
*/


//#define VIENNACL_DEBUG_ALL
#ifndef NDEBUG
 #define NDEBUG
#endif

#include "viennacl/scalar.hpp"
#include "viennacl/vector.hpp"
#include "viennacl/matrix.hpp"
#include "viennacl/linalg/inner_prod.hpp"
#include "viennacl/linalg/norm_2.hpp"
#include "viennacl/scheduler/execute.hpp"
#include "viennacl/tools/timer.hpp"

#include <iostream>
#include <vector>

using std::cout;
using std::cin;
using std::endl;


#define BENCHMARK_VECTOR_SIZE   2
#define BENCHMARK_RUNS          1000


template<typename ScalarType>
int run_benchmark()
{

  viennacl::tools::timer timer;
  double exec_time;

  std::vector<ScalarType> std_vec1(BENCHMARK_VECTOR_SIZE);
  std::vector<ScalarType> std_vec2(BENCHMARK_VECTOR_SIZE);
  viennacl::vector<ScalarType> vcl_vec1(BENCHMARK_VECTOR_SIZE);
  viennacl::vector<ScalarType> vcl_vec2(BENCHMARK_VECTOR_SIZE);
  ScalarType alpha = ScalarType(1.1415);
  ScalarType beta  = ScalarType(0.97172);


  ///////////// Vector operations /////////////////

  std_vec1[0] = 1.0;
  std_vec2[0] = 1.0;
  for (std::size_t i=1; i<BENCHMARK_VECTOR_SIZE; ++i)
  {
    std_vec1[i] = std_vec1[i-1] * ScalarType(1.000001);
    std_vec2[i] = std_vec1[i-1] * ScalarType(0.999999);
  }

  viennacl::copy(std_vec1, vcl_vec1);
  viennacl::fast_copy(std_vec1, vcl_vec1);
  viennacl::copy(std_vec2, vcl_vec2);

  viennacl::backend::finish();
  vcl_vec2 = alpha * vcl_vec1 + beta * vcl_vec2;
  viennacl::backend::finish();
  timer.start();
  for (std::size_t runs=0; runs<BENCHMARK_RUNS; ++runs)
  {
    vcl_vec2 = alpha * vcl_vec1 + beta * vcl_vec2;
  }
  viennacl::backend::finish();
  exec_time = timer.get();
  std::cout << "Execution time per operation, no scheduler: " << exec_time / BENCHMARK_RUNS << " sec" << std::endl;
  std::cout << "Result: " << vcl_vec2[0] << std::endl;

  viennacl::backend::finish();
  timer.start();
  for (std::size_t runs=0; runs<BENCHMARK_RUNS; ++runs)
  {
    viennacl::scheduler::statement   my_statement(vcl_vec2, viennacl::op_assign(), alpha * vcl_vec1 + beta * vcl_vec2); // same as vcl_v1 = alpha * vcl_vec1 + beta * vcl_vec2;
    viennacl::scheduler::execute(my_statement);
  }
  viennacl::backend::finish();
  exec_time = timer.get();
  std::cout << "Execution time per operation, with scheduler including statement generation: " << exec_time / BENCHMARK_RUNS << " sec" << std::endl;
  std::cout << "Result: " << vcl_vec2[0] << std::endl;

  viennacl::scheduler::statement   my_statement(vcl_vec2, viennacl::op_assign(), alpha * vcl_vec1 + beta * vcl_vec2); // same as vcl_v1 = alpha * vcl_vec1 + beta * vcl_vec2;
  viennacl::backend::finish();
  timer.start();
  for (std::size_t runs=0; runs<BENCHMARK_RUNS; ++runs)
  {
    viennacl::scheduler::execute(my_statement);
  }
  viennacl::backend::finish();
  exec_time = timer.get();
  std::cout << "Execution time per operation, only execution: " << exec_time / BENCHMARK_RUNS << " sec" << std::endl;
  std::cout << "Result: " << vcl_vec2[0] << std::endl;

  return 0;
}

int main()
{
  std::cout << std::endl;
  std::cout << "----------------------------------------------" << std::endl;
  std::cout << "               Device Info" << std::endl;
  std::cout << "----------------------------------------------" << std::endl;

#ifdef VIENNACL_WITH_OPENCL
  std::cout << viennacl::ocl::current_device().info() << std::endl;
#endif

  std::cout << std::endl;
  std::cout << "----------------------------------------------" << std::endl;
  std::cout << "----------------------------------------------" << std::endl;
  std::cout << "## Benchmark :: Vector" << std::endl;
  std::cout << "----------------------------------------------" << std::endl;
  std::cout << std::endl;
  std::cout << "   -------------------------------" << std::endl;
  std::cout << "   # benchmarking single-precision" << std::endl;
  std::cout << "   -------------------------------" << std::endl;
  run_benchmark<float>();
#ifdef VIENNACL_WITH_OPENCL
  if ( viennacl::ocl::current_device().double_support() )
#endif
  {
    std::cout << std::endl;
    std::cout << "   -------------------------------" << std::endl;
    std::cout << "   # benchmarking double-precision" << std::endl;
    std::cout << "   -------------------------------" << std::endl;
    run_benchmark<double>();
  }
  return 0;
}

