/* =========================================================================
   Copyright (c) 2010-2013, Institute for Microelectronics,
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
#include "viennacl/linalg/inner_prod.hpp"
#include "viennacl/linalg/norm_2.hpp"

#include <iostream>
#include <vector>
#include "benchmark-utils.hpp"

using std::cout;
using std::cin;
using std::endl;


#define BENCHMARK_VECTOR_SIZE   3000000
#define BENCHMARK_RUNS          10


template<typename ScalarType>
int run_benchmark()
{
   
   Timer timer;
   double exec_time;
   
  ScalarType std_result = 0;
   
  ScalarType std_factor1 = static_cast<ScalarType>(3.1415);
  ScalarType std_factor2 = static_cast<ScalarType>(42.0);
  viennacl::scalar<ScalarType> vcl_factor1(std_factor1);
  viennacl::scalar<ScalarType> vcl_factor2(std_factor2);
  
  std::vector<ScalarType> std_vec1(BENCHMARK_VECTOR_SIZE);
  std::vector<ScalarType> std_vec2(BENCHMARK_VECTOR_SIZE);
  std::vector<ScalarType> std_vec3(BENCHMARK_VECTOR_SIZE);
  viennacl::vector<ScalarType> vcl_vec1(BENCHMARK_VECTOR_SIZE);
  viennacl::vector<ScalarType> vcl_vec2(BENCHMARK_VECTOR_SIZE); 
  viennacl::vector<ScalarType> vcl_vec3(BENCHMARK_VECTOR_SIZE); 

  
  ///////////// Vector operations /////////////////
  
  std_vec1[0] = 1.0;
  std_vec2[0] = 1.0;
  for (int i=1; i<BENCHMARK_VECTOR_SIZE; ++i)
  {
    std_vec1[i] = std_vec1[i-1] * ScalarType(1.000001);
    std_vec2[i] = std_vec1[i-1] * ScalarType(0.999999);
  }

  viennacl::copy(std_vec1, vcl_vec1);
  viennacl::fast_copy(std_vec1, vcl_vec1);
  viennacl::copy(std_vec2, vcl_vec2);
  
  viennacl::swap(vcl_vec1, vcl_vec2);
  //check that vcl_vec1 is now equal to std_vec2:
  viennacl::fast_copy(vcl_vec1, std_vec3);
  for (int i=0; i<BENCHMARK_VECTOR_SIZE; ++i)
    if (std_vec3[i] != std_vec2[i])
      std::cout << "ERROR in swap(): Failed at entry " << i << std::endl;
  
  viennacl::fast_swap(vcl_vec1, vcl_vec2);
  //check that vcl_vec1 is now equal to std_vec1 again:
  viennacl::copy(vcl_vec1, std_vec3);
  for (int i=0; i<BENCHMARK_VECTOR_SIZE; ++i)
    if (std_vec3[i] != std_vec1[i])
      std::cout << "ERROR in fast_swap(): Failed at entry " << i << std::endl;
  
  
  // inner product
  viennacl::backend::finish();
  std::cout << "------- Vector inner products ----------" << std::endl;
  timer.start();
  for (int runs=0; runs<BENCHMARK_RUNS; ++runs)
  {
    std_result = 0;
    for (int i=0; i<BENCHMARK_VECTOR_SIZE; ++i)
      std_result += std_vec1[i] * std_vec2[i];
  }
  exec_time = timer.get();
  std::cout << "CPU time: " << exec_time << std::endl;
  std::cout << "CPU "; printOps(2.0 * static_cast<double>(std_vec1.size()), static_cast<double>(exec_time) / static_cast<double>(BENCHMARK_RUNS));
  std::cout << "Result:" << std_result << std::endl;
  
  
  std_result = viennacl::linalg::inner_prod(vcl_vec1, vcl_vec2); //startup calculation
  std_result = 0.0;
  viennacl::backend::finish();
  timer.start();
  for (int runs=0; runs<BENCHMARK_RUNS; ++runs)
  {
    vcl_factor2 = viennacl::linalg::inner_prod(vcl_vec1, vcl_vec2);
  }
  viennacl::backend::finish();
  exec_time = timer.get();
  std::cout << "GPU time: " << exec_time << std::endl;
  std::cout << "GPU "; printOps(2.0 * static_cast<double>(std_vec1.size()), static_cast<double>(exec_time) / static_cast<double>(BENCHMARK_RUNS));
  std::cout << "Result: " << vcl_factor2 << std::endl;

  // inner product
  viennacl::backend::finish();
  std::cout << "------- Vector norm_2 ----------" << std::endl;
  timer.start();
  for (int runs=0; runs<BENCHMARK_RUNS; ++runs)
  {
    std_result = 0;
    for (int i=0; i<BENCHMARK_VECTOR_SIZE; ++i)
    {
      ScalarType entry = std_vec1[i]; 
      std_result += entry * entry;
    }
  }
  std_result = std::sqrt(std_result);
  exec_time = timer.get();
  std::cout << "CPU time: " << exec_time << std::endl;
  std::cout << "CPU "; printOps(2.0 * static_cast<double>(std_vec1.size()), static_cast<double>(exec_time) / static_cast<double>(BENCHMARK_RUNS));
  std::cout << "Result:" << std_result << std::endl;
  
  
  std_result = viennacl::linalg::norm_2(vcl_vec1); //startup calculation
  std_result = 0.0;
  viennacl::backend::finish();
  timer.start();
  for (int runs=0; runs<BENCHMARK_RUNS; ++runs)
  {
    vcl_factor2 = viennacl::linalg::norm_2(vcl_vec1);
  }
  viennacl::backend::finish();
  exec_time = timer.get();
  std::cout << "GPU time: " << exec_time << std::endl;
  std::cout << "GPU "; printOps(2.0 * static_cast<double>(std_vec1.size()), static_cast<double>(exec_time) / static_cast<double>(BENCHMARK_RUNS));
  std::cout << "Result: " << vcl_factor2 << std::endl;
  
  // vector addition
  
  std::cout << "------- Vector addition ----------" << std::endl;
  timer.start();
  for (int runs=0; runs<BENCHMARK_RUNS; ++runs)
  {
    for (int i=0; i<BENCHMARK_VECTOR_SIZE; ++i)
      std_vec3[i] = std_vec1[i] + std_vec2[i];
  }
  exec_time = timer.get();
  std::cout << "CPU time: " << exec_time << std::endl;
  std::cout << "CPU "; printOps(2.0 * static_cast<double>(std_vec1.size()), static_cast<double>(exec_time) / static_cast<double>(BENCHMARK_RUNS));
  
  
  vcl_vec3 = vcl_vec1 + vcl_vec2; //startup calculation
  viennacl::backend::finish();
  std_result = 0.0;
  timer.start();
  for (int runs=0; runs<BENCHMARK_RUNS; ++runs)
  {
    vcl_vec3 = vcl_vec1 + vcl_vec2;
  }
  viennacl::backend::finish();
  exec_time = timer.get();
  std::cout << "GPU time: " << exec_time << std::endl;
  std::cout << "GPU "; printOps(2.0 * static_cast<double>(std_vec1.size()), static_cast<double>(exec_time) / static_cast<double>(BENCHMARK_RUNS));
  
  
  
  
  // multiply add:
  std::cout << "------- Vector multiply add ----------" << std::endl;
  timer.start();
  for (int runs=0; runs<BENCHMARK_RUNS; ++runs)
  {
    for (int i=0; i<BENCHMARK_VECTOR_SIZE; ++i)
      std_vec1[i] += std_factor1 * std_vec2[i];
  }
  exec_time = timer.get();
  std::cout << "CPU time: " << exec_time << std::endl;
  std::cout << "CPU "; printOps(2.0 * static_cast<double>(std_vec1.size()), static_cast<double>(exec_time) / static_cast<double>(BENCHMARK_RUNS));
  
  
  vcl_vec1 += vcl_factor1 * vcl_vec2; //startup calculation
  viennacl::backend::finish();
  timer.start();
  for (int runs=0; runs<BENCHMARK_RUNS; ++runs)
  {
    vcl_vec1 += vcl_factor1 * vcl_vec2;
  }
  viennacl::backend::finish();
  exec_time = timer.get();
  std::cout << "GPU time: " << exec_time << std::endl;
  std::cout << "GPU "; printOps(2.0 * static_cast<double>(std_vec1.size()), static_cast<double>(exec_time) / static_cast<double>(BENCHMARK_RUNS));
  
 
  
  
  //complicated vector addition:
  std::cout << "------- Vector complicated expression ----------" << std::endl;
  timer.start();
  for (int runs=0; runs<BENCHMARK_RUNS; ++runs)
  {
    for (int i=0; i<BENCHMARK_VECTOR_SIZE; ++i)
      std_vec3[i] += std_vec2[i] / std_factor1 + std_factor2 * (std_vec1[i] - std_factor1 * std_vec2[i]);
  }
  exec_time = timer.get();
  std::cout << "CPU time: " << exec_time << std::endl;
  std::cout << "CPU "; printOps(6.0 * static_cast<double>(std_vec1.size()), static_cast<double>(exec_time) / static_cast<double>(BENCHMARK_RUNS));
  
  vcl_vec3 = vcl_vec2 / vcl_factor1 + vcl_factor2 * (vcl_vec1 - vcl_factor1*vcl_vec2); //startup calculation
  viennacl::backend::finish();
  timer.start();
  for (int runs=0; runs<BENCHMARK_RUNS; ++runs)
  {
    vcl_vec3 = vcl_vec2 / vcl_factor1 + vcl_factor2 * (vcl_vec1 - vcl_factor1*vcl_vec2);
  }
  viennacl::backend::finish();
  exec_time = timer.get();
  std::cout << "GPU time: " << exec_time << std::endl;
  std::cout << "GPU "; printOps(6.0 * static_cast<double>(std_vec1.size()), static_cast<double>(exec_time) / static_cast<double>(BENCHMARK_RUNS));
  
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
  if( viennacl::ocl::current_device().double_support() )
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

