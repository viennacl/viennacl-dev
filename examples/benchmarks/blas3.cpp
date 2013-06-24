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
*   Benchmark: BLAS level 3 functionality for dense matrices (blas3.cpp and blas3.cu are identical, the latter being required for compilation using CUDA nvcc)
*
*/

//disable debug mechanisms to have a fair benchmark environment
#ifndef NDEBUG
 #define NDEBUG
#endif

//
// include necessary system headers
//
#include <iostream>

//
// ViennaCL includes
//
#include "viennacl/scalar.hpp"
#include "viennacl/vector.hpp"
#include "viennacl/matrix.hpp"
#include "viennacl/linalg/prod.hpp"
#include "viennacl/matrix_proxy.hpp"
#include "viennacl/linalg/lu.hpp"

// Some helper functions for this tutorial:
#include "../tutorial/Random.hpp"


#include "benchmark-utils.hpp"


#define BLAS3_MATRIX_SIZE   2048

template<typename ScalarType>
int run_benchmark()
{
  Timer timer;
  double exec_time;

  //
  // One alternative: Put the matrices into a contiguous block of memory (allows to use viennacl::fast_copy(), avoiding temporary memory)
  //
  std::vector<ScalarType> stl_A(BLAS3_MATRIX_SIZE * BLAS3_MATRIX_SIZE);
  std::vector<ScalarType> stl_B(BLAS3_MATRIX_SIZE * BLAS3_MATRIX_SIZE);
  std::vector<ScalarType> stl_C(BLAS3_MATRIX_SIZE * BLAS3_MATRIX_SIZE);

  //
  // Fill the matrix
  //
  for (unsigned int i = 0; i < BLAS3_MATRIX_SIZE; ++i)
    for (unsigned int j = 0; j < BLAS3_MATRIX_SIZE; ++j)
      stl_A[i*BLAS3_MATRIX_SIZE + j] = random<ScalarType>();

  for (unsigned int i = 0; i < BLAS3_MATRIX_SIZE; ++i)
    for (unsigned int j = 0; j < BLAS3_MATRIX_SIZE; ++j)
      stl_B[i + j*BLAS3_MATRIX_SIZE] = random<ScalarType>();

  //
  // Set up some ViennaCL objects
  //
#ifdef VIENNACL_WITH_OPENCL
  viennacl::ocl::set_context_device_type(0, viennacl::ocl::gpu_tag());
#endif

  //viennacl::ocl::current_context().build_options("-cl-mad-enable -cl-fast-relaxed-math");   //uncomment for additional optimizations
  //viennacl::ocl::current_context().build_options("-cl-opt-disable");                        //uncomment to get poor performance
  viennacl::matrix<ScalarType> vcl_A(BLAS3_MATRIX_SIZE, BLAS3_MATRIX_SIZE);
  viennacl::matrix<ScalarType> vcl_B(BLAS3_MATRIX_SIZE, BLAS3_MATRIX_SIZE);
  viennacl::matrix<ScalarType> vcl_C(BLAS3_MATRIX_SIZE, BLAS3_MATRIX_SIZE);


  /////////////////////////////////////////////////
  //////////// Matrix-matrix products /////////////
  /////////////////////////////////////////////////

  //
  // Now iterate over all OpenCL devices in the context and compute the matrix-matrix product
  //

  std::cout << " ------ Benchmark 1: Matrix-Matrix product ------ " << std::endl;


#ifdef VIENNACL_WITH_OPENCL
  std::vector<viennacl::ocl::device> devices = viennacl::ocl::current_context().devices();
#else
  std::vector<long> devices(1);
#endif
  for (std::size_t i=0; i<devices.size(); ++i)
  {
#ifdef VIENNACL_WITH_OPENCL
    viennacl::ocl::current_context().switch_device(devices[i]);
    std::cout << " - Device Name: " << viennacl::ocl::current_device().name() << std::endl;
#endif

    viennacl::fast_copy(&(stl_A[0]),
                        &(stl_A[0]) + stl_A.size(),
                        vcl_A);
    viennacl::fast_copy(&(stl_B[0]),
                        &(stl_B[0]) + stl_B.size(),
                        vcl_B);
    vcl_C = viennacl::linalg::prod(vcl_A, vcl_B);
    viennacl::backend::finish();
    timer.start();
    vcl_C = viennacl::linalg::prod(vcl_A, vcl_B);
    viennacl::backend::finish();
    exec_time = timer.get();
    std::cout << " - Execution time on device (no setup time included): " << exec_time << std::endl;
    std::cout << " - GFLOPs (counting multiply&add as separate operations): " << 2.0 * (vcl_A.size1() / 1000.0) * (vcl_A.size2() / 1000.0) * (vcl_B.size2() / 1000.0) / exec_time << std::endl;
    std::cout << std::endl;
  }

  std::cout << " ------ Benchmark 2: Matrix-Matrix product using ranges ------ " << std::endl;

  viennacl::range r(BLAS3_MATRIX_SIZE/4, 3 * BLAS3_MATRIX_SIZE/4);
  for (std::size_t i=0; i<devices.size(); ++i)
  {
#ifdef VIENNACL_WITH_OPENCL
    viennacl::ocl::current_context().switch_device(devices[i]);
    std::cout << " - Device Name: " << viennacl::ocl::current_device().name() << std::endl;
#endif

    viennacl::fast_copy(&(stl_A[0]),
                        &(stl_A[0]) + stl_A.size(),
                        vcl_A);
    viennacl::fast_copy(&(stl_B[0]),
                        &(stl_B[0]) + stl_B.size(),
                        vcl_B);
    viennacl::project(vcl_C, r, r) = viennacl::linalg::prod(viennacl::project(vcl_A, r, r), viennacl::project(vcl_B, r, r));
    viennacl::backend::finish();
    timer.start();
    viennacl::project(vcl_C, r, r) = viennacl::linalg::prod(viennacl::project(vcl_A, r, r), viennacl::project(vcl_B, r, r));
    viennacl::backend::finish();
    exec_time = timer.get();
    std::cout << " - Execution time on device (no setup time included): " << exec_time << std::endl;
    std::cout << " - GFLOPs (counting multiply&add as separate operations): " << 2.0 * (vcl_A.size1() / 2000.0) * (vcl_A.size2() / 2000.0) * (vcl_B.size2() / 2000.0) / exec_time << std::endl;
    std::cout << std::endl;
  }

  std::cout << " ------ Benchmark 3: Matrix-Matrix product using slices ------ " << std::endl;

  viennacl::slice s(0, 2, BLAS3_MATRIX_SIZE/2);
  for (std::size_t i=0; i<devices.size(); ++i)
  {
#ifdef VIENNACL_WITH_OPENCL
    viennacl::ocl::current_context().switch_device(devices[i]);
    std::cout << " - Device Name: " << viennacl::ocl::current_device().name() << std::endl;
#endif

    viennacl::fast_copy(&(stl_A[0]),
                        &(stl_A[0]) + stl_A.size(),
                        vcl_A);
    viennacl::fast_copy(&(stl_B[0]),
                        &(stl_B[0]) + stl_B.size(),
                        vcl_B);
    viennacl::project(vcl_C, s, s) = viennacl::linalg::prod(viennacl::project(vcl_A, s, s), viennacl::project(vcl_B, s, s));
    viennacl::backend::finish();
    timer.start();
    viennacl::project(vcl_C, s, s) = viennacl::linalg::prod(viennacl::project(vcl_A, s, s), viennacl::project(vcl_B, s, s));
    viennacl::backend::finish();
    exec_time = timer.get();
    std::cout << " - Execution time on device (no setup time included): " << exec_time << std::endl;
    std::cout << " - GFLOPs (counting multiply&add as separate operations): " << 2.0 * (vcl_A.size1() / 2000.0) * (vcl_A.size2() / 2000.0) * (vcl_B.size2() / 2000.0) / exec_time << std::endl;
    std::cout << std::endl;
  }


  std::cout << " ------ Benchmark 4: LU factorization ------ " << std::endl;

  for (std::size_t i=0; i<devices.size(); ++i)
  {
#ifdef VIENNACL_WITH_OPENCL
    viennacl::ocl::current_context().switch_device(devices[i]);
    std::cout << " - Device Name: " << viennacl::ocl::current_device().name() << std::endl;
#endif

    viennacl::fast_copy(&(stl_A[0]),
                        &(stl_A[0]) + stl_A.size(),
                        vcl_A);
    viennacl::linalg::lu_factorize(vcl_A);
    viennacl::backend::finish();
    timer.start();
    viennacl::linalg::lu_factorize(vcl_A);
    viennacl::backend::finish();
    exec_time = timer.get();
    std::cout << " - Execution time on device (no setup time included): " << exec_time << std::endl;
    std::cout << " - GFLOPs (counting multiply&add as separate operations): " << 2.0 * (vcl_A.size1() / 1000.0) * (vcl_A.size2() / 1000.0) * (vcl_A.size2() / 1000.0) / exec_time << std::endl;
    std::cout << std::endl;
  }

  return EXIT_SUCCESS;
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
  std::cout << "## Benchmark :: Dense Matrix-Matrix product " << std::endl;
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
