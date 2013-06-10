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
*   Tutorial: Using ViennaCL with multiple threads, one thread per GPU
*
*/

#ifndef VIENNACL_WITH_OPENCL
  #define VIENNACL_WITH_OPENCL
#endif

// include necessary system headers
#include <iostream>

//include basic scalar and vector types of ViennaCL
#include "viennacl/scalar.hpp"
#include "viennacl/vector.hpp"

#include "viennacl/ocl/device.hpp"
#include "viennacl/ocl/platform.hpp"
#include "viennacl/ocl/backend.hpp"

//include the generic inner product functions of ViennaCL
#include "viennacl/linalg/norm_2.hpp"

// Some helper functions for this tutorial:
#include "Random.hpp"

#include <boost/thread.hpp>

template <typename T>
void thread_func(viennacl::vector<T> * u, viennacl::vector<T> const * v, T * result, std::size_t thread_id) //Note: using references instead of pointers leads to some troubles with boost.thread
{
  *u += *v;
  *result = viennacl::linalg::norm_2(*u);
}


int main()
{
  //Change this type definition to double if your gpu supports that
  typedef float       ScalarType;

  if (viennacl::ocl::get_platforms().size() == 0)
  {
    std::cerr << "Error: No platform found!" << std::endl;
    return EXIT_FAILURE;
  }
  
  //
  // Part 1: Setup first device for first context, second device for second context:
  //  
  viennacl::ocl::platform pf = viennacl::ocl::get_platforms()[0];
  std::vector<viennacl::ocl::device> const & devices = pf.devices();
  
  // Set first device to first context:
  viennacl::ocl::setup_context(0, devices[0]);

  // Set second device for second context (use the same device for the second context if only one device available):
  if (devices.size() > 1)
    viennacl::ocl::setup_context(1, devices[1]);
  else
    viennacl::ocl::setup_context(1, devices[0]);

  //
  // Part 2: Now create vectors in the two contexts and let two threads process the results on two GPUs in parallel
  //
  
  std::size_t N = 10;
  
  viennacl::ocl::switch_context(0);
  viennacl::vector<ScalarType> u1 = viennacl::scalar_vector<ScalarType>(N, 1.0);
  viennacl::vector<ScalarType> v1 = viennacl::scalar_vector<ScalarType>(N, 2.0);
  ScalarType result1 = 0;
  
  viennacl::ocl::switch_context(1);
  viennacl::vector<ScalarType> u2 = viennacl::scalar_vector<ScalarType>(N, 2.0);
  viennacl::vector<ScalarType> v2 = viennacl::scalar_vector<ScalarType>(N, 4.0);
  ScalarType result2 = 0;
  
  // Two threads operating on both contexts at the same time:
  boost::thread worker_1(thread_func<ScalarType>, &u1, &v1, &result1, 1);
  boost::thread worker_2(thread_func<ScalarType>, &u2, &v2, &result2, 2);
  
  worker_1.join();
  worker_2.join();

  std::cout << "Result of thread 1: " << result1 << std::endl;
  std::cout << "Result of thread 2: " << result2 << std::endl;
  
  std::cout << "!!!! TUTORIAL COMPLETED SUCCESSFULLY !!!!" << std::endl;

  return EXIT_SUCCESS;
}

