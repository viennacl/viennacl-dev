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
void thread_func(std::string * message, std::size_t thread_id) //Note: using references instead of pointers leads to some troubles with boost.thread
{
  std::size_t N = 10;

  viennacl::vector<T> u = viennacl::scalar_vector<T>(N, 1.0 * (thread_id + 1), viennacl::ocl::get_context(thread_id));
  viennacl::vector<T> v = viennacl::scalar_vector<T>(N, 2.0 * (thread_id + 1), viennacl::ocl::get_context(thread_id));

  u += v;
  T result = viennacl::linalg::norm_2(u);

  std::stringstream ss;
  ss << "Result of thread " << thread_id << " on device " << viennacl::ocl::get_context(thread_id).devices()[0].name() << ": " << result << std::endl;
  *message = ss.str();
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
  // Part 2: Now let two threads operate on two GPUs in parallel
  //

  std::string message0;
  std::string message1;
  boost::thread worker_0(thread_func<ScalarType>, &message0, 0);
  boost::thread worker_1(thread_func<ScalarType>, &message1, 1);

  worker_0.join();
  worker_1.join();

  std::cout << message0 << std::endl;
  std::cout << message1 << std::endl;

  std::cout << "!!!! TUTORIAL COMPLETED SUCCESSFULLY !!!!" << std::endl;

  return EXIT_SUCCESS;
}

