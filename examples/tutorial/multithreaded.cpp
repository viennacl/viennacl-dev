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

/** \example multithreaded.cpp
*
*   This tutorial shows how to use ViennaCL with multiple threads, one thread per GPU.
*   The use of one thread per context (host, CUDA, OpenCL) is supported.
*   However, using more than one thread per context is not fully covered by the OpenCL standard.
*   It is, however, perfectly fine to use multiple OpenCL contexts simultaneously, each using one thread.
*
*   We start with including the necessary headers:
**/

#ifndef VIENNACL_WITH_OPENCL
  #define VIENNACL_WITH_OPENCL
#endif

// include necessary system headers
#include <iostream>

//include basic scalar and vector types of ViennaCL
#include "viennacl/scalar.hpp"
#include "viennacl/vector.hpp"
#include "viennacl/matrix.hpp"

#include "viennacl/ocl/device.hpp"
#include "viennacl/ocl/platform.hpp"
#include "viennacl/ocl/backend.hpp"

//include the generic inner product functions of ViennaCL
#include "viennacl/linalg/norm_2.hpp"


#include <boost/thread.hpp>


/**
*  Each thread runs a separate task, for which we provide the following functor.
*  For each thread, vector operations and a vector norm gets computed.
**/
template<typename NumericT>
class worker
{
public:
  worker(std::size_t tid) : thread_id_(tid) {}

  void operator()()
  {
    std::size_t N = 6;

    viennacl::context ctx(viennacl::ocl::get_context(static_cast<long>(thread_id_)));
    viennacl::vector<NumericT> u = viennacl::scalar_vector<NumericT>(N, NumericT(1) * NumericT(thread_id_ + 1), ctx);
    viennacl::vector<NumericT> v = viennacl::scalar_vector<NumericT>(N, NumericT(2) * NumericT(thread_id_ + 1), ctx);
    viennacl::matrix<NumericT> A = viennacl::linalg::outer_prod(u, v);
    viennacl::vector<NumericT> x(u);

    u += v;
    NumericT result = viennacl::linalg::norm_2(u);

    std::stringstream ss;
    ss << "Result of thread " << thread_id_ << " on device " << viennacl::ocl::get_context(static_cast<long>(thread_id_)).devices()[0].name() << ": " << result << std::endl;
    ss << "  A: " << A << std::endl;
    ss << "  x: " << x << std::endl;
    message_ = ss.str();
  }

  std::string message() const { return message_; }

private:
  std::string message_;
  std::size_t thread_id_;
};

/**
*   In the main routine we create two OpenCL contexts and then use one thread per context to run the operations in the functor defined above.
**/
int main()
{
  // Change this type definition to double if your gpu supports that
  typedef float       ScalarType;

  if (viennacl::ocl::get_platforms().size() == 0)
  {
    std::cerr << "Error: No platform found!" << std::endl;
    return EXIT_FAILURE;
  }

  /**
  *   Part 1: Setup first device for first context, second device for second context:
  **/
  viennacl::ocl::platform pf = viennacl::ocl::get_platforms()[0];
  std::vector<viennacl::ocl::device> const & devices = pf.devices();

  // Set first device to first context:
  viennacl::ocl::setup_context(0, devices[0]);

  // Set second device for second context (use the same device for the second context if only one device available):
  if (devices.size() > 1)
    viennacl::ocl::setup_context(1, devices[1]);
  else
    viennacl::ocl::setup_context(1, devices[0]);

  /**
  *   Part 2: Now let two threads operate on two GPUs in parallel.
  **/

  worker<ScalarType> work_functor0(0);
  worker<ScalarType> work_functor1(1);
  boost::thread worker_thread_0(boost::ref(work_functor0));
  boost::thread worker_thread_1(boost::ref(work_functor1));

  worker_thread_0.join();
  worker_thread_1.join();

  std::cout << work_functor0.message() << std::endl;
  std::cout << work_functor1.message() << std::endl;

  /**
  *  We're done - print a success message.
  **/
  std::cout << "!!!! TUTORIAL COMPLETED SUCCESSFULLY !!!!" << std::endl;

  return EXIT_SUCCESS;
}

