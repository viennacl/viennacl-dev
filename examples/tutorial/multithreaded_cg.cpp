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

/** \example multithreaded_cg.cpp
*
*   This tutorial shows how to run multiple instances of a conjugate gradient solver, one instance per GPU.
*
*   We start with including the necessary headers:
**/

#ifndef VIENNACL_WITH_OPENCL
  #define VIENNACL_WITH_OPENCL
#endif

// include necessary system headers
#include <iostream>

//
// ublas includes
//
#include <boost/numeric/ublas/io.hpp>
#include <boost/numeric/ublas/triangular.hpp>
#include <boost/numeric/ublas/matrix_sparse.hpp>
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/matrix_proxy.hpp>
#include <boost/numeric/ublas/operation.hpp>
#include <boost/numeric/ublas/operation_sparse.hpp>
#include <boost/numeric/ublas/io.hpp>
#include <boost/numeric/ublas/lu.hpp>

// Must be set if you want to use ViennaCL algorithms on ublas objects
#define VIENNACL_WITH_UBLAS 1


//include basic scalar and vector types of ViennaCL
#include "viennacl/scalar.hpp"
#include "viennacl/vector.hpp"
#include "viennacl/compressed_matrix.hpp"
#include "viennacl/io/matrix_market.hpp"

#include "viennacl/ocl/device.hpp"
#include "viennacl/ocl/platform.hpp"
#include "viennacl/ocl/backend.hpp"

//include the generic inner product functions of ViennaCL
#include "viennacl/linalg/norm_2.hpp"
#include "viennacl/linalg/cg.hpp"

// Some helper functions for this tutorial:
#include "vector-io.hpp"

using namespace boost::numeric;

/**
*   This tutorial uses Boost.Thread for threading. Other threading approaches (e.g. pthreads) also work.
**/
#include <boost/thread.hpp>

/**
*   This functor represents the work carried out in each thread.
*   It creates the necessary objects, loads the data, and executes the CG solver.
**/
template<typename NumericT>
class worker
{
public:
  worker(std::size_t tid) : thread_id_(tid) {}

  /**
  *   The functor interface, entry point for each thread.
  **/
  void operator()()
  {
    /**
    * Set up some ublas objects
    **/
    ublas::vector<NumericT> rhs;
    ublas::vector<NumericT> ref_result;
    ublas::compressed_matrix<NumericT> ublas_matrix;

    /**
    * Read system from file. You may also assemble everything on the fly here.
    **/
    if (!viennacl::io::read_matrix_market_file(ublas_matrix, "../examples/testdata/mat65k.mtx"))
    {
      std::cout << "Error reading Matrix file" << std::endl;
      return;
    }

    if (!readVectorFromFile("../examples/testdata/rhs65025.txt", rhs))
    {
      std::cout << "Error reading RHS file" << std::endl;
      return;
    }

    if (!readVectorFromFile("../examples/testdata/result65025.txt", ref_result))
    {
      std::cout << "Error reading Result file" << std::endl;
      return;
    }

    /**
    *  Set up some ViennaCL objects in the respective context.
    *  It is important to place the objects in the correct context (associated with each thread)
    **/
    viennacl::context ctx(viennacl::ocl::get_context(static_cast<long>(thread_id_)));

    std::size_t vcl_size = rhs.size();
    viennacl::compressed_matrix<NumericT> vcl_compressed_matrix(ctx);
    viennacl::vector<NumericT> vcl_rhs(vcl_size, ctx);
    viennacl::vector<NumericT> vcl_ref_result(vcl_size, ctx);

    viennacl::copy(rhs.begin(), rhs.end(), vcl_rhs.begin());
    viennacl::copy(ref_result.begin(), ref_result.end(), vcl_ref_result.begin());


    /**
    * Transfer ublas-matrix to ViennaCL objects sitting on the GPU:
    **/
    viennacl::copy(ublas_matrix, vcl_compressed_matrix);

    viennacl::vector<NumericT> vcl_result = viennacl::linalg::solve(vcl_compressed_matrix, vcl_rhs, viennacl::linalg::cg_tag());

    std::stringstream ss;
    ss << "Result of thread " << thread_id_ << " on device " << viennacl::ocl::get_context(static_cast<long>(thread_id_)).devices()[0].name() << ": " << vcl_result[0] << ", should: " << ref_result[0] << std::endl;
    message_ = ss.str();
  }

  std::string message() const { return message_; }

private:
  std::string message_;
  std::size_t thread_id_;
};

/**
*   In the main routine we create two OpenCL contexts and then use one thread per context to run the CG solver in the functor defined above.
**/
int main()
{
  //Change this type definition to double if your gpu supports that
  typedef float       ScalarType;

  if (viennacl::ocl::get_platforms().size() == 0)
  {
    std::cerr << "Error: No platform found!" << std::endl;
    return EXIT_FAILURE;
  }

  /**
  * Part 1: Setup first device for first context, second device for second context:
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
  * Part 2: Now let two threads operate on two GPUs in parallel, each running a CG solver
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
  *  That's it. Print a success message and exit.
  **/
  std::cout << "!!!! TUTORIAL COMPLETED SUCCESSFULLY !!!!" << std::endl;

  return EXIT_SUCCESS;
}

