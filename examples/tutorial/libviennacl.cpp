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

/** \example examples/tutorial/libviennacl.cpp
*
*   In this example we show how one can directly interface the ViennaCL BLAS-like shared library.
*   For simplicity, C++ is used as a host language, but one may also use C or any other language which is able to call C functions.
*
*   The first step is to include the necessary headers:
**/


// include necessary system headers
#include <iostream>
#include <vector>

// Some helper functions for this tutorial:
#include "viennacl.hpp"

#include "viennacl/vector.hpp"


/**
*  In this example we only create two vectors and swap values between them.
**/
int main()
{
  std::size_t size = 10;

  ViennaCLInt half_size = static_cast<ViennaCLInt>(size / 2);


  /**
  * Before we start we need to create a backend.
  * This allows one later to specify OpenCL command queues, CPU threads, or CUDA streams while preserving common interfaces.
  **/
  ViennaCLBackend my_backend;
  ViennaCLBackendCreate(&my_backend);


  /**
  *  <h2>Host-based Execution</h2>
  *
  *  We use the host to swap all odd entries of x (all ones) with all even entries in y (all twos):
  **/

  viennacl::vector<float> host_x = viennacl::scalar_vector<float>(size, 1.0, viennacl::context(viennacl::MAIN_MEMORY));
  viennacl::vector<float> host_y = viennacl::scalar_vector<float>(size, 2.0, viennacl::context(viennacl::MAIN_MEMORY));

  ViennaCLHostSswap(my_backend, half_size,
                    viennacl::linalg::host_based::detail::extract_raw_pointer<float>(host_x), 1, 2,
                    viennacl::linalg::host_based::detail::extract_raw_pointer<float>(host_y), 0, 2);

  std::cout << " --- Host ---" << std::endl;
  std::cout << "host_x: " << host_x << std::endl;
  std::cout << "host_y: " << host_y << std::endl;

  /**
  *   <h2>CUDA-based Execution</h2>
  *
  *  We use CUDA to swap all even entries in x (all ones) with all odd entries in y (all twos)
  **/

#ifdef VIENNACL_WITH_CUDA
  viennacl::vector<float> cuda_x = viennacl::scalar_vector<float>(size, 1.0, viennacl::context(viennacl::CUDA_MEMORY));
  viennacl::vector<float> cuda_y = viennacl::scalar_vector<float>(size, 2.0, viennacl::context(viennacl::CUDA_MEMORY));

  ViennaCLCUDASswap(my_backend, half_size,
                    viennacl::cuda_arg(cuda_x), 0, 2,
                    viennacl::cuda_arg(cuda_y), 1, 2);

  std::cout << " --- CUDA ---" << std::endl;
  std::cout << "cuda_x: " << cuda_x << std::endl;
  std::cout << "cuda_y: " << cuda_y << std::endl;
#endif

  /**
  *  <h2>OpenCL-based Execution</h2>
  *
  *  Use OpenCL to swap all odd entries in x (all ones) with all odd entries in y (all twos)
  **/

#ifdef VIENNACL_WITH_OPENCL
  long context_id = 0;
  viennacl::vector<float> opencl_x = viennacl::scalar_vector<float>(size, 1.0, viennacl::context(viennacl::ocl::get_context(context_id)));
  viennacl::vector<float> opencl_y = viennacl::scalar_vector<float>(size, 2.0, viennacl::context(viennacl::ocl::get_context(context_id)));

  ViennaCLBackendSetOpenCLContextID(my_backend, static_cast<ViennaCLInt>(context_id));

  ViennaCLOpenCLSswap(my_backend, half_size,
                      viennacl::traits::opencl_handle(opencl_x).get(), 1, 2,
                      viennacl::traits::opencl_handle(opencl_y).get(), 1, 2);

  std::cout << " --- OpenCL ---" << std::endl;
  std::cout << "opencl_x: " << opencl_x << std::endl;
  std::cout << "opencl_y: " << opencl_y << std::endl;
#endif

  /**
  *  The last step is to clean up by destroying the backend:
  **/
  ViennaCLBackendDestroy(&my_backend);

  std::cout << "!!!! TUTORIAL COMPLETED SUCCESSFULLY !!!!" << std::endl;

  return EXIT_SUCCESS;
}

