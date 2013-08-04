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
*   Tutorial: Using the ViennaCL BLAS-like shared library
*
*/


// include necessary system headers
#include <iostream>
#include <vector>

// Some helper functions for this tutorial:
#include "viennacl.hpp"

#include "viennacl/vector.hpp"

int main()
{
  std::size_t size = 10;

  //
  // Part 1: Host-based execution
  //

  viennacl::vector<float> host_x = viennacl::scalar_vector<float>(size, 1.0, viennacl::context(viennacl::MAIN_MEMORY));
  viennacl::vector<float> host_y = viennacl::scalar_vector<float>(size, 2.0, viennacl::context(viennacl::MAIN_MEMORY));

  // Create backend:
  ViennaCLHostBackend my_host_backend = NULL;

  ViennaCLHostSswap(my_host_backend, size/2,
                    viennacl::linalg::host_based::detail::extract_raw_pointer<float>(host_x), 1, 2,
                    viennacl::linalg::host_based::detail::extract_raw_pointer<float>(host_y), 0, 2);

  ViennaCLHostSscal(my_host_backend, size,
                    2.0f,
                    viennacl::linalg::host_based::detail::extract_raw_pointer<float>(host_x), 0, 1);

  std::cout << " --- Host ---" << std::endl;
  std::cout << "host_x: " << host_x << std::endl;
  std::cout << "host_y: " << host_y << std::endl;

  //
  // Part 2: CUDA-based execution
  //

#ifdef VIENNACL_WITH_CUDA
  ViennaCLCUDABackend my_cuda_backend = NULL;

  viennacl::vector<float> cuda_x = viennacl::scalar_vector<float>(size, 1.0, viennacl::context(viennacl::CUDA_MEMORY));
  viennacl::vector<float> cuda_y = viennacl::scalar_vector<float>(size, 2.0, viennacl::context(viennacl::CUDA_MEMORY));

  ViennaCLCUDASswap(my_cuda_backend, size/2,
                    viennacl::linalg::cuda::detail::cuda_arg<float>(cuda_x), 0, 2,
                    viennacl::linalg::cuda::detail::cuda_arg<float>(cuda_y), 1, 2);

  ViennaCLCUDASscal(my_cuda_backend, size,
                    2.0f,
                    viennacl::linalg::cuda::detail::cuda_arg<float>(cuda_x), 0, 1);

  std::cout << " --- CUDA ---" << std::endl;
  std::cout << "cuda_x: " << cuda_x << std::endl;
  std::cout << "cuda_y: " << cuda_y << std::endl;
#endif

  //
  // Part 3: OpenCL-based execution
  //

#ifdef VIENNACL_WITH_OPENCL
  std::size_t context_id = 0;
  viennacl::vector<float> opencl_x = viennacl::scalar_vector<float>(size, 1.0, viennacl::context(viennacl::ocl::get_context(context_id)));
  viennacl::vector<float> opencl_y = viennacl::scalar_vector<float>(size, 2.0, viennacl::context(viennacl::ocl::get_context(context_id)));

  ViennaCLOpenCLBackend_impl my_opencl_backend_impl;
  my_opencl_backend_impl.context_id = context_id;
  ViennaCLOpenCLBackend my_opencl_backend = &my_opencl_backend_impl;

  ViennaCLOpenCLSswap(my_opencl_backend, size/2,
                      viennacl::traits::opencl_handle(opencl_x).get(), 1, 2,
                      viennacl::traits::opencl_handle(opencl_y).get(), 1, 2);

  ViennaCLOpenCLSscal(my_opencl_backend, size,
                      2.0f,
                      viennacl::traits::opencl_handle(opencl_x).get(), 0, 1);

  std::cout << " --- OpenCL ---" << std::endl;
  std::cout << "opencl_x: " << opencl_x << std::endl;
  std::cout << "opencl_y: " << opencl_y << std::endl;
#endif

  //
  //  That's it.
  //
  std::cout << "!!!! TEST COMPLETED SUCCESSFULLY !!!!" << std::endl;

  return EXIT_SUCCESS;
}

