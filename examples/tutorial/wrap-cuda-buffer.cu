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
*   Tutorial:  Use ViennaCL within user-defined (i.e. your own) OpenCL contexts
*
*/


//
// include necessary system headers
//
#include <iostream>
#include <string>

#include <cuda.h>

//
// ViennaCL includes
//
#include "viennacl/vector.hpp"
#include "viennacl/matrix.hpp"
#include "viennacl/linalg/matrix_operations.hpp"
#include "viennacl/linalg/norm_2.hpp"
#include "viennacl/linalg/prod.hpp"


// Some helper functions for this tutorial:
#include "Random.hpp"

template <typename T>
__global__ void my_inplace_add_kernel(T * vec1, T * vec2, unsigned int size)
{
    for (unsigned int i = blockDim.x * blockIdx.x + threadIdx.x;
                      i < size;
                      i += gridDim.x * blockDim.x)
      vec1[i] += vec2[i];
}



int main()
{
  typedef float       ScalarType;

  //
  // Part 1: Allocate some CUDA memory
  //
  std::size_t size = 10;
  ScalarType *cuda_x;
  ScalarType *cuda_y;

  cudaMalloc(&cuda_x, size * sizeof(ScalarType));
  cudaMalloc(&cuda_y, size * sizeof(ScalarType));

  std::vector<ScalarType> host_x(size, 1.0);
  std::vector<ScalarType> host_y(size, 2.0);

  cudaMemcpy(cuda_x, &(host_x[0]), size * sizeof(ScalarType), cudaMemcpyHostToDevice);
  cudaMemcpy(cuda_y, &(host_y[0]), size * sizeof(ScalarType), cudaMemcpyHostToDevice);

  my_inplace_add_kernel<<<128, 128>>>(cuda_x, cuda_y, static_cast<unsigned int>(1000));

  // check result
  std::vector<ScalarType> result_cuda(size);
  cudaMemcpy(&(result_cuda[0]), cuda_x, size * sizeof(ScalarType), cudaMemcpyDeviceToHost);

  std::cout << "result_cuda: ";
  for (std::size_t i=0; i<size; ++i)
    std::cout << result_cuda[i] << " ";
  std::cout << std::endl;

  //
  // Part 2: Now do the same within ViennaCL
  //

  viennacl::vector<ScalarType> vcl_vec1(cuda_x, size, viennacl::CUDA_MEMORY); // Third parameter specifies that this is CUDA memory rather than host memory
  viennacl::vector<ScalarType> vcl_vec2(cuda_y, size, viennacl::CUDA_MEMORY); // Third parameter specifies that this is CUDA memory rather than host memory

  vcl_vec1 = viennacl::scalar_vector<ScalarType>(size, ScalarType(1.0));
  vcl_vec2 = viennacl::scalar_vector<ScalarType>(size, ScalarType(2.0));

  vcl_vec1 += vcl_vec2;

  std::cout << "Result with ViennaCL: ";
  std::cout << vcl_vec1 << std::endl;

  //
  //  That's it.
  //
  std::cout << "!!!! TUTORIAL COMPLETED SUCCESSFULLY !!!!" << std::endl;

  return 0;
}

