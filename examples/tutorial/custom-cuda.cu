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

/** \example "CUDA: User-Provided Memory and Kernels"
*
*   This tutorial shows how you can use your own CUDA buffers and CUDA kernels with ViennaCL.
*   We demonstrate this for simple vector and matrix-vector operations.
*   For simplicity, error-checks of CUDA API calls are omitted.
*
*   We begin with including the necessary headers:
**/

// System headers
#include <iostream>
#include <string>

#ifndef VIENNACL_WITH_CUDA
  #define VIENNACL_WITH_CUDA
#endif


// ViennaCL headers
#include "viennacl/vector.hpp"
#include "viennacl/matrix.hpp"
#include "viennacl/linalg/matrix_operations.hpp"
#include "viennacl/linalg/norm_2.hpp"
#include "viennacl/linalg/prod.hpp"


/** <h2>Defining a Compute Kernel</h2>
*
* In the following we define a custom compute kernel which computes an elementwise product of two vectors. <br />
* Input: v1 ... vector<br />
*        v2 ... vector<br />
* Output: result ... vector<br />
*
* Algorithm: set result[i] <- v1[i] * v2[i]<br />
*            (in MATLAB notation this is 'result = v1 .* v2');<br />
**/

template<typename NumericT>
__global__ void elementwise_multiplication(const NumericT * vec1,
                                           const NumericT * vec2,
                                                 NumericT * result,
                                           unsigned int size)
{
  for (unsigned int i = blockDim.x * blockIdx.x + threadIdx.x;
                    i < size;
                    i += gridDim.x * blockDim.x)
    result[i] = vec1[i] * vec2[i];
}

/**
* With this let us go right to main():
**/
int main()
{
  typedef float       NumericType;


  /**
  * <h2>Part 1: Set up custom CUDA buffers</h2>
  *
  * The following is rather lengthy because OpenCL is a fairly low-level framework.
  * For comparison, the subsequent code explicitly performs the OpenCL setup that is done
  * in the background within the 'custom_kernels'-tutorial
  **/

  //manually set up a custom OpenCL context:
  std::size_t N = 5;

  NumericType * device_vec1;
  NumericType * device_vec2;
  NumericType * device_result;
  NumericType * device_A;

  cudaMalloc(&device_vec1,   N * sizeof(NumericType));
  cudaMalloc(&device_vec2,   N * sizeof(NumericType));
  cudaMalloc(&device_result, N * sizeof(NumericType));
  cudaMalloc(&device_A,  N * N * sizeof(NumericType));

  // fill vectors and matrix with data:
  std::vector<NumericType> temp(N);
  for (std::size_t i=0; i<temp.size(); ++i)
    temp[i] = NumericType(i);
  cudaMemcpy(device_vec1, &(temp[0]), temp.size() * sizeof(NumericType), cudaMemcpyHostToDevice);

  for (std::size_t i=0; i<temp.size(); ++i)
    temp[i] = NumericType(2*i);
  cudaMemcpy(device_vec2, &(temp[0]), temp.size() * sizeof(NumericType), cudaMemcpyHostToDevice);

  temp.resize(N*N);
  for (std::size_t i=0; i<temp.size(); ++i)
    temp[i] = NumericType(i);
  cudaMemcpy(device_A, &(temp[0]), temp.size() * sizeof(NumericType), cudaMemcpyHostToDevice);


  /**
  * <h2>Part 2: Reuse Custom CUDA buffers with ViennaCL</h2>
  *
  * Wrap existing OpenCL objects into ViennaCL:
  **/
  viennacl::vector<NumericType> vcl_vec1(device_vec1, viennacl::CUDA_MEMORY, N);
  viennacl::vector<NumericType> vcl_vec2(device_vec2, viennacl::CUDA_MEMORY, N);
  viennacl::vector<NumericType> vcl_result(device_result, viennacl::CUDA_MEMORY, N);

  std::cout << "Standard vector operations within ViennaCL:" << std::endl;
  vcl_result = NumericType(3.1415) * vcl_vec1 + vcl_vec2;

  std::cout << "vec1  : " << vcl_vec1 << std::endl;
  std::cout << "vec2  : " << vcl_vec2 << std::endl;
  std::cout << "result: " << vcl_result << std::endl;

  /**
  * We can also reuse the existing elementwise_prod kernel.
  * Therefore, we first have to make the existing program known to ViennaCL
  * For more details on the three lines, see tutorial 'custom-kernels'
  **/
  std::cout << "Using existing kernel within ViennaCL:" << std::endl;
  elementwise_multiplication<<<128, 128>>>(viennacl::cuda_arg(vcl_vec1),
                                           viennacl::cuda_arg(vcl_vec2),
                                           viennacl::cuda_arg(vcl_result),
                                           N);

  std::cout << "vec1  : " << vcl_vec1 << std::endl;
  std::cout << "vec2  : " << vcl_vec2 << std::endl;
  std::cout << "result: " << vcl_result << std::endl;


  /**
  * Since a linear piece of memory can be interpreted in several ways,
  * we will now create a 5x5 row-major matrix out of the linear memory in device_A
  * The entries in vcl_vec2 and vcl_result are used to carry out matrix-vector products:
  **/
  viennacl::matrix<NumericType> vcl_matrix(device_A, viennacl::CUDA_MEMORY,
                                           N, // number of rows.
                                           N);// number of colums.

  vcl_result = viennacl::linalg::prod(vcl_matrix, vcl_vec2);

  std::cout << "result of matrix-vector product: ";
  std::cout << vcl_result << std::endl;

  /**
  *  Any further operations can be carried out in the same way.
  *  Just keep in mind that any resizing of vectors or matrices leads to a reallocation of the underlying memory buffer, through which the 'wrapper' is lost.
  **/
  std::cout << "!!!! TUTORIAL COMPLETED SUCCESSFULLY !!!!" << std::endl;

  return EXIT_SUCCESS;
}

