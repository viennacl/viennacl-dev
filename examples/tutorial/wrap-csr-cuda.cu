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

/*
*
*   Tutorial: How to wrap user-provided CSR sparse matrix data available on the host or in CUDA buffers in a viennacl::compressed_matrix<>
*
*/


//
// include necessary system headers
//
#include <iostream>
#include <cstdlib>
#include <string>

#include <cuda.h>

//
// ViennaCL includes
//
#include "viennacl/vector.hpp"
#include "viennacl/compressed_matrix.hpp"
#include "viennacl/linalg/prod.hpp"


int main()
{
  typedef double       ScalarType;

  //
  // Part 1: Allocate CUDA memory for representing the following 6-by-6 sparse matrix:
  //
  //  (  4 -1  0 -1  0  0 )
  //  ( -1  4 -1  0 -1  0 )
  //  (  0 -1  4  0  0 -1 )
  //  ( -1  0  0  4 -1  0 )
  //  (  0 -1  0 -1  4 -1 )
  //  (  0  0 -1  0 -1  4 )
  //
  std::size_t N = 6;
  std::size_t nonzeros = 20;
  //                             row 0,          row 1,               row 2,         row 3,          row 4,               row 5           end of matrix
  unsigned int row_start[]   = { 0,              3,                   7,             10,             13,                  17,             20};
  unsigned int col_indices[] = { 0,  1,   3,     0,    1,  2,   4,    1,  2,   5,     0,  3,   4,     1,   3,  4,   5,     2,   4,  5};
  ScalarType   entries[]     = {4., -1., -1.,   -1.,  4., -1., -1.,  -1., 4., -1.,   -1., 4., -1.,   -1., -1., 4., -1.,   -1., -1., 4.};

  // CSR arrays (note that ViennaCL expects 'unsigned int', not 'unsigned long')
  unsigned int *cuda_row_start;
  unsigned int *cuda_col_indices;
  ScalarType   *cuda_entries;

  cudaMalloc(&cuda_row_start,   (N+1)    * sizeof(unsigned int));
  cudaMalloc(&cuda_col_indices, nonzeros * sizeof(unsigned int));
  cudaMalloc(&cuda_entries,     nonzeros * sizeof(ScalarType));

  // copy data from host
  cudaMemcpy(cuda_row_start,     row_start, (N+1)    * sizeof(unsigned int), cudaMemcpyHostToDevice);
  cudaMemcpy(cuda_col_indices, col_indices, nonzeros * sizeof(unsigned int), cudaMemcpyHostToDevice);
  cudaMemcpy(cuda_entries,         entries, nonzeros * sizeof(ScalarType),   cudaMemcpyHostToDevice);



  //
  // Part 1: Wrap CUDA buffers with ViennaCL types
  //

  // wrap the existing CUDA CSR data in a viennacl::compressed_matrix:
  viennacl::compressed_matrix<ScalarType> vcl_A_cuda(cuda_row_start, cuda_col_indices, cuda_entries, viennacl::CUDA_MEMORY, N, N, nonzeros);

  // perform a matrix-vector product for demonstration purposes:
  viennacl::vector<ScalarType> vcl_x_cuda = viennacl::scalar_vector<ScalarType>(N, 1.0, viennacl::traits::context(vcl_A_cuda));
  viennacl::vector<ScalarType> vcl_y_cuda = viennacl::linalg::prod(vcl_A_cuda, vcl_x_cuda);

  std::cout << "Result of matrix-vector product with CUDA: " << std::endl;
  std::cout << vcl_y_cuda << std::endl;


  // ViennaCL does not automatically free your buffers (you're still the owner), so don't forget to clean up :-)
  cudaFree(cuda_row_start);
  cudaFree(cuda_col_indices);
  cudaFree(cuda_entries);


  //
  // Part 2: Wrap host buffers with ViennaCL types
  //

  // wrap the existing CSR data on the host in a viennacl::compressed_matrix:
  viennacl::compressed_matrix<ScalarType> vcl_A_host(row_start, col_indices, entries, viennacl::MAIN_MEMORY, N, N, nonzeros);

  // perform a matrix-vector product for demonstration purposes:
  viennacl::vector<ScalarType> vcl_x_host = viennacl::scalar_vector<ScalarType>(N, 1.0, viennacl::traits::context(vcl_A_host));
  viennacl::vector<ScalarType> vcl_y_host = viennacl::linalg::prod(vcl_A_host, vcl_x_host);

  std::cout << "Result of matrix-vector product wrapping host-buffers: " << std::endl;
  std::cout << vcl_y_host << std::endl;



  //
  //  That's it.
  //
  std::cout << "!!!! TUTORIAL COMPLETED SUCCESSFULLY !!!!" << std::endl;

  return EXIT_SUCCESS;
}

