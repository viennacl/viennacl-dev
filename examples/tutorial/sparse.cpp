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

/** \example sparse.cpp
*
*   This tutorial demonstrates the use of sparse matrices.
*   The primary operation for sparse matrices in ViennaCL is the sparse matrix-vector product.
*
*   We start with including the respective headers:
**/

// system headers
#include <iostream>

// ublas headers
#include <boost/numeric/ublas/matrix_sparse.hpp>
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/operation.hpp>
#include <boost/numeric/ublas/operation_sparse.hpp>
#include <boost/numeric/ublas/io.hpp>

// Must be set if you want to use ViennaCL algorithms on ublas objects
#define VIENNACL_WITH_UBLAS 1

// ViennaCL includes
#include "viennacl/scalar.hpp"
#include "viennacl/vector.hpp"
#include "viennacl/compressed_matrix.hpp"
#include "viennacl/linalg/prod.hpp"


// Additional helper functions for this tutorial:
#include "vector-io.hpp"

// Shortcut for writing 'ublas::' instead of 'boost::numeric::ublas::'
using namespace boost::numeric;

/**
*   We setup a sparse matrix in uBLAS and populate it with values.
*   Then, the respective ViennaCL sparse matrix is created and initialized with data from the uBLAS matrix.
*   After a direct manipulation of the ViennaCL matrix, matrix-vector products are computed with both matrices.
**/
int main()
{
  typedef float       ScalarType;

  std::size_t size = 5;

  /**
  * Set up some ublas objects
  **/
  ublas::vector<ScalarType> rhs = ublas::scalar_vector<ScalarType>(size, ScalarType(size));
  ublas::compressed_matrix<ScalarType> ublas_matrix(size, size);

  ublas_matrix(0,0) =  2.0f; ublas_matrix(0,1) = -1.0f;
  ublas_matrix(1,0) = -1.0f; ublas_matrix(1,1) =  2.0f; ublas_matrix(1,2) = -1.0f;
  ublas_matrix(2,1) = -1.0f; ublas_matrix(2,2) =  2.0f; ublas_matrix(2,3) = -1.0f;
  ublas_matrix(3,2) = -1.0f; ublas_matrix(3,3) =  2.0f; ublas_matrix(3,4) = -1.0f;
  ublas_matrix(4,3) = -1.0f; ublas_matrix(4,4) =  2.0f;

  std::cout << "ublas matrix: " << ublas_matrix << std::endl;

  /**
  * Set up some ViennaCL objects and initialize with data from uBLAS objects
  **/
  viennacl::vector<ScalarType> vcl_rhs(size);
  viennacl::compressed_matrix<ScalarType> vcl_compressed_matrix(size, size);

  viennacl::copy(rhs, vcl_rhs);
  viennacl::copy(ublas_matrix, vcl_compressed_matrix);

  // just get the data directly from the GPU and print it:
  ublas::compressed_matrix<ScalarType> temp(size, size);
  viennacl::copy(vcl_compressed_matrix, temp);
  std::cout << "ViennaCL: " << temp << std::endl;

  // now modify GPU data directly:
  std::cout << "Modifying vcl_compressed_matrix a bit: " << std::endl;
  vcl_compressed_matrix(0, 0) =  3.0f;
  vcl_compressed_matrix(2, 3) = -3.0f;
  vcl_compressed_matrix(4, 2) = -3.0f;  //this is a new nonzero entry
  vcl_compressed_matrix(4, 3) = -3.0f;

  // and print it again:
  viennacl::copy(vcl_compressed_matrix, temp);
  std::cout << "ViennaCL matrix copied to uBLAS matrix: " << temp << std::endl;

  /**
  *  Compute matrix-vector products and output the results (should match):
  **/
  std::cout << "ublas: " << ublas::prod(temp, rhs) << std::endl;
  std::cout << "ViennaCL: " << viennacl::linalg::prod(vcl_compressed_matrix, vcl_rhs) << std::endl;

  /**
  *  That's it. Print a success message and exit.
  **/
  std::cout << "!!!! TUTORIAL COMPLETED SUCCESSFULLY !!!!" << std::endl;

  return EXIT_SUCCESS;
}

