/* =========================================================================
   Copyright (c) 2010-2014, Institute for Microelectronics,
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

/** \example "Eigenvalues: Lanczos' Method"
*
*   This tutorial shows how to calculate the largest eigenvalues of a matrix using Lanczos' method.
*
*   The Lanczos method is particularly attractive for use with large, sparse matrices, since the only requirement on the matrix is to provide a matrix-vector product.
*   Although less common, the method is sometimes also with dense matrices.
*
*   We start with including the necessary headers:
**/

// include necessary system headers
#include <iostream>

#ifndef NDEBUG
  #define NDEBUG
#endif

#define VIENNACL_WITH_UBLAS

//include basic scalar and vector types of ViennaCL
#include "viennacl/scalar.hpp"
#include "viennacl/vector.hpp"
#include "viennacl/compressed_matrix.hpp"

#include "viennacl/linalg/lanczos.hpp"
#include "viennacl/io/matrix_market.hpp"

// Some helper functions for this tutorial:
#include <iostream>
#include <fstream>
#include <limits>
#include <string>
#include <iomanip>
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/matrix_proxy.hpp>
#include <boost/numeric/ublas/matrix_expression.hpp>
#include <boost/numeric/ublas/matrix_sparse.hpp>
#include <boost/numeric/ublas/vector.hpp>
#include <boost/numeric/ublas/operation.hpp>
#include <boost/numeric/ublas/vector_expression.hpp>


/**
*  We read a sparse matrix (from Boost.uBLAS) from a matrix-market file, then run the Lanczos method.
*  Finally, the computed eigenvalues are printed.
**/
int main()
{
  // If you GPU does not support double precision, use `float` instead of `double`:
  typedef double     ScalarType;

  /**
  *  Create the uBLAS-matrix and read the sparse matrix:
  **/
  boost::numeric::ublas::compressed_matrix<ScalarType> ublas_A;
  if (!viennacl::io::read_matrix_market_file(ublas_A, "../examples/testdata/mat65k.mtx"))
  {
    std::cout << "Error reading Matrix file" << std::endl;
    return EXIT_FAILURE;
  }

  /**
  *  Create the configuration for the Lanczos method.
  **/
  viennacl::linalg::lanczos_tag ltag(0.75,    // Select a power of 0.75 as the tolerance for the machine precision.
                                     10,      // Compute (approximations to) the 10 largest eigenvalues
                                     viennacl::linalg::lanczos_tag::partial_reorthogonalization, // use partial reorthogonalization
                                     1700);   // Maximum size of the Krylov space

  /**
  *  Run the Lanczos method by passing the tag to the routine viennacl::linalg::eig()
  **/
  std::cout << "Running Lanczos algorithm (this might take a while)..." << std::endl;
  std::vector<double> lanczos_eigenvalues = viennacl::linalg::eig(ublas_A, ltag);

  /**
  *  Print the computed eigenvalues and exit:
  **/
  for (std::size_t i = 0; i< lanczos_eigenvalues.size(); i++)
    std::cout << "Eigenvalue " << i+1 << ": " << std::setprecision(10) << lanczos_eigenvalues[i] << std::endl;

  return EXIT_SUCCESS;
}

