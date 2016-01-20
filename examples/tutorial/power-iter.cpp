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

/** \example power-iter.cpp
*
*   This tutorial demonstrates the calculation of the eigenvalue with largest modulus using the power iteration method.
*
*   We start with including the necessary headers:
**/

// Sparse matrices in uBLAS are *very* slow if debug mode is enabled. Disable it:
#ifndef NDEBUG
  #define BOOST_UBLAS_NDEBUG
#endif

// Include necessary system headers
#include <iostream>
#include <fstream>
#include <limits>
#include <string>

#define VIENNACL_WITH_UBLAS

// Include basic scalar and vector types of ViennaCL
#include "viennacl/scalar.hpp"
#include "viennacl/vector.hpp"
#include "viennacl/compressed_matrix.hpp"

#include "viennacl/linalg/power_iter.hpp"
#include "viennacl/io/matrix_market.hpp"

// Some helper functions for this tutorial:
#include <iostream>

// Boost includes:
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/matrix_proxy.hpp>
#include <boost/numeric/ublas/matrix_expression.hpp>
#include <boost/numeric/ublas/matrix_sparse.hpp>
#include <boost/numeric/ublas/vector.hpp>
#include <boost/numeric/ublas/operation.hpp>
#include <boost/numeric/ublas/vector_expression.hpp>


/**
*  To run the power iteration, we set up a sparse matrix using Boost.uBLAS, transfer it over to a ViennaCL matrix, and then run the algorithm.
**/
int main()
{
  // This example relies on double precision to be available and will provide only poor results with single precision
  typedef double     ScalarType;

  /**
  * Create the sparse uBLAS matrix and read the matrix from the matrix-market file:
  **/
  boost::numeric::ublas::compressed_matrix<ScalarType> ublas_A;

  if (!viennacl::io::read_matrix_market_file(ublas_A, "../examples/testdata/mat65k.mtx"))
  {
    std::cout << "Error reading Matrix file" << std::endl;
    return EXIT_FAILURE;
  }

  /**
  *  Transfer the data from the uBLAS matrix over to the ViennaCL sparse matrix:
  **/
  viennacl::compressed_matrix<ScalarType>  vcl_A(ublas_A.size1(), ublas_A.size2());
  viennacl::copy(ublas_A, vcl_A);

  /**
  *  Run the power iteration up until the largest eigenvalue changes by less than the specified tolerance.
  *  Print the results of running with uBLAS as well as ViennaCL and exit.
  **/
  viennacl::linalg::power_iter_tag ptag(1e-6);

  std::cout << "Starting computation of eigenvalue with largest modulus (might take about a minute)..." << std::endl;
  std::cout << "Result of power iteration with ublas matrix (single-threaded): " << viennacl::linalg::eig(ublas_A, ptag) << std::endl;
  std::cout << "Result of power iteration with ViennaCL (OpenCL accelerated): " << viennacl::linalg::eig(vcl_A, ptag) << std::endl;

  /**
   *  You can also obtain the associated *approximated* eigenvector by passing it as a third argument to eig()
   *  Tighten the tolerance passed to ptag above in order to obtain more accurate results.
   **/
  viennacl::vector<ScalarType> eigenvector(vcl_A.size1());
  viennacl::linalg::eig(vcl_A, ptag, eigenvector);
  std::cout << "First three entries in eigenvector: " << eigenvector[0] << " " << eigenvector[1] << " " << eigenvector[2] << std::endl;
  viennacl::vector<ScalarType> Ax = viennacl::linalg::prod(vcl_A, eigenvector);
  std::cout << "First three entries in A*eigenvector: " << Ax[0] << " " << Ax[1] << " " << Ax[2] << std::endl;

  return EXIT_SUCCESS;
}

