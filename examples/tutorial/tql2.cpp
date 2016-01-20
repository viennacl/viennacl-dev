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

/** \example tql2.cpp
*
*   This tutorial explains how one can use the tql-algorithm to compute the eigenvalues of tridiagonal matrices.
*
*   We start with including the necessary headers:
**/

// include necessary system headers
#include <iostream>
#include <fstream>
#include <limits>
#include <string>
#include <iomanip>

#define VIENNACL_WITH_UBLAS

//include basic scalar and vector types of ViennaCL
#include "viennacl/scalar.hpp"
#include "viennacl/vector.hpp"
#include "viennacl/compressed_matrix.hpp"
#include "viennacl/io/matrix_market.hpp"

#include "viennacl/linalg/qr-method.hpp"
#include "viennacl/linalg/qr-method-common.hpp"
#include "viennacl/linalg/host_based/matrix_operations.hpp"

// Use the shortcut 'ublas::' instead of 'boost::numeric::ublas::'
namespace ublas = boost::numeric::ublas;

// Run in single precision. Change to double precision if provided by your GPU.
typedef float     ScalarType;

/**
*  We generate a symmetric tridiagonal matrix with known eigenvalues, call the tql-algorithm, and then print the results.
**/
int main()
{
  std::size_t sz = 10;
  std::cout << "Compute eigenvalues and eigenvectors of matrix of size " << sz << "-by-" << sz << std::endl << std::endl;

  std::vector<ScalarType> d(sz), e(sz);
  // Initialize diagonal and superdiagonal elements of the tridiagonal matrix
  d[0] = 1; e[0] = 0;
  d[1] = 2; e[1] = 4;
  d[2] =-4; e[2] = 5;
  d[3] = 6; e[3] = 1;
  d[4] = 3; e[4] = 2;
  d[5] = 4; e[5] =-3;
  d[6] = 7; e[6] = 5;
  d[7] = 9; e[7] = 1;
  d[8] = 3; e[8] = 5;
  d[9] = 8; e[9] = 2;

  /**
  * Initialize the matrix Q as the identity matrix. It will hold the eigenvectors.
  **/
  viennacl::matrix<ScalarType> Q = viennacl::identity_matrix<ScalarType>(sz);

  /**
  * Compute the eigenvalues and eigenvectors
  **/
  viennacl::linalg::tql2(Q, d, e);

  /**
  * Print the results:
  **/
  std::cout << "Eigenvalues: " << std::endl;
  for (unsigned int i = 0; i < d.size(); i++)
    std::cout << std::setprecision(6) << std::fixed << d[i] << " ";
  std::cout << std::endl;

  std::cout << std::endl;
  std::cout << "Eigenvectors corresponding to the eigenvalues above are the columns: " << std::endl << std::endl;
  std::cout << Q << std::endl;

  /**
  * That's it. Print success message and exit.
  **/
  std::cout << std::endl <<"--------TUTORIAL COMPLETED----------" << std::endl;

  return EXIT_SUCCESS;
}
