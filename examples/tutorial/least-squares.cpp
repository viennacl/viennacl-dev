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

/** \example least-squares.cpp
*
*   This tutorial shows how least Squares problems for matrices from ViennaCL or Boost.uBLAS can be solved solved.
*
*   We start with including the respective header files:
**/

// activate ublas support in ViennaCL
#define VIENNACL_WITH_UBLAS

//
// include necessary system headers
//
#include <iostream>

// Boost headers
#include <boost/numeric/ublas/triangular.hpp>
#include <boost/numeric/ublas/vector.hpp>
#include <boost/numeric/ublas/vector_proxy.hpp>
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/matrix_proxy.hpp>
#include <boost/numeric/ublas/lu.hpp>
#include <boost/numeric/ublas/io.hpp>


// ViennaCL headers
#include "viennacl/vector.hpp"
#include "viennacl/matrix.hpp"
#include "viennacl/matrix_proxy.hpp"
#include "viennacl/linalg/qr.hpp"
#include "viennacl/linalg/lu.hpp"
#include "viennacl/linalg/direct_solve.hpp"


/**
*  The minimization problem of finding x such that \f$ \Vert Ax - b \Vert \f$ is solved as follows:
*   - Compute the QR-factorization of A = QR.
*   - Compute \f$ b' = Q^{\mathrm{T}} b \f$ for the equivalent minimization problem \f$ \Vert Rx - Q^{\mathrm{T}} b \f$.
*   - Solve the triangular system \f$ \tilde{R} x = b' \f$, where \f$ \tilde{R} \f$ is the upper square matrix of R.
*
**/
int main (int, const char **)
{
  typedef float               ScalarType;     //feel free to change this to 'double' if supported by your hardware

  typedef boost::numeric::ublas::matrix<ScalarType>              MatrixType;
  typedef boost::numeric::ublas::vector<ScalarType>              VectorType;
  typedef viennacl::matrix<ScalarType, viennacl::column_major>   VCLMatrixType;
  typedef viennacl::vector<ScalarType>                           VCLVectorType;

  /**
  *  Create vectors and matrices with data:
  **/
  VectorType ublas_b(4);
  ublas_b(0) = -4;
  ublas_b(1) =  2;
  ublas_b(2) =  5;
  ublas_b(3) = -1;

  MatrixType ublas_A(4, 3);

  ublas_A(0, 0) =  2; ublas_A(0, 1) = -1; ublas_A(0, 2) =  1;
  ublas_A(1, 0) =  1; ublas_A(1, 1) = -5; ublas_A(1, 2) =  2;
  ublas_A(2, 0) = -3; ublas_A(2, 1) =  1; ublas_A(2, 2) = -4;
  ublas_A(3, 0) =  1; ublas_A(3, 1) = -1; ublas_A(3, 2) =  1;

  /**
  * Setup the matrix and vector with ViennaCL objects and copy the data from the uBLAS objects:
  **/
  VCLVectorType vcl_b(ublas_b.size());
  VCLMatrixType vcl_A(ublas_A.size1(), ublas_A.size2());

  viennacl::copy(ublas_b, vcl_b);
  viennacl::copy(ublas_A, vcl_A);


  /**
  * <h2>Option 1: Using Boost.uBLAS</h2>
  *
  * The implementation in ViennaCL accepts both uBLAS and ViennaCL types.
  * We start with a single-threaded implementation using Boost.uBLAS.
  **/

  std::cout << "--- Boost.uBLAS ---" << std::endl;
  /**
  * The first (and computationally most expensive) step is to compute the QR factorization of A.
  * Since we do not need A later, we directly overwrite A with the householder reflectors and the upper triangular matrix R.
  * The returned vector holds the scalar coefficients (betas) for the Householder reflections \f$ I - \beta v v^{\mathrm{T}} \f$
  **/
  std::vector<ScalarType> ublas_betas = viennacl::linalg::inplace_qr(ublas_A);

  /**
  * Compute the modified RHS of the minimization problem from the QR factorization, but do not form \f$ Q^{\mathrm{T}} \f$ explicitly:
  * b' := Q^T b
  **/
  viennacl::linalg::inplace_qr_apply_trans_Q(ublas_A, ublas_betas, ublas_b);

  /**
  * Final step: triangular solve: Rx = b'', where b'' are the first three entries in b'
  * We only need the upper left square part of A, which defines the upper triangular matrix R
  **/
  boost::numeric::ublas::range ublas_range(0, 3);
  boost::numeric::ublas::matrix_range<MatrixType> ublas_R(ublas_A, ublas_range, ublas_range);
  boost::numeric::ublas::vector_range<VectorType> ublas_b2(ublas_b, ublas_range);
  boost::numeric::ublas::inplace_solve(ublas_R, ublas_b2, boost::numeric::ublas::upper_tag());

  std::cout << "Result: " << ublas_b2 << std::endl;

  /**
  *  <h2>Option 2: Use ViennaCL types</h2>
  *
  *  ViennaCL is used for the computationally intensive BLAS 3 computations.
  *  Boost.uBLAS is used for the panel factorization on the host (CPU).
  */

  std::cout << "--- ViennaCL (hybrid implementation)  ---" << std::endl;
  std::vector<ScalarType> hybrid_betas = viennacl::linalg::inplace_qr(vcl_A);

  /**
  * compute modified RHS of the minimization problem: \f$ b' := Q^T b \f$
  **/
  viennacl::linalg::inplace_qr_apply_trans_Q(vcl_A, hybrid_betas, vcl_b);

  /**
  * Final step: triangular solve: Rx = b'.
  * We only need the upper part of A such that R is a square matrix
  **/
  viennacl::range vcl_range(0, 3);
  viennacl::matrix_range<VCLMatrixType> vcl_R(vcl_A, vcl_range, vcl_range);
  viennacl::vector_range<VCLVectorType> vcl_b2(vcl_b, vcl_range);
  viennacl::linalg::inplace_solve(vcl_R, vcl_b2, viennacl::linalg::upper_tag());

  std::cout << "Result: " << vcl_b2 << std::endl;

  /**
  *  That's it.
  **/
  std::cout << "!!!! TUTORIAL COMPLETED SUCCESSFULLY !!!!" << std::endl;

  return EXIT_SUCCESS;
}

