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
*   Tutorial: Least Squares problem for matrices from ViennaCL or Boost.uBLAS (least-squares.cpp and least-squares.cu are identical, the latter being required for compilation using CUDA nvcc)
* 
*   See Example 2 at http://tutorial.math.lamar.edu/Classes/LinAlg/QRDecomposition.aspx for a reference solution.
*
*/

// activate ublas support in ViennaCL
#define VIENNACL_WITH_UBLAS 

//
// include necessary system headers
//
#include <iostream>

//
// Boost includes
//
#include <boost/numeric/ublas/vector.hpp>
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/matrix_proxy.hpp>
#include <boost/numeric/ublas/lu.hpp>
#include <boost/numeric/ublas/io.hpp>


//
// ViennaCL includes
//
#include "viennacl/vector.hpp"
#include "viennacl/matrix.hpp"
#include "viennacl/matrix_proxy.hpp"
#include "viennacl/linalg/qr.hpp"
#include "viennacl/linalg/lu.hpp"


/*
*   Tutorial: Least Squares problem for matrices from ViennaCL or Boost.uBLAS
* 
*   See Example 2 at http://tutorial.math.lamar.edu/Classes/LinAlg/QRDecomposition.aspx for a reference solution.
*/

int main (int, const char **)
{
  typedef float               ScalarType;     //feel free to change this to 'double' if supported by your hardware
  typedef boost::numeric::ublas::matrix<ScalarType>              MatrixType;
  typedef boost::numeric::ublas::vector<ScalarType>              VectorType;
  typedef viennacl::matrix<ScalarType, viennacl::column_major>   VCLMatrixType;
  typedef viennacl::vector<ScalarType>                           VCLVectorType;

  //
  // Create vectors and matrices with data, cf. http://tutorial.math.lamar.edu/Classes/LinAlg/QRDecomposition.aspx
  //
  VectorType ublas_b(4);
  ublas_b(0) = -4;
  ublas_b(1) =  2;
  ublas_b(2) =  5;
  ublas_b(3) = -1;
  
  MatrixType ublas_A(4, 3);
  MatrixType Q = boost::numeric::ublas::zero_matrix<ScalarType>(4, 4);
  MatrixType R = boost::numeric::ublas::zero_matrix<ScalarType>(4, 3);
  
  ublas_A(0, 0) =  2; ublas_A(0, 1) = -1; ublas_A(0, 2) =  1;
  ublas_A(1, 0) =  1; ublas_A(1, 1) = -5; ublas_A(1, 2) =  2;
  ublas_A(2, 0) = -3; ublas_A(2, 1) =  1; ublas_A(2, 2) = -4;
  ublas_A(3, 0) =  1; ublas_A(3, 1) = -1; ublas_A(3, 2) =  1;
  
  //
  // Setup the matrix in ViennaCL:
  //
  VCLVectorType vcl_b(ublas_b.size());
  VCLMatrixType vcl_A(ublas_A.size1(), ublas_A.size2());
  
  viennacl::copy(ublas_b, vcl_b);
  viennacl::copy(ublas_A, vcl_A);
  

  
  //////////// Part 1: Use Boost.uBLAS for all computations ////////////////
  
  std::cout << "--- Boost.uBLAS ---" << std::endl;
  std::vector<ScalarType> ublas_betas = viennacl::linalg::inplace_qr(ublas_A);  //computes the QR factorization
  
  // compute modified RHS of the minimization problem: 
  // b' := Q^T b
  viennacl::linalg::inplace_qr_apply_trans_Q(ublas_A, ublas_betas, ublas_b);
  
  // Final step: triangular solve: Rx = b'', where b'' are the first three entries in b'
  // We only need the upper left square part of A, which defines the upper triangular matrix R
  boost::numeric::ublas::range ublas_range(0, 3);
  boost::numeric::ublas::matrix_range<MatrixType> ublas_R(ublas_A, ublas_range, ublas_range);
  boost::numeric::ublas::vector_range<VectorType> ublas_b2(ublas_b, ublas_range);
  boost::numeric::ublas::inplace_solve(ublas_R, ublas_b2, boost::numeric::ublas::upper_tag());

  std::cout << "Result: " << ublas_b2 << std::endl;
  
  //////////// Part 2: Use ViennaCL types for BLAS 3 computations, but use Boost.uBLAS for the panel factorization ////////////////
  
  std::cout << "--- ViennaCL (hybrid implementation)  ---" << std::endl;
  std::vector<ScalarType> hybrid_betas = viennacl::linalg::inplace_qr(vcl_A);
  
  // compute modified RHS of the minimization problem: 
  // b := Q^T b
  viennacl::linalg::inplace_qr_apply_trans_Q(vcl_A, hybrid_betas, vcl_b);

  // Final step: triangular solve: Rx = b'.
  // We only need the upper part of A such that R is a square matrix
  viennacl::range vcl_range(0, 3);
  viennacl::matrix_range<VCLMatrixType> vcl_R(vcl_A, vcl_range, vcl_range);
  viennacl::vector_range<VCLVectorType> vcl_b2(vcl_b, vcl_range);
  viennacl::linalg::inplace_solve(vcl_R, vcl_b2, viennacl::linalg::upper_tag());

  std::cout << "Result: " << vcl_b2 << std::endl;
  
  
  
  //
  //  That's it.
  //
  std::cout << "!!!! TUTORIAL COMPLETED SUCCESSFULLY !!!!" << std::endl;

  return EXIT_SUCCESS;
}

