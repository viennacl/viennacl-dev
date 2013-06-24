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
*   Tutorial: QR factorization of matrices from ViennaCL or Boost.uBLAS (qr.cpp and qr.cu are identical, the latter being required for compilation using CUDA nvcc)
*
*/

// activate ublas support in ViennaCL
#define VIENNACL_WITH_UBLAS

//
// include necessary system headers
//
#include <iostream>

//
// ViennaCL includes: We only need the qr-header
//
#include "viennacl/linalg/qr.hpp"

//
// Boost includes
//
#include <boost/numeric/ublas/vector.hpp>
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/io.hpp>


//
// Testing
//
#include "viennacl/range.hpp"
#include "viennacl/linalg/prod.hpp"
#include "viennacl/matrix.hpp"
#include "viennacl/matrix_proxy.hpp"


//
// A helper function checking the result
//
template <typename MatrixType>
double check(MatrixType const & qr, MatrixType const & ref)
{
  bool do_break = false;
  double max_error = 0;
  for (std::size_t i=0; i<ref.size1(); ++i)
  {
    for (std::size_t j=0; j<ref.size2(); ++j)
    {
      if (qr(i,j) != 0.0 && ref(i,j) != 0.0)
      {
        double rel_err = fabs(qr(i,j) - ref(i,j)) / fabs(ref(i,j) );

        if (rel_err > max_error)
          max_error = rel_err;
      }


      if (qr(i,j) != qr(i,j))
      {
        std::cout << "!!!" << std::endl;
        std::cout << "!!! NaN detected at i=" << i << " and j=" << j << std::endl;
        std::cout << "!!!" << std::endl;
        do_break = true;
        break;
      }
    }
    if (do_break)
      break;
  }
  return max_error;
}


int main (int, const char **)
{
  typedef double               ScalarType;     //feel free to change this to 'double' if supported by your hardware
  typedef boost::numeric::ublas::matrix<ScalarType>        MatrixType;
  typedef boost::numeric::ublas::vector<ScalarType>        VectorType;
  typedef viennacl::matrix<ScalarType, viennacl::column_major>        VCLMatrixType;
  typedef viennacl::vector<ScalarType>        VCLVectorType;

  std::size_t rows = 113;   //number of rows in the matrix
  std::size_t cols = 54;   //number of columns

  //
  // Create matrices with some data
  //
  MatrixType ublas_A(rows, cols);
  MatrixType Q(rows, rows);
  MatrixType R(rows, cols);

  // Some random data with a bit of extra weight on the diagonal
  for (std::size_t i=0; i<rows; ++i)
  {
    for (std::size_t j=0; j<cols; ++j)
    {
      ublas_A(i,j) = -1.0 + (i+1)*(j+1)
                     + ( (rand() % 1000) - 500.0) / 1000.0;

      if (i == j)
        ublas_A(i,j) += 10.0;

      R(i,j) = 0.0;
    }

    for (std::size_t j=0; j<rows; ++j)
      Q(i,j) = 0.0;
  }

  // keep initial input matrix for comparison
  MatrixType ublas_A_backup(ublas_A);


  //
  // Setup the matrix in ViennaCL:
  //
  VCLVectorType dummy(10);
  VCLMatrixType vcl_A(ublas_A.size1(), ublas_A.size2());

  viennacl::copy(ublas_A, vcl_A);

  //
  // Compute QR factorization of A. A is overwritten with Householder vectors. Coefficients are returned and a block size of 3 is used.
  // Note that at the moment the number of columns of A must be divisible by the block size
  //

  std::cout << "--- Boost.uBLAS ---" << std::endl;
  std::vector<ScalarType> ublas_betas = viennacl::linalg::inplace_qr(ublas_A);  //computes the QR factorization

  //
  // A check for the correct result:
  //
  viennacl::linalg::recoverQ(ublas_A, ublas_betas, Q, R);
  MatrixType ublas_QR = prod(Q, R);
  double ublas_error = check(ublas_QR, ublas_A_backup);
  std::cout << "Max rel error (ublas): " << ublas_error << std::endl;

  //
  // QR factorization in ViennaCL using Boost.uBLAS for the panel factorization
  //
  std::cout << "--- Hybrid (default) ---" << std::endl;
  viennacl::copy(ublas_A_backup, vcl_A);
  std::vector<ScalarType> hybrid_betas = viennacl::linalg::inplace_qr(vcl_A);


  //
  // A check for the correct result:
  //
  viennacl::copy(vcl_A, ublas_A);
  Q.clear(); R.clear();
  viennacl::linalg::recoverQ(ublas_A, hybrid_betas, Q, R);
  double hybrid_error = check(ublas_QR, ublas_A_backup);
  std::cout << "Max rel error (hybrid): " << hybrid_error << std::endl;


  //
  //  That's it.
  //
  std::cout << "!!!! TUTORIAL COMPLETED SUCCESSFULLY !!!!" << std::endl;

  return EXIT_SUCCESS;
}

