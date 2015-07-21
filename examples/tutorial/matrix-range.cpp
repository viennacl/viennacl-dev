/* =========================================================================
   Copyright (c) 2010-2015, Institute for Microelectronics,
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

/** \example matrix-range.cpp
*
*   This tutorial explains the use of matrix ranges with simple BLAS level 1 and 2 operations.
*
*   We start with including the necessary headers:
**/


// activate ublas support in ViennaCL
#define VIENNACL_WITH_UBLAS

// System headers
#include <iostream>
#include <string>


// ViennaCL headers
#include "viennacl/scalar.hpp"
#include "viennacl/matrix.hpp"
#include "viennacl/linalg/prod.hpp"
#include "viennacl/matrix_proxy.hpp"


// Boost headers
#include "boost/numeric/ublas/vector.hpp"
#include "boost/numeric/ublas/matrix.hpp"
#include "boost/numeric/ublas/matrix_proxy.hpp"
#include "boost/numeric/ublas/io.hpp"

/**
*   In the main() routine we set up Boost.uBLAS as well as ViennaCL objects.
*   A few standard operations on submatrices are performed by using the matrix_range<> view available in both libraries.
**/
int main (int, const char **)
{
  // feel free to change this to 'double' if supported by your hardware
  typedef float                                           ScalarType;
  typedef boost::numeric::ublas::matrix<ScalarType>       MatrixType;

  typedef viennacl::matrix<ScalarType, viennacl::row_major>    VCLMatrixType;

  /**
  * Setup ublas objects and fill with data:
  **/
  std::size_t dim_large = 5;
  std::size_t dim_small = 3;

  MatrixType ublas_A(dim_large, dim_large);
  MatrixType ublas_B(dim_small, dim_small);
  MatrixType ublas_C(dim_large, dim_small);
  MatrixType ublas_D(dim_small, dim_large);


  for (std::size_t i=0; i<ublas_A.size1(); ++i)
    for (std::size_t j=0; j<ublas_A.size2(); ++j)
      ublas_A(i,j) = static_cast<ScalarType>((i+1) + (j+1)*(i+1));

  for (std::size_t i=0; i<ublas_B.size1(); ++i)
    for (std::size_t j=0; j<ublas_B.size2(); ++j)
      ublas_B(i,j) = static_cast<ScalarType>((i+1) + (j+1)*(i+1));

  for (std::size_t i=0; i<ublas_C.size1(); ++i)
    for (std::size_t j=0; j<ublas_C.size2(); ++j)
      ublas_C(i,j) = static_cast<ScalarType>((j+2) + (j+1)*(i+1));

  for (std::size_t i=0; i<ublas_D.size1(); ++i)
    for (std::size_t j=0; j<ublas_D.size2(); ++j)
      ublas_D(i,j) = static_cast<ScalarType>((j+2) + (j+1)*(i+1));

  /**
  * Extract submatrices using the ranges in ublas
  **/
  boost::numeric::ublas::range ublas_r1(0, dim_small);                      //the first 'dim_small' entries
  boost::numeric::ublas::range ublas_r2(dim_large - dim_small, dim_large);  //the last 'dim_small' entries
  boost::numeric::ublas::matrix_range<MatrixType> ublas_A_sub1(ublas_A, ublas_r1, ublas_r1); //upper left part of A
  boost::numeric::ublas::matrix_range<MatrixType> ublas_A_sub2(ublas_A, ublas_r2, ublas_r2); //lower right part of A

  boost::numeric::ublas::matrix_range<MatrixType> ublas_C_sub(ublas_C, ublas_r1, ublas_r1); //upper left part of C
  boost::numeric::ublas::matrix_range<MatrixType> ublas_D_sub(ublas_D, ublas_r1, ublas_r1); //upper left part of D

  /**
  * Setup ViennaCL objects and copy data from uBLAS objects
  **/
  VCLMatrixType vcl_A(dim_large, dim_large);
  VCLMatrixType vcl_B(dim_small, dim_small);
  VCLMatrixType vcl_C(dim_large, dim_small);
  VCLMatrixType vcl_D(dim_small, dim_large);

  viennacl::copy(ublas_A, vcl_A);
  viennacl::copy(ublas_B, vcl_B);
  viennacl::copy(ublas_C, vcl_C);
  viennacl::copy(ublas_D, vcl_D);

  /**
  * Extract submatrices using the ranges in ViennaCL. Similar to the code above for uBLAS.
  **/
  viennacl::range vcl_r1(0, dim_small);   //the first 'dim_small' entries
  viennacl::range vcl_r2(dim_large - dim_small, dim_large); //the last 'dim_small' entries
  viennacl::matrix_range<VCLMatrixType>   vcl_A_sub1(vcl_A, vcl_r1, vcl_r1); //upper left part of A
  viennacl::matrix_range<VCLMatrixType>   vcl_A_sub2(vcl_A, vcl_r2, vcl_r2); //lower right part of A

  viennacl::matrix_range<VCLMatrixType>   vcl_C_sub(vcl_C, vcl_r1, vcl_r1); //upper left part of C
  viennacl::matrix_range<VCLMatrixType>   vcl_D_sub(vcl_D, vcl_r1, vcl_r1); //upper left part of D

  /**
  * First use case: Copy from ublas to submatrices and back:
  **/

  ublas_A_sub1 = ublas_B;
  viennacl::copy(ublas_B, vcl_A_sub1);
  viennacl::copy(vcl_A_sub1, ublas_B);

  /**
  * Second use case: Addition of matrices.
  **/

  // range to range:
  ublas_A_sub2 += ublas_A_sub2;
  vcl_A_sub2 += vcl_A_sub2;

  // range to matrix:
  ublas_B += ublas_A_sub2;
  vcl_B += vcl_A_sub2;


  /**
  * Third use case: Matrix range with matrix-matrix product:
  **/
  ublas_A_sub1 += prod(ublas_C_sub, ublas_D_sub);
  vcl_A_sub1 += viennacl::linalg::prod(vcl_C_sub, vcl_D_sub);

  /**
  * Print result matrices:
  **/
  std::cout << "Result ublas:    " << ublas_A << std::endl;
  std::cout << "Result ViennaCL: " << vcl_A << std::endl;

  /**
  *  That's it. Print a success message:
  **/
  std::cout << "!!!! TUTORIAL COMPLETED SUCCESSFULLY !!!!" << std::endl;

  return EXIT_SUCCESS;
}

