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

/** \example blas2.cpp
*
*   In this tutorial the BLAS level 2 functionality in ViennaCL is demonstrated.
*
*   We start with including the required header files:
**/

// System headers
#include <iostream>

// uBLAS headers
#include <boost/numeric/ublas/io.hpp>
#include <boost/numeric/ublas/triangular.hpp>
#include <boost/numeric/ublas/matrix_sparse.hpp>
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/matrix_proxy.hpp>
#include <boost/numeric/ublas/vector_proxy.hpp>
#include <boost/numeric/ublas/lu.hpp>
#include <boost/numeric/ublas/io.hpp>


// Must be set if you want to use ViennaCL algorithms on ublas objects
#define VIENNACL_WITH_UBLAS 1

// ViennaCL headers
#include "viennacl/scalar.hpp"
#include "viennacl/vector.hpp"
#include "viennacl/matrix.hpp"
#include "viennacl/linalg/direct_solve.hpp"
#include "viennacl/linalg/prod.hpp"       //generic matrix-vector product
#include "viennacl/linalg/norm_2.hpp"     //generic l2-norm for vectors
#include "viennacl/linalg/lu.hpp"         //LU substitution routines
#include "viennacl/tools/random.hpp"

// Make `boost::numeric::ublas` available under the shortcut `ublas`:
using namespace boost::numeric;

/**
* We do not need any auxiliary functions in this example, so let us start directly in main():
**/
int main()
{
  typedef float       ScalarType;

  viennacl::tools::uniform_random_numbers<ScalarType> randomNumber;

  /**
  * Set up some uBLAS vectors and a matrix.
  * They will be later used for filling the ViennaCL objects with data.
  **/
  ublas::vector<ScalarType> rhs(12);
  for (unsigned int i = 0; i < rhs.size(); ++i)
    rhs(i) = randomNumber();
  ublas::vector<ScalarType> rhs2 = rhs;
  ublas::vector<ScalarType> result = ublas::zero_vector<ScalarType>(10);
  ublas::vector<ScalarType> result2 = result;
  ublas::vector<ScalarType> rhs_trans = rhs;
  rhs_trans.resize(result.size(), true);
  ublas::vector<ScalarType> result_trans = ublas::zero_vector<ScalarType>(rhs.size());

  ublas::matrix<ScalarType> matrix(result.size(),rhs.size());

  /**
  * Fill the uBLAS-matrix
  **/
  for (unsigned int i = 0; i < matrix.size1(); ++i)
    for (unsigned int j = 0; j < matrix.size2(); ++j)
      matrix(i,j) = randomNumber();

  /**
  * Use some plain STL types:
  **/
  std::vector< ScalarType > stl_result(result.size());
  std::vector< ScalarType > stl_rhs(rhs.size());
  std::vector< std::vector<ScalarType> > stl_matrix(result.size());
  for (unsigned int i=0; i < result.size(); ++i)
  {
    stl_matrix[i].resize(rhs.size());
    for (unsigned int j = 0; j < matrix.size2(); ++j)
    {
      stl_rhs[j] = rhs[j];
      stl_matrix[i][j] = matrix(i,j);
    }
  }

  /**
  * Set up some ViennaCL objects (initialized with zeros) and then copy data from the uBLAS objects.
  **/
  viennacl::vector<ScalarType> vcl_rhs(rhs.size());
  viennacl::vector<ScalarType> vcl_result(result.size());
  viennacl::matrix<ScalarType> vcl_matrix(result.size(), rhs.size());
  viennacl::matrix<ScalarType> vcl_matrix2(result.size(), rhs.size());

  viennacl::copy(rhs.begin(), rhs.end(), vcl_rhs.begin());
  viennacl::copy(matrix, vcl_matrix);     //copy from ublas dense matrix type to ViennaCL type

  /**
  * Some basic matrix operations with ViennaCL are as follows:
  **/
  vcl_matrix2 = vcl_matrix;
  vcl_matrix2 += vcl_matrix;
  vcl_matrix2 -= vcl_matrix;
  vcl_matrix2 = vcl_matrix2 + vcl_matrix;
  vcl_matrix2 = vcl_matrix2 - vcl_matrix;

  viennacl::scalar<ScalarType> vcl_3(3.0);
  vcl_matrix2 *= ScalarType(2.0);
  vcl_matrix2 /= ScalarType(2.0);
  vcl_matrix2 *= vcl_3;
  vcl_matrix2 /= vcl_3;

  /**
  * A matrix can be cleared directly:
  **/
  vcl_matrix.clear();

  /**
  * Other ways of data transfers between matrices in main memory and a ViennaCL matrix:
  **/
  viennacl::copy(stl_matrix, vcl_matrix); //alternative: copy from STL vector< vector<> > type to ViennaCL type

  //for demonstration purposes (no effect):
  viennacl::copy(vcl_matrix, matrix); //copy back from ViennaCL to ublas type.
  viennacl::copy(vcl_matrix, stl_matrix); //copy back from ViennaCL to STL type.

  /**
  * <h2> Matrix-Vector Products </h2>
  *
  * Compute matrix-vector products
  **/
  std::cout << "----- Matrix-Vector product -----" << std::endl;
  result = ublas::prod(matrix, rhs);                            //the ublas way
  stl_result = viennacl::linalg::prod(stl_matrix, stl_rhs);     //using STL
  vcl_result = viennacl::linalg::prod(vcl_matrix, vcl_rhs);     //the ViennaCL way

  /**
  * Compute transposed matrix-vector products
  **/
  std::cout << "----- Transposed Matrix-Vector product -----" << std::endl;
  result_trans = prod(trans(matrix), rhs_trans);

  viennacl::vector<ScalarType> vcl_rhs_trans(rhs_trans.size());
  viennacl::vector<ScalarType> vcl_result_trans(result_trans.size());
  viennacl::copy(rhs_trans.begin(), rhs_trans.end(), vcl_rhs_trans.begin());
  vcl_result_trans = viennacl::linalg::prod(trans(vcl_matrix), vcl_rhs_trans);



  /**
  * <h2>Direct Solver</h2>
  *
  * In order to demonstrate the direct solvers, we first need to setup suitable square matrices.
  * This is again achieved by running the setup on the CPU and then copy the data over to ViennaCL types:
  **/
  ublas::matrix<ScalarType> tri_matrix(10,10);
  for (std::size_t i=0; i<tri_matrix.size1(); ++i)
  {
    for (std::size_t j=0; j<i; ++j)
      tri_matrix(i,j) = 0.0;

    for (std::size_t j=i; j<tri_matrix.size2(); ++j)
      tri_matrix(i,j) = matrix(i,j);
  }

  viennacl::matrix<ScalarType> vcl_tri_matrix = viennacl::identity_matrix<ScalarType>(tri_matrix.size1());
  viennacl::copy(tri_matrix, vcl_tri_matrix);

  // Bring vectors to correct size:
  rhs.resize(tri_matrix.size1(), true);
  rhs2.resize(tri_matrix.size1(), true);
  vcl_rhs.resize(tri_matrix.size1(), true);

  viennacl::copy(rhs.begin(), rhs.end(), vcl_rhs.begin());
  vcl_result.resize(10);


  /**
  * Run a triangular solver on the upper triangular part of the matrix:
  **/
  std::cout << "----- Upper Triangular solve -----" << std::endl;
  result = ublas::solve(tri_matrix, rhs, ublas::upper_tag());                                    //ublas
  vcl_result = viennacl::linalg::solve(vcl_tri_matrix, vcl_rhs, viennacl::linalg::upper_tag());  //ViennaCL

  /**
  * Inplace variants of triangular solvers:
  **/
  ublas::inplace_solve(tri_matrix, rhs, ublas::upper_tag());                                //ublas
  viennacl::linalg::inplace_solve(vcl_tri_matrix, vcl_rhs, viennacl::linalg::upper_tag());  //ViennaCL


  /**
  * Set up a full system for full solver using LU factorizations:
  **/
  std::cout << "----- LU factorization -----" << std::endl;
  std::size_t lu_dim = 300;
  ublas::matrix<ScalarType> square_matrix(lu_dim, lu_dim);
  ublas::vector<ScalarType> lu_rhs(lu_dim);
  viennacl::matrix<ScalarType> vcl_square_matrix(lu_dim, lu_dim);
  viennacl::vector<ScalarType> vcl_lu_rhs(lu_dim);

  for (std::size_t i=0; i<lu_dim; ++i)
    for (std::size_t j=0; j<lu_dim; ++j)
      square_matrix(i,j) = randomNumber();

  //put some more weight on diagonal elements:
  for (std::size_t j=0; j<lu_dim; ++j)
  {
    square_matrix(j,j) += ScalarType(10.0);
    lu_rhs(j) = randomNumber();
  }

  viennacl::copy(square_matrix, vcl_square_matrix);
  viennacl::copy(lu_rhs, vcl_lu_rhs);
  viennacl::linalg::lu_factorize(vcl_square_matrix);
  viennacl::linalg::lu_substitute(vcl_square_matrix, vcl_lu_rhs);
  viennacl::copy(square_matrix, vcl_square_matrix);
  viennacl::copy(lu_rhs, vcl_lu_rhs);


  /**
  * Full solver with Boost.uBLAS:
  **/
  ublas::lu_factorize(square_matrix);
  ublas::inplace_solve (square_matrix, lu_rhs, ublas::unit_lower_tag ());
  ublas::inplace_solve (square_matrix, lu_rhs, ublas::upper_tag ());


  /**
  * Full solver with ViennaCL:
  **/
  viennacl::linalg::lu_factorize(vcl_square_matrix);
  viennacl::linalg::lu_substitute(vcl_square_matrix, vcl_lu_rhs);

  /**
  *  That's it.
  **/
  std::cout << "!!!! TUTORIAL COMPLETED SUCCESSFULLY !!!!" << std::endl;

  return EXIT_SUCCESS;
}

