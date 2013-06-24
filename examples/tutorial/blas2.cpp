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
*   Tutorial: BLAS level 2 functionality (blas2.cpp and blas2.cu are identical, the latter being required for compilation using CUDA nvcc)
*
*/


//
// include necessary system headers
//
#include <iostream>

//
// ublas includes
//
#include <boost/numeric/ublas/io.hpp>
#include <boost/numeric/ublas/triangular.hpp>
#include <boost/numeric/ublas/matrix_sparse.hpp>
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/matrix_proxy.hpp>
#include <boost/numeric/ublas/lu.hpp>
#include <boost/numeric/ublas/io.hpp>


// Must be set if you want to use ViennaCL algorithms on ublas objects
#define VIENNACL_WITH_UBLAS 1


//
// ViennaCL includes
//
#include "viennacl/scalar.hpp"
#include "viennacl/vector.hpp"
#include "viennacl/matrix.hpp"
#include "viennacl/linalg/direct_solve.hpp"
#include "viennacl/linalg/prod.hpp"       //generic matrix-vector product
#include "viennacl/linalg/norm_2.hpp"     //generic l2-norm for vectors
#include "viennacl/linalg/lu.hpp"         //LU substitution routines

// Some helper functions for this tutorial:
#include "Random.hpp"
#include "vector-io.hpp"

using namespace boost::numeric;

int main()
{
  typedef float       ScalarType;

  //
  // Set up some ublas objects
  //
  ublas::vector<ScalarType> rhs(12);
  for (unsigned int i = 0; i < rhs.size(); ++i)
    rhs(i) = random<ScalarType>();
  ublas::vector<ScalarType> rhs2 = rhs;
  ublas::vector<ScalarType> result = ublas::zero_vector<ScalarType>(10);
  ublas::vector<ScalarType> result2 = result;
  ublas::vector<ScalarType> rhs_trans = rhs;
  rhs_trans.resize(result.size(), true);
  ublas::vector<ScalarType> result_trans = ublas::zero_vector<ScalarType>(rhs.size());


  ublas::matrix<ScalarType> matrix(result.size(),rhs.size());

  //
  // Fill the matrix
  //
  for (unsigned int i = 0; i < matrix.size1(); ++i)
    for (unsigned int j = 0; j < matrix.size2(); ++j)
      matrix(i,j) = random<ScalarType>();

  //
  // Use some plain STL types:
  //
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

  //
  // Set up some ViennaCL objects
  //
  viennacl::vector<ScalarType> vcl_rhs(rhs.size());
  viennacl::vector<ScalarType> vcl_result(result.size());
  viennacl::matrix<ScalarType> vcl_matrix(result.size(), rhs.size());
  viennacl::matrix<ScalarType> vcl_matrix2(result.size(), rhs.size());

  viennacl::copy(rhs.begin(), rhs.end(), vcl_rhs.begin());
  viennacl::copy(matrix, vcl_matrix);     //copy from ublas dense matrix type to ViennaCL type

  //
  // Some basic matrix operations
  //
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

  //
  // A matrix can be cleared directly:
  //
  vcl_matrix.clear();

  viennacl::copy(stl_matrix, vcl_matrix); //alternative: copy from STL vector< vector<> > type to ViennaCL type

  //for demonstration purposes (no effect):
  viennacl::copy(vcl_matrix, matrix); //copy back from ViennaCL to ublas type.
  viennacl::copy(vcl_matrix, stl_matrix); //copy back from ViennaCL to STL type.

  /////////////////////////////////////////////////
  //////////// Matrix vector products /////////////
  /////////////////////////////////////////////////


  //
  // Compute matrix-vector products
  //
  std::cout << "----- Matrix-Vector product -----" << std::endl;
  result = ublas::prod(matrix, rhs);                            //the ublas way
  stl_result = viennacl::linalg::prod(stl_matrix, stl_rhs);     //using STL
  vcl_result = viennacl::linalg::prod(vcl_matrix, vcl_rhs);     //the ViennaCL way

  //
  // Compute transposed matrix-vector products
  //
  std::cout << "----- Transposed Matrix-Vector product -----" << std::endl;
  result_trans = prod(trans(matrix), rhs_trans);

  viennacl::vector<ScalarType> vcl_rhs_trans(rhs_trans.size());
  viennacl::vector<ScalarType> vcl_result_trans(result_trans.size());
  viennacl::copy(rhs_trans.begin(), rhs_trans.end(), vcl_rhs_trans.begin());
  vcl_result_trans = viennacl::linalg::prod(trans(vcl_matrix), vcl_rhs_trans);



  /////////////////////////////////////////////////
  //////////////// Direct solver  /////////////////
  /////////////////////////////////////////////////


  //
  // Setup suitable matrices
  //
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

  rhs.resize(tri_matrix.size1(), true);
  rhs2.resize(tri_matrix.size1(), true);
  vcl_rhs.resize(tri_matrix.size1(), true);

  viennacl::copy(rhs.begin(), rhs.end(), vcl_rhs.begin());
  vcl_result.resize(10);


  //
  // Triangular solver
  //
  std::cout << "----- Upper Triangular solve -----" << std::endl;
  result = ublas::solve(tri_matrix, rhs, ublas::upper_tag());                                    //ublas
  vcl_result = viennacl::linalg::solve(vcl_tri_matrix, vcl_rhs, viennacl::linalg::upper_tag());  //ViennaCL

  //
  // Inplace variants of the above
  //
  ublas::inplace_solve(tri_matrix, rhs, ublas::upper_tag());                                //ublas
  viennacl::linalg::inplace_solve(vcl_tri_matrix, vcl_rhs, viennacl::linalg::upper_tag());  //ViennaCL


  //
  // Set up a full system for LU solver:
  //
  std::cout << "----- LU factorization -----" << std::endl;
  std::size_t lu_dim = 300;
  ublas::matrix<ScalarType> square_matrix(lu_dim, lu_dim);
  ublas::vector<ScalarType> lu_rhs(lu_dim);
  viennacl::matrix<ScalarType> vcl_square_matrix(lu_dim, lu_dim);
  viennacl::vector<ScalarType> vcl_lu_rhs(lu_dim);

  for (std::size_t i=0; i<lu_dim; ++i)
    for (std::size_t j=0; j<lu_dim; ++j)
      square_matrix(i,j) = random<ScalarType>();

  //put some more weight on diagonal elements:
  for (std::size_t j=0; j<lu_dim; ++j)
  {
    square_matrix(j,j) += 10.0;
    lu_rhs(j) = random<ScalarType>();
  }

  viennacl::copy(square_matrix, vcl_square_matrix);
  viennacl::copy(lu_rhs, vcl_lu_rhs);
  viennacl::linalg::lu_factorize(vcl_square_matrix);
  viennacl::linalg::lu_substitute(vcl_square_matrix, vcl_lu_rhs);
  viennacl::copy(square_matrix, vcl_square_matrix);
  viennacl::copy(lu_rhs, vcl_lu_rhs);


  //
  // ublas:
  //
  ublas::lu_factorize(square_matrix);
  ublas::inplace_solve (square_matrix, lu_rhs, ublas::unit_lower_tag ());
  ublas::inplace_solve (square_matrix, lu_rhs, ublas::upper_tag ());


  //
  // ViennaCL:
  //
  viennacl::linalg::lu_factorize(vcl_square_matrix);
  viennacl::linalg::lu_substitute(vcl_square_matrix, vcl_lu_rhs);

  //
  //  That's it.
  //
  std::cout << "!!!! TUTORIAL COMPLETED SUCCESSFULLY !!!!" << std::endl;

  return 0;
}

