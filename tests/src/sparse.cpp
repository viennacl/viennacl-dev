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



/** \file tests/src/sparse.cpp  Tests sparse matrix operations.
*   \test  Tests sparse matrix operations.
**/

//
// *** System
//
#include <iostream>
#include <vector>
#include <map>
#include <cmath>

//
// *** ViennaCL
//
#include "viennacl/scalar.hpp"
#include "viennacl/compressed_matrix.hpp"
#include "viennacl/compressed_compressed_matrix.hpp"
#include "viennacl/coordinate_matrix.hpp"
#include "viennacl/ell_matrix.hpp"
#include "viennacl/sliced_ell_matrix.hpp"
#include "viennacl/hyb_matrix.hpp"
#include "viennacl/vector.hpp"
#include "viennacl/vector_proxy.hpp"
#include "viennacl/linalg/prod.hpp"
#include "viennacl/linalg/norm_2.hpp"
#include "viennacl/linalg/ilu.hpp"
#include "viennacl/linalg/detail/ilu/common.hpp"
#include "viennacl/io/matrix_market.hpp"
#include "viennacl/tools/random.hpp"



//
// -------------------------------------------------------------
//
template<typename ScalarType>
ScalarType diff(ScalarType & s1, viennacl::scalar<ScalarType> & s2)
{
   if (s1 != s2)
      return (s1 - s2) / std::max(fabs(s1), std::fabs(s2));
   return 0;
}

template<typename ScalarType>
ScalarType diff(std::vector<ScalarType> & v1, viennacl::vector<ScalarType> & v2)
{
   std::vector<ScalarType> v2_cpu(v2.size());
   viennacl::backend::finish();
   viennacl::copy(v2.begin(), v2.end(), v2_cpu.begin());

   for (unsigned int i=0;i<v1.size(); ++i)
   {
      if ( std::max( std::fabs(v2_cpu[i]), std::fabs(v1[i]) ) > 0 )
      {
        //if (std::max( std::fabs(v2_cpu[i]), std::fabs(v1[i]) ) < 1e-10 )  //absolute tolerance (avoid round-off issues)
        //  v2_cpu[i] = 0;
        //else
          v2_cpu[i] = std::fabs(v2_cpu[i] - v1[i]) / std::max( std::fabs(v2_cpu[i]), std::fabs(v1[i]) );
      }
      else
         v2_cpu[i] = 0.0;

      if (v2_cpu[i] > 0.0001)
      {
        //std::cout << "Neighbor: "      << i-1 << ": " << v1[i-1] << " vs. " << v2_cpu[i-1] << std::endl;
        std::cout << "Error at entry " << i   << ": Should: " << v1[i]   << " vs. Is: " << v2[i]   << std::endl;
        //std::cout << "Neighbor: "      << i+1 << ": " << v1[i+1] << " vs. " << v2_cpu[i+1] << std::endl;
        exit(EXIT_FAILURE);
      }
   }

   ScalarType norm_inf = 0;
   for (std::size_t i=0; i<v2_cpu.size(); ++i)
     norm_inf = std::max<ScalarType>(norm_inf, std::fabs(v2_cpu[i]));

   return norm_inf;
}


template<typename IndexT, typename NumericT, typename SparseMatrixT>
NumericT diff(std::vector<std::map<IndexT, NumericT> > & cpu_A, SparseMatrixT & vcl_A)
{
  typedef typename std::map<IndexT, NumericT>::const_iterator  RowIterator;

  std::vector<std::map<IndexT, NumericT> > from_gpu(vcl_A.size1());

  viennacl::backend::finish();
  viennacl::copy(vcl_A, from_gpu);

  NumericT error = 0;

  //step 1: compare all entries from cpu_A with vcl_A:
  for (std::size_t i=0; i<cpu_A.size(); ++i)
  {
    //std::cout << "Row " << row_it.index1() << ": " << std::endl;
    for (RowIterator it = cpu_A[i].begin(); it != cpu_A[i].end(); ++it)
    {
      //std::cout << "(" << col_it.index2() << ", " << *col_it << std::endl;
      NumericT current_error = 0;
      NumericT val_cpu_A = it->second;
      NumericT val_gpu_A = from_gpu[i][it->first];

      NumericT max_val = std::max(std::fabs(val_cpu_A), std::fabs(val_gpu_A));
      if (max_val > 0)
        current_error = std::fabs(val_cpu_A - val_gpu_A) / max_val;
      if (current_error > error)
        error = current_error;
    }
  }

  //step 2: compare all entries from gpu_matrix with cpu_matrix (sparsity pattern might differ):
  //std::cout << "ViennaCL matrix: " << std::endl;
  for (std::size_t i=0; i<from_gpu.size(); ++i)
  {
    //std::cout << "Row " << row_it.index1() << ": " << std::endl;
    for (RowIterator it = from_gpu[i].begin(); it != from_gpu[i].end(); ++it)
    {
      //std::cout << "(" << col_it.index2() << ", " << *col_it << std::endl;
      NumericT current_error = 0;
      NumericT val_gpu_A = it->second;
      NumericT val_cpu_A = cpu_A[i][it->first];

      NumericT max_val = std::max(std::fabs(val_cpu_A), std::fabs(val_gpu_A));
      if (max_val > 0)
        current_error = std::fabs(val_cpu_A - val_gpu_A) / max_val;
      if (current_error > error)
        error = current_error;
    }
  }

  return error;
}


template<typename NumericT, typename VCL_MatrixT, typename Epsilon, typename STLVectorT, typename VCLVectorT>
int strided_matrix_vector_product_test(Epsilon epsilon,
                                       STLVectorT & result,     STLVectorT const & rhs,
                                       VCLVectorT & vcl_result, VCLVectorT & vcl_rhs)
{
  typedef typename std::map<unsigned int, NumericT>::const_iterator    RowIterator;
    int retval = EXIT_SUCCESS;

    std::vector<std::map<unsigned int, NumericT> > std_A(5);
    std_A[0][0] = NumericT(2.0); std_A[0][2] = NumericT(-1.0);
    std_A[1][0] = NumericT(3.0); std_A[1][2] = NumericT(-5.0);
    std_A[2][1] = NumericT(5.0); std_A[2][2] = NumericT(-2.0);
    std_A[3][2] = NumericT(1.0); std_A[3][3] = NumericT(-6.0);
    std_A[4][1] = NumericT(7.0); std_A[4][2] = NumericT(-5.0);
    //the following computes project(result, slice(1, 3, 5)) = prod(std_A, project(rhs, slice(3, 2, 4)));
    for (std::size_t i=0; i<5; ++i)
    {
      NumericT val = 0;
      for (RowIterator it = std_A[i].begin(); it != std_A[i].end(); ++it)
        val += it->second * rhs[3 + 2*it->first];
      result[1 + 3*i] = val;
    }

    VCL_MatrixT vcl_sparse_matrix2;
    viennacl::copy(std_A, vcl_sparse_matrix2);
    viennacl::vector<NumericT> vec(4);
    vec(0) = rhs[3];
    vec(1) = rhs[5];
    vec(2) = rhs[7];
    vec(3) = rhs[9];
    viennacl::project(vcl_result, viennacl::slice(1, 3, 5)) = viennacl::linalg::prod(vcl_sparse_matrix2, viennacl::project(vcl_rhs, viennacl::slice(3, 2, 4)));

    if ( std::fabs(diff(result, vcl_result)) > epsilon )
    {
      std::cout << "# Error at operation: matrix-vector product with stided vectors, part 1" << std::endl;
      std::cout << "  diff: " << std::fabs(diff(result, vcl_result)) << std::endl;
      retval = EXIT_FAILURE;
    }
    vcl_result(1)  = NumericT(1.0);
    vcl_result(4)  = NumericT(1.0);
    vcl_result(7)  = NumericT(1.0);
    vcl_result(10) = NumericT(1.0);
    vcl_result(13) = NumericT(1.0);

    viennacl::project(vcl_result, viennacl::slice(1, 3, 5)) = viennacl::linalg::prod(vcl_sparse_matrix2, vec);

    if ( std::fabs(diff(result, vcl_result)) > epsilon )
    {
      std::cout << "# Error at operation: matrix-vector product with strided vectors, part 2" << std::endl;
      std::cout << "  diff: " << std::fabs(diff(result, vcl_result)) << std::endl;
      retval = EXIT_FAILURE;
    }

    return retval;
}


template< typename NumericT, typename VCL_MATRIX, typename Epsilon >
int resize_test(Epsilon const& epsilon)
{
   int retval = EXIT_SUCCESS;

   std::vector<std::map<unsigned int, NumericT> > std_A(5);
   VCL_MATRIX vcl_matrix;

   std_A[0][0] = NumericT(10.0); std_A[0][1] = NumericT(0.1); std_A[0][2] = NumericT(0.2); std_A[0][3] = NumericT(0.3); std_A[0][4] = NumericT(0.4);
   std_A[1][0] = NumericT(1.0);  std_A[1][1] = NumericT(1.1); std_A[1][2] = NumericT(1.2); std_A[1][3] = NumericT(1.3); std_A[1][4] = NumericT(1.4);
   std_A[2][0] = NumericT(2.0);  std_A[2][1] = NumericT(2.1); std_A[2][2] = NumericT(2.2); std_A[2][3] = NumericT(2.3); std_A[2][4] = NumericT(2.4);
   std_A[3][0] = NumericT(3.0);  std_A[3][1] = NumericT(3.1); std_A[3][2] = NumericT(3.2); std_A[3][3] = NumericT(3.3); std_A[3][4] = NumericT(3.4);
   std_A[4][0] = NumericT(4.0);  std_A[4][1] = NumericT(4.1); std_A[4][2] = NumericT(4.2); std_A[4][3] = NumericT(4.3); std_A[4][4] = NumericT(4.4);

   viennacl::copy(std_A, vcl_matrix);
   std::vector<std::map<unsigned int, NumericT> > std_B(std_A.size());
   viennacl::copy(vcl_matrix, std_B);

   std::cout << "Checking for equality after copy..." << std::endl;
    if ( std::fabs(diff(std_A, vcl_matrix)) > epsilon )
    {
        std::cout << "# Error at operation: equality after copy with sparse matrix" << std::endl;
        std::cout << "  diff: " << std::fabs(diff(std_A, vcl_matrix)) << std::endl;
        return EXIT_FAILURE;
    }

   std::cout << "Testing resize to larger..." << std::endl;
   std_A.resize(10);
   std_A[0][0] = NumericT(10.0); std_A[0][1] = NumericT(0.1); std_A[0][2] = NumericT(0.2); std_A[0][3] = NumericT(0.3); std_A[0][4] = NumericT(0.4);
   std_A[1][0] = NumericT( 1.0); std_A[1][1] = NumericT(1.1); std_A[1][2] = NumericT(1.2); std_A[1][3] = NumericT(1.3); std_A[1][4] = NumericT(1.4);
   std_A[2][0] = NumericT( 2.0); std_A[2][1] = NumericT(2.1); std_A[2][2] = NumericT(2.2); std_A[2][3] = NumericT(2.3); std_A[2][4] = NumericT(2.4);
   std_A[3][0] = NumericT( 3.0); std_A[3][1] = NumericT(3.1); std_A[3][2] = NumericT(3.2); std_A[3][3] = NumericT(3.3); std_A[3][4] = NumericT(3.4);
   std_A[4][0] = NumericT( 4.0); std_A[4][1] = NumericT(4.1); std_A[4][2] = NumericT(4.2); std_A[4][3] = NumericT(4.3); std_A[4][4] = NumericT(4.4);

   vcl_matrix.resize(10, 10, true);

    if ( std::fabs(diff(std_A, vcl_matrix)) > epsilon )
    {
        std::cout << "# Error at operation: resize (to larger) with sparse matrix" << std::endl;
        std::cout << "  diff: " << std::fabs(diff(std_A, vcl_matrix)) << std::endl;
        return EXIT_FAILURE;
    }

   std_A[5][5] = NumericT(5.5); std_A[5][6] = NumericT(5.6); std_A[5][7] = NumericT(5.7); std_A[5][8] = NumericT(5.8); std_A[5][9] = NumericT(5.9);
   std_A[6][5] = NumericT(6.5); std_A[6][6] = NumericT(6.6); std_A[6][7] = NumericT(6.7); std_A[6][8] = NumericT(6.8); std_A[6][9] = NumericT(6.9);
   std_A[7][5] = NumericT(7.5); std_A[7][6] = NumericT(7.6); std_A[7][7] = NumericT(7.7); std_A[7][8] = NumericT(7.8); std_A[7][9] = NumericT(7.9);
   std_A[8][5] = NumericT(8.5); std_A[8][6] = NumericT(8.6); std_A[8][7] = NumericT(8.7); std_A[8][8] = NumericT(8.8); std_A[8][9] = NumericT(8.9);
   std_A[9][5] = NumericT(9.5); std_A[9][6] = NumericT(9.6); std_A[9][7] = NumericT(9.7); std_A[9][8] = NumericT(9.8); std_A[9][9] = NumericT(9.9);
   viennacl::copy(std_A, vcl_matrix);

   std::cout << "Testing resize to smaller..." << std::endl;
   std_A.clear();
   std_A.resize(7);
   std_A[0][0] = NumericT(10.0); std_A[0][1] = NumericT(0.1); std_A[0][2] = NumericT(0.2); std_A[0][3] = NumericT(0.3); std_A[0][4] = NumericT(0.4);
   std_A[1][0] = NumericT( 1.0); std_A[1][1] = NumericT(1.1); std_A[1][2] = NumericT(1.2); std_A[1][3] = NumericT(1.3); std_A[1][4] = NumericT(1.4);
   std_A[2][0] = NumericT( 2.0); std_A[2][1] = NumericT(2.1); std_A[2][2] = NumericT(2.2); std_A[2][3] = NumericT(2.3); std_A[2][4] = NumericT(2.4);
   std_A[3][0] = NumericT( 3.0); std_A[3][1] = NumericT(3.1); std_A[3][2] = NumericT(3.2); std_A[3][3] = NumericT(3.3); std_A[3][4] = NumericT(3.4);
   std_A[4][0] = NumericT( 4.0); std_A[4][1] = NumericT(4.1); std_A[4][2] = NumericT(4.2); std_A[4][3] = NumericT(4.3); std_A[4][4] = NumericT(4.4);
   std_A[5][5] = NumericT( 5.5); std_A[5][6] = NumericT(5.6); //std_A[5][7] = NumericT(5.7); std_A[5][8] = NumericT(5.8); std_A[5][9] = NumericT(5.9);
   std_A[6][5] = NumericT( 6.5); std_A[6][6] = NumericT(6.6); //std_A[6][7] = NumericT(6.7); std_A[6][8] = NumericT(6.8); std_A[6][9] = NumericT(6.9);

   vcl_matrix.resize(7, 7);

   //std::cout << std_A << std::endl;
    if ( std::fabs(diff(std_A, vcl_matrix)) > epsilon )
    {
        std::cout << "# Error at operation: resize (to smaller) with sparse matrix" << std::endl;
        std::cout << "  diff: " << std::fabs(diff(std_A, vcl_matrix)) << std::endl;
        retval = EXIT_FAILURE;
    }

   std::vector<NumericT> std_vec(std_A.size(), NumericT(3.1415));
   viennacl::vector<NumericT> vcl_vec(std_A.size());


  std::cout << "Testing unit lower triangular solve: compressed_matrix" << std::endl;
  viennacl::copy(std_vec, vcl_vec);

  std::cout << "STL..." << std::endl;
  //boost::numeric::ublas::inplace_solve((ublas_matrix), ublas_vec, boost::numeric::ublas::unit_lower_tag());
  for (std::size_t i=1; i<std_A.size(); ++i)
    for (typename std::map<unsigned int, NumericT>::const_iterator it = std_A[i].begin(); it != std_A[i].end(); ++it)
    {
      if (it->first < static_cast<unsigned int>(i))
        std_vec[i] -= it->second * std_vec[it->first];
      else
        continue;
    }

  std::cout << "ViennaCL..." << std::endl;
  viennacl::linalg::inplace_solve((vcl_matrix), vcl_vec, viennacl::linalg::unit_lower_tag());

  if ( std::fabs(diff(std_vec, vcl_vec)) > epsilon )
  {
      std::cout << "# Error at operation: unit lower triangular solve" << std::endl;
      std::cout << "  diff: " << std::fabs(diff(std_vec, vcl_vec)) << std::endl;
      retval = EXIT_FAILURE;
  }
  return retval;
}


//
// -------------------------------------------------------------
//
template< typename NumericT, typename Epsilon >
int test(Epsilon const& epsilon)
{
  viennacl::tools::uniform_random_numbers<NumericT> randomNumber;

  std::cout << "Testing resizing of compressed_matrix..." << std::endl;
  int retval = resize_test<NumericT, viennacl::compressed_matrix<NumericT> >(epsilon);
  if (retval != EXIT_SUCCESS)
    return retval;

  // --------------------------------------------------------------------------
  std::vector<NumericT> rhs;
  std::vector<NumericT> result;
  std::vector<std::map<unsigned int, NumericT> > std_matrix;

  if (viennacl::io::read_matrix_market_file(std_matrix, "../examples/testdata/mat65k.mtx") == EXIT_FAILURE)
  {
    std::cout << "Error reading Matrix file" << std::endl;
    return EXIT_FAILURE;
  }

  //unsigned int cg_mat_size = cg_mat.size();
  std::cout << "done reading matrix" << std::endl;


  rhs.resize(std_matrix.size());
  for (std::size_t i=0; i<rhs.size(); ++i)
  {
    std_matrix[i][i] = NumericT(0.5);   // Get rid of round-off errors by making row-sums unequal to zero:
    rhs[i] = NumericT(1) + randomNumber();
  }

  // add some random numbers to the double-compressed matrix:
  std::vector<std::map<unsigned int, NumericT> > std_cc_matrix(std_matrix.size());
  std_cc_matrix[42][199] = NumericT(3.1415);
  std_cc_matrix[31][69] = NumericT(2.71);
  std_cc_matrix[23][32] = NumericT(6);
  std_cc_matrix[177][57] = NumericT(4);
  std_cc_matrix[21][97] = NumericT(-4);
  std_cc_matrix[92][25] = NumericT(2);
  std_cc_matrix[89][62] = NumericT(11);
  std_cc_matrix[ 1][ 7] = NumericT(8);
  std_cc_matrix[85][41] = NumericT(13);
  std_cc_matrix[66][28] = NumericT(8);
  std_cc_matrix[21][74] = NumericT(-2);
  viennacl::tools::sparse_matrix_adapter<NumericT> adapted_std_cc_matrix(std_cc_matrix, std_matrix.size(), std_matrix.size());


  result = rhs;


  viennacl::vector<NumericT> vcl_rhs(rhs.size());
  viennacl::vector<NumericT> vcl_result(result.size());
  viennacl::vector<NumericT> vcl_result2(result.size());
  viennacl::compressed_matrix<NumericT> vcl_compressed_matrix(rhs.size(), rhs.size());
  viennacl::compressed_compressed_matrix<NumericT> vcl_compressed_compressed_matrix(rhs.size(), rhs.size());
  viennacl::coordinate_matrix<NumericT> vcl_coordinate_matrix(rhs.size(), rhs.size());
  viennacl::ell_matrix<NumericT> vcl_ell_matrix;
  viennacl::sliced_ell_matrix<NumericT> vcl_sliced_ell_matrix;
  viennacl::hyb_matrix<NumericT> vcl_hyb_matrix;

  viennacl::copy(rhs.begin(), rhs.end(), vcl_rhs.begin());
  viennacl::copy(std_matrix, vcl_compressed_matrix);
  viennacl::copy(adapted_std_cc_matrix, vcl_compressed_compressed_matrix);
  viennacl::copy(std_matrix, vcl_coordinate_matrix);

  // --------------------------------------------------------------------------
  std::cout << "Testing products: STL" << std::endl;
  result = viennacl::linalg::prod(std_matrix, rhs);

  std::cout << "Testing products: compressed_matrix" << std::endl;
  vcl_result = viennacl::linalg::prod(vcl_compressed_matrix, vcl_rhs);

  if ( std::fabs(diff(result, vcl_result)) > epsilon )
  {
    std::cout << "# Error at operation: matrix-vector product with compressed_matrix" << std::endl;
    std::cout << "  diff: " << std::fabs(diff(result, vcl_result)) << std::endl;
    retval = EXIT_FAILURE;
  }

  std::cout << "Testing products: compressed_matrix, strided vectors" << std::endl;
  retval = strided_matrix_vector_product_test<NumericT, viennacl::compressed_matrix<NumericT> >(epsilon, result, rhs, vcl_result, vcl_rhs);
  if (retval != EXIT_SUCCESS)
    return retval;

  //
  // Triangular solvers for A \ b:
  //

  std::cout << "Testing unit upper triangular solve: compressed_matrix" << std::endl;
  result = rhs;
  viennacl::copy(result, vcl_result);
  //boost::numeric::ublas::inplace_solve(trans(ublas_matrix_trans), result, boost::numeric::ublas::unit_upper_tag());
  for (std::size_t i2=0; i2<std_matrix.size(); ++i2)
  {
    std::size_t row = std_matrix.size() - i2 - 1;
    for (typename std::map<unsigned int, NumericT>::const_iterator it = std_matrix[row].begin(); it != std_matrix[row].end(); ++it)
    {
      if (it->first > static_cast<unsigned int>(row))
        result[row] -= it->second * result[it->first];
      else
        continue;
    }
  }

  viennacl::linalg::inplace_solve(vcl_compressed_matrix, vcl_result, viennacl::linalg::unit_upper_tag());

  if ( std::fabs(diff(result, vcl_result)) > epsilon )
  {
    std::cout << "# Error at operation: unit upper triangular solve with compressed_matrix" << std::endl;
    std::cout << "  diff: " << std::fabs(diff(result, vcl_result)) << std::endl;
    retval = EXIT_FAILURE;
  }

  ////////////////////////////

  std::cout << "Testing upper triangular solve: compressed_matrix" << std::endl;
  result = rhs;
  viennacl::copy(result, vcl_result);
  //boost::numeric::ublas::inplace_solve(trans(ublas_matrix_trans), result, boost::numeric::ublas::upper_tag());
  for (std::size_t i2=0; i2<std_matrix.size(); ++i2)
  {
    std::size_t row = std_matrix.size() - i2 - 1;
    NumericT diag = 0;
    for (typename std::map<unsigned int, NumericT>::const_iterator it = std_matrix[row].begin(); it != std_matrix[row].end(); ++it)
    {
      if (it->first > static_cast<unsigned int>(row))
        result[row] -= it->second * result[it->first];
      else if (it->first == static_cast<unsigned int>(row))
        diag = it->second;
      else
        continue;
    }
    result[row] /= diag;
  }

  viennacl::linalg::inplace_solve(vcl_compressed_matrix, vcl_result, viennacl::linalg::upper_tag());

  if ( std::fabs(diff(result, vcl_result)) > epsilon )
  {
    std::cout << "# Error at operation: upper triangular solve with compressed_matrix" << std::endl;
    std::cout << "  diff: " << std::fabs(diff(result, vcl_result)) << std::endl;
    retval = EXIT_FAILURE;
  }

  ////////////////////////////

  std::cout << "Testing unit lower triangular solve: compressed_matrix" << std::endl;
  result = rhs;
  viennacl::copy(result, vcl_result);
  //boost::numeric::ublas::inplace_solve(trans(ublas_matrix_trans), result, boost::numeric::ublas::unit_lower_tag());
  for (std::size_t i=1; i<std_matrix.size(); ++i)
    for (typename std::map<unsigned int, NumericT>::const_iterator it = std_matrix[i].begin(); it != std_matrix[i].end(); ++it)
    {
      if (it->first < static_cast<unsigned int>(i))
        result[i] -= it->second * result[it->first];
      else
        continue;
    }
  viennacl::linalg::inplace_solve(vcl_compressed_matrix, vcl_result, viennacl::linalg::unit_lower_tag());


  if ( std::fabs(diff(result, vcl_result)) > epsilon )
  {
    std::cout << "# Error at operation: unit lower triangular solve with compressed_matrix" << std::endl;
    std::cout << "  diff: " << std::fabs(diff(result, vcl_result)) << std::endl;
    retval = EXIT_FAILURE;
  }


  std::cout << "Testing lower triangular solve: compressed_matrix" << std::endl;
  result = rhs;
  viennacl::copy(result, vcl_result);
  //boost::numeric::ublas::inplace_solve(trans(ublas_matrix_trans), result, boost::numeric::ublas::lower_tag());
  for (std::size_t i=0; i<std_matrix.size(); ++i)
  {
    NumericT diag = 0;
    for (typename std::map<unsigned int, NumericT>::const_iterator it = std_matrix[i].begin(); it != std_matrix[i].end(); ++it)
    {
      if (it->first < static_cast<unsigned int>(i))
        result[i] -= it->second * result[it->first];
      else if (it->first == static_cast<unsigned int>(i))
        diag = it->second;
      else
        continue;
    }
    result[i] /= diag;
  }
  viennacl::linalg::inplace_solve(vcl_compressed_matrix, vcl_result, viennacl::linalg::lower_tag());


  if ( std::fabs(diff(result, vcl_result)) > epsilon )
  {
    std::cout << "# Error at operation: lower triangular solve with compressed_matrix" << std::endl;
    std::cout << "  diff: " << std::fabs(diff(result, vcl_result)) << std::endl;
    retval = EXIT_FAILURE;
  }



  //
  // Triangular solvers for A^T \ b
  //
  std::vector<std::map<unsigned int, NumericT> > std_matrix_trans(std_matrix.size());

  // compute transpose:
  for (std::size_t i=0; i<std_matrix.size(); ++i)
    for (typename std::map<unsigned int, NumericT>::const_iterator it  = std_matrix[i].begin(); it != std_matrix[i].end(); ++it)
      std_matrix_trans[i][it->first] = it->second;

  std::cout << "Testing transposed unit upper triangular solve: compressed_matrix" << std::endl;
  result = rhs;
  viennacl::copy(result, vcl_result);
  //boost::numeric::ublas::inplace_solve(trans(ublas_matrix), result, boost::numeric::ublas::unit_upper_tag());
  for (std::size_t i2=0; i2<std_matrix_trans.size(); ++i2)
  {
    std::size_t row = std_matrix_trans.size() - i2 - 1;
    for (typename std::map<unsigned int, NumericT>::const_iterator it = std_matrix_trans[row].begin(); it != std_matrix_trans[row].end(); ++it)
    {
      if (it->first > static_cast<unsigned int>(row))
        result[row] -= it->second * result[it->first];
      else
        continue;
    }
  }
  viennacl::linalg::inplace_solve(trans(vcl_compressed_matrix), vcl_result, viennacl::linalg::unit_upper_tag());

  if ( std::fabs(diff(result, vcl_result)) > epsilon )
  {
    std::cout << "# Error at operation: unit upper triangular solve with compressed_matrix" << std::endl;
    std::cout << "  diff: " << std::fabs(diff(result, vcl_result)) << std::endl;
    retval = EXIT_FAILURE;
  }

  /////////////////////////

  std::cout << "Testing transposed upper triangular solve: compressed_matrix" << std::endl;
  result = rhs;
  viennacl::copy(result, vcl_result);
  //boost::numeric::ublas::inplace_solve(trans(ublas_matrix), result, boost::numeric::ublas::upper_tag());
  for (std::size_t i2=0; i2<std_matrix_trans.size(); ++i2)
  {
    std::size_t row = std_matrix_trans.size() - i2 - 1;
    NumericT diag = 0;
    for (typename std::map<unsigned int, NumericT>::const_iterator it = std_matrix_trans[row].begin(); it != std_matrix_trans[row].end(); ++it)
    {
      if (it->first > static_cast<unsigned int>(row))
        result[row] -= it->second * result[it->first];
      else if (it->first == static_cast<unsigned int>(row))
        diag = it->second;
      else
        continue;
    }
    result[row] /= diag;
  }
  viennacl::linalg::inplace_solve(trans(vcl_compressed_matrix), vcl_result, viennacl::linalg::upper_tag());

  if ( std::fabs(diff(result, vcl_result)) > epsilon )
  {
    std::cout << "# Error at operation: upper triangular solve with compressed_matrix" << std::endl;
    std::cout << "  diff: " << std::fabs(diff(result, vcl_result)) << std::endl;
    retval = EXIT_FAILURE;
  }

  /////////////////////////

  std::cout << "Testing transposed unit lower triangular solve: compressed_matrix" << std::endl;
  result = rhs;
  viennacl::copy(result, vcl_result);
  //boost::numeric::ublas::inplace_solve(trans(ublas_matrix), result, boost::numeric::ublas::unit_lower_tag());
  for (std::size_t i=1; i<std_matrix_trans.size(); ++i)
    for (typename std::map<unsigned int, NumericT>::const_iterator it = std_matrix_trans[i].begin(); it != std_matrix_trans[i].end(); ++it)
    {
      if (it->first < static_cast<unsigned int>(i))
        result[i] -= it->second * result[it->first];
      else
        continue;
    }
  viennacl::linalg::inplace_solve(trans(vcl_compressed_matrix), vcl_result, viennacl::linalg::unit_lower_tag());

  if ( std::fabs(diff(result, vcl_result)) > epsilon )
  {
    std::cout << "# Error at operation: unit lower triangular solve with compressed_matrix" << std::endl;
    std::cout << "  diff: " << std::fabs(diff(result, vcl_result)) << std::endl;
    retval = EXIT_FAILURE;
  }

  /////////////////////////

  std::cout << "Testing transposed lower triangular solve: compressed_matrix" << std::endl;
  result = rhs;
  viennacl::copy(result, vcl_result);
  //boost::numeric::ublas::inplace_solve(trans(ublas_matrix), result, boost::numeric::ublas::lower_tag());
  for (std::size_t i=0; i<std_matrix_trans.size(); ++i)
  {
    NumericT diag = 0;
    for (typename std::map<unsigned int, NumericT>::const_iterator it = std_matrix_trans[i].begin(); it != std_matrix_trans[i].end(); ++it)
    {
      if (it->first < static_cast<unsigned int>(i))
        result[i] -= it->second * result[it->first];
      else if (it->first == static_cast<unsigned int>(i))
        diag = it->second;
      else
        continue;
    }
    result[i] /= diag;
  }
  viennacl::linalg::inplace_solve(trans(vcl_compressed_matrix), vcl_result, viennacl::linalg::lower_tag());

  if ( std::fabs(diff(result, vcl_result)) > epsilon )
  {
    std::cout << "# Error at operation: lower triangular solve with compressed_matrix" << std::endl;
    std::cout << "  diff: " << std::fabs(diff(result, vcl_result)) << std::endl;
    retval = EXIT_FAILURE;
  }


  //
  /////////////////////////
  //


  std::cout << "Testing products: compressed_compressed_matrix" << std::endl;
  result     = viennacl::linalg::prod(std_cc_matrix, rhs);
  vcl_result = viennacl::linalg::prod(vcl_compressed_compressed_matrix, vcl_rhs);

  if ( std::fabs(diff(result, vcl_result)) > epsilon )
  {
    std::cout << "# Error at operation: matrix-vector product with compressed_compressed_matrix" << std::endl;
    std::cout << "  diff: " << std::fabs(diff(result, vcl_result)) << std::endl;
    retval = EXIT_FAILURE;
  }

  {
    std::vector<std::map<unsigned int, NumericT> > temp(vcl_compressed_compressed_matrix.size1());
    viennacl::copy(vcl_compressed_compressed_matrix, temp);

    // check that entries are correct by computing the product again:
    result     = viennacl::linalg::prod(temp, rhs);

    if ( std::fabs(diff(result, vcl_result)) > epsilon )
    {
      std::cout << "# Error at operation: matrix-vector product with compressed_compressed_matrix (after copy back)" << std::endl;
      std::cout << "  diff: " << std::fabs(diff(result, vcl_result)) << std::endl;
      retval = EXIT_FAILURE;
    }

  }


  //
  /////////////////////////
  //


  std::cout << "Testing products: coordinate_matrix" << std::endl;
  result     = viennacl::linalg::prod(std_matrix, rhs);
  vcl_result = viennacl::linalg::prod(vcl_coordinate_matrix, vcl_rhs);

  if ( std::fabs(diff(result, vcl_result)) > epsilon )
  {
    std::cout << "# Error at operation: matrix-vector product with coordinate_matrix" << std::endl;
    std::cout << "  diff: " << std::fabs(diff(result, vcl_result)) << std::endl;
    retval = EXIT_FAILURE;
  }

  std::cout << "Testing products: coordinate_matrix, strided vectors" << std::endl;
  //std::cout << " --> SKIPPING <--" << std::endl;
  retval = strided_matrix_vector_product_test<NumericT, viennacl::coordinate_matrix<NumericT> >(epsilon, result, rhs, vcl_result, vcl_rhs);
  if (retval != EXIT_SUCCESS)
    return retval;


  //
  /////////////////////////
  //


  //std::cout << "Copying ell_matrix" << std::endl;
  viennacl::copy(std_matrix, vcl_ell_matrix);
  std_matrix.clear();
  viennacl::copy(vcl_ell_matrix, std_matrix);// just to check that it works


  std::cout << "Testing products: ell_matrix" << std::endl;
  result     = viennacl::linalg::prod(std_matrix, rhs);
  vcl_result.clear();
  vcl_result = viennacl::linalg::prod(vcl_ell_matrix, vcl_rhs);
  //viennacl::linalg::prod_impl(vcl_ell_matrix, vcl_rhs, vcl_result);
  //std::cout << vcl_result << "\n";
  //std::cout << "  diff: " << std::fabs(diff(result, vcl_result)) << std::endl;
  //std::cout << "First entry of result vector: " << vcl_result[0] << std::endl;

  if ( std::fabs(diff(result, vcl_result)) > epsilon )
  {
    std::cout << "# Error at operation: matrix-vector product with ell_matrix" << std::endl;
    std::cout << "  diff: " << std::fabs(diff(result, vcl_result)) << std::endl;
    retval = EXIT_FAILURE;
  }

  std::cout << "Testing products: ell_matrix, strided vectors" << std::endl;
  retval = strided_matrix_vector_product_test<NumericT, viennacl::ell_matrix<NumericT> >(epsilon, result, rhs, vcl_result, vcl_rhs);
  if (retval != EXIT_SUCCESS)
    return retval;


  //
  /////////////////////////
  //


  //std::cout << "Copying sliced_ell_matrix" << std::endl;
  viennacl::copy(std_matrix, vcl_sliced_ell_matrix);

  std::cout << "Testing products: sliced_ell_matrix" << std::endl;
  result     = viennacl::linalg::prod(std_matrix, rhs);
  vcl_result.clear();
  vcl_result = viennacl::linalg::prod(vcl_sliced_ell_matrix, vcl_rhs);

  if ( std::fabs(diff(result, vcl_result)) > epsilon )
  {
    std::cout << "# Error at operation: matrix-vector product with sliced_ell_matrix" << std::endl;
    std::cout << "  diff: " << std::fabs(diff(result, vcl_result)) << std::endl;
    retval = EXIT_FAILURE;
  }

  std::cout << "Testing products: sliced_ell_matrix, strided vectors" << std::endl;
  retval = strided_matrix_vector_product_test<NumericT, viennacl::sliced_ell_matrix<NumericT> >(epsilon, result, rhs, vcl_result, vcl_rhs);
  if (retval != EXIT_SUCCESS)
    return retval;


  //
  /////////////////////////
  //


  //std::cout << "Copying hyb_matrix" << std::endl;
  viennacl::copy(std_matrix, vcl_hyb_matrix);
  std_matrix.clear();
  viennacl::copy(vcl_hyb_matrix, std_matrix);// just to check that it works
  viennacl::copy(std_matrix, vcl_hyb_matrix);

  std::cout << "Testing products: hyb_matrix" << std::endl;
  result     = viennacl::linalg::prod(std_matrix, rhs);
  vcl_result.clear();
  vcl_result = viennacl::linalg::prod(vcl_hyb_matrix, vcl_rhs);
  //viennacl::linalg::prod_impl(vcl_hyb_matrix, vcl_rhs, vcl_result);
  //std::cout << vcl_result << "\n";
  //std::cout << "  diff: " << std::fabs(diff(result, vcl_result)) << std::endl;
  //std::cout << "First entry of result vector: " << vcl_result[0] << std::endl;

  if ( std::fabs(diff(result, vcl_result)) > epsilon )
  {
    std::cout << "# Error at operation: matrix-vector product with hyb_matrix" << std::endl;
    std::cout << "  diff: " << std::fabs(diff(result, vcl_result)) << std::endl;
    retval = EXIT_FAILURE;
  }

  std::cout << "Testing products: hyb_matrix, strided vectors" << std::endl;
  retval = strided_matrix_vector_product_test<NumericT, viennacl::hyb_matrix<NumericT> >(epsilon, result, rhs, vcl_result, vcl_rhs);
  if (retval != EXIT_SUCCESS)
    return retval;


  // --------------------------------------------------------------------------
  // --------------------------------------------------------------------------
  NumericT alpha = static_cast<NumericT>(2.786);
  NumericT beta = static_cast<NumericT>(1.432);
  copy(rhs.begin(), rhs.end(), vcl_rhs.begin());
  copy(result.begin(), result.end(), vcl_result.begin());
  copy(result.begin(), result.end(), vcl_result2.begin());

  std::cout << "Testing scaled additions of products and vectors" << std::endl;
  std::vector<NumericT> result2(result);
  result2 = viennacl::linalg::prod(std_matrix, rhs);
  for (std::size_t i=0; i<result.size(); ++i)
    result[i] = alpha * result2[i] + beta * result[i];
  vcl_result2 = alpha * viennacl::linalg::prod(vcl_compressed_matrix, vcl_rhs) + beta * vcl_result;

  if ( std::fabs(diff(result, vcl_result2)) > epsilon )
  {
    std::cout << "# Error at operation: matrix-vector product (compressed_matrix) with scaled additions" << std::endl;
    std::cout << "  diff: " << std::fabs(diff(result, vcl_result2)) << std::endl;
    retval = EXIT_FAILURE;
  }


  vcl_result2.clear();
  vcl_result2 = alpha * viennacl::linalg::prod(vcl_coordinate_matrix, vcl_rhs) + beta * vcl_result;

  if ( std::fabs(diff(result, vcl_result2)) > epsilon )
  {
    std::cout << "# Error at operation: matrix-vector product (coordinate_matrix) with scaled additions" << std::endl;
    std::cout << "  diff: " << std::fabs(diff(result, vcl_result2)) << std::endl;
    retval = EXIT_FAILURE;
  }

  vcl_result2.clear();
  vcl_result2 = alpha * viennacl::linalg::prod(vcl_ell_matrix, vcl_rhs) + beta * vcl_result;

  if ( std::fabs(diff(result, vcl_result2)) > epsilon )
  {
    std::cout << "# Error at operation: matrix-vector product (ell_matrix) with scaled additions" << std::endl;
    std::cout << "  diff: " << std::fabs(diff(result, vcl_result2)) << std::endl;
    retval = EXIT_FAILURE;
  }

  vcl_result2.clear();
  vcl_result2 = alpha * viennacl::linalg::prod(vcl_hyb_matrix, vcl_rhs) + beta * vcl_result;

  if ( std::fabs(diff(result, vcl_result2)) > epsilon )
  {
    std::cout << "# Error at operation: matrix-vector product (hyb_matrix) with scaled additions" << std::endl;
    std::cout << "  diff: " << std::fabs(diff(result, vcl_result2)) << std::endl;
    retval = EXIT_FAILURE;
  }

  ////////////// Test of .clear() ////////////////
  std_matrix.clear();

  std::cout << "Testing products after clear(): compressed_matrix" << std::endl;
  vcl_compressed_matrix.clear();
  result     = std::vector<NumericT>(result.size(), NumericT(1));
  result     = viennacl::linalg::prod(std_matrix, rhs);
  vcl_result = viennacl::linalg::prod(vcl_compressed_matrix, vcl_rhs);

  if ( std::fabs(diff(result, vcl_result)) > epsilon )
  {
    std::cout << "# Error at operation: matrix-vector product with compressed_matrix" << std::endl;
    std::cout << "  diff: " << std::fabs(diff(result, vcl_result)) << std::endl;
    retval = EXIT_FAILURE;
  }

  std::cout << "Testing products after clear(): compressed_compressed_matrix" << std::endl;
  vcl_compressed_compressed_matrix.clear();
  result     = std::vector<NumericT>(result.size(), NumericT(1));
  result     = viennacl::linalg::prod(std_matrix, rhs);
  vcl_result = viennacl::linalg::prod(vcl_compressed_compressed_matrix, vcl_rhs);

  if ( std::fabs(diff(result, vcl_result)) > epsilon )
  {
    std::cout << "# Error at operation: matrix-vector product with compressed_compressed_matrix" << std::endl;
    std::cout << "  diff: " << std::fabs(diff(result, vcl_result)) << std::endl;
    retval = EXIT_FAILURE;
  }

  std::cout << "Testing products after clear(): coordinate_matrix" << std::endl;
  vcl_coordinate_matrix.clear();
  result     = std::vector<NumericT>(result.size(), NumericT(1));
  result     = viennacl::linalg::prod(std_matrix, rhs);
  vcl_result = viennacl::linalg::prod(vcl_coordinate_matrix, vcl_rhs);

  if ( std::fabs(diff(result, vcl_result)) > epsilon )
  {
    std::cout << "# Error at operation: matrix-vector product with coordinate_matrix" << std::endl;
    std::cout << "  diff: " << std::fabs(diff(result, vcl_result)) << std::endl;
    retval = EXIT_FAILURE;
  }

  std::cout << "Testing products after clear(): ell_matrix" << std::endl;
  vcl_ell_matrix.clear();
  result     = std::vector<NumericT>(result.size(), NumericT(1));
  result     = viennacl::linalg::prod(std_matrix, rhs);
  vcl_result = viennacl::linalg::prod(vcl_ell_matrix, vcl_rhs);

  if ( std::fabs(diff(result, vcl_result)) > epsilon )
  {
    std::cout << "# Error at operation: matrix-vector product with ell_matrix" << std::endl;
    std::cout << "  diff: " << std::fabs(diff(result, vcl_result)) << std::endl;
    retval = EXIT_FAILURE;
  }

  std::cout << "Testing products after clear(): hyb_matrix" << std::endl;
  vcl_hyb_matrix.clear();
  result     = std::vector<NumericT>(result.size(), NumericT(1));
  result     = viennacl::linalg::prod(std_matrix, rhs);
  vcl_result = viennacl::linalg::prod(vcl_hyb_matrix, vcl_rhs);

  if ( std::fabs(diff(result, vcl_result)) > epsilon )
  {
    std::cout << "# Error at operation: matrix-vector product with hyb_matrix" << std::endl;
    std::cout << "  diff: " << std::fabs(diff(result, vcl_result)) << std::endl;
    retval = EXIT_FAILURE;
  }

  std::cout << "Testing products after clear(): sliced_ell_matrix" << std::endl;
  vcl_sliced_ell_matrix.clear();
  result     = std::vector<NumericT>(result.size(), NumericT(1));
  result     = viennacl::linalg::prod(std_matrix, rhs);
  vcl_result = viennacl::linalg::prod(vcl_sliced_ell_matrix, vcl_rhs);

  if ( std::fabs(diff(result, vcl_result)) > epsilon )
  {
    std::cout << "# Error at operation: matrix-vector product with sliced_ell_matrix" << std::endl;
    std::cout << "  diff: " << std::fabs(diff(result, vcl_result)) << std::endl;
    retval = EXIT_FAILURE;
  }


  // --------------------------------------------------------------------------
  return retval;
}
//
// -------------------------------------------------------------
//
int main()
{
  std::cout << std::endl;
  std::cout << "----------------------------------------------" << std::endl;
  std::cout << "----------------------------------------------" << std::endl;
  std::cout << "## Test :: Sparse Matrices" << std::endl;
  std::cout << "----------------------------------------------" << std::endl;
  std::cout << "----------------------------------------------" << std::endl;
  std::cout << std::endl;

  int retval = EXIT_SUCCESS;

  std::cout << std::endl;
  std::cout << "----------------------------------------------" << std::endl;
  std::cout << std::endl;
  {
    typedef float NumericT;
    NumericT epsilon = static_cast<NumericT>(1E-4);
    std::cout << "# Testing setup:" << std::endl;
    std::cout << "  eps:     " << epsilon << std::endl;
    std::cout << "  numeric: float" << std::endl;
    retval = test<NumericT>(epsilon);
    if ( retval == EXIT_SUCCESS )
        std::cout << "# Test passed" << std::endl;
    else
        return retval;
  }
  std::cout << std::endl;
  std::cout << "----------------------------------------------" << std::endl;
  std::cout << std::endl;

#ifdef VIENNACL_WITH_OPENCL
  if ( viennacl::ocl::current_device().double_support() )
#endif
  {
    {
      typedef double NumericT;
      NumericT epsilon = 1.0E-12;
      std::cout << "# Testing setup:" << std::endl;
      std::cout << "  eps:     " << epsilon << std::endl;
      std::cout << "  numeric: double" << std::endl;
      retval = test<NumericT>(epsilon);
      if ( retval == EXIT_SUCCESS )
        std::cout << "# Test passed" << std::endl;
      else
        return retval;
    }
    std::cout << std::endl;
    std::cout << "----------------------------------------------" << std::endl;
    std::cout << std::endl;
  }
#ifdef VIENNACL_WITH_OPENCL
  else
    std::cout << "No double precision support, skipping test..." << std::endl;
#endif


  std::cout << std::endl;
  std::cout << "------- Test completed --------" << std::endl;
  std::cout << std::endl;

  return retval;
}
