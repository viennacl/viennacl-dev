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

#ifndef NDEBUG
 #define NDEBUG
#endif

//
// *** System
//
#include <iostream>

//
// *** Boost
//
#include <boost/numeric/ublas/io.hpp>
#include <boost/numeric/ublas/triangular.hpp>
#include <boost/numeric/ublas/matrix_sparse.hpp>
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/matrix_proxy.hpp>
#include <boost/numeric/ublas/lu.hpp>
#include <boost/numeric/ublas/io.hpp>
#include <boost/numeric/ublas/operation_sparse.hpp>
#include <boost/numeric/ublas/vector_proxy.hpp>

//
// *** ViennaCL
//
//#define VIENNACL_DEBUG_ALL
#define VIENNACL_WITH_UBLAS 1
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
using namespace boost::numeric;
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
ScalarType diff(ublas::vector<ScalarType> & v1, viennacl::vector<ScalarType> & v2)
{
   ublas::vector<ScalarType> v2_cpu(v2.size());
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

   return norm_inf(v2_cpu);
}


template<typename ScalarType, typename VCL_MATRIX>
ScalarType diff(ublas::compressed_matrix<ScalarType> & cpu_matrix, VCL_MATRIX & gpu_matrix)
{
  typedef ublas::compressed_matrix<ScalarType>  CPU_MATRIX;
  CPU_MATRIX from_gpu(gpu_matrix.size1(), gpu_matrix.size2());

  viennacl::backend::finish();
  viennacl::copy(gpu_matrix, from_gpu);

  ScalarType error = 0;

  //step 1: compare all entries from cpu_matrix with gpu_matrix:
  //std::cout << "Ublas matrix: " << std::endl;
  for (typename CPU_MATRIX::const_iterator1 row_it = cpu_matrix.begin1();
        row_it != cpu_matrix.end1();
        ++row_it)
  {
    //std::cout << "Row " << row_it.index1() << ": " << std::endl;
    for (typename CPU_MATRIX::const_iterator2 col_it = row_it.begin();
          col_it != row_it.end();
          ++col_it)
    {
      //std::cout << "(" << col_it.index2() << ", " << *col_it << std::endl;
      ScalarType current_error = 0;

      if ( std::max( std::fabs(cpu_matrix(col_it.index1(), col_it.index2())),
                      std::fabs(from_gpu(col_it.index1(), col_it.index2()))   ) > 0 )
        current_error = std::fabs(cpu_matrix(col_it.index1(), col_it.index2()) - from_gpu(col_it.index1(), col_it.index2()))
                          / std::max( std::fabs(cpu_matrix(col_it.index1(), col_it.index2())),
                                      std::fabs(from_gpu(col_it.index1(), col_it.index2()))   );
      if (current_error > error)
        error = current_error;
    }
  }

  //step 2: compare all entries from gpu_matrix with cpu_matrix (sparsity pattern might differ):
  //std::cout << "ViennaCL matrix: " << std::endl;
  for (typename CPU_MATRIX::const_iterator1 row_it = from_gpu.begin1();
        row_it != from_gpu.end1();
        ++row_it)
  {
    //std::cout << "Row " << row_it.index1() << ": " << std::endl;
    for (typename CPU_MATRIX::const_iterator2 col_it = row_it.begin();
          col_it != row_it.end();
          ++col_it)
    {
      //std::cout << "(" << col_it.index2() << ", " << *col_it << std::endl;
      ScalarType current_error = 0;

      if ( std::max( std::fabs(cpu_matrix(col_it.index1(), col_it.index2())),
                      std::fabs(from_gpu(col_it.index1(), col_it.index2()))   ) > 0 )
        current_error = std::fabs(cpu_matrix(col_it.index1(), col_it.index2()) - from_gpu(col_it.index1(), col_it.index2()))
                          / std::max( std::fabs(cpu_matrix(col_it.index1(), col_it.index2())),
                                      std::fabs(from_gpu(col_it.index1(), col_it.index2()))   );
      if (current_error > error)
        error = current_error;
    }
  }

  return error;
}


template<typename NumericT, typename VCL_MatrixT, typename Epsilon, typename UblasVectorT, typename VCLVectorT>
int strided_matrix_vector_product_test(Epsilon epsilon,
                                        UblasVectorT & result, UblasVectorT const & rhs,
                                        VCLVectorT & vcl_result, VCLVectorT & vcl_rhs)
{
    int retval = EXIT_SUCCESS;

    ublas::compressed_matrix<NumericT> ublas_matrix2(5, 4);
    ublas_matrix2(0, 0) = NumericT(2.0); ublas_matrix2(0, 2) = NumericT(-1.0);
    ublas_matrix2(1, 0) = NumericT(3.0); ublas_matrix2(1, 2) = NumericT(-5.0);
    ublas_matrix2(2, 1) = NumericT(5.0); ublas_matrix2(2, 2) = NumericT(-2.0);
    ublas_matrix2(3, 2) = NumericT(1.0); ublas_matrix2(3, 3) = NumericT(-6.0);
    ublas_matrix2(4, 1) = NumericT(7.0); ublas_matrix2(4, 2) = NumericT(-5.0);
    project(result, ublas::slice(1, 3, 5))     = ublas::prod(ublas_matrix2, project(rhs, ublas::slice(3, 2, 4)));

    VCL_MatrixT vcl_sparse_matrix2;
    viennacl::copy(ublas_matrix2, vcl_sparse_matrix2);
    viennacl::vector<NumericT> vec(4);
    vec(0) = rhs(3);
    vec(1) = rhs(5);
    vec(2) = rhs(7);
    vec(3) = rhs(9);
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

   ublas::compressed_matrix<NumericT> ublas_matrix(5,5);
   VCL_MATRIX vcl_matrix;

   ublas_matrix(0,0) = NumericT(10.0); ublas_matrix(0, 1) = NumericT(0.1); ublas_matrix(0, 2) = NumericT(0.2); ublas_matrix(0, 3) = NumericT(0.3); ublas_matrix(0, 4) = NumericT(0.4);
   ublas_matrix(1,0) = NumericT(1.0);  ublas_matrix(1, 1) = NumericT(1.1); ublas_matrix(1, 2) = NumericT(1.2); ublas_matrix(1, 3) = NumericT(1.3); ublas_matrix(1, 4) = NumericT(1.4);
   ublas_matrix(2,0) = NumericT(2.0);  ublas_matrix(2, 1) = NumericT(2.1); ublas_matrix(2, 2) = NumericT(2.2); ublas_matrix(2, 3) = NumericT(2.3); ublas_matrix(2, 4) = NumericT(2.4);
   ublas_matrix(3,0) = NumericT(3.0);  ublas_matrix(3, 1) = NumericT(3.1); ublas_matrix(3, 2) = NumericT(3.2); ublas_matrix(3, 3) = NumericT(3.3); ublas_matrix(3, 4) = NumericT(3.4);
   ublas_matrix(4,0) = NumericT(4.0);  ublas_matrix(4, 1) = NumericT(4.1); ublas_matrix(4, 2) = NumericT(4.2); ublas_matrix(4, 3) = NumericT(4.3); ublas_matrix(4, 4) = NumericT(4.4);

   viennacl::copy(ublas_matrix, vcl_matrix);
   ublas::compressed_matrix<NumericT> other_matrix(ublas_matrix.size1(), ublas_matrix.size2());
   viennacl::copy(vcl_matrix, other_matrix);

   std::cout << "Checking for equality after copy..." << std::endl;
    if ( std::fabs(diff(ublas_matrix, vcl_matrix)) > epsilon )
    {
        std::cout << "# Error at operation: equality after copy with sparse matrix" << std::endl;
        std::cout << "  diff: " << std::fabs(diff(ublas_matrix, vcl_matrix)) << std::endl;
        return EXIT_FAILURE;
    }

   std::cout << "Testing resize to larger..." << std::endl;
   ublas_matrix.resize(10, 10, false); //ublas does not allow preserve = true here
   ublas_matrix(0,0) = NumericT(10.0); ublas_matrix(0, 1) = NumericT(0.1); ublas_matrix(0, 2) = NumericT(0.2); ublas_matrix(0, 3) = NumericT(0.3); ublas_matrix(0, 4) = NumericT(0.4);
   ublas_matrix(1,0) = NumericT( 1.0); ublas_matrix(1, 1) = NumericT(1.1); ublas_matrix(1, 2) = NumericT(1.2); ublas_matrix(1, 3) = NumericT(1.3); ublas_matrix(1, 4) = NumericT(1.4);
   ublas_matrix(2,0) = NumericT( 2.0); ublas_matrix(2, 1) = NumericT(2.1); ublas_matrix(2, 2) = NumericT(2.2); ublas_matrix(2, 3) = NumericT(2.3); ublas_matrix(2, 4) = NumericT(2.4);
   ublas_matrix(3,0) = NumericT( 3.0); ublas_matrix(3, 1) = NumericT(3.1); ublas_matrix(3, 2) = NumericT(3.2); ublas_matrix(3, 3) = NumericT(3.3); ublas_matrix(3, 4) = NumericT(3.4);
   ublas_matrix(4,0) = NumericT( 4.0); ublas_matrix(4, 1) = NumericT(4.1); ublas_matrix(4, 2) = NumericT(4.2); ublas_matrix(4, 3) = NumericT(4.3); ublas_matrix(4, 4) = NumericT(4.4);
   //std::cout << ublas_matrix << std::endl;

   vcl_matrix.resize(10, 10, true);

    if ( std::fabs(diff(ublas_matrix, vcl_matrix)) > epsilon )
    {
        std::cout << "# Error at operation: resize (to larger) with sparse matrix" << std::endl;
        std::cout << "  diff: " << std::fabs(diff(ublas_matrix, vcl_matrix)) << std::endl;
        return EXIT_FAILURE;
    }

   ublas_matrix(5,5) = NumericT(5.5); ublas_matrix(5, 6) = NumericT(5.6); ublas_matrix(5, 7) = NumericT(5.7); ublas_matrix(5, 8) = NumericT(5.8); ublas_matrix(5, 9) = NumericT(5.9);
   ublas_matrix(6,5) = NumericT(6.5); ublas_matrix(6, 6) = NumericT(6.6); ublas_matrix(6, 7) = NumericT(6.7); ublas_matrix(6, 8) = NumericT(6.8); ublas_matrix(6, 9) = NumericT(6.9);
   ublas_matrix(7,5) = NumericT(7.5); ublas_matrix(7, 6) = NumericT(7.6); ublas_matrix(7, 7) = NumericT(7.7); ublas_matrix(7, 8) = NumericT(7.8); ublas_matrix(7, 9) = NumericT(7.9);
   ublas_matrix(8,5) = NumericT(8.5); ublas_matrix(8, 6) = NumericT(8.6); ublas_matrix(8, 7) = NumericT(8.7); ublas_matrix(8, 8) = NumericT(8.8); ublas_matrix(8, 9) = NumericT(8.9);
   ublas_matrix(9,5) = NumericT(9.5); ublas_matrix(9, 6) = NumericT(9.6); ublas_matrix(9, 7) = NumericT(9.7); ublas_matrix(9, 8) = NumericT(9.8); ublas_matrix(9, 9) = NumericT(9.9);
   viennacl::copy(ublas_matrix, vcl_matrix);

   std::cout << "Testing resize to smaller..." << std::endl;
   ublas_matrix.resize(7, 7, false); //ublas does not allow preserve = true here
   ublas_matrix(0,0) = NumericT(10.0); ublas_matrix(0, 1) = NumericT(0.1); ublas_matrix(0, 2) = NumericT(0.2); ublas_matrix(0, 3) = NumericT(0.3); ublas_matrix(0, 4) = NumericT(0.4);
   ublas_matrix(1,0) = NumericT( 1.0); ublas_matrix(1, 1) = NumericT(1.1); ublas_matrix(1, 2) = NumericT(1.2); ublas_matrix(1, 3) = NumericT(1.3); ublas_matrix(1, 4) = NumericT(1.4);
   ublas_matrix(2,0) = NumericT( 2.0); ublas_matrix(2, 1) = NumericT(2.1); ublas_matrix(2, 2) = NumericT(2.2); ublas_matrix(2, 3) = NumericT(2.3); ublas_matrix(2, 4) = NumericT(2.4);
   ublas_matrix(3,0) = NumericT( 3.0); ublas_matrix(3, 1) = NumericT(3.1); ublas_matrix(3, 2) = NumericT(3.2); ublas_matrix(3, 3) = NumericT(3.3); ublas_matrix(3, 4) = NumericT(3.4);
   ublas_matrix(4,0) = NumericT( 4.0); ublas_matrix(4, 1) = NumericT(4.1); ublas_matrix(4, 2) = NumericT(4.2); ublas_matrix(4, 3) = NumericT(4.3); ublas_matrix(4, 4) = NumericT(4.4);
   ublas_matrix(5,5) = NumericT( 5.5); ublas_matrix(5, 6) = NumericT(5.6); ublas_matrix(5, 7) = NumericT(5.7); ublas_matrix(5, 8) = NumericT(5.8); ublas_matrix(5, 9) = NumericT(5.9);
   ublas_matrix(6,5) = NumericT( 6.5); ublas_matrix(6, 6) = NumericT(6.6); ublas_matrix(6, 7) = NumericT(6.7); ublas_matrix(6, 8) = NumericT(6.8); ublas_matrix(6, 9) = NumericT(6.9);

   vcl_matrix.resize(7, 7);

   //std::cout << ublas_matrix << std::endl;
    if ( std::fabs(diff(ublas_matrix, vcl_matrix)) > epsilon )
    {
        std::cout << "# Error at operation: resize (to smaller) with sparse matrix" << std::endl;
        std::cout << "  diff: " << std::fabs(diff(ublas_matrix, vcl_matrix)) << std::endl;
        retval = EXIT_FAILURE;
    }

   ublas::vector<NumericT> ublas_vec = ublas::scalar_vector<NumericT>(ublas_matrix.size1(), NumericT(3.1415));
   viennacl::vector<NumericT> vcl_vec(ublas_matrix.size1());


  std::cout << "Testing transposed unit lower triangular solve: compressed_matrix" << std::endl;
  viennacl::copy(ublas_vec, vcl_vec);
  std::cout << "matrix: " << ublas_matrix << std::endl;
  std::cout << "vector: " << ublas_vec << std::endl;
  std::cout << "ViennaCL matrix size: " << vcl_matrix.size1() << " x " << vcl_matrix.size2() << std::endl;

  std::cout << "ublas..." << std::endl;
  boost::numeric::ublas::inplace_solve((ublas_matrix), ublas_vec, boost::numeric::ublas::unit_lower_tag());
  std::cout << "ViennaCL..." << std::endl;
  viennacl::linalg::inplace_solve((vcl_matrix), vcl_vec, viennacl::linalg::unit_lower_tag());

  /*
  std::list< viennacl::backend::mem_handle > multifrontal_L_row_index_arrays_;
  std::list< viennacl::backend::mem_handle > multifrontal_L_row_buffers_;
  std::list< viennacl::backend::mem_handle > multifrontal_L_col_buffers_;
  std::list< viennacl::backend::mem_handle > multifrontal_L_element_buffers_;
  std::list< std::size_t > multifrontal_L_row_elimination_num_list_;

  viennacl::vector<NumericT> multifrontal_U_diagonal_;

  viennacl::linalg::detail::multifrontal_setup_L(vcl_matrix,
                                                  multifrontal_U_diagonal_, //dummy
                                                  multifrontal_L_row_index_arrays_,
                                                  multifrontal_L_row_buffers_,
                                                  multifrontal_L_col_buffers_,
                                                  multifrontal_L_element_buffers_,
                                                  multifrontal_L_row_elimination_num_list_);

  viennacl::linalg::detail::multifrontal_substitute(vcl_vec,
                                                    multifrontal_L_row_index_arrays_,
                                                    multifrontal_L_row_buffers_,
                                                    multifrontal_L_col_buffers_,
                                                    multifrontal_L_element_buffers_,
                                                    multifrontal_L_row_elimination_num_list_);


  std::cout << "ublas..." << std::endl;
  boost::numeric::ublas::inplace_solve((ublas_matrix), ublas_vec, boost::numeric::ublas::upper_tag());
  std::cout << "ViennaCL..." << std::endl;
  std::list< viennacl::backend::mem_handle > multifrontal_U_row_index_arrays_;
  std::list< viennacl::backend::mem_handle > multifrontal_U_row_buffers_;
  std::list< viennacl::backend::mem_handle > multifrontal_U_col_buffers_;
  std::list< viennacl::backend::mem_handle > multifrontal_U_element_buffers_;
  std::list< std::size_t > multifrontal_U_row_elimination_num_list_;

  multifrontal_U_diagonal_.resize(vcl_matrix.size1(), false);
  viennacl::linalg::single_threaded::detail::row_info(vcl_matrix, multifrontal_U_diagonal_, viennacl::linalg::detail::SPARSE_ROW_DIAGONAL);
  viennacl::linalg::detail::multifrontal_setup_U(vcl_matrix,
                                                 multifrontal_U_diagonal_,
                                                 multifrontal_U_row_index_arrays_,
                                                 multifrontal_U_row_buffers_,
                                                 multifrontal_U_col_buffers_,
                                                 multifrontal_U_element_buffers_,
                                                 multifrontal_U_row_elimination_num_list_);

  vcl_vec = viennacl::linalg::element_div(vcl_vec, multifrontal_U_diagonal_);
  viennacl::linalg::detail::multifrontal_substitute(vcl_vec,
                                                    multifrontal_U_row_index_arrays_,
                                                    multifrontal_U_row_buffers_,
                                                    multifrontal_U_col_buffers_,
                                                    multifrontal_U_element_buffers_,
                                                    multifrontal_U_row_elimination_num_list_);
  */
  for (std::size_t i=0; i<ublas_vec.size(); ++i)
  {
    std::cout << ublas_vec[i] << " vs. " << vcl_vec[i] << std::endl;
  }

  /*std::cout << "Testing transposed unit upper triangular solve: compressed_matrix" << std::endl;
  viennacl::copy(ublas_vec, vcl_vec);
  std::cout << "matrix: " << ublas_matrix << std::endl;
  std::cout << "vector: " << ublas_vec << std::endl;
  std::cout << "ViennaCL matrix size: " << vcl_matrix.size1() << " x " << vcl_matrix.size2() << std::endl;

  std::cout << "ublas..." << std::endl;
  boost::numeric::ublas::inplace_solve((ublas_matrix), ublas_vec, boost::numeric::ublas::lower_tag());
  std::cout << "ViennaCL..." << std::endl;
  viennacl::linalg::inplace_solve((vcl_matrix), vcl_vec, viennacl::linalg::lower_tag());

  for (std::size_t i=0; i<ublas_vec.size(); ++i)
  {
    std::cout << ublas_vec[i] << " vs. " << vcl_vec[i] << std::endl;
  }*/

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
  std::cout << "Testing resizing of coordinate_matrix..." << std::endl;
  //if (retval != EXIT_FAILURE)
  //  retval = resize_test<NumericT, viennacl::coordinate_matrix<NumericT> >(epsilon);
  //else
  //  return retval;

  // --------------------------------------------------------------------------
  ublas::vector<NumericT> rhs;
  ublas::vector<NumericT> result;
  ublas::compressed_matrix<NumericT> ublas_matrix;

  if (viennacl::io::read_matrix_market_file(ublas_matrix, "../examples/testdata/mat65k.mtx") == EXIT_FAILURE)
  {
    std::cout << "Error reading Matrix file" << std::endl;
    return EXIT_FAILURE;
  }

  //unsigned int cg_mat_size = cg_mat.size();
  std::cout << "done reading matrix" << std::endl;


  rhs.resize(ublas_matrix.size2());
  for (std::size_t i=0; i<rhs.size(); ++i)
  {
    ublas_matrix(i,i) = NumericT(0.5);   // Get rid of round-off errors by making row-sums unequal to zero:
    rhs[i] = NumericT(1) + randomNumber();
  }

  // add some random numbers to the double-compressed matrix:
  ublas::compressed_matrix<NumericT> ublas_cc_matrix(ublas_matrix.size1(), ublas_matrix.size2());
  ublas_cc_matrix(42,199) = NumericT(3.1415);
  ublas_cc_matrix(31, 69) = NumericT(2.71);
  ublas_cc_matrix(23, 32) = NumericT(6);
  ublas_cc_matrix(177,57) = NumericT(4);
  ublas_cc_matrix(21, 97) = NumericT(-4);
  ublas_cc_matrix(92, 25) = NumericT(2);
  ublas_cc_matrix(89, 62) = NumericT(11);
  ublas_cc_matrix(1,   7) = NumericT(8);
  ublas_cc_matrix(85, 41) = NumericT(13);
  ublas_cc_matrix(66, 28) = NumericT(8);
  ublas_cc_matrix(21, 74) = NumericT(-2);


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
  viennacl::copy(ublas_matrix, vcl_compressed_matrix);
  viennacl::copy(ublas_cc_matrix, vcl_compressed_compressed_matrix);
  viennacl::copy(ublas_matrix, vcl_coordinate_matrix);

  // --------------------------------------------------------------------------
  std::cout << "Testing products: ublas" << std::endl;
  result     = viennacl::linalg::prod(ublas_matrix, rhs);

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
  ublas::compressed_matrix<NumericT> ublas_matrix_trans(ublas_matrix.size2(), ublas_matrix.size1(), ublas_matrix.nnz()); // = trans(ublas_matrix); //note: triangular solvers with uBLAS show atrocious performance, while transposed solvers are quite okay. To keep execution times short, we use a double-transpose-trick in the following.

  // fast transpose:
  for (typename ublas::compressed_matrix<NumericT>::iterator1 row_it  = ublas_matrix.begin1();
                                                              row_it != ublas_matrix.end1();
                                                            ++row_it)
  {
    for (typename ublas::compressed_matrix<NumericT>::iterator2 col_it  = row_it.begin();
                                                                col_it != row_it.end();
                                                              ++col_it)
    {
      ublas_matrix_trans(col_it.index1(), col_it.index2()) = *col_it;
    }
  }


  std::cout << "Testing unit upper triangular solve: compressed_matrix" << std::endl;
  result = rhs;
  viennacl::copy(result, vcl_result);
  boost::numeric::ublas::inplace_solve(trans(ublas_matrix_trans), result, boost::numeric::ublas::unit_upper_tag());
  viennacl::linalg::inplace_solve(vcl_compressed_matrix, vcl_result, viennacl::linalg::unit_upper_tag());

  if ( std::fabs(diff(result, vcl_result)) > epsilon )
  {
    std::cout << "# Error at operation: unit upper triangular solve with compressed_matrix" << std::endl;
    std::cout << "  diff: " << std::fabs(diff(result, vcl_result)) << std::endl;
    retval = EXIT_FAILURE;
  }

  std::cout << "Testing upper triangular solve: compressed_matrix" << std::endl;
  result = rhs;
  viennacl::copy(result, vcl_result);
  boost::numeric::ublas::inplace_solve(trans(ublas_matrix_trans), result, boost::numeric::ublas::upper_tag());
  viennacl::linalg::inplace_solve(vcl_compressed_matrix, vcl_result, viennacl::linalg::upper_tag());

  if ( std::fabs(diff(result, vcl_result)) > epsilon )
  {
    std::cout << "# Error at operation: upper triangular solve with compressed_matrix" << std::endl;
    std::cout << "  diff: " << std::fabs(diff(result, vcl_result)) << std::endl;
    retval = EXIT_FAILURE;
  }

  std::cout << "Testing unit lower triangular solve: compressed_matrix" << std::endl;
  result = rhs;
  viennacl::copy(result, vcl_result);
  boost::numeric::ublas::inplace_solve(trans(ublas_matrix_trans), result, boost::numeric::ublas::unit_lower_tag());
  viennacl::linalg::inplace_solve(vcl_compressed_matrix, vcl_result, viennacl::linalg::unit_lower_tag());

  /*std::list< viennacl::backend::mem_handle > multifrontal_L_row_index_arrays_;
  std::list< viennacl::backend::mem_handle > multifrontal_L_row_buffers_;
  std::list< viennacl::backend::mem_handle > multifrontal_L_col_buffers_;
  std::list< viennacl::backend::mem_handle > multifrontal_L_element_buffers_;
  std::list< std::size_t > multifrontal_L_row_elimination_num_list_;

  viennacl::vector<NumericT> multifrontal_U_diagonal_;

  viennacl::switch_memory_domain(multifrontal_U_diagonal_, viennacl::MAIN_MEMORY);
  multifrontal_U_diagonal_.resize(vcl_compressed_matrix.size1(), false);
  viennacl::linalg::single_threaded::detail::row_info(vcl_compressed_matrix, multifrontal_U_diagonal_, viennacl::linalg::detail::SPARSE_ROW_DIAGONAL);

  viennacl::linalg::detail::multifrontal_setup_L(vcl_compressed_matrix,
                                                  multifrontal_U_diagonal_, //dummy
                                                  multifrontal_L_row_index_arrays_,
                                                  multifrontal_L_row_buffers_,
                                                  multifrontal_L_col_buffers_,
                                                  multifrontal_L_element_buffers_,
                                                  multifrontal_L_row_elimination_num_list_);

  viennacl::linalg::detail::multifrontal_substitute(vcl_result,
                                                    multifrontal_L_row_index_arrays_,
                                                    multifrontal_L_row_buffers_,
                                                    multifrontal_L_col_buffers_,
                                                    multifrontal_L_element_buffers_,
                                                    multifrontal_L_row_elimination_num_list_);*/


  if ( std::fabs(diff(result, vcl_result)) > epsilon )
  {
    std::cout << "# Error at operation: unit lower triangular solve with compressed_matrix" << std::endl;
    std::cout << "  diff: " << std::fabs(diff(result, vcl_result)) << std::endl;
    retval = EXIT_FAILURE;
  }


  std::cout << "Testing lower triangular solve: compressed_matrix" << std::endl;
  result = rhs;
  viennacl::copy(result, vcl_result);
  boost::numeric::ublas::inplace_solve(trans(ublas_matrix_trans), result, boost::numeric::ublas::lower_tag());
  viennacl::linalg::inplace_solve(vcl_compressed_matrix, vcl_result, viennacl::linalg::lower_tag());

  /*std::list< viennacl::backend::mem_handle > multifrontal_U_row_index_arrays_;
  std::list< viennacl::backend::mem_handle > multifrontal_U_row_buffers_;
  std::list< viennacl::backend::mem_handle > multifrontal_U_col_buffers_;
  std::list< viennacl::backend::mem_handle > multifrontal_U_element_buffers_;
  std::list< std::size_t > multifrontal_U_row_elimination_num_list_;

  multifrontal_U_diagonal_.resize(vcl_compressed_matrix.size1(), false);
  viennacl::linalg::single_threaded::detail::row_info(vcl_compressed_matrix, multifrontal_U_diagonal_, viennacl::linalg::detail::SPARSE_ROW_DIAGONAL);
  viennacl::linalg::detail::multifrontal_setup_U(vcl_compressed_matrix,
                                                 multifrontal_U_diagonal_,
                                                 multifrontal_U_row_index_arrays_,
                                                 multifrontal_U_row_buffers_,
                                                 multifrontal_U_col_buffers_,
                                                 multifrontal_U_element_buffers_,
                                                 multifrontal_U_row_elimination_num_list_);

  vcl_result = viennacl::linalg::element_div(vcl_result, multifrontal_U_diagonal_);
  viennacl::linalg::detail::multifrontal_substitute(vcl_result,
                                                    multifrontal_U_row_index_arrays_,
                                                    multifrontal_U_row_buffers_,
                                                    multifrontal_U_col_buffers_,
                                                    multifrontal_U_element_buffers_,
                                                    multifrontal_U_row_elimination_num_list_);*/


  if ( std::fabs(diff(result, vcl_result)) > epsilon )
  {
    std::cout << "# Error at operation: lower triangular solve with compressed_matrix" << std::endl;
    std::cout << "  diff: " << std::fabs(diff(result, vcl_result)) << std::endl;
    retval = EXIT_FAILURE;
  }

/*
  std::cout << "Testing lower triangular solve: compressed_matrix" << std::endl;
  result = rhs;
  viennacl::copy(result, vcl_result);
  boost::numeric::ublas::inplace_solve(ublas_matrix, result, boost::numeric::ublas::lower_tag());
  viennacl::linalg::inplace_solve(vcl_compressed_matrix, vcl_result, viennacl::linalg::lower_tag());

  if ( std::fabs(diff(result, vcl_result)) > epsilon )
  {
    std::cout << "# Error at operation: lower triangular solve with compressed_matrix" << std::endl;
    std::cout << "  diff: " << std::fabs(diff(result, vcl_result)) << std::endl;
    retval = EXIT_FAILURE;
  }*/

  //
  // Triangular solvers for A^T \ b
  //

  std::cout << "Testing transposed unit upper triangular solve: compressed_matrix" << std::endl;
  result = rhs;
  viennacl::copy(result, vcl_result);
  boost::numeric::ublas::inplace_solve(trans(ublas_matrix), result, boost::numeric::ublas::unit_upper_tag());
  viennacl::linalg::inplace_solve(trans(vcl_compressed_matrix), vcl_result, viennacl::linalg::unit_upper_tag());

  if ( std::fabs(diff(result, vcl_result)) > epsilon )
  {
    std::cout << "# Error at operation: unit upper triangular solve with compressed_matrix" << std::endl;
    std::cout << "  diff: " << std::fabs(diff(result, vcl_result)) << std::endl;
    retval = EXIT_FAILURE;
  }

  std::cout << "Testing transposed upper triangular solve: compressed_matrix" << std::endl;
  result = rhs;
  viennacl::copy(result, vcl_result);
  boost::numeric::ublas::inplace_solve(trans(ublas_matrix), result, boost::numeric::ublas::upper_tag());
  viennacl::linalg::inplace_solve(trans(vcl_compressed_matrix), vcl_result, viennacl::linalg::upper_tag());

  if ( std::fabs(diff(result, vcl_result)) > epsilon )
  {
    std::cout << "# Error at operation: upper triangular solve with compressed_matrix" << std::endl;
    std::cout << "  diff: " << std::fabs(diff(result, vcl_result)) << std::endl;
    retval = EXIT_FAILURE;
  }


  std::cout << "Testing transposed unit lower triangular solve: compressed_matrix" << std::endl;
  result = rhs;
  viennacl::copy(result, vcl_result);
  boost::numeric::ublas::inplace_solve(trans(ublas_matrix), result, boost::numeric::ublas::unit_lower_tag());
  viennacl::linalg::inplace_solve(trans(vcl_compressed_matrix), vcl_result, viennacl::linalg::unit_lower_tag());

  if ( std::fabs(diff(result, vcl_result)) > epsilon )
  {
    std::cout << "# Error at operation: unit lower triangular solve with compressed_matrix" << std::endl;
    std::cout << "  diff: " << std::fabs(diff(result, vcl_result)) << std::endl;
    retval = EXIT_FAILURE;
  }

  std::cout << "Testing transposed lower triangular solve: compressed_matrix" << std::endl;
  result = rhs;
  viennacl::copy(result, vcl_result);
  boost::numeric::ublas::inplace_solve(trans(ublas_matrix), result, boost::numeric::ublas::lower_tag());
  viennacl::linalg::inplace_solve(trans(vcl_compressed_matrix), vcl_result, viennacl::linalg::lower_tag());

  if ( std::fabs(diff(result, vcl_result)) > epsilon )
  {
    std::cout << "# Error at operation: lower triangular solve with compressed_matrix" << std::endl;
    std::cout << "  diff: " << std::fabs(diff(result, vcl_result)) << std::endl;
    retval = EXIT_FAILURE;
  }

  std::cout << "Testing products: compressed_compressed_matrix" << std::endl;
  result     = viennacl::linalg::prod(ublas_cc_matrix, rhs);
  vcl_result = viennacl::linalg::prod(vcl_compressed_compressed_matrix, vcl_rhs);

  if ( std::fabs(diff(result, vcl_result)) > epsilon )
  {
    std::cout << "# Error at operation: matrix-vector product with compressed_compressed_matrix" << std::endl;
    std::cout << "  diff: " << std::fabs(diff(result, vcl_result)) << std::endl;
    retval = EXIT_FAILURE;
  }

  {
    ublas::compressed_matrix<NumericT> temp(vcl_compressed_compressed_matrix.size1(), vcl_compressed_compressed_matrix.size2());
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




  std::cout << "Testing products: coordinate_matrix" << std::endl;
  result     = viennacl::linalg::prod(ublas_matrix, rhs);
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


  //std::cout << "Copying ell_matrix" << std::endl;
  viennacl::copy(ublas_matrix, vcl_ell_matrix);
  ublas_matrix.clear();
  viennacl::copy(vcl_ell_matrix, ublas_matrix);// just to check that it's works


  std::cout << "Testing products: ell_matrix" << std::endl;
  result     = viennacl::linalg::prod(ublas_matrix, rhs);
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

  //std::cout << "Copying sliced_ell_matrix" << std::endl;
  viennacl::copy(ublas_matrix, vcl_sliced_ell_matrix);
  //ublas_matrix.clear();
  //viennacl::copy(vcl_hyb_matrix, ublas_matrix);// just to check that it's works
  //viennacl::copy(ublas_matrix, vcl_hyb_matrix);

  std::cout << "Testing products: sliced_ell_matrix" << std::endl;
  result     = viennacl::linalg::prod(ublas_matrix, rhs);
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


  //std::cout << "Copying hyb_matrix" << std::endl;
  viennacl::copy(ublas_matrix, vcl_hyb_matrix);
  ublas_matrix.clear();
  viennacl::copy(vcl_hyb_matrix, ublas_matrix);// just to check that it's works
  viennacl::copy(ublas_matrix, vcl_hyb_matrix);

  std::cout << "Testing products: hyb_matrix" << std::endl;
  result     = viennacl::linalg::prod(ublas_matrix, rhs);
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
  result     = alpha * viennacl::linalg::prod(ublas_matrix, rhs) + beta * result;
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
  ublas_matrix.clear();

  std::cout << "Testing products after clear(): compressed_matrix" << std::endl;
  vcl_compressed_matrix.clear();
  result     = ublas::scalar_vector<NumericT>(result.size(), NumericT(1));
  result     = viennacl::linalg::prod(ublas_matrix, rhs);
  vcl_result = viennacl::linalg::prod(vcl_compressed_matrix, vcl_rhs);

  if ( std::fabs(diff(result, vcl_result)) > epsilon )
  {
    std::cout << "# Error at operation: matrix-vector product with compressed_matrix" << std::endl;
    std::cout << "  diff: " << std::fabs(diff(result, vcl_result)) << std::endl;
    retval = EXIT_FAILURE;
  }

  std::cout << "Testing products after clear(): compressed_compressed_matrix" << std::endl;
  vcl_compressed_compressed_matrix.clear();
  result     = ublas::scalar_vector<NumericT>(result.size(), NumericT(1));
  result     = viennacl::linalg::prod(ublas_matrix, rhs);
  vcl_result = viennacl::linalg::prod(vcl_compressed_compressed_matrix, vcl_rhs);

  if ( std::fabs(diff(result, vcl_result)) > epsilon )
  {
    std::cout << "# Error at operation: matrix-vector product with compressed_compressed_matrix" << std::endl;
    std::cout << "  diff: " << std::fabs(diff(result, vcl_result)) << std::endl;
    retval = EXIT_FAILURE;
  }

  std::cout << "Testing products after clear(): coordinate_matrix" << std::endl;
  vcl_coordinate_matrix.clear();
  result     = ublas::scalar_vector<NumericT>(result.size(), NumericT(1));
  result     = viennacl::linalg::prod(ublas_matrix, rhs);
  vcl_result = viennacl::linalg::prod(vcl_coordinate_matrix, vcl_rhs);

  if ( std::fabs(diff(result, vcl_result)) > epsilon )
  {
    std::cout << "# Error at operation: matrix-vector product with coordinate_matrix" << std::endl;
    std::cout << "  diff: " << std::fabs(diff(result, vcl_result)) << std::endl;
    retval = EXIT_FAILURE;
  }

  std::cout << "Testing products after clear(): ell_matrix" << std::endl;
  vcl_ell_matrix.clear();
  result     = ublas::scalar_vector<NumericT>(result.size(), NumericT(1));
  result     = viennacl::linalg::prod(ublas_matrix, rhs);
  vcl_result = viennacl::linalg::prod(vcl_ell_matrix, vcl_rhs);

  if ( std::fabs(diff(result, vcl_result)) > epsilon )
  {
    std::cout << "# Error at operation: matrix-vector product with ell_matrix" << std::endl;
    std::cout << "  diff: " << std::fabs(diff(result, vcl_result)) << std::endl;
    retval = EXIT_FAILURE;
  }

  std::cout << "Testing products after clear(): hyb_matrix" << std::endl;
  vcl_hyb_matrix.clear();
  result     = ublas::scalar_vector<NumericT>(result.size(), NumericT(1));
  result     = viennacl::linalg::prod(ublas_matrix, rhs);
  vcl_result = viennacl::linalg::prod(vcl_hyb_matrix, vcl_rhs);

  if ( std::fabs(diff(result, vcl_result)) > epsilon )
  {
    std::cout << "# Error at operation: matrix-vector product with hyb_matrix" << std::endl;
    std::cout << "  diff: " << std::fabs(diff(result, vcl_result)) << std::endl;
    retval = EXIT_FAILURE;
  }

  std::cout << "Testing products after clear(): sliced_ell_matrix" << std::endl;
  vcl_sliced_ell_matrix.clear();
  result     = ublas::scalar_vector<NumericT>(result.size(), NumericT(1));
  result     = viennacl::linalg::prod(ublas_matrix, rhs);
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
