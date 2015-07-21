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


/** \file tests/src/blas3_solve.cpp   Tests the BLAS level 3 triangular solvers
*   \test  Tests the BLAS level 3 triangular solvers
**/

//#define NDEBUG
//#define VIENNACL_DEBUG_BUILD

//
// *** System
//
#include <iostream>

// We don't need debug mode in UBLAS:
#ifndef NDEBUG
  #define BOOST_UBLAS_NDEBUG
#endif

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

//
// *** ViennaCL
//
//#define VIENNACL_DEBUG_ALL
//#define VIENNACL_DEBUG_BUILD
#define VIENNACL_WITH_UBLAS 1
#include "viennacl/scalar.hpp"
#include "viennacl/matrix.hpp"
#include "viennacl/matrix_proxy.hpp"
#include "viennacl/vector.hpp"
#include "viennacl/linalg/prod.hpp"
#include "viennacl/linalg/norm_2.hpp"
#include "viennacl/linalg/direct_solve.hpp"
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
   viennacl::backend::finish();
   if (s1 != s2)
      return (s1 - s2) / std::max(fabs(s1), fabs(s2));
   return 0;
}

template<typename ScalarType>
ScalarType diff(ublas::vector<ScalarType> & v1, viennacl::vector<ScalarType> & v2)
{
   ublas::vector<ScalarType> v2_cpu(v2.size());
   viennacl::backend::finish();
   viennacl::copy(v2.begin(), v2.end(), v2_cpu.begin());
   viennacl::backend::finish();

   for (std::size_t i=0;i<v1.size(); ++i)
   {
      if ( std::max( fabs(v2_cpu[i]), fabs(v1[i]) ) > 0 )
         v2_cpu[i] = fabs(v2_cpu[i] - v1[i]) / std::max( fabs(v2_cpu[i]), fabs(v1[i]) );
      else
         v2_cpu[i] = 0.0;
   }

   return norm_inf(v2_cpu);
}


template<typename ScalarType, typename VCLMatrixType>
ScalarType diff(ublas::matrix<ScalarType> & mat1, VCLMatrixType & mat2)
{
   ublas::matrix<ScalarType> mat2_cpu(mat2.size1(), mat2.size2());
   viennacl::backend::finish();  //workaround for a bug in APP SDK 2.7 on Trinity APUs (with Catalyst 12.8)
   viennacl::copy(mat2, mat2_cpu);
   ScalarType ret = 0;
   ScalarType act = 0;

    for (unsigned int i = 0; i < mat2_cpu.size1(); ++i)
    {
      for (unsigned int j = 0; j < mat2_cpu.size2(); ++j)
      {
         act = std::fabs(mat2_cpu(i,j) - mat1(i,j)) / std::max( std::fabs(mat2_cpu(i, j)), std::fabs(mat1(i,j)) );
         if (act > ret)
           ret = act;
      }
    }
   //std::cout << ret << std::endl;
   return ret;
}



//
// Triangular solvers
//



template<typename RHSTypeRef, typename RHSTypeCheck, typename Epsilon >
void run_solver_check(RHSTypeRef & B_ref, RHSTypeCheck & B_check, int & retval, Epsilon const & epsilon)
{
   double act_diff = fabs(diff(B_ref, B_check));
   if ( act_diff > epsilon )
   {
     std::cout << " FAILED!" << std::endl;
     std::cout << "# Error at operation: matrix-matrix solve" << std::endl;
     std::cout << "  diff: " << act_diff << std::endl;
     retval = EXIT_FAILURE;
   }
   else
     std::cout << " passed! " << act_diff << std::endl;

}


template< typename NumericT, typename Epsilon,
          typename ReferenceMatrixTypeA, typename ReferenceMatrixTypeB, typename ReferenceMatrixTypeC,
          typename MatrixTypeA, typename MatrixTypeB, typename MatrixTypeC, typename MatrixTypeResult>
int test_solve(Epsilon const& epsilon,

              ReferenceMatrixTypeA const & A,
              ReferenceMatrixTypeB const & B_start,
              ReferenceMatrixTypeC const & C_start,

              MatrixTypeA const & vcl_A,
              MatrixTypeB & vcl_B,
              MatrixTypeC & vcl_C,
              MatrixTypeResult const &
             )
{
   int retval = EXIT_SUCCESS;

   // --------------------------------------------------------------------------

   ReferenceMatrixTypeA result;
   ReferenceMatrixTypeC C_trans;

   ReferenceMatrixTypeB B = B_start;
   ReferenceMatrixTypeC C = C_start;

   MatrixTypeResult vcl_result;

   // Test: A \ B with various tags --------------------------------------------------------------------------
   std::cout << "Testing A \\ B: " << std::endl;
   std::cout << " * upper_tag:      ";
   result = ublas::solve(A, B, ublas::upper_tag());
   vcl_result = viennacl::linalg::solve(vcl_A, vcl_B, viennacl::linalg::upper_tag());
   run_solver_check(result, vcl_result, retval, epsilon);

   std::cout << " * unit_upper_tag: ";
   result = ublas::solve(A, B, ublas::unit_upper_tag());
   vcl_result = viennacl::linalg::solve(vcl_A, vcl_B, viennacl::linalg::unit_upper_tag());
   run_solver_check(result, vcl_result, retval, epsilon);

   std::cout << " * lower_tag:      ";
   result = ublas::solve(A, B, ublas::lower_tag());
   vcl_result = viennacl::linalg::solve(vcl_A, vcl_B, viennacl::linalg::lower_tag());
   run_solver_check(result, vcl_result, retval, epsilon);

   std::cout << " * unit_lower_tag: ";
   result = ublas::solve(A, B, ublas::unit_lower_tag());
   vcl_result = viennacl::linalg::solve(vcl_A, vcl_B, viennacl::linalg::unit_lower_tag());
   run_solver_check(result, vcl_result, retval, epsilon);

   if (retval == EXIT_SUCCESS)
     std::cout << "Test A \\ B passed!" << std::endl;

   B = B_start;
   C = C_start;

   // Test: A \ B^T --------------------------------------------------------------------------
   std::cout << "Testing A \\ B^T: " << std::endl;
   std::cout << " * upper_tag:      ";
   viennacl::copy(C, vcl_C); C_trans = trans(C);
   //check solve():
   result = ublas::solve(A, C_trans, ublas::upper_tag());
   vcl_result = viennacl::linalg::solve(vcl_A, trans(vcl_C), viennacl::linalg::upper_tag());
   run_solver_check(result, vcl_result, retval, epsilon);
   //check compute kernels:
   std::cout << " * upper_tag:      ";
   ublas::inplace_solve(A, C_trans, ublas::upper_tag());
   viennacl::linalg::inplace_solve(vcl_A, trans(vcl_C), viennacl::linalg::upper_tag());
   C = trans(C_trans); run_solver_check(C, vcl_C, retval, epsilon);

   std::cout << " * unit_upper_tag: ";
   viennacl::copy(C, vcl_C); C_trans = trans(C);
   ublas::inplace_solve(A, C_trans, ublas::unit_upper_tag());
   viennacl::linalg::inplace_solve(vcl_A, trans(vcl_C), viennacl::linalg::unit_upper_tag());
   C = trans(C_trans); run_solver_check(C, vcl_C, retval, epsilon);

   std::cout << " * lower_tag:      ";
   viennacl::copy(C, vcl_C); C_trans = trans(C);
   ublas::inplace_solve(A, C_trans, ublas::lower_tag());
   viennacl::linalg::inplace_solve(vcl_A, trans(vcl_C), viennacl::linalg::lower_tag());
   C = trans(C_trans); run_solver_check(C, vcl_C, retval, epsilon);

   std::cout << " * unit_lower_tag: ";
   viennacl::copy(C, vcl_C); C_trans = trans(C);
   ublas::inplace_solve(A, C_trans, ublas::unit_lower_tag());
   viennacl::linalg::inplace_solve(vcl_A, trans(vcl_C), viennacl::linalg::unit_lower_tag());
   C = trans(C_trans); run_solver_check(C, vcl_C, retval, epsilon);

   if (retval == EXIT_SUCCESS)
     std::cout << "Test A \\ B^T passed!" << std::endl;

   B = B_start;
   C = C_start;

   // Test: A \ B with various tags --------------------------------------------------------------------------
   std::cout << "Testing A^T \\ B: " << std::endl;
   std::cout << " * upper_tag:      ";
   viennacl::copy(B, vcl_B);
   result = ublas::solve(trans(A), B, ublas::upper_tag());
   vcl_result = viennacl::linalg::solve(trans(vcl_A), vcl_B, viennacl::linalg::upper_tag());
   run_solver_check(result, vcl_result, retval, epsilon);

   std::cout << " * unit_upper_tag: ";
   viennacl::copy(B, vcl_B);
   result = ublas::solve(trans(A), B, ublas::unit_upper_tag());
   vcl_result = viennacl::linalg::solve(trans(vcl_A), vcl_B, viennacl::linalg::unit_upper_tag());
   run_solver_check(result, vcl_result, retval, epsilon);

   std::cout << " * lower_tag:      ";
   viennacl::copy(B, vcl_B);
   result = ublas::solve(trans(A), B, ublas::lower_tag());
   vcl_result = viennacl::linalg::solve(trans(vcl_A), vcl_B, viennacl::linalg::lower_tag());
   run_solver_check(result, vcl_result, retval, epsilon);

   std::cout << " * unit_lower_tag: ";
   viennacl::copy(B, vcl_B);
   result = ublas::solve(trans(A), B, ublas::unit_lower_tag());
   vcl_result = viennacl::linalg::solve(trans(vcl_A), vcl_B, viennacl::linalg::unit_lower_tag());
   run_solver_check(result, vcl_result, retval, epsilon);

   if (retval == EXIT_SUCCESS)
     std::cout << "Test A^T \\ B passed!" << std::endl;

   B = B_start;
   C = C_start;

   // Test: A^T \ B^T --------------------------------------------------------------------------
   std::cout << "Testing A^T \\ B^T: " << std::endl;
   std::cout << " * upper_tag:      ";
   viennacl::copy(C, vcl_C); C_trans = trans(C);
   //check solve():
   result = ublas::solve(trans(A), C_trans, ublas::upper_tag());
   vcl_result = viennacl::linalg::solve(trans(vcl_A), trans(vcl_C), viennacl::linalg::upper_tag());
   run_solver_check(result, vcl_result, retval, epsilon);
   //check kernels:
   std::cout << " * upper_tag:      ";
   ublas::inplace_solve(trans(A), C_trans, ublas::upper_tag());
   viennacl::linalg::inplace_solve(trans(vcl_A), trans(vcl_C), viennacl::linalg::upper_tag());
   C = trans(C_trans); run_solver_check(C, vcl_C, retval, epsilon);

   std::cout << " * unit_upper_tag: ";
   viennacl::copy(C, vcl_C); C_trans = trans(C);
   ublas::inplace_solve(trans(A), C_trans, ublas::unit_upper_tag());
   viennacl::linalg::inplace_solve(trans(vcl_A), trans(vcl_C), viennacl::linalg::unit_upper_tag());
   C = trans(C_trans); run_solver_check(C, vcl_C, retval, epsilon);

   std::cout << " * lower_tag:      ";
   viennacl::copy(C, vcl_C); C_trans = trans(C);
   ublas::inplace_solve(trans(A), C_trans, ublas::lower_tag());
   viennacl::linalg::inplace_solve(trans(vcl_A), trans(vcl_C), viennacl::linalg::lower_tag());
   C = trans(C_trans); run_solver_check(C, vcl_C, retval, epsilon);

   std::cout << " * unit_lower_tag: ";
   viennacl::copy(C, vcl_C); C_trans = trans(C);
   ublas::inplace_solve(trans(A), C_trans, ublas::unit_lower_tag());
   viennacl::linalg::inplace_solve(trans(vcl_A), trans(vcl_C), viennacl::linalg::unit_lower_tag());
   C = trans(C_trans); run_solver_check(C, vcl_C, retval, epsilon);

   if (retval == EXIT_SUCCESS)
     std::cout << "Test A^T \\ B^T passed!" << std::endl;

   return retval;
}


template< typename NumericT, typename F_A, typename F_B, typename Epsilon >
int test_solve(Epsilon const& epsilon)
{
  viennacl::tools::uniform_random_numbers<NumericT> randomNumber;

  int ret = EXIT_SUCCESS;
  std::size_t matrix_size = 135;  //some odd number, not too large
  std::size_t rhs_num = 67;

  std::cout << "--- Part 2: Testing matrix-matrix solver ---" << std::endl;


  ublas::matrix<NumericT> A(matrix_size, matrix_size);
  ublas::matrix<NumericT> B_start(matrix_size, rhs_num);
  ublas::matrix<NumericT> C_start(rhs_num, matrix_size);

  for (std::size_t i = 0; i < A.size1(); ++i)
  {
    for (std::size_t j = 0; j < A.size2(); ++j)
        A(i,j) = static_cast<NumericT>(-0.5) * randomNumber();
    A(i,i) = NumericT(1.0) + NumericT(2.0) * randomNumber(); //some extra weight on diagonal for stability
  }

  for (std::size_t i = 0; i < B_start.size1(); ++i)
    for (std::size_t j = 0; j < B_start.size2(); ++j)
        B_start(i,j) = randomNumber();

  for (std::size_t i = 0; i < C_start.size1(); ++i)
    for (std::size_t j = 0; j < C_start.size2(); ++j)
        C_start(i,j) = randomNumber();


  // A
  viennacl::range range1_A(matrix_size, 2*matrix_size);
  viennacl::range range2_A(2*matrix_size, 3*matrix_size);
  viennacl::slice slice1_A(matrix_size, 2, matrix_size);
  viennacl::slice slice2_A(0, 3, matrix_size);

  viennacl::matrix<NumericT, F_A>    vcl_A(matrix_size, matrix_size);
  viennacl::copy(A, vcl_A);

  viennacl::matrix<NumericT, F_A>    vcl_big_range_A(4*matrix_size, 4*matrix_size);
  viennacl::matrix_range<viennacl::matrix<NumericT, F_A> > vcl_range_A(vcl_big_range_A, range1_A, range2_A);
  viennacl::copy(A, vcl_range_A);

  viennacl::matrix<NumericT, F_A>    vcl_big_slice_A(4*matrix_size, 4*matrix_size);
  viennacl::matrix_slice<viennacl::matrix<NumericT, F_A> > vcl_slice_A(vcl_big_slice_A, slice1_A, slice2_A);
  viennacl::copy(A, vcl_slice_A);


  // B
  viennacl::range range1_B(matrix_size, 2*matrix_size);
  viennacl::range range2_B(2*rhs_num, 3*rhs_num);
  viennacl::slice slice1_B(matrix_size, 2, matrix_size);
  viennacl::slice slice2_B(0, 3, rhs_num);

  viennacl::matrix<NumericT, F_B>    vcl_B(matrix_size, rhs_num);
  viennacl::copy(B_start, vcl_B);

  viennacl::matrix<NumericT, F_B>    vcl_big_range_B(4*matrix_size, 4*rhs_num);
  viennacl::matrix_range<viennacl::matrix<NumericT, F_B> > vcl_range_B(vcl_big_range_B, range1_B, range2_B);
  viennacl::copy(B_start, vcl_range_B);

  viennacl::matrix<NumericT, F_B>    vcl_big_slice_B(4*matrix_size, 4*rhs_num);
  viennacl::matrix_slice<viennacl::matrix<NumericT, F_B> > vcl_slice_B(vcl_big_slice_B, slice1_B, slice2_B);
  viennacl::copy(B_start, vcl_slice_B);


  // C
  viennacl::range range1_C(rhs_num, 2*rhs_num);
  viennacl::range range2_C(2*matrix_size, 3*matrix_size);
  viennacl::slice slice1_C(rhs_num, 2, rhs_num);
  viennacl::slice slice2_C(0, 3, matrix_size);

  viennacl::matrix<NumericT, F_B>    vcl_C(rhs_num, matrix_size);
  viennacl::copy(C_start, vcl_C);

  viennacl::matrix<NumericT, F_B>    vcl_big_range_C(4*rhs_num, 4*matrix_size);
  viennacl::matrix_range<viennacl::matrix<NumericT, F_B> > vcl_range_C(vcl_big_range_C, range1_C, range2_C);
  viennacl::copy(C_start, vcl_range_C);

  viennacl::matrix<NumericT, F_B>    vcl_big_slice_C(4*rhs_num, 4*matrix_size);
  viennacl::matrix_slice<viennacl::matrix<NumericT, F_B> > vcl_slice_C(vcl_big_slice_C, slice1_C, slice2_C);
  viennacl::copy(C_start, vcl_slice_C);


  std::cout << "Now using A=matrix, B=matrix" << std::endl;
  ret = test_solve<NumericT>(epsilon,
                             A, B_start, C_start,
                             vcl_A, vcl_B, vcl_C, vcl_B
                            );
  if (ret != EXIT_SUCCESS)
    return ret;

  std::cout << "Now using A=matrix, B=range" << std::endl;
  ret = test_solve<NumericT>(epsilon,
                             A, B_start, C_start,
                             vcl_A, vcl_range_B, vcl_range_C, vcl_B
                            );
  if (ret != EXIT_SUCCESS)
    return ret;

  std::cout << "Now using A=matrix, B=slice" << std::endl;
  ret = test_solve<NumericT>(epsilon,
                             A, B_start, C_start,
                             vcl_A, vcl_slice_B, vcl_slice_C, vcl_B
                            );
  if (ret != EXIT_SUCCESS)
    return ret;



  std::cout << "Now using A=range, B=matrix" << std::endl;
  ret = test_solve<NumericT>(epsilon,
                             A, B_start, C_start,
                             vcl_range_A, vcl_B, vcl_C, vcl_B
                            );
  if (ret != EXIT_SUCCESS)
    return ret;

  std::cout << "Now using A=range, B=range" << std::endl;
  ret = test_solve<NumericT>(epsilon,
                             A, B_start, C_start,
                             vcl_range_A, vcl_range_B, vcl_range_C, vcl_B
                            );
  if (ret != EXIT_SUCCESS)
    return ret;

  std::cout << "Now using A=range, B=slice" << std::endl;
  ret = test_solve<NumericT>(epsilon,
                             A, B_start, C_start,
                             vcl_range_A, vcl_slice_B, vcl_slice_C, vcl_B
                            );
  if (ret != EXIT_SUCCESS)
    return ret;




  std::cout << "Now using A=slice, B=matrix" << std::endl;
  ret = test_solve<NumericT>(epsilon,
                             A, B_start, C_start,
                             vcl_slice_A, vcl_B, vcl_C, vcl_B
                            );
  if (ret != EXIT_SUCCESS)
    return ret;

  std::cout << "Now using A=slice, B=range" << std::endl;
  ret = test_solve<NumericT>(epsilon,
                             A, B_start, C_start,
                             vcl_slice_A, vcl_range_B, vcl_range_C, vcl_B
                            );
  if (ret != EXIT_SUCCESS)
    return ret;

  std::cout << "Now using A=slice, B=slice" << std::endl;
  ret = test_solve<NumericT>(epsilon,
                             A, B_start, C_start,
                             vcl_slice_A, vcl_slice_B, vcl_slice_C, vcl_B
                            );
  if (ret != EXIT_SUCCESS)
    return ret;




  return ret;

}



//
// Control functions
//


template< typename NumericT, typename Epsilon >
int test(Epsilon const& epsilon)
{
  int ret;

  std::cout << "////////////////////////////////" << std::endl;
  std::cout << "/// Now testing A=row, B=row ///" << std::endl;
  std::cout << "////////////////////////////////" << std::endl;
  ret = test_solve<NumericT, viennacl::row_major, viennacl::row_major>(epsilon);
  if (ret != EXIT_SUCCESS)
    return ret;


  std::cout << "////////////////////////////////" << std::endl;
  std::cout << "/// Now testing A=row, B=col ///" << std::endl;
  std::cout << "////////////////////////////////" << std::endl;
  ret = test_solve<NumericT, viennacl::row_major, viennacl::column_major>(epsilon);
  if (ret != EXIT_SUCCESS)
    return ret;

  std::cout << "////////////////////////////////" << std::endl;
  std::cout << "/// Now testing A=col, B=row ///" << std::endl;
  std::cout << "////////////////////////////////" << std::endl;
  ret = test_solve<NumericT, viennacl::column_major, viennacl::row_major>(epsilon);
  if (ret != EXIT_SUCCESS)
    return ret;

  std::cout << "////////////////////////////////" << std::endl;
  std::cout << "/// Now testing A=col, B=col ///" << std::endl;
  std::cout << "////////////////////////////////" << std::endl;
  ret = test_solve<NumericT, viennacl::column_major, viennacl::column_major>(epsilon);
  if (ret != EXIT_SUCCESS)
    return ret;



  return ret;
}

//
// -------------------------------------------------------------
//
int main()
{
   std::cout << std::endl;
   std::cout << "----------------------------------------------" << std::endl;
   std::cout << "----------------------------------------------" << std::endl;
   std::cout << "## Test :: BLAS 3 routines" << std::endl;
   std::cout << "----------------------------------------------" << std::endl;
   std::cout << "----------------------------------------------" << std::endl;
   std::cout << std::endl;

   int retval = EXIT_SUCCESS;

   std::cout << std::endl;
   std::cout << "----------------------------------------------" << std::endl;
   std::cout << std::endl;
   {
      typedef float NumericT;
      NumericT epsilon = NumericT(1.0E-3);
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
        NumericT epsilon = 1.0E-11;
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

   std::cout << std::endl;
   std::cout << "------- Test completed --------" << std::endl;
   std::cout << std::endl;


   return retval;
}
