/* =========================================================================
   Copyright (c) 2010-2011, Institute for Microelectronics,
                            Institute for Analysis and Scientific Computing,
                            TU Wien.

                            -----------------
                  ViennaCL - The Vienna Computing Library
                            -----------------

   Project Head:    Karl Rupp                   rupp@iue.tuwien.ac.at
               
   (A list of authors and contributors can be found in the PDF manual)

   License:         MIT (X11), see file LICENSE in the base directory
============================================================================= */

//#define NDEBUG

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

//
// *** ViennaCL
//
//#define VIENNACL_DEBUG_INFO_ALL
//#define VIENNACL_DEBUG_BUILD
#define VIENNACL_HAVE_UBLAS 1
#include "viennacl/scalar.hpp"
#include "viennacl/matrix.hpp"
#include "viennacl/vector.hpp"
#include "viennacl/matrix_proxy.hpp"
#include "viennacl/linalg/prod.hpp"
#include "viennacl/linalg/norm_2.hpp"
#include "viennacl/linalg/direct_solve.hpp"
#include "examples/tutorial/Random.hpp"
//
// -------------------------------------------------------------
//
using namespace boost::numeric;
//
// -------------------------------------------------------------
//
template <typename ScalarType>
ScalarType diff(ScalarType & s1, viennacl::scalar<ScalarType> & s2) 
{
   if (s1 != s2)
      return (s1 - s2) / std::max(fabs(s1), fabs(s2));
   return 0;
}

template <typename ScalarType>
ScalarType diff(ublas::vector<ScalarType> & v1, viennacl::vector<ScalarType> & v2)
{
   ublas::vector<ScalarType> v2_cpu(v2.size());
   copy(v2.begin(), v2.end(), v2_cpu.begin());

   for (unsigned int i=0;i<v1.size(); ++i)
   {
      if ( std::max( fabs(v2_cpu[i]), fabs(v1[i]) ) > 0 )
         v2_cpu[i] = fabs(v2_cpu[i] - v1[i]) / std::max( fabs(v2_cpu[i]), fabs(v1[i]) );
      else
         v2_cpu[i] = 0.0;
   }

   return norm_inf(v2_cpu);
}

template <typename ScalarType, typename VCLMatrixType>
ScalarType diff(ublas::matrix<ScalarType> & mat1, VCLMatrixType & mat2)
{
   ublas::matrix<ScalarType> mat2_cpu(mat2.size1(), mat2.size2());
   viennacl::copy(mat2, mat2_cpu);
   double ret = 0;
   double act = 0;

    for (unsigned int i = 0; i < mat2_cpu.size1(); ++i)
    {
      for (unsigned int j = 0; j < mat2_cpu.size2(); ++j)
      {
         act = fabs(mat2_cpu(i,j) - mat1(i,j)) / std::max( fabs(mat2_cpu(i, j)), fabs(mat1(i,j)) );
         if (act > ret)
           ret = act;
      }
    }
   //std::cout << ret << std::endl;
   return ret;
}

//
// -------------------------------------------------------------
//
template< typename NumericT, typename MatrixTypeA, typename MatrixTypeB, typename MatrixTypeC, typename Epsilon >
int test_prod(Epsilon const& epsilon)
{
   int retval = EXIT_SUCCESS;
   long matrix_size1 = 157;  //some odd number, not too large
   long matrix_size2 = 91;  //some odd number, not too large
   long matrix_size3 = 73;  //some odd number, not too large
   NumericT act_diff = 0;
   
   // --------------------------------------------------------------------------            
   ublas::matrix<NumericT> A(matrix_size1, matrix_size2);
   ublas::matrix<NumericT> B(matrix_size2, matrix_size3);
   ublas::matrix<NumericT> C(matrix_size1, matrix_size3);

   //fill A and B:
   for (unsigned int i = 0; i < A.size1(); ++i)
      for (unsigned int j = 0; j < A.size2(); ++j)
         A(i,j) = static_cast<NumericT>(0.1) * random<NumericT>();
   for (unsigned int i = 0; i < B.size1(); ++i)
      for (unsigned int j = 0; j < B.size2(); ++j)
         B(i,j) = static_cast<NumericT>(0.1) * random<NumericT>();

   ublas::matrix<NumericT> A_trans = trans(A);
   ublas::matrix<NumericT> B_trans = trans(B);
   
   MatrixTypeA vcl_A_full(3*matrix_size1, 3*matrix_size2); vcl_A_full.clear();
   MatrixTypeB vcl_B_full(3*matrix_size2, 3*matrix_size3); vcl_B_full.clear();
   MatrixTypeA vcl_A_trans_full(3*matrix_size2, 3*matrix_size1); vcl_A_trans_full.clear();
   MatrixTypeB vcl_B_trans_full(3*matrix_size3, 3*matrix_size2); vcl_B_trans_full.clear();
   MatrixTypeC vcl_C_full(3*matrix_size1, 3*matrix_size3); vcl_C_full.clear();

   viennacl::range r1(matrix_size1, 2*matrix_size1);
   viennacl::range r2(matrix_size2, 2*matrix_size2);
   viennacl::range r3(matrix_size3, 2*matrix_size3);
   viennacl::matrix_range<MatrixTypeA> vcl_A(vcl_A_full, r1, r2);
   viennacl::matrix_range<MatrixTypeB> vcl_B(vcl_B_full, r2, r3);
   viennacl::matrix_range<MatrixTypeA> vcl_A_trans(vcl_A_trans_full, r2, r1);
   viennacl::matrix_range<MatrixTypeB> vcl_B_trans(vcl_B_trans_full, r3, r2);
   viennacl::matrix_range<MatrixTypeC> vcl_C(vcl_C_full, r1, r3);

   
   viennacl::copy(A, vcl_A);
   viennacl::copy(B, vcl_B);
   viennacl::copy(A_trans, vcl_A_trans);
   viennacl::copy(B_trans, vcl_B_trans);

   // Test: C = A * B --------------------------------------------------------------------------       
   C     = viennacl::linalg::prod(A, B);
   vcl_C = viennacl::linalg::prod(vcl_A, vcl_B);
   act_diff = fabs(diff(C, vcl_C));
   
   if( act_diff > epsilon )
   {
     std::cout << "# Error at operation: matrix-matrix product" << std::endl;
     std::cout << "  diff: " << act_diff << std::endl;
     retval = EXIT_FAILURE;
   }
   else
     std::cout << "Test C = A * B passed!" << std::endl;
   
   // Test: C = A * trans(B) --------------------------------------------------------------------------       
   C     = boost::numeric::ublas::prod(A, trans(B_trans));
   vcl_C = viennacl::linalg::prod(vcl_A, trans(vcl_B_trans));
   act_diff = fabs(diff(C, vcl_C));
   
   if( act_diff > epsilon )
   {
     std::cout << "# Error at operation: matrix-matrix product" << std::endl;
     std::cout << "  diff: " << act_diff << std::endl;
     retval = EXIT_FAILURE;
   }
   else
     std::cout << "Test C = A * trans(B) passed!" << std::endl;
   
   // Test: C = trans(A) * B --------------------------------------------------------------------------       
   C     = boost::numeric::ublas::prod(trans(A_trans), B);
   vcl_C = viennacl::linalg::prod(trans(vcl_A_trans), vcl_B);
   act_diff = fabs(diff(C, vcl_C));
   
   if( act_diff > epsilon )
   {
     std::cout << "# Error at operation: matrix-matrix product" << std::endl;
     std::cout << "  diff: " << act_diff << std::endl;
     retval = EXIT_FAILURE;
   }
   else
     std::cout << "Test C = trans(A) * B passed!" << std::endl;
   
   
   // Test: C = trans(A) * trans(B) --------------------------------------------------------------------------       
   C     = boost::numeric::ublas::prod(trans(A_trans), trans(B_trans));
   vcl_C = viennacl::linalg::prod(trans(vcl_A_trans), trans(vcl_B_trans));
   act_diff = fabs(diff(C, vcl_C));
   
   if( act_diff > epsilon )
   {
     std::cout << "# Error at operation: matrix-matrix product" << std::endl;
     std::cout << "  diff: " << act_diff << std::endl;
     retval = EXIT_FAILURE;
   }
   else
     std::cout << "Test C = trans(A) * trans(B) passed!" << std::endl;
   
   
   
   return retval;
}


/*
template <typename RHSTypeRef, typename RHSTypeCheck, typename Epsilon >
void run_solver_check(RHSTypeRef & B_ref, RHSTypeCheck & B_check, int & retval, Epsilon const & epsilon)
{
   double act_diff = fabs(diff(B_ref, B_check));
   if( act_diff > epsilon )
   {
     std::cout << " FAILED!" << std::endl;
     std::cout << "# Error at operation: matrix-matrix solve" << std::endl;
     std::cout << "  diff: " << act_diff << std::endl;
     retval = EXIT_FAILURE;
   }
   else
     std::cout << " passed! " << act_diff << std::endl;
   
}

template< typename NumericT, typename MatrixTypeA, typename MatrixTypeB, typename Epsilon >
int test_solve(Epsilon const& epsilon)
{
   int retval = EXIT_SUCCESS;
   long matrix_size = 83;  //some odd number, not too large
   long rhs_num = 61;
   
   // --------------------------------------------------------------------------            
   ublas::matrix<NumericT> A(matrix_size, matrix_size);
   ublas::matrix<NumericT> B_start(matrix_size, rhs_num);
   ublas::matrix<NumericT> C_start(rhs_num, matrix_size);

   //fill A and B:
   for (unsigned int i = 0; i < A.size1(); ++i)
   {
      for (unsigned int j = 0; j < A.size2(); ++j)
         A(i,j) = static_cast<NumericT>(-0.5) * random<NumericT>();
      A(i,i) = 1.0 + 2.0 * random<NumericT>(); //some extra weight on diagonal for stability
   }
   
   for (unsigned int i = 0; i < B_start.size1(); ++i)
      for (unsigned int j = 0; j < B_start.size2(); ++j)
         B_start(i,j) = random<NumericT>();

   for (unsigned int i = 0; i < C_start.size1(); ++i)
      for (unsigned int j = 0; j < C_start.size2(); ++j)
         C_start(i,j) = random<NumericT>();
      
   ublas::matrix<NumericT> B = B_start;
   ublas::matrix<NumericT> result = B_start;
   ublas::matrix<NumericT> C = C_start;
   ublas::matrix<NumericT> A_trans = trans(A);
   ublas::matrix<NumericT> C_trans = trans(C);

   
   MatrixTypeA vcl_A(matrix_size, matrix_size);
   MatrixTypeB vcl_B(matrix_size, rhs_num);
   MatrixTypeB vcl_result(matrix_size, rhs_num);
   MatrixTypeB vcl_C(rhs_num, matrix_size);
   MatrixTypeB vcl_C_result(rhs_num, matrix_size);

   
   viennacl::copy(A, vcl_A);
   viennacl::copy(B, vcl_B);
   viennacl::copy(C, vcl_C);
   
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
} */

template< typename NumericT, typename Epsilon >
int test(Epsilon const& epsilon)
{
  int ret;

  std::cout << "--- Part 1: Testing matrix-matrix products ---" << std::endl;
  
  //
  //
  std::cout << "Now using A=row, B=row, C=row" << std::endl;
  ret = test_prod<NumericT,
             viennacl::matrix<NumericT, viennacl::row_major>,
             viennacl::matrix<NumericT, viennacl::row_major>,
             viennacl::matrix<NumericT, viennacl::row_major>  >(epsilon);
  if (ret != EXIT_SUCCESS)
    return ret;

  //
  //
  std::cout << "Now using A=row, B=row, C=column" << std::endl;
  ret = test_prod<NumericT,
             viennacl::matrix<NumericT, viennacl::row_major>,
             viennacl::matrix<NumericT, viennacl::row_major>,
             viennacl::matrix<NumericT, viennacl::column_major>  >(epsilon);
  if (ret != EXIT_SUCCESS)
    return ret;

  //
  //
  std::cout << "Now using A=row, B=column, C=row" << std::endl;
  ret = test_prod<NumericT,
             viennacl::matrix<NumericT, viennacl::row_major>,
             viennacl::matrix<NumericT, viennacl::column_major>,
             viennacl::matrix<NumericT, viennacl::row_major>  >(epsilon);
  if (ret != EXIT_SUCCESS)
    return ret;
  
  //
  //
  std::cout << "Now using A=row, B=column, C=column" << std::endl;
  ret = test_prod<NumericT,
             viennacl::matrix<NumericT, viennacl::row_major>,
             viennacl::matrix<NumericT, viennacl::column_major>,
             viennacl::matrix<NumericT, viennacl::column_major>  >(epsilon);
  if (ret != EXIT_SUCCESS)
    return ret;

  
  
  //
  //
  std::cout << "Now using A=column, B=row, C=row" << std::endl;
  ret = test_prod<NumericT,
             viennacl::matrix<NumericT, viennacl::column_major>,
             viennacl::matrix<NumericT, viennacl::row_major>,
             viennacl::matrix<NumericT, viennacl::row_major>  >(epsilon);
  if (ret != EXIT_SUCCESS)
    return ret;

  //
  //
  std::cout << "Now using A=column, B=row, C=column" << std::endl;
  ret = test_prod<NumericT,
             viennacl::matrix<NumericT, viennacl::column_major>,
             viennacl::matrix<NumericT, viennacl::row_major>,
             viennacl::matrix<NumericT, viennacl::column_major>  >(epsilon);
  if (ret != EXIT_SUCCESS)
    return ret;

  //
  //
  std::cout << "Now using A=column, B=column, C=row" << std::endl;
  ret = test_prod<NumericT,
             viennacl::matrix<NumericT, viennacl::column_major>,
             viennacl::matrix<NumericT, viennacl::column_major>,
             viennacl::matrix<NumericT, viennacl::row_major>  >(epsilon);
  if (ret != EXIT_SUCCESS)
    return ret;
  
  //
  //
  std::cout << "Now using A=column, B=column, C=column" << std::endl;
  ret = test_prod<NumericT,
             viennacl::matrix<NumericT, viennacl::column_major>,
             viennacl::matrix<NumericT, viennacl::column_major>,
             viennacl::matrix<NumericT, viennacl::column_major>  >(epsilon);
  if (ret != EXIT_SUCCESS)
    return ret;
  
  /*
  std::cout << "--- Part 2: Testing matrix-matrix solver ---" << std::endl;
  
  std::cout << "Now using A=row, B=row" << std::endl;
  ret = test_solve<NumericT,
             viennacl::matrix<NumericT, viennacl::row_major>,
             viennacl::matrix<NumericT, viennacl::row_major>  >(epsilon);
  if (ret != EXIT_SUCCESS)
    return ret;

  std::cout << "Now using A=row, B=col" << std::endl;
  ret = test_solve<NumericT,
             viennacl::matrix<NumericT, viennacl::row_major>,
             viennacl::matrix<NumericT, viennacl::column_major>  >(epsilon);
  if (ret != EXIT_SUCCESS)
    return ret;

  std::cout << "Now using A=col, B=row" << std::endl;
  ret = test_solve<NumericT,
             viennacl::matrix<NumericT, viennacl::column_major>,
             viennacl::matrix<NumericT, viennacl::row_major>  >(epsilon);
  if (ret != EXIT_SUCCESS)
    return ret;

  std::cout << "Now using A=col, B=col" << std::endl;
  ret = test_solve<NumericT,
             viennacl::matrix<NumericT, viennacl::column_major>,
             viennacl::matrix<NumericT, viennacl::column_major>  >(epsilon);
  if (ret != EXIT_SUCCESS)
    return ret; */
  
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
      NumericT epsilon = 1.0E-3;
      std::cout << "# Testing setup:" << std::endl;
      std::cout << "  eps:     " << epsilon << std::endl;
      std::cout << "  numeric: float" << std::endl;
      retval = test<NumericT>(epsilon);
      if( retval == EXIT_SUCCESS )
        std::cout << "# Test passed" << std::endl;
      else
        return retval;
   }
   std::cout << std::endl;
   std::cout << "----------------------------------------------" << std::endl;
   std::cout << std::endl;
   if( viennacl::ocl::current_device().double_support() )
   {
      {
        typedef double NumericT;
        NumericT epsilon = 1.0E-11;
        std::cout << "# Testing setup:" << std::endl;
        std::cout << "  eps:     " << epsilon << std::endl;
        std::cout << "  numeric: double" << std::endl;
        retval = test<NumericT>(epsilon);
        if( retval == EXIT_SUCCESS )
          std::cout << "# Test passed" << std::endl;
        else
          return retval;
      }
      std::cout << std::endl;
      std::cout << "----------------------------------------------" << std::endl;
      std::cout << std::endl;
   }
   return retval;
}
