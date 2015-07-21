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



/** \file tests/src/scheduler_matrix_vector.cpp  Tests the scheduler for matrix-vector-operations.
*   \test Tests the scheduler for matrix-vector-operations.
**/

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
//#define VIENNACL_DEBUG_ALL
#define VIENNACL_WITH_UBLAS 1
#include "viennacl/scalar.hpp"
#include "viennacl/matrix.hpp"
#include "viennacl/vector.hpp"
#include "viennacl/linalg/prod.hpp"
#include "viennacl/linalg/norm_2.hpp"
#include "viennacl/linalg/direct_solve.hpp"
#include "viennacl/linalg/lu.hpp"
#include "viennacl/tools/random.hpp"

#include "viennacl/scheduler/execute.hpp"
#include "viennacl/scheduler/io.hpp"

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
      return (s1 - s2) / std::max(std::fabs(s1), std::fabs(s2));
   return 0;
}

template<typename ScalarType, typename VCLVectorType>
ScalarType diff(ublas::vector<ScalarType> const & v1, VCLVectorType const & v2)
{
   ublas::vector<ScalarType> v2_cpu(v2.size());
   viennacl::backend::finish();  //workaround for a bug in APP SDK 2.7 on Trinity APUs (with Catalyst 12.8)
   viennacl::copy(v2.begin(), v2.end(), v2_cpu.begin());

   for (unsigned int i=0;i<v1.size(); ++i)
   {
      if ( std::max( std::fabs(v2_cpu[i]), std::fabs(v1[i]) ) > 0 )
         v2_cpu[i] = std::fabs(v2_cpu[i] - v1[i]) / std::max( std::fabs(v2_cpu[i]), std::fabs(v1[i]) );
      else
         v2_cpu[i] = 0.0;
   }

   return norm_inf(v2_cpu);
}

template<typename ScalarType, typename VCLMatrixType>
ScalarType diff(ublas::matrix<ScalarType> const & mat1, VCLMatrixType const & mat2)
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
// -------------------------------------------------------------
//

template<typename NumericT, typename Epsilon,
          typename UblasMatrixType, typename UblasVectorType,
          typename VCLMatrixType, typename VCLVectorType1, typename VCLVectorType2>
int test_prod_rank1(Epsilon const & epsilon,
                    UblasMatrixType & ublas_m1, UblasVectorType & ublas_v1, UblasVectorType & ublas_v2,
                    VCLMatrixType & vcl_m1, VCLVectorType1 & vcl_v1, VCLVectorType2 & vcl_v2)
{
   int retval = EXIT_SUCCESS;

   // sync data:
   viennacl::copy(ublas_v1.begin(), ublas_v1.end(), vcl_v1.begin());
   viennacl::copy(ublas_v2.begin(), ublas_v2.end(), vcl_v2.begin());
   viennacl::copy(ublas_m1, vcl_m1);

   /* TODO: Add rank-1 operations here */

   //reset vcl_matrix:
   viennacl::copy(ublas_m1, vcl_m1);

   // --------------------------------------------------------------------------
   std::cout << "Matrix-Vector product" << std::endl;
   ublas_v1 = viennacl::linalg::prod(ublas_m1, ublas_v2);
   {
   viennacl::scheduler::statement   my_statement(vcl_v1, viennacl::op_assign(), viennacl::linalg::prod(vcl_m1, vcl_v2));
   viennacl::scheduler::execute(my_statement);
   }

   if ( std::fabs(diff(ublas_v1, vcl_v1)) > epsilon )
   {
      std::cout << "# Error at operation: matrix-vector product" << std::endl;
      std::cout << "  diff: " << std::fabs(diff(ublas_v1, vcl_v1)) << std::endl;
      retval = EXIT_FAILURE;
   }

   std::cout << "Matrix-Vector product with inplace-add" << std::endl;
   ublas_v1 += viennacl::linalg::prod(ublas_m1, ublas_v2);
   {
   viennacl::scheduler::statement   my_statement(vcl_v1, viennacl::op_inplace_add(), viennacl::linalg::prod(vcl_m1, vcl_v2));
   viennacl::scheduler::execute(my_statement);
   }

   if ( std::fabs(diff(ublas_v1, vcl_v1)) > epsilon )
   {
      std::cout << "# Error at operation: matrix-vector product" << std::endl;
      std::cout << "  diff: " << std::fabs(diff(ublas_v1, vcl_v1)) << std::endl;
      retval = EXIT_FAILURE;
   }

   std::cout << "Matrix-Vector product with inplace-sub" << std::endl;
   ublas_v1 -= viennacl::linalg::prod(ublas_m1, ublas_v2);
   {
   viennacl::scheduler::statement   my_statement(vcl_v1, viennacl::op_inplace_sub(), viennacl::linalg::prod(vcl_m1, vcl_v2));
   viennacl::scheduler::execute(my_statement);
   }

   if ( std::fabs(diff(ublas_v1, vcl_v1)) > epsilon )
   {
      std::cout << "# Error at operation: matrix-vector product" << std::endl;
      std::cout << "  diff: " << std::fabs(diff(ublas_v1, vcl_v1)) << std::endl;
      retval = EXIT_FAILURE;
   }

   // --------------------------------------------------------------------------
   /*
   std::cout << "Matrix-Vector product with scaled matrix" << std::endl;
   ublas_v1 = viennacl::linalg::prod(NumericT(2.0) * ublas_m1, ublas_v2);
   {
   viennacl::scheduler::statement   my_statement(vcl_v1, viennacl::op_assign(), viennacl::linalg::prod(NumericT(2.0) * vcl_m1, vcl_v2));
   viennacl::scheduler::execute(my_statement);
   }

   if ( std::fabs(diff(ublas_v1, vcl_v1)) > epsilon )
   {
      std::cout << "# Error at operation: matrix-vector product" << std::endl;
      std::cout << "  diff: " << std::fabs(diff(ublas_v1, vcl_v1)) << std::endl;
      retval = EXIT_FAILURE;
   }*/

   // --------------------------------------------------------------------------
   std::cout << "Matrix-Vector product with scaled vector" << std::endl;
   /*
   ublas_v1 = viennacl::linalg::prod(ublas_m1, NumericT(2.0) * ublas_v2);
   {
   viennacl::scheduler::statement   my_statement(vcl_v1, viennacl::op_assign(), viennacl::linalg::prod(vcl_m1, NumericT(2.0) * vcl_v2));
   viennacl::scheduler::execute(my_statement);
   }

   if ( std::fabs(diff(ublas_v1, vcl_v1)) > epsilon )
   {
      std::cout << "# Error at operation: matrix-vector product" << std::endl;
      std::cout << "  diff: " << std::fabs(diff(ublas_v1, vcl_v1)) << std::endl;
      retval = EXIT_FAILURE;
   }*/

   // --------------------------------------------------------------------------
   std::cout << "Matrix-Vector product with scaled matrix and scaled vector" << std::endl;
   /*
   ublas_v1 = viennacl::linalg::prod(NumericT(2.0) * ublas_m1, NumericT(2.0) * ublas_v2);
   {
   viennacl::scheduler::statement   my_statement(vcl_v1, viennacl::op_assign(), viennacl::linalg::prod(NumericT(2.0) * vcl_m1, NumericT(2.0) * vcl_v2));
   viennacl::scheduler::execute(my_statement);
   }

   if ( std::fabs(diff(ublas_v1, vcl_v1)) > epsilon )
   {
      std::cout << "# Error at operation: matrix-vector product" << std::endl;
      std::cout << "  diff: " << std::fabs(diff(ublas_v1, vcl_v1)) << std::endl;
      retval = EXIT_FAILURE;
   }*/


   // --------------------------------------------------------------------------
   std::cout << "Matrix-Vector product with scaled add" << std::endl;
   NumericT alpha = static_cast<NumericT>(2.786);
   NumericT beta = static_cast<NumericT>(3.1415);
   viennacl::copy(ublas_v1.begin(), ublas_v1.end(), vcl_v1.begin());
   viennacl::copy(ublas_v2.begin(), ublas_v2.end(), vcl_v2.begin());

   ublas_v1 = alpha * viennacl::linalg::prod(ublas_m1, ublas_v2) - beta * ublas_v1;
   {
   viennacl::scheduler::statement   my_statement(vcl_v1, viennacl::op_assign(), alpha * viennacl::linalg::prod(vcl_m1, vcl_v2) - beta * vcl_v1);
   viennacl::scheduler::execute(my_statement);
   }

   if ( std::fabs(diff(ublas_v1, vcl_v1)) > epsilon )
   {
      std::cout << "# Error at operation: matrix-vector product with scaled additions" << std::endl;
      std::cout << "  diff: " << std::fabs(diff(ublas_v1, vcl_v1)) << std::endl;
      retval = EXIT_FAILURE;
   }

   std::cout << "Matrix-Vector product with scaled add, inplace-add" << std::endl;
   viennacl::copy(ublas_v1.begin(), ublas_v1.end(), vcl_v1.begin());
   viennacl::copy(ublas_v2.begin(), ublas_v2.end(), vcl_v2.begin());

   ublas_v1 += alpha * viennacl::linalg::prod(ublas_m1, ublas_v2) - beta * ublas_v1;
   {
   viennacl::scheduler::statement   my_statement(vcl_v1, viennacl::op_inplace_add(), alpha * viennacl::linalg::prod(vcl_m1, vcl_v2) - beta * vcl_v1);
   viennacl::scheduler::execute(my_statement);
   }

   if ( std::fabs(diff(ublas_v1, vcl_v1)) > epsilon )
   {
      std::cout << "# Error at operation: matrix-vector product with scaled additions" << std::endl;
      std::cout << "  diff: " << std::fabs(diff(ublas_v1, vcl_v1)) << std::endl;
      retval = EXIT_FAILURE;
   }

   std::cout << "Matrix-Vector product with scaled add, inplace-sub" << std::endl;
   viennacl::copy(ublas_v1.begin(), ublas_v1.end(), vcl_v1.begin());
   viennacl::copy(ublas_v2.begin(), ublas_v2.end(), vcl_v2.begin());

   ublas_v1 -= alpha * viennacl::linalg::prod(ublas_m1, ublas_v2) - beta * ublas_v1;
   {
   viennacl::scheduler::statement   my_statement(vcl_v1, viennacl::op_inplace_sub(), alpha * viennacl::linalg::prod(vcl_m1, vcl_v2) - beta * vcl_v1);
   viennacl::scheduler::execute(my_statement);
   }

   if ( std::fabs(diff(ublas_v1, vcl_v1)) > epsilon )
   {
      std::cout << "# Error at operation: matrix-vector product with scaled additions" << std::endl;
      std::cout << "  diff: " << std::fabs(diff(ublas_v1, vcl_v1)) << std::endl;
      retval = EXIT_FAILURE;
   }

   // --------------------------------------------------------------------------

   viennacl::copy(ublas_v1.begin(), ublas_v1.end(), vcl_v1.begin());
   viennacl::copy(ublas_v2.begin(), ublas_v2.end(), vcl_v2.begin());

   std::cout << "Transposed Matrix-Vector product" << std::endl;
   ublas_v2 = viennacl::linalg::prod(trans(ublas_m1), ublas_v1);
   {
   viennacl::scheduler::statement   my_statement(vcl_v2, viennacl::op_assign(), viennacl::linalg::prod(trans(vcl_m1), vcl_v1));
   viennacl::scheduler::execute(my_statement);
   }

   if ( std::fabs(diff(ublas_v2, vcl_v2)) > epsilon )
   {
      std::cout << "# Error at operation: transposed matrix-vector product" << std::endl;
      std::cout << "  diff: " << std::fabs(diff(ublas_v2, vcl_v2)) << std::endl;
      retval = EXIT_FAILURE;
   }

   std::cout << "Transposed Matrix-Vector product, inplace-add" << std::endl;
   ublas_v2 += viennacl::linalg::prod(trans(ublas_m1), ublas_v1);
   {
   viennacl::scheduler::statement   my_statement(vcl_v2, viennacl::op_inplace_add(), viennacl::linalg::prod(trans(vcl_m1), vcl_v1));
   viennacl::scheduler::execute(my_statement);
   }

   if ( std::fabs(diff(ublas_v2, vcl_v2)) > epsilon )
   {
      std::cout << "# Error at operation: transposed matrix-vector product" << std::endl;
      std::cout << "  diff: " << std::fabs(diff(ublas_v2, vcl_v2)) << std::endl;
      retval = EXIT_FAILURE;
   }

   std::cout << "Transposed Matrix-Vector product, inplace-sub" << std::endl;
   ublas_v2 -= viennacl::linalg::prod(trans(ublas_m1), ublas_v1);
   {
   viennacl::scheduler::statement   my_statement(vcl_v2, viennacl::op_inplace_sub(), viennacl::linalg::prod(trans(vcl_m1), vcl_v1));
   viennacl::scheduler::execute(my_statement);
   }

   if ( std::fabs(diff(ublas_v2, vcl_v2)) > epsilon )
   {
      std::cout << "# Error at operation: transposed matrix-vector product" << std::endl;
      std::cout << "  diff: " << std::fabs(diff(ublas_v2, vcl_v2)) << std::endl;
      retval = EXIT_FAILURE;
   }

   // --------------------------------------------------------------------------
   std::cout << "Transposed Matrix-Vector product with scaled add" << std::endl;
   ublas_v2 = alpha * viennacl::linalg::prod(trans(ublas_m1), ublas_v1) + beta * ublas_v2;
   {
   viennacl::scheduler::statement   my_statement(vcl_v2, viennacl::op_assign(), alpha * viennacl::linalg::prod(trans(vcl_m1), vcl_v1) + beta * vcl_v2);
   viennacl::scheduler::execute(my_statement);
   }

   if ( std::fabs(diff(ublas_v2, vcl_v2)) > epsilon )
   {
      std::cout << "# Error at operation: transposed matrix-vector product with scaled additions" << std::endl;
      std::cout << "  diff: " << std::fabs(diff(ublas_v2, vcl_v2)) << std::endl;
      retval = EXIT_FAILURE;
   }

   std::cout << "Transposed Matrix-Vector product with scaled add, inplace-add" << std::endl;
   ublas_v2 += alpha * viennacl::linalg::prod(trans(ublas_m1), ublas_v1) + beta * ublas_v2;
   {
   viennacl::scheduler::statement   my_statement(vcl_v2, viennacl::op_inplace_add(), alpha * viennacl::linalg::prod(trans(vcl_m1), vcl_v1) + beta * vcl_v2);
   viennacl::scheduler::execute(my_statement);
   }

   if ( std::fabs(diff(ublas_v2, vcl_v2)) > epsilon )
   {
      std::cout << "# Error at operation: transposed matrix-vector product with scaled additions" << std::endl;
      std::cout << "  diff: " << std::fabs(diff(ublas_v2, vcl_v2)) << std::endl;
      retval = EXIT_FAILURE;
   }

   std::cout << "Transposed Matrix-Vector product with scaled add, inplace-sub" << std::endl;
   ublas_v2 -= alpha * viennacl::linalg::prod(trans(ublas_m1), ublas_v1) + beta * ublas_v2;
   {
   viennacl::scheduler::statement   my_statement(vcl_v2, viennacl::op_inplace_sub(), alpha * viennacl::linalg::prod(trans(vcl_m1), vcl_v1) + beta * vcl_v2);
   viennacl::scheduler::execute(my_statement);
   }

   if ( std::fabs(diff(ublas_v2, vcl_v2)) > epsilon )
   {
      std::cout << "# Error at operation: transposed matrix-vector product with scaled additions" << std::endl;
      std::cout << "  diff: " << std::fabs(diff(ublas_v2, vcl_v2)) << std::endl;
      retval = EXIT_FAILURE;
   }

   // --------------------------------------------------------------------------

   return retval;
}


//
// -------------------------------------------------------------
//
template< typename NumericT, typename F, typename Epsilon >
int test(Epsilon const& epsilon)
{
   int retval = EXIT_SUCCESS;

   viennacl::tools::uniform_random_numbers<NumericT> randomNumber;

   std::size_t num_rows = 141;
   std::size_t num_cols = 79;

   // --------------------------------------------------------------------------
   ublas::vector<NumericT> ublas_v1(num_rows);
   for (std::size_t i = 0; i < ublas_v1.size(); ++i)
     ublas_v1(i) = randomNumber();
   ublas::vector<NumericT> ublas_v2 = ublas::scalar_vector<NumericT>(num_cols, NumericT(3.1415));


   ublas::matrix<NumericT> ublas_m1(ublas_v1.size(), ublas_v2.size());

   for (std::size_t i = 0; i < ublas_m1.size1(); ++i)
      for (std::size_t j = 0; j < ublas_m1.size2(); ++j)
         ublas_m1(i,j) = static_cast<NumericT>(0.1) * randomNumber();


   ublas::matrix<NumericT> ublas_m2(ublas_v1.size(), ublas_v1.size());

   for (std::size_t i = 0; i < ublas_m2.size1(); ++i)
   {
      for (std::size_t j = 0; j < ublas_m2.size2(); ++j)
         ublas_m2(i,j) = static_cast<NumericT>(-0.1) * randomNumber();
      ublas_m2(i, i) = static_cast<NumericT>(2) + randomNumber();
   }


   viennacl::vector<NumericT> vcl_v1_native(ublas_v1.size());
   viennacl::vector<NumericT> vcl_v1_large(4 * ublas_v1.size());
   viennacl::vector_range< viennacl::vector<NumericT> > vcl_v1_range(vcl_v1_large, viennacl::range(3, ublas_v1.size() + 3));
   viennacl::vector_slice< viennacl::vector<NumericT> > vcl_v1_slice(vcl_v1_large, viennacl::slice(2, 3, ublas_v1.size()));

   viennacl::vector<NumericT> vcl_v2_native(ublas_v2.size());
   viennacl::vector<NumericT> vcl_v2_large(4 * ublas_v2.size());
   viennacl::vector_range< viennacl::vector<NumericT> > vcl_v2_range(vcl_v2_large, viennacl::range(8, ublas_v2.size() + 8));
   viennacl::vector_slice< viennacl::vector<NumericT> > vcl_v2_slice(vcl_v2_large, viennacl::slice(6, 2, ublas_v2.size()));

   viennacl::matrix<NumericT, F> vcl_m1_native(ublas_m1.size1(), ublas_m1.size2());
   viennacl::matrix<NumericT, F> vcl_m1_large(4 * ublas_m1.size1(), 4 * ublas_m1.size2());
   viennacl::matrix_range< viennacl::matrix<NumericT, F> > vcl_m1_range(vcl_m1_large,
                                                                        viennacl::range(8, ublas_m1.size1() + 8),
                                                                        viennacl::range(ublas_m1.size2(), 2 * ublas_m1.size2()) );
   viennacl::matrix_slice< viennacl::matrix<NumericT, F> > vcl_m1_slice(vcl_m1_large,
                                                                        viennacl::slice(6, 2, ublas_m1.size1()),
                                                                        viennacl::slice(ublas_m1.size2(), 2, ublas_m1.size2()) );

   viennacl::matrix<NumericT, F> vcl_m2_native(ublas_m2.size1(), ublas_m2.size2());
   viennacl::matrix<NumericT, F> vcl_m2_large(4 * ublas_m2.size1(), 4 * ublas_m2.size2());
   viennacl::matrix_range< viennacl::matrix<NumericT, F> > vcl_m2_range(vcl_m2_large,
                                                                        viennacl::range(8, ublas_m2.size1() + 8),
                                                                        viennacl::range(ublas_m2.size2(), 2 * ublas_m2.size2()) );
   viennacl::matrix_slice< viennacl::matrix<NumericT, F> > vcl_m2_slice(vcl_m2_large,
                                                                        viennacl::slice(6, 2, ublas_m2.size1()),
                                                                        viennacl::slice(ublas_m2.size2(), 2, ublas_m2.size2()) );


/*   std::cout << "Matrix resizing (to larger)" << std::endl;
   matrix.resize(2*num_rows, 2*num_cols, true);
   for (unsigned int i = 0; i < matrix.size1(); ++i)
   {
      for (unsigned int j = (i<result.size() ? rhs.size() : 0); j < matrix.size2(); ++j)
         matrix(i,j) = 0;
   }
   vcl_matrix.resize(2*num_rows, 2*num_cols, true);
   viennacl::copy(vcl_matrix, matrix);
   if ( std::fabs(diff(matrix, vcl_matrix)) > epsilon )
   {
      std::cout << "# Error at operation: matrix resize (to larger)" << std::endl;
      std::cout << "  diff: " << std::fabs(diff(matrix, vcl_matrix)) << std::endl;
      return EXIT_FAILURE;
   }

   matrix(12, 14) = NumericT(1.9);
   matrix(19, 16) = NumericT(1.0);
   matrix (13, 15) =  NumericT(-9);
   vcl_matrix(12, 14) = NumericT(1.9);
   vcl_matrix(19, 16) = NumericT(1.0);
   vcl_matrix (13, 15) =  NumericT(-9);

   std::cout << "Matrix resizing (to smaller)" << std::endl;
   matrix.resize(result.size(), rhs.size(), true);
   vcl_matrix.resize(result.size(), rhs.size(), true);
   if ( std::fabs(diff(matrix, vcl_matrix)) > epsilon )
   {
      std::cout << "# Error at operation: matrix resize (to smaller)" << std::endl;
      std::cout << "  diff: " << std::fabs(diff(matrix, vcl_matrix)) << std::endl;
      return EXIT_FAILURE;
   }
   */

   //
   // Run a bunch of tests for rank-1-updates, matrix-vector products
   //
   std::cout << "------------ Testing rank-1-updates and matrix-vector products ------------------" << std::endl;

   std::cout << "* m = full, v1 = full, v2 = full" << std::endl;
   retval = test_prod_rank1<NumericT>(epsilon,
                                      ublas_m1, ublas_v1, ublas_v2,
                                      vcl_m1_native, vcl_v1_native, vcl_v2_native);
   if (retval == EXIT_FAILURE)
   {
     std::cout << " --- FAILED! ---" << std::endl;
     return retval;
   }
   else
     std::cout << " --- PASSED ---" << std::endl;


   std::cout << "* m = full, v1 = full, v2 = range" << std::endl;
   retval = test_prod_rank1<NumericT>(epsilon,
                                      ublas_m1, ublas_v1, ublas_v2,
                                      vcl_m1_native, vcl_v1_native, vcl_v2_range);
   if (retval == EXIT_FAILURE)
   {
     std::cout << " --- FAILED! ---" << std::endl;
     return retval;
   }
   else
     std::cout << " --- PASSED ---" << std::endl;


   std::cout << "* m = full, v1 = full, v2 = slice" << std::endl;
   retval = test_prod_rank1<NumericT>(epsilon,
                                      ublas_m1, ublas_v1, ublas_v2,
                                      vcl_m1_native, vcl_v1_native, vcl_v2_slice);
   if (retval == EXIT_FAILURE)
   {
     std::cout << " --- FAILED! ---" << std::endl;
     return retval;
   }
   else
     std::cout << " --- PASSED ---" << std::endl;


   // v1 = range


   std::cout << "* m = full, v1 = range, v2 = full" << std::endl;
   retval = test_prod_rank1<NumericT>(epsilon,
                                      ublas_m1, ublas_v1, ublas_v2,
                                      vcl_m1_native, vcl_v1_range, vcl_v2_native);
   if (retval == EXIT_FAILURE)
   {
     std::cout << " --- FAILED! ---" << std::endl;
     return retval;
   }
   else
     std::cout << " --- PASSED ---" << std::endl;


   std::cout << "* m = full, v1 = range, v2 = range" << std::endl;
   retval = test_prod_rank1<NumericT>(epsilon,
                                      ublas_m1, ublas_v1, ublas_v2,
                                      vcl_m1_native, vcl_v1_range, vcl_v2_range);
   if (retval == EXIT_FAILURE)
   {
     std::cout << " --- FAILED! ---" << std::endl;
     return retval;
   }
   else
     std::cout << " --- PASSED ---" << std::endl;


   std::cout << "* m = full, v1 = range, v2 = slice" << std::endl;
   retval = test_prod_rank1<NumericT>(epsilon,
                                      ublas_m1, ublas_v1, ublas_v2,
                                      vcl_m1_native, vcl_v1_range, vcl_v2_slice);
   if (retval == EXIT_FAILURE)
   {
     std::cout << " --- FAILED! ---" << std::endl;
     return retval;
   }
   else
     std::cout << " --- PASSED ---" << std::endl;



   // v1 = slice

   std::cout << "* m = full, v1 = slice, v2 = full" << std::endl;
   retval = test_prod_rank1<NumericT>(epsilon,
                                      ublas_m1, ublas_v1, ublas_v2,
                                      vcl_m1_native, vcl_v1_slice, vcl_v2_native);
   if (retval == EXIT_FAILURE)
   {
     std::cout << " --- FAILED! ---" << std::endl;
     return retval;
   }
   else
     std::cout << " --- PASSED ---" << std::endl;


   std::cout << "* m = full, v1 = slice, v2 = range" << std::endl;
   retval = test_prod_rank1<NumericT>(epsilon,
                                      ublas_m1, ublas_v1, ublas_v2,
                                      vcl_m1_native, vcl_v1_slice, vcl_v2_range);
   if (retval == EXIT_FAILURE)
   {
     std::cout << " --- FAILED! ---" << std::endl;
     return retval;
   }
   else
     std::cout << " --- PASSED ---" << std::endl;


   std::cout << "* m = full, v1 = slice, v2 = slice" << std::endl;
   retval = test_prod_rank1<NumericT>(epsilon,
                                      ublas_m1, ublas_v1, ublas_v2,
                                      vcl_m1_native, vcl_v1_slice, vcl_v2_slice);
   if (retval == EXIT_FAILURE)
   {
     std::cout << " --- FAILED! ---" << std::endl;
     return retval;
   }
   else
     std::cout << " --- PASSED ---" << std::endl;


   ///////////////////////////// matrix_range

   std::cout << "* m = range, v1 = full, v2 = full" << std::endl;
   retval = test_prod_rank1<NumericT>(epsilon,
                                      ublas_m1, ublas_v1, ublas_v2,
                                      vcl_m1_range, vcl_v1_native, vcl_v2_native);
   if (retval == EXIT_FAILURE)
   {
     std::cout << " --- FAILED! ---" << std::endl;
     return retval;
   }
   else
     std::cout << " --- PASSED ---" << std::endl;


   std::cout << "* m = range, v1 = full, v2 = range" << std::endl;
   retval = test_prod_rank1<NumericT>(epsilon,
                                      ublas_m1, ublas_v1, ublas_v2,
                                      vcl_m1_range, vcl_v1_native, vcl_v2_range);
   if (retval == EXIT_FAILURE)
   {
     std::cout << " --- FAILED! ---" << std::endl;
     return retval;
   }
   else
     std::cout << " --- PASSED ---" << std::endl;


   std::cout << "* m = range, v1 = full, v2 = slice" << std::endl;
   retval = test_prod_rank1<NumericT>(epsilon,
                                      ublas_m1, ublas_v1, ublas_v2,
                                      vcl_m1_range, vcl_v1_native, vcl_v2_slice);
   if (retval == EXIT_FAILURE)
   {
     std::cout << " --- FAILED! ---" << std::endl;
     return retval;
   }
   else
     std::cout << " --- PASSED ---" << std::endl;


   // v1 = range


   std::cout << "* m = range, v1 = range, v2 = full" << std::endl;
   retval = test_prod_rank1<NumericT>(epsilon,
                                      ublas_m1, ublas_v1, ublas_v2,
                                      vcl_m1_range, vcl_v1_range, vcl_v2_native);
   if (retval == EXIT_FAILURE)
   {
     std::cout << " --- FAILED! ---" << std::endl;
     return retval;
   }
   else
     std::cout << " --- PASSED ---" << std::endl;


   std::cout << "* m = range, v1 = range, v2 = range" << std::endl;
   retval = test_prod_rank1<NumericT>(epsilon,
                                      ublas_m1, ublas_v1, ublas_v2,
                                      vcl_m1_range, vcl_v1_range, vcl_v2_range);
   if (retval == EXIT_FAILURE)
   {
     std::cout << " --- FAILED! ---" << std::endl;
     return retval;
   }
   else
     std::cout << " --- PASSED ---" << std::endl;


   std::cout << "* m = range, v1 = range, v2 = slice" << std::endl;
   retval = test_prod_rank1<NumericT>(epsilon,
                                      ublas_m1, ublas_v1, ublas_v2,
                                      vcl_m1_range, vcl_v1_range, vcl_v2_slice);
   if (retval == EXIT_FAILURE)
   {
     std::cout << " --- FAILED! ---" << std::endl;
     return retval;
   }
   else
     std::cout << " --- PASSED ---" << std::endl;



   // v1 = slice

   std::cout << "* m = range, v1 = slice, v2 = full" << std::endl;
   retval = test_prod_rank1<NumericT>(epsilon,
                                      ublas_m1, ublas_v1, ublas_v2,
                                      vcl_m1_range, vcl_v1_slice, vcl_v2_native);
   if (retval == EXIT_FAILURE)
   {
     std::cout << " --- FAILED! ---" << std::endl;
     return retval;
   }
   else
     std::cout << " --- PASSED ---" << std::endl;


   std::cout << "* m = range, v1 = slice, v2 = range" << std::endl;
   retval = test_prod_rank1<NumericT>(epsilon,
                                      ublas_m1, ublas_v1, ublas_v2,
                                      vcl_m1_range, vcl_v1_slice, vcl_v2_range);
   if (retval == EXIT_FAILURE)
   {
     std::cout << " --- FAILED! ---" << std::endl;
     return retval;
   }
   else
     std::cout << " --- PASSED ---" << std::endl;


   std::cout << "* m = range, v1 = slice, v2 = slice" << std::endl;
   retval = test_prod_rank1<NumericT>(epsilon,
                                      ublas_m1, ublas_v1, ublas_v2,
                                      vcl_m1_range, vcl_v1_slice, vcl_v2_slice);
   if (retval == EXIT_FAILURE)
   {
     std::cout << " --- FAILED! ---" << std::endl;
     return retval;
   }
   else
     std::cout << " --- PASSED ---" << std::endl;


   ///////////////////////////// matrix_slice

   std::cout << "* m = slice, v1 = full, v2 = full" << std::endl;
   retval = test_prod_rank1<NumericT>(epsilon,
                                      ublas_m1, ublas_v1, ublas_v2,
                                      vcl_m1_slice, vcl_v1_native, vcl_v2_native);
   if (retval == EXIT_FAILURE)
   {
     std::cout << " --- FAILED! ---" << std::endl;
     return retval;
   }
   else
     std::cout << " --- PASSED ---" << std::endl;


   std::cout << "* m = slice, v1 = full, v2 = range" << std::endl;
   retval = test_prod_rank1<NumericT>(epsilon,
                                      ublas_m1, ublas_v1, ublas_v2,
                                      vcl_m1_slice, vcl_v1_native, vcl_v2_range);
   if (retval == EXIT_FAILURE)
   {
     std::cout << " --- FAILED! ---" << std::endl;
     return retval;
   }
   else
     std::cout << " --- PASSED ---" << std::endl;


   std::cout << "* m = slice, v1 = full, v2 = slice" << std::endl;
   retval = test_prod_rank1<NumericT>(epsilon,
                                      ublas_m1, ublas_v1, ublas_v2,
                                      vcl_m1_slice, vcl_v1_native, vcl_v2_slice);
   if (retval == EXIT_FAILURE)
   {
     std::cout << " --- FAILED! ---" << std::endl;
     return retval;
   }
   else
     std::cout << " --- PASSED ---" << std::endl;


   // v1 = range


   std::cout << "* m = slice, v1 = range, v2 = full" << std::endl;
   retval = test_prod_rank1<NumericT>(epsilon,
                                      ublas_m1, ublas_v1, ublas_v2,
                                      vcl_m1_slice, vcl_v1_range, vcl_v2_native);
   if (retval == EXIT_FAILURE)
   {
     std::cout << " --- FAILED! ---" << std::endl;
     return retval;
   }
   else
     std::cout << " --- PASSED ---" << std::endl;


   std::cout << "* m = slice, v1 = range, v2 = range" << std::endl;
   retval = test_prod_rank1<NumericT>(epsilon,
                                      ublas_m1, ublas_v1, ublas_v2,
                                      vcl_m1_slice, vcl_v1_range, vcl_v2_range);
   if (retval == EXIT_FAILURE)
   {
     std::cout << " --- FAILED! ---" << std::endl;
     return retval;
   }
   else
     std::cout << " --- PASSED ---" << std::endl;


   std::cout << "* m = slice, v1 = range, v2 = slice" << std::endl;
   retval = test_prod_rank1<NumericT>(epsilon,
                                      ublas_m1, ublas_v1, ublas_v2,
                                      vcl_m1_slice, vcl_v1_range, vcl_v2_slice);
   if (retval == EXIT_FAILURE)
   {
     std::cout << " --- FAILED! ---" << std::endl;
     return retval;
   }
   else
     std::cout << " --- PASSED ---" << std::endl;



   // v1 = slice

   std::cout << "* m = slice, v1 = slice, v2 = full" << std::endl;
   retval = test_prod_rank1<NumericT>(epsilon,
                                      ublas_m1, ublas_v1, ublas_v2,
                                      vcl_m1_slice, vcl_v1_slice, vcl_v2_native);
   if (retval == EXIT_FAILURE)
   {
     std::cout << " --- FAILED! ---" << std::endl;
     return retval;
   }
   else
     std::cout << " --- PASSED ---" << std::endl;


   std::cout << "* m = slice, v1 = slice, v2 = range" << std::endl;
   retval = test_prod_rank1<NumericT>(epsilon,
                                      ublas_m1, ublas_v1, ublas_v2,
                                      vcl_m1_slice, vcl_v1_slice, vcl_v2_range);
   if (retval == EXIT_FAILURE)
   {
     std::cout << " --- FAILED! ---" << std::endl;
     return retval;
   }
   else
     std::cout << " --- PASSED ---" << std::endl;


   std::cout << "* m = slice, v1 = slice, v2 = slice" << std::endl;
   retval = test_prod_rank1<NumericT>(epsilon,
                                      ublas_m1, ublas_v1, ublas_v2,
                                      vcl_m1_slice, vcl_v1_slice, vcl_v2_slice);
   if (retval == EXIT_FAILURE)
   {
     std::cout << " --- FAILED! ---" << std::endl;
     return retval;
   }
   else
     std::cout << " --- PASSED ---" << std::endl;

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
   std::cout << "## Test :: Matrix" << std::endl;
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
      std::cout << "  layout: row-major" << std::endl;
      retval = test<NumericT, viennacl::row_major>(epsilon);
      if ( retval == EXIT_SUCCESS )
         std::cout << "# Test passed" << std::endl;
      else
         return retval;
   }
   std::cout << std::endl;
   std::cout << "----------------------------------------------" << std::endl;
   std::cout << std::endl;
   {
      typedef float NumericT;
      NumericT epsilon = NumericT(1.0E-3);
      std::cout << "# Testing setup:" << std::endl;
      std::cout << "  eps:     " << epsilon << std::endl;
      std::cout << "  numeric: float" << std::endl;
      std::cout << "  layout: column-major" << std::endl;
      retval = test<NumericT, viennacl::column_major>(epsilon);
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
         std::cout << "  layout: row-major" << std::endl;
         retval = test<NumericT, viennacl::row_major>(epsilon);
            if ( retval == EXIT_SUCCESS )
               std::cout << "# Test passed" << std::endl;
            else
              return retval;
      }
      std::cout << std::endl;
      std::cout << "----------------------------------------------" << std::endl;
      std::cout << std::endl;
      {
         typedef double NumericT;
         NumericT epsilon = 1.0E-11;
         std::cout << "# Testing setup:" << std::endl;
         std::cout << "  eps:     " << epsilon << std::endl;
         std::cout << "  numeric: double" << std::endl;
         std::cout << "  layout: column-major" << std::endl;
         retval = test<NumericT, viennacl::column_major>(epsilon);
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
