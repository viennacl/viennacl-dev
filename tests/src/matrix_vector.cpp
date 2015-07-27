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


/** \file tests/src/matrix_vector.cpp  Tests routines for matrix-vector operaions (BLAS level 2) using floating point arithmetic.
*   \test Tests routines for matrix-vector operaions (BLAS level 2) using floating point arithmetic.
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
#include <boost/numeric/ublas/vector_proxy.hpp>

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
#include "viennacl/linalg/sum.hpp"
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
                    UblasMatrixType & ublas_m1, UblasVectorType & ublas_v1, UblasVectorType & ublas_v2, UblasMatrixType & ublas_m2,
                    VCLMatrixType & vcl_m1, VCLVectorType1 & vcl_v1, VCLVectorType2 & vcl_v2, VCLMatrixType & vcl_m2)
{
   int retval = EXIT_SUCCESS;

   // sync data:
   ublas_v1 = ublas::scalar_vector<NumericT>(ublas_v1.size(), NumericT(0.1234));
   ublas_v2 = ublas::scalar_vector<NumericT>(ublas_v2.size(), NumericT(0.4321));
   viennacl::copy(ublas_v1.begin(), ublas_v1.end(), vcl_v1.begin());
   viennacl::copy(ublas_v2.begin(), ublas_v2.end(), vcl_v2.begin());
   viennacl::copy(ublas_m1, vcl_m1);

   // --------------------------------------------------------------------------
   std::cout << "Rank 1 update" << std::endl;

   ublas_m1 += ublas::outer_prod(ublas_v1, ublas_v2);
   vcl_m1 += viennacl::linalg::outer_prod(vcl_v1, vcl_v2);
   if ( std::fabs(diff(ublas_m1, vcl_m1)) > epsilon )
   {
      std::cout << "# Error at operation: rank 1 update" << std::endl;
      std::cout << "  diff: " << std::fabs(diff(ublas_m1, vcl_m1)) << std::endl;
      return EXIT_FAILURE;
   }



   // --------------------------------------------------------------------------
   std::cout << "Scaled rank 1 update - CPU Scalar" << std::endl;
   ublas_m1 += NumericT(4.2) * ublas::outer_prod(ublas_v1, ublas_v2);
   vcl_m1 += NumericT(2.1) * viennacl::linalg::outer_prod(vcl_v1, vcl_v2);
   vcl_m1 += viennacl::linalg::outer_prod(vcl_v1, vcl_v2) * NumericT(2.1);  //check proper compilation
   if ( std::fabs(diff(ublas_m1, vcl_m1)) > epsilon )
   {
      std::cout << "# Error at operation: scaled rank 1 update - CPU Scalar" << std::endl;
      std::cout << "  diff: " << std::fabs(diff(ublas_m1, vcl_m1)) << std::endl;
      return EXIT_FAILURE;
   }

      // --------------------------------------------------------------------------
   std::cout << "Scaled rank 1 update - GPU Scalar" << std::endl;
   ublas_m1 += NumericT(4.2) * ublas::outer_prod(ublas_v1, ublas_v2);
   vcl_m1 += viennacl::scalar<NumericT>(NumericT(2.1)) * viennacl::linalg::outer_prod(vcl_v1, vcl_v2);
   vcl_m1 += viennacl::linalg::outer_prod(vcl_v1, vcl_v2) * viennacl::scalar<NumericT>(NumericT(2.1));  //check proper compilation
   if ( std::fabs(diff(ublas_m1, vcl_m1)) > epsilon )
   {
      std::cout << "# Error at operation: scaled rank 1 update - GPU Scalar" << std::endl;
      std::cout << "  diff: " << std::fabs(diff(ublas_m1, vcl_m1)) << std::endl;
      return EXIT_FAILURE;
   }

   //reset vcl_matrix:
   viennacl::copy(ublas_m1, vcl_m1);

   // --------------------------------------------------------------------------
   std::cout << "Matrix-Vector product" << std::endl;
   ublas_v1 = viennacl::linalg::prod(ublas_m1, ublas_v2);
   vcl_v1   = viennacl::linalg::prod(vcl_m1, vcl_v2);

   if ( std::fabs(diff(ublas_v1, vcl_v1)) > epsilon )
   {
      std::cout << "# Error at operation: matrix-vector product" << std::endl;
      std::cout << "  diff: " << std::fabs(diff(ublas_v1, vcl_v1)) << std::endl;
      retval = EXIT_FAILURE;
   }
   // --------------------------------------------------------------------------
   std::cout << "Matrix-Vector product with scaled add" << std::endl;
   NumericT alpha = static_cast<NumericT>(2.786);
   NumericT beta = static_cast<NumericT>(1.432);
   viennacl::copy(ublas_v1.begin(), ublas_v1.end(), vcl_v1.begin());
   viennacl::copy(ublas_v2.begin(), ublas_v2.end(), vcl_v2.begin());

   ublas_v1 = alpha * viennacl::linalg::prod(ublas_m1, ublas_v2) + beta * ublas_v1;
   vcl_v1   = alpha * viennacl::linalg::prod(vcl_m1, vcl_v2) + beta * vcl_v1;

   if ( std::fabs(diff(ublas_v1, vcl_v1)) > epsilon )
   {
      std::cout << "# Error at operation: matrix-vector product with scaled additions" << std::endl;
      std::cout << "  diff: " << std::fabs(diff(ublas_v1, vcl_v1)) << std::endl;
      retval = EXIT_FAILURE;
   }
   // --------------------------------------------------------------------------
   std::cout << "Matrix-Vector product with matrix expression" << std::endl;
   ublas_v1 = ublas::prod(ublas_m1 + ublas_m1, ublas_v2);
   vcl_v1   = viennacl::linalg::prod(vcl_m1 + vcl_m1, vcl_v2);

   if ( std::fabs(diff(ublas_v1, vcl_v1)) > epsilon )
   {
      std::cout << "# Error at operation: matrix-vector product" << std::endl;
      std::cout << "  diff: " << std::fabs(diff(ublas_v1, vcl_v1)) << std::endl;
      retval = EXIT_FAILURE;
   }
   // --------------------------------------------------------------------------
   std::cout << "Matrix-Vector product with vector expression" << std::endl;
   ublas_v1 = ublas::prod(ublas_m1, NumericT(3) * ublas_v2);
   vcl_v1   = viennacl::linalg::prod(vcl_m1, NumericT(3) * vcl_v2);

   if ( std::fabs(diff(ublas_v1, vcl_v1)) > epsilon )
   {
      std::cout << "# Error at operation: matrix-vector product" << std::endl;
      std::cout << "  diff: " << std::fabs(diff(ublas_v1, vcl_v1)) << std::endl;
      retval = EXIT_FAILURE;
   }
   // --------------------------------------------------------------------------
   std::cout << "Matrix-Vector product with matrix and vector expression" << std::endl;
   ublas_v1 = ublas::prod(ublas_m1 + ublas_m1, ublas_v2 + ublas_v2);
   vcl_v1   = viennacl::linalg::prod(vcl_m1 + vcl_m1, vcl_v2 + vcl_v2);

   if ( std::fabs(diff(ublas_v1, vcl_v1)) > epsilon )
   {
      std::cout << "# Error at operation: matrix-vector product" << std::endl;
      std::cout << "  diff: " << std::fabs(diff(ublas_v1, vcl_v1)) << std::endl;
      retval = EXIT_FAILURE;
   }
   // --------------------------------------------------------------------------

   viennacl::copy(ublas_v1.begin(), ublas_v1.end(), vcl_v1.begin());
   viennacl::copy(ublas_v2.begin(), ublas_v2.end(), vcl_v2.begin());

   std::cout << "Transposed Matrix-Vector product" << std::endl;
   ublas_v2 = alpha * viennacl::linalg::prod(trans(ublas_m1), ublas_v1);
   vcl_v2   = alpha * viennacl::linalg::prod(trans(vcl_m1), vcl_v1);

   if ( std::fabs(diff(ublas_v2, vcl_v2)) > epsilon )
   {
      std::cout << "# Error at operation: transposed matrix-vector product" << std::endl;
      std::cout << "  diff: " << std::fabs(diff(ublas_v2, vcl_v2)) << std::endl;
      retval = EXIT_FAILURE;
   }

   std::cout << "Transposed Matrix-Vector product with scaled add" << std::endl;
   ublas_v2 = alpha * viennacl::linalg::prod(trans(ublas_m1), ublas_v1) + beta * ublas_v2;
   vcl_v2   = alpha * viennacl::linalg::prod(trans(vcl_m1), vcl_v1) + beta * vcl_v2;

   if ( std::fabs(diff(ublas_v2, vcl_v2)) > epsilon )
   {
      std::cout << "# Error at operation: transposed matrix-vector product with scaled additions" << std::endl;
      std::cout << "  diff: " << std::fabs(diff(ublas_v2, vcl_v2)) << std::endl;
      retval = EXIT_FAILURE;
   }
   // --------------------------------------------------------------------------

   std::cout << "Row sum with matrix" << std::endl;
   ublas_v1 = ublas::prod(ublas_m1, ublas::scalar_vector<NumericT>(ublas_m1.size2(), NumericT(1)));
   vcl_v1   = viennacl::linalg::row_sum(vcl_m1);

   if ( std::fabs(diff(ublas_v1, vcl_v1)) > epsilon )
   {
      std::cout << "# Error at operation: row sum" << std::endl;
      std::cout << "  diff: " << std::fabs(diff(ublas_v1, vcl_v1)) << std::endl;
      retval = EXIT_FAILURE;
   }
   // --------------------------------------------------------------------------

   std::cout << "Row sum with matrix expression" << std::endl;
   ublas_v1 = ublas::prod(ublas_m1 + ublas_m1, ublas::scalar_vector<NumericT>(ublas_m1.size2(), NumericT(1)));
   vcl_v1   = viennacl::linalg::row_sum(vcl_m1 + vcl_m1);

   if ( std::fabs(diff(ublas_v1, vcl_v1)) > epsilon )
   {
      std::cout << "# Error at operation: row sum (with expression)" << std::endl;
      std::cout << "  diff: " << std::fabs(diff(ublas_v1, vcl_v1)) << std::endl;
      retval = EXIT_FAILURE;
   }
   // --------------------------------------------------------------------------

   std::cout << "Column sum with matrix" << std::endl;
   ublas_v2 = ublas::prod(trans(ublas_m1), ublas::scalar_vector<NumericT>(ublas_m1.size1(), NumericT(1)));
   vcl_v2   = viennacl::linalg::column_sum(vcl_m1);

   if ( std::fabs(diff(ublas_v2, vcl_v2)) > epsilon )
   {
      std::cout << "# Error at operation: column sum" << std::endl;
      std::cout << "  diff: " << std::fabs(diff(ublas_v2, vcl_v2)) << std::endl;
      retval = EXIT_FAILURE;
   }
   // --------------------------------------------------------------------------

   std::cout << "Column sum with matrix expression" << std::endl;
   ublas_v2 = ublas::prod(trans(ublas_m1 + ublas_m1), ublas::scalar_vector<NumericT>(ublas_m1.size1(), NumericT(1)));
   vcl_v2   = viennacl::linalg::column_sum(vcl_m1 + vcl_m1);

   if ( std::fabs(diff(ublas_v2, vcl_v2)) > epsilon )
   {
      std::cout << "# Error at operation: column sum (with expression)" << std::endl;
      std::cout << "  diff: " << std::fabs(diff(ublas_v2, vcl_v2)) << std::endl;
      retval = EXIT_FAILURE;
   }
   // --------------------------------------------------------------------------


   viennacl::copy(ublas_v1.begin(), ublas_v1.end(), vcl_v1.begin());
   viennacl::copy(ublas_v2.begin(), ublas_v2.end(), vcl_v2.begin());

   std::cout << "Row extraction from matrix" << std::endl;
   ublas_v2 = row(ublas_m1, std::size_t(7));
   vcl_v2   = row(vcl_m1, std::size_t(7));

   if ( std::fabs(diff(ublas_v2, vcl_v2)) > epsilon )
   {
      std::cout << "# Error at operation: diagonal extraction from matrix" << std::endl;
      std::cout << "  diff: " << std::fabs(diff(ublas_v2, vcl_v2)) << std::endl;
      retval = EXIT_FAILURE;
   }

   std::cout << "Column extraction from matrix" << std::endl;
   ublas_v1 = column(ublas_m1, std::size_t(7));
   vcl_v1   = column(vcl_m1, std::size_t(7));

   if ( std::fabs(diff(ublas_v1, vcl_v1)) > epsilon )
   {
      std::cout << "# Error at operation: diagonal extraction from matrix" << std::endl;
      std::cout << "  diff: " << std::fabs(diff(ublas_v2, vcl_v2)) << std::endl;
      retval = EXIT_FAILURE;
   }

   // --------------------------------------------------------------------------

   viennacl::copy(ublas_v1.begin(), ublas_v1.end(), vcl_v1.begin());
   viennacl::copy(ublas_v2.begin(), ublas_v2.end(), vcl_v2.begin());
   viennacl::copy(ublas_m2, vcl_m2);
   UblasMatrixType A = ublas_m2;

   std::cout << "Diagonal extraction from matrix" << std::endl;
   for (std::size_t i=0; i<ublas_m1.size2(); ++i)
     ublas_v2[i] = ublas_m1(i + 3, i);
   vcl_v2   = diag(vcl_m1, static_cast<int>(-3));

   if ( std::fabs(diff(ublas_v2, vcl_v2)) > epsilon )
   {
      std::cout << "# Error at operation: diagonal extraction from matrix" << std::endl;
      std::cout << "  diff: " << std::fabs(diff(ublas_v2, vcl_v2)) << std::endl;
      retval = EXIT_FAILURE;
   }

   std::cout << "Matrix diagonal assignment from vector" << std::endl;
   A = ublas::scalar_matrix<NumericT>(A.size1(), A.size2(), NumericT(0));
   for (std::size_t i=0; i<ublas_m1.size2(); ++i)
     A(i + (A.size1() - ublas_m1.size2()), i) = ublas_v2[i];
   vcl_m2 = diag(vcl_v2, static_cast<int>(ublas_m1.size2()) - static_cast<int>(A.size1()));

   if ( std::fabs(diff(A, vcl_m2)) > epsilon )
   {
      std::cout << "# Error at operation: Matrix assignment from diagonal" << std::endl;
      std::cout << "  diff: " << std::fabs(diff(A, vcl_m2)) << std::endl;
      retval = EXIT_FAILURE;
   }
   // --------------------------------------------------------------------------

   return retval;
}



template<typename NumericT, typename Epsilon,
          typename UblasMatrixType, typename UblasVectorType,
          typename VCLMatrixType, typename VCLVectorType1>
int test_solve(Epsilon const & epsilon,
               UblasMatrixType & ublas_m1, UblasVectorType & ublas_v1,
               VCLMatrixType & vcl_m1, VCLVectorType1 & vcl_v1)
{
   int retval = EXIT_SUCCESS;

   // sync data:
   //viennacl::copy(ublas_v1.begin(), ublas_v1.end(), vcl_v1.begin());
   viennacl::copy(ublas_v1, vcl_v1);
   viennacl::copy(ublas_m1, vcl_m1);

   /////////////////// test direct solvers ////////////////////////////

   //upper triangular:
   std::cout << "Upper triangular solver" << std::endl;
   ublas_v1 = ublas::solve(ublas_m1, ublas_v1, ublas::upper_tag());
   vcl_v1 = viennacl::linalg::solve(vcl_m1, vcl_v1, viennacl::linalg::upper_tag());
   if ( std::fabs(diff(ublas_v1, vcl_v1)) > epsilon )
   {
      std::cout << "# Error at operation: upper triangular solver" << std::endl;
      std::cout << "  diff: " << std::fabs(diff(ublas_v1, vcl_v1)) << std::endl;
      retval = EXIT_FAILURE;
   }

   //upper unit triangular:
   std::cout << "Upper unit triangular solver" << std::endl;
   viennacl::copy(ublas_v1, vcl_v1);
   ublas_v1 = ublas::solve(ublas_m1, ublas_v1, ublas::unit_upper_tag());
   vcl_v1 = viennacl::linalg::solve(vcl_m1, vcl_v1, viennacl::linalg::unit_upper_tag());
   if ( std::fabs(diff(ublas_v1, vcl_v1)) > epsilon )
   {
      std::cout << "# Error at operation: unit upper triangular solver" << std::endl;
      std::cout << "  diff: " << std::fabs(diff(ublas_v1, vcl_v1)) << std::endl;
      retval = EXIT_FAILURE;
   }

   //lower triangular:
   std::cout << "Lower triangular solver" << std::endl;
   viennacl::copy(ublas_v1, vcl_v1);
   ublas_v1 = ublas::solve(ublas_m1, ublas_v1, ublas::lower_tag());
   vcl_v1 = viennacl::linalg::solve(vcl_m1, vcl_v1, viennacl::linalg::lower_tag());
   if ( std::fabs(diff(ublas_v1, vcl_v1)) > epsilon )
   {
      std::cout << "# Error at operation: lower triangular solver" << std::endl;
      std::cout << "  diff: " << std::fabs(diff(ublas_v1, vcl_v1)) << std::endl;
      retval = EXIT_FAILURE;
   }

   //lower unit triangular:
   std::cout << "Lower unit triangular solver" << std::endl;
   viennacl::copy(ublas_v1, vcl_v1);
   ublas_v1 = ublas::solve(ublas_m1, ublas_v1, ublas::unit_lower_tag());
   vcl_v1 = viennacl::linalg::solve(vcl_m1, vcl_v1, viennacl::linalg::unit_lower_tag());
   if ( std::fabs(diff(ublas_v1, vcl_v1)) > epsilon )
   {
      std::cout << "# Error at operation: unit lower triangular solver" << std::endl;
      std::cout << "  diff: " << std::fabs(diff(ublas_v1, vcl_v1)) << std::endl;
      retval = EXIT_FAILURE;
   }





   //transposed upper triangular:
   std::cout << "Transposed upper triangular solver" << std::endl;
   viennacl::copy(ublas_v1, vcl_v1);
   ublas_v1 = ublas::solve(trans(ublas_m1), ublas_v1, ublas::upper_tag());
   vcl_v1 = viennacl::linalg::solve(trans(vcl_m1), vcl_v1, viennacl::linalg::upper_tag());
   if ( std::fabs(diff(ublas_v1, vcl_v1)) > epsilon )
   {
      std::cout << "# Error at operation: upper triangular solver" << std::endl;
      std::cout << "  diff: " << std::fabs(diff(ublas_v1, vcl_v1)) << std::endl;
      retval = EXIT_FAILURE;
   }

   //transposed upper unit triangular:
   std::cout << "Transposed unit upper triangular solver" << std::endl;
   viennacl::copy(ublas_v1, vcl_v1);
   ublas_v1 = ublas::solve(trans(ublas_m1), ublas_v1, ublas::unit_upper_tag());
   vcl_v1 = viennacl::linalg::solve(trans(vcl_m1), vcl_v1, viennacl::linalg::unit_upper_tag());
   if ( std::fabs(diff(ublas_v1, vcl_v1)) > epsilon )
   {
      std::cout << "# Error at operation: unit upper triangular solver" << std::endl;
      std::cout << "  diff: " << std::fabs(diff(ublas_v1, vcl_v1)) << std::endl;
      retval = EXIT_FAILURE;
   }

   //transposed lower triangular:
   std::cout << "Transposed lower triangular solver" << std::endl;
   viennacl::copy(ublas_v1, vcl_v1);
   ublas_v1 = ublas::solve(trans(ublas_m1), ublas_v1, ublas::lower_tag());
   vcl_v1 = viennacl::linalg::solve(trans(vcl_m1), vcl_v1, viennacl::linalg::lower_tag());
   if ( std::fabs(diff(ublas_v1, vcl_v1)) > epsilon )
   {
      std::cout << "# Error at operation: lower triangular solver" << std::endl;
      std::cout << "  diff: " << std::fabs(diff(ublas_v1, vcl_v1)) << std::endl;
      retval = EXIT_FAILURE;
   }

   //transposed lower unit triangular:
   std::cout << "Transposed unit lower triangular solver" << std::endl;
   viennacl::copy(ublas_v1, vcl_v1);
   ublas_v1 = ublas::solve(trans(ublas_m1), ublas_v1, ublas::unit_lower_tag());
   vcl_v1 = viennacl::linalg::solve(trans(vcl_m1), vcl_v1, viennacl::linalg::unit_lower_tag());
   if ( std::fabs(diff(ublas_v1, vcl_v1)) > epsilon )
   {
      std::cout << "# Error at operation: unit lower triangular solver" << std::endl;
      std::cout << "  diff: " << std::fabs(diff(ublas_v1, vcl_v1)) << std::endl;
      retval = EXIT_FAILURE;
   }

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

   std::size_t num_rows = 141; //note: use num_rows > num_cols + 3 for diag() tests to work
   std::size_t num_cols = 103;

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
                                      ublas_m1, ublas_v1, ublas_v2, ublas_m2,
                                      vcl_m1_native, vcl_v1_native, vcl_v2_native, vcl_m2_native);
   if (retval == EXIT_FAILURE)
   {
     std::cout << " --- FAILED! ---" << std::endl;
     return retval;
   }
   else
     std::cout << " --- PASSED ---" << std::endl;


   std::cout << "* m = full, v1 = full, v2 = range" << std::endl;
   retval = test_prod_rank1<NumericT>(epsilon,
                                      ublas_m1, ublas_v1, ublas_v2, ublas_m2,
                                      vcl_m1_native, vcl_v1_native, vcl_v2_range, vcl_m2_native);
   if (retval == EXIT_FAILURE)
   {
     std::cout << " --- FAILED! ---" << std::endl;
     return retval;
   }
   else
     std::cout << " --- PASSED ---" << std::endl;


   std::cout << "* m = full, v1 = full, v2 = slice" << std::endl;
   retval = test_prod_rank1<NumericT>(epsilon,
                                      ublas_m1, ublas_v1, ublas_v2, ublas_m2,
                                      vcl_m1_native, vcl_v1_native, vcl_v2_slice, vcl_m2_native);
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
                                      ublas_m1, ublas_v1, ublas_v2, ublas_m2,
                                      vcl_m1_native, vcl_v1_range, vcl_v2_native, vcl_m2_native);
   if (retval == EXIT_FAILURE)
   {
     std::cout << " --- FAILED! ---" << std::endl;
     return retval;
   }
   else
     std::cout << " --- PASSED ---" << std::endl;


   std::cout << "* m = full, v1 = range, v2 = range" << std::endl;
   retval = test_prod_rank1<NumericT>(epsilon,
                                      ublas_m1, ublas_v1, ublas_v2, ublas_m2,
                                      vcl_m1_native, vcl_v1_range, vcl_v2_range, vcl_m2_native);
   if (retval == EXIT_FAILURE)
   {
     std::cout << " --- FAILED! ---" << std::endl;
     return retval;
   }
   else
     std::cout << " --- PASSED ---" << std::endl;


   std::cout << "* m = full, v1 = range, v2 = slice" << std::endl;
   retval = test_prod_rank1<NumericT>(epsilon,
                                      ublas_m1, ublas_v1, ublas_v2, ublas_m2,
                                      vcl_m1_native, vcl_v1_range, vcl_v2_slice, vcl_m2_native);
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
                                      ublas_m1, ublas_v1, ublas_v2, ublas_m2,
                                      vcl_m1_native, vcl_v1_slice, vcl_v2_native, vcl_m2_native);
   if (retval == EXIT_FAILURE)
   {
     std::cout << " --- FAILED! ---" << std::endl;
     return retval;
   }
   else
     std::cout << " --- PASSED ---" << std::endl;


   std::cout << "* m = full, v1 = slice, v2 = range" << std::endl;
   retval = test_prod_rank1<NumericT>(epsilon,
                                      ublas_m1, ublas_v1, ublas_v2, ublas_m2,
                                      vcl_m1_native, vcl_v1_slice, vcl_v2_range, vcl_m2_native);
   if (retval == EXIT_FAILURE)
   {
     std::cout << " --- FAILED! ---" << std::endl;
     return retval;
   }
   else
     std::cout << " --- PASSED ---" << std::endl;


   std::cout << "* m = full, v1 = slice, v2 = slice" << std::endl;
   retval = test_prod_rank1<NumericT>(epsilon,
                                      ublas_m1, ublas_v1, ublas_v2, ublas_m2,
                                      vcl_m1_native, vcl_v1_slice, vcl_v2_slice, vcl_m2_native);
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
                                      ublas_m1, ublas_v1, ublas_v2, ublas_m2,
                                      vcl_m1_range, vcl_v1_native, vcl_v2_native, vcl_m2_range);
   if (retval == EXIT_FAILURE)
   {
     std::cout << " --- FAILED! ---" << std::endl;
     return retval;
   }
   else
     std::cout << " --- PASSED ---" << std::endl;


   std::cout << "* m = range, v1 = full, v2 = range" << std::endl;
   retval = test_prod_rank1<NumericT>(epsilon,
                                      ublas_m1, ublas_v1, ublas_v2, ublas_m2,
                                      vcl_m1_range, vcl_v1_native, vcl_v2_range, vcl_m2_range);
   if (retval == EXIT_FAILURE)
   {
     std::cout << " --- FAILED! ---" << std::endl;
     return retval;
   }
   else
     std::cout << " --- PASSED ---" << std::endl;


   std::cout << "* m = range, v1 = full, v2 = slice" << std::endl;
   retval = test_prod_rank1<NumericT>(epsilon,
                                      ublas_m1, ublas_v1, ublas_v2, ublas_m2,
                                      vcl_m1_range, vcl_v1_native, vcl_v2_slice, vcl_m2_range);
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
                                      ublas_m1, ublas_v1, ublas_v2, ublas_m2,
                                      vcl_m1_range, vcl_v1_range, vcl_v2_native, vcl_m2_range);
   if (retval == EXIT_FAILURE)
   {
     std::cout << " --- FAILED! ---" << std::endl;
     return retval;
   }
   else
     std::cout << " --- PASSED ---" << std::endl;


   std::cout << "* m = range, v1 = range, v2 = range" << std::endl;
   retval = test_prod_rank1<NumericT>(epsilon,
                                      ublas_m1, ublas_v1, ublas_v2, ublas_m2,
                                      vcl_m1_range, vcl_v1_range, vcl_v2_range, vcl_m2_range);
   if (retval == EXIT_FAILURE)
   {
     std::cout << " --- FAILED! ---" << std::endl;
     return retval;
   }
   else
     std::cout << " --- PASSED ---" << std::endl;


   std::cout << "* m = range, v1 = range, v2 = slice" << std::endl;
   retval = test_prod_rank1<NumericT>(epsilon,
                                      ublas_m1, ublas_v1, ublas_v2, ublas_m2,
                                      vcl_m1_range, vcl_v1_range, vcl_v2_slice, vcl_m2_range);
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
                                      ublas_m1, ublas_v1, ublas_v2, ublas_m2,
                                      vcl_m1_range, vcl_v1_slice, vcl_v2_native, vcl_m2_range);
   if (retval == EXIT_FAILURE)
   {
     std::cout << " --- FAILED! ---" << std::endl;
     return retval;
   }
   else
     std::cout << " --- PASSED ---" << std::endl;


   std::cout << "* m = range, v1 = slice, v2 = range" << std::endl;
   retval = test_prod_rank1<NumericT>(epsilon,
                                      ublas_m1, ublas_v1, ublas_v2, ublas_m2,
                                      vcl_m1_range, vcl_v1_slice, vcl_v2_range, vcl_m2_range);
   if (retval == EXIT_FAILURE)
   {
     std::cout << " --- FAILED! ---" << std::endl;
     return retval;
   }
   else
     std::cout << " --- PASSED ---" << std::endl;


   std::cout << "* m = range, v1 = slice, v2 = slice" << std::endl;
   retval = test_prod_rank1<NumericT>(epsilon,
                                      ublas_m1, ublas_v1, ublas_v2, ublas_m2,
                                      vcl_m1_range, vcl_v1_slice, vcl_v2_slice, vcl_m2_range);
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
                                      ublas_m1, ublas_v1, ublas_v2, ublas_m2,
                                      vcl_m1_slice, vcl_v1_native, vcl_v2_native, vcl_m2_slice);
   if (retval == EXIT_FAILURE)
   {
     std::cout << " --- FAILED! ---" << std::endl;
     return retval;
   }
   else
     std::cout << " --- PASSED ---" << std::endl;


   std::cout << "* m = slice, v1 = full, v2 = range" << std::endl;
   retval = test_prod_rank1<NumericT>(epsilon,
                                      ublas_m1, ublas_v1, ublas_v2, ublas_m2,
                                      vcl_m1_slice, vcl_v1_native, vcl_v2_range, vcl_m2_slice);
   if (retval == EXIT_FAILURE)
   {
     std::cout << " --- FAILED! ---" << std::endl;
     return retval;
   }
   else
     std::cout << " --- PASSED ---" << std::endl;


   std::cout << "* m = slice, v1 = full, v2 = slice" << std::endl;
   retval = test_prod_rank1<NumericT>(epsilon,
                                      ublas_m1, ublas_v1, ublas_v2, ublas_m2,
                                      vcl_m1_slice, vcl_v1_native, vcl_v2_slice, vcl_m2_slice);
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
                                      ublas_m1, ublas_v1, ublas_v2, ublas_m2,
                                      vcl_m1_slice, vcl_v1_range, vcl_v2_native, vcl_m2_slice);
   if (retval == EXIT_FAILURE)
   {
     std::cout << " --- FAILED! ---" << std::endl;
     return retval;
   }
   else
     std::cout << " --- PASSED ---" << std::endl;


   std::cout << "* m = slice, v1 = range, v2 = range" << std::endl;
   retval = test_prod_rank1<NumericT>(epsilon,
                                      ublas_m1, ublas_v1, ublas_v2, ublas_m2,
                                      vcl_m1_slice, vcl_v1_range, vcl_v2_range, vcl_m2_slice);
   if (retval == EXIT_FAILURE)
   {
     std::cout << " --- FAILED! ---" << std::endl;
     return retval;
   }
   else
     std::cout << " --- PASSED ---" << std::endl;


   std::cout << "* m = slice, v1 = range, v2 = slice" << std::endl;
   retval = test_prod_rank1<NumericT>(epsilon,
                                      ublas_m1, ublas_v1, ublas_v2, ublas_m2,
                                      vcl_m1_slice, vcl_v1_range, vcl_v2_slice, vcl_m2_slice);
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
                                      ublas_m1, ublas_v1, ublas_v2, ublas_m2,
                                      vcl_m1_slice, vcl_v1_slice, vcl_v2_native, vcl_m2_slice);
   if (retval == EXIT_FAILURE)
   {
     std::cout << " --- FAILED! ---" << std::endl;
     return retval;
   }
   else
     std::cout << " --- PASSED ---" << std::endl;


   std::cout << "* m = slice, v1 = slice, v2 = range" << std::endl;
   retval = test_prod_rank1<NumericT>(epsilon,
                                      ublas_m1, ublas_v1, ublas_v2, ublas_m2,
                                      vcl_m1_slice, vcl_v1_slice, vcl_v2_range, vcl_m2_slice);
   if (retval == EXIT_FAILURE)
   {
     std::cout << " --- FAILED! ---" << std::endl;
     return retval;
   }
   else
     std::cout << " --- PASSED ---" << std::endl;


   std::cout << "* m = slice, v1 = slice, v2 = slice" << std::endl;
   retval = test_prod_rank1<NumericT>(epsilon,
                                      ublas_m1, ublas_v1, ublas_v2, ublas_m2,
                                      vcl_m1_slice, vcl_v1_slice, vcl_v2_slice, vcl_m2_slice);
   if (retval == EXIT_FAILURE)
   {
     std::cout << " --- FAILED! ---" << std::endl;
     return retval;
   }
   else
     std::cout << " --- PASSED ---" << std::endl;



   //
   // Testing triangular solve() routines
   //

   std::cout << "------------ Testing triangular solves ------------------" << std::endl;

   std::cout << "* m = full, v1 = full" << std::endl;
   retval = test_solve<NumericT>(epsilon,
                                 ublas_m2, ublas_v1,
                                 vcl_m2_native, vcl_v1_native);
   if (retval == EXIT_FAILURE)
   {
     std::cout << " --- FAILED! ---" << std::endl;
     return retval;
   }
   else
     std::cout << " --- PASSED ---" << std::endl;

   std::cout << "* m = full, v1 = range" << std::endl;
   retval = test_solve<NumericT>(epsilon,
                                 ublas_m2, ublas_v1,
                                 vcl_m2_native, vcl_v1_range);
   if (retval == EXIT_FAILURE)
   {
     std::cout << " --- FAILED! ---" << std::endl;
     return retval;
   }
   else
     std::cout << " --- PASSED ---" << std::endl;

   std::cout << "* m = full, v1 = slice" << std::endl;
   retval = test_solve<NumericT>(epsilon,
                                 ublas_m2, ublas_v1,
                                 vcl_m2_native, vcl_v1_slice);
   if (retval == EXIT_FAILURE)
   {
     std::cout << " --- FAILED! ---" << std::endl;
     return retval;
   }
   else
     std::cout << " --- PASSED ---" << std::endl;

   ///////// matrix_range


   std::cout << "* m = range, v1 = full" << std::endl;
   retval = test_solve<NumericT>(epsilon,
                                 ublas_m2, ublas_v1,
                                 vcl_m2_range, vcl_v1_native);
   if (retval == EXIT_FAILURE)
   {
     std::cout << " --- FAILED! ---" << std::endl;
     return retval;
   }
   else
     std::cout << " --- PASSED ---" << std::endl;

   std::cout << "* m = range, v1 = range" << std::endl;
   retval = test_solve<NumericT>(epsilon,
                                 ublas_m2, ublas_v1,
                                 vcl_m2_range, vcl_v1_range);
   if (retval == EXIT_FAILURE)
   {
     std::cout << " --- FAILED! ---" << std::endl;
     return retval;
   }
   else
     std::cout << " --- PASSED ---" << std::endl;

   std::cout << "* m = range, v1 = slice" << std::endl;
   retval = test_solve<NumericT>(epsilon,
                                 ublas_m2, ublas_v1,
                                 vcl_m2_range, vcl_v1_slice);
   if (retval == EXIT_FAILURE)
   {
     std::cout << " --- FAILED! ---" << std::endl;
     return retval;
   }
   else
     std::cout << " --- PASSED ---" << std::endl;

   //////// matrix_slice

   std::cout << "* m = slice, v1 = full" << std::endl;
   retval = test_solve<NumericT>(epsilon,
                                 ublas_m2, ublas_v1,
                                 vcl_m2_slice, vcl_v1_native);
   if (retval == EXIT_FAILURE)
   {
     std::cout << " --- FAILED! ---" << std::endl;
     return retval;
   }
   else
     std::cout << " --- PASSED ---" << std::endl;

   std::cout << "* m = slice, v1 = range" << std::endl;
   retval = test_solve<NumericT>(epsilon,
                                 ublas_m2, ublas_v1,
                                 vcl_m2_slice, vcl_v1_range);
   if (retval == EXIT_FAILURE)
   {
     std::cout << " --- FAILED! ---" << std::endl;
     return retval;
   }
   else
     std::cout << " --- PASSED ---" << std::endl;

   std::cout << "* m = slice, v1 = slice" << std::endl;
   retval = test_solve<NumericT>(epsilon,
                                 ublas_m2, ublas_v1,
                                 vcl_m2_slice, vcl_v1_slice);
   if (retval == EXIT_FAILURE)
   {
     std::cout << " --- FAILED! ---" << std::endl;
     return retval;
   }
   else
     std::cout << " --- PASSED ---" << std::endl;







   ////////////// Final test for full LU decomposition:

   //full solver:
   std::cout << "Full solver" << std::endl;
   unsigned int lu_dim = 100;
   ublas::matrix<NumericT> square_matrix(lu_dim, lu_dim);
   ublas::vector<NumericT> lu_rhs(lu_dim);
   viennacl::matrix<NumericT, F> vcl_square_matrix(lu_dim, lu_dim);
   viennacl::vector<NumericT> vcl_lu_rhs(lu_dim);

   for (std::size_t i=0; i<lu_dim; ++i)
     for (std::size_t j=0; j<lu_dim; ++j)
       square_matrix(i,j) = -static_cast<NumericT>(0.5) * randomNumber();

   //put some more weight on diagonal elements:
   for (std::size_t j=0; j<lu_dim; ++j)
   {
     square_matrix(j,j) = static_cast<NumericT>(20.0) + randomNumber();
     lu_rhs(j) = randomNumber();
   }

   viennacl::copy(square_matrix, vcl_square_matrix);
   viennacl::copy(lu_rhs, vcl_lu_rhs);

   //ublas::
   ublas::lu_factorize(square_matrix);
   ublas::inplace_solve (square_matrix, lu_rhs, ublas::unit_lower_tag ());
   ublas::inplace_solve (square_matrix, lu_rhs, ublas::upper_tag ());

   // ViennaCL:
   viennacl::linalg::lu_factorize(vcl_square_matrix);
   //viennacl::copy(square_matrix, vcl_square_matrix);
   viennacl::linalg::lu_substitute(vcl_square_matrix, vcl_lu_rhs);

   if ( std::fabs(diff(lu_rhs, vcl_lu_rhs)) > epsilon )
   {
      std::cout << "# Error at operation: dense solver" << std::endl;
      std::cout << "  diff: " << std::fabs(diff(lu_rhs, vcl_lu_rhs)) << std::endl;
      retval = EXIT_FAILURE;
   }



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

//   std::cout << std::endl;
//   std::cout << "----------------------------------------------" << std::endl;
//   std::cout << std::endl;
//   {
//      typedef float NumericT;
//      NumericT epsilon = NumericT(1.0E-3);
//      std::cout << "# Testing setup:" << std::endl;
//      std::cout << "  eps:     " << epsilon << std::endl;
//      std::cout << "  numeric: float" << std::endl;
//      std::cout << "  layout: row-major" << std::endl;
//      retval = test<NumericT, viennacl::row_major>(epsilon);
//      if ( retval == EXIT_SUCCESS )
//         std::cout << "# Test passed" << std::endl;
//      else
//         return retval;
//   }
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
