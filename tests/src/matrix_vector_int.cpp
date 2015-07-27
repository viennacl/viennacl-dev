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

/** \file tests/src/matrix_vector_int.cpp  Tests routines for matrix-vector operaions (BLAS level 2) using integer arithmetic.
*   \test Tests routines for matrix-vector operaions (BLAS level 2) using integer arithmetic.
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
#include "viennacl/linalg/sum.hpp"

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
      return 1;
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
      if (v2_cpu[i] != v1[i])
        return 1;
   }

   return 0;
}

template<typename ScalarType, typename VCLMatrixType>
ScalarType diff(ublas::matrix<ScalarType> const & mat1, VCLMatrixType const & mat2)
{
   ublas::matrix<ScalarType> mat2_cpu(mat2.size1(), mat2.size2());
   viennacl::backend::finish();  //workaround for a bug in APP SDK 2.7 on Trinity APUs (with Catalyst 12.8)
   viennacl::copy(mat2, mat2_cpu);

    for (unsigned int i = 0; i < mat2_cpu.size1(); ++i)
    {
      for (unsigned int j = 0; j < mat2_cpu.size2(); ++j)
      {
         if (mat2_cpu(i,j) != mat1(i,j))
           return 1;
      }
    }
   //std::cout << ret << std::endl;
   return 0;
}
//
// -------------------------------------------------------------
//

template<typename NumericT,
          typename UblasMatrixType, typename UblasVectorType,
          typename VCLMatrixType, typename VCLVectorType1, typename VCLVectorType2>
int test_prod_rank1(UblasMatrixType & ublas_m1, UblasVectorType & ublas_v1, UblasVectorType & ublas_v2,
                    VCLMatrixType & vcl_m1, VCLVectorType1 & vcl_v1, VCLVectorType2 & vcl_v2)
{
   int retval = EXIT_SUCCESS;

   // sync data:
   ublas_v1 = ublas::scalar_vector<NumericT>(ublas_v1.size(), NumericT(2));
   ublas_v2 = ublas::scalar_vector<NumericT>(ublas_v2.size(), NumericT(3));
   viennacl::copy(ublas_v1.begin(), ublas_v1.end(), vcl_v1.begin());
   viennacl::copy(ublas_v2.begin(), ublas_v2.end(), vcl_v2.begin());
   viennacl::copy(ublas_m1, vcl_m1);

   // --------------------------------------------------------------------------
   std::cout << "Rank 1 update" << std::endl;

   ublas_m1 += ublas::outer_prod(ublas_v1, ublas_v2);
   vcl_m1 += viennacl::linalg::outer_prod(vcl_v1, vcl_v2);
   if ( diff(ublas_m1, vcl_m1) != 0 )
   {
      std::cout << "# Error at operation: rank 1 update" << std::endl;
      std::cout << "  diff: " << diff(ublas_m1, vcl_m1) << std::endl;
      return EXIT_FAILURE;
   }



   // --------------------------------------------------------------------------
   std::cout << "Scaled rank 1 update - CPU Scalar" << std::endl;
   ublas_m1 += NumericT(4) * ublas::outer_prod(ublas_v1, ublas_v2);
   vcl_m1 += NumericT(2) * viennacl::linalg::outer_prod(vcl_v1, vcl_v2);
   vcl_m1 += viennacl::linalg::outer_prod(vcl_v1, vcl_v2) * NumericT(2);  //check proper compilation
   if ( diff(ublas_m1, vcl_m1) != 0 )
   {
      std::cout << "# Error at operation: scaled rank 1 update - CPU Scalar" << std::endl;
      std::cout << "  diff: " << diff(ublas_m1, vcl_m1) << std::endl;
      return EXIT_FAILURE;
   }

      // --------------------------------------------------------------------------
   std::cout << "Scaled rank 1 update - GPU Scalar" << std::endl;
   ublas_m1 += NumericT(4) * ublas::outer_prod(ublas_v1, ublas_v2);
   vcl_m1 += viennacl::scalar<NumericT>(2) * viennacl::linalg::outer_prod(vcl_v1, vcl_v2);
   vcl_m1 += viennacl::linalg::outer_prod(vcl_v1, vcl_v2) * viennacl::scalar<NumericT>(2);  //check proper compilation
   if ( diff(ublas_m1, vcl_m1) != 0 )
   {
      std::cout << "# Error at operation: scaled rank 1 update - GPU Scalar" << std::endl;
      std::cout << "  diff: " << diff(ublas_m1, vcl_m1) << std::endl;
      return EXIT_FAILURE;
   }

   //reset vcl_matrix:
   viennacl::copy(ublas_m1, vcl_m1);

   // --------------------------------------------------------------------------
   std::cout << "Matrix-Vector product" << std::endl;
   ublas_v1 = viennacl::linalg::prod(ublas_m1, ublas_v2);
   vcl_v1   = viennacl::linalg::prod(vcl_m1, vcl_v2);

   if ( diff(ublas_v1, vcl_v1) != 0 )
   {
      std::cout << "# Error at operation: matrix-vector product" << std::endl;
      std::cout << "  diff: " << diff(ublas_v1, vcl_v1) << std::endl;
      retval = EXIT_FAILURE;
   }
   // --------------------------------------------------------------------------
   std::cout << "Matrix-Vector product with scaled add" << std::endl;
   NumericT alpha = static_cast<NumericT>(2);
   NumericT beta = static_cast<NumericT>(3);
   viennacl::copy(ublas_v1.begin(), ublas_v1.end(), vcl_v1.begin());
   viennacl::copy(ublas_v2.begin(), ublas_v2.end(), vcl_v2.begin());

   ublas_v1 = alpha * viennacl::linalg::prod(ublas_m1, ublas_v2) + beta * ublas_v1;
   vcl_v1   = alpha * viennacl::linalg::prod(vcl_m1, vcl_v2) + beta * vcl_v1;

   if ( diff(ublas_v1, vcl_v1) != 0 )
   {
      std::cout << "# Error at operation: matrix-vector product with scaled additions" << std::endl;
      std::cout << "  diff: " << diff(ublas_v1, vcl_v1) << std::endl;
      retval = EXIT_FAILURE;
   }
   // --------------------------------------------------------------------------

   viennacl::copy(ublas_v1.begin(), ublas_v1.end(), vcl_v1.begin());
   viennacl::copy(ublas_v2.begin(), ublas_v2.end(), vcl_v2.begin());

   std::cout << "Transposed Matrix-Vector product" << std::endl;
   ublas_v2 = alpha * viennacl::linalg::prod(trans(ublas_m1), ublas_v1);
   vcl_v2   = alpha * viennacl::linalg::prod(trans(vcl_m1), vcl_v1);

   if ( diff(ublas_v2, vcl_v2) != 0 )
   {
      std::cout << "# Error at operation: transposed matrix-vector product" << std::endl;
      std::cout << "  diff: " << diff(ublas_v2, vcl_v2) << std::endl;
      retval = EXIT_FAILURE;
   }

   std::cout << "Transposed Matrix-Vector product with scaled add" << std::endl;
   ublas_v2 = alpha * viennacl::linalg::prod(trans(ublas_m1), ublas_v1) + beta * ublas_v2;
   vcl_v2   = alpha * viennacl::linalg::prod(trans(vcl_m1), vcl_v1) + beta * vcl_v2;

   if ( diff(ublas_v2, vcl_v2) != 0 )
   {
      std::cout << "# Error at operation: transposed matrix-vector product with scaled additions" << std::endl;
      std::cout << "  diff: " << diff(ublas_v2, vcl_v2) << std::endl;
      retval = EXIT_FAILURE;
   }
   // --------------------------------------------------------------------------


   std::cout << "Row sum with matrix" << std::endl;
   ublas_v1 = ublas::prod(ublas_m1, ublas::scalar_vector<NumericT>(ublas_m1.size2(), NumericT(1)));
   vcl_v1   = viennacl::linalg::row_sum(vcl_m1);

   if ( diff(ublas_v1, vcl_v1) != 0 )
   {
      std::cout << "# Error at operation: row sum" << std::endl;
      std::cout << "  diff: " << diff(ublas_v1, vcl_v1) << std::endl;
      retval = EXIT_FAILURE;
   }
   // --------------------------------------------------------------------------

   std::cout << "Row sum with matrix expression" << std::endl;
   ublas_v1 = ublas::prod(ublas_m1 + ublas_m1, ublas::scalar_vector<NumericT>(ublas_m1.size2(), NumericT(1)));
   vcl_v1   = viennacl::linalg::row_sum(vcl_m1 + vcl_m1);

   if ( diff(ublas_v1, vcl_v1) != 0 )
   {
      std::cout << "# Error at operation: row sum (with expression)" << std::endl;
      std::cout << "  diff: " << diff(ublas_v1, vcl_v1) << std::endl;
      retval = EXIT_FAILURE;
   }
   // --------------------------------------------------------------------------

   std::cout << "Column sum with matrix" << std::endl;
   ublas_v2 = ublas::prod(trans(ublas_m1), ublas::scalar_vector<NumericT>(ublas_m1.size1(), NumericT(1)));
   vcl_v2   = viennacl::linalg::column_sum(vcl_m1);

   if ( diff(ublas_v2, vcl_v2) != 0 )
   {
      std::cout << "# Error at operation: column sum" << std::endl;
      std::cout << "  diff: " << diff(ublas_v2, vcl_v2) << std::endl;
      retval = EXIT_FAILURE;
   }
   // --------------------------------------------------------------------------

   std::cout << "Column sum with matrix expression" << std::endl;
   ublas_v2 = ublas::prod(trans(ublas_m1 + ublas_m1), ublas::scalar_vector<NumericT>(ublas_m1.size1(), NumericT(1)));
   vcl_v2   = viennacl::linalg::column_sum(vcl_m1 + vcl_m1);

   if ( diff(ublas_v2, vcl_v2) != 0 )
   {
      std::cout << "# Error at operation: column sum (with expression)" << std::endl;
      std::cout << "  diff: " << diff(ublas_v2, vcl_v2) << std::endl;
      retval = EXIT_FAILURE;
   }
   // --------------------------------------------------------------------------

   return retval;
}


//
// -------------------------------------------------------------
//
template< typename NumericT, typename F>
int test()
{
   int retval = EXIT_SUCCESS;

   std::size_t num_rows = 141;
   std::size_t num_cols = 103;

   // --------------------------------------------------------------------------
   ublas::vector<NumericT> ublas_v1(num_rows);
   for (std::size_t i = 0; i < ublas_v1.size(); ++i)
     ublas_v1(i) = NumericT(i);
   ublas::vector<NumericT> ublas_v2 = ublas::scalar_vector<NumericT>(num_cols, NumericT(3));


   ublas::matrix<NumericT> ublas_m1(ublas_v1.size(), ublas_v2.size());
   ublas::matrix<NumericT> ublas_m2(ublas_v1.size(), ublas_v1.size());


   for (std::size_t i = 0; i < ublas_m1.size1(); ++i)
    for (std::size_t j = 0; j < ublas_m1.size2(); ++j)
      ublas_m1(i,j) = NumericT(i+j);


   for (std::size_t i = 0; i < ublas_m2.size1(); ++i)
    for (std::size_t j = 0; j < ublas_m2.size2(); ++j)
       ublas_m2(i,j) = NumericT(j - i*j + i);


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


   //
   // Run a bunch of tests for rank-1-updates, matrix-vector products
   //
   std::cout << "------------ Testing rank-1-updates and matrix-vector products ------------------" << std::endl;

   std::cout << "* m = full, v1 = full, v2 = full" << std::endl;
   retval = test_prod_rank1<NumericT>(ublas_m1, ublas_v1, ublas_v2,
                                      vcl_m1_native, vcl_v1_native, vcl_v2_native);
   if (retval == EXIT_FAILURE)
   {
     std::cout << " --- FAILED! ---" << std::endl;
     return retval;
   }
   else
     std::cout << " --- PASSED ---" << std::endl;

   for (std::size_t i = 0; i < ublas_m1.size1(); ++i)
    for (std::size_t j = 0; j < ublas_m1.size2(); ++j)
      ublas_m1(i,j) = NumericT(i+j);

   std::cout << "* m = full, v1 = full, v2 = range" << std::endl;
   retval = test_prod_rank1<NumericT>(ublas_m1, ublas_v1, ublas_v2,
                                      vcl_m1_native, vcl_v1_native, vcl_v2_range);
   if (retval == EXIT_FAILURE)
   {
     std::cout << " --- FAILED! ---" << std::endl;
     return retval;
   }
   else
     std::cout << " --- PASSED ---" << std::endl;


   for (std::size_t i = 0; i < ublas_m1.size1(); ++i)
    for (std::size_t j = 0; j < ublas_m1.size2(); ++j)
      ublas_m1(i,j) = NumericT(i+j);

   std::cout << "* m = full, v1 = full, v2 = slice" << std::endl;
   retval = test_prod_rank1<NumericT>(ublas_m1, ublas_v1, ublas_v2,
                                      vcl_m1_native, vcl_v1_native, vcl_v2_slice);
   if (retval == EXIT_FAILURE)
   {
     std::cout << " --- FAILED! ---" << std::endl;
     return retval;
   }
   else
     std::cout << " --- PASSED ---" << std::endl;


   for (std::size_t i = 0; i < ublas_m1.size1(); ++i)
    for (std::size_t j = 0; j < ublas_m1.size2(); ++j)
      ublas_m1(i,j) = NumericT(i+j);

   // v1 = range


   std::cout << "* m = full, v1 = range, v2 = full" << std::endl;
   retval = test_prod_rank1<NumericT>(ublas_m1, ublas_v1, ublas_v2,
                                      vcl_m1_native, vcl_v1_range, vcl_v2_native);
   if (retval == EXIT_FAILURE)
   {
     std::cout << " --- FAILED! ---" << std::endl;
     return retval;
   }
   else
     std::cout << " --- PASSED ---" << std::endl;


   for (std::size_t i = 0; i < ublas_m1.size1(); ++i)
    for (std::size_t j = 0; j < ublas_m1.size2(); ++j)
      ublas_m1(i,j) = NumericT(i+j);

   std::cout << "* m = full, v1 = range, v2 = range" << std::endl;
   retval = test_prod_rank1<NumericT>(ublas_m1, ublas_v1, ublas_v2,
                                      vcl_m1_native, vcl_v1_range, vcl_v2_range);
   if (retval == EXIT_FAILURE)
   {
     std::cout << " --- FAILED! ---" << std::endl;
     return retval;
   }
   else
     std::cout << " --- PASSED ---" << std::endl;


   for (std::size_t i = 0; i < ublas_m1.size1(); ++i)
    for (std::size_t j = 0; j < ublas_m1.size2(); ++j)
      ublas_m1(i,j) = NumericT(i+j);

   std::cout << "* m = full, v1 = range, v2 = slice" << std::endl;
   retval = test_prod_rank1<NumericT>(ublas_m1, ublas_v1, ublas_v2,
                                      vcl_m1_native, vcl_v1_range, vcl_v2_slice);
   if (retval == EXIT_FAILURE)
   {
     std::cout << " --- FAILED! ---" << std::endl;
     return retval;
   }
   else
     std::cout << " --- PASSED ---" << std::endl;


   for (std::size_t i = 0; i < ublas_m1.size1(); ++i)
    for (std::size_t j = 0; j < ublas_m1.size2(); ++j)
      ublas_m1(i,j) = NumericT(i+j);


   // v1 = slice

   std::cout << "* m = full, v1 = slice, v2 = full" << std::endl;
   retval = test_prod_rank1<NumericT>(ublas_m1, ublas_v1, ublas_v2,
                                      vcl_m1_native, vcl_v1_slice, vcl_v2_native);
   if (retval == EXIT_FAILURE)
   {
     std::cout << " --- FAILED! ---" << std::endl;
     return retval;
   }
   else
     std::cout << " --- PASSED ---" << std::endl;


   for (std::size_t i = 0; i < ublas_m1.size1(); ++i)
    for (std::size_t j = 0; j < ublas_m1.size2(); ++j)
      ublas_m1(i,j) = NumericT(i+j);

   std::cout << "* m = full, v1 = slice, v2 = range" << std::endl;
   retval = test_prod_rank1<NumericT>(ublas_m1, ublas_v1, ublas_v2,
                                      vcl_m1_native, vcl_v1_slice, vcl_v2_range);
   if (retval == EXIT_FAILURE)
   {
     std::cout << " --- FAILED! ---" << std::endl;
     return retval;
   }
   else
     std::cout << " --- PASSED ---" << std::endl;


   for (std::size_t i = 0; i < ublas_m1.size1(); ++i)
    for (std::size_t j = 0; j < ublas_m1.size2(); ++j)
      ublas_m1(i,j) = NumericT(i+j);

   std::cout << "* m = full, v1 = slice, v2 = slice" << std::endl;
   retval = test_prod_rank1<NumericT>(ublas_m1, ublas_v1, ublas_v2,
                                      vcl_m1_native, vcl_v1_slice, vcl_v2_slice);
   if (retval == EXIT_FAILURE)
   {
     std::cout << " --- FAILED! ---" << std::endl;
     return retval;
   }
   else
     std::cout << " --- PASSED ---" << std::endl;


   for (std::size_t i = 0; i < ublas_m1.size1(); ++i)
    for (std::size_t j = 0; j < ublas_m1.size2(); ++j)
      ublas_m1(i,j) = NumericT(i+j);

   ///////////////////////////// matrix_range

   std::cout << "* m = range, v1 = full, v2 = full" << std::endl;
   retval = test_prod_rank1<NumericT>(ublas_m1, ublas_v1, ublas_v2,
                                      vcl_m1_range, vcl_v1_native, vcl_v2_native);
   if (retval == EXIT_FAILURE)
   {
     std::cout << " --- FAILED! ---" << std::endl;
     return retval;
   }
   else
     std::cout << " --- PASSED ---" << std::endl;


   for (std::size_t i = 0; i < ublas_m1.size1(); ++i)
    for (std::size_t j = 0; j < ublas_m1.size2(); ++j)
      ublas_m1(i,j) = NumericT(i+j);

   std::cout << "* m = range, v1 = full, v2 = range" << std::endl;
   retval = test_prod_rank1<NumericT>(ublas_m1, ublas_v1, ublas_v2,
                                      vcl_m1_range, vcl_v1_native, vcl_v2_range);
   if (retval == EXIT_FAILURE)
   {
     std::cout << " --- FAILED! ---" << std::endl;
     return retval;
   }
   else
     std::cout << " --- PASSED ---" << std::endl;


   for (std::size_t i = 0; i < ublas_m1.size1(); ++i)
    for (std::size_t j = 0; j < ublas_m1.size2(); ++j)
      ublas_m1(i,j) = NumericT(i+j);

   std::cout << "* m = range, v1 = full, v2 = slice" << std::endl;
   retval = test_prod_rank1<NumericT>(ublas_m1, ublas_v1, ublas_v2,
                                      vcl_m1_range, vcl_v1_native, vcl_v2_slice);
   if (retval == EXIT_FAILURE)
   {
     std::cout << " --- FAILED! ---" << std::endl;
     return retval;
   }
   else
     std::cout << " --- PASSED ---" << std::endl;


   for (std::size_t i = 0; i < ublas_m1.size1(); ++i)
    for (std::size_t j = 0; j < ublas_m1.size2(); ++j)
      ublas_m1(i,j) = NumericT(i+j);

   // v1 = range


   std::cout << "* m = range, v1 = range, v2 = full" << std::endl;
   retval = test_prod_rank1<NumericT>(ublas_m1, ublas_v1, ublas_v2,
                                      vcl_m1_range, vcl_v1_range, vcl_v2_native);
   if (retval == EXIT_FAILURE)
   {
     std::cout << " --- FAILED! ---" << std::endl;
     return retval;
   }
   else
     std::cout << " --- PASSED ---" << std::endl;


   for (std::size_t i = 0; i < ublas_m1.size1(); ++i)
    for (std::size_t j = 0; j < ublas_m1.size2(); ++j)
      ublas_m1(i,j) = NumericT(i+j);

   std::cout << "* m = range, v1 = range, v2 = range" << std::endl;
   retval = test_prod_rank1<NumericT>(ublas_m1, ublas_v1, ublas_v2,
                                      vcl_m1_range, vcl_v1_range, vcl_v2_range);
   if (retval == EXIT_FAILURE)
   {
     std::cout << " --- FAILED! ---" << std::endl;
     return retval;
   }
   else
     std::cout << " --- PASSED ---" << std::endl;


   for (std::size_t i = 0; i < ublas_m1.size1(); ++i)
    for (std::size_t j = 0; j < ublas_m1.size2(); ++j)
      ublas_m1(i,j) = NumericT(i+j);

   std::cout << "* m = range, v1 = range, v2 = slice" << std::endl;
   retval = test_prod_rank1<NumericT>(ublas_m1, ublas_v1, ublas_v2,
                                      vcl_m1_range, vcl_v1_range, vcl_v2_slice);
   if (retval == EXIT_FAILURE)
   {
     std::cout << " --- FAILED! ---" << std::endl;
     return retval;
   }
   else
     std::cout << " --- PASSED ---" << std::endl;


   for (std::size_t i = 0; i < ublas_m1.size1(); ++i)
    for (std::size_t j = 0; j < ublas_m1.size2(); ++j)
      ublas_m1(i,j) = NumericT(i+j);


   // v1 = slice

   std::cout << "* m = range, v1 = slice, v2 = full" << std::endl;
   retval = test_prod_rank1<NumericT>(ublas_m1, ublas_v1, ublas_v2,
                                      vcl_m1_range, vcl_v1_slice, vcl_v2_native);
   if (retval == EXIT_FAILURE)
   {
     std::cout << " --- FAILED! ---" << std::endl;
     return retval;
   }
   else
     std::cout << " --- PASSED ---" << std::endl;


   for (std::size_t i = 0; i < ublas_m1.size1(); ++i)
    for (std::size_t j = 0; j < ublas_m1.size2(); ++j)
      ublas_m1(i,j) = NumericT(i+j);

   std::cout << "* m = range, v1 = slice, v2 = range" << std::endl;
   retval = test_prod_rank1<NumericT>(ublas_m1, ublas_v1, ublas_v2,
                                      vcl_m1_range, vcl_v1_slice, vcl_v2_range);
   if (retval == EXIT_FAILURE)
   {
     std::cout << " --- FAILED! ---" << std::endl;
     return retval;
   }
   else
     std::cout << " --- PASSED ---" << std::endl;


   for (std::size_t i = 0; i < ublas_m1.size1(); ++i)
    for (std::size_t j = 0; j < ublas_m1.size2(); ++j)
      ublas_m1(i,j) = NumericT(i+j);

   std::cout << "* m = range, v1 = slice, v2 = slice" << std::endl;
   retval = test_prod_rank1<NumericT>(ublas_m1, ublas_v1, ublas_v2,
                                      vcl_m1_range, vcl_v1_slice, vcl_v2_slice);
   if (retval == EXIT_FAILURE)
   {
     std::cout << " --- FAILED! ---" << std::endl;
     return retval;
   }
   else
     std::cout << " --- PASSED ---" << std::endl;


   for (std::size_t i = 0; i < ublas_m1.size1(); ++i)
    for (std::size_t j = 0; j < ublas_m1.size2(); ++j)
      ublas_m1(i,j) = NumericT(i+j);

   ///////////////////////////// matrix_slice

   std::cout << "* m = slice, v1 = full, v2 = full" << std::endl;
   retval = test_prod_rank1<NumericT>(ublas_m1, ublas_v1, ublas_v2,
                                      vcl_m1_slice, vcl_v1_native, vcl_v2_native);
   if (retval == EXIT_FAILURE)
   {
     std::cout << " --- FAILED! ---" << std::endl;
     return retval;
   }
   else
     std::cout << " --- PASSED ---" << std::endl;


   for (std::size_t i = 0; i < ublas_m1.size1(); ++i)
    for (std::size_t j = 0; j < ublas_m1.size2(); ++j)
      ublas_m1(i,j) = NumericT(i+j);

   std::cout << "* m = slice, v1 = full, v2 = range" << std::endl;
   retval = test_prod_rank1<NumericT>(ublas_m1, ublas_v1, ublas_v2,
                                      vcl_m1_slice, vcl_v1_native, vcl_v2_range);
   if (retval == EXIT_FAILURE)
   {
     std::cout << " --- FAILED! ---" << std::endl;
     return retval;
   }
   else
     std::cout << " --- PASSED ---" << std::endl;


   for (std::size_t i = 0; i < ublas_m1.size1(); ++i)
    for (std::size_t j = 0; j < ublas_m1.size2(); ++j)
      ublas_m1(i,j) = NumericT(i+j);

   std::cout << "* m = slice, v1 = full, v2 = slice" << std::endl;
   retval = test_prod_rank1<NumericT>(ublas_m1, ublas_v1, ublas_v2,
                                      vcl_m1_slice, vcl_v1_native, vcl_v2_slice);
   if (retval == EXIT_FAILURE)
   {
     std::cout << " --- FAILED! ---" << std::endl;
     return retval;
   }
   else
     std::cout << " --- PASSED ---" << std::endl;


   for (std::size_t i = 0; i < ublas_m1.size1(); ++i)
    for (std::size_t j = 0; j < ublas_m1.size2(); ++j)
      ublas_m1(i,j) = NumericT(i+j);

   // v1 = range


   std::cout << "* m = slice, v1 = range, v2 = full" << std::endl;
   retval = test_prod_rank1<NumericT>(ublas_m1, ublas_v1, ublas_v2,
                                      vcl_m1_slice, vcl_v1_range, vcl_v2_native);
   if (retval == EXIT_FAILURE)
   {
     std::cout << " --- FAILED! ---" << std::endl;
     return retval;
   }
   else
     std::cout << " --- PASSED ---" << std::endl;


   for (std::size_t i = 0; i < ublas_m1.size1(); ++i)
    for (std::size_t j = 0; j < ublas_m1.size2(); ++j)
      ublas_m1(i,j) = NumericT(i+j);

   std::cout << "* m = slice, v1 = range, v2 = range" << std::endl;
   retval = test_prod_rank1<NumericT>(ublas_m1, ublas_v1, ublas_v2,
                                      vcl_m1_slice, vcl_v1_range, vcl_v2_range);
   if (retval == EXIT_FAILURE)
   {
     std::cout << " --- FAILED! ---" << std::endl;
     return retval;
   }
   else
     std::cout << " --- PASSED ---" << std::endl;


   for (std::size_t i = 0; i < ublas_m1.size1(); ++i)
    for (std::size_t j = 0; j < ublas_m1.size2(); ++j)
      ublas_m1(i,j) = NumericT(i+j);

   std::cout << "* m = slice, v1 = range, v2 = slice" << std::endl;
   retval = test_prod_rank1<NumericT>(ublas_m1, ublas_v1, ublas_v2,
                                      vcl_m1_slice, vcl_v1_range, vcl_v2_slice);
   if (retval == EXIT_FAILURE)
   {
     std::cout << " --- FAILED! ---" << std::endl;
     return retval;
   }
   else
     std::cout << " --- PASSED ---" << std::endl;



   for (std::size_t i = 0; i < ublas_m1.size1(); ++i)
    for (std::size_t j = 0; j < ublas_m1.size2(); ++j)
      ublas_m1(i,j) = NumericT(i+j);

   // v1 = slice

   std::cout << "* m = slice, v1 = slice, v2 = full" << std::endl;
   retval = test_prod_rank1<NumericT>(ublas_m1, ublas_v1, ublas_v2,
                                      vcl_m1_slice, vcl_v1_slice, vcl_v2_native);
   if (retval == EXIT_FAILURE)
   {
     std::cout << " --- FAILED! ---" << std::endl;
     return retval;
   }
   else
     std::cout << " --- PASSED ---" << std::endl;

   for (std::size_t i = 0; i < ublas_m1.size1(); ++i)
    for (std::size_t j = 0; j < ublas_m1.size2(); ++j)
      ublas_m1(i,j) = NumericT(i+j);


   std::cout << "* m = slice, v1 = slice, v2 = range" << std::endl;
   retval = test_prod_rank1<NumericT>(ublas_m1, ublas_v1, ublas_v2,
                                      vcl_m1_slice, vcl_v1_slice, vcl_v2_range);
   if (retval == EXIT_FAILURE)
   {
     std::cout << " --- FAILED! ---" << std::endl;
     return retval;
   }
   else
     std::cout << " --- PASSED ---" << std::endl;


   for (std::size_t i = 0; i < ublas_m1.size1(); ++i)
    for (std::size_t j = 0; j < ublas_m1.size2(); ++j)
      ublas_m1(i,j) = NumericT(i+j);

   std::cout << "* m = slice, v1 = slice, v2 = slice" << std::endl;
   retval = test_prod_rank1<NumericT>(ublas_m1, ublas_v1, ublas_v2,
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
      typedef int NumericT;
      std::cout << "# Testing setup:" << std::endl;
      std::cout << "  numeric: int" << std::endl;
      std::cout << "  layout: row-major" << std::endl;
      retval = test<NumericT, viennacl::row_major>();
      if ( retval == EXIT_SUCCESS )
         std::cout << "# Test passed" << std::endl;
      else
         return retval;
   }
   std::cout << std::endl;
   std::cout << "----------------------------------------------" << std::endl;
   std::cout << std::endl;
   {
      typedef int NumericT;
      std::cout << "# Testing setup:" << std::endl;
      std::cout << "  numeric: int" << std::endl;
      std::cout << "  layout: column-major" << std::endl;
      retval = test<NumericT, viennacl::column_major>();
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
         typedef long NumericT;
         std::cout << "# Testing setup:" << std::endl;
         std::cout << "  numeric: double" << std::endl;
         std::cout << "  layout: row-major" << std::endl;
         retval = test<NumericT, viennacl::row_major>();
            if ( retval == EXIT_SUCCESS )
               std::cout << "# Test passed" << std::endl;
            else
              return retval;
      }
      std::cout << std::endl;
      std::cout << "----------------------------------------------" << std::endl;
      std::cout << std::endl;
      {
         typedef long NumericT;
         std::cout << "# Testing setup:" << std::endl;
         std::cout << "  numeric: double" << std::endl;
         std::cout << "  layout: column-major" << std::endl;
         retval = test<NumericT, viennacl::column_major>();
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
