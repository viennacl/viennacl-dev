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
#define VIENNACL_HAVE_UBLAS 1
#include "viennacl/scalar.hpp"
#include "viennacl/matrix.hpp"
#include "viennacl/vector.hpp"
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

template <typename ScalarType, typename F, unsigned int ALIGNMENT>
ScalarType diff(ublas::matrix<ScalarType> & mat1, viennacl::matrix<ScalarType, F, ALIGNMENT> & mat2)
{
   ublas::matrix<ScalarType> mat2_cpu(mat2.size1(), mat2.size2());
   copy(mat2, mat2_cpu);
   ScalarType ret = 0;
   ScalarType act = 0;

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
template< typename NumericT, typename F, typename Epsilon >
int test(Epsilon const& epsilon)
{
   int retval = EXIT_SUCCESS;
   
   std::size_t num_rows = 121;
   std::size_t num_cols = 103;
   
   // --------------------------------------------------------------------------            
   ublas::vector<NumericT> rhs(num_rows);
   for (unsigned int i = 0; i < rhs.size(); ++i)
     rhs(i) = random<NumericT>();
   ublas::vector<NumericT> rhs2 = rhs;
   ublas::vector<NumericT> result = ublas::scalar_vector<NumericT>(num_cols, NumericT(3.1415));
   ublas::vector<NumericT> result2 = result;
   ublas::vector<NumericT> rhs_trans = result;
   ublas::vector<NumericT> result_trans = ublas::zero_vector<NumericT>(rhs.size());

  
   ublas::matrix<NumericT> matrix(result.size(), rhs.size());
  
   for (unsigned int i = 0; i < matrix.size1(); ++i)
      for (unsigned int j = 0; j < matrix.size2(); ++j)
         matrix(i,j) = static_cast<NumericT>(0.1) * random<NumericT>();

   viennacl::vector<NumericT> vcl_rhs(rhs.size());
   viennacl::vector<NumericT> vcl_rhs_trans(rhs_trans.size());
   viennacl::vector<NumericT> vcl_result_trans(result_trans.size());
   viennacl::vector<NumericT> vcl_result(result.size()); 
   viennacl::matrix<NumericT, F> vcl_matrix(result.size(), rhs.size());

   std::cout << "Creating mem" << std::endl;
   viennacl::copy(rhs.begin(), rhs.end(), vcl_rhs.begin());
   viennacl::copy(result, vcl_result);
   viennacl::copy(matrix, vcl_matrix);
   
   std::cout << "Matrix resizing (to larger)" << std::endl;
   matrix.resize(2*num_rows, 2*num_cols, true);
   for (unsigned int i = 0; i < matrix.size1(); ++i)
   {
      for (unsigned int j = (i<result.size() ? rhs.size() : 0); j < matrix.size2(); ++j)
         matrix(i,j) = 0;
   }
   vcl_matrix.resize(2*num_rows, 2*num_cols, true);
   viennacl::copy(vcl_matrix, matrix);
   if( fabs(diff(matrix, vcl_matrix)) > epsilon )
   {
      std::cout << "# Error at operation: matrix resize (to larger)" << std::endl;
      std::cout << "  diff: " << fabs(diff(matrix, vcl_matrix)) << std::endl;
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
   if( fabs(diff(matrix, vcl_matrix)) > epsilon )
   {
      std::cout << "# Error at operation: matrix resize (to smaller)" << std::endl;
      std::cout << "  diff: " << fabs(diff(matrix, vcl_matrix)) << std::endl;
      return EXIT_FAILURE;
   }


   std::cout << "Matrix addition and subtraction" << std::endl;
   viennacl::matrix<NumericT, F> vcl_matrix2 = vcl_matrix;
   vcl_matrix2 += vcl_matrix;
   vcl_matrix2 = vcl_matrix2 + vcl_matrix;
   matrix *= 3.0;

   if( fabs(diff(matrix, vcl_matrix2)) > epsilon )
   {
      std::cout << "# Error at operation: matrix addition and subtraction (part 1b)" << std::endl;
      std::cout << "  diff: " << fabs(diff(matrix, vcl_matrix2)) << std::endl;
      return EXIT_FAILURE;
   }

   vcl_matrix2 -= vcl_matrix;
   vcl_matrix2 = vcl_matrix2 - vcl_matrix;
   matrix /= 3.0;

   if( fabs(diff(matrix, vcl_matrix2)) > epsilon )
   {
      std::cout << "# Error at operation: matrix addition and subtraction (part 2)" << std::endl;
      std::cout << "  diff: " << fabs(diff(matrix, vcl_matrix2)) << std::endl;
      return EXIT_FAILURE;
   }
   
   // --------------------------------------------------------------------------            
   std::cout << "Rank 1 update" << std::endl;
   ublas::matrix<NumericT> matrix2 = matrix;
   
   matrix2 += ublas::outer_prod(result, rhs);
   vcl_matrix += viennacl::linalg::outer_prod(vcl_result, vcl_rhs);
   if( fabs(diff(matrix2, vcl_matrix)) > epsilon )
   {
      std::cout << "# Error at operation: rank 1 update" << std::endl;
      std::cout << "  diff: " << fabs(diff(matrix2, vcl_matrix)) << std::endl;
      return EXIT_FAILURE;
   }
   // --------------------------------------------------------------------------            
   std::cout << "Scaled rank 1 update" << std::endl;
   matrix2 += 4.2f * ublas::outer_prod(result, rhs);
   vcl_matrix += 2.1f * viennacl::linalg::outer_prod(vcl_result, vcl_rhs);
   vcl_matrix += viennacl::linalg::outer_prod(vcl_result, vcl_rhs) * 2.1f;  //check proper compilation
   if( fabs(diff(matrix2, vcl_matrix)) > epsilon )
   {
      std::cout << "# Error at operation: scaled rank 1 update" << std::endl;
      std::cout << "  diff: " << fabs(diff(matrix2, vcl_matrix)) << std::endl;
      return EXIT_FAILURE;
   }
   
   //reset vcl_matrix:
   viennacl::copy(matrix, vcl_matrix);
   
   // --------------------------------------------------------------------------            
   std::cout << "Matrix-Vector product" << std::endl;
   result     = viennacl::linalg::prod(matrix, rhs);
   vcl_result = viennacl::linalg::prod(vcl_matrix, vcl_rhs);
   
   if( fabs(diff(result, vcl_result)) > epsilon )
   {
      std::cout << "# Error at operation: matrix-vector product" << std::endl;
      std::cout << "  diff: " << fabs(diff(result, vcl_result)) << std::endl;
      retval = EXIT_FAILURE;
   }
   // --------------------------------------------------------------------------            
   std::cout << "Matrix-Vector product with scaled add" << std::endl;
   NumericT alpha = static_cast<NumericT>(2.786);
   NumericT beta = static_cast<NumericT>(1.432);
   viennacl::copy(rhs.begin(), rhs.end(), vcl_rhs.begin());
   viennacl::copy(result.begin(), result.end(), vcl_result.begin());

   result     = alpha * viennacl::linalg::prod(matrix, rhs) + beta * result;
   vcl_result = alpha * viennacl::linalg::prod(vcl_matrix, vcl_rhs) + beta * vcl_result;

   if( fabs(diff(result, vcl_result)) > epsilon )
   {
      std::cout << "# Error at operation: matrix-vector product with scaled additions" << std::endl;
      std::cout << "  diff: " << fabs(diff(result, vcl_result)) << std::endl;
      retval = EXIT_FAILURE;
   }
   // --------------------------------------------------------------------------            

   viennacl::copy(rhs_trans.begin(), rhs_trans.end(), vcl_rhs_trans.begin());
   viennacl::copy(result_trans.begin(), result_trans.end(), vcl_result_trans.begin());

   std::cout << "Transposed Matrix-Vector product" << std::endl;
   result_trans     = alpha * viennacl::linalg::prod(trans(matrix), rhs_trans);  
   vcl_result_trans = alpha * viennacl::linalg::prod(trans(vcl_matrix), vcl_rhs_trans);

   if( fabs(diff(result_trans, vcl_result_trans)) > epsilon )
   {
      std::cout << "# Error at operation: transposed matrix-vector product" << std::endl;
      std::cout << "  diff: " << fabs(diff(result_trans, vcl_result_trans)) << std::endl;
      retval = EXIT_FAILURE;
   }

   std::cout << "Transposed Matrix-Vector product with scaled add" << std::endl;
   result_trans     = alpha * viennacl::linalg::prod(trans(matrix), rhs_trans) + beta * result_trans;  
   vcl_result_trans = alpha * viennacl::linalg::prod(trans(vcl_matrix), vcl_rhs_trans) + beta * vcl_result_trans;

   if( fabs(diff(result_trans, vcl_result_trans)) > epsilon )
   {
      std::cout << "# Error at operation: transposed matrix-vector product with scaled additions" << std::endl;
      std::cout << "  diff: " << fabs(diff(result_trans, vcl_result_trans)) << std::endl;
      retval = EXIT_FAILURE;
   }
   // --------------------------------------------------------------------------            

   /////////////////// test direct solvers ////////////////////////////
   
   rhs.resize(40);
   matrix.resize(rhs.size(), rhs.size());
   result.resize(rhs.size());

   std::cout << "Resizing vcl_rhs..." << std::endl;
   vcl_rhs.resize(rhs.size());
   std::cout << "Resizing vcl_rhs done" << std::endl;
   vcl_matrix.resize(rhs.size(), rhs.size());
   std::cout << "Resizing vcl_result..." << std::endl;
   vcl_result.resize(rhs.size());
   std::cout << "Resizing vcl_result done" << std::endl;

   for (unsigned int i = 0; i < matrix.size1(); ++i)
   {
      for (unsigned int j = 0; j < matrix.size2(); ++j)
         matrix(i,j) = -random<NumericT>();
      rhs(i) = random<NumericT>();
   }

   //force unit diagonal
   for (unsigned int i = 0; i < matrix.size1(); ++i)
      matrix(i,i) = static_cast<NumericT>(3) + random<NumericT>();

   viennacl::copy(matrix, vcl_matrix);
   viennacl::copy(rhs, vcl_rhs);

   //upper triangular:
   std::cout << "Upper triangular solver" << std::endl;
   result = ublas::solve(matrix, rhs, ublas::upper_tag());
   vcl_result = viennacl::linalg::solve(vcl_matrix, vcl_rhs, viennacl::linalg::upper_tag());
   if( fabs(diff(result, vcl_result)) > epsilon )
   {
      std::cout << "# Error at operation: upper triangular solver" << std::endl;
      std::cout << "  diff: " << fabs(diff(result, vcl_result)) << std::endl;
      retval = EXIT_FAILURE;
   }

   //upper unit triangular:
   std::cout << "Upper unit triangular solver" << std::endl;
   viennacl::copy(rhs, vcl_rhs);
   result = ublas::solve(matrix, rhs, ublas::unit_upper_tag());
   vcl_result = viennacl::linalg::solve(vcl_matrix, vcl_rhs, viennacl::linalg::unit_upper_tag());
   if( fabs(diff(result, vcl_result)) > epsilon )
   {
      std::cout << "# Error at operation: unit upper triangular solver" << std::endl;
      std::cout << "  diff: " << fabs(diff(result, vcl_result)) << std::endl;
      retval = EXIT_FAILURE;
   }

   //lower triangular:
   std::cout << "Lower triangular solver" << std::endl;
   viennacl::copy(rhs, vcl_rhs);
   result = ublas::solve(matrix, rhs, ublas::lower_tag());
   vcl_result = viennacl::linalg::solve(vcl_matrix, vcl_rhs, viennacl::linalg::lower_tag());
   if( fabs(diff(result, vcl_result)) > epsilon )
   {
      std::cout << "# Error at operation: lower triangular solver" << std::endl;
      std::cout << "  diff: " << fabs(diff(result, vcl_result)) << std::endl;
      retval = EXIT_FAILURE;
   }

   //lower unit triangular:
   std::cout << "Lower unit triangular solver" << std::endl;
   viennacl::copy(rhs, vcl_rhs);
   result = ublas::solve(matrix, rhs, ublas::unit_lower_tag());
   vcl_result = viennacl::linalg::solve(vcl_matrix, vcl_rhs, viennacl::linalg::unit_lower_tag());
   if( fabs(diff(result, vcl_result)) > epsilon )
   {
      std::cout << "# Error at operation: unit lower triangular solver" << std::endl;
      std::cout << "  diff: " << fabs(diff(result, vcl_result)) << std::endl;
      retval = EXIT_FAILURE;
   }





   //transposed upper triangular:
   std::cout << "Transposed upper triangular solver" << std::endl;
   viennacl::copy(rhs, vcl_rhs);
   result = ublas::solve(trans(matrix), rhs, ublas::upper_tag());
   vcl_result = viennacl::linalg::solve(trans(vcl_matrix), vcl_rhs, viennacl::linalg::upper_tag());
   if( fabs(diff(result, vcl_result)) > epsilon )
   {
      std::cout << "# Error at operation: upper triangular solver" << std::endl;
      std::cout << "  diff: " << fabs(diff(result, vcl_result)) << std::endl;
      retval = EXIT_FAILURE;
   }

   //transposed upper unit triangular:
   std::cout << "Transposed unit upper triangular solver" << std::endl;
   viennacl::copy(rhs, vcl_rhs);
   result = ublas::solve(trans(matrix), rhs, ublas::unit_upper_tag());
   vcl_result = viennacl::linalg::solve(trans(vcl_matrix), vcl_rhs, viennacl::linalg::unit_upper_tag());
   if( fabs(diff(result, vcl_result)) > epsilon )
   {
      std::cout << "# Error at operation: unit upper triangular solver" << std::endl;
      std::cout << "  diff: " << fabs(diff(result, vcl_result)) << std::endl;
      retval = EXIT_FAILURE;
   }

   //transposed lower triangular:
   std::cout << "Transposed lower triangular solver" << std::endl;
   viennacl::copy(rhs, vcl_rhs);
   result = ublas::solve(trans(matrix), rhs, ublas::lower_tag());
   vcl_result = viennacl::linalg::solve(trans(vcl_matrix), vcl_rhs, viennacl::linalg::lower_tag());
   if( fabs(diff(result, vcl_result)) > epsilon )
   {
      std::cout << "# Error at operation: lower triangular solver" << std::endl;
      std::cout << "  diff: " << fabs(diff(result, vcl_result)) << std::endl;
      retval = EXIT_FAILURE;
   }

   //transposed lower unit triangular:
   std::cout << "Transposed unit lower triangular solver" << std::endl;
   viennacl::copy(rhs, vcl_rhs);
   result = ublas::solve(trans(matrix), rhs, ublas::unit_lower_tag());
   vcl_result = viennacl::linalg::solve(trans(vcl_matrix), vcl_rhs, viennacl::linalg::unit_lower_tag());
   if( fabs(diff(result, vcl_result)) > epsilon )
   {
      std::cout << "# Error at operation: unit lower triangular solver" << std::endl;
      std::cout << "  diff: " << fabs(diff(result, vcl_result)) << std::endl;
      retval = EXIT_FAILURE;
   }
   
   
   //full solver:
   std::cout << "Full solver" << std::endl;
   unsigned int lu_dim = 100;
   ublas::matrix<NumericT> square_matrix(lu_dim, lu_dim);
   ublas::vector<NumericT> lu_rhs(lu_dim);
   viennacl::matrix<NumericT, F> vcl_square_matrix(lu_dim, lu_dim);
   viennacl::vector<NumericT> vcl_lu_rhs(lu_dim);

   for (std::size_t i=0; i<lu_dim; ++i)
     for (std::size_t j=0; j<lu_dim; ++j)
       square_matrix(i,j) = -static_cast<NumericT>(0.5) * random<NumericT>();

   //put some more weight on diagonal elements:
   for (std::size_t j=0; j<lu_dim; ++j)
   {
     square_matrix(j,j) = static_cast<NumericT>(20.0) + random<NumericT>();
     lu_rhs(j) = random<NumericT>();
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

   if( fabs(diff(lu_rhs, vcl_lu_rhs)) > epsilon )
   {
      std::cout << "# Error at operation: dense solver" << std::endl;
      std::cout << "  diff: " << fabs(diff(lu_rhs, vcl_lu_rhs)) << std::endl;
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
      if( retval == EXIT_SUCCESS )
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
         std::cout << "  layout: row-major" << std::endl;
         retval = test<NumericT, viennacl::row_major>(epsilon);
            if( retval == EXIT_SUCCESS )
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
            if( retval == EXIT_SUCCESS )
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
