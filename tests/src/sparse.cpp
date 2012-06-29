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

//
// *** ViennaCL
//
//#define VIENNACL_DEBUG_ALL
#define VIENNACL_HAVE_UBLAS 1
#include "viennacl/scalar.hpp"
#include "viennacl/compressed_matrix.hpp"
#include "viennacl/coordinate_matrix.hpp"
#include "viennacl/ell_matrix.hpp"
#include "viennacl/hyb_matrix.hpp"
#include "viennacl/vector.hpp"
#include "viennacl/linalg/prod.hpp"
#include "viennacl/linalg/norm_2.hpp"
#include "viennacl/linalg/jacobi_precond.hpp"
#include "viennacl/io/matrix_market.hpp"
#include "examples/tutorial/Random.hpp"
#include "examples/tutorial/vector-io.hpp"

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


template <typename ScalarType, typename VCL_MATRIX>
ScalarType diff(ublas::compressed_matrix<ScalarType> & cpu_matrix, VCL_MATRIX & gpu_matrix)
{
  typedef ublas::compressed_matrix<ScalarType>  CPU_MATRIX;
   CPU_MATRIX from_gpu;
   
   copy(gpu_matrix, from_gpu);

   ScalarType error = 0;
   
   //step 1: compare all entries from cpu_matrix with gpu_matrix:
    for (typename CPU_MATRIX::const_iterator1 row_it = cpu_matrix.begin1();
          row_it != cpu_matrix.end1();
          ++row_it)
    {
      for (typename CPU_MATRIX::const_iterator2 col_it = row_it.begin();
            col_it != row_it.end();
            ++col_it)
      {
        ScalarType current_error = 0;
        
        if ( std::max( fabs(cpu_matrix(col_it.index1(), col_it.index2())), 
                       fabs(from_gpu(col_it.index1(), col_it.index2()))   ) > 0 )
          current_error = fabs(cpu_matrix(col_it.index1(), col_it.index2()) - from_gpu(col_it.index1(), col_it.index2())) 
                            / std::max( fabs(cpu_matrix(col_it.index1(), col_it.index2())), 
                                        fabs(from_gpu(col_it.index1(), col_it.index2()))   );
        if (current_error > error)
          error = current_error;
      }
    }

   //step 2: compare all entries from gpu_matrix with cpu_matrix (sparsity pattern might differ):
    for (typename CPU_MATRIX::const_iterator1 row_it = from_gpu.begin1();
          row_it != from_gpu.end1();
          ++row_it)
    {
      for (typename CPU_MATRIX::const_iterator2 col_it = row_it.begin();
            col_it != row_it.end();
            ++col_it)
      {
        ScalarType current_error = 0;
        
        if ( std::max( fabs(cpu_matrix(col_it.index1(), col_it.index2())), 
                       fabs(from_gpu(col_it.index1(), col_it.index2()))   ) > 0 )
          current_error = fabs(cpu_matrix(col_it.index1(), col_it.index2()) - from_gpu(col_it.index1(), col_it.index2())) 
                            / std::max( fabs(cpu_matrix(col_it.index1(), col_it.index2())), 
                                        fabs(from_gpu(col_it.index1(), col_it.index2()))   );
        if (current_error > error)
          error = current_error;
      }
    }

   return error;
}


template< typename NumericT, typename VCL_MATRIX, typename Epsilon >
int resize_test(Epsilon const& epsilon)
{
   int retval = EXIT_SUCCESS;
   
   ublas::compressed_matrix<NumericT> ublas_matrix(5,5);
   VCL_MATRIX vcl_matrix;    
   
   ublas_matrix(0,0) = 10.0; ublas_matrix(0, 1) = 0.1; ublas_matrix(0, 2) = 0.2; ublas_matrix(0, 3) = 0.3; ublas_matrix(0, 4) = 0.4;
   ublas_matrix(1,0) = 1.0; ublas_matrix(1, 1) = 1.1; ublas_matrix(1, 2) = 1.2; ublas_matrix(1, 3) = 1.3; ublas_matrix(1, 4) = 1.4;
   ublas_matrix(2,0) = 2.0; ublas_matrix(2, 1) = 2.1; ublas_matrix(2, 2) = 2.2; ublas_matrix(2, 3) = 2.3; ublas_matrix(2, 4) = 2.4;
   ublas_matrix(3,0) = 3.0; ublas_matrix(3, 1) = 3.1; ublas_matrix(3, 2) = 3.2; ublas_matrix(3, 3) = 3.3; ublas_matrix(3, 4) = 3.4;
   ublas_matrix(4,0) = 4.0; ublas_matrix(4, 1) = 4.1; ublas_matrix(4, 2) = 4.2; ublas_matrix(4, 3) = 4.3; ublas_matrix(4, 4) = 4.4;
   
   copy(ublas_matrix, vcl_matrix); ublas_matrix.clear();
   copy(vcl_matrix, ublas_matrix);
   
   std::cout << "Checking for equality after copy..." << std::endl;   
    if( fabs(diff(ublas_matrix, vcl_matrix)) > epsilon )
    {
        std::cout << "# Error at operation: equality after copy with sparse matrix" << std::endl;
        std::cout << "  diff: " << fabs(diff(ublas_matrix, vcl_matrix)) << std::endl;
        retval = EXIT_FAILURE;
    }
   
   std::cout << "Testing resize to larger..." << std::endl;
   ublas_matrix.resize(10, 10, false); //ublas does not allow preserve = true here
   ublas_matrix(0,0) = 10.0; ublas_matrix(0, 1) = 0.1; ublas_matrix(0, 2) = 0.2; ublas_matrix(0, 3) = 0.3; ublas_matrix(0, 4) = 0.4;
   ublas_matrix(1,0) = 1.0; ublas_matrix(1, 1) = 1.1; ublas_matrix(1, 2) = 1.2; ublas_matrix(1, 3) = 1.3; ublas_matrix(1, 4) = 1.4;
   ublas_matrix(2,0) = 2.0; ublas_matrix(2, 1) = 2.1; ublas_matrix(2, 2) = 2.2; ublas_matrix(2, 3) = 2.3; ublas_matrix(2, 4) = 2.4;
   ublas_matrix(3,0) = 3.0; ublas_matrix(3, 1) = 3.1; ublas_matrix(3, 2) = 3.2; ublas_matrix(3, 3) = 3.3; ublas_matrix(3, 4) = 3.4;
   ublas_matrix(4,0) = 4.0; ublas_matrix(4, 1) = 4.1; ublas_matrix(4, 2) = 4.2; ublas_matrix(4, 3) = 4.3; ublas_matrix(4, 4) = 4.4;
   //std::cout << ublas_matrix << std::endl;
   
   vcl_matrix.resize(10, 10, true);
   
    if( fabs(diff(ublas_matrix, vcl_matrix)) > epsilon )
    {
        std::cout << "# Error at operation: resize (to larger) with sparse matrix" << std::endl;
        std::cout << "  diff: " << fabs(diff(ublas_matrix, vcl_matrix)) << std::endl;
        retval = EXIT_FAILURE;
    }

   ublas_matrix(5,5) = 5.5; ublas_matrix(5, 6) = 5.6; ublas_matrix(5, 7) = 5.7; ublas_matrix(5, 8) = 5.8; ublas_matrix(5, 9) = 5.9;
   ublas_matrix(6,5) = 6.5; ublas_matrix(6, 6) = 6.6; ublas_matrix(6, 7) = 6.7; ublas_matrix(6, 8) = 6.8; ublas_matrix(6, 9) = 6.9;
   ublas_matrix(7,5) = 7.5; ublas_matrix(7, 6) = 7.6; ublas_matrix(7, 7) = 7.7; ublas_matrix(7, 8) = 7.8; ublas_matrix(7, 9) = 7.9;
   ublas_matrix(8,5) = 8.5; ublas_matrix(8, 6) = 8.6; ublas_matrix(8, 7) = 8.7; ublas_matrix(8, 8) = 8.8; ublas_matrix(8, 9) = 8.9;
   ublas_matrix(9,5) = 9.5; ublas_matrix(9, 6) = 9.6; ublas_matrix(9, 7) = 9.7; ublas_matrix(9, 8) = 9.8; ublas_matrix(9, 9) = 9.9;
   copy(ublas_matrix, vcl_matrix);
    
   std::cout << "Testing resize to smaller..." << std::endl;
   ublas_matrix.resize(7, 7, false); //ublas does not allow preserve = true here
   ublas_matrix(0,0) = 10.0; ublas_matrix(0, 1) = 0.1; ublas_matrix(0, 2) = 0.2; ublas_matrix(0, 3) = 0.3; ublas_matrix(0, 4) = 0.4;
   ublas_matrix(1,0) = 1.0; ublas_matrix(1, 1) = 1.1; ublas_matrix(1, 2) = 1.2; ublas_matrix(1, 3) = 1.3; ublas_matrix(1, 4) = 1.4;
   ublas_matrix(2,0) = 2.0; ublas_matrix(2, 1) = 2.1; ublas_matrix(2, 2) = 2.2; ublas_matrix(2, 3) = 2.3; ublas_matrix(2, 4) = 2.4;
   ublas_matrix(3,0) = 3.0; ublas_matrix(3, 1) = 3.1; ublas_matrix(3, 2) = 3.2; ublas_matrix(3, 3) = 3.3; ublas_matrix(3, 4) = 3.4;
   ublas_matrix(4,0) = 4.0; ublas_matrix(4, 1) = 4.1; ublas_matrix(4, 2) = 4.2; ublas_matrix(4, 3) = 4.3; ublas_matrix(4, 4) = 4.4;
   ublas_matrix(5,5) = 5.5; ublas_matrix(5, 6) = 5.6; ublas_matrix(5, 7) = 5.7; ublas_matrix(5, 8) = 5.8; ublas_matrix(5, 9) = 5.9;
   ublas_matrix(6,5) = 6.5; ublas_matrix(6, 6) = 6.6; ublas_matrix(6, 7) = 6.7; ublas_matrix(6, 8) = 6.8; ublas_matrix(6, 9) = 6.9;

   vcl_matrix.resize(7, 7);

   //std::cout << ublas_matrix << std::endl;
    if( fabs(diff(ublas_matrix, vcl_matrix)) > epsilon )
    {
        std::cout << "# Error at operation: resize (to smaller) with sparse matrix" << std::endl;
        std::cout << "  diff: " << fabs(diff(ublas_matrix, vcl_matrix)) << std::endl;
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
   std::cout << "Testing resizing of compressed_matrix..." << std::endl;
   int retval = resize_test<NumericT, viennacl::compressed_matrix<NumericT> >(epsilon);
   std::cout << "Testing resizing of coordinate_matrix..." << std::endl;
   if (retval != EXIT_FAILURE)
     retval = resize_test<NumericT, viennacl::coordinate_matrix<NumericT> >(epsilon);
   
   // --------------------------------------------------------------------------            
   ublas::vector<NumericT> rhs;
   ublas::vector<NumericT> result;
   ublas::compressed_matrix<NumericT> ublas_matrix;

    if (!viennacl::io::read_matrix_market_file(ublas_matrix, "../../examples/testdata/mat65k.mtx"))
    {
      std::cout << "Error reading Matrix file" << std::endl;
      return EXIT_FAILURE;
    }
    //unsigned int cg_mat_size = cg_mat.size(); 
    std::cout << "done reading matrix" << std::endl;

    if (!readVectorFromFile("../../examples/testdata/rhs65025.txt", rhs))
    {
      std::cout << "Error reading RHS file" << std::endl;
      return EXIT_FAILURE;
    }
    std::cout << "done reading rhs" << std::endl;

    if (!readVectorFromFile("../../examples/testdata/result65025.txt", result))
    {
      std::cout << "Error reading Result file" << std::endl;
      return EXIT_FAILURE;
    }
    std::cout << "done reading result" << std::endl;
   

   viennacl::vector<NumericT> vcl_rhs(rhs.size());
   viennacl::vector<NumericT> vcl_result(result.size()); 
   viennacl::vector<NumericT> vcl_result2(result.size()); 
   viennacl::compressed_matrix<NumericT> vcl_compressed_matrix(rhs.size(), rhs.size());
   viennacl::coordinate_matrix<NumericT> vcl_coordinate_matrix(rhs.size(), rhs.size());
   viennacl::ell_matrix<NumericT> vcl_ell_matrix(rhs.size(), rhs.size());
   viennacl::hyb_matrix<NumericT> vcl_hyb_matrix(rhs.size(), rhs.size());

   copy(rhs.begin(), rhs.end(), vcl_rhs.begin());
   copy(ublas_matrix, vcl_compressed_matrix);
   copy(ublas_matrix, vcl_coordinate_matrix);

   // --------------------------------------------------------------------------          
   std::cout << "Testing products: ublas" << std::endl;
   result     = viennacl::linalg::prod(ublas_matrix, rhs);
   
   std::cout << "Testing products: compressed_matrix" << std::endl;
   vcl_result = viennacl::linalg::prod(vcl_compressed_matrix, vcl_rhs);
   
   if( fabs(diff(result, vcl_result)) > epsilon )
   {
      std::cout << "# Error at operation: matrix-vector product with compressed_matrix" << std::endl;
      std::cout << "  diff: " << fabs(diff(result, vcl_result)) << std::endl;
      retval = EXIT_FAILURE;
   }
   
   std::cout << "Copying ell_matrix" << std::endl;
   copy(ublas_matrix, vcl_ell_matrix);
   ublas_matrix.clear();
   copy(vcl_ell_matrix, ublas_matrix);// just to check that it's works


   std::cout << "Testing products: ell_matrix" << std::endl;
   vcl_result.clear();
   vcl_result = viennacl::linalg::prod(vcl_ell_matrix, vcl_rhs);
   //viennacl::linalg::prod_impl(vcl_ell_matrix, vcl_rhs, vcl_result);
   //std::cout << vcl_result << "\n";
   std::cout << "  diff: " << fabs(diff(result, vcl_result)) << std::endl;
   std::cout << "First entry of result vector: " << vcl_result[0] << std::endl;
   
   if( fabs(diff(result, vcl_result)) > epsilon )
   {
      std::cout << "# Error at operation: matrix-vector product with ell_matrix" << std::endl;
      std::cout << "  diff: " << fabs(diff(result, vcl_result)) << std::endl;
      retval = EXIT_FAILURE;
   }
   
   
   std::cout << "Copying hyb_matrix" << std::endl;
   copy(ublas_matrix, vcl_hyb_matrix);
   ublas_matrix.clear();
   copy(vcl_hyb_matrix, ublas_matrix);// just to check that it's works
   copy(ublas_matrix, vcl_hyb_matrix);
 
   std::cout << "Testing products: hyb_matrix" << std::endl;
   vcl_result.clear();
   vcl_result = viennacl::linalg::prod(vcl_ell_matrix, vcl_rhs);
   //viennacl::linalg::prod_impl(vcl_hyb_matrix, vcl_rhs, vcl_result);
   //std::cout << vcl_result << "\n";
   std::cout << "  diff: " << fabs(diff(result, vcl_result)) << std::endl;
   std::cout << "First entry of result vector: " << vcl_result[0] << std::endl;
   
   if( fabs(diff(result, vcl_result)) > epsilon )
   {
      std::cout << "# Error at operation: matrix-vector product with hyb_matrix" << std::endl;
      std::cout << "  diff: " << fabs(diff(result, vcl_result)) << std::endl;
      retval = EXIT_FAILURE;
   }

   
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

   if( fabs(diff(result, vcl_result2)) > epsilon )
   {
      std::cout << "# Error at operation: matrix-vector product (compressed_matrix) with scaled additions" << std::endl;
      std::cout << "  diff: " << fabs(diff(result, vcl_result2)) << std::endl;
      retval = EXIT_FAILURE;
   }

   
/*   vcl_result2 = alpha * viennacl::linalg::prod(vcl_coordinate_matrix, vcl_rhs) + beta * vcl_result;

   if( fabs(diff(result, vcl_result2)) > epsilon )
   {
      std::cout << "# Error at operation: matrix-vector product (coordinate_matrix) with scaled additions" << std::endl;
      std::cout << "  diff: " << fabs(diff(result, vcl_result2)) << std::endl;
      retval = EXIT_FAILURE;
   }*/

   
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
      NumericT epsilon = static_cast<NumericT>(5.0E-2);
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
   
/*   {
      typedef float NumericT;
      NumericT epsilon = 1.0E-6;
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
   std::cout << std::endl;*/
   
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
      
/*      {
         typedef double NumericT;
         NumericT epsilon = 1.0E-15;
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
      std::cout << std::endl;*/
   }
   else
     std::cout << "No double precision support..." << std::endl;
   return retval;
}
