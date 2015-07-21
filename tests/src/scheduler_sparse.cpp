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


/** \file tests/src/scheduler_sparse.cpp  Tests the scheduler for sparse matrix operations.
*   \test Tests the scheduler for sparse matrix operations.
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

//
// *** ViennaCL
//
//#define VIENNACL_DEBUG_ALL
#define VIENNACL_WITH_UBLAS 1
#include "viennacl/scalar.hpp"
#include "viennacl/compressed_matrix.hpp"
#include "viennacl/coordinate_matrix.hpp"
#include "viennacl/ell_matrix.hpp"
#include "viennacl/hyb_matrix.hpp"
#include "viennacl/vector.hpp"
#include "viennacl/vector_proxy.hpp"
#include "viennacl/linalg/prod.hpp"
#include "viennacl/linalg/norm_2.hpp"
#include "viennacl/linalg/ilu.hpp"
#include "viennacl/linalg/detail/ilu/common.hpp"
#include "viennacl/io/matrix_market.hpp"
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
        std::cout << "Error at entry " << i   << ": " << v1[i]   << " vs. " << v2_cpu[i]   << std::endl;
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
  CPU_MATRIX from_gpu;

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

//
// -------------------------------------------------------------
//
template< typename NumericT, typename Epsilon >
int test(Epsilon const& epsilon)
{
  int retval = EXIT_SUCCESS;

  viennacl::tools::uniform_random_numbers<NumericT> randomNumber;

  // --------------------------------------------------------------------------
  NumericT alpha = static_cast<NumericT>(2.786);
  NumericT beta = static_cast<NumericT>(1.432);

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

  result = rhs;


  viennacl::vector<NumericT> vcl_rhs(rhs.size());
  viennacl::vector<NumericT> vcl_result(result.size());
  viennacl::vector<NumericT> vcl_result2(result.size());
  viennacl::compressed_matrix<NumericT> vcl_compressed_matrix(rhs.size(), rhs.size());
  viennacl::coordinate_matrix<NumericT> vcl_coordinate_matrix(rhs.size(), rhs.size());
  viennacl::ell_matrix<NumericT> vcl_ell_matrix;
  viennacl::hyb_matrix<NumericT> vcl_hyb_matrix;

  viennacl::copy(rhs.begin(), rhs.end(), vcl_rhs.begin());
  viennacl::copy(ublas_matrix, vcl_compressed_matrix);
  viennacl::copy(ublas_matrix, vcl_coordinate_matrix);

  // --------------------------------------------------------------------------
  std::cout << "Testing products: compressed_matrix" << std::endl;
  result     = viennacl::linalg::prod(ublas_matrix, rhs);
  {
  viennacl::scheduler::statement my_statement(vcl_result, viennacl::op_assign(), viennacl::linalg::prod(vcl_compressed_matrix, vcl_rhs));
  viennacl::scheduler::execute(my_statement);
  }
  vcl_result = viennacl::linalg::prod(vcl_compressed_matrix, vcl_rhs);

  if ( std::fabs(diff(result, vcl_result)) > epsilon )
  {
    std::cout << "# Error at operation: matrix-vector product with compressed_matrix" << std::endl;
    std::cout << "  diff: " << std::fabs(diff(result, vcl_result)) << std::endl;
    retval = EXIT_FAILURE;
  }

  std::cout << "Testing products: coordinate_matrix" << std::endl;
  rhs *= NumericT(1.1);
  vcl_rhs *= NumericT(1.1);
  result     = viennacl::linalg::prod(ublas_matrix, rhs);
  {
  viennacl::scheduler::statement my_statement(vcl_result, viennacl::op_assign(), viennacl::linalg::prod(vcl_coordinate_matrix, vcl_rhs));
  viennacl::scheduler::execute(my_statement);
  }

  if ( std::fabs(diff(result, vcl_result)) > epsilon )
  {
    std::cout << "# Error at operation: matrix-vector product with coordinate_matrix" << std::endl;
    std::cout << "  diff: " << std::fabs(diff(result, vcl_result)) << std::endl;
    retval = EXIT_FAILURE;
  }

  result     = alpha * viennacl::linalg::prod(ublas_matrix, rhs) + beta * result;
  {
  viennacl::scheduler::statement my_statement(vcl_result2, viennacl::op_assign(), alpha * viennacl::linalg::prod(vcl_coordinate_matrix, vcl_rhs) + beta * vcl_result);
  viennacl::scheduler::execute(my_statement);
  }

  if ( std::fabs(diff(result, vcl_result2)) > epsilon )
  {
    std::cout << "# Error at operation: matrix-vector product (coordinate_matrix) with scaled additions" << std::endl;
    std::cout << "  diff: " << std::fabs(diff(result, vcl_result2)) << std::endl;
    retval = EXIT_FAILURE;
  }

  //std::cout << "Copying ell_matrix" << std::endl;
  viennacl::copy(ublas_matrix, vcl_ell_matrix);
  ublas_matrix.clear();
  viennacl::copy(vcl_ell_matrix, ublas_matrix);// just to check that it's works


  std::cout << "Testing products: ell_matrix" << std::endl;
  rhs *= NumericT(1.1);
  vcl_rhs *= NumericT(1.1);
  result     = viennacl::linalg::prod(ublas_matrix, rhs);
  {
  //viennacl::scheduler::statement my_statement(vcl_result, viennacl::op_assign(), viennacl::linalg::prod(vcl_ell_matrix, vcl_rhs));
  //viennacl::scheduler::execute(my_statement);
  }
  vcl_result = viennacl::linalg::prod(vcl_ell_matrix, vcl_rhs);

  if ( std::fabs(diff(result, vcl_result)) > epsilon )
  {
    std::cout << "# Error at operation: matrix-vector product with ell_matrix" << std::endl;
    std::cout << "  diff: " << std::fabs(diff(result, vcl_result)) << std::endl;
    retval = EXIT_FAILURE;
  }

  //std::cout << "Copying hyb_matrix" << std::endl;
  viennacl::copy(ublas_matrix, vcl_hyb_matrix);
  ublas_matrix.clear();
  viennacl::copy(vcl_hyb_matrix, ublas_matrix);// just to check that it's works
  viennacl::copy(ublas_matrix, vcl_hyb_matrix);

  std::cout << "Testing products: hyb_matrix" << std::endl;
  rhs *= NumericT(1.1);
  vcl_rhs *= NumericT(1.1);
  result     = viennacl::linalg::prod(ublas_matrix, rhs);
  {
  viennacl::scheduler::statement my_statement(vcl_result, viennacl::op_assign(), viennacl::linalg::prod(vcl_hyb_matrix, vcl_rhs));
  viennacl::scheduler::execute(my_statement);
  }

  if ( std::fabs(diff(result, vcl_result)) > epsilon )
  {
    std::cout << "# Error at operation: matrix-vector product with hyb_matrix" << std::endl;
    std::cout << "  diff: " << std::fabs(diff(result, vcl_result)) << std::endl;
    retval = EXIT_FAILURE;
  }


  // --------------------------------------------------------------------------
  // --------------------------------------------------------------------------
  copy(rhs.begin(), rhs.end(), vcl_rhs.begin());
  copy(result.begin(), result.end(), vcl_result.begin());
  copy(result.begin(), result.end(), vcl_result2.begin());
  copy(ublas_matrix, vcl_compressed_matrix);
  copy(ublas_matrix, vcl_coordinate_matrix);
  copy(ublas_matrix, vcl_ell_matrix);
  copy(ublas_matrix, vcl_hyb_matrix);

  std::cout << "Testing scaled additions of products and vectors: compressed_matrix" << std::endl;
  rhs *= NumericT(1.1);
  vcl_rhs *= NumericT(1.1);
  result     = alpha * viennacl::linalg::prod(ublas_matrix, rhs) + beta * result;
  {
  viennacl::scheduler::statement my_statement(vcl_result2, viennacl::op_assign(), alpha * viennacl::linalg::prod(vcl_compressed_matrix, vcl_rhs) + beta * vcl_result);
  viennacl::scheduler::execute(my_statement);
  }

  if ( std::fabs(diff(result, vcl_result2)) > epsilon )
  {
    std::cout << "# Error at operation: matrix-vector product (compressed_matrix) with scaled additions" << std::endl;
    std::cout << "  diff: " << std::fabs(diff(result, vcl_result2)) << std::endl;
    retval = EXIT_FAILURE;
  }


  std::cout << "Testing scaled additions of products and vectors: coordinate_matrix" << std::endl;
  copy(result.begin(), result.end(), vcl_result.begin());
  rhs *= NumericT(1.1);
  vcl_rhs *= NumericT(1.1);
  result     = alpha * viennacl::linalg::prod(ublas_matrix, rhs) + beta * result;
  {
  viennacl::scheduler::statement my_statement(vcl_result2, viennacl::op_assign(), alpha * viennacl::linalg::prod(vcl_coordinate_matrix, vcl_rhs) + beta * vcl_result);
  viennacl::scheduler::execute(my_statement);
  }

  if ( std::fabs(diff(result, vcl_result2)) > epsilon )
  {
    std::cout << "# Error at operation: matrix-vector product (coordinate_matrix) with scaled additions" << std::endl;
    std::cout << "  diff: " << std::fabs(diff(result, vcl_result2)) << std::endl;
    retval = EXIT_FAILURE;
  }

  std::cout << "Testing scaled additions of products and vectors: ell_matrix" << std::endl;
  copy(result.begin(), result.end(), vcl_result.begin());
  rhs *= NumericT(1.1);
  vcl_rhs *= NumericT(1.1);
  result     = alpha * viennacl::linalg::prod(ublas_matrix, rhs) + beta * result;
  {
  viennacl::scheduler::statement my_statement(vcl_result2, viennacl::op_assign(), alpha * viennacl::linalg::prod(vcl_ell_matrix, vcl_rhs) + beta * vcl_result);
  viennacl::scheduler::execute(my_statement);
  }

  if ( std::fabs(diff(result, vcl_result2)) > epsilon )
  {
    std::cout << "# Error at operation: matrix-vector product (ell_matrix) with scaled additions" << std::endl;
    std::cout << "  diff: " << std::fabs(diff(result, vcl_result2)) << std::endl;
    retval = EXIT_FAILURE;
  }

  std::cout << "Testing scaled additions of products and vectors: hyb_matrix" << std::endl;
  copy(result.begin(), result.end(), vcl_result.begin());
  rhs *= NumericT(1.1);
  vcl_rhs *= NumericT(1.1);
  result     = alpha * viennacl::linalg::prod(ublas_matrix, rhs) + beta * result;
  {
  viennacl::scheduler::statement my_statement(vcl_result2, viennacl::op_assign(), alpha * viennacl::linalg::prod(vcl_hyb_matrix, vcl_rhs) + beta * vcl_result);
  viennacl::scheduler::execute(my_statement);
  }

  if ( std::fabs(diff(result, vcl_result2)) > epsilon )
  {
    std::cout << "# Error at operation: matrix-vector product (hyb_matrix) with scaled additions" << std::endl;
    std::cout << "  diff: " << std::fabs(diff(result, vcl_result2)) << std::endl;
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
      NumericT epsilon = 1.0E-13;
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
