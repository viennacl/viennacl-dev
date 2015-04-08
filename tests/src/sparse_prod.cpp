/* =========================================================================
   Copyright (c) 2010-2014, Institute for Microelectronics,
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
#include <boost/numeric/ublas/matrix_sparse.hpp>
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/matrix_proxy.hpp>
#include <boost/numeric/ublas/operation_sparse.hpp>

//
// *** ViennaCL
//
//#define VIENNACL_DEBUG_ALL
#define VIENNACL_WITH_UBLAS 1

#include "viennacl/scalar.hpp"
#include "viennacl/compressed_matrix.hpp"
#include "viennacl/linalg/prod.hpp"

#include "examples/tutorial/Random.hpp"

//
// -------------------------------------------------------------
//
using namespace boost::numeric;
//
// -------------------------------------------------------------
//

/* Routine for computing the relative difference of two matrices. 1 is returned if the sparsity patterns do not match. */
template<typename NumericT, typename MatrixT>
NumericT diff(ublas::compressed_matrix<NumericT> const & ublas_A,
              MatrixT & vcl_A)
{
  viennacl::switch_memory_context(vcl_A, viennacl::context(viennacl::MAIN_MEMORY));

  NumericT error = NumericT(-1.0);

  NumericT     const * vcl_A_elements   = viennacl::linalg::host_based::detail::extract_raw_pointer<NumericT>(vcl_A.handle());
  unsigned int const * vcl_A_row_buffer = viennacl::linalg::host_based::detail::extract_raw_pointer<unsigned int>(vcl_A.handle1());
  unsigned int const * vcl_A_col_buffer = viennacl::linalg::host_based::detail::extract_raw_pointer<unsigned int>(vcl_A.handle2());


  /* Simultaneously compare the sparsity patterns of both matrices against each other. */
  //std::cout << "Ublas matrix: " << std::endl;

  unsigned int const * vcl_A_current_col_ptr = vcl_A_col_buffer;
  NumericT     const * vcl_A_current_val_ptr = vcl_A_elements;

  for (typename ublas::compressed_matrix<NumericT>::const_iterator1 ublas_A_row_it = ublas_A.begin1();
        ublas_A_row_it != ublas_A.end1();
        ++ublas_A_row_it)
  {
    std::size_t row_index = ublas_A_row_it.index1();
    if (vcl_A_current_col_ptr != vcl_A_col_buffer + vcl_A_row_buffer[row_index])
    {
      std::cerr << "Sparsity pattern mismatch detected: Start of row out of sync!" << std::endl;
      std::cerr << " ublas row: " << ublas_A_row_it.index1() << std::endl;
      std::cerr << " ViennaCL col ptr is: " << vcl_A_current_col_ptr << std::endl;
      std::cerr << " ViennaCL col ptr should: " << vcl_A_col_buffer + vcl_A_row_buffer[row_index] << std::endl;
      std::cerr << " ViennaCL col ptr value: " << *vcl_A_current_col_ptr << std::endl;
      return NumericT(1.0);
    }

    //std::cout << "Row " << row_it.index1() << ": " << std::endl;
    for (typename ublas::compressed_matrix<NumericT>::const_iterator2 ublas_A_col_it = ublas_A_row_it.begin();
          ublas_A_col_it != ublas_A_row_it.end();
          ++ublas_A_col_it, ++vcl_A_current_col_ptr, ++vcl_A_current_val_ptr)
    {
      if (ublas_A_col_it.index2() != std::size_t(*vcl_A_current_col_ptr))
      {
        std::cerr << "Sparsity pattern mismatch detected!" << std::endl;
        std::cerr << " ublas row: " << ublas_A_col_it.index1() << std::endl;
        std::cerr << " ublas col: " << ublas_A_col_it.index2() << std::endl;
        std::cerr << " ViennaCL row entries: " << vcl_A_row_buffer[row_index] << ", " << vcl_A_row_buffer[row_index + 1] << std::endl;
        std::cerr << " ViennaCL entry in row: " << vcl_A_current_col_ptr - (vcl_A_col_buffer + vcl_A_row_buffer[row_index]) << std::endl;
        std::cerr << " ViennaCL col: " << *vcl_A_current_col_ptr << std::endl;
        return NumericT(1.0);
      }

      // compute relative error (we know for sure that the uBLAS matrix only carries nonzero entries:
      NumericT current_error = std::fabs(*ublas_A_col_it - *vcl_A_current_val_ptr) / std::max(std::fabs(*ublas_A_col_it), std::fabs(*vcl_A_current_val_ptr));

      if (current_error > 0.1)
      {
        std::cerr << "Value mismatch detected!" << std::endl;
        std::cerr << " ublas row: " << ublas_A_col_it.index1() << std::endl;
        std::cerr << " ublas col: " << ublas_A_col_it.index2() << std::endl;
        std::cerr << " ublas value: " << *ublas_A_col_it << std::endl;
        std::cerr << " ViennaCL value: " << *vcl_A_current_val_ptr << std::endl;
        return NumericT(1.0);
      }

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

  std::size_t N = 400;
  std::size_t K = 100;
  std::size_t M = 300;
  std::size_t nnz_row = 20;
  // --------------------------------------------------------------------------
  ublas::compressed_matrix<NumericT> ublas_A(N, K);
  ublas::compressed_matrix<NumericT> ublas_B(K, M);
  ublas::compressed_matrix<NumericT> ublas_C;

  for (std::size_t i=0; i<ublas_A.size1(); ++i)
    for (std::size_t j=0; j<nnz_row; ++j)
      ublas_A(i, std::size_t(random<double>() * double(ublas_A.size2()))) = NumericT(1.0) + random<NumericT>();

  for (std::size_t i=0; i<ublas_B.size1(); ++i)
    for (std::size_t j=0; j<nnz_row; ++j)
      ublas_B(i, std::size_t(random<double>() * double(ublas_B.size2()))) = NumericT(1.0) + random<NumericT>();


  viennacl::compressed_matrix<NumericT>  vcl_A(ublas_A.size1(), ublas_A.size2());
  viennacl::compressed_matrix<NumericT>  vcl_B(ublas_B.size1(), ublas_B.size2());
  viennacl::compressed_matrix<NumericT>  vcl_C;

  viennacl::copy(ublas_A, vcl_A);
  viennacl::copy(ublas_B, vcl_B);

  // --------------------------------------------------------------------------
  std::cout << "Testing products: ublas" << std::endl;
  ublas_C = prod(ublas_A, ublas_B);

  std::cout << "Testing products: compressed_matrix" << std::endl;
  vcl_C = viennacl::linalg::prod(vcl_A, vcl_B);

  if ( std::fabs(diff(ublas_C, vcl_C)) > epsilon )
  {
    std::cout << "# Error at operation: matrix-matrix product with compressed_matrix" << std::endl;
    std::cout << "  diff: " << std::fabs(diff(ublas_C, vcl_C)) << std::endl;
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
  std::cout << "## Test :: Sparse Matrix Product" << std::endl;
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
