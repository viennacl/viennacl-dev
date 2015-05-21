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

//
// *** System
//
#include <iostream>
#include <vector>
#include <map>


//
// *** ViennaCL
//
#include "viennacl/scalar.hpp"
#include "viennacl/compressed_matrix.hpp"
#include "viennacl/linalg/prod.hpp"

#include "examples/tutorial/Random.hpp"

//
// -------------------------------------------------------------
//

/* Routine for computing the relative difference of two matrices. 1 is returned if the sparsity patterns do not match. */
template<typename IndexT, typename NumericT, typename MatrixT>
NumericT diff(std::vector<std::map<IndexT, NumericT> > const & stl_A,
              MatrixT & vcl_A)
{
  viennacl::switch_memory_context(vcl_A, viennacl::context(viennacl::MAIN_MEMORY));

  NumericT error = NumericT(-1.0);

  NumericT     const * vcl_A_elements   = viennacl::linalg::host_based::detail::extract_raw_pointer<NumericT>(vcl_A.handle());
  unsigned int const * vcl_A_row_buffer = viennacl::linalg::host_based::detail::extract_raw_pointer<unsigned int>(vcl_A.handle1());
  unsigned int const * vcl_A_col_buffer = viennacl::linalg::host_based::detail::extract_raw_pointer<unsigned int>(vcl_A.handle2());


  /* Simultaneously compare the sparsity patterns of both matrices against each other. */

  unsigned int const * vcl_A_current_col_ptr = vcl_A_col_buffer;
  NumericT     const * vcl_A_current_val_ptr = vcl_A_elements;

  for (std::size_t row = 0; row < stl_A.size(); ++row)
  {
    if (vcl_A_current_col_ptr != vcl_A_col_buffer + vcl_A_row_buffer[row])
    {
      std::cerr << "Sparsity pattern mismatch detected: Start of row out of sync!" << std::endl;
      std::cerr << " STL row: " << row << std::endl;
      std::cerr << " ViennaCL col ptr is: " << vcl_A_current_col_ptr << std::endl;
      std::cerr << " ViennaCL col ptr should: " << vcl_A_col_buffer + vcl_A_row_buffer[row] << std::endl;
      std::cerr << " ViennaCL col ptr value: " << *vcl_A_current_col_ptr << std::endl;
      return NumericT(1.0);
    }

    //std::cout << "Row " << row_it.index1() << ": " << std::endl;
    for (typename std::map<IndexT, NumericT>::const_iterator col_it = stl_A[row].begin();
          col_it != stl_A[row].end();
          ++col_it, ++vcl_A_current_col_ptr, ++vcl_A_current_val_ptr)
    {
      if (col_it->first != std::size_t(*vcl_A_current_col_ptr))
      {
        std::cerr << "Sparsity pattern mismatch detected!" << std::endl;
        std::cerr << " STL row: " << row << std::endl;
        std::cerr << " STL col: " << col_it->first << std::endl;
        std::cerr << " ViennaCL row entries: " << vcl_A_row_buffer[row] << ", " << vcl_A_row_buffer[row + 1] << std::endl;
        std::cerr << " ViennaCL entry in row: " << vcl_A_current_col_ptr - (vcl_A_col_buffer + vcl_A_row_buffer[row]) << std::endl;
        std::cerr << " ViennaCL col: " << *vcl_A_current_col_ptr << std::endl;
        return NumericT(1.0);
      }

      // compute relative error (we know for sure that the uBLAS matrix only carries nonzero entries:
      NumericT current_error = std::fabs(col_it->second - *vcl_A_current_val_ptr) / std::max(std::fabs(col_it->second), std::fabs(*vcl_A_current_val_ptr));

      if (current_error > 0.1)
      {
        std::cerr << "Value mismatch detected!" << std::endl;
        std::cerr << " STL row: " << row << std::endl;
        std::cerr << " STL col: " << col_it->first << std::endl;
        std::cerr << " STL value: " << col_it->second << std::endl;
        std::cerr << " ViennaCL value: " << *vcl_A_current_val_ptr << std::endl;
        return NumericT(1.0);
      }

      if (current_error > error)
        error = current_error;
    }
  }

  return error;
}

template<typename IndexT, typename NumericT>
void prod(std::vector<std::map<IndexT, NumericT> > const & stl_A,
          std::vector<std::map<IndexT, NumericT> > const & stl_B,
          std::vector<std::map<IndexT, NumericT> >       & stl_C)
{
  for (std::size_t i=0; i<stl_A.size(); ++i)
    for (typename std::map<IndexT, NumericT>::const_iterator it_A = stl_A[i].begin(); it_A != stl_A[i].end(); ++it_A)
    {
      IndexT row_B = it_A->first;
      for (typename std::map<IndexT, NumericT>::const_iterator it_B = stl_B[row_B].begin(); it_B != stl_B[row_B].end(); ++it_B)
        stl_C[i][it_B->first] += it_A->second * it_B->second;
    }
}


//
// -------------------------------------------------------------
//
template< typename NumericT, typename Epsilon >
int test(Epsilon const& epsilon)
{
  int retval = EXIT_SUCCESS;

  std::size_t N = 210;
  std::size_t K = 300;
  std::size_t M = 420;
  std::size_t nnz_row = 40;
  // --------------------------------------------------------------------------
  std::vector<std::map<unsigned int, NumericT> > stl_A(N);
  std::vector<std::map<unsigned int, NumericT> > stl_B(K);
  std::vector<std::map<unsigned int, NumericT> > stl_C(N);

  for (std::size_t i=0; i<stl_A.size(); ++i)
    for (std::size_t j=0; j<nnz_row; ++j)
      stl_A[i][static_cast<unsigned int>(random<double>() * double(K))] = NumericT(1.0) + random<NumericT>();

  for (std::size_t i=0; i<stl_B.size(); ++i)
    for (std::size_t j=0; j<nnz_row; ++j)
      stl_B[i][static_cast<unsigned int>(random<double>() * double(M))] = NumericT(1.0) + random<NumericT>();


  viennacl::compressed_matrix<NumericT>  vcl_A(N, K);
  viennacl::compressed_matrix<NumericT>  vcl_B(K, M);
  viennacl::compressed_matrix<NumericT>  vcl_C;

  viennacl::tools::sparse_matrix_adapter<NumericT> adapted_stl_A(stl_A, N, K);
  viennacl::tools::sparse_matrix_adapter<NumericT> adapted_stl_B(stl_B, K, M);
  viennacl::copy(adapted_stl_A, vcl_A);
  viennacl::copy(adapted_stl_B, vcl_B);

  // --------------------------------------------------------------------------
  std::cout << "Testing products: STL" << std::endl;
  prod(stl_A, stl_B, stl_C);

  std::cout << "Testing products: compressed_matrix" << std::endl;
  vcl_C = viennacl::linalg::prod(vcl_A, vcl_B);

  if ( std::fabs(diff(stl_C, vcl_C)) > epsilon )
  {
    std::cout << "# Error at operation: matrix-matrix product with compressed_matrix" << std::endl;
    std::cout << "  diff: " << std::fabs(diff(stl_C, vcl_C)) << std::endl;
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
