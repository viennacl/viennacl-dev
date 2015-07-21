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


/** \file tests/src/spmdm.cpp  Tests sparse-matrix-dense-matrix products.
*   \test  Tests sparse-matrix-dense-matrix products.
**/

//
// include necessary system headers
//
#include <iostream>
#include <cmath>

//
// ublas includes
//
#include <boost/numeric/ublas/io.hpp>
#include <boost/numeric/ublas/triangular.hpp>
#include <boost/numeric/ublas/matrix_sparse.hpp>
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/matrix_proxy.hpp>
#include <boost/numeric/ublas/operation.hpp>
#include <boost/numeric/ublas/operation_sparse.hpp>
#include <boost/numeric/ublas/io.hpp>
#include <boost/numeric/ublas/lu.hpp>

// Must be set if you want to use ViennaCL algorithms on ublas objects
#define VIENNACL_WITH_UBLAS 1
//#define VIENNACL_WITH_OPENCL 1
//#define VIENNACL_WITH_CUDA 1
//#define VIENNACL_DEBUG_KERNEL 1
//#define VIENNACL_BUILD_INFO 1

//
// ViennaCL includes
//
#include "viennacl/scalar.hpp"
#include "viennacl/vector.hpp"
#include "viennacl/matrix.hpp"
#include "viennacl/linalg/direct_solve.hpp"
#include "viennacl/compressed_matrix.hpp"
#include "viennacl/coordinate_matrix.hpp"
#include "viennacl/ell_matrix.hpp"
#include "viennacl/hyb_matrix.hpp"
#include "viennacl/linalg/prod.hpp"       //generic matrix-vector product
#include "viennacl/linalg/norm_2.hpp"     //generic l2-norm for vectors
#include "viennacl/io/matrix_market.hpp"
#include "viennacl/tools/random.hpp"


using namespace boost::numeric;

template< typename ScalarType >
int check_matrices(const ublas::matrix< ScalarType >& ref_mat, const ublas::matrix< ScalarType >& mat, ScalarType eps) {

  std::size_t size1, size2;
  size1 = ref_mat.size1(); size2 = ref_mat.size2();
  if ( (size1 != mat.size1()) || (size2 != mat.size2()) )
    return EXIT_FAILURE;

  for (unsigned int i = 0; i < size1; i++)
    for (unsigned int j = 0; j < size2; j++)
    {
      ScalarType rel_error = std::abs(ref_mat(i,j) - mat(i,j)) / std::max(std::abs(ref_mat(i,j)), std::abs(mat(i,j)));
      if ( rel_error > eps ) {
        std::cout << "ERROR: Verification failed at (" << i <<", "<< j << "): "
                  << " Expected: " << ref_mat(i,j) << ", got: " << mat(i,j) << " (relative error: " << rel_error << ")" << std::endl;
        return EXIT_FAILURE;
      }
    }

  std::cout << "Everything went well!" << std::endl;
  return EXIT_SUCCESS;
}

template<typename NumericT, typename ResultLayoutT, typename FactorLayoutT>
int test(NumericT epsilon)
{
  int retVal = EXIT_SUCCESS;

  viennacl::tools::uniform_random_numbers<NumericT> randomNumber;

  ublas::compressed_matrix<NumericT>    ublas_lhs;

  if (viennacl::io::read_matrix_market_file(ublas_lhs, "../examples/testdata/mat65k.mtx") == EXIT_FAILURE)
  {
    std::cout << "Error reading Matrix file" << std::endl;
    return EXIT_FAILURE;
  }

  // add some extra weight to diagonal in order to avoid issues with round-off errors
  for (std::size_t i=0; i<ublas_lhs.size1(); ++i)
    ublas_lhs(i,i) *= NumericT(1.5);

  std::size_t cols_rhs = 1;

  viennacl::compressed_matrix<NumericT> compressed_lhs;
  viennacl::ell_matrix<NumericT>        ell_lhs;
  viennacl::coordinate_matrix<NumericT> coo_lhs;
  viennacl::hyb_matrix<NumericT>     hyb_lhs;

  ublas::matrix<NumericT> ublas_result;
  viennacl::matrix<NumericT, ResultLayoutT> result;

  viennacl::copy( ublas_lhs, compressed_lhs);
  viennacl::copy( ublas_lhs, ell_lhs);
  viennacl::copy( ublas_lhs, coo_lhs);
  viennacl::copy( ublas_lhs, hyb_lhs);

  ublas::matrix<NumericT> ublas_rhs1(ublas_lhs.size2(), cols_rhs);
  viennacl::matrix<NumericT, FactorLayoutT> rhs1(ublas_lhs.size2(), cols_rhs);

  ublas::matrix<NumericT> ublas_rhs2;
  viennacl::matrix<NumericT, FactorLayoutT> rhs2;

  ublas::matrix<NumericT> temp(ublas_rhs1.size1(), cols_rhs);

  for (unsigned int i = 0; i < ublas_rhs1.size1(); i++)
    for (unsigned int j = 0; j < ublas_rhs1.size2(); j++)
      ublas_rhs1(i,j) = NumericT(0.5) + NumericT(0.1) * randomNumber();
  viennacl::copy( ublas_rhs1, rhs1);

  ublas_rhs2 = ublas::trans( ublas_rhs1);
  viennacl::copy( ublas_rhs2, rhs2);

  /* gold result */
  ublas_result = ublas::prod( ublas_lhs, ublas_rhs1);

  /******************************************************************/
  std::cout << "Testing compressed(CSR) lhs * dense rhs" << std::endl;
  result = viennacl::linalg::prod( compressed_lhs, rhs1);

  temp.clear();
  viennacl::copy( result, temp);
  retVal = check_matrices(ublas_result, temp, epsilon);

  /******************************************************************/
  std::cout << "Testing compressed(ELL) lhs * dense rhs" << std::endl;
  result.clear();
  result = viennacl::linalg::prod( ell_lhs, rhs1);

  temp.clear();
  viennacl::copy( result, temp);
  check_matrices(ublas_result, temp, epsilon);

  /******************************************************************/

  std::cout << "Testing compressed(COO) lhs * dense rhs" << std::endl;
  result.clear();
  result = viennacl::linalg::prod( coo_lhs, rhs1);

  temp.clear();
  viennacl::copy( result, temp);
  check_matrices(ublas_result, temp, epsilon);

  /******************************************************************/

  std::cout << "Testing compressed(HYB) lhs * dense rhs" << std::endl;
  result.clear();
  result = viennacl::linalg::prod( hyb_lhs, rhs1);

  temp.clear();
  viennacl::copy( result, temp);
  check_matrices(ublas_result, temp, epsilon);

  /******************************************************************/

  /* gold result */
  ublas_result = ublas::prod( ublas_lhs, ublas::trans(ublas_rhs2));

  /******************************************************************/
  std::cout << std::endl << "Testing compressed(CSR) lhs * transposed dense rhs:" << std::endl;
  result.clear();
  result = viennacl::linalg::prod( compressed_lhs, viennacl::trans(rhs2));

  temp.clear();
  viennacl::copy( result, temp);
  retVal = check_matrices(ublas_result, temp, epsilon);

  /******************************************************************/
  std::cout << "Testing compressed(ELL) lhs * transposed dense rhs" << std::endl;
  result.clear();
  result = viennacl::linalg::prod( ell_lhs, viennacl::trans(rhs2));

  temp.clear();
  viennacl::copy( result, temp);
  check_matrices(ublas_result, temp, epsilon);

  /******************************************************************/
  std::cout << "Testing compressed(COO) lhs * transposed dense rhs" << std::endl;
  result.clear();
  result = viennacl::linalg::prod( coo_lhs, viennacl::trans(rhs2));

  temp.clear();
  viennacl::copy( result, temp);
  check_matrices(ublas_result, temp, epsilon);

  /******************************************************************/

  std::cout << "Testing compressed(HYB) lhs * transposed dense rhs" << std::endl;
  result.clear();
  result = viennacl::linalg::prod( hyb_lhs, viennacl::trans(rhs2));

  temp.clear();
  viennacl::copy( result, temp);
  check_matrices(ublas_result, temp, epsilon);

  /******************************************************************/
  if (retVal == EXIT_SUCCESS) {
    std::cout << "Tests passed successfully" << std::endl;
  }

  return retVal;
}

//
// -------------------------------------------------------------
//
int main()
{
  std::cout << std::endl;
  std::cout << "----------------------------------------------" << std::endl;
  std::cout << "----------------------------------------------" << std::endl;
  std::cout << "## Test :: Sparse-Dense Matrix Multiplication" << std::endl;
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
    std::cout << "  layout:  row-major, row-major" << std::endl;
    retval = test<NumericT, viennacl::row_major, viennacl::row_major>(epsilon);
    if ( retval == EXIT_SUCCESS )
        std::cout << "# Test passed" << std::endl;
    else
        return retval;

    std::cout << "# Testing setup:" << std::endl;
    std::cout << "  eps:     " << epsilon << std::endl;
    std::cout << "  numeric: float" << std::endl;
    std::cout << "  layout:  row-major, column-major" << std::endl;
    retval = test<NumericT, viennacl::row_major, viennacl::column_major>(epsilon);
    if ( retval == EXIT_SUCCESS )
        std::cout << "# Test passed" << std::endl;
    else
        return retval;

    std::cout << "# Testing setup:" << std::endl;
    std::cout << "  eps:     " << epsilon << std::endl;
    std::cout << "  numeric: float" << std::endl;
    std::cout << "  layout:  column-major, row-major" << std::endl;
    retval = test<NumericT, viennacl::column_major, viennacl::row_major>(epsilon);
    if ( retval == EXIT_SUCCESS )
        std::cout << "# Test passed" << std::endl;
    else
        return retval;

    std::cout << "# Testing setup:" << std::endl;
    std::cout << "  eps:     " << epsilon << std::endl;
    std::cout << "  numeric: float" << std::endl;
    std::cout << "  layout:  column-major, column-major" << std::endl;
    retval = test<NumericT, viennacl::column_major, viennacl::column_major>(epsilon);
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
      std::cout << "  layout:  row-major, row-major" << std::endl;
      retval = test<NumericT, viennacl::row_major, viennacl::row_major>(epsilon);
      if ( retval == EXIT_SUCCESS )
        std::cout << "# Test passed" << std::endl;
      else
        return retval;

      std::cout << "# Testing setup:" << std::endl;
      std::cout << "  eps:     " << epsilon << std::endl;
      std::cout << "  numeric: double" << std::endl;
      std::cout << "  layout:  row-major, column-major" << std::endl;
      retval = test<NumericT, viennacl::row_major, viennacl::column_major>(epsilon);
      if ( retval == EXIT_SUCCESS )
        std::cout << "# Test passed" << std::endl;
      else
        return retval;

      std::cout << "# Testing setup:" << std::endl;
      std::cout << "  eps:     " << epsilon << std::endl;
      std::cout << "  numeric: double" << std::endl;
      std::cout << "  layout:  column-major, row-major" << std::endl;
      retval = test<NumericT, viennacl::column_major, viennacl::row_major>(epsilon);
      if ( retval == EXIT_SUCCESS )
        std::cout << "# Test passed" << std::endl;
      else
        return retval;

      std::cout << "# Testing setup:" << std::endl;
      std::cout << "  eps:     " << epsilon << std::endl;
      std::cout << "  numeric: double" << std::endl;
      std::cout << "  layout:  column-major, column-major" << std::endl;
      retval = test<NumericT, viennacl::column_major, viennacl::column_major>(epsilon);
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

