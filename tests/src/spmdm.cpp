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
#include "viennacl/linalg/lu.hpp"         //LU substitution routines


// Some helper functions for this tutorial:
#include "Random.hpp"


using namespace boost::numeric;

template < typename ScalarType >
int check_matrices(const ublas::matrix< ScalarType >& ref_mat, const ublas::matrix< ScalarType >& mat) {

  std::size_t size1, size2;
  ScalarType eps = 0.00001;
  size1 = ref_mat.size1(); size2 = ref_mat.size2();
  if( (size1 != mat.size1()) || (size2 != mat.size2()) )
    return EXIT_FAILURE;

  for (unsigned int i = 0; i < size1; i++)
    for (unsigned int j = 0; j < size2; j++)
      if ( abs(ref_mat(i,j) - mat(i,j)) > eps ) {
        std::cout << "!!Verification failed at " << i <<" : "<< j
                  << "(expected: " << ref_mat(i,j) << " get: " << mat(i,j) << " )" << std::endl;
        return EXIT_FAILURE;
      }

  std::cout << "Everything went well!" << std::endl;
  return EXIT_SUCCESS;
}


int main()
{
  typedef float       ScalarType;
  
  std::size_t size = 1024, size1, size2;
  int retVal = EXIT_SUCCESS;

  ublas::compressed_matrix<ScalarType> ublas_lhs(size/2, size);
  viennacl::compressed_matrix<ScalarType> compressed_lhs(size/2, size);
  viennacl::ell_matrix<ScalarType> ell_lhs;
  viennacl::coordinate_matrix<ScalarType> coo_lhs;

  ublas::matrix<ScalarType> ublas_rhs1(size, size/2);
  viennacl::matrix<ScalarType> rhs1(size, size/2);

//  ublas::matrix<ScalarType> ublas_rhs2(size/2, size);
//  viennacl::matrix<ScalarType> rhs2(size/2, size);
  ublas::matrix<ScalarType> ublas_rhs2;
  viennacl::matrix<ScalarType> rhs2;

//  ublas::matrix<ScalarType> ublas_result( size/2, size/2);
//  viennacl::matrix<ScalarType> result( size/2, size/2);
  ublas::matrix<ScalarType> ublas_result;
  viennacl::matrix<ScalarType> result;

  ublas::matrix<ScalarType> temp( size/2, size/2);


  size1 = size/2;
  size2 = size;
  ublas_lhs(0,0) = 3.3f; ublas_lhs(0,1) = 2.2f; ublas_lhs(0,2) = 1.1f;
  ublas_lhs(1,0) = -2.2f; ublas_lhs(1,1) = 3.3f; ublas_lhs(1,2) = 2.2f; ublas_lhs(1,3) = 1.1f;
  for (unsigned int i = 2; i < size1; i++) {
    ublas_lhs(i, i-2) = -1.1f; ublas_lhs(i, i-1) = -2.2f; ublas_lhs(i, i) = 3.3f; ublas_lhs(i, i+1) = 2.2f; ublas_lhs(i, i+2) = 1.1f;
  }

  viennacl::copy( ublas_lhs, compressed_lhs);
  viennacl::copy( ublas_lhs, ell_lhs);
  viennacl::copy( ublas_lhs, coo_lhs);

  size1 = size;
  size2 = size/2;
  for (unsigned int i = 0; i < size1; i++)
    for (unsigned int j = 0; j < size2; j++)
      ublas_rhs1(i,j) = random<ScalarType>();
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
  retVal = check_matrices(ublas_result, temp);

  /******************************************************************/
  std::cout << "Testing compressed(ELL) lhs * dense rhs" << std::endl;
  result.clear();
  result = viennacl::linalg::prod( ell_lhs, rhs1);

  temp.clear();
  viennacl::copy( result, temp);
  check_matrices(ublas_result, temp);

  /******************************************************************/

//  std::cout << "Testing compressed(COO) lhs * dense rhs" << std::endl;
//  result.clear();
//  result = viennacl::linalg::prod( coo_lhs, rhs1);
//
//  temp.clear();
//  viennacl::copy( result, temp);
//  check_matrices(ublas_result, temp);

  /******************************************************************/

  /* gold result */
  ublas_result = ublas::prod( ublas_lhs, ublas::trans(ublas_rhs2));

  /******************************************************************/
  std::cout << std::endl << "Testing compressed(CSR) lhs * transposed dense rhs:" << std::endl;
  result.clear();
  result = viennacl::linalg::prod( compressed_lhs, viennacl::trans(rhs2));

  temp.clear();
  viennacl::copy( result, temp);
  retVal = check_matrices(ublas_result, temp);

  /******************************************************************/
  std::cout << "Testing compressed(ELL) lhs * transposed dense rhs" << std::endl;
  result.clear();
  result = viennacl::linalg::prod( ell_lhs, viennacl::trans(rhs2));

  temp.clear();
  viennacl::copy( result, temp);
  check_matrices(ublas_result, temp);

  /******************************************************************/
//  std::cout << "Testing compressed(COO) lhs * transposed dense rhs" << std::endl;
//  result.clear();
//  result = viennacl::linalg::prod( coo_lhs, viennacl::trans(rhs2));
//
//  temp.clear();
//  viennacl::copy( result, temp);
//  check_matrices(ublas_result, temp);

  /******************************************************************/
  if(retVal == EXIT_SUCCESS) {
    std::cout << "Tests passed successfully" << std::endl;
  }

  return retVal;
}
