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
#include "viennacl/linalg/prod.hpp"       //generic matrix-vector product
#include "viennacl/linalg/norm_2.hpp"     //generic l2-norm for vectors
#include "viennacl/linalg/lu.hpp"         //LU substitution routines


// Some helper functions for this tutorial:
#include "Random.hpp"


using namespace boost::numeric;


int main()
{
  typedef float       ScalarType;
  
  std::size_t size = 1024, size1, size2;

  ublas::compressed_matrix<ScalarType> u_lhs(size/2, size);
  viennacl::compressed_matrix<ScalarType> lhs(size/2, size);

  ublas::matrix<ScalarType> u_rhs1(size, size/2);
  viennacl::matrix<ScalarType> rhs1(size, size/2);

//  ublas::matrix<ScalarType> u_rhs2(size/2, size);
//  viennacl::matrix<ScalarType> rhs2(size/2, size);
  ublas::matrix<ScalarType> u_rhs2;
  viennacl::matrix<ScalarType> rhs2;

//  ublas::matrix<ScalarType> u_result( size/2, size/2);
//  viennacl::matrix<ScalarType> result( size/2, size/2);
  ublas::matrix<ScalarType> u_result;
  viennacl::matrix<ScalarType> result;

  ublas::matrix<ScalarType> temp( size/2, size/2);

  
  size1 = size/2;
  size2 = size;
  u_lhs(0,0) = 3.3f; u_lhs(0,1) = 2.2f; u_lhs(0,2) = 1.1f;
  u_lhs(1,0) = -2.2f; u_lhs(1,1) = 3.3f; u_lhs(1,2) = 2.2f; u_lhs(1,3) = 1.1f;
  for (unsigned int i = 2; i < size1; i++) {
    u_lhs(i, i-2) = -1.1f; u_lhs(i, i-1) = -2.2f; u_lhs(i, i) = 3.3f; u_lhs(i, i+1) = 2.2f; u_lhs(i, i+2) = 1.1f;
  }
  viennacl::copy( u_lhs, lhs);

  size1 = size;
  size2 = size/2;
  for (unsigned int i = 0; i < size1; i++)
    for (unsigned int j = 0; j < size2; j++)
      u_rhs1(i,j) = random<ScalarType>();
  viennacl::copy( u_rhs1, rhs1);

  u_rhs2 = ublas::trans( u_rhs1);
  viennacl::copy( u_rhs2, rhs2);

  u_result = ublas::prod( u_lhs, u_rhs1);
  result = viennacl::linalg::prod( lhs, rhs1);

  viennacl::copy( result, temp);
  
  ScalarType eps = 0.00001;
  std::cout << "dense rhs:" << std::endl;
  std::cout << "Checking results: " << std::endl;

  size1 = size/2;
  size2 = size/2;
  for (unsigned int i = 0; i < size1; i++)
    for (unsigned int j = 0; j < size2; j++)
      if ( abs(temp(i,j) - u_result(i,j)) > eps ) {
        std::cout << "!!Verification failed at " << i <<" : "<< j  << "(expected: " << u_result(i,j) << " get: " << temp(i,j) << " )" << std::endl;
        return EXIT_FAILURE;
      }

  std::cout << "Everything went well!" << std::endl;

  std::cout << std::endl << "dense transposed rhs:" << std::endl;

  u_result = ublas::prod( u_lhs, ublas::trans(u_rhs2));
  result = viennacl::linalg::prod( lhs, viennacl::trans(rhs2));

  viennacl::copy( result, temp);
  
  std::cout << "Checking results: " << std::endl;

  size1 = size/2;
  size2 = size/2;
  for (unsigned int i = 0; i < size1; i++)
    for (unsigned int j = 0; j < size2; j++)
      if ( abs(temp(i,j) - u_result(i,j)) > eps ) {
        std::cout << "!!Verification failed at " << i <<" : "<< j  << "(expected: " << u_result(i,j) << " get: " << temp(i,j) << " )" << std::endl;
        return EXIT_FAILURE;
      }

  std::cout << "Everything went well!" << std::endl;

  return EXIT_SUCCESS;
}
