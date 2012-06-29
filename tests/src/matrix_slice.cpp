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

#define VIENNACL_HAVE_UBLAS
//#define NDEBUG
//#define VIENNACL_BUILD_INFO

#include <utility>
#include <iostream>
#include <fstream>
#include <string>
#include <cmath>
#include <algorithm>
#include <stdio.h>
#include <time.h>
//#include "../benchmarks/benchmark-utils.hpp"
#include "viennacl/scalar.hpp"
#include "viennacl/matrix.hpp"
#include "viennacl/linalg/prod.hpp"
/*#include "viennacl/compressed_matrix.hpp"
#include "viennacl/linalg/cg.hpp"
#include "viennacl/linalg/inner_prod.hpp"
#include "viennacl/linalg/ilu.hpp"
#include "viennacl/linalg/norm_2.hpp"
#include "viennacl/io/matrix_market.hpp"*/
#include "viennacl/matrix_proxy.hpp"
#include "viennacl/vector_proxy.hpp"
#include "boost/numeric/ublas/vector.hpp"
#include "boost/numeric/ublas/matrix.hpp"
#include "boost/numeric/ublas/matrix_proxy.hpp"
#include "boost/numeric/ublas/vector_proxy.hpp"
#include "boost/numeric/ublas/io.hpp"

           
template <typename VectorType, typename VCLVectorType>
bool check_for_equality_vector(VectorType const & ublas_v, VCLVectorType const & vcl_v)
{
  typedef typename VectorType::value_type   value_type;
  
  boost::numeric::ublas::vector<value_type> vcl_v_cpu(vcl_v.size());
  viennacl::copy(vcl_v, vcl_v_cpu);
  
  for (std::size_t i=0; i<ublas_v.size(); ++i)
  {
    if (ublas_v(i) != vcl_v_cpu(i))
    {
      if ( std::abs(ublas_v(i) - vcl_v_cpu(i)) / std::max(ublas_v(i), vcl_v_cpu(i)) > 1e-5 ) 
      {
        std::cout << "Error at index (" << i << "): " << ublas_v(i) << " vs " << vcl_v_cpu(i) << std::endl;
        std::cout << ublas_v << std::endl;
        std::cout << vcl_v_cpu << std::endl;
        return false;
      }
    }
  }
  return true;
}


template <typename MatrixType, typename VCLMatrixType>
bool check_for_equality(MatrixType const & ublas_A, VCLMatrixType const & vcl_A)
{
  typedef typename MatrixType::value_type   value_type;
  
  boost::numeric::ublas::matrix<value_type> vcl_A_cpu(vcl_A.size1(), vcl_A.size2());
  viennacl::copy(vcl_A, vcl_A_cpu);
  
  for (std::size_t i=0; i<ublas_A.size1(); ++i)
  {
    for (std::size_t j=0; j<ublas_A.size2(); ++j)
    {
      if (ublas_A(i,j) != vcl_A_cpu(i,j))
      {
        if ( std::abs(ublas_A(i,j) - vcl_A_cpu(i,j)) / std::max(ublas_A(i,j), vcl_A_cpu(i,j)) > 1e-5 ) 
        {
          std::cout << "Error at index (" << i << ", " << j << "): " << ublas_A(i,j) << " vs " << vcl_A_cpu(i,j) << std::endl;
          std::cout << ublas_A << std::endl;
          std::cout << vcl_A_cpu << std::endl;
          return false;
        }
      }
    }
  }
  return true;
}


           
template <typename T, typename ScalarType>
int run_test()
{
    //typedef float               ScalarType;
    typedef boost::numeric::ublas::matrix<ScalarType>       MatrixType;
    typedef boost::numeric::ublas::vector<ScalarType>       VectorType;
    
    typedef viennacl::matrix<ScalarType, T>    VCLMatrixType;
    typedef viennacl::vector<ScalarType>       VCLVectorType;
    
    viennacl::scalar<ScalarType> gpu_pi = ScalarType(3.1415);
    
    //std::size_t dim_large = 196;
    //std::size_t dim_small = 64;
    //std::size_t dim_large = 75;  //Note: ensure dim_large > 2 * dim_small
    //std::size_t dim_small = 34;

    std::size_t dim_large = 75;  //Note: ensure dim_large > 2 * dim_small
    std::size_t dim_small = 34;
    
    //setup ublas objects:
    MatrixType ublas_A(dim_large, dim_large);
    for (std::size_t i=0; i<ublas_A.size1(); ++i)
      for (std::size_t j=0; j<ublas_A.size2(); ++j)
        ublas_A(i,j) = ScalarType((i+1) + (j+1)*(i+1));

    MatrixType ublas_B(dim_small, dim_small);
    for (std::size_t i=0; i<ublas_B.size1(); ++i)
      for (std::size_t j=0; j<ublas_B.size2(); ++j)
        ublas_B(i,j) = ScalarType((i+1) + (j+1)*(i+1));

    MatrixType ublas_C(dim_large, 2 * dim_small);
    for (std::size_t i=0; i<ublas_C.size1(); ++i)
      for (std::size_t j=0; j<ublas_C.size2(); ++j)
        ublas_C(i,j) = ScalarType((j+2) + (j+1)*(i+1));

    MatrixType ublas_D(2 * dim_small, dim_large);
    for (std::size_t i=0; i<ublas_D.size1(); ++i)
      for (std::size_t j=0; j<ublas_D.size2(); ++j)
        ublas_D(i,j) = ScalarType((j+2) + (j+1)*(i+1));
      
    boost::numeric::ublas::slice ublas_s1(0, 2, dim_small);
    boost::numeric::ublas::slice ublas_s2(dim_large - 2 * dim_small, 2, dim_small);
    boost::numeric::ublas::matrix_slice<MatrixType> ublas_A_sub1(ublas_A, ublas_s1, ublas_s1);
    boost::numeric::ublas::matrix_slice<MatrixType> ublas_A_sub2(ublas_A, ublas_s2, ublas_s2);

    boost::numeric::ublas::matrix_slice<MatrixType> ublas_C_sub(ublas_C, ublas_s1, ublas_s1);
    boost::numeric::ublas::matrix_slice<MatrixType> ublas_D_sub(ublas_D, ublas_s1, ublas_s1);

    //Setup ViennaCL objects    
    VCLMatrixType vcl_A(dim_large, dim_large);
    viennacl::copy(ublas_A, vcl_A);
    VCLMatrixType vcl_B(dim_small, dim_small);
    viennacl::copy(ublas_B, vcl_B);
    VCLMatrixType vcl_C(dim_large, 2 * dim_small);
    viennacl::copy(ublas_C, vcl_C);
    VCLMatrixType vcl_D(2 * dim_small, dim_large);
    viennacl::copy(ublas_D, vcl_D);
    
    viennacl::slice vcl_s1(0, 2, dim_small);
    viennacl::slice vcl_s2(dim_large - 2 * dim_small, 2, dim_small);
    viennacl::matrix_slice<VCLMatrixType>   vcl_A_sub1(vcl_A, vcl_s1, vcl_s1);
    viennacl::matrix_slice<VCLMatrixType>   vcl_A_sub2(vcl_A, vcl_s2, vcl_s2);
    
    viennacl::matrix_slice<VCLMatrixType>   vcl_C_sub(vcl_C, vcl_s1, vcl_s1);
    viennacl::matrix_slice<VCLMatrixType>   vcl_D_sub(vcl_D, vcl_s1, vcl_s1);
    
    std::cout << std::endl;
    std::cout << "//" << std::endl;
    std::cout << "////////// Test 1: Copy to GPU //////////" << std::endl;
    std::cout << "//" << std::endl;
    
    std::cout << "Testing upper left copy to A... ";
    ublas_A_sub1 = ublas_B;
    viennacl::copy(ublas_B, vcl_A_sub1);
    if (check_for_equality(ublas_A, vcl_A))
      std::cout << "PASSED!" << std::endl;
    else
    {
      std::cout << std::endl << "TEST failed!";
      return EXIT_FAILURE;
    }
    
    
    std::cout << "Testing lower right copy to A... ";
    ublas_A_sub2 = ublas_B;
    viennacl::copy(ublas_B, vcl_A_sub2);
    if (check_for_equality(ublas_A, vcl_A))
      std::cout << "PASSED!" << std::endl;
    else
    {
      std::cout << std::endl << "TEST failed!";
      return EXIT_FAILURE;
    }
    
    
    
    std::cout << "Testing upper copy to C... ";
    ublas_C_sub = ublas_B;
    viennacl::copy(ublas_B, vcl_C_sub);
    if (check_for_equality(ublas_C, vcl_C))
      std::cout << "PASSED!" << std::endl;
    else
    {
      std::cout << std::endl << "TEST failed!";
      return EXIT_FAILURE;
    }

    
    std::cout << "Testing left copy to D... ";
    ublas_D_sub = ublas_B;
    viennacl::copy(ublas_B, vcl_D_sub);
    if (check_for_equality(ublas_D, vcl_D))
      std::cout << "PASSED!" << std::endl;
    else
      std::cout << std::endl << "TEST failed!";
    
    std::cout << std::endl;
    std::cout << "//" << std::endl;
    std::cout << "////////// Test 2: Copy from GPU //////////" << std::endl;
    std::cout << "//" << std::endl;
    
    std::cout << "Testing upper left copy to A... ";
    if (check_for_equality(ublas_A_sub1, vcl_A_sub1))
      std::cout << "PASSED!" << std::endl;
    else
    {
      std::cout << std::endl << "TEST failed!";
      return EXIT_FAILURE;
    }
    
    std::cout << "Testing lower right copy to A... ";
    if (check_for_equality(ublas_A_sub2, vcl_A_sub2))
      std::cout << "PASSED!" << std::endl;
    else
    {
      std::cout << std::endl << "TEST failed!";
      return EXIT_FAILURE;
    }
    
    std::cout << "Testing upper copy to C... ";
    if (check_for_equality(ublas_C_sub, vcl_C_sub))
      std::cout << "PASSED!" << std::endl;
    else
    {
      std::cout << std::endl << "TEST failed!";
      return EXIT_FAILURE;
    }

    std::cout << "Testing left copy to D... ";
    if (check_for_equality(ublas_D_sub, vcl_D_sub))
      std::cout << "PASSED!" << std::endl;
    else
    {
      std::cout << std::endl << "TEST failed!";
      return EXIT_FAILURE;
    }
    
    
    std::cout << "//" << std::endl;
    std::cout << "////////// Test 3: Addition //////////" << std::endl;
    std::cout << "//" << std::endl;
    viennacl::copy(ublas_A_sub2, vcl_A_sub2);
    
    std::cout << "Inplace add to submatrix: ";
    ublas_A_sub2 += ublas_A_sub2;
    vcl_A_sub2 += vcl_A_sub2;

    if (check_for_equality(ublas_A, vcl_A))
      std::cout << "PASSED!" << std::endl;
    else
    {
      std::cout << std::endl << "TEST failed!";
      return EXIT_FAILURE;
    }

    std::cout << "Inplace add to matrix: ";
    ublas_B += ublas_A_sub2;
    vcl_B += vcl_A_sub2;

    if (check_for_equality(ublas_B, vcl_B))
      std::cout << "PASSED!" << std::endl;
    else
    {
      std::cout << std::endl << "TEST failed!";
      return EXIT_FAILURE;
    }
    
    std::cout << "Add to submatrix: ";
    ublas_A_sub2 = ublas_A_sub2 + ublas_A_sub2;
    vcl_A_sub2 = vcl_A_sub2 + vcl_A_sub2;

    if (check_for_equality(ublas_A, vcl_A))
      std::cout << "PASSED!" << std::endl;
    else
    {
      std::cout << std::endl << "TEST failed!";
      return EXIT_FAILURE;
    }

    std::cout << "Add to matrix: ";
    ublas_B = ublas_A_sub2 + ublas_A_sub2;
    vcl_B = vcl_A_sub2 + vcl_A_sub2;

    if (check_for_equality(ublas_B, vcl_B))
      std::cout << "PASSED!" << std::endl;
    else
    {
      std::cout << std::endl << "TEST failed!";
      return EXIT_FAILURE;
    }
    
    std::cout << "//" << std::endl;
    std::cout << "////////// Test 4: Subtraction //////////" << std::endl;
    std::cout << "//" << std::endl;
    viennacl::copy(ublas_A_sub2, vcl_A_sub2);
    
    std::cout << "Inplace add to submatrix: ";
    ublas_A_sub2 -= ublas_A_sub2;
    vcl_A_sub2 -= vcl_A_sub2;

    if (check_for_equality(ublas_A, vcl_A))
      std::cout << "PASSED!" << std::endl;
    else
    {
      std::cout << std::endl << "TEST failed!";
      return EXIT_FAILURE;
    }

    std::cout << "Inplace add to matrix: ";
    ublas_B -= ublas_A_sub2;
    vcl_B -= vcl_A_sub2;

    if (check_for_equality(ublas_B, vcl_B))
      std::cout << "PASSED!" << std::endl;
    else
    {
      std::cout << std::endl << "TEST failed!";
      return EXIT_FAILURE;
    }
    
    std::cout << "Add to submatrix: ";
    ublas_A_sub2 = ublas_A_sub2 - ublas_A_sub2;
    vcl_A_sub2 = vcl_A_sub2 - vcl_A_sub2;

    if (check_for_equality(ublas_A, vcl_A))
      std::cout << "PASSED!" << std::endl;
    else
    {
      std::cout << std::endl << "TEST failed!";
      return EXIT_FAILURE;
    }

    std::cout << "Add to matrix: ";
    ublas_B = ublas_A_sub2 - ublas_A_sub2;
    vcl_B = vcl_A_sub2 - vcl_A_sub2;

    if (check_for_equality(ublas_B, vcl_B))
      std::cout << "PASSED!" << std::endl;
    else
    {
      std::cout << std::endl << "TEST failed!";
      return EXIT_FAILURE;
    }
    
    std::cout << "//" << std::endl;
    std::cout << "////////// Test 5: Scaling //////////" << std::endl;
    std::cout << "//" << std::endl;
    viennacl::copy(ublas_A, vcl_A);
    
    std::cout << "Multiplication with CPU scalar: ";
    ublas_A_sub2 *= ScalarType(3.1415);
    vcl_A_sub2 *= ScalarType(3.1415);

    if (check_for_equality(ublas_A, vcl_A))
      std::cout << "PASSED!" << std::endl;
    else
    {
      std::cout << std::endl << "TEST failed!";
      return EXIT_FAILURE;
    }

    std::cout << "Multiplication with GPU scalar: ";
    ublas_A_sub2 *= gpu_pi;
    vcl_A_sub2 *= gpu_pi;

    if (check_for_equality(ublas_A, vcl_A))
      std::cout << "PASSED!" << std::endl;
    else
    {
      std::cout << std::endl << "TEST failed!";
      return EXIT_FAILURE;
    }
    
    
    std::cout << "Division with CPU scalar: ";
    ublas_A_sub2 /= ScalarType(3.1415);
    vcl_A_sub2 /= ScalarType(3.1415);

    if (check_for_equality(ublas_A, vcl_A))
      std::cout << "PASSED!" << std::endl;
    else
    {
      std::cout << std::endl << "TEST failed!";
      return EXIT_FAILURE;
    }

    std::cout << "Division with GPU scalar: ";
    ublas_A_sub2 /= gpu_pi;
    vcl_A_sub2 /= gpu_pi;

    if (check_for_equality(ublas_A, vcl_A))
      std::cout << "PASSED!" << std::endl;
    else
    {
      std::cout << std::endl << "TEST failed!";
      return EXIT_FAILURE;
    }
    
    

    std::cout << "//" << std::endl;
    std::cout << "////////// Test 6: Matrix-Matrix Products //////////" << std::endl;
    std::cout << "//" << std::endl;
    viennacl::copy(ublas_A, vcl_A);
    viennacl::copy(ublas_B, vcl_B);
    viennacl::copy(ublas_C, vcl_C);

    std::cout << "Assigned C = A * B: ";
    ublas_A_sub1 = prod(ublas_C_sub, ublas_D_sub);
    vcl_A_sub1 = viennacl::linalg::prod(vcl_C_sub, vcl_D_sub);

    if (check_for_equality(ublas_A, vcl_A))
      std::cout << "PASSED!" << std::endl;
    else
    {
      std::cout << std::endl << "TEST failed!";
      return EXIT_FAILURE;
    }

    std::cout << "Assigned C = A^T * B: ";
    ublas_A_sub1 = prod(trans(ublas_C_sub), ublas_D_sub);
    vcl_A_sub1 = viennacl::linalg::prod(trans(vcl_C_sub), vcl_D_sub);

    if (check_for_equality(ublas_A, vcl_A))
      std::cout << "PASSED!" << std::endl;
    else
    {
      std::cout << std::endl << "TEST failed!";
      return EXIT_FAILURE;
    }

    std::cout << "Assigned C = A * B^T: ";
    ublas_A_sub1 = prod(ublas_C_sub, trans(ublas_D_sub));
    vcl_A_sub1 = viennacl::linalg::prod(vcl_C_sub, trans(vcl_D_sub));

    if (check_for_equality(ublas_A, vcl_A))
      std::cout << "PASSED!" << std::endl;
    else
    {
      std::cout << std::endl << "TEST failed!";
      return EXIT_FAILURE;
    }

    std::cout << "Assigned C = A^T * B^T: ";
    ublas_A_sub1 = prod(trans(ublas_C_sub), trans(ublas_D_sub));
    vcl_A_sub1 = viennacl::linalg::prod(trans(vcl_C_sub), trans(vcl_D_sub));

    if (check_for_equality(ublas_A, vcl_A))
      std::cout << "PASSED!" << std::endl;
    else
    {
      std::cout << std::endl << "TEST failed!";
      return EXIT_FAILURE;
    }

    std::cout << "Inplace add of prod(): ";
    ublas_A_sub1 += prod(ublas_C_sub, ublas_D_sub);
    vcl_A_sub1 += viennacl::linalg::prod(vcl_C_sub, vcl_D_sub);

    if (check_for_equality(ublas_A, vcl_A))
      std::cout << "PASSED!" << std::endl;
    else
    {
      std::cout << std::endl << "TEST failed!";
      return EXIT_FAILURE;
    }


    std::cout << "//" << std::endl;
    std::cout << "////////// Test 7: Matrix-Vector Products //////////" << std::endl;
    std::cout << "//" << std::endl;

    VectorType ublas_v1(dim_large);
    for (std::size_t i=0; i<ublas_v1.size(); ++i)
      ublas_v1(i) = static_cast<ScalarType>(i);
    boost::numeric::ublas::vector_slice<VectorType> ublas_v1_sub(ublas_v1, ublas_s1);

    VectorType ublas_v2(dim_large);
    for (std::size_t i=0; i<ublas_v2.size(); ++i)
      ublas_v2(i) = static_cast<ScalarType>(i) - static_cast<ScalarType>(5);
    boost::numeric::ublas::vector_slice<VectorType> ublas_v2_sub(ublas_v2, ublas_s1);

    
    VCLVectorType vcl_v1(ublas_v1.size());
    viennacl::vector_slice<VCLVectorType> vcl_v1_sub(vcl_v1, vcl_s1);
    VCLVectorType vcl_v2(ublas_v2.size());
    viennacl::vector_slice<VCLVectorType> vcl_v2_sub(vcl_v2, vcl_s1);
    viennacl::copy(ublas_v1, vcl_v1);
    viennacl::copy(ublas_v2, vcl_v2);
    viennacl::copy(ublas_A_sub1, vcl_A_sub1);
    
    
    ublas_v2_sub = prod(ublas_A_sub1, ublas_v1_sub);
    vcl_v2_sub = viennacl::linalg::prod(vcl_A_sub1, vcl_v1_sub);

    if (check_for_equality_vector(ublas_v2, vcl_v2))
      std::cout << "PASSED!" << std::endl;
    else
    {
      std::cout << std::endl << "TEST failed!";
      return EXIT_FAILURE;
    }


    return EXIT_SUCCESS;
}    

int main (int argc, const char * argv[])
{
  std::cout << std::endl;
  std::cout << "----------------------------------------------" << std::endl;
  std::cout << "----------------------------------------------" << std::endl;
  std::cout << "## Test :: Matrix Slice" << std::endl;
  std::cout << "----------------------------------------------" << std::endl;
  std::cout << "----------------------------------------------" << std::endl;
  std::cout << std::endl;
   
  std::cout << "# Testing setup:" << std::endl;
  std::cout << "  eps:     " << 0 << std::endl;
  std::cout << "  numeric: float" << std::endl;
  if (run_test<viennacl::row_major, float>() != EXIT_SUCCESS)
    return EXIT_FAILURE;
  if (run_test<viennacl::column_major, float>() != EXIT_SUCCESS)
    return EXIT_FAILURE;
  
  
  if( viennacl::ocl::current_device().double_support() )
  {
    std::cout << "# Testing setup:" << std::endl;
    std::cout << "  eps:     " << 0 << std::endl;
    std::cout << "  numeric: double" << std::endl;
    
    if (run_test<viennacl::row_major, double>() != EXIT_SUCCESS)
      return EXIT_FAILURE;
    if (run_test<viennacl::column_major, double>() != EXIT_SUCCESS)
      return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}

