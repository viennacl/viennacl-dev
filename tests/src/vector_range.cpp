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
#include "viennacl/linalg/norm_1.hpp"
#include "viennacl/linalg/norm_2.hpp"
#include "viennacl/linalg/norm_inf.hpp"
#include "viennacl/linalg/inner_prod.hpp"
/*#include "viennacl/compressed_matrix.hpp"
#include "viennacl/linalg/cg.hpp"
#include "viennacl/linalg/inner_prod.hpp"
#include "viennacl/linalg/ilu.hpp"
#include "viennacl/linalg/norm_2.hpp"
#include "viennacl/io/matrix_market.hpp"*/
#include "viennacl/vector_proxy.hpp"
#include "boost/numeric/ublas/vector.hpp"
#include "boost/numeric/ublas/matrix.hpp"
#include "boost/numeric/ublas/vector_proxy.hpp"
#include "boost/numeric/ublas/io.hpp"


template <typename VectorType, typename VCLVectorType>
bool check_for_equality(VectorType const & ublas_v, VCLVectorType const & vcl_v)
{
  typedef typename VectorType::value_type   value_type;
  
  std::vector<value_type> vcl_v_cpu(vcl_v.size());
  viennacl::copy(vcl_v, vcl_v_cpu);

  bool error_detected = false;
  for (size_t i=0; i<ublas_v.size(); ++i)
  {
    if (ublas_v[i] != vcl_v_cpu[i])
    {
      //check whether there are just some round-off errors:
      if (std::abs(ublas_v[i] - vcl_v_cpu[i]) / std::max(ublas_v[i], vcl_v_cpu[i]) > 1e-5)
      {
        std::cout << "Error at index (" << i << "): " << ublas_v[i] << " vs " << vcl_v_cpu[i] << std::endl;
        error_detected = true;
      }
    }
  }
  
  if (!error_detected)
    std::cout << "PASSED!" << std::endl;
  else
  {
    std::cout << std::endl << "TEST failed!" << std::endl;
    exit(EXIT_FAILURE);
  }
  
  return true;
}


           
template <typename ScalarType>
int run_test()
{
    typedef boost::numeric::ublas::vector<ScalarType>       VectorType;
    
    typedef viennacl::vector<ScalarType>                    VCLVectorType;
    
    std::size_t dim_large = 70;
    std::size_t dim_small = 27;
    
    //setup ublas objects:
    VectorType ublas_v1(dim_large);
    for (std::size_t i=0; i<ublas_v1.size(); ++i)
      ublas_v1(i) = static_cast<ScalarType>(i+1);

    VectorType ublas_v2(dim_small);
    for (std::size_t i=0; i<ublas_v2.size(); ++i)
      ublas_v2(i) = static_cast<ScalarType>(dim_large + i);
      
    boost::numeric::ublas::range ublas_r1(0, dim_small);
    boost::numeric::ublas::range ublas_r2(dim_small - 1, 2*dim_small - 1);
    boost::numeric::ublas::range ublas_r3(dim_large - dim_small, dim_large);
    boost::numeric::ublas::vector_range<VectorType> ublas_v1_sub1(ublas_v1, ublas_r1);
    boost::numeric::ublas::vector_range<VectorType> ublas_v1_sub2(ublas_v1, ublas_r2);
    boost::numeric::ublas::vector_range<VectorType> ublas_v1_sub3(ublas_v1, ublas_r3);

    //Setup ViennaCL objects    
    VCLVectorType vcl_v1(dim_large);
    viennacl::copy(ublas_v1, vcl_v1);
    VCLVectorType vcl_v2(dim_small);
    viennacl::copy(ublas_v2, vcl_v2);
    
    viennacl::range vcl_r1(0, dim_small);
    viennacl::range vcl_r2(dim_small - 1, 2*dim_small - 1);
    viennacl::range vcl_r3(dim_large - dim_small, dim_large);
    viennacl::vector_range<VCLVectorType>   vcl_v1_sub1(vcl_v1, vcl_r1);
    viennacl::vector_range<VCLVectorType>   vcl_v1_sub2(vcl_v1, vcl_r2);
    viennacl::vector_range<VCLVectorType>   vcl_v1_sub3(vcl_v1, vcl_r3);
    
    std::cout << std::endl;
    std::cout << "//" << std::endl;
    std::cout << "////////// Test: Copy to GPU //////////" << std::endl;
    std::cout << "//" << std::endl;
    
    ublas_v1_sub1 = ublas_v2;
    viennacl::copy(ublas_v2, vcl_v1_sub1);
    std::cout << "Testing copy to begin of v1... ";
    check_for_equality(ublas_v1, vcl_v1);
    
    
    ublas_v1_sub2 = ublas_v2;
    viennacl::copy(ublas_v2, vcl_v1_sub2);
    std::cout << "Testing copy to middle of v1... ";
    check_for_equality(ublas_v1, vcl_v1);
    
    
    
    ublas_v1_sub3 = ublas_v2;
    viennacl::copy(ublas_v2, vcl_v1_sub3);
    std::cout << "Testing copy to bottom of v1... ";
    check_for_equality(ublas_v1, vcl_v1);

    
    std::cout << std::endl;
    std::cout << "//" << std::endl;
    std::cout << "////////// Test: Copy from GPU //////////" << std::endl;
    std::cout << "//" << std::endl;
    
    std::cout << "Testing beginning of v1... ";
    check_for_equality(ublas_v1_sub1, vcl_v1_sub1);
    
    std::cout << "Testing middle of v1... ";
    check_for_equality(ublas_v1_sub2, vcl_v1_sub2);
    
    std::cout << "Testing bottom of v1... ";
    check_for_equality(ublas_v1_sub3, vcl_v1_sub3);


    
    std::cout << std::endl;
    std::cout << "//" << std::endl;
    std::cout << "////////// Test: Assignments //////////" << std::endl;
    std::cout << "//" << std::endl;

    viennacl::copy(ublas_v1, vcl_v1);
    viennacl::copy(ublas_v2, vcl_v2);

    std::cout << "Testing vector assigned to range... ";
    ublas_v1_sub1 = ublas_v2;
    vcl_v1_sub1 = vcl_v2;
    check_for_equality(ublas_v1, vcl_v1);
    
    std::cout << "Testing range assigned to vector... ";
    ublas_v2 = ublas_v1_sub1;
    vcl_v2 = vcl_v1_sub1;
    check_for_equality(ublas_v1, vcl_v1);
    
    std::cout << "Testing range assigned to range... ";
    ublas_v1_sub1 = ublas_v1_sub3;
    vcl_v1_sub1 = vcl_v1_sub3;
    check_for_equality(ublas_v1, vcl_v1);
    
    
    std::cout << std::endl;
    std::cout << "//" << std::endl;
    std::cout << "////////// Test: Copy CTORs //////////" << std::endl;
    std::cout << "//" << std::endl;

    viennacl::copy(ublas_v1, vcl_v1);
    viennacl::copy(ublas_v2, vcl_v2);

    {
      std::cout << "Testing vector created from range... ";
      ublas_v2 = ublas_v1_sub2;
      VCLVectorType vcl_ctor_1 = vcl_v1_sub2;
      check_for_equality(ublas_v2, vcl_ctor_1);
      viennacl::copy(ublas_v2, vcl_v2);
    }
    
    
    std::cout << "//" << std::endl;
    std::cout << "////////// Test: Inplace add //////////" << std::endl;
    std::cout << "//" << std::endl;
    viennacl::copy(ublas_v1_sub1, vcl_v1_sub1);
    
    std::cout << "Testing inplace add at beginning of v1: ";
    ublas_v1_sub1 += ublas_v1_sub1;
    vcl_v1_sub1 += vcl_v1_sub1;
    check_for_equality(ublas_v1, vcl_v1);

    std::cout << "Testing inplace add at middle of v1: ";
    ublas_v1_sub2 += ublas_v1_sub2;
    vcl_v1_sub2 += vcl_v1_sub2;
    check_for_equality(ublas_v1, vcl_v1);

    std::cout << "Testing inplace add at end of v1: ";
    ublas_v1_sub3 += ublas_v1_sub3;
    vcl_v1_sub3 += vcl_v1_sub3;
    check_for_equality(ublas_v1, vcl_v1);
    
    std::cout << "Testing inplace add of range with vector ";
    ublas_v2 += ublas_v1_sub3;
    vcl_v2 += vcl_v1_sub3;
    check_for_equality(ublas_v2, vcl_v2);

    std::cout << "Testing inplace add of vector with range: ";
    viennacl::copy(ublas_v2, vcl_v2);
    ublas_v1_sub2 += ublas_v2;
    vcl_v1_sub2 += vcl_v2;
    
    check_for_equality(ublas_v1, vcl_v1);
    
    
    std::cout << "//" << std::endl;
    std::cout << "////////// Test: Inplace sub //////////" << std::endl;
    std::cout << "//" << std::endl;
    viennacl::copy(ublas_v1_sub1, vcl_v1_sub1);
    
    std::cout << "Testing inplace sub at beginning of v1: ";
    ublas_v1_sub1 -= ublas_v1_sub1;
    vcl_v1_sub1 -= vcl_v1_sub1;
    check_for_equality(ublas_v1, vcl_v1);
    
    std::cout << "Testing inplace sub at middle of v1: ";
    ublas_v1_sub2 -= ublas_v1_sub2;
    vcl_v1_sub2 -= vcl_v1_sub2;
    check_for_equality(ublas_v1, vcl_v1);

    std::cout << "Testing inplace sub at end of v1: ";
    ublas_v1_sub3 -= ublas_v1_sub3;
    vcl_v1_sub3 -= vcl_v1_sub3;
    check_for_equality(ublas_v1, vcl_v1);

    std::cout << "Testing inplace sub of range with vector ";
    ublas_v2 -= ublas_v1_sub3;
    vcl_v2 -= vcl_v1_sub3;
    check_for_equality(ublas_v2, vcl_v2);
    
    std::cout << "Testing inplace sub of vector with range: ";
    viennacl::copy(ublas_v2, vcl_v2);
    ublas_v1_sub2 -= ublas_v2;
    vcl_v1_sub2 -= vcl_v2;
    
    check_for_equality(ublas_v1, vcl_v1);

    
    std::cout << "//" << std::endl;
    std::cout << "////////// Test: Inplace mult/div //////////" << std::endl;
    std::cout << "//" << std::endl;
    viennacl::copy(ublas_v1, vcl_v1);
    viennacl::copy(ublas_v2, vcl_v2);
    ScalarType s = 3.14;
    viennacl::scalar<ScalarType>  vcl_s = s;

    std::cout << "Multiplication with CPU scalar: ";
    ublas_v1_sub1 *= s;
    vcl_v1_sub1   *= s;
    
    check_for_equality(ublas_v1, vcl_v1);
    
    std::cout << "Multiplication with GPU scalar: ";
    ublas_v1_sub3 *= vcl_s;
    vcl_v1_sub3   *= vcl_s;
    
    check_for_equality(ublas_v1, vcl_v1);
    

    std::cout << "Division with CPU scalar: ";
    ublas_v1_sub1 /= s;
    vcl_v1_sub1   /= s;
    
    check_for_equality(ublas_v1, vcl_v1);
    
    std::cout << "Division with GPU scalar: ";
    ublas_v1_sub3 /= vcl_s;
    vcl_v1_sub3   /= vcl_s;
    
    check_for_equality(ublas_v1, vcl_v1);
    

    
    
    std::cout << "//" << std::endl;
    std::cout << "////////// Test: Vector Operations (norm_X, inner_prod, etc.) //////////" << std::endl;
    std::cout << "//" << std::endl;
    
    for (std::size_t i=0; i<ublas_v1.size(); ++i) //reinit values
      ublas_v1(i) = static_cast<ScalarType>(i+1);
    
    viennacl::copy(ublas_v1_sub1, vcl_v1_sub1);
    viennacl::copy(ublas_v1_sub2, vcl_v1_sub2);
    viennacl::copy(ublas_v1_sub3, vcl_v1_sub3);
    
    double result_ublas = 0;
    double result_viennacl = 0;

    std::cout << "Testing norm_1: ";
    result_ublas = norm_1(ublas_v1_sub2);
    result_viennacl = viennacl::linalg::norm_1(vcl_v1_sub2);
    
    if (std::abs(result_ublas - result_viennacl) / std::abs(result_ublas) < 1e-3)
      std::cout << "PASSED!" << std::endl;
    else
    {
      std::cout << std::endl << "TEST failed!";
      std::cout << "Ublas: "    << result_ublas << std::endl;
      std::cout << "ViennaCL: " << result_viennacl << std::endl;
      return EXIT_FAILURE;
    }

    std::cout << "Testing norm_2: ";
    result_ublas = norm_2(ublas_v1_sub2);
    result_viennacl = viennacl::linalg::norm_2(vcl_v1_sub2);
    
    if (std::abs(result_ublas - result_viennacl) / std::abs(result_ublas) < 1e-3)
      std::cout << "PASSED!" << std::endl;
    else
    {
      std::cout << std::endl << "TEST failed!";
      std::cout << "Ublas: "    << result_ublas << std::endl;
      std::cout << "ViennaCL: " << result_viennacl << std::endl;
      return EXIT_FAILURE;
    }

    std::cout << "Testing norm_inf: ";
    result_ublas = norm_inf(ublas_v1_sub2);
    result_viennacl = viennacl::linalg::norm_inf(vcl_v1_sub2);
    
    if (std::abs(result_ublas - result_viennacl) / std::abs(result_ublas) < 1e-3)
      std::cout << "PASSED!" << std::endl;
    else
    {
      std::cout << std::endl << "TEST failed!";
      std::cout << "Ublas: "    << result_ublas << std::endl;
      std::cout << "ViennaCL: " << result_viennacl << std::endl;
      return EXIT_FAILURE;
    }
    
    std::cout << "Testing inner_prod: ";
    result_ublas = inner_prod(ublas_v1_sub1, ublas_v1_sub3);
    result_viennacl = viennacl::linalg::inner_prod(vcl_v1_sub1, vcl_v1_sub3);
    
    if (std::abs(result_ublas - result_viennacl) / std::abs(result_ublas) < 1e-3)
      std::cout << "PASSED!" << std::endl;
    else
    {
      std::cout << std::endl << "TEST failed!";
      std::cout << "Ublas: "    << result_ublas << std::endl;
      std::cout << "ViennaCL: " << result_viennacl << std::endl;
      return EXIT_FAILURE;
    }
    
    std::cout << std::endl;
    std::cout << "----------------------------------------------" << std::endl;
    std::cout << std::endl;

    return EXIT_SUCCESS;
}    

int main (int argc, const char * argv[])
{
  std::cout << std::endl;
  std::cout << "----------------------------------------------" << std::endl;
  std::cout << "----------------------------------------------" << std::endl;
  std::cout << "## Test :: Vector Range" << std::endl;
  std::cout << "----------------------------------------------" << std::endl;
  std::cout << "----------------------------------------------" << std::endl;
  std::cout << std::endl;
   
  std::cout << "# Testing setup:" << std::endl;
  //std::cout << "  eps:     " << 0 << std::endl;
  std::cout << "  numeric: float" << std::endl;
  if (run_test<float>() != EXIT_SUCCESS)
    return EXIT_FAILURE;
  
  if( viennacl::ocl::current_device().double_support() )
  {
    std::cout << "# Testing setup:" << std::endl;
    //std::cout << "  eps:     " << 0 << std::endl;
    std::cout << "  numeric: double" << std::endl;
    
    if (run_test<double>() != EXIT_SUCCESS)
      return EXIT_FAILURE;
  }
  
  std::cout << std::endl;
  std::cout << "------- Test completed --------" << std::endl;
  std::cout << std::endl;
   
  
  return EXIT_SUCCESS;
}

