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



/** \file tests/src/vector_uint.cpp  Tests vector operations (BLAS level 1) for unsigned integer arithmetic.
*   \test  Tests vector operations (BLAS level 1) for unsigned integer arithmetic.
**/

//
// *** System
//
#include <iostream>
#include <iomanip>
#include <vector>

//
// *** ViennaCL
//
#include "viennacl/vector.hpp"
#include "viennacl/vector_proxy.hpp"
#include "viennacl/linalg/inner_prod.hpp"
#include "viennacl/linalg/norm_1.hpp"
#include "viennacl/linalg/norm_2.hpp"
#include "viennacl/linalg/norm_inf.hpp"
#include "viennacl/linalg/maxmin.hpp"
#include "viennacl/linalg/sum.hpp"


//
// -------------------------------------------------------------
//
template<typename ScalarType>
ScalarType diff(ScalarType const & s1, ScalarType const & s2)
{
  viennacl::backend::finish();
  return s1 - s2;
}
//
// -------------------------------------------------------------
//
template<typename ScalarType>
ScalarType diff(ScalarType const & s1, viennacl::scalar<ScalarType> const & s2)
{
  viennacl::backend::finish();
  return s1 - s2;
}
//
// -------------------------------------------------------------
//
template<typename ScalarType>
ScalarType diff(ScalarType const & s1, viennacl::entry_proxy<ScalarType> const & s2)
{
  viennacl::backend::finish();
  return s1 - s2;
}
//
// -------------------------------------------------------------
//
template<typename ScalarType, typename VCLVectorType>
ScalarType diff(std::vector<ScalarType> const & v1, VCLVectorType const & v2)
{
   std::vector<ScalarType> v2_cpu(v2.size());
   viennacl::backend::finish();  //workaround for a bug in APP SDK 2.7 on Trinity APUs (with Catalyst 12.8)
   viennacl::copy(v2.begin(), v2.end(), v2_cpu.begin());

   for (unsigned int i=0;i<v1.size(); ++i)
   {
      if (v2_cpu[i] != v1[i])
        return 1;
   }

   return 0;
}


template<typename T1, typename T2>
int check(T1 const & t1, T2 const & t2)
{
  int retval = EXIT_SUCCESS;

  if (diff(t1, t2) != 0)
  {
    std::cout << "# Error! Difference: " << diff(t1, t2) << std::endl;
    retval = EXIT_FAILURE;
  }
  return retval;
}


//
// -------------------------------------------------------------
//
template< typename NumericT, typename STLVectorType, typename ViennaCLVectorType1, typename ViennaCLVectorType2 >
int test(STLVectorType       & std_v1, STLVectorType       & std_v2,
         ViennaCLVectorType1 & vcl_v1, ViennaCLVectorType2 & vcl_v2)
{
  int retval = EXIT_SUCCESS;

  NumericT                    cpu_result = 42;
  viennacl::scalar<NumericT>  gpu_result = 43;

  //
  // Initializer:
  //
  std::cout << "Checking for zero_vector initializer..." << std::endl;
  for (std::size_t i=0; i<std_v1.size(); ++i)
    std_v1[i] = 0;
  vcl_v1 = viennacl::zero_vector<NumericT>(vcl_v1.size());
  if (check(std_v1, vcl_v1) != EXIT_SUCCESS)
    return EXIT_FAILURE;

  std::cout << "Checking for scalar_vector initializer..." << std::endl;
  for (std::size_t i=0; i<std_v1.size(); ++i)
    std_v1[i] = cpu_result;
  vcl_v1 = viennacl::scalar_vector<NumericT>(vcl_v1.size(), cpu_result);
  if (check(std_v1, vcl_v1) != EXIT_SUCCESS)
    return EXIT_FAILURE;

  for (std::size_t i=0; i<std_v1.size(); ++i)
    std_v1[i] = cpu_result + 1;
  vcl_v1 = viennacl::scalar_vector<NumericT>(vcl_v1.size(), gpu_result);
  if (check(std_v1, vcl_v1) != EXIT_SUCCESS)
    return EXIT_FAILURE;

  std::cout << "Checking for unit_vector initializer..." << std::endl;
  for (std::size_t i=0; i<std_v1.size(); ++i)
    std_v1[i] = (i == 5) ? 1 : 0;
  vcl_v1 = viennacl::unit_vector<NumericT>(vcl_v1.size(), 5);
  if (check(std_v1, vcl_v1) != EXIT_SUCCESS)
    return EXIT_FAILURE;

  for (std::size_t i=0; i<std_v1.size(); ++i)
  {
    std_v1[i] = NumericT(i);
    std_v2[i] = NumericT(i+42);
  }

  viennacl::copy(std_v1.begin(), std_v1.end(), vcl_v1.begin());  //resync
  viennacl::copy(std_v2.begin(), std_v2.end(), vcl_v2.begin());

  std::cout << "Checking for successful copy..." << std::endl;
  if (check(std_v1, vcl_v1) != EXIT_SUCCESS)
    return EXIT_FAILURE;
  if (check(std_v2, vcl_v2) != EXIT_SUCCESS)
    return EXIT_FAILURE;

  //
  // Part 1: Norms and inner product
  //

  // --------------------------------------------------------------------------
  std::cout << "Testing inner_prod..." << std::endl;
  cpu_result = 0;
  for (std::size_t i=0; i<std_v1.size(); ++i)
    cpu_result += std_v1[i] * std_v2[i];
  NumericT cpu_result2 = viennacl::linalg::inner_prod(vcl_v1, vcl_v2);
  gpu_result = viennacl::linalg::inner_prod(vcl_v1, vcl_v2);

  if (check(cpu_result, cpu_result2) != EXIT_SUCCESS)
    return EXIT_FAILURE;
  if (check(cpu_result, gpu_result) != EXIT_SUCCESS)
    return EXIT_FAILURE;

  cpu_result = 0;
  for (std::size_t i=0; i<std_v1.size(); ++i)
    cpu_result += (std_v1[i] + std_v2[i]) * (2*std_v2[i]);
  NumericT cpu_result3 = viennacl::linalg::inner_prod(vcl_v1 + vcl_v2, 2*vcl_v2);
  gpu_result = viennacl::linalg::inner_prod(vcl_v1 + vcl_v2, 2*vcl_v2);

  if (check(cpu_result, cpu_result3) != EXIT_SUCCESS)
    return EXIT_FAILURE;
  if (check(cpu_result, gpu_result) != EXIT_SUCCESS)
    return EXIT_FAILURE;

  // --------------------------------------------------------------------------
  std::cout << "Testing norm_1..." << std::endl;
  cpu_result = 0;
  for (std::size_t i=0; i<std_v1.size(); ++i)   //note: norm_1 broken for unsigned ints on MacOS
    cpu_result += std_v1[i];
  gpu_result = viennacl::linalg::norm_1(vcl_v1);

  if (check(cpu_result, gpu_result) != EXIT_SUCCESS)
    return EXIT_FAILURE;

  cpu_result2 = 0; //reset
  for (std::size_t i=0; i<std_v1.size(); ++i)   //note: norm_1 broken for unsigned ints on MacOS
    cpu_result2 += std_v1[i];
  cpu_result = viennacl::linalg::norm_1(vcl_v1);

  if (check(cpu_result, cpu_result2) != EXIT_SUCCESS)
    return EXIT_FAILURE;

  cpu_result2 = 0;
  for (std::size_t i=0; i<std_v1.size(); ++i)   //note: norm_1 broken for unsigned ints on MacOS
    cpu_result2 += std_v1[i] + std_v2[i];
  cpu_result = viennacl::linalg::norm_1(vcl_v1 + vcl_v2);

  if (check(cpu_result, cpu_result2) != EXIT_SUCCESS)
    return EXIT_FAILURE;

  // --------------------------------------------------------------------------
  std::cout << "Testing norm_inf..." << std::endl;
  cpu_result = 0;
  for (std::size_t i=0; i<std_v1.size(); ++i)
    if (std_v1[i] > cpu_result)
      cpu_result = std_v1[i];
  gpu_result = viennacl::linalg::norm_inf(vcl_v1);

  if (check(cpu_result, gpu_result) != EXIT_SUCCESS)
    return EXIT_FAILURE;

  cpu_result2 = 0;
  for (std::size_t i=0; i<std_v1.size(); ++i)
    if (std_v1[i] > cpu_result2)
      cpu_result2 = std_v1[i];
  cpu_result = viennacl::linalg::norm_inf(vcl_v1);

  if (check(cpu_result, cpu_result2) != EXIT_SUCCESS)
    return EXIT_FAILURE;

  cpu_result2 = 0;
  for (std::size_t i=0; i<std_v1.size(); ++i)
    if (std_v1[i] + std_v2[i] > cpu_result2)
      cpu_result2 = std_v1[i] + std_v2[i];
  cpu_result = viennacl::linalg::norm_inf(vcl_v1 + vcl_v2);

  if (check(cpu_result, cpu_result2) != EXIT_SUCCESS)
    return EXIT_FAILURE;

  // --------------------------------------------------------------------------
  std::cout << "Testing index_norm_inf..." << std::endl;

  std::size_t cpu_index = 0;
  cpu_result = 0;
  for (std::size_t i=0; i<std_v1.size(); ++i)
    if (std_v1[i] > cpu_result)
    {
      cpu_result = std_v1[i];
      cpu_index = i;
    }
  std::size_t gpu_index = viennacl::linalg::index_norm_inf(vcl_v1);

  if (check(static_cast<NumericT>(cpu_index), static_cast<NumericT>(gpu_index)) != EXIT_SUCCESS)
    return EXIT_FAILURE;
  // --------------------------------------------------------------------------
  gpu_result = vcl_v1[viennacl::linalg::index_norm_inf(vcl_v1)];

  if (check(cpu_result, gpu_result) != EXIT_SUCCESS)
    return EXIT_FAILURE;

  cpu_index = 0;
  cpu_result = 0;
  for (std::size_t i=0; i<std_v1.size(); ++i)
    if (std_v1[i] + std_v2[i] > cpu_result)
    {
      cpu_result = std_v1[i];
      cpu_index = i;
    }
  gpu_result = vcl_v1[viennacl::linalg::index_norm_inf(vcl_v1 + vcl_v2)];

  if (check(cpu_result, gpu_result) != EXIT_SUCCESS)
    return EXIT_FAILURE;

  // --------------------------------------------------------------------------
  std::cout << "Testing max..." << std::endl;
  cpu_result = std_v1[0];
  for (std::size_t i=0; i<std_v1.size(); ++i)
    cpu_result = std::max<NumericT>(cpu_result, std_v1[i]);
  gpu_result = viennacl::linalg::max(vcl_v1);

  if (check(cpu_result, gpu_result) != EXIT_SUCCESS)
    return EXIT_FAILURE;

  cpu_result = std_v1[0];
  for (std::size_t i=0; i<std_v1.size(); ++i)
    cpu_result = std::max<NumericT>(cpu_result, std_v1[i]);
  gpu_result = cpu_result;
  cpu_result *= 2; //reset
  cpu_result = viennacl::linalg::max(vcl_v1);

  if (check(cpu_result, gpu_result) != EXIT_SUCCESS)
    return EXIT_FAILURE;

  cpu_result = std_v1[0] + std_v2[0];
  for (std::size_t i=0; i<std_v1.size(); ++i)
    cpu_result = std::max<NumericT>(cpu_result, std_v1[i] + std_v2[i]);
  gpu_result = cpu_result;
  cpu_result *= 2; //reset
  cpu_result = viennacl::linalg::max(vcl_v1 + vcl_v2);

  if (check(cpu_result, gpu_result) != EXIT_SUCCESS)
    return EXIT_FAILURE;


  // --------------------------------------------------------------------------
  std::cout << "Testing min..." << std::endl;
  cpu_result = std_v1[0];
  for (std::size_t i=0; i<std_v1.size(); ++i)
    cpu_result = std::min<NumericT>(cpu_result, std_v1[i]);
  gpu_result = viennacl::linalg::min(vcl_v1);

  if (check(cpu_result, gpu_result) != EXIT_SUCCESS)
    return EXIT_FAILURE;

  cpu_result = std_v1[0];
  for (std::size_t i=0; i<std_v1.size(); ++i)
    cpu_result = std::min<NumericT>(cpu_result, std_v1[i]);
  gpu_result = cpu_result;
  cpu_result *= 2; //reset
  cpu_result = viennacl::linalg::min(vcl_v1);

  if (check(cpu_result, gpu_result) != EXIT_SUCCESS)
    return EXIT_FAILURE;

  cpu_result = std_v1[0] + std_v2[0];
  for (std::size_t i=0; i<std_v1.size(); ++i)
    cpu_result = std::min<NumericT>(cpu_result, std_v1[i] + std_v2[i]);
  gpu_result = cpu_result;
  cpu_result *= 2; //reset
  cpu_result = viennacl::linalg::min(vcl_v1 + vcl_v2);

  if (check(cpu_result, gpu_result) != EXIT_SUCCESS)
    return EXIT_FAILURE;

  // --------------------------------------------------------------------------
  std::cout << "Testing sum..." << std::endl;
  cpu_result = 0;
  for (std::size_t i=0; i<std_v1.size(); ++i)
    cpu_result += std_v1[i];
  cpu_result2 = viennacl::linalg::sum(vcl_v1);
  gpu_result = viennacl::linalg::sum(vcl_v1);

  if (check(cpu_result, cpu_result2) != EXIT_SUCCESS)
    return EXIT_FAILURE;
  if (check(cpu_result, gpu_result) != EXIT_SUCCESS)
    return EXIT_FAILURE;

  cpu_result = 0;
  for (std::size_t i=0; i<std_v1.size(); ++i)
    cpu_result += std_v1[i] + std_v2[i];
  cpu_result3 = viennacl::linalg::sum(vcl_v1 + vcl_v2);
  gpu_result = viennacl::linalg::sum(vcl_v1 + vcl_v2);

  if (check(cpu_result, cpu_result3) != EXIT_SUCCESS)
    return EXIT_FAILURE;
  if (check(cpu_result, gpu_result) != EXIT_SUCCESS)
    return EXIT_FAILURE;


  // --------------------------------------------------------------------------

  std::cout << "Testing assignments..." << std::endl;
  NumericT val = static_cast<NumericT>(1);
  for (size_t i=0; i < std_v1.size(); ++i)
    std_v1[i] = val;

  for (size_t i=0; i < vcl_v1.size(); ++i)
    vcl_v1(i) = val;

  if (check(std_v1, vcl_v1) != EXIT_SUCCESS)
    return EXIT_FAILURE;


  //
  // multiplication and division of vectors by scalars
  //
  std::cout << "Testing scaling with CPU scalar..." << std::endl;
  NumericT alpha = static_cast<NumericT>(3);
  viennacl::scalar<NumericT> gpu_alpha = alpha;

  for (size_t i=0; i<std_v1.size(); ++i)
    std_v1[i] *= alpha;
  vcl_v1 *= alpha;

  if (check(std_v1, vcl_v1) != EXIT_SUCCESS)
    return EXIT_FAILURE;

  std::cout << "Testing scaling with GPU scalar..." << std::endl;
  for (size_t i=0; i<std_v1.size(); ++i)
    std_v1[i] *= alpha;
  vcl_v1 *= gpu_alpha;

  if (check(std_v1, vcl_v1) != EXIT_SUCCESS)
    return EXIT_FAILURE;

  NumericT beta  = static_cast<NumericT>(2);
  viennacl::scalar<NumericT> gpu_beta = beta;

  std::cout << "Testing shrinking with CPU scalar..." << std::endl;
  for (size_t i=0; i<std_v1.size(); ++i)
    std_v1[i] /= beta;
  vcl_v1 /= beta;

  if (check(std_v1, vcl_v1) != EXIT_SUCCESS)
    return EXIT_FAILURE;

  std::cout << "Testing shrinking with GPU scalar..." << std::endl;
  for (size_t i=0; i<std_v1.size(); ++i)
    std_v1[i] /= beta;
  vcl_v1 /= gpu_beta;

  if (check(std_v1, vcl_v1) != EXIT_SUCCESS)
    return EXIT_FAILURE;


  //
  // add and inplace_add of vectors
  //
  for (size_t i=0; i<std_v1.size(); ++i)
    std_v1[i] = NumericT(i);
  for (size_t i=0; i<std_v1.size(); ++i)
    std_v2[i] = 3 * std_v1[i];
  viennacl::copy(std_v1.begin(), std_v1.end(), vcl_v1.begin());  //resync
  viennacl::copy(std_v2.begin(), std_v2.end(), vcl_v2.begin());

  std::cout << "Testing add on vector..." << std::endl;

  std::cout << "Checking for successful copy..." << std::endl;
  if (check(std_v1, vcl_v1) != EXIT_SUCCESS)
    return EXIT_FAILURE;
  if (check(std_v2, vcl_v2) != EXIT_SUCCESS)
    return EXIT_FAILURE;

  for (size_t i=0; i<std_v1.size(); ++i)
    std_v1[i] = std_v1[i] + std_v2[i];
  vcl_v1 = vcl_v1 + vcl_v2;

  if (check(std_v1, vcl_v1) != EXIT_SUCCESS)
    return EXIT_FAILURE;

  std::cout << "Testing inplace-add on vector..." << std::endl;
  for (size_t i=0; i<std_v1.size(); ++i)
    std_v1[i] += std_v2[i];
  vcl_v1 += vcl_v2;

  if (check(std_v1, vcl_v1) != EXIT_SUCCESS)
    return EXIT_FAILURE;


  //
  // multiply-add
  //
  std::cout << "Testing multiply-add on vector with CPU scalar (right)..." << std::endl;
  for (size_t i=0; i < std_v1.size(); ++i)
    std_v1[i] = NumericT(i);
  for (std::size_t i=0; i<std_v1.size(); ++i)
    std_v2[i] = 3 * std_v1[i];
  viennacl::copy(std_v1.begin(), std_v1.end(), vcl_v1.begin());
  viennacl::copy(std_v2.begin(), std_v2.end(), vcl_v2.begin());

  for (std::size_t i=0; i<std_v1.size(); ++i)
    std_v1[i] = std_v1[i] + alpha * std_v2[i];
  vcl_v1 = vcl_v1 + alpha * vcl_v2;

  if (check(std_v1, vcl_v1) != EXIT_SUCCESS)
    return EXIT_FAILURE;

  std::cout << "Testing multiply-add on vector with CPU scalar (left)..." << std::endl;
  for (std::size_t i=0; i<std_v1.size(); ++i)
    std_v2[i] = 3 * std_v1[i];
  viennacl::copy(std_v1.begin(), std_v1.end(), vcl_v1.begin());
  viennacl::copy(std_v2.begin(), std_v2.end(), vcl_v2.begin());

  for (std::size_t i=0; i<std_v1.size(); ++i)
    std_v1[i] = alpha * std_v1[i] + std_v2[i];
  vcl_v1 = alpha * vcl_v1 + vcl_v2;

  if (check(std_v1, vcl_v1) != EXIT_SUCCESS)
    return EXIT_FAILURE;

  std::cout << "Testing multiply-add on vector with CPU scalar (both)..." << std::endl;
  for (std::size_t i=0; i<std_v1.size(); ++i)
    std_v2[i] = 3 * std_v1[i];
  viennacl::copy(std_v1.begin(), std_v1.end(), vcl_v1.begin());
  viennacl::copy(std_v2.begin(), std_v2.end(), vcl_v2.begin());

  for (std::size_t i=0; i<std_v1.size(); ++i)
    std_v1[i] = alpha * std_v1[i] + beta * std_v2[i];
  vcl_v1 = alpha * vcl_v1 + beta * vcl_v2;

  if (check(std_v1, vcl_v1) != EXIT_SUCCESS)
    return EXIT_FAILURE;


  std::cout << "Testing inplace multiply-add on vector with CPU scalar..." << std::endl;
  for (std::size_t i=0; i<std_v1.size(); ++i)
    std_v2[i] = 3 * std_v1[i];
  viennacl::copy(std_v1.begin(), std_v1.end(), vcl_v1.begin());
  viennacl::copy(std_v2.begin(), std_v2.end(), vcl_v2.begin());

  for (std::size_t i=0; i<std_v1.size(); ++i)
    std_v1[i] += alpha * std_v2[i];
  vcl_v1 += alpha * vcl_v2;

  if (check(std_v1, vcl_v1) != EXIT_SUCCESS)
    return EXIT_FAILURE;


  std::cout << "Testing multiply-add on vector with GPU scalar (right)..." << std::endl;
  for (std::size_t i=0; i<std_v1.size(); ++i)
    std_v2[i] = 3 * std_v1[i];
  viennacl::copy(std_v1.begin(), std_v1.end(), vcl_v1.begin());
  viennacl::copy(std_v2.begin(), std_v2.end(), vcl_v2.begin());

  for (std::size_t i=0; i<std_v1.size(); ++i)
    std_v1[i] = std_v1[i] + alpha * std_v2[i];
  vcl_v1   = vcl_v1   + gpu_alpha *   vcl_v2;

  if (check(std_v1, vcl_v1) != EXIT_SUCCESS)
    return EXIT_FAILURE;

  std::cout << "Testing multiply-add on vector with GPU scalar (left)..." << std::endl;
  for (std::size_t i=0; i<std_v1.size(); ++i)
    std_v2[i] = 3 * std_v1[i];
  viennacl::copy(std_v1.begin(), std_v1.end(), vcl_v1.begin());
  viennacl::copy(std_v2.begin(), std_v2.end(), vcl_v2.begin());

  for (std::size_t i=0; i<std_v1.size(); ++i)
    std_v1[i] = std_v1[i] + alpha * std_v2[i];
  vcl_v1 = vcl_v1 + gpu_alpha * vcl_v2;

  if (check(std_v1, vcl_v1) != EXIT_SUCCESS)
    return EXIT_FAILURE;

  std::cout << "Testing multiply-add on vector with GPU scalar (both)..." << std::endl;
  for (std::size_t i=0; i<std_v1.size(); ++i)
    std_v2[i] = 3 * std_v1[i];
  viennacl::copy(std_v1.begin(), std_v1.end(), vcl_v1.begin());
  viennacl::copy(std_v2.begin(), std_v2.end(), vcl_v2.begin());

  for (std::size_t i=0; i<std_v1.size(); ++i)
    std_v1[i] = alpha * std_v1[i] + beta * std_v2[i];
  vcl_v1 = gpu_alpha * vcl_v1 + gpu_beta * vcl_v2;

  if (check(std_v1, vcl_v1) != EXIT_SUCCESS)
    return EXIT_FAILURE;


  std::cout << "Testing inplace multiply-add on vector with GPU scalar (both, adding)..." << std::endl;
  for (std::size_t i=0; i<std_v1.size(); ++i)
    std_v2[i] = 3 * std_v1[i];
  viennacl::copy(std_v1.begin(), std_v1.end(), vcl_v1.begin());
  viennacl::copy(std_v2.begin(), std_v2.end(), vcl_v2.begin());

  for (std::size_t i=0; i<std_v1.size(); ++i)
    std_v1[i] += alpha * std_v1[i] + beta * std_v2[i];
  vcl_v1 += gpu_alpha * vcl_v1 + gpu_beta * vcl_v2;

  if (check(std_v1, vcl_v1) != EXIT_SUCCESS)
    return EXIT_FAILURE;


  std::cout << "Testing inplace multiply-add on vector with GPU scalar..." << std::endl;
  for (std::size_t i=0; i<std_v1.size(); ++i)
    std_v2[i] = 3 * std_v1[i];
  viennacl::copy(std_v1.begin(), std_v1.end(), vcl_v1.begin());
  viennacl::copy(std_v2.begin(), std_v2.end(), vcl_v2.begin());

  for (std::size_t i=0; i<std_v1.size(); ++i)
    std_v1[i] += alpha * std_v2[i];
  vcl_v1 += gpu_alpha * vcl_v2;

  if (check(std_v1, vcl_v1) != EXIT_SUCCESS)
    return EXIT_FAILURE;


  //
  // division-add
  //
  std::cout << "Testing division-add on vector with CPU scalar (right)..." << std::endl;
  for (size_t i=0; i < std_v1.size(); ++i)
    std_v1[i] = NumericT(i);
  for (std::size_t i=0; i<std_v1.size(); ++i)
    std_v2[i] = 3 * std_v1[i];
  viennacl::copy(std_v1.begin(), std_v1.end(), vcl_v1.begin());
  viennacl::copy(std_v2.begin(), std_v2.end(), vcl_v2.begin());

  for (std::size_t i=0; i<std_v1.size(); ++i)
    std_v1[i] = std_v1[i] + std_v2[i] / alpha;
  vcl_v1 = vcl_v1 + vcl_v2 / alpha;

  if (check(std_v1, vcl_v1) != EXIT_SUCCESS)
    return EXIT_FAILURE;


  std::cout << "Testing division-add on vector with CPU scalar (left)..." << std::endl;
  for (std::size_t i=0; i<std_v1.size(); ++i)
    std_v2[i] = 3 * std_v1[i];
  viennacl::copy(std_v1.begin(), std_v1.end(), vcl_v1.begin());
  viennacl::copy(std_v2.begin(), std_v2.end(), vcl_v2.begin());

  for (std::size_t i=0; i<std_v1.size(); ++i)
    std_v1[i] = std_v1[i] / alpha + std_v2[i];
  vcl_v1 = vcl_v1 / alpha + vcl_v2;

  if (check(std_v1, vcl_v1) != EXIT_SUCCESS)
    return EXIT_FAILURE;

  std::cout << "Testing division-add on vector with CPU scalar (both)..." << std::endl;
  for (std::size_t i=0; i<std_v1.size(); ++i)
    std_v2[i] = 3 * std_v1[i];
  viennacl::copy(std_v1.begin(), std_v1.end(), vcl_v1.begin());
  viennacl::copy(std_v2.begin(), std_v2.end(), vcl_v2.begin());

  for (std::size_t i=0; i<std_v1.size(); ++i)
    std_v1[i] = std_v1[i] / alpha + std_v2[i] / beta;
  vcl_v1 = vcl_v1 / alpha + vcl_v2 / beta;

  if (check(std_v1, vcl_v1) != EXIT_SUCCESS)
    return EXIT_FAILURE;

  std::cout << "Testing division-multiply-add on vector with CPU scalar..." << std::endl;
  for (std::size_t i=0; i<std_v1.size(); ++i)
    std_v2[i] = 3 * std_v1[i];
  viennacl::copy(std_v1.begin(), std_v1.end(), vcl_v1.begin());
  viennacl::copy(std_v2.begin(), std_v2.end(), vcl_v2.begin());

  for (std::size_t i=0; i<std_v1.size(); ++i)
    std_v1[i] = std_v1[i] / alpha + std_v2[i] * beta;
  vcl_v1 = vcl_v1 / alpha + vcl_v2 * beta;

  if (check(std_v1, vcl_v1) != EXIT_SUCCESS)
    return EXIT_FAILURE;


  std::cout << "Testing multiply-division-add on vector with CPU scalar..." << std::endl;
  for (std::size_t i=0; i<std_v1.size(); ++i)
    std_v2[i] = 3 * std_v1[i];
  viennacl::copy(std_v1.begin(), std_v1.end(), vcl_v1.begin());
  viennacl::copy(std_v2.begin(), std_v2.end(), vcl_v2.begin());

  for (std::size_t i=0; i<std_v1.size(); ++i)
    std_v1[i] = std_v1[i] * alpha + std_v2[i] / beta;
  vcl_v1 = vcl_v1 * alpha + vcl_v2 / beta;

  if (check(std_v1, vcl_v1) != EXIT_SUCCESS)
    return EXIT_FAILURE;



  std::cout << "Testing inplace division-add on vector with CPU scalar..." << std::endl;
  for (std::size_t i=0; i<std_v1.size(); ++i)
    std_v2[i] = 3 * std_v1[i];
  viennacl::copy(std_v1.begin(), std_v1.end(), vcl_v1.begin());
  viennacl::copy(std_v2.begin(), std_v2.end(), vcl_v2.begin());

  for (std::size_t i=0; i<std_v1.size(); ++i)
    std_v1[i] += std_v2[i] / alpha;
  vcl_v1 += vcl_v2 / alpha;

  if (check(std_v1, vcl_v1) != EXIT_SUCCESS)
    return EXIT_FAILURE;


  std::cout << "Testing division-add on vector with GPU scalar (right)..." << std::endl;
  for (std::size_t i=0; i<std_v1.size(); ++i)
    std_v2[i] = 3 * std_v1[i];
  viennacl::copy(std_v1.begin(), std_v1.end(), vcl_v1.begin());
  viennacl::copy(std_v2.begin(), std_v2.end(), vcl_v2.begin());

  for (std::size_t i=0; i<std_v1.size(); ++i)
    std_v1[i] = std_v1[i] + std_v2[i] / alpha;
  vcl_v1 = vcl_v1 + vcl_v2 / gpu_alpha;

  if (check(std_v1, vcl_v1) != EXIT_SUCCESS)
    return EXIT_FAILURE;

  std::cout << "Testing division-add on vector with GPU scalar (left)..." << std::endl;
  for (std::size_t i=0; i<std_v1.size(); ++i)
    std_v2[i] = 3 * std_v1[i];
  viennacl::copy(std_v1.begin(), std_v1.end(), vcl_v1.begin());
  viennacl::copy(std_v2.begin(), std_v2.end(), vcl_v2.begin());

  for (std::size_t i=0; i<std_v1.size(); ++i)
    std_v1[i] = std_v1[i] + std_v2[i] / alpha;
  vcl_v1   = vcl_v1   +   vcl_v2 / gpu_alpha;

  if (check(std_v1, vcl_v1) != EXIT_SUCCESS)
    return EXIT_FAILURE;

  std::cout << "Testing division-add on vector with GPU scalar (both)..." << std::endl;
  for (std::size_t i=0; i<std_v1.size(); ++i)
    std_v2[i] = 3 * std_v1[i];
  viennacl::copy(std_v1.begin(), std_v1.end(), vcl_v1.begin());
  viennacl::copy(std_v2.begin(), std_v2.end(), vcl_v2.begin());

  for (std::size_t i=0; i<std_v1.size(); ++i)
    std_v1[i] = std_v1[i] / alpha + std_v2[i] / beta;
  vcl_v1 = vcl_v1 / gpu_alpha + vcl_v2 / gpu_beta;

  if (check(std_v1, vcl_v1) != EXIT_SUCCESS)
    return EXIT_FAILURE;


  std::cout << "Testing inplace division-add on vector with GPU scalar (both, adding)..." << std::endl;
  for (std::size_t i=0; i<std_v1.size(); ++i)
    std_v2[i] = 3 * std_v1[i];
  viennacl::copy(std_v1.begin(), std_v1.end(), vcl_v1.begin());
  viennacl::copy(std_v2.begin(), std_v2.end(), vcl_v2.begin());

  for (std::size_t i=0; i<std_v1.size(); ++i)
    std_v1[i] += std_v1[i] / alpha + std_v2[i] / beta;
  vcl_v1 += vcl_v1 / gpu_alpha + vcl_v2 / gpu_beta;

  if (check(std_v1, vcl_v1) != EXIT_SUCCESS)
    return EXIT_FAILURE;

  std::cout << "Testing inplace division-multiply-add on vector with GPU scalar (adding)..." << std::endl;
  for (std::size_t i=0; i<std_v1.size(); ++i)
    std_v2[i] = 3 * std_v1[i];
  viennacl::copy(std_v1.begin(), std_v1.end(), vcl_v1.begin());
  viennacl::copy(std_v2.begin(), std_v2.end(), vcl_v2.begin());

  for (std::size_t i=0; i<std_v1.size(); ++i)
    std_v1[i] += std_v1[i] / alpha + std_v2[i] * beta;
  vcl_v1 += vcl_v1 / gpu_alpha + vcl_v2 * gpu_beta;

  if (check(std_v1, vcl_v1) != EXIT_SUCCESS)
    return EXIT_FAILURE;


  std::cout << "Testing inplace division-add on vector with GPU scalar..." << std::endl;
  for (std::size_t i=0; i<std_v1.size(); ++i)
    std_v2[i] = 3 * std_v1[i];
  viennacl::copy(std_v1.begin(), std_v1.end(), vcl_v1.begin());
  viennacl::copy(std_v2.begin(), std_v2.end(), vcl_v2.begin());

  for (std::size_t i=0; i<std_v1.size(); ++i)
    std_v1[i] += std_v2[i] * alpha;
  vcl_v1 += vcl_v2 * gpu_alpha;

  if (check(std_v1, vcl_v1) != EXIT_SUCCESS)
    return EXIT_FAILURE;

  //
  // More complicated expressions (for ensuring the operator overloads work correctly)
  //
  for (size_t i=0; i < std_v1.size(); ++i)
    std_v1[i] = NumericT(i);
  for (std::size_t i=0; i<std_v1.size(); ++i)
    std_v2[i] = 3 * std_v1[i];
  viennacl::copy(std_v1.begin(), std_v1.end(), vcl_v1.begin());
  viennacl::copy(std_v2.begin(), std_v2.end(), vcl_v2.begin());

  std::cout << "Testing three vector additions..." << std::endl;
  for (std::size_t i=0; i<std_v1.size(); ++i)
    std_v1[i] = std_v2[i] + std_v1[i] + std_v2[i];
  vcl_v1 = vcl_v2 + vcl_v1 + vcl_v2;

  if (check(std_v1, vcl_v1) != EXIT_SUCCESS)
    return EXIT_FAILURE;

  // --------------------------------------------------------------------------
  for (std::size_t i=0; i<std_v1.size(); ++i)
    std_v2[i] = 3 * std_v1[i];
  viennacl::copy(std_v1.begin(), std_v1.end(), vcl_v1.begin());
  viennacl::copy(std_v2.begin(), std_v2.end(), vcl_v2.begin());

  std::cout << "Testing swap..." << std::endl;
  swap(std_v1, std_v2);
  swap(vcl_v1, vcl_v2);

  if (check(std_v1, vcl_v1) != EXIT_SUCCESS)
    return EXIT_FAILURE;

  std::cout << "Testing elementwise multiplication..." << std::endl;
  std::cout << " v1 = element_prod(v1, v2);" << std::endl;
  for (std::size_t i=0; i<std_v1.size(); ++i)
    std_v1[i] = std_v1[i] * std_v2[i];
  vcl_v1 = viennacl::linalg::element_prod(vcl_v1, vcl_v2);

  if (check(std_v1, vcl_v1) != EXIT_SUCCESS)
    return EXIT_FAILURE;

  std::cout << " v1 += element_prod(v1, v2);" << std::endl;
  for (std::size_t i=0; i<std_v1.size(); ++i)
    std_v1[i] += std_v1[i] * std_v2[i];
  vcl_v1 += viennacl::linalg::element_prod(vcl_v1, vcl_v2);

  if (check(std_v1, vcl_v1) != EXIT_SUCCESS)
    return EXIT_FAILURE;

  ///////
  std::cout << " v1 = element_prod(v1 + v2, v2);" << std::endl;
  for (std::size_t i=0; i<std_v1.size(); ++i)
    std_v1[i] = (std_v1[i] + std_v2[i]) * std_v2[i];
  vcl_v1 = viennacl::linalg::element_prod(vcl_v1 + vcl_v2, vcl_v2);

  if (check(std_v1, vcl_v1) != EXIT_SUCCESS)
    return EXIT_FAILURE;

  std::cout << " v1 += element_prod(v1 + v2, v2);" << std::endl;
  for (std::size_t i=0; i<std_v1.size(); ++i)
    std_v1[i] += (std_v1[i] + std_v2[i]) * std_v2[i];
  vcl_v1 += viennacl::linalg::element_prod(vcl_v1 + vcl_v2, vcl_v2);

  if (check(std_v1, vcl_v1) != EXIT_SUCCESS)
    return EXIT_FAILURE;

  ///////
  std::cout << " v1 = element_prod(v1, v2 + v1);" << std::endl;
  for (std::size_t i=0; i<std_v1.size(); ++i)
    std_v1[i] = std_v1[i] * (std_v2[i] + std_v1[i]);
  vcl_v1 = viennacl::linalg::element_prod(vcl_v1, vcl_v2 + vcl_v1);

  if (check(std_v1, vcl_v1) != EXIT_SUCCESS)
    return EXIT_FAILURE;

  std::cout << " v1 += element_prod(v1, v2 + v1);" << std::endl;
  for (std::size_t i=0; i<std_v1.size(); ++i)
    std_v1[i] += std_v1[i] * (std_v2[i] + std_v1[i]);
  vcl_v1 += viennacl::linalg::element_prod(vcl_v1, vcl_v2 + vcl_v1);

  if (check(std_v1, vcl_v1) != EXIT_SUCCESS)
    return EXIT_FAILURE;

  ///////
  std::cout << " v1 = element_prod(v1 + v2, v2 + v1);" << std::endl;
  for (std::size_t i=0; i<std_v1.size(); ++i)
    std_v1[i] = (std_v1[i] + std_v2[i]) * (std_v2[i] + std_v1[i]);
  vcl_v1 = viennacl::linalg::element_prod(vcl_v1 + vcl_v2, vcl_v2 + vcl_v1);

  if (check(std_v1, vcl_v1) != EXIT_SUCCESS)
    return EXIT_FAILURE;

  std::cout << " v1 += element_prod(v1 + v2, v2 + v1);" << std::endl;
  for (std::size_t i=0; i<std_v1.size(); ++i)
    std_v1[i] += (std_v1[i] + std_v2[i]) * (std_v2[i] + std_v1[i]);
  vcl_v1 += viennacl::linalg::element_prod(vcl_v1 + vcl_v2, vcl_v2 + vcl_v1);

  if (check(std_v1, vcl_v1) != EXIT_SUCCESS)
    return EXIT_FAILURE;


  std::cout << "Testing elementwise division..." << std::endl;
  for (std::size_t i=0; i<std_v1.size(); ++i)
  {
    std_v1[i] = NumericT(1 + i);
    std_v2[i] = NumericT(5 + i);
  }

  viennacl::copy(std_v1.begin(), std_v1.end(), vcl_v1.begin());
  viennacl::copy(std_v2.begin(), std_v2.end(), vcl_v2.begin());

  for (std::size_t i=0; i<std_v1.size(); ++i)
    std_v1[i] = std_v1[i] / std_v2[i];
  vcl_v1 = viennacl::linalg::element_div(vcl_v1, vcl_v2);

  if (check(std_v1, vcl_v1) != EXIT_SUCCESS)
    return EXIT_FAILURE;

  for (std::size_t i=0; i<std_v1.size(); ++i)
    std_v1[i] += std_v1[i] / std_v2[i];
  vcl_v1 += viennacl::linalg::element_div(vcl_v1, vcl_v2);

  if (check(std_v1, vcl_v1) != EXIT_SUCCESS)
    return EXIT_FAILURE;

  ///////
  for (std::size_t i=0; i<std_v1.size(); ++i)
    std_v1[i] = (std_v1[i] + std_v2[i]) / std_v2[i];
  vcl_v1 = viennacl::linalg::element_div(vcl_v1 + vcl_v2, vcl_v2);

  if (check(std_v1, vcl_v1) != EXIT_SUCCESS)
    return EXIT_FAILURE;

  for (std::size_t i=0; i<std_v1.size(); ++i)
    std_v1[i] += (std_v1[i] + std_v2[i]) / std_v2[i];
  vcl_v1 += viennacl::linalg::element_div(vcl_v1 + vcl_v2, vcl_v2);

  if (check(std_v1, vcl_v1) != EXIT_SUCCESS)
    return EXIT_FAILURE;

  ///////
  for (std::size_t i=0; i<std_v1.size(); ++i)
    std_v1[i] = std_v1[i] / (std_v2[i] + std_v1[i]);
  vcl_v1 = viennacl::linalg::element_div(vcl_v1, vcl_v2 + vcl_v1);

  if (check(std_v1, vcl_v1) != EXIT_SUCCESS)
    return EXIT_FAILURE;

  for (std::size_t i=0; i<std_v1.size(); ++i)
    std_v1[i] += std_v1[i] / (std_v2[i] + std_v1[i]);
  vcl_v1 += viennacl::linalg::element_div(vcl_v1, vcl_v2 + vcl_v1);

  if (check(std_v1, vcl_v1) != EXIT_SUCCESS)
    return EXIT_FAILURE;

  ///////
  for (std::size_t i=0; i<std_v1.size(); ++i)
    std_v1[i] = (std_v1[i] + std_v2[i]) / (std_v2[i] + std_v1[i]);
  vcl_v1 = viennacl::linalg::element_div(vcl_v1 + vcl_v2, vcl_v2 + vcl_v1);

  if (check(std_v1, vcl_v1) != EXIT_SUCCESS)
    return EXIT_FAILURE;

  for (std::size_t i=0; i<std_v1.size(); ++i)
    std_v1[i] += (std_v1[i] + std_v2[i]) / (std_v2[i] + std_v1[i]);
  vcl_v1 += viennacl::linalg::element_div(vcl_v1 + vcl_v2, vcl_v2 + vcl_v1);

  if (check(std_v1, vcl_v1) != EXIT_SUCCESS)
    return EXIT_FAILURE;

  // --------------------------------------------------------------------------
  return retval;
}


template< typename NumericT >
int test()
{
  int retval = EXIT_SUCCESS;
  std::size_t size = 12345;

  std::cout << "Running tests for vector of size " << size << std::endl;

  //
  // Set up STL objects
  //
  std::vector<NumericT> std_full_vec(size);
  std::vector<NumericT> std_full_vec2(std_full_vec.size());

  for (std::size_t i=0; i<std_full_vec.size(); ++i)
  {
    std_full_vec[i]  = NumericT(1.0) + NumericT(i);
    std_full_vec2[i] = NumericT(2.0) + NumericT(i) / NumericT(2);
  }

  std::vector<NumericT> std_range_vec (2 * std_full_vec.size() / 4 - std_full_vec.size() / 4);
  std::vector<NumericT> std_range_vec2(2 * std_full_vec.size() / 4 - std_full_vec.size() / 4);

  for (std::size_t i=0; i<std_range_vec.size(); ++i)
    std_range_vec[i] = std_full_vec[i + std_full_vec.size() / 4];
  for (std::size_t i=0; i<std_range_vec2.size(); ++i)
    std_range_vec2[i] = std_full_vec2[i + 2 * std_full_vec2.size() / 4];

  std::vector<NumericT> std_slice_vec (std_full_vec.size() / 4);
  std::vector<NumericT> std_slice_vec2(std_full_vec.size() / 4);

  for (std::size_t i=0; i<std_slice_vec.size(); ++i)
    std_slice_vec[i] = std_full_vec[3*i + std_full_vec.size() / 4];
  for (std::size_t i=0; i<std_slice_vec2.size(); ++i)
    std_slice_vec2[i] = std_full_vec2[2*i + 2 * std_full_vec2.size() / 4];

  //
  // Set up ViennaCL objects
  //
  viennacl::vector<NumericT> vcl_full_vec(std_full_vec.size());
  viennacl::vector<NumericT> vcl_full_vec2(std_full_vec2.size());

  viennacl::fast_copy(std_full_vec.begin(), std_full_vec.end(), vcl_full_vec.begin());
  viennacl::copy(std_full_vec2.begin(), std_full_vec2.end(), vcl_full_vec2.begin());

  viennacl::range vcl_r1(    vcl_full_vec.size() / 4, 2 * vcl_full_vec.size() / 4);
  viennacl::range vcl_r2(2 * vcl_full_vec2.size() / 4, 3 * vcl_full_vec2.size() / 4);
  viennacl::vector_range< viennacl::vector<NumericT> > vcl_range_vec(vcl_full_vec, vcl_r1);
  viennacl::vector_range< viennacl::vector<NumericT> > vcl_range_vec2(vcl_full_vec2, vcl_r2);

  {
    viennacl::vector<NumericT> vcl_short_vec(vcl_range_vec);
    viennacl::vector<NumericT> vcl_short_vec2 = vcl_range_vec2;

    std::vector<NumericT> std_short_vec(std_range_vec);
    std::vector<NumericT> std_short_vec2(std_range_vec2);

    std::cout << "Testing creation of vectors from range..." << std::endl;
    if (check(std_short_vec, vcl_short_vec) != EXIT_SUCCESS)
      return EXIT_FAILURE;
    if (check(std_short_vec2, vcl_short_vec2) != EXIT_SUCCESS)
      return EXIT_FAILURE;
  }

  viennacl::slice vcl_s1(    vcl_full_vec.size() / 4, 3, vcl_full_vec.size() / 4);
  viennacl::slice vcl_s2(2 * vcl_full_vec2.size() / 4, 2, vcl_full_vec2.size() / 4);
  viennacl::vector_slice< viennacl::vector<NumericT> > vcl_slice_vec(vcl_full_vec, vcl_s1);
  viennacl::vector_slice< viennacl::vector<NumericT> > vcl_slice_vec2(vcl_full_vec2, vcl_s2);

  viennacl::vector<NumericT> vcl_short_vec(vcl_slice_vec);
  viennacl::vector<NumericT> vcl_short_vec2 = vcl_slice_vec2;

  std::vector<NumericT> std_short_vec(std_slice_vec);
  std::vector<NumericT> std_short_vec2(std_slice_vec2);

  std::cout << "Testing creation of vectors from slice..." << std::endl;
  if (check(std_short_vec, vcl_short_vec) != EXIT_SUCCESS)
    return EXIT_FAILURE;
  if (check(std_short_vec2, vcl_short_vec2) != EXIT_SUCCESS)
    return EXIT_FAILURE;


  //
  // Now start running tests for vectors, ranges and slices:
  //

  std::cout << " ** vcl_v1 = vector, vcl_v2 = vector **" << std::endl;
  retval = test<NumericT>(std_short_vec, std_short_vec2,
                          vcl_short_vec, vcl_short_vec2);
  if (retval != EXIT_SUCCESS)
    return EXIT_FAILURE;

  std::cout << " ** vcl_v1 = vector, vcl_v2 = range **" << std::endl;
  retval = test<NumericT>(std_short_vec, std_short_vec2,
                          vcl_short_vec, vcl_range_vec2);
  if (retval != EXIT_SUCCESS)
    return EXIT_FAILURE;

  std::cout << " ** vcl_v1 = vector, vcl_v2 = slice **" << std::endl;
  retval = test<NumericT>(std_short_vec, std_short_vec2,
                          vcl_short_vec, vcl_slice_vec2);
  if (retval != EXIT_SUCCESS)
    return EXIT_FAILURE;

  ///////

  std::cout << " ** vcl_v1 = range, vcl_v2 = vector **" << std::endl;
  retval = test<NumericT>(std_short_vec, std_short_vec2,
                          vcl_range_vec, vcl_short_vec2);
  if (retval != EXIT_SUCCESS)
    return EXIT_FAILURE;

  std::cout << " ** vcl_v1 = range, vcl_v2 = range **" << std::endl;
  retval = test<NumericT>(std_short_vec, std_short_vec2,
                          vcl_range_vec, vcl_range_vec2);
  if (retval != EXIT_SUCCESS)
    return EXIT_FAILURE;

  std::cout << " ** vcl_v1 = range, vcl_v2 = slice **" << std::endl;
  retval = test<NumericT>(std_short_vec, std_short_vec2,
                          vcl_range_vec, vcl_slice_vec2);
  if (retval != EXIT_SUCCESS)
    return EXIT_FAILURE;

  ///////

  std::cout << " ** vcl_v1 = slice, vcl_v2 = vector **" << std::endl;
  retval = test<NumericT>(std_short_vec, std_short_vec2,
                          vcl_slice_vec, vcl_short_vec2);
  if (retval != EXIT_SUCCESS)
    return EXIT_FAILURE;

  std::cout << " ** vcl_v1 = slice, vcl_v2 = range **" << std::endl;
  retval = test<NumericT>(std_short_vec, std_short_vec2,
                          vcl_slice_vec, vcl_range_vec2);
  if (retval != EXIT_SUCCESS)
    return EXIT_FAILURE;

  std::cout << " ** vcl_v1 = slice, vcl_v2 = slice **" << std::endl;
  retval = test<NumericT>(std_short_vec, std_short_vec2,
                          vcl_slice_vec, vcl_slice_vec2);
  if (retval != EXIT_SUCCESS)
    return EXIT_FAILURE;

  return EXIT_SUCCESS;
}



//
// -------------------------------------------------------------
//
int main()
{
  std::cout << std::endl;
  std::cout << "----------------------------------------------" << std::endl;
  std::cout << "----------------------------------------------" << std::endl;
  std::cout << "## Test :: Vector with Integer types" << std::endl;
  std::cout << "----------------------------------------------" << std::endl;
  std::cout << "----------------------------------------------" << std::endl;
  std::cout << std::endl;

  int retval = EXIT_SUCCESS;

  std::cout << std::endl;
  std::cout << "----------------------------------------------" << std::endl;
  std::cout << std::endl;
  {
    std::cout << "# Testing setup:" << std::endl;
    std::cout << "  numeric: unsigned int" << std::endl;
    retval = test<unsigned int>();
    if ( retval == EXIT_SUCCESS )
      std::cout << "# Test passed" << std::endl;
    else
      return retval;
  }
  std::cout << std::endl;
  std::cout << "----------------------------------------------" << std::endl;
  std::cout << std::endl;
  {
    std::cout << "# Testing setup:" << std::endl;
    std::cout << "  numeric: long" << std::endl;
    retval = test<unsigned long>();
    if ( retval == EXIT_SUCCESS )
      std::cout << "# Test passed" << std::endl;
    else
      return retval;
  }
  std::cout << std::endl;
  std::cout << "----------------------------------------------" << std::endl;
  std::cout << std::endl;

  std::cout << std::endl;
  std::cout << "------- Test completed --------" << std::endl;
  std::cout << std::endl;

  return retval;
}
