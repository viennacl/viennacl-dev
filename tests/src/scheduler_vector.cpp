/* =========================================================================
   Copyright (c) 2010-2016, Institute for Microelectronics,
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



/** \file tests/src/scheduler_vector.cpp  Tests the scheduler for vector-operations.
*   \test Tests the scheduler for vector-operations.
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
#include "viennacl/matrix.hpp"
#include "viennacl/vector_proxy.hpp"
#include "viennacl/linalg/inner_prod.hpp"
#include "viennacl/linalg/norm_1.hpp"
#include "viennacl/linalg/norm_2.hpp"
#include "viennacl/linalg/norm_inf.hpp"

#include "viennacl/scheduler/execute.hpp"
#include "viennacl/scheduler/io.hpp"

#include "viennacl/tools/random.hpp"


//
// -------------------------------------------------------------
//
template<typename ScalarType>
ScalarType diff(ScalarType const & s1, ScalarType const & s2)
{
   viennacl::backend::finish();
   if (std::fabs(s1 - s2) > 0)
      return (s1 - s2) / std::max(std::fabs(s1), std::fabs(s2));
   return 0;
}
//
// -------------------------------------------------------------
//
template<typename ScalarType>
ScalarType diff(ScalarType const & s1, viennacl::scalar<ScalarType> const & s2)
{
   viennacl::backend::finish();
   if (std::fabs(s1 - s2) > 0)
      return (s1 - s2) / std::max(std::fabs(s1), std::fabs(s2));
   return 0;
}
//
// -------------------------------------------------------------
//
template<typename ScalarType>
ScalarType diff(ScalarType const & s1, viennacl::entry_proxy<ScalarType> const & s2)
{
   viennacl::backend::finish();
   if (std::fabs(s1 - s2) > 0)
      return (s1 - s2) / std::max(std::fabs(s1), std::fabs(s2));
   return 0;
}
//
// -------------------------------------------------------------
//
template<typename ScalarType, typename ViennaCLVectorType>
ScalarType diff(std::vector<ScalarType> const & v1, ViennaCLVectorType const & vcl_vec)
{
  std::vector<ScalarType> v2_cpu(vcl_vec.size());
  viennacl::backend::finish();
  viennacl::copy(vcl_vec, v2_cpu);

  ScalarType norm_inf_value = 0;
  for (std::size_t i=0;i<v1.size(); ++i)
  {
    ScalarType tmp = 0;
    if ( std::max( std::fabs(v2_cpu[i]), std::fabs(v1[i]) ) > 0 )
       tmp = std::fabs(v2_cpu[i] - v1[i]) / std::max( std::fabs(v2_cpu[i]), std::fabs(v1[i]) );

    norm_inf_value = (tmp > norm_inf_value) ? tmp : norm_inf_value;
  }

  return norm_inf_value;
}


template<typename T1, typename T2>
int check(T1 const & t1, T2 const & t2, double epsilon)
{
  int retval = EXIT_SUCCESS;

  double temp = std::fabs(diff(t1, t2));
  if (temp > epsilon)
  {
    std::cout << "# Error! Relative difference: " << temp << std::endl;
    retval = EXIT_FAILURE;
  }
  else
    std::cout << "PASSED!" << std::endl;
  return retval;
}


//
// -------------------------------------------------------------
//
template< typename NumericT, typename Epsilon, typename STLVectorType, typename ViennaCLVectorType1, typename ViennaCLVectorType2 >
int test(Epsilon const& epsilon,
         STLVectorType       & std_v1, STLVectorType       & std_v2,
         ViennaCLVectorType1 & vcl_v1, ViennaCLVectorType2 & vcl_v2)
{
  int retval = EXIT_SUCCESS;

  viennacl::tools::uniform_random_numbers<NumericT> randomNumber;

  NumericT                    cpu_result = 42.0;
  viennacl::scalar<NumericT>  gpu_result = 43.0;
  NumericT                    alpha      = NumericT(3.1415);
  NumericT                    beta       = NumericT(2.7172);

  //
  // Initializer:
  //
  std::cout << "Checking for zero_vector initializer..." << std::endl;
  std_v1 = std::vector<NumericT>(std_v1.size());
  vcl_v1 = viennacl::zero_vector<NumericT>(vcl_v1.size());
  if (check(std_v1, vcl_v1, epsilon) != EXIT_SUCCESS)
    return EXIT_FAILURE;

  std::cout << "Checking for scalar_vector initializer..." << std::endl;
  std_v1 = std::vector<NumericT>(std_v1.size(), cpu_result);
  vcl_v1 = viennacl::scalar_vector<NumericT>(vcl_v1.size(), cpu_result);
  if (check(std_v1, vcl_v1, epsilon) != EXIT_SUCCESS)
    return EXIT_FAILURE;

  std_v1 = std::vector<NumericT>(std_v1.size(), gpu_result);
  vcl_v1 = viennacl::scalar_vector<NumericT>(vcl_v1.size(), gpu_result);
  if (check(std_v1, vcl_v1, epsilon) != EXIT_SUCCESS)
    return EXIT_FAILURE;

  std::cout << "Checking for unit_vector initializer..." << std::endl;
  std_v1 = std::vector<NumericT>(std_v1.size()); std_v1[5] = NumericT(1);
  vcl_v1 = viennacl::unit_vector<NumericT>(vcl_v1.size(), 5);
  if (check(std_v1, vcl_v1, epsilon) != EXIT_SUCCESS)
    return EXIT_FAILURE;


  for (std::size_t i=0; i<std_v1.size(); ++i)
  {
    std_v1[i] = NumericT(1.0) + randomNumber();
    std_v2[i] = NumericT(1.0) + randomNumber();
  }

  viennacl::copy(std_v1.begin(), std_v1.end(), vcl_v1.begin());  //resync
  viennacl::copy(std_v2.begin(), std_v2.end(), vcl_v2.begin());

  std::cout << "Checking for successful copy..." << std::endl;
  if (check(std_v1, vcl_v1, epsilon) != EXIT_SUCCESS)
    return EXIT_FAILURE;
  if (check(std_v2, vcl_v2, epsilon) != EXIT_SUCCESS)
    return EXIT_FAILURE;


  // --------------------------------------------------------------------------

  std::cout << "Testing simple assignments..." << std::endl;

  {
  for (std::size_t i=0; i<std_v1.size(); ++i)
    std_v1[i] = std_v2[i];
  viennacl::scheduler::statement   my_statement(vcl_v1, viennacl::op_assign(), vcl_v2); // same as vcl_v1 = vcl_v2;
  viennacl::scheduler::execute(my_statement);

  if (check(std_v1, vcl_v1, epsilon) != EXIT_SUCCESS)
    return EXIT_FAILURE;
  }

  {
  for (std::size_t i=0; i<std_v1.size(); ++i)
    std_v1[i] += std_v2[i];
  viennacl::scheduler::statement   my_statement(vcl_v1, viennacl::op_inplace_add(), vcl_v2); // same as vcl_v1 += vcl_v2;
  viennacl::scheduler::execute(my_statement);

  if (check(std_v1, vcl_v1, epsilon) != EXIT_SUCCESS)
    return EXIT_FAILURE;
  }

  {
  for (std::size_t i=0; i<std_v1.size(); ++i)
    std_v1[i] -= std_v2[i];
  viennacl::scheduler::statement   my_statement(vcl_v1, viennacl::op_inplace_sub(), vcl_v2); // same as vcl_v1 -= vcl_v2;
  viennacl::scheduler::execute(my_statement);

  if (check(std_v1, vcl_v1, epsilon) != EXIT_SUCCESS)
    return EXIT_FAILURE;
  }

  std::cout << "Testing composite assignments..." << std::endl;
  {
  for (std::size_t i=0; i<std_v1.size(); ++i)
    std_v1[i] = std_v1[i] + std_v2[i];
  viennacl::scheduler::statement   my_statement(vcl_v1, viennacl::op_assign(), vcl_v1 + vcl_v2); // same as vcl_v1 = vcl_v1 + vcl_v2;
  viennacl::scheduler::execute(my_statement);

  if (check(std_v1, vcl_v1, epsilon) != EXIT_SUCCESS)
    return EXIT_FAILURE;
  }
  {
  for (std::size_t i=0; i<std_v1.size(); ++i)
    std_v1[i] += alpha * std_v1[i] - beta * std_v2[i] + std_v1[i] / beta - std_v2[i] / alpha;
  viennacl::scheduler::statement   my_statement(vcl_v1, viennacl::op_inplace_add(), alpha * vcl_v1 - beta * vcl_v2 + vcl_v1 / beta - vcl_v2 / alpha); // same as vcl_v1 += alpha * vcl_v1 - beta * vcl_v2 + beta * vcl_v1 - alpha * vcl_v2;
  viennacl::scheduler::execute(my_statement);

  if (check(std_v1, vcl_v1, epsilon) != EXIT_SUCCESS)
    return EXIT_FAILURE;
  }

  {
  for (std::size_t i=0; i<std_v1.size(); ++i)
    std_v1[i] = std_v1[i] - std_v2[i];
  viennacl::scheduler::statement   my_statement(vcl_v1, viennacl::op_assign(), vcl_v1 - vcl_v2); // same as vcl_v1 = vcl_v1 - vcl_v2;
  viennacl::scheduler::execute(my_statement);

  if (check(std_v1, vcl_v1, epsilon) != EXIT_SUCCESS)
    return EXIT_FAILURE;
  }

  std::cout << "--- Testing reductions ---" << std::endl;
  std::cout << "inner_prod..." << std::endl;
  {
  cpu_result = 0;
  for (std::size_t i=0; i<std_v1.size(); ++i)
    cpu_result += std_v1[i] * std_v2[i];
  viennacl::scheduler::statement   my_statement(gpu_result, viennacl::op_assign(), viennacl::linalg::inner_prod(vcl_v1, vcl_v2)); // same as gpu_result = inner_prod(vcl_v1, vcl_v2);
  viennacl::scheduler::execute(my_statement);

  if (check(cpu_result, gpu_result, epsilon) != EXIT_SUCCESS)
    return EXIT_FAILURE;
  }

  {
  cpu_result = 0;
  for (std::size_t i=0; i<std_v1.size(); ++i)
    cpu_result += (std_v1[i] + std_v2[i]) * std_v2[i];
  viennacl::scheduler::statement   my_statement(gpu_result, viennacl::op_assign(), viennacl::linalg::inner_prod(vcl_v1 + vcl_v2, vcl_v2)); // same as gpu_result = inner_prod(vcl_v1 + vcl_v2, vcl_v2);
  viennacl::scheduler::execute(my_statement);

  if (check(cpu_result, gpu_result, epsilon) != EXIT_SUCCESS)
    return EXIT_FAILURE;
  }

  {
  cpu_result = 0;
  for (std::size_t i=0; i<std_v1.size(); ++i)
    cpu_result += std_v1[i] * (std_v2[i] - std_v1[i]);
  viennacl::scheduler::statement   my_statement(gpu_result, viennacl::op_assign(), viennacl::linalg::inner_prod(vcl_v1, vcl_v2 - vcl_v1)); // same as gpu_result = inner_prod(vcl_v1, vcl_v2 - vcl_v1);
  viennacl::scheduler::execute(my_statement);

  if (check(cpu_result, gpu_result, epsilon) != EXIT_SUCCESS)
    return EXIT_FAILURE;
  }

  {
  cpu_result = 0;
  for (std::size_t i=0; i<std_v1.size(); ++i)
    cpu_result += (std_v1[i] - std_v2[i]) * (std_v2[i] + std_v1[i]);
  viennacl::scheduler::statement   my_statement(gpu_result, viennacl::op_assign(), viennacl::linalg::inner_prod(vcl_v1 - vcl_v2, vcl_v2 + vcl_v1)); // same as gpu_result = inner_prod(vcl_v1 - vcl_v2, vcl_v2 + vcl_v1);
  viennacl::scheduler::execute(my_statement);

  if (check(cpu_result, gpu_result, epsilon) != EXIT_SUCCESS)
    return EXIT_FAILURE;
  }

  std::cout << "norm_1..." << std::endl;
  {
  cpu_result = 0;
  for (std::size_t i=0; i<std_v1.size(); ++i)
    cpu_result += std::fabs(std_v1[i]);
  viennacl::scheduler::statement   my_statement(gpu_result, viennacl::op_assign(), viennacl::linalg::norm_1(vcl_v1)); // same as gpu_result = norm_1(vcl_v1);
  viennacl::scheduler::execute(my_statement);

  if (check(cpu_result, gpu_result, epsilon) != EXIT_SUCCESS)
    return EXIT_FAILURE;
  }

  {
  cpu_result = 0;
  for (std::size_t i=0; i<std_v1.size(); ++i)
    cpu_result += std::fabs(std_v1[i] + std_v2[i]);
  viennacl::scheduler::statement   my_statement(gpu_result, viennacl::op_assign(), viennacl::linalg::norm_1(vcl_v1 + vcl_v2)); // same as gpu_result = norm_1(vcl_v1 + vcl_v2);
  viennacl::scheduler::execute(my_statement);

  if (check(cpu_result, gpu_result, epsilon) != EXIT_SUCCESS)
    return EXIT_FAILURE;
  }

  std::cout << "norm_2..." << std::endl;
  {
  cpu_result = 0;
  for (std::size_t i=0; i<std_v1.size(); ++i)
    cpu_result += std_v1[i] * std_v1[i];
  cpu_result = std::sqrt(cpu_result);
  viennacl::scheduler::statement   my_statement(gpu_result, viennacl::op_assign(), viennacl::linalg::norm_2(vcl_v1)); // same as gpu_result = norm_2(vcl_v1);
  viennacl::scheduler::execute(my_statement);

  if (check(cpu_result, gpu_result, epsilon) != EXIT_SUCCESS)
    return EXIT_FAILURE;
  }

  {
  cpu_result = 0;
  for (std::size_t i=0; i<std_v1.size(); ++i)
    cpu_result += (std_v1[i] + std_v2[i]) * (std_v1[i] + std_v2[i]);
  cpu_result = std::sqrt(cpu_result);
  viennacl::scheduler::statement   my_statement(gpu_result, viennacl::op_assign(), viennacl::linalg::norm_2(vcl_v1 + vcl_v2)); // same as gpu_result = norm_2(vcl_v1 + vcl_v2);
  viennacl::scheduler::execute(my_statement);

  if (check(cpu_result, gpu_result, epsilon) != EXIT_SUCCESS)
    return EXIT_FAILURE;
  }

  std::cout << "norm_inf..." << std::endl;
  {
  cpu_result = 0;
  for (std::size_t i=0; i<std_v1.size(); ++i)
    cpu_result = std::max(cpu_result, std::fabs(std_v1[i]));
  viennacl::scheduler::statement   my_statement(gpu_result, viennacl::op_assign(), viennacl::linalg::norm_inf(vcl_v1)); // same as gpu_result = norm_inf(vcl_v1);
  viennacl::scheduler::execute(my_statement);

  if (check(cpu_result, gpu_result, epsilon) != EXIT_SUCCESS)
    return EXIT_FAILURE;
  }

  {
  cpu_result = 0;
  for (std::size_t i=0; i<std_v1.size(); ++i)
    cpu_result = std::max(cpu_result, std::fabs(std_v1[i] - std_v2[i]));
  viennacl::scheduler::statement   my_statement(gpu_result, viennacl::op_assign(), viennacl::linalg::norm_inf(vcl_v1 - vcl_v2)); // same as gpu_result = norm_inf(vcl_v1 - vcl_v2);
  viennacl::scheduler::execute(my_statement);

  if (check(cpu_result, gpu_result, epsilon) != EXIT_SUCCESS)
    return EXIT_FAILURE;
  }

  std::cout << "--- Testing elementwise operations (binary) ---" << std::endl;
  std::cout << "x = element_prod(x, y)... ";
  {
  for (std::size_t i=0; i<std_v1.size(); ++i)
    std_v1[i] = std_v1[i] * std_v2[i];
  viennacl::scheduler::statement   my_statement(vcl_v1, viennacl::op_assign(), viennacl::linalg::element_prod(vcl_v1, vcl_v2));
  viennacl::scheduler::execute(my_statement);

  if (check(std_v1, vcl_v1, epsilon) != EXIT_SUCCESS)
    return EXIT_FAILURE;
  }

  std::cout << "x = element_prod(x + y, y)... ";
  {
  for (std::size_t i=0; i<std_v1.size(); ++i)
    std_v1[i] = (std_v1[i] + std_v2[i]) * std_v2[i];
  viennacl::scheduler::statement   my_statement(vcl_v1, viennacl::op_assign(), viennacl::linalg::element_prod(vcl_v1 + vcl_v2, vcl_v2));
  viennacl::scheduler::execute(my_statement);

  if (check(std_v1, vcl_v1, epsilon) != EXIT_SUCCESS)
    return EXIT_FAILURE;
  }

  std::cout << "x = element_prod(x, x + y)... ";
  {
  for (std::size_t i=0; i<std_v1.size(); ++i)
    std_v1[i] = std_v1[i] * (std_v1[i] + std_v2[i]);
  viennacl::scheduler::statement   my_statement(vcl_v1, viennacl::op_assign(), viennacl::linalg::element_prod(vcl_v1, vcl_v2 + vcl_v1));
  viennacl::scheduler::execute(my_statement);

  if (check(std_v1, vcl_v1, epsilon) != EXIT_SUCCESS)
    return EXIT_FAILURE;
  }

  std::cout << "x = element_prod(x - y, y + x)... ";
  {
  for (std::size_t i=0; i<std_v1.size(); ++i)
    std_v1[i] = (std_v1[i] - std_v2[i]) * (std_v2[i] + std_v1[i]);
  viennacl::scheduler::statement   my_statement(vcl_v1, viennacl::op_assign(), viennacl::linalg::element_prod(vcl_v1 - vcl_v2, vcl_v2 + vcl_v1));
  viennacl::scheduler::execute(my_statement);

  if (check(std_v1, vcl_v1, epsilon) != EXIT_SUCCESS)
    return EXIT_FAILURE;
  }



  std::cout << "x = element_div(x, y)... ";
  {
  for (std::size_t i=0; i<std_v1.size(); ++i)
    std_v1[i] = std_v1[i] / std_v2[i];
  viennacl::scheduler::statement   my_statement(vcl_v1, viennacl::op_assign(), viennacl::linalg::element_div(vcl_v1, vcl_v2));
  viennacl::scheduler::execute(my_statement);

  if (check(std_v1, vcl_v1, epsilon) != EXIT_SUCCESS)
    return EXIT_FAILURE;
  }

  std::cout << "x = element_div(x + y, y)... ";
  {
  for (std::size_t i=0; i<std_v1.size(); ++i)
    std_v1[i] = (std_v1[i] + std_v2[i]) / std_v2[i];
  viennacl::scheduler::statement   my_statement(vcl_v1, viennacl::op_assign(), viennacl::linalg::element_div(vcl_v1 + vcl_v2, vcl_v2));
  viennacl::scheduler::execute(my_statement);

  if (check(std_v1, vcl_v1, epsilon) != EXIT_SUCCESS)
    return EXIT_FAILURE;
  }

  std::cout << "x = element_div(x, x + y)... ";
  {
  for (std::size_t i=0; i<std_v1.size(); ++i)
    std_v1[i] = std_v1[i] / (std_v1[i] + std_v2[i]);
  viennacl::scheduler::statement   my_statement(vcl_v1, viennacl::op_assign(), viennacl::linalg::element_div(vcl_v1, vcl_v2 + vcl_v1));
  viennacl::scheduler::execute(my_statement);

  if (check(std_v1, vcl_v1, epsilon) != EXIT_SUCCESS)
    return EXIT_FAILURE;
  }

  std::cout << "x = element_div(x - y, y + x)... ";
  {
  for (std::size_t i=0; i<std_v1.size(); ++i)
    std_v1[i] = (std_v1[i] - std_v2[i]) / (std_v2[i] + std_v1[i]);
  viennacl::scheduler::statement   my_statement(vcl_v1, viennacl::op_assign(), viennacl::linalg::element_div(vcl_v1 - vcl_v2, vcl_v2 + vcl_v1));
  viennacl::scheduler::execute(my_statement);

  if (check(std_v1, vcl_v1, epsilon) != EXIT_SUCCESS)
    return EXIT_FAILURE;
  }


  std::cout << "x = element_pow(x, y)... ";
  {
  for (std::size_t i=0; i<std_v1.size(); ++i)
  {
    std_v1[i] = NumericT(2.0) + randomNumber();
    std_v2[i] = NumericT(1.0) + randomNumber();
  }
  viennacl::copy(std_v1, vcl_v1);
  viennacl::copy(std_v2, vcl_v2);

  for (std::size_t i=0; i<std_v1.size(); ++i)
    std_v1[i] = std::pow(std_v1[i], std_v2[i]);
  viennacl::scheduler::statement   my_statement(vcl_v1, viennacl::op_assign(), viennacl::linalg::element_pow(vcl_v1, vcl_v2));
  viennacl::scheduler::execute(my_statement);

  if (check(std_v1, vcl_v1, epsilon) != EXIT_SUCCESS)
    return EXIT_FAILURE;
  }

  std::cout << "x = element_pow(x + y, y)... ";
  {
  for (std::size_t i=0; i<std_v1.size(); ++i)
  {
    std_v1[i] = NumericT(2.0) + randomNumber();
    std_v2[i] = NumericT(1.0) + randomNumber();
  }
  viennacl::copy(std_v1, vcl_v1);
  viennacl::copy(std_v2, vcl_v2);

  for (std::size_t i=0; i<std_v1.size(); ++i)
    std_v1[i] = std::pow(std_v1[i]  + std_v2[i], std_v2[i]);
  viennacl::scheduler::statement   my_statement(vcl_v1, viennacl::op_assign(), viennacl::linalg::element_pow(vcl_v1 + vcl_v2, vcl_v2));
  viennacl::scheduler::execute(my_statement);

  if (check(std_v1, vcl_v1, epsilon) != EXIT_SUCCESS)
    return EXIT_FAILURE;
  }

  std::cout << "x = element_pow(x, x + y)... ";
  {
  for (std::size_t i=0; i<std_v1.size(); ++i)
  {
    std_v1[i] = NumericT(2.0) + randomNumber();
    std_v2[i] = NumericT(1.0) + randomNumber();
  }
  viennacl::copy(std_v1, vcl_v1);
  viennacl::copy(std_v2, vcl_v2);

  for (std::size_t i=0; i<std_v1.size(); ++i)
    std_v1[i] = std::pow(std_v1[i], std_v1[i] + std_v2[i]);
  viennacl::scheduler::statement   my_statement(vcl_v1, viennacl::op_assign(), viennacl::linalg::element_pow(vcl_v1, vcl_v2 + vcl_v1));
  viennacl::scheduler::execute(my_statement);

  if (check(std_v1, vcl_v1, epsilon) != EXIT_SUCCESS)
    return EXIT_FAILURE;
  }

  std::cout << "x = element_pow(x - y, y + x)... ";
  {
  for (std::size_t i=0; i<std_v1.size(); ++i)
  {
    std_v1[i] = NumericT(2.0) + randomNumber();
    std_v2[i] = NumericT(1.0) + randomNumber();
  }
  viennacl::copy(std_v1, vcl_v1);
  viennacl::copy(std_v2, vcl_v2);

  for (std::size_t i=0; i<std_v1.size(); ++i)
    std_v1[i] = std::pow(std_v1[i] - std_v2[i], std_v2[i] + std_v1[i]);
  viennacl::scheduler::statement   my_statement(vcl_v1, viennacl::op_assign(), viennacl::linalg::element_pow(vcl_v1 - vcl_v2, vcl_v2 + vcl_v1));
  viennacl::scheduler::execute(my_statement);

  if (check(std_v1, vcl_v1, epsilon) != EXIT_SUCCESS)
    return EXIT_FAILURE;
  }

  std::cout << "--- Testing elementwise operations (unary) ---" << std::endl;
#define GENERATE_UNARY_OP_TEST(OPNAME) \
  std_v1 = std::vector<NumericT>(std_v1.size(), NumericT(0.21)); \
  for (std::size_t i=0; i<std_v1.size(); ++i) \
    std_v2[i] = NumericT(3.1415) * std_v1[i]; \
  viennacl::copy(std_v1.begin(), std_v1.end(), vcl_v1.begin()); \
  viennacl::copy(std_v2.begin(), std_v2.end(), vcl_v2.begin()); \
  { \
  for (std::size_t i=0; i<std_v1.size(); ++i) \
    std_v1[i] = std::OPNAME(std_v2[i]); \
  viennacl::scheduler::statement my_statement(vcl_v1, viennacl::op_assign(), viennacl::linalg::element_##OPNAME(vcl_v2)); \
  viennacl::scheduler::execute(my_statement); \
  if (check(std_v1, vcl_v1, epsilon) != EXIT_SUCCESS) \
    return EXIT_FAILURE; \
  } \
  { \
  for (std::size_t i=0; i<std_v1.size(); ++i) \
  std_v1[i] = std::OPNAME(std_v2[i] / NumericT(2)); \
  viennacl::scheduler::statement my_statement(vcl_v1, viennacl::op_assign(), viennacl::linalg::element_##OPNAME(vcl_v2 / NumericT(2))); \
  viennacl::scheduler::execute(my_statement); \
  if (check(std_v1, vcl_v1, epsilon) != EXIT_SUCCESS) \
    return EXIT_FAILURE; \
  }

  GENERATE_UNARY_OP_TEST(cos);
  GENERATE_UNARY_OP_TEST(cosh);
  GENERATE_UNARY_OP_TEST(exp);
  GENERATE_UNARY_OP_TEST(floor);
  GENERATE_UNARY_OP_TEST(fabs);
  GENERATE_UNARY_OP_TEST(log);
  GENERATE_UNARY_OP_TEST(log10);
  GENERATE_UNARY_OP_TEST(sin);
  GENERATE_UNARY_OP_TEST(sinh);
  GENERATE_UNARY_OP_TEST(fabs);
  //GENERATE_UNARY_OP_TEST(abs); //OpenCL allows abs on integers only
  GENERATE_UNARY_OP_TEST(sqrt);
  GENERATE_UNARY_OP_TEST(tan);
  GENERATE_UNARY_OP_TEST(tanh);

#undef GENERATE_UNARY_OP_TEST

  std::cout << "--- Testing complicated composite operations ---" << std::endl;
  std::cout << "x = inner_prod(x, y) * y..." << std::endl;
  {
  cpu_result = 0;
  for (std::size_t i=0; i<std_v1.size(); ++i)
    cpu_result += std_v1[i] * std_v2[i];
  for (std::size_t i=0; i<std_v1.size(); ++i)
    std_v1[i] = cpu_result * std_v2[i];
  viennacl::scheduler::statement   my_statement(vcl_v1, viennacl::op_assign(), viennacl::linalg::inner_prod(vcl_v1, vcl_v2) * vcl_v2);
  viennacl::scheduler::execute(my_statement);

  if (check(std_v1, vcl_v1, epsilon) != EXIT_SUCCESS)
    return EXIT_FAILURE;
  }

  std::cout << "x = y / norm_1(x)..." << std::endl;
  {
  cpu_result = 0;
  for (std::size_t i=0; i<std_v1.size(); ++i)
    cpu_result += std::fabs(std_v1[i]);
  for (std::size_t i=0; i<std_v1.size(); ++i)
    std_v1[i] = std_v2[i] / cpu_result;
  viennacl::scheduler::statement   my_statement(vcl_v1, viennacl::op_assign(), vcl_v2 / viennacl::linalg::norm_1(vcl_v1) );
  viennacl::scheduler::execute(my_statement);

  if (check(std_v1, vcl_v1, epsilon) != EXIT_SUCCESS)
    return EXIT_FAILURE;
  }


  // --------------------------------------------------------------------------
  return retval;
}


template< typename NumericT, typename Epsilon >
int test(Epsilon const& epsilon)
{
  int retval = EXIT_SUCCESS;
  std::size_t size = 24656;

  viennacl::tools::uniform_random_numbers<NumericT> randomNumber;

  std::cout << "Running tests for vector of size " << size << std::endl;

  //
  // Set up STL objects
  //
  std::vector<NumericT> std_full_vec(size);
  std::vector<NumericT> std_full_vec2(std_full_vec.size());

  for (std::size_t i=0; i<std_full_vec.size(); ++i)
  {
    std_full_vec[i]  = NumericT(1.0) + randomNumber();
    std_full_vec2[i] = NumericT(1.0) + randomNumber();
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
    if (check(std_short_vec, vcl_short_vec, epsilon) != EXIT_SUCCESS)
      return EXIT_FAILURE;
    if (check(std_short_vec2, vcl_short_vec2, epsilon) != EXIT_SUCCESS)
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
  if (check(std_short_vec, vcl_short_vec, epsilon) != EXIT_SUCCESS)
    return EXIT_FAILURE;
  if (check(std_short_vec2, vcl_short_vec2, epsilon) != EXIT_SUCCESS)
    return EXIT_FAILURE;


  //
  // Now start running tests for vectors, ranges and slices:
  //

  std::cout << " ** vcl_v1 = vector, vcl_v2 = vector **" << std::endl;
  retval = test<NumericT>(epsilon,
                          std_short_vec, std_short_vec2,
                          vcl_short_vec, vcl_short_vec2);
  if (retval != EXIT_SUCCESS)
    return EXIT_FAILURE;

  std::cout << " ** vcl_v1 = vector, vcl_v2 = range **" << std::endl;
  retval = test<NumericT>(epsilon,
                          std_short_vec, std_short_vec2,
                          vcl_short_vec, vcl_range_vec2);
  if (retval != EXIT_SUCCESS)
    return EXIT_FAILURE;

  std::cout << " ** vcl_v1 = vector, vcl_v2 = slice **" << std::endl;
  retval = test<NumericT>(epsilon,
                          std_short_vec, std_short_vec2,
                          vcl_short_vec, vcl_slice_vec2);
  if (retval != EXIT_SUCCESS)
    return EXIT_FAILURE;

  ///////

  std::cout << " ** vcl_v1 = range, vcl_v2 = vector **" << std::endl;
  retval = test<NumericT>(epsilon,
                          std_short_vec, std_short_vec2,
                          vcl_range_vec, vcl_short_vec2);
  if (retval != EXIT_SUCCESS)
    return EXIT_FAILURE;

  std::cout << " ** vcl_v1 = range, vcl_v2 = range **" << std::endl;
  retval = test<NumericT>(epsilon,
                          std_short_vec, std_short_vec2,
                          vcl_range_vec, vcl_range_vec2);
  if (retval != EXIT_SUCCESS)
    return EXIT_FAILURE;

  std::cout << " ** vcl_v1 = range, vcl_v2 = slice **" << std::endl;
  retval = test<NumericT>(epsilon,
                          std_short_vec, std_short_vec2,
                          vcl_range_vec, vcl_slice_vec2);
  if (retval != EXIT_SUCCESS)
    return EXIT_FAILURE;

  ///////

  std::cout << " ** vcl_v1 = slice, vcl_v2 = vector **" << std::endl;
  retval = test<NumericT>(epsilon,
                          std_short_vec, std_short_vec2,
                          vcl_slice_vec, vcl_short_vec2);
  if (retval != EXIT_SUCCESS)
    return EXIT_FAILURE;

  std::cout << " ** vcl_v1 = slice, vcl_v2 = range **" << std::endl;
  retval = test<NumericT>(epsilon,
                          std_short_vec, std_short_vec2,
                          vcl_slice_vec, vcl_range_vec2);
  if (retval != EXIT_SUCCESS)
    return EXIT_FAILURE;

  std::cout << " ** vcl_v1 = slice, vcl_v2 = slice **" << std::endl;
  retval = test<NumericT>(epsilon,
                          std_short_vec, std_short_vec2,
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
   std::cout << "## Test :: Vector" << std::endl;
   std::cout << "----------------------------------------------" << std::endl;
   std::cout << "----------------------------------------------" << std::endl;
   std::cout << std::endl;

   int retval = EXIT_SUCCESS;

   std::cout << std::endl;
   std::cout << "----------------------------------------------" << std::endl;
   std::cout << std::endl;
   {
      typedef float NumericT;
      NumericT epsilon = static_cast<NumericT>(1.0E-4);
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

  std::cout << std::endl;
  std::cout << "------- Test completed --------" << std::endl;
  std::cout << std::endl;


   return retval;
}
