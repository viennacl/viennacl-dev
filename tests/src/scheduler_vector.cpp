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



/** \file tests/src/scheduler_vector.cpp  Tests the scheduler for vector-operations.
*   \test Tests the scheduler for vector-operations.
**/

//
// *** System
//
#include <iostream>
#include <iomanip>

//
// *** Boost
//
#include <boost/numeric/ublas/io.hpp>
#include <boost/numeric/ublas/vector.hpp>
#include <boost/numeric/ublas/vector_proxy.hpp>

//
// *** ViennaCL
//
//#define VIENNACL_DEBUG_ALL
#define VIENNACL_WITH_UBLAS 1
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

using namespace boost::numeric;


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
ScalarType diff(ublas::vector<ScalarType> const & v1, ViennaCLVectorType const & vcl_vec)
{
   ublas::vector<ScalarType> v2_cpu(vcl_vec.size());
   viennacl::backend::finish();
   viennacl::copy(vcl_vec, v2_cpu);

   for (unsigned int i=0;i<v1.size(); ++i)
   {
      if ( std::max( std::fabs(v2_cpu[i]), std::fabs(v1[i]) ) > 0 )
         v2_cpu[i] = std::fabs(v2_cpu[i] - v1[i]) / std::max( std::fabs(v2_cpu[i]), std::fabs(v1[i]) );
      else
         v2_cpu[i] = 0.0;
   }

   return ublas::norm_inf(v2_cpu);
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
template< typename NumericT, typename Epsilon, typename UblasVectorType, typename ViennaCLVectorType1, typename ViennaCLVectorType2 >
int test(Epsilon const& epsilon,
         UblasVectorType     & ublas_v1, UblasVectorType     & ublas_v2,
         ViennaCLVectorType1 &   vcl_v1, ViennaCLVectorType2 &   vcl_v2)
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
  ublas_v1 = ublas::zero_vector<NumericT>(ublas_v1.size());
  vcl_v1 = viennacl::zero_vector<NumericT>(vcl_v1.size());
  if (check(ublas_v1, vcl_v1, epsilon) != EXIT_SUCCESS)
    return EXIT_FAILURE;

  std::cout << "Checking for scalar_vector initializer..." << std::endl;
  ublas_v1 = ublas::scalar_vector<NumericT>(ublas_v1.size(), cpu_result);
  vcl_v1 = viennacl::scalar_vector<NumericT>(vcl_v1.size(), cpu_result);
  if (check(ublas_v1, vcl_v1, epsilon) != EXIT_SUCCESS)
    return EXIT_FAILURE;

  ublas_v1 = ublas::scalar_vector<NumericT>(ublas_v1.size(), gpu_result);
  vcl_v1 = viennacl::scalar_vector<NumericT>(vcl_v1.size(), gpu_result);
  if (check(ublas_v1, vcl_v1, epsilon) != EXIT_SUCCESS)
    return EXIT_FAILURE;

  std::cout << "Checking for unit_vector initializer..." << std::endl;
  ublas_v1 = ublas::unit_vector<NumericT>(ublas_v1.size(), 5);
  vcl_v1 = viennacl::unit_vector<NumericT>(vcl_v1.size(), 5);
  if (check(ublas_v1, vcl_v1, epsilon) != EXIT_SUCCESS)
    return EXIT_FAILURE;


  for (std::size_t i=0; i<ublas_v1.size(); ++i)
  {
    ublas_v1[i] = NumericT(1.0) + randomNumber();
    ublas_v2[i] = NumericT(1.0) + randomNumber();
  }

  viennacl::copy(ublas_v1.begin(), ublas_v1.end(), vcl_v1.begin());  //resync
  viennacl::copy(ublas_v2.begin(), ublas_v2.end(), vcl_v2.begin());

  std::cout << "Checking for successful copy..." << std::endl;
  if (check(ublas_v1, vcl_v1, epsilon) != EXIT_SUCCESS)
    return EXIT_FAILURE;
  if (check(ublas_v2, vcl_v2, epsilon) != EXIT_SUCCESS)
    return EXIT_FAILURE;


  // --------------------------------------------------------------------------

  std::cout << "Testing simple assignments..." << std::endl;

  {
  ublas_v1 = ublas_v2;
  viennacl::scheduler::statement   my_statement(vcl_v1, viennacl::op_assign(), vcl_v2); // same as vcl_v1 = vcl_v2;
  viennacl::scheduler::execute(my_statement);

  if (check(ublas_v1, vcl_v1, epsilon) != EXIT_SUCCESS)
    return EXIT_FAILURE;
  }

  {
  ublas_v1 += ublas_v2;
  viennacl::scheduler::statement   my_statement(vcl_v1, viennacl::op_inplace_add(), vcl_v2); // same as vcl_v1 += vcl_v2;
  viennacl::scheduler::execute(my_statement);

  if (check(ublas_v1, vcl_v1, epsilon) != EXIT_SUCCESS)
    return EXIT_FAILURE;
  }

  {
  ublas_v1 -= ublas_v2;
  viennacl::scheduler::statement   my_statement(vcl_v1, viennacl::op_inplace_sub(), vcl_v2); // same as vcl_v1 -= vcl_v2;
  viennacl::scheduler::execute(my_statement);

  if (check(ublas_v1, vcl_v1, epsilon) != EXIT_SUCCESS)
    return EXIT_FAILURE;
  }

  std::cout << "Testing composite assignments..." << std::endl;
  {
  ublas_v1 = ublas_v1 + ublas_v2;
  viennacl::scheduler::statement   my_statement(vcl_v1, viennacl::op_assign(), vcl_v1 + vcl_v2); // same as vcl_v1 = vcl_v1 + vcl_v2;
  viennacl::scheduler::execute(my_statement);

  if (check(ublas_v1, vcl_v1, epsilon) != EXIT_SUCCESS)
    return EXIT_FAILURE;
  }
  {
  ublas_v1 += alpha * ublas_v1 - beta * ublas_v2 + ublas_v1 / beta - ublas_v2 / alpha;
  viennacl::scheduler::statement   my_statement(vcl_v1, viennacl::op_inplace_add(), alpha * vcl_v1 - beta * vcl_v2 + vcl_v1 / beta - vcl_v2 / alpha); // same as vcl_v1 += alpha * vcl_v1 - beta * vcl_v2 + beta * vcl_v1 - alpha * vcl_v2;
  viennacl::scheduler::execute(my_statement);

  if (check(ublas_v1, vcl_v1, epsilon) != EXIT_SUCCESS)
    return EXIT_FAILURE;
  }

  {
  ublas_v1 = ublas_v1 - ublas_v2;
  viennacl::scheduler::statement   my_statement(vcl_v1, viennacl::op_assign(), vcl_v1 - vcl_v2); // same as vcl_v1 = vcl_v1 - vcl_v2;
  viennacl::scheduler::execute(my_statement);

  if (check(ublas_v1, vcl_v1, epsilon) != EXIT_SUCCESS)
    return EXIT_FAILURE;
  }

  std::cout << "--- Testing reductions ---" << std::endl;
  std::cout << "inner_prod..." << std::endl;
  {
  cpu_result = inner_prod(ublas_v1, ublas_v2);
  viennacl::scheduler::statement   my_statement(gpu_result, viennacl::op_assign(), viennacl::linalg::inner_prod(vcl_v1, vcl_v2)); // same as gpu_result = inner_prod(vcl_v1, vcl_v2);
  viennacl::scheduler::execute(my_statement);

  if (check(cpu_result, gpu_result, epsilon) != EXIT_SUCCESS)
    return EXIT_FAILURE;
  }

  {
  cpu_result = inner_prod(ublas_v1 + ublas_v2, ublas_v2);
  viennacl::scheduler::statement   my_statement(gpu_result, viennacl::op_assign(), viennacl::linalg::inner_prod(vcl_v1 + vcl_v2, vcl_v2)); // same as gpu_result = inner_prod(vcl_v1 + vcl_v2, vcl_v2);
  viennacl::scheduler::execute(my_statement);

  if (check(cpu_result, gpu_result, epsilon) != EXIT_SUCCESS)
    return EXIT_FAILURE;
  }

  {
  cpu_result = inner_prod(ublas_v1, ublas_v2 - ublas_v1);
  viennacl::scheduler::statement   my_statement(gpu_result, viennacl::op_assign(), viennacl::linalg::inner_prod(vcl_v1, vcl_v2 - vcl_v1)); // same as gpu_result = inner_prod(vcl_v1, vcl_v2 - vcl_v1);
  viennacl::scheduler::execute(my_statement);

  if (check(cpu_result, gpu_result, epsilon) != EXIT_SUCCESS)
    return EXIT_FAILURE;
  }

  {
  cpu_result = inner_prod(ublas_v1 - ublas_v2, ublas_v2 + ublas_v1);
  viennacl::scheduler::statement   my_statement(gpu_result, viennacl::op_assign(), viennacl::linalg::inner_prod(vcl_v1 - vcl_v2, vcl_v2 + vcl_v1)); // same as gpu_result = inner_prod(vcl_v1 - vcl_v2, vcl_v2 + vcl_v1);
  viennacl::scheduler::execute(my_statement);

  if (check(cpu_result, gpu_result, epsilon) != EXIT_SUCCESS)
    return EXIT_FAILURE;
  }

  std::cout << "norm_1..." << std::endl;
  {
  cpu_result = norm_1(ublas_v1);
  viennacl::scheduler::statement   my_statement(gpu_result, viennacl::op_assign(), viennacl::linalg::norm_1(vcl_v1)); // same as gpu_result = norm_1(vcl_v1);
  viennacl::scheduler::execute(my_statement);

  if (check(cpu_result, gpu_result, epsilon) != EXIT_SUCCESS)
    return EXIT_FAILURE;
  }

  {
  cpu_result = norm_1(ublas_v1 + ublas_v2);
  viennacl::scheduler::statement   my_statement(gpu_result, viennacl::op_assign(), viennacl::linalg::norm_1(vcl_v1 + vcl_v2)); // same as gpu_result = norm_1(vcl_v1 + vcl_v2);
  viennacl::scheduler::execute(my_statement);

  if (check(cpu_result, gpu_result, epsilon) != EXIT_SUCCESS)
    return EXIT_FAILURE;
  }

  std::cout << "norm_2..." << std::endl;
  {
  cpu_result = norm_2(ublas_v1);
  viennacl::scheduler::statement   my_statement(gpu_result, viennacl::op_assign(), viennacl::linalg::norm_2(vcl_v1)); // same as gpu_result = norm_2(vcl_v1);
  viennacl::scheduler::execute(my_statement);

  if (check(cpu_result, gpu_result, epsilon) != EXIT_SUCCESS)
    return EXIT_FAILURE;
  }

  {
  cpu_result = norm_2(ublas_v1 + ublas_v2);
  viennacl::scheduler::statement   my_statement(gpu_result, viennacl::op_assign(), viennacl::linalg::norm_2(vcl_v1 + vcl_v2)); // same as gpu_result = norm_2(vcl_v1 + vcl_v2);
  viennacl::scheduler::execute(my_statement);

  if (check(cpu_result, gpu_result, epsilon) != EXIT_SUCCESS)
    return EXIT_FAILURE;
  }

  std::cout << "norm_inf..." << std::endl;
  {
  cpu_result = norm_inf(ublas_v1);
  viennacl::scheduler::statement   my_statement(gpu_result, viennacl::op_assign(), viennacl::linalg::norm_inf(vcl_v1)); // same as gpu_result = norm_inf(vcl_v1);
  viennacl::scheduler::execute(my_statement);

  if (check(cpu_result, gpu_result, epsilon) != EXIT_SUCCESS)
    return EXIT_FAILURE;
  }

  {
  cpu_result = norm_inf(ublas_v1 - ublas_v2);
  viennacl::scheduler::statement   my_statement(gpu_result, viennacl::op_assign(), viennacl::linalg::norm_inf(vcl_v1 - vcl_v2)); // same as gpu_result = norm_inf(vcl_v1 - vcl_v2);
  viennacl::scheduler::execute(my_statement);

  if (check(cpu_result, gpu_result, epsilon) != EXIT_SUCCESS)
    return EXIT_FAILURE;
  }

  std::cout << "--- Testing elementwise operations (binary) ---" << std::endl;
  std::cout << "x = element_prod(x, y)... ";
  {
  ublas_v1 = element_prod(ublas_v1, ublas_v2);
  viennacl::scheduler::statement   my_statement(vcl_v1, viennacl::op_assign(), viennacl::linalg::element_prod(vcl_v1, vcl_v2));
  viennacl::scheduler::execute(my_statement);

  if (check(ublas_v1, vcl_v1, epsilon) != EXIT_SUCCESS)
    return EXIT_FAILURE;
  }

  std::cout << "x = element_prod(x + y, y)... ";
  {
  ublas_v1 = element_prod(ublas_v1 + ublas_v2, ublas_v2);
  viennacl::scheduler::statement   my_statement(vcl_v1, viennacl::op_assign(), viennacl::linalg::element_prod(vcl_v1 + vcl_v2, vcl_v2));
  viennacl::scheduler::execute(my_statement);

  if (check(ublas_v1, vcl_v1, epsilon) != EXIT_SUCCESS)
    return EXIT_FAILURE;
  }

  std::cout << "x = element_prod(x, x + y)... ";
  {
  ublas_v1 = element_prod(ublas_v1, ublas_v1 + ublas_v2);
  viennacl::scheduler::statement   my_statement(vcl_v1, viennacl::op_assign(), viennacl::linalg::element_prod(vcl_v1, vcl_v2 + vcl_v1));
  viennacl::scheduler::execute(my_statement);

  if (check(ublas_v1, vcl_v1, epsilon) != EXIT_SUCCESS)
    return EXIT_FAILURE;
  }

  std::cout << "x = element_prod(x - y, y + x)... ";
  {
  ublas_v1 = element_prod(ublas_v1 - ublas_v2, ublas_v2 + ublas_v1);
  viennacl::scheduler::statement   my_statement(vcl_v1, viennacl::op_assign(), viennacl::linalg::element_prod(vcl_v1 - vcl_v2, vcl_v2 + vcl_v1));
  viennacl::scheduler::execute(my_statement);

  if (check(ublas_v1, vcl_v1, epsilon) != EXIT_SUCCESS)
    return EXIT_FAILURE;
  }



  std::cout << "x = element_div(x, y)... ";
  {
  ublas_v1 = element_div(ublas_v1, ublas_v2);
  viennacl::scheduler::statement   my_statement(vcl_v1, viennacl::op_assign(), viennacl::linalg::element_div(vcl_v1, vcl_v2));
  viennacl::scheduler::execute(my_statement);

  if (check(ublas_v1, vcl_v1, epsilon) != EXIT_SUCCESS)
    return EXIT_FAILURE;
  }

  std::cout << "x = element_div(x + y, y)... ";
  {
  ublas_v1 = element_div(ublas_v1 + ublas_v2, ublas_v2);
  viennacl::scheduler::statement   my_statement(vcl_v1, viennacl::op_assign(), viennacl::linalg::element_div(vcl_v1 + vcl_v2, vcl_v2));
  viennacl::scheduler::execute(my_statement);

  if (check(ublas_v1, vcl_v1, epsilon) != EXIT_SUCCESS)
    return EXIT_FAILURE;
  }

  std::cout << "x = element_div(x, x + y)... ";
  {
  ublas_v1 = element_div(ublas_v1, ublas_v1 + ublas_v2);
  viennacl::scheduler::statement   my_statement(vcl_v1, viennacl::op_assign(), viennacl::linalg::element_div(vcl_v1, vcl_v2 + vcl_v1));
  viennacl::scheduler::execute(my_statement);

  if (check(ublas_v1, vcl_v1, epsilon) != EXIT_SUCCESS)
    return EXIT_FAILURE;
  }

  std::cout << "x = element_div(x - y, y + x)... ";
  {
  ublas_v1 = element_div(ublas_v1 - ublas_v2, ublas_v2 + ublas_v1);
  viennacl::scheduler::statement   my_statement(vcl_v1, viennacl::op_assign(), viennacl::linalg::element_div(vcl_v1 - vcl_v2, vcl_v2 + vcl_v1));
  viennacl::scheduler::execute(my_statement);

  if (check(ublas_v1, vcl_v1, epsilon) != EXIT_SUCCESS)
    return EXIT_FAILURE;
  }


  std::cout << "x = element_pow(x, y)... ";
  {
  for (std::size_t i=0; i<ublas_v1.size(); ++i)
  {
    ublas_v1[i] = NumericT(2.0) + randomNumber();
    ublas_v2[i] = NumericT(1.0) + randomNumber();
  }
  viennacl::copy(ublas_v1, vcl_v1);
  viennacl::copy(ublas_v2, vcl_v2);

  for (std::size_t i=0; i<ublas_v1.size(); ++i)
    ublas_v1[i] = std::pow(ublas_v1[i], ublas_v2[i]);
  viennacl::scheduler::statement   my_statement(vcl_v1, viennacl::op_assign(), viennacl::linalg::element_pow(vcl_v1, vcl_v2));
  viennacl::scheduler::execute(my_statement);

  if (check(ublas_v1, vcl_v1, epsilon) != EXIT_SUCCESS)
    return EXIT_FAILURE;
  }

  std::cout << "x = element_pow(x + y, y)... ";
  {
  for (std::size_t i=0; i<ublas_v1.size(); ++i)
  {
    ublas_v1[i] = NumericT(2.0) + randomNumber();
    ublas_v2[i] = NumericT(1.0) + randomNumber();
  }
  viennacl::copy(ublas_v1, vcl_v1);
  viennacl::copy(ublas_v2, vcl_v2);

  for (std::size_t i=0; i<ublas_v1.size(); ++i)
    ublas_v1[i] = std::pow(ublas_v1[i]  + ublas_v2[i], ublas_v2[i]);
  viennacl::scheduler::statement   my_statement(vcl_v1, viennacl::op_assign(), viennacl::linalg::element_pow(vcl_v1 + vcl_v2, vcl_v2));
  viennacl::scheduler::execute(my_statement);

  if (check(ublas_v1, vcl_v1, epsilon) != EXIT_SUCCESS)
    return EXIT_FAILURE;
  }

  std::cout << "x = element_pow(x, x + y)... ";
  {
  for (std::size_t i=0; i<ublas_v1.size(); ++i)
  {
    ublas_v1[i] = NumericT(2.0) + randomNumber();
    ublas_v2[i] = NumericT(1.0) + randomNumber();
  }
  viennacl::copy(ublas_v1, vcl_v1);
  viennacl::copy(ublas_v2, vcl_v2);

  for (std::size_t i=0; i<ublas_v1.size(); ++i)
    ublas_v1[i] = std::pow(ublas_v1[i], ublas_v1[i] + ublas_v2[i]);
  viennacl::scheduler::statement   my_statement(vcl_v1, viennacl::op_assign(), viennacl::linalg::element_pow(vcl_v1, vcl_v2 + vcl_v1));
  viennacl::scheduler::execute(my_statement);

  if (check(ublas_v1, vcl_v1, epsilon) != EXIT_SUCCESS)
    return EXIT_FAILURE;
  }

  std::cout << "x = element_pow(x - y, y + x)... ";
  {
  for (std::size_t i=0; i<ublas_v1.size(); ++i)
  {
    ublas_v1[i] = NumericT(2.0) + randomNumber();
    ublas_v2[i] = NumericT(1.0) + randomNumber();
  }
  viennacl::copy(ublas_v1, vcl_v1);
  viennacl::copy(ublas_v2, vcl_v2);

  for (std::size_t i=0; i<ublas_v1.size(); ++i)
    ublas_v1[i] = std::pow(ublas_v1[i] - ublas_v2[i], ublas_v2[i] + ublas_v1[i]);
  viennacl::scheduler::statement   my_statement(vcl_v1, viennacl::op_assign(), viennacl::linalg::element_pow(vcl_v1 - vcl_v2, vcl_v2 + vcl_v1));
  viennacl::scheduler::execute(my_statement);

  if (check(ublas_v1, vcl_v1, epsilon) != EXIT_SUCCESS)
    return EXIT_FAILURE;
  }

  std::cout << "--- Testing elementwise operations (unary) ---" << std::endl;
#define GENERATE_UNARY_OP_TEST(OPNAME) \
  ublas_v1 = ublas::scalar_vector<NumericT>(ublas_v1.size(), NumericT(0.21)); \
  ublas_v2 = NumericT(3.1415) * ublas_v1; \
  viennacl::copy(ublas_v1.begin(), ublas_v1.end(), vcl_v1.begin()); \
  viennacl::copy(ublas_v2.begin(), ublas_v2.end(), vcl_v2.begin()); \
  { \
  for (std::size_t i=0; i<ublas_v1.size(); ++i) \
    ublas_v1[i] = std::OPNAME(ublas_v2[i]); \
  viennacl::scheduler::statement my_statement(vcl_v1, viennacl::op_assign(), viennacl::linalg::element_##OPNAME(vcl_v2)); \
  viennacl::scheduler::execute(my_statement); \
  if (check(ublas_v1, vcl_v1, epsilon) != EXIT_SUCCESS) \
    return EXIT_FAILURE; \
  } \
  { \
  for (std::size_t i=0; i<ublas_v1.size(); ++i) \
  ublas_v1[i] = std::OPNAME(ublas_v2[i] / NumericT(2)); \
  viennacl::scheduler::statement my_statement(vcl_v1, viennacl::op_assign(), viennacl::linalg::element_##OPNAME(vcl_v2 / NumericT(2))); \
  viennacl::scheduler::execute(my_statement); \
  if (check(ublas_v1, vcl_v1, epsilon) != EXIT_SUCCESS) \
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
  ublas_v1 = inner_prod(ublas_v1, ublas_v2) * ublas_v2;
  viennacl::scheduler::statement   my_statement(vcl_v1, viennacl::op_assign(), viennacl::linalg::inner_prod(vcl_v1, vcl_v2) * vcl_v2);
  viennacl::scheduler::execute(my_statement);

  if (check(ublas_v1, vcl_v1, epsilon) != EXIT_SUCCESS)
    return EXIT_FAILURE;
  }

  std::cout << "x = y / norm_1(x)..." << std::endl;
  {
  ublas_v1 = ublas_v2 / norm_1(ublas_v1);
  viennacl::scheduler::statement   my_statement(vcl_v1, viennacl::op_assign(), vcl_v2 / viennacl::linalg::norm_1(vcl_v1) );
  viennacl::scheduler::execute(my_statement);

  if (check(ublas_v1, vcl_v1, epsilon) != EXIT_SUCCESS)
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
  // Set up UBLAS objects
  //
  ublas::vector<NumericT> ublas_full_vec(size);
  ublas::vector<NumericT> ublas_full_vec2(ublas_full_vec.size());

  for (std::size_t i=0; i<ublas_full_vec.size(); ++i)
  {
    ublas_full_vec[i]  = NumericT(1.0) + randomNumber();
    ublas_full_vec2[i] = NumericT(1.0) + randomNumber();
  }

  ublas::range r1(    ublas_full_vec.size() / 4, 2 * ublas_full_vec.size() / 4);
  ublas::range r2(2 * ublas_full_vec2.size() / 4, 3 * ublas_full_vec2.size() / 4);
  ublas::vector_range< ublas::vector<NumericT> > ublas_range_vec(ublas_full_vec, r1);
  ublas::vector_range< ublas::vector<NumericT> > ublas_range_vec2(ublas_full_vec2, r2);

  ublas::slice s1(    ublas_full_vec.size() / 4, 3, ublas_full_vec.size() / 4);
  ublas::slice s2(2 * ublas_full_vec2.size() / 4, 2, ublas_full_vec2.size() / 4);
  ublas::vector_slice< ublas::vector<NumericT> > ublas_slice_vec(ublas_full_vec, s1);
  ublas::vector_slice< ublas::vector<NumericT> > ublas_slice_vec2(ublas_full_vec2, s2);

  //
  // Set up ViennaCL objects
  //
  viennacl::vector<NumericT> vcl_full_vec(ublas_full_vec.size());
  viennacl::vector<NumericT> vcl_full_vec2(ublas_full_vec2.size());

  viennacl::fast_copy(ublas_full_vec.begin(), ublas_full_vec.end(), vcl_full_vec.begin());
  viennacl::copy(ublas_full_vec2.begin(), ublas_full_vec2.end(), vcl_full_vec2.begin());

  viennacl::range vcl_r1(    vcl_full_vec.size() / 4, 2 * vcl_full_vec.size() / 4);
  viennacl::range vcl_r2(2 * vcl_full_vec2.size() / 4, 3 * vcl_full_vec2.size() / 4);
  viennacl::vector_range< viennacl::vector<NumericT> > vcl_range_vec(vcl_full_vec, vcl_r1);
  viennacl::vector_range< viennacl::vector<NumericT> > vcl_range_vec2(vcl_full_vec2, vcl_r2);

  {
    viennacl::vector<NumericT> vcl_short_vec(vcl_range_vec);
    viennacl::vector<NumericT> vcl_short_vec2 = vcl_range_vec2;

    ublas::vector<NumericT> ublas_short_vec(ublas_range_vec);
    ublas::vector<NumericT> ublas_short_vec2(ublas_range_vec2);

    std::cout << "Testing creation of vectors from range..." << std::endl;
    if (check(ublas_short_vec, vcl_short_vec, epsilon) != EXIT_SUCCESS)
      return EXIT_FAILURE;
    if (check(ublas_short_vec2, vcl_short_vec2, epsilon) != EXIT_SUCCESS)
      return EXIT_FAILURE;
  }

  viennacl::slice vcl_s1(    vcl_full_vec.size() / 4, 3, vcl_full_vec.size() / 4);
  viennacl::slice vcl_s2(2 * vcl_full_vec2.size() / 4, 2, vcl_full_vec2.size() / 4);
  viennacl::vector_slice< viennacl::vector<NumericT> > vcl_slice_vec(vcl_full_vec, vcl_s1);
  viennacl::vector_slice< viennacl::vector<NumericT> > vcl_slice_vec2(vcl_full_vec2, vcl_s2);

  viennacl::vector<NumericT> vcl_short_vec(vcl_slice_vec);
  viennacl::vector<NumericT> vcl_short_vec2 = vcl_slice_vec2;

  ublas::vector<NumericT> ublas_short_vec(ublas_slice_vec);
  ublas::vector<NumericT> ublas_short_vec2(ublas_slice_vec2);

  std::cout << "Testing creation of vectors from slice..." << std::endl;
  if (check(ublas_short_vec, vcl_short_vec, epsilon) != EXIT_SUCCESS)
    return EXIT_FAILURE;
  if (check(ublas_short_vec2, vcl_short_vec2, epsilon) != EXIT_SUCCESS)
    return EXIT_FAILURE;


  //
  // Now start running tests for vectors, ranges and slices:
  //

  std::cout << " ** vcl_v1 = vector, vcl_v2 = vector **" << std::endl;
  retval = test<NumericT>(epsilon,
                          ublas_short_vec, ublas_short_vec2,
                          vcl_short_vec, vcl_short_vec2);
  if (retval != EXIT_SUCCESS)
    return EXIT_FAILURE;

  std::cout << " ** vcl_v1 = vector, vcl_v2 = range **" << std::endl;
  retval = test<NumericT>(epsilon,
                          ublas_short_vec, ublas_short_vec2,
                          vcl_short_vec, vcl_range_vec2);
  if (retval != EXIT_SUCCESS)
    return EXIT_FAILURE;

  std::cout << " ** vcl_v1 = vector, vcl_v2 = slice **" << std::endl;
  retval = test<NumericT>(epsilon,
                          ublas_short_vec, ublas_short_vec2,
                          vcl_short_vec, vcl_slice_vec2);
  if (retval != EXIT_SUCCESS)
    return EXIT_FAILURE;

  ///////

  std::cout << " ** vcl_v1 = range, vcl_v2 = vector **" << std::endl;
  retval = test<NumericT>(epsilon,
                          ublas_short_vec, ublas_short_vec2,
                          vcl_range_vec, vcl_short_vec2);
  if (retval != EXIT_SUCCESS)
    return EXIT_FAILURE;

  std::cout << " ** vcl_v1 = range, vcl_v2 = range **" << std::endl;
  retval = test<NumericT>(epsilon,
                          ublas_short_vec, ublas_short_vec2,
                          vcl_range_vec, vcl_range_vec2);
  if (retval != EXIT_SUCCESS)
    return EXIT_FAILURE;

  std::cout << " ** vcl_v1 = range, vcl_v2 = slice **" << std::endl;
  retval = test<NumericT>(epsilon,
                          ublas_short_vec, ublas_short_vec2,
                          vcl_range_vec, vcl_slice_vec2);
  if (retval != EXIT_SUCCESS)
    return EXIT_FAILURE;

  ///////

  std::cout << " ** vcl_v1 = slice, vcl_v2 = vector **" << std::endl;
  retval = test<NumericT>(epsilon,
                          ublas_short_vec, ublas_short_vec2,
                          vcl_slice_vec, vcl_short_vec2);
  if (retval != EXIT_SUCCESS)
    return EXIT_FAILURE;

  std::cout << " ** vcl_v1 = slice, vcl_v2 = range **" << std::endl;
  retval = test<NumericT>(epsilon,
                          ublas_short_vec, ublas_short_vec2,
                          vcl_slice_vec, vcl_range_vec2);
  if (retval != EXIT_SUCCESS)
    return EXIT_FAILURE;

  std::cout << " ** vcl_v1 = slice, vcl_v2 = slice **" << std::endl;
  retval = test<NumericT>(epsilon,
                          ublas_short_vec, ublas_short_vec2,
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
