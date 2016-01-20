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


/** \file tests/src/vector_float_double.cpp  Tests vector operations (BLAS level 1) for floating point arithmetic.
*   \test  Tests vector operations (BLAS level 1) for floating point arithmetic.
**/

//
// *** System
//
#include <iostream>
#include <iomanip>
#include <cmath>

//
// *** ViennaCL
//
//#define VIENNACL_DEBUG_ALL

#include "viennacl/vector.hpp"
#include "viennacl/vector_proxy.hpp"
#include "viennacl/linalg/inner_prod.hpp"
#include "viennacl/linalg/norm_1.hpp"
#include "viennacl/linalg/norm_2.hpp"
#include "viennacl/linalg/norm_inf.hpp"
#include "viennacl/linalg/maxmin.hpp"
#include "viennacl/linalg/sum.hpp"

#include "viennacl/tools/random.hpp"


template<typename NumericT>
class vector_proxy
{
public:
  vector_proxy(NumericT * p_values, std::size_t start_idx, std::size_t increment, std::size_t num_elements)
    : values_(p_values), start_(start_idx), inc_(increment), size_(num_elements) {}

  NumericT const & operator[](std::size_t index) const { return values_[start_ + index * inc_]; }
  NumericT       & operator[](std::size_t index)       { return values_[start_ + index * inc_]; }

  std::size_t size() const { return size_; }

private:
  NumericT * values_;
  std::size_t start_;
  std::size_t inc_;
  std::size_t size_;
};

template<typename NumericT>
void proxy_copy(vector_proxy<NumericT> const & host_vec, viennacl::vector_base<NumericT> & vcl_vec)
{
  std::vector<NumericT> std_vec(host_vec.size());

  for (std::size_t i=0; i<host_vec.size(); ++i)
    std_vec[i] = host_vec[i];

  viennacl::copy(std_vec.begin(), std_vec.end(), vcl_vec.begin());
}

template<typename NumericT>
void proxy_copy(viennacl::vector_base<NumericT> const & vcl_vec, vector_proxy<NumericT> & host_vec)
{
  std::vector<NumericT> std_vec(vcl_vec.size());

  viennacl::copy(vcl_vec.begin(), vcl_vec.end(), std_vec.begin());

  for (std::size_t i=0; i<host_vec.size(); ++i)
    host_vec[i] = std_vec[i];
}


//
// -------------------------------------------------------------
//
template<typename ScalarType>
ScalarType diff(ScalarType const & s1, ScalarType const & s2)
{
   viennacl::backend::finish();
   if (std::fabs(s1 - s2) > 0 )
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
   if (std::fabs(s1 - s2) > 0 )
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
   if (std::fabs(s1 - s2) > 0 )
      return (s1 - s2) / std::max(std::fabs(s1), std::fabs(s2));
   return 0;
}
//
// -------------------------------------------------------------
//
template<typename ScalarType, typename ViennaCLVectorType>
ScalarType diff(vector_proxy<ScalarType> const & v1, ViennaCLVectorType const & vcl_vec)
{
   std::vector<ScalarType> v2_cpu(vcl_vec.size());
   viennacl::backend::finish();
   viennacl::copy(vcl_vec, v2_cpu);

   for (unsigned int i=0;i<v1.size(); ++i)
   {
      if ( std::max( std::fabs(v2_cpu[i]), std::fabs(v1[i]) ) > 0 )
         v2_cpu[i] = std::fabs(v2_cpu[i] - v1[i]) / std::max( std::fabs(v2_cpu[i]), std::fabs(v1[i]) );
      else
         v2_cpu[i] = 0.0;
   }

   ScalarType ret = 0;
   for (std::size_t i=0; i<v2_cpu.size(); ++i)
     ret = std::max(ret, std::fabs(v2_cpu[i]));
   return ret;
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
  return retval;
}


//
// -------------------------------------------------------------
//
template< typename NumericT, typename Epsilon, typename HostVectorType, typename ViennaCLVectorType1, typename ViennaCLVectorType2 >
int test(Epsilon const& epsilon,
         HostVectorType      &  host_v1, HostVectorType      &  host_v2,
         ViennaCLVectorType1 &   vcl_v1, ViennaCLVectorType2 &   vcl_v2)
{
  int retval = EXIT_SUCCESS;

  viennacl::tools::uniform_random_numbers<NumericT> randomNumber;

  NumericT                    cpu_result = NumericT(42.0);
  viennacl::scalar<NumericT>  gpu_result = NumericT(43.0);

  //
  // Initializer:
  //
  std::cout << "Checking for zero_vector initializer..." << std::endl;
  for (std::size_t i=0; i<host_v1.size(); ++i)
    host_v1[i] = NumericT(0);
  vcl_v1 = viennacl::zero_vector<NumericT>(vcl_v1.size());
  if (check(host_v1, vcl_v1, epsilon) != EXIT_SUCCESS)
    return EXIT_FAILURE;

  std::cout << "Checking for scalar_vector initializer..." << std::endl;
  for (std::size_t i=0; i<host_v1.size(); ++i)
    host_v1[i] = NumericT(cpu_result);
  vcl_v1 = viennacl::scalar_vector<NumericT>(vcl_v1.size(), cpu_result);
  if (check(host_v1, vcl_v1, epsilon) != EXIT_SUCCESS)
    return EXIT_FAILURE;

  for (std::size_t i=0; i<host_v1.size(); ++i)
    host_v1[i] = NumericT(gpu_result);
  vcl_v1 = viennacl::scalar_vector<NumericT>(vcl_v1.size(), gpu_result);
  if (check(host_v1, vcl_v1, epsilon) != EXIT_SUCCESS)
    return EXIT_FAILURE;

  std::cout << "Checking for unit_vector initializer..." << std::endl;
  for (std::size_t i=0; i<host_v1.size(); ++i)
    host_v1[i] = NumericT(0);
  host_v1[5] = NumericT(1);
  vcl_v1 = viennacl::unit_vector<NumericT>(vcl_v1.size(), 5);
  if (check(host_v1, vcl_v1, epsilon) != EXIT_SUCCESS)
    return EXIT_FAILURE;


  for (std::size_t i=0; i<host_v1.size(); ++i)
  {
    host_v1[i] = NumericT(1.0) + randomNumber();
    host_v2[i] = NumericT(1.0) + randomNumber();
  }

  proxy_copy(host_v1, vcl_v1);  //resync
  proxy_copy(host_v2, vcl_v2);

  std::cout << "Checking for successful copy..." << std::endl;
  if (check(host_v1, vcl_v1, epsilon) != EXIT_SUCCESS)
    return EXIT_FAILURE;
  if (check(host_v2, vcl_v2, epsilon) != EXIT_SUCCESS)
    return EXIT_FAILURE;

  //
  // Part 1: Norms and inner product
  //

  // --------------------------------------------------------------------------
  std::cout << "Testing inner_prod..." << std::endl;
  cpu_result = 0;
  for (std::size_t i=0; i<host_v1.size(); ++i)
    cpu_result += host_v1[i] * host_v2[i];
  NumericT cpu_result2 = viennacl::linalg::inner_prod(vcl_v1, vcl_v2);
  gpu_result = viennacl::linalg::inner_prod(vcl_v1, vcl_v2);

  std::cout << "Reference: " << cpu_result << std::endl;
  std::cout << cpu_result2 << std::endl;
  std::cout << gpu_result << std::endl;
  if (check(cpu_result, cpu_result2, epsilon) != EXIT_SUCCESS)
    return EXIT_FAILURE;
  if (check(cpu_result, gpu_result, epsilon) != EXIT_SUCCESS)
    return EXIT_FAILURE;

  cpu_result = 0;
  for (std::size_t i=0; i<host_v1.size(); ++i)
    cpu_result += (host_v1[i] + host_v2[i]) * (host_v2[i] - host_v1[i]);
  NumericT cpu_result3 = viennacl::linalg::inner_prod(vcl_v1 + vcl_v2, vcl_v2 - vcl_v1);
  gpu_result = viennacl::linalg::inner_prod(vcl_v1 + vcl_v2, vcl_v2 - vcl_v1);

  std::cout << "Reference: " << cpu_result << std::endl;
  std::cout << cpu_result3 << std::endl;
  std::cout << gpu_result << std::endl;
  if (check(cpu_result, cpu_result3, epsilon) != EXIT_SUCCESS)
    return EXIT_FAILURE;
  if (check(cpu_result, gpu_result, epsilon) != EXIT_SUCCESS)
    return EXIT_FAILURE;

  // --------------------------------------------------------------------------
  std::cout << "Testing norm_1..." << std::endl;
  cpu_result = 0;
  for (std::size_t i=0; i<host_v1.size(); ++i)
    cpu_result += std::fabs(host_v1[i]);
  gpu_result = viennacl::linalg::norm_1(vcl_v1);

  if (check(cpu_result, gpu_result, epsilon) != EXIT_SUCCESS)
    return EXIT_FAILURE;

  gpu_result = 2 * cpu_result; //reset
  cpu_result = 0;
  for (std::size_t i=0; i<host_v1.size(); ++i)
    cpu_result += std::fabs(host_v1[i]);
  gpu_result = cpu_result;
  cpu_result = 0;
  cpu_result = viennacl::linalg::norm_1(vcl_v1);

  if (check(cpu_result, gpu_result, epsilon) != EXIT_SUCCESS)
    return EXIT_FAILURE;

  cpu_result = 0;
  for (std::size_t i=0; i<host_v1.size(); ++i)
    cpu_result += std::fabs(host_v1[i] + host_v2[i]);
  gpu_result = cpu_result;
  cpu_result = 0;
  cpu_result = viennacl::linalg::norm_1(vcl_v1 + vcl_v2);

  if (check(cpu_result, gpu_result, epsilon) != EXIT_SUCCESS)
    return EXIT_FAILURE;

  // --------------------------------------------------------------------------
  std::cout << "Testing norm_2..." << std::endl;
  cpu_result = 0;
  for (std::size_t i=0; i<host_v1.size(); ++i)
    cpu_result += host_v1[i] * host_v1[i];
  cpu_result = std::sqrt(cpu_result);
  gpu_result = viennacl::linalg::norm_2(vcl_v1);

  if (check(cpu_result, gpu_result, epsilon) != EXIT_SUCCESS)
    return EXIT_FAILURE;

  gpu_result = 2 * cpu_result; //reset
  cpu_result = 0;
  for (std::size_t i=0; i<host_v1.size(); ++i)
    cpu_result += host_v1[i] * host_v1[i];
  gpu_result = std::sqrt(cpu_result);
  cpu_result = viennacl::linalg::norm_2(vcl_v1);

  if (check(cpu_result, gpu_result, epsilon) != EXIT_SUCCESS)
    return EXIT_FAILURE;

  cpu_result = 0;
  for (std::size_t i=0; i<host_v1.size(); ++i)
    cpu_result += (host_v1[i] + host_v2[i]) * (host_v1[i] + host_v2[i]);
  gpu_result = std::sqrt(cpu_result);
  cpu_result = viennacl::linalg::norm_2(vcl_v1 + vcl_v2);

  if (check(cpu_result, gpu_result, epsilon) != EXIT_SUCCESS)
    return EXIT_FAILURE;

  // --------------------------------------------------------------------------
  std::cout << "Testing norm_inf..." << std::endl;
  cpu_result = std::fabs(host_v1[0]);
  for (std::size_t i=0; i<host_v1.size(); ++i)
    cpu_result = std::max(std::fabs(host_v1[i]), cpu_result);
  gpu_result = viennacl::linalg::norm_inf(vcl_v1);

  if (check(cpu_result, gpu_result, epsilon) != EXIT_SUCCESS)
    return EXIT_FAILURE;

  gpu_result = 2 * cpu_result; //reset
  cpu_result = std::fabs(host_v1[0]);
  for (std::size_t i=0; i<host_v1.size(); ++i)
    cpu_result = std::max(std::fabs(host_v1[i]), cpu_result);
  gpu_result = cpu_result;
  cpu_result = 0;
  cpu_result = viennacl::linalg::norm_inf(vcl_v1);

  if (check(cpu_result, gpu_result, epsilon) != EXIT_SUCCESS)
    return EXIT_FAILURE;

  cpu_result = std::fabs(host_v1[0]);
  for (std::size_t i=0; i<host_v1.size(); ++i)
    cpu_result = std::max(std::fabs(host_v1[i] + host_v2[i]), cpu_result);
  gpu_result = cpu_result;
  cpu_result = 0;
  cpu_result = viennacl::linalg::norm_inf(vcl_v1 + vcl_v2);

  if (check(cpu_result, gpu_result, epsilon) != EXIT_SUCCESS)
    return EXIT_FAILURE;

  // --------------------------------------------------------------------------
  std::cout << "Testing index_norm_inf..." << std::endl;
  std::size_t cpu_index = 0;
  cpu_result = std::fabs(host_v1[0]);
  for (std::size_t i=0; i<host_v1.size(); ++i)
  {
    if (std::fabs(host_v1[i]) > cpu_result)
    {
      cpu_result = std::fabs(host_v1[i]);
      cpu_index = i;
    }
  }
  std::size_t gpu_index = viennacl::linalg::index_norm_inf(vcl_v1);

  if (check(static_cast<NumericT>(cpu_index), static_cast<NumericT>(gpu_index), epsilon) != EXIT_SUCCESS)
    return EXIT_FAILURE;
  // --------------------------------------------------------------------------
  cpu_result = host_v1[cpu_index];
  gpu_result = vcl_v1[viennacl::linalg::index_norm_inf(vcl_v1)];

  if (check(cpu_result, gpu_result, epsilon) != EXIT_SUCCESS)
    return EXIT_FAILURE;

  cpu_result = std::fabs(host_v1[0] + host_v2[0]);
  for (std::size_t i=0; i<host_v1.size(); ++i)
  {
    if (std::fabs(host_v1[i] + host_v2[i]) > cpu_result)
    {
      cpu_result = std::fabs(host_v1[i] + host_v2[i]);
      cpu_index = i;
    }
  }
  cpu_result = host_v1[cpu_index];
  gpu_result = vcl_v1[viennacl::linalg::index_norm_inf(vcl_v1 + vcl_v2)];

  if (check(cpu_result, gpu_result, epsilon) != EXIT_SUCCESS)
    return EXIT_FAILURE;


  // --------------------------------------------------------------------------
  std::cout << "Testing max..." << std::endl;
  cpu_result = host_v1[0];
  for (std::size_t i=0; i<host_v1.size(); ++i)
    cpu_result = std::max<NumericT>(cpu_result, host_v1[i]);
  gpu_result = viennacl::linalg::max(vcl_v1);

  if (check(cpu_result, gpu_result, epsilon) != EXIT_SUCCESS)
    return EXIT_FAILURE;

  cpu_result = host_v1[0];
  for (std::size_t i=0; i<host_v1.size(); ++i)
    cpu_result = std::max<NumericT>(cpu_result, host_v1[i]);
  gpu_result = cpu_result;
  cpu_result *= 2; //reset
  cpu_result = viennacl::linalg::max(vcl_v1);

  if (check(cpu_result, gpu_result, epsilon) != EXIT_SUCCESS)
    return EXIT_FAILURE;

  cpu_result = host_v1[0] + host_v2[0];
  for (std::size_t i=0; i<host_v1.size(); ++i)
    cpu_result = std::max<NumericT>(cpu_result, host_v1[i] + host_v2[i]);
  gpu_result = cpu_result;
  cpu_result *= 2; //reset
  cpu_result = viennacl::linalg::max(vcl_v1 + vcl_v2);

  if (check(cpu_result, gpu_result, epsilon) != EXIT_SUCCESS)
    return EXIT_FAILURE;


  // --------------------------------------------------------------------------
  std::cout << "Testing min..." << std::endl;
  cpu_result = host_v1[0];
  for (std::size_t i=0; i<host_v1.size(); ++i)
    cpu_result = std::min<NumericT>(cpu_result, host_v1[i]);
  gpu_result = viennacl::linalg::min(vcl_v1);

  if (check(cpu_result, gpu_result, epsilon) != EXIT_SUCCESS)
    return EXIT_FAILURE;

  cpu_result = host_v1[0];
  for (std::size_t i=0; i<host_v1.size(); ++i)
    cpu_result = std::min<NumericT>(cpu_result, host_v1[i]);
  gpu_result = cpu_result;
  cpu_result *= 2; //reset
  cpu_result = viennacl::linalg::min(vcl_v1);

  if (check(cpu_result, gpu_result, epsilon) != EXIT_SUCCESS)
    return EXIT_FAILURE;

  cpu_result = host_v1[0] + host_v2[0];
  for (std::size_t i=0; i<host_v1.size(); ++i)
    cpu_result = std::min<NumericT>(cpu_result, host_v1[i] + host_v2[i]);
  gpu_result = cpu_result;
  cpu_result *= 2; //reset
  cpu_result = viennacl::linalg::min(vcl_v1 + vcl_v2);

  if (check(cpu_result, gpu_result, epsilon) != EXIT_SUCCESS)
    return EXIT_FAILURE;

  // --------------------------------------------------------------------------
  std::cout << "Testing sum..." << std::endl;
  cpu_result = 0;
  for (std::size_t i=0; i<host_v1.size(); ++i)
    cpu_result += host_v1[i];
  cpu_result2 = viennacl::linalg::sum(vcl_v1);
  gpu_result = viennacl::linalg::sum(vcl_v1);

  if (check(cpu_result, cpu_result2, epsilon) != EXIT_SUCCESS)
    return EXIT_FAILURE;
  if (check(cpu_result, gpu_result, epsilon) != EXIT_SUCCESS)
    return EXIT_FAILURE;

  cpu_result = 0;
  for (std::size_t i=0; i<host_v1.size(); ++i)
    cpu_result += host_v1[i] + host_v2[i];
  cpu_result3 = viennacl::linalg::sum(vcl_v1 + vcl_v2);
  gpu_result = viennacl::linalg::sum(vcl_v1 + vcl_v2);

  if (check(cpu_result, cpu_result3, epsilon) != EXIT_SUCCESS)
    return EXIT_FAILURE;
  if (check(cpu_result, gpu_result, epsilon) != EXIT_SUCCESS)
    return EXIT_FAILURE;


  //
  // Plane rotation and assignments
  //

  // --------------------------------------------------------------------------

  for (std::size_t i=0; i<host_v1.size(); ++i)
  {
    NumericT temp =   NumericT(1.1) * host_v1[i] + NumericT(2.3) * host_v2[i];
    host_v2[i]    = - NumericT(2.3) * host_v1[i] + NumericT(1.1) * host_v2[i];
    host_v1[i]    = temp;
  }
  viennacl::linalg::plane_rotation(vcl_v1, vcl_v2, NumericT(1.1), NumericT(2.3));

  if (check(host_v1, vcl_v1, epsilon) != EXIT_SUCCESS)
    return EXIT_FAILURE;
  if (check(host_v2, vcl_v2, epsilon) != EXIT_SUCCESS)
    return EXIT_FAILURE;

  // --------------------------------------------------------------------------

  std::cout << "Testing assignments..." << std::endl;
  NumericT val = static_cast<NumericT>(1e-1);
  for (size_t i=0; i < host_v1.size(); ++i)
    host_v1[i] = val;

  for (size_t i=0; i < vcl_v1.size(); ++i)
    vcl_v1(i) = val;

  if (check(host_v1, vcl_v1, epsilon) != EXIT_SUCCESS)
    return EXIT_FAILURE;

  std::cout << "Testing assignments via iterators..." << std::endl;

  host_v1[2] = static_cast<NumericT>(1.9);
   vcl_v1[2] = static_cast<NumericT>(1.9);

  host_v1[2] = static_cast<NumericT>(1.5);
  typename ViennaCLVectorType1::iterator vcl_v1_it = vcl_v1.begin();
  ++vcl_v1_it;
  ++vcl_v1_it;
  *vcl_v1_it = static_cast<NumericT>(1.5);

  if (check(host_v1, vcl_v1, epsilon) != EXIT_SUCCESS)
    return EXIT_FAILURE;

  //
  // multiplication and division of vectors by scalars
  //
  for (std::size_t i=0; i < host_v1.size(); ++i)
  {
    host_v1[i] = NumericT(1.0) + randomNumber();
    host_v2[i] = NumericT(3.1415) * host_v1[i];
  }
  proxy_copy(host_v1, vcl_v1);  //resync
  proxy_copy(host_v2, vcl_v2);

  std::cout << "Testing scaling with CPU scalar..." << std::endl;
  NumericT alpha = static_cast<NumericT>(1.7182);
  viennacl::scalar<NumericT> gpu_alpha = alpha;

  for (std::size_t i=0; i < host_v1.size(); ++i)
    host_v1[i]  *= NumericT(long(alpha));
  vcl_v1    *= long(alpha);

  if (check(host_v1, vcl_v1, epsilon) != EXIT_SUCCESS)
    return EXIT_FAILURE;

  for (std::size_t i=0; i < host_v1.size(); ++i)
    host_v1[i]  *= NumericT(float(alpha));
  vcl_v1    *= float(alpha);

  if (check(host_v1, vcl_v1, epsilon) != EXIT_SUCCESS)
    return EXIT_FAILURE;

  for (std::size_t i=0; i < host_v1.size(); ++i)
    host_v1[i]  *= NumericT(double(alpha));
  vcl_v1    *= double(alpha);

  if (check(host_v1, vcl_v1, epsilon) != EXIT_SUCCESS)
    return EXIT_FAILURE;


  std::cout << "Testing scaling with GPU scalar..." << std::endl;
  for (std::size_t i=0; i < host_v1.size(); ++i)
    host_v1[i]  *= alpha;
  vcl_v1    *= gpu_alpha;

  if (check(host_v1, vcl_v1, epsilon) != EXIT_SUCCESS)
    return EXIT_FAILURE;

  std::cout << "Testing scaling with scalar expression..." << std::endl;
  cpu_result = 0;
  for (std::size_t i=0; i < host_v1.size(); ++i)
    cpu_result += host_v1[i] * host_v2[i];
  for (std::size_t i=0; i < host_v1.size(); ++i)
    host_v1[i] *= cpu_result;
  vcl_v1    *= viennacl::linalg::inner_prod(vcl_v1, vcl_v2);

  if (check(host_v1, vcl_v1, epsilon) != EXIT_SUCCESS)
    return EXIT_FAILURE;

  NumericT beta  = static_cast<NumericT>(1.4153);
  viennacl::scalar<NumericT> gpu_beta = beta;

  std::cout << "Testing shrinking with CPU scalar..." << std::endl;
  for (std::size_t i=0; i < host_v1.size(); ++i)
    host_v1[i] /= NumericT(long(beta));
  vcl_v1   /= long(beta);

  if (check(host_v1, vcl_v1, epsilon) != EXIT_SUCCESS)
    return EXIT_FAILURE;

  for (std::size_t i=0; i < host_v1.size(); ++i)
    host_v1[i] /= NumericT(float(beta));
  vcl_v1   /= float(beta);

  if (check(host_v1, vcl_v1, epsilon) != EXIT_SUCCESS)
    return EXIT_FAILURE;

  for (std::size_t i=0; i < host_v1.size(); ++i)
    host_v1[i] /= NumericT(double(beta));
  vcl_v1   /= double(beta);

  if (check(host_v1, vcl_v1, epsilon) != EXIT_SUCCESS)
    return EXIT_FAILURE;


  std::cout << "Testing shrinking with GPU scalar..." << std::endl;
  for (std::size_t i=0; i < host_v1.size(); ++i)
    host_v1[i] /= beta;
  vcl_v1   /= gpu_beta;

  if (check(host_v1, vcl_v1, epsilon) != EXIT_SUCCESS)
    return EXIT_FAILURE;



  //
  // add and inplace_add of vectors
  //
  for (size_t i=0; i < host_v1.size(); ++i)
  {
    host_v1[i] = NumericT(1.0) + randomNumber();
    host_v2[i] = NumericT(3.1415) * host_v1[i];
  }
  proxy_copy(host_v1, vcl_v1);  //resync
  proxy_copy(host_v2, vcl_v2);

  std::cout << "Testing add on vector..." << std::endl;

  std::cout << "Checking for successful copy..." << std::endl;
  if (check(host_v1, vcl_v1, epsilon) != EXIT_SUCCESS)
    return EXIT_FAILURE;
  if (check(host_v2, vcl_v2, epsilon) != EXIT_SUCCESS)
    return EXIT_FAILURE;

  for (size_t i=0; i < host_v1.size(); ++i)
    host_v1[i] = host_v1[i] + host_v2[i];
  vcl_v1       =   vcl_v1 +   vcl_v2;

  if (check(host_v1, vcl_v1, epsilon) != EXIT_SUCCESS)
    return EXIT_FAILURE;

  std::cout << "Testing add on vector with flipsign..." << std::endl;
  for (size_t i=0; i < host_v1.size(); ++i)
    host_v1[i] = - host_v1[i] + host_v2[i];
  vcl_v1       = -   vcl_v1 +   vcl_v2;

  if (check(host_v1, vcl_v1, epsilon) != EXIT_SUCCESS)
    return EXIT_FAILURE;

  std::cout << "Testing inplace-add on vector..." << std::endl;
  for (size_t i=0; i < host_v1.size(); ++i)
    host_v1[i] += host_v2[i];
  vcl_v1   +=   vcl_v2;

  if (check(host_v1, vcl_v1, epsilon) != EXIT_SUCCESS)
    return EXIT_FAILURE;

  std::cout << "Testing assignment to vector with vector multiplied by scalar expression..." << std::endl;
  cpu_result = 0;
  for (std::size_t i=0; i < host_v1.size(); ++i)
    cpu_result += host_v1[i] * host_v2[i];
  for (std::size_t i=0; i < host_v1.size(); ++i)
    host_v1[i] = cpu_result * host_v2[i];
  //host_v1  = inner_prod(host_v1, host_v2) * host_v2;
  vcl_v1    = viennacl::linalg::inner_prod(vcl_v1, vcl_v2) * vcl_v2;

  //
  // subtract and inplace_subtract of vectors
  //
  std::cout << "Testing sub on vector..." << std::endl;
  for (std::size_t i=0; i < host_v1.size(); ++i)
    host_v2[i] = NumericT(3.1415) * host_v1[i];
  proxy_copy(host_v1, vcl_v1);
  proxy_copy(host_v2, vcl_v2);

  for (std::size_t i=0; i < host_v1.size(); ++i)
    host_v1[i] = host_v1[i] - host_v2[i];
  vcl_v1       =   vcl_v1 -   vcl_v2;

  if (check(host_v1, vcl_v1, epsilon) != EXIT_SUCCESS)
    return EXIT_FAILURE;

  std::cout << "Testing inplace-sub on vector..." << std::endl;
  for (std::size_t i=0; i < host_v1.size(); ++i)
    host_v1[i] -= host_v2[i];
  vcl_v1   -= vcl_v2;

  if (check(host_v1, vcl_v1, epsilon) != EXIT_SUCCESS)
    return EXIT_FAILURE;



  //
  // multiply-add
  //
  std::cout << "Testing multiply-add on vector with CPU scalar (right)..." << std::endl;
  for (size_t i=0; i < host_v1.size(); ++i)
  {
    host_v1[i] = NumericT(1.0) + randomNumber();
    host_v2[i] = NumericT(3.1415) * host_v1[i];
  }
  proxy_copy(host_v1, vcl_v1);
  proxy_copy(host_v2, vcl_v2);

  for (std::size_t i=0; i < host_v1.size(); ++i)
    host_v1[i] = host_v1[i] + host_v2[i] * NumericT(float(alpha));
  vcl_v1   = vcl_v1   +   vcl_v2 * float(alpha);

  if (check(host_v1, vcl_v1, epsilon) != EXIT_SUCCESS)
    return EXIT_FAILURE;

  for (std::size_t i=0; i < host_v1.size(); ++i)
    host_v1[i] = host_v1[i] + host_v2[i] * NumericT(double(alpha));
  vcl_v1   = vcl_v1   +   vcl_v2 * double(alpha);

  if (check(host_v1, vcl_v1, epsilon) != EXIT_SUCCESS)
    return EXIT_FAILURE;


  std::cout << "Testing multiply-add on vector with CPU scalar (left)..." << std::endl;
  for (std::size_t i=0; i < host_v1.size(); ++i)
    host_v2[i] = NumericT(3.1415) * host_v1[i];
  proxy_copy(host_v1, vcl_v1);
  proxy_copy(host_v2, vcl_v2);

  for (std::size_t i=0; i < host_v1.size(); ++i)
    host_v1[i] = NumericT(long(alpha)) * host_v1[i] + host_v2[i];
  vcl_v1   = long(alpha) *   vcl_v1 +   vcl_v2;

  if (check(host_v1, vcl_v1, epsilon) != EXIT_SUCCESS)
    return EXIT_FAILURE;

  for (std::size_t i=0; i < host_v1.size(); ++i)
    host_v1[i] = NumericT(float(alpha)) * host_v1[i] + host_v2[i];
  vcl_v1   = float(alpha) *   vcl_v1 +   vcl_v2;

  if (check(host_v1, vcl_v1, epsilon) != EXIT_SUCCESS)
    return EXIT_FAILURE;

  for (std::size_t i=0; i < host_v1.size(); ++i)
    host_v1[i] = NumericT(double(alpha)) * host_v1[i] + host_v2[i];
  vcl_v1   = double(alpha) *   vcl_v1 +   vcl_v2;

  if (check(host_v1, vcl_v1, epsilon) != EXIT_SUCCESS)
    return EXIT_FAILURE;


  std::cout << "Testing multiply-add on vector with CPU scalar (both)..." << std::endl;
  for (std::size_t i=0; i < host_v1.size(); ++i)
    host_v2[i] = NumericT(3.1415) * host_v1[i];
  proxy_copy(host_v1, vcl_v1);
  proxy_copy(host_v2, vcl_v2);

  for (std::size_t i=0; i < host_v1.size(); ++i)
    host_v1[i] = NumericT(long(alpha)) * host_v1[i] + NumericT(long(beta)) * host_v2[i];
  vcl_v1   = long(alpha) *   vcl_v1 + long(beta) *   vcl_v2;

  if (check(host_v1, vcl_v1, epsilon) != EXIT_SUCCESS)
    return EXIT_FAILURE;

  for (std::size_t i=0; i < host_v1.size(); ++i)
    host_v1[i] = NumericT(float(alpha)) * host_v1[i] + NumericT(float(beta)) * host_v2[i];
  vcl_v1   = float(alpha) *   vcl_v1 + float(beta) *   vcl_v2;

  if (check(host_v1, vcl_v1, epsilon) != EXIT_SUCCESS)
    return EXIT_FAILURE;

  for (std::size_t i=0; i < host_v1.size(); ++i)
    host_v1[i] = NumericT(double(alpha)) * host_v1[i] + NumericT(double(beta)) * host_v2[i];
  vcl_v1   = double(alpha) *   vcl_v1 + double(beta) *   vcl_v2;

  if (check(host_v1, vcl_v1, epsilon) != EXIT_SUCCESS)
    return EXIT_FAILURE;


  std::cout << "Testing inplace multiply-add on vector with CPU scalar..." << std::endl;
  for (std::size_t i=0; i < host_v1.size(); ++i)
    host_v2[i] = NumericT(3.1415) * host_v1[i];
  proxy_copy(host_v1, vcl_v1);
  proxy_copy(host_v2, vcl_v2);

  for (std::size_t i=0; i < host_v1.size(); ++i)
    host_v1[i] += host_v2[i] * NumericT(long(alpha));
  vcl_v1   +=   vcl_v2 * long(alpha);

  if (check(host_v1, vcl_v1, epsilon) != EXIT_SUCCESS)
    return EXIT_FAILURE;

  for (std::size_t i=0; i < host_v1.size(); ++i)
    host_v1[i] += host_v2[i] * NumericT(float(alpha));
  vcl_v1   +=   vcl_v2 * float(alpha);

  if (check(host_v1, vcl_v1, epsilon) != EXIT_SUCCESS)
    return EXIT_FAILURE;

  for (std::size_t i=0; i < host_v1.size(); ++i)
    host_v1[i] += NumericT(double(alpha)) * host_v2[i];
  vcl_v1   += double(alpha) *   vcl_v2;

  if (check(host_v1, vcl_v1, epsilon) != EXIT_SUCCESS)
    return EXIT_FAILURE;


  std::cout << "Testing multiply-add on vector with GPU scalar (right)..." << std::endl;
  for (std::size_t i=0; i < host_v1.size(); ++i)
    host_v2[i] = NumericT(3.1415) * host_v1[i];
  proxy_copy(host_v1, vcl_v1);
  proxy_copy(host_v2, vcl_v2);

  for (std::size_t i=0; i < host_v1.size(); ++i)
    host_v1[i] = host_v1[i] +     alpha * host_v2[i];
  vcl_v1   = vcl_v1   + gpu_alpha *   vcl_v2;

  if (check(host_v1, vcl_v1, epsilon) != EXIT_SUCCESS)
    return EXIT_FAILURE;

  std::cout << "Testing multiply-add on vector with GPU scalar (left)..." << std::endl;
  for (std::size_t i=0; i < host_v1.size(); ++i)
    host_v2[i] = NumericT(3.1415) * host_v1[i];
  proxy_copy(host_v1, vcl_v1);
  proxy_copy(host_v2, vcl_v2);

  for (std::size_t i=0; i < host_v1.size(); ++i)
    host_v1[i] = host_v1[i] +     alpha * host_v2[i];
  vcl_v1   = vcl_v1   + gpu_alpha *   vcl_v2;

  if (check(host_v1, vcl_v1, epsilon) != EXIT_SUCCESS)
    return EXIT_FAILURE;

  std::cout << "Testing multiply-add on vector with GPU scalar (both)..." << std::endl;
  for (std::size_t i=0; i < host_v1.size(); ++i)
    host_v2[i] = NumericT(3.1415) * host_v1[i];
  proxy_copy(host_v1, vcl_v1);
  proxy_copy(host_v2, vcl_v2);

  for (std::size_t i=0; i < host_v1.size(); ++i)
    host_v1[i] = alpha * host_v1[i] + beta * host_v2[i];
  vcl_v1   = gpu_alpha *   vcl_v1 + gpu_beta *   vcl_v2;

  if (check(host_v1, vcl_v1, epsilon) != EXIT_SUCCESS)
    return EXIT_FAILURE;


  std::cout << "Testing inplace multiply-add on vector with GPU scalar (both, adding)..." << std::endl;
  for (std::size_t i=0; i < host_v1.size(); ++i)
    host_v2[i] = NumericT(3.1415) * host_v1[i];
  proxy_copy(host_v1, vcl_v1);
  proxy_copy(host_v2, vcl_v2);

  for (std::size_t i=0; i < host_v1.size(); ++i)
    host_v1[i] +=     alpha * host_v1[i] +     beta * host_v2[i];
  vcl_v1   += gpu_alpha *   vcl_v1 + gpu_beta *   vcl_v2;

  if (check(host_v1, vcl_v1, epsilon) != EXIT_SUCCESS)
    return EXIT_FAILURE;

  std::cout << "Testing inplace multiply-add on vector with GPU scalar (both, subtracting)..." << std::endl;
  for (std::size_t i=0; i < host_v1.size(); ++i)
    host_v2[i] = NumericT(3.1415) * host_v1[i];
  proxy_copy(host_v1, vcl_v1);
  proxy_copy(host_v2, vcl_v2);

  for (std::size_t i=0; i < host_v1.size(); ++i)
    host_v1[i] +=     alpha * host_v1[i] -     beta * host_v2[i];
  vcl_v1   += gpu_alpha *   vcl_v1 - gpu_beta *   vcl_v2;

  if (check(host_v1, vcl_v1, epsilon) != EXIT_SUCCESS)
    return EXIT_FAILURE;



  std::cout << "Testing inplace multiply-add on vector with GPU scalar..." << std::endl;
  for (std::size_t i=0; i < host_v1.size(); ++i)
    host_v2[i] = NumericT(3.1415) * host_v1[i];
  proxy_copy(host_v1, vcl_v1);
  proxy_copy(host_v2, vcl_v2);

  for (std::size_t i=0; i < host_v1.size(); ++i)
    host_v1[i] +=     alpha * host_v2[i];
  vcl_v1   += gpu_alpha *   vcl_v2;

  if (check(host_v1, vcl_v1, epsilon) != EXIT_SUCCESS)
    return EXIT_FAILURE;


  //
  // division-add
  //
  std::cout << "Testing division-add on vector with CPU scalar (right)..." << std::endl;
  for (size_t i=0; i < host_v1.size(); ++i)
  {
    host_v1[i] = NumericT(1.0) + randomNumber();
    host_v2[i] = NumericT(3.1415) * host_v1[i];
  }
  proxy_copy(host_v1, vcl_v1);
  proxy_copy(host_v2, vcl_v2);

  for (std::size_t i=0; i < host_v1.size(); ++i)
    host_v1[i] = host_v1[i] + host_v2[i] / NumericT(long(alpha));
  vcl_v1   = vcl_v1   + vcl_v2 / long(alpha);

  if (check(host_v1, vcl_v1, epsilon) != EXIT_SUCCESS)
    return EXIT_FAILURE;

  for (std::size_t i=0; i < host_v1.size(); ++i)
    host_v1[i] = host_v1[i] + host_v2[i] / NumericT(float(alpha));
  vcl_v1   = vcl_v1   + vcl_v2 / float(alpha);

  if (check(host_v1, vcl_v1, epsilon) != EXIT_SUCCESS)
    return EXIT_FAILURE;

  for (std::size_t i=0; i < host_v1.size(); ++i)
    host_v1[i] = host_v1[i] + host_v2[i] / NumericT(double(alpha));
  vcl_v1   = vcl_v1   + vcl_v2 / double(alpha);

  if (check(host_v1, vcl_v1, epsilon) != EXIT_SUCCESS)
    return EXIT_FAILURE;


  std::cout << "Testing division-add on vector with CPU scalar (left)..." << std::endl;
  for (std::size_t i=0; i < host_v1.size(); ++i)
    host_v2[i] = NumericT(3.1415) * host_v1[i];
  proxy_copy(host_v1, vcl_v1);
  proxy_copy(host_v2, vcl_v2);

  for (std::size_t i=0; i < host_v1.size(); ++i)
    host_v1[i] = host_v1[i] / NumericT(float(alpha)) + host_v2[i];
  vcl_v1   =   vcl_v1 / float(alpha) +   vcl_v2;

  if (check(host_v1, vcl_v1, epsilon) != EXIT_SUCCESS)
    return EXIT_FAILURE;

  for (std::size_t i=0; i < host_v1.size(); ++i)
    host_v1[i] = host_v1[i] / NumericT(double(alpha)) + host_v2[i];
  vcl_v1   =   vcl_v1 / double(alpha) +   vcl_v2;

  if (check(host_v1, vcl_v1, epsilon) != EXIT_SUCCESS)
    return EXIT_FAILURE;


  std::cout << "Testing division-add on vector with CPU scalar (both)..." << std::endl;
  for (std::size_t i=0; i < host_v1.size(); ++i)
    host_v2[i] = NumericT(3.1415) * host_v1[i];
  proxy_copy(host_v1, vcl_v1);
  proxy_copy(host_v2, vcl_v2);

  for (std::size_t i=0; i < host_v1.size(); ++i)
    host_v1[i] = host_v1[i] / NumericT(float(alpha)) + host_v2[i] / NumericT(float(beta));
  vcl_v1   =   vcl_v1 / float(alpha) +   vcl_v2 / float(beta);

  if (check(host_v1, vcl_v1, epsilon) != EXIT_SUCCESS)
    return EXIT_FAILURE;

  for (std::size_t i=0; i < host_v1.size(); ++i)
    host_v1[i] = host_v1[i] / NumericT(double(alpha)) + host_v2[i] / NumericT(double(beta));
  vcl_v1   =   vcl_v1 / double(alpha) +   vcl_v2 / double(beta);

  if (check(host_v1, vcl_v1, epsilon) != EXIT_SUCCESS)
    return EXIT_FAILURE;

  std::cout << "Testing division-multiply-add on vector with CPU scalar..." << std::endl;
  for (std::size_t i=0; i < host_v1.size(); ++i)
    host_v2[i] = NumericT(3.1415) * host_v1[i];
  proxy_copy(host_v1, vcl_v1);
  proxy_copy(host_v2, vcl_v2);

  for (std::size_t i=0; i < host_v1.size(); ++i)
    host_v1[i] = host_v1[i] / alpha + host_v2[i] * beta;
  vcl_v1   =   vcl_v1 / alpha +   vcl_v2 * beta;

  if (check(host_v1, vcl_v1, epsilon) != EXIT_SUCCESS)
    return EXIT_FAILURE;


  std::cout << "Testing multiply-division-add on vector with CPU scalar..." << std::endl;
  for (std::size_t i=0; i < host_v1.size(); ++i)
    host_v2[i] = NumericT(3.1415) * host_v1[i];
  proxy_copy(host_v1, vcl_v1);
  proxy_copy(host_v2, vcl_v2);

  for (std::size_t i=0; i < host_v1.size(); ++i)
    host_v1[i] = host_v1[i] * alpha + host_v2[i] / beta;
  vcl_v1   =   vcl_v1 * alpha +   vcl_v2 / beta;

  if (check(host_v1, vcl_v1, epsilon) != EXIT_SUCCESS)
    return EXIT_FAILURE;



  std::cout << "Testing inplace division-add on vector with CPU scalar..." << std::endl;
  for (std::size_t i=0; i < host_v1.size(); ++i)
    host_v2[i] = NumericT(3.1415) * host_v1[i];
  proxy_copy(host_v1, vcl_v1);
  proxy_copy(host_v2, vcl_v2);

  for (std::size_t i=0; i < host_v1.size(); ++i)
    host_v1[i] += host_v2[i] / alpha;
  vcl_v1   += vcl_v2 / alpha;

  if (check(host_v1, vcl_v1, epsilon) != EXIT_SUCCESS)
    return EXIT_FAILURE;


  std::cout << "Testing division-add on vector with GPU scalar (right)..." << std::endl;
  for (std::size_t i=0; i < host_v1.size(); ++i)
    host_v2[i] = NumericT(3.1415) * host_v1[i];
  proxy_copy(host_v1, vcl_v1);
  proxy_copy(host_v2, vcl_v2);

  for (std::size_t i=0; i < host_v1.size(); ++i)
    host_v1[i] = host_v1[i] + host_v2[i] / alpha;
  vcl_v1   = vcl_v1   +   vcl_v2 / gpu_alpha;

  if (check(host_v1, vcl_v1, epsilon) != EXIT_SUCCESS)
    return EXIT_FAILURE;

  std::cout << "Testing division-add on vector with GPU scalar (left)..." << std::endl;
  for (std::size_t i=0; i < host_v1.size(); ++i)
    host_v2[i] = NumericT(3.1415) * host_v1[i];
  proxy_copy(host_v1, vcl_v1);
  proxy_copy(host_v2, vcl_v2);

  for (std::size_t i=0; i < host_v1.size(); ++i)
    host_v1[i] = host_v1[i] + host_v2[i] / alpha;
  vcl_v1   = vcl_v1   +   vcl_v2 / gpu_alpha;

  if (check(host_v1, vcl_v1, epsilon) != EXIT_SUCCESS)
    return EXIT_FAILURE;

  std::cout << "Testing division-add on vector with GPU scalar (both)..." << std::endl;
  for (std::size_t i=0; i < host_v1.size(); ++i)
    host_v2[i] = NumericT(3.1415) * host_v1[i];
  proxy_copy(host_v1, vcl_v1);
  proxy_copy(host_v2, vcl_v2);

  for (std::size_t i=0; i < host_v1.size(); ++i)
    host_v1[i] = host_v1[i] / alpha     + host_v2[i] / beta;
  vcl_v1   =   vcl_v1 / gpu_alpha +   vcl_v2 / gpu_beta;

  if (check(host_v1, vcl_v1, epsilon) != EXIT_SUCCESS)
    return EXIT_FAILURE;


  std::cout << "Testing inplace division-add on vector with GPU scalar (both, adding)..." << std::endl;
  for (std::size_t i=0; i < host_v1.size(); ++i)
    host_v2[i] = NumericT(3.1415) * host_v1[i];
  proxy_copy(host_v1, vcl_v1);
  proxy_copy(host_v2, vcl_v2);

  for (std::size_t i=0; i < host_v1.size(); ++i)
    host_v1[i] += host_v1[i] / alpha     + host_v2[i] / beta;
  vcl_v1   +=   vcl_v1 / gpu_alpha +   vcl_v2 / gpu_beta;

  if (check(host_v1, vcl_v1, epsilon) != EXIT_SUCCESS)
    return EXIT_FAILURE;

  std::cout << "Testing inplace division-add on vector with GPU scalar (both, subtracting)..." << std::endl;
  for (std::size_t i=0; i < host_v1.size(); ++i)
    host_v2[i] = NumericT(3.1415) * host_v1[i];
  proxy_copy(host_v1, vcl_v1);
  proxy_copy(host_v2, vcl_v2);

  for (std::size_t i=0; i < host_v1.size(); ++i)
    host_v1[i] += host_v1[i] / alpha     - host_v2[i] / beta;
  vcl_v1   +=   vcl_v1 / gpu_alpha -   vcl_v2 / gpu_beta;

  if (check(host_v1, vcl_v1, epsilon) != EXIT_SUCCESS)
    return EXIT_FAILURE;

  std::cout << "Testing inplace division-multiply-add on vector with GPU scalar (adding)..." << std::endl;
  for (std::size_t i=0; i < host_v1.size(); ++i)
    host_v2[i] = NumericT(3.1415) * host_v1[i];
  proxy_copy(host_v1, vcl_v1);
  proxy_copy(host_v2, vcl_v2);

  for (std::size_t i=0; i < host_v1.size(); ++i)
    host_v1[i] += host_v1[i] / alpha     + host_v2[i] * beta;
  vcl_v1   +=   vcl_v1 / gpu_alpha +   vcl_v2 * gpu_beta;

  if (check(host_v1, vcl_v1, epsilon) != EXIT_SUCCESS)
    return EXIT_FAILURE;

  std::cout << "Testing inplace multiply-division-add on vector with GPU scalar (subtracting)..." << std::endl;
  for (std::size_t i=0; i < host_v1.size(); ++i)
    host_v2[i] = NumericT(3.1415) * host_v1[i];
  proxy_copy(host_v1, vcl_v1);
  proxy_copy(host_v2, vcl_v2);

  for (std::size_t i=0; i < host_v1.size(); ++i)
    host_v1[i] += host_v1[i] * alpha     - host_v2[i] / beta;
  vcl_v1   +=   vcl_v1 * gpu_alpha -   vcl_v2 / gpu_beta;

  if (check(host_v1, vcl_v1, epsilon) != EXIT_SUCCESS)
    return EXIT_FAILURE;



  std::cout << "Testing inplace division-add on vector with GPU scalar..." << std::endl;
  for (std::size_t i=0; i < host_v1.size(); ++i)
    host_v2[i] = NumericT(3.1415) * host_v1[i];
  proxy_copy(host_v1, vcl_v1);
  proxy_copy(host_v2, vcl_v2);

  for (std::size_t i=0; i < host_v1.size(); ++i)
    host_v1[i] += host_v2[i] * alpha;
  vcl_v1   +=   vcl_v2 * gpu_alpha;

  if (check(host_v1, vcl_v1, epsilon) != EXIT_SUCCESS)
    return EXIT_FAILURE;



  //
  // multiply-subtract
  //
  std::cout << "Testing multiply-subtract on vector with CPU scalar (right)..." << std::endl;
  for (size_t i=0; i < host_v1.size(); ++i)
  {
    host_v1[i] = NumericT(1.0) + randomNumber();
    host_v2[i] = NumericT(3.1415) * host_v1[i];
  }
  proxy_copy(host_v1, vcl_v1);
  proxy_copy(host_v2, vcl_v2);

  for (std::size_t i=0; i < host_v1.size(); ++i)
    host_v1[i] = host_v1[i] - alpha * host_v2[i];
  vcl_v1   = vcl_v1   - alpha *   vcl_v2;

  if (check(host_v1, vcl_v1, epsilon) != EXIT_SUCCESS)
    return EXIT_FAILURE;


  std::cout << "Testing multiply-subtract on vector with CPU scalar (left)..." << std::endl;
  for (std::size_t i=0; i < host_v1.size(); ++i)
    host_v2[i] = NumericT(3.1415) * host_v1[i];
  proxy_copy(host_v1, vcl_v1);
  proxy_copy(host_v2, vcl_v2);

  for (std::size_t i=0; i < host_v1.size(); ++i)
    host_v1[i] = alpha * host_v1[i] - host_v2[i];
  vcl_v1   = alpha * vcl_v1   -   vcl_v2;

  if (check(host_v1, vcl_v1, epsilon) != EXIT_SUCCESS)
    return EXIT_FAILURE;

  std::cout << "Testing multiply-subtract on vector with CPU scalar (both)..." << std::endl;
  for (std::size_t i=0; i < host_v1.size(); ++i)
    host_v2[i] = NumericT(3.1415) * host_v1[i];
  proxy_copy(host_v1, vcl_v1);
  proxy_copy(host_v2, vcl_v2);

  for (std::size_t i=0; i < host_v1.size(); ++i)
    host_v1[i] = alpha * host_v1[i] - beta * host_v2[i];
  vcl_v1   = alpha * vcl_v1   - beta *   vcl_v2;

  if (check(host_v1, vcl_v1, epsilon) != EXIT_SUCCESS)
    return EXIT_FAILURE;


  std::cout << "Testing inplace multiply-subtract on vector with CPU scalar..." << std::endl;
  for (std::size_t i=0; i < host_v1.size(); ++i)
    host_v2[i] = NumericT(3.1415) * host_v1[i];
  proxy_copy(host_v1, vcl_v1);
  proxy_copy(host_v2, vcl_v2);

  for (std::size_t i=0; i < host_v1.size(); ++i)
    host_v1[i] -= alpha * host_v2[i];
  vcl_v1   -= alpha *   vcl_v2;

  if (check(host_v1, vcl_v1, epsilon) != EXIT_SUCCESS)
    return EXIT_FAILURE;


  std::cout << "Testing multiply-subtract on vector with GPU scalar (right)..." << std::endl;
  for (std::size_t i=0; i < host_v1.size(); ++i)
    host_v2[i] = NumericT(3.1415) * host_v1[i];
  proxy_copy(host_v1, vcl_v1);
  proxy_copy(host_v2, vcl_v2);

  for (std::size_t i=0; i < host_v1.size(); ++i)
    host_v1[i] = host_v1[i] -     alpha * host_v2[i];
  vcl_v1   = vcl_v1   - gpu_alpha *   vcl_v2;

  if (check(host_v1, vcl_v1, epsilon) != EXIT_SUCCESS)
    return EXIT_FAILURE;

  std::cout << "Testing multiply-subtract on vector with GPU scalar (left)..." << std::endl;
  for (std::size_t i=0; i < host_v1.size(); ++i)
    host_v2[i] = NumericT(3.1415) * host_v1[i];
  proxy_copy(host_v1, vcl_v1);
  proxy_copy(host_v2, vcl_v2);

  for (std::size_t i=0; i < host_v1.size(); ++i)
    host_v1[i] = host_v1[i] -     alpha * host_v2[i];
  vcl_v1   = vcl_v1   - gpu_alpha *   vcl_v2;

  if (check(host_v1, vcl_v1, epsilon) != EXIT_SUCCESS)
    return EXIT_FAILURE;

  std::cout << "Testing multiply-subtract on vector with GPU scalar (both)..." << std::endl;
  for (std::size_t i=0; i < host_v1.size(); ++i)
    host_v2[i] = NumericT(3.1415) * host_v1[i];
  proxy_copy(host_v1, vcl_v1);
  proxy_copy(host_v2, vcl_v2);

  for (std::size_t i=0; i < host_v1.size(); ++i)
    host_v1[i] =     alpha * host_v1[i] -     beta * host_v2[i];
  vcl_v1   = gpu_alpha * vcl_v1   - gpu_beta *   vcl_v2;

  if (check(host_v1, vcl_v1, epsilon) != EXIT_SUCCESS)
    return EXIT_FAILURE;

  std::cout << "Testing inplace multiply-subtract on vector with GPU scalar (both, adding)..." << std::endl;
  for (std::size_t i=0; i < host_v1.size(); ++i)
    host_v2[i] = NumericT(3.1415) * host_v1[i];
  proxy_copy(host_v1, vcl_v1);
  proxy_copy(host_v2, vcl_v2);

  for (std::size_t i=0; i < host_v1.size(); ++i)
    host_v1[i] -=     alpha * host_v1[i] +     beta * host_v2[i];
  vcl_v1   -= gpu_alpha * vcl_v1   + gpu_beta *   vcl_v2;

  if (check(host_v1, vcl_v1, epsilon) != EXIT_SUCCESS)
    return EXIT_FAILURE;

  std::cout << "Testing inplace multiply-subtract on vector with GPU scalar (both, subtracting)..." << std::endl;
  for (std::size_t i=0; i < host_v1.size(); ++i)
    host_v2[i] = NumericT(3.1415) * host_v1[i];
  proxy_copy(host_v1, vcl_v1);
  proxy_copy(host_v2, vcl_v2);

  for (std::size_t i=0; i < host_v1.size(); ++i)
    host_v1[i] -=     alpha * host_v1[i] -     beta * host_v2[i];
  vcl_v1   -= gpu_alpha * vcl_v1   - gpu_beta *   vcl_v2;

  if (check(host_v1, vcl_v1, epsilon) != EXIT_SUCCESS)
    return EXIT_FAILURE;


  std::cout << "Testing inplace multiply-subtract on vector with GPU scalar..." << std::endl;
  for (std::size_t i=0; i < host_v1.size(); ++i)
    host_v2[i] = NumericT(3.1415) * host_v1[i];
  proxy_copy(host_v1, vcl_v1);
  proxy_copy(host_v2, vcl_v2);

  for (std::size_t i=0; i < host_v1.size(); ++i)
    host_v1[i] -=     alpha * host_v2[i];
  vcl_v1   -= gpu_alpha *   vcl_v2;

  if (check(host_v1, vcl_v1, epsilon) != EXIT_SUCCESS)
    return EXIT_FAILURE;



  //
  // division-subtract
  //
  std::cout << "Testing division-subtract on vector with CPU scalar (right)..." << std::endl;
  for (size_t i=0; i < host_v1.size(); ++i)
  {
    host_v1[i] = NumericT(1.0) + randomNumber();
    host_v2[i] = NumericT(3.1415) * host_v1[i];
  }
  proxy_copy(host_v1, vcl_v1);
  proxy_copy(host_v2, vcl_v2);

  for (std::size_t i=0; i < host_v1.size(); ++i)
    host_v1[i] = host_v1[i] - host_v2[i] / alpha;
  vcl_v1   = vcl_v1   -   vcl_v2 / alpha;

  if (check(host_v1, vcl_v1, epsilon) != EXIT_SUCCESS)
    return EXIT_FAILURE;


  std::cout << "Testing division-subtract on vector with CPU scalar (left)..." << std::endl;
  for (std::size_t i=0; i < host_v1.size(); ++i)
    host_v2[i] = NumericT(3.1415) * host_v1[i];
  proxy_copy(host_v1, vcl_v1);
  proxy_copy(host_v2, vcl_v2);

  for (std::size_t i=0; i < host_v1.size(); ++i)
    host_v1[i] = host_v1[i] / alpha - host_v2[i];
  vcl_v1   =   vcl_v1 / alpha -   vcl_v2;

  if (check(host_v1, vcl_v1, epsilon) != EXIT_SUCCESS)
    return EXIT_FAILURE;

  std::cout << "Testing division-subtract on vector with CPU scalar (both)..." << std::endl;
  for (std::size_t i=0; i < host_v1.size(); ++i)
    host_v2[i] = NumericT(3.1415) * host_v1[i];
  proxy_copy(host_v1, vcl_v1);
  proxy_copy(host_v2, vcl_v2);

  for (std::size_t i=0; i < host_v1.size(); ++i)
    host_v1[i] = host_v1[i] / alpha - host_v2[i] / alpha;
  vcl_v1   =   vcl_v1 / alpha -   vcl_v2 / alpha;

  if (check(host_v1, vcl_v1, epsilon) != EXIT_SUCCESS)
    return EXIT_FAILURE;


  std::cout << "Testing inplace division-subtract on vector with CPU scalar..." << std::endl;
  for (std::size_t i=0; i < host_v1.size(); ++i)
    host_v2[i] = NumericT(3.1415) * host_v1[i];
  proxy_copy(host_v1, vcl_v1);
  proxy_copy(host_v2, vcl_v2);

  for (std::size_t i=0; i < host_v1.size(); ++i)
    host_v1[i] -= host_v2[i] / alpha;
  vcl_v1   -=   vcl_v2 / alpha;

  if (check(host_v1, vcl_v1, epsilon) != EXIT_SUCCESS)
    return EXIT_FAILURE;

  std::cout << "Testing inplace division-subtract on vector with GPU scalar..." << std::endl;
  for (std::size_t i=0; i < host_v1.size(); ++i)
    host_v2[i] = NumericT(3.1415) * host_v1[i];
  proxy_copy(host_v1, vcl_v1);
  proxy_copy(host_v2, vcl_v2);

  for (std::size_t i=0; i < host_v1.size(); ++i)
    host_v1[i] -= host_v2[i] / alpha;
  vcl_v1   -=   vcl_v2 / gpu_alpha;

  if (check(host_v1, vcl_v1, epsilon) != EXIT_SUCCESS)
    return EXIT_FAILURE;


  std::cout << "Testing division-subtract on vector with GPU scalar (right)..." << std::endl;
  for (std::size_t i=0; i < host_v1.size(); ++i)
    host_v2[i] = NumericT(3.1415) * host_v1[i];
  proxy_copy(host_v1, vcl_v1);
  proxy_copy(host_v2, vcl_v2);

  for (std::size_t i=0; i < host_v1.size(); ++i)
    host_v1[i] = host_v1[i] - host_v2[i] / alpha;
  vcl_v1   = vcl_v1   -   vcl_v2 / gpu_alpha;

  if (check(host_v1, vcl_v1, epsilon) != EXIT_SUCCESS)
    return EXIT_FAILURE;

  std::cout << "Testing division-subtract on vector with GPU scalar (left)..." << std::endl;
  for (std::size_t i=0; i < host_v1.size(); ++i)
    host_v2[i] = NumericT(3.1415) * host_v1[i];
  proxy_copy(host_v1, vcl_v1);
  proxy_copy(host_v2, vcl_v2);

  for (std::size_t i=0; i < host_v1.size(); ++i)
    host_v1[i] = host_v1[i] - host_v2[i] / alpha;
  vcl_v1   = vcl_v1   -   vcl_v2 / gpu_alpha;

  if (check(host_v1, vcl_v1, epsilon) != EXIT_SUCCESS)
    return EXIT_FAILURE;

  std::cout << "Testing division-subtract on vector with GPU scalar (both)..." << std::endl;
  for (std::size_t i=0; i < host_v1.size(); ++i)
    host_v2[i] = NumericT(3.1415) * host_v1[i];
  proxy_copy(host_v1, vcl_v1);
  proxy_copy(host_v2, vcl_v2);

  for (std::size_t i=0; i < host_v1.size(); ++i)
    host_v1[i] = host_v1[i] / alpha     - host_v2[i] / beta;
  vcl_v1   =   vcl_v1 / gpu_alpha -   vcl_v2 / gpu_beta;

  if (check(host_v1, vcl_v1, epsilon) != EXIT_SUCCESS)
    return EXIT_FAILURE;

  std::cout << "Testing inplace division-subtract on vector with GPU scalar (both, adding)..." << std::endl;
  for (std::size_t i=0; i < host_v1.size(); ++i)
    host_v2[i] = NumericT(3.1415) * host_v1[i];
  proxy_copy(host_v1, vcl_v1);
  proxy_copy(host_v2, vcl_v2);

  for (std::size_t i=0; i < host_v1.size(); ++i)
    host_v1[i] -= host_v1[i] / alpha     + host_v2[i] / beta;
  vcl_v1   -=   vcl_v1 / gpu_alpha +   vcl_v2 / gpu_beta;

  if (check(host_v1, vcl_v1, epsilon) != EXIT_SUCCESS)
    return EXIT_FAILURE;

  std::cout << "Testing inplace division-subtract on vector with GPU scalar (both, subtracting)..." << std::endl;
  for (std::size_t i=0; i < host_v1.size(); ++i)
    host_v2[i] = NumericT(3.1415) * host_v1[i];
  proxy_copy(host_v1, vcl_v1);
  proxy_copy(host_v2, vcl_v2);

  for (std::size_t i=0; i < host_v1.size(); ++i)
    host_v1[i] -= host_v1[i] / alpha     - host_v2[i] / beta;
  vcl_v1   -=   vcl_v1 / gpu_alpha -   vcl_v2 / gpu_beta;

  if (check(host_v1, vcl_v1, epsilon) != EXIT_SUCCESS)
    return EXIT_FAILURE;

  std::cout << "Testing multiply-division-subtract on vector with GPU scalar..." << std::endl;
  for (std::size_t i=0; i < host_v1.size(); ++i)
    host_v2[i] = NumericT(3.1415) * host_v1[i];
  proxy_copy(host_v1, vcl_v1);
  proxy_copy(host_v2, vcl_v2);

  for (std::size_t i=0; i < host_v1.size(); ++i)
    host_v1[i] = host_v1[i] * alpha     - host_v2[i] / beta;
  vcl_v1   =   vcl_v1 * gpu_alpha -   vcl_v2 / gpu_beta;

  if (check(host_v1, vcl_v1, epsilon) != EXIT_SUCCESS)
    return EXIT_FAILURE;

  std::cout << "Testing division-multiply-subtract on vector with GPU scalar..." << std::endl;
  for (std::size_t i=0; i < host_v1.size(); ++i)
    host_v2[i] = NumericT(3.1415) * host_v1[i];
  proxy_copy(host_v1, vcl_v1);
  proxy_copy(host_v2, vcl_v2);

  for (std::size_t i=0; i < host_v1.size(); ++i)
    host_v1[i] = host_v1[i] / alpha     - host_v2[i] * beta;
  vcl_v1   =   vcl_v1 / gpu_alpha -   vcl_v2 * gpu_beta;

  if (check(host_v1, vcl_v1, epsilon) != EXIT_SUCCESS)
    return EXIT_FAILURE;

  std::cout << "Testing inplace multiply-division-subtract on vector with GPU scalar (adding)..." << std::endl;
  for (std::size_t i=0; i < host_v1.size(); ++i)
    host_v2[i] = NumericT(3.1415) * host_v1[i];
  proxy_copy(host_v1, vcl_v1);
  proxy_copy(host_v2, vcl_v2);

  for (std::size_t i=0; i < host_v1.size(); ++i)
    host_v1[i] -= host_v1[i] * alpha     + host_v2[i] / beta;
  vcl_v1   -=   vcl_v1 * gpu_alpha +   vcl_v2 / gpu_beta;

  if (check(host_v1, vcl_v1, epsilon) != EXIT_SUCCESS)
    return EXIT_FAILURE;

  std::cout << "Testing inplace division-multiply-subtract on vector with GPU scalar (adding)..." << std::endl;
  for (std::size_t i=0; i < host_v1.size(); ++i)
    host_v2[i] = NumericT(3.1415) * host_v1[i];
  proxy_copy(host_v1, vcl_v1);
  proxy_copy(host_v2, vcl_v2);

  for (std::size_t i=0; i < host_v1.size(); ++i)
    host_v1[i] -= host_v1[i] / alpha     + host_v2[i] * beta;
  vcl_v1   -=   vcl_v1 / gpu_alpha +   vcl_v2 * gpu_beta;

  if (check(host_v1, vcl_v1, epsilon) != EXIT_SUCCESS)
    return EXIT_FAILURE;

  std::cout << "Testing inplace multiply-division-subtract on vector with GPU scalar (subtracting)..." << std::endl;
  for (std::size_t i=0; i < host_v1.size(); ++i)
    host_v2[i] = NumericT(3.1415) * host_v1[i];
  proxy_copy(host_v1, vcl_v1);
  proxy_copy(host_v2, vcl_v2);

  for (std::size_t i=0; i < host_v1.size(); ++i)
    host_v1[i] -= host_v1[i] * alpha     - host_v2[i] / beta;
  vcl_v1   -=   vcl_v1 * gpu_alpha -   vcl_v2 / gpu_beta;

  if (check(host_v1, vcl_v1, epsilon) != EXIT_SUCCESS)
    return EXIT_FAILURE;

  std::cout << "Testing inplace division-multiply-subtract on vector with GPU scalar (subtracting)..." << std::endl;
  for (std::size_t i=0; i < host_v1.size(); ++i)
    host_v2[i] = NumericT(3.1415) * host_v1[i];
  proxy_copy(host_v1, vcl_v1);
  proxy_copy(host_v2, vcl_v2);

  for (std::size_t i=0; i < host_v1.size(); ++i)
    host_v1[i] -= host_v1[i] / alpha     - host_v2[i] * beta;
  vcl_v1   -=   vcl_v1 / gpu_alpha -   vcl_v2 * gpu_beta;

  if (check(host_v1, vcl_v1, epsilon) != EXIT_SUCCESS)
    return EXIT_FAILURE;


  std::cout << "Testing inplace division-subtract on vector with GPU scalar..." << std::endl;
  for (std::size_t i=0; i < host_v1.size(); ++i)
    host_v2[i] = NumericT(3.1415) * host_v1[i];
  proxy_copy(host_v1, vcl_v1);
  proxy_copy(host_v2, vcl_v2);

  for (std::size_t i=0; i < host_v1.size(); ++i)
    host_v1[i] -=     alpha * host_v2[i];
  vcl_v1   -= gpu_alpha *   vcl_v2;

  if (check(host_v1, vcl_v1, epsilon) != EXIT_SUCCESS)
    return EXIT_FAILURE;



  //
  // More complicated expressions (for ensuring the operator overloads work correctly)
  //
  for (std::size_t i=0; i < host_v1.size(); ++i)
  {
    host_v1[i] = NumericT(1.0) + randomNumber();
    host_v2[i] = NumericT(3.1415) * host_v1[i];
  }
  proxy_copy(host_v1, vcl_v1);
  proxy_copy(host_v2, vcl_v2);

  std::cout << "Testing three vector additions..." << std::endl;
  for (std::size_t i=0; i < host_v1.size(); ++i)
    host_v1[i] = host_v2[i] + host_v1[i] + host_v2[i];
  vcl_v1   =   vcl_v2 +   vcl_v1 +   vcl_v2;

  if (check(host_v1, vcl_v1, epsilon) != EXIT_SUCCESS)
    return EXIT_FAILURE;


  for (std::size_t i=0; i < host_v1.size(); ++i)
    host_v2[i] = NumericT(3.1415) * host_v1[i];
  proxy_copy(host_v1, vcl_v1);
  proxy_copy(host_v2, vcl_v2);

  std::cout << "Testing complicated vector expression with CPU scalar..." << std::endl;
  for (std::size_t i=0; i < host_v1.size(); ++i)
    host_v1[i] = beta * (host_v1[i] - alpha * host_v2[i]);
  vcl_v1   = beta * (vcl_v1   - alpha * vcl_v2);

  if (check(host_v1, vcl_v1, epsilon) != EXIT_SUCCESS)
    return EXIT_FAILURE;

  std::cout << "Testing complicated vector expression with GPU scalar..." << std::endl;
  for (std::size_t i=0; i < host_v1.size(); ++i)
    host_v1[i] =     beta * (host_v1[i] -     alpha * host_v2[i]);
  vcl_v1   = gpu_beta * (vcl_v1   - gpu_alpha * vcl_v2);

  if (check(host_v1, vcl_v1, epsilon) != EXIT_SUCCESS)
    return EXIT_FAILURE;

  // --------------------------------------------------------------------------
  for (std::size_t i=0; i < host_v1.size(); ++i)
    host_v2[i] = NumericT(3.1415) * host_v1[i];
  proxy_copy(host_v1, vcl_v1);
  proxy_copy(host_v2, vcl_v2);

  std::cout << "Testing swap..." << std::endl;
  for (std::size_t i=0; i < host_v1.size(); ++i)
  {
    NumericT temp = host_v1[i];
    host_v1[i] = host_v2[i];
    host_v2[i] = temp;
  }
  swap(vcl_v1, vcl_v2);

  if (check(host_v1, vcl_v1, epsilon) != EXIT_SUCCESS)
    return EXIT_FAILURE;

  // --------------------------------------------------------------------------
  for (std::size_t i=0; i<host_v1.size(); ++i)
  {
    host_v1[i] = NumericT(1.0) + randomNumber();
    host_v2[i] = NumericT(5.0) + randomNumber();
  }

  proxy_copy(host_v1, vcl_v1);
  proxy_copy(host_v2, vcl_v2);

  std::cout << "Testing unary operator-..." << std::endl;
  for (std::size_t i=0; i < host_v1.size(); ++i)
    host_v1[i] = - host_v2[i];
  vcl_v1   = -   vcl_v2;

  if (check(host_v1, vcl_v1, epsilon) != EXIT_SUCCESS)
    return EXIT_FAILURE;


  std::cout << "Testing elementwise multiplication..." << std::endl;
  std::cout << " v1 = element_prod(v1, v2);" << std::endl;
  for (std::size_t i=0; i < host_v1.size(); ++i)
    host_v1[i] = host_v1[i] * host_v2[i];
  vcl_v1 = viennacl::linalg::element_prod(vcl_v1, vcl_v2);

  if (check(host_v1, vcl_v1, epsilon) != EXIT_SUCCESS)
    return EXIT_FAILURE;

  std::cout << " v1 += element_prod(v1, v2);" << std::endl;
  for (std::size_t i=0; i < host_v1.size(); ++i)
    host_v1[i] += host_v1[i] * host_v2[i];
  vcl_v1 += viennacl::linalg::element_prod(vcl_v1, vcl_v2);

  if (check(host_v1, vcl_v1, epsilon) != EXIT_SUCCESS)
    return EXIT_FAILURE;

  std::cout << " v1 -= element_prod(v1, v2);" << std::endl;
  for (std::size_t i=0; i < host_v1.size(); ++i)
    host_v1[i] -= host_v1[i] * host_v2[i];
  vcl_v1 -= viennacl::linalg::element_prod(vcl_v1, vcl_v2);

  if (check(host_v1, vcl_v1, epsilon) != EXIT_SUCCESS)
    return EXIT_FAILURE;

  ///////
  std::cout << " v1 = element_prod(v1 + v2, v2);" << std::endl;
  for (std::size_t i=0; i < host_v1.size(); ++i)
    host_v1[i] = (host_v1[i] + host_v2[i]) * host_v2[i];
  vcl_v1 = viennacl::linalg::element_prod(vcl_v1 + vcl_v2, vcl_v2);

  if (check(host_v1, vcl_v1, epsilon) != EXIT_SUCCESS)
    return EXIT_FAILURE;

  std::cout << " v1 += element_prod(v1 + v2, v2);" << std::endl;
  for (std::size_t i=0; i < host_v1.size(); ++i)
    host_v1[i] += (host_v1[i] + host_v2[i]) * host_v2[i];
  vcl_v1 += viennacl::linalg::element_prod(vcl_v1 + vcl_v2, vcl_v2);

  if (check(host_v1, vcl_v1, epsilon) != EXIT_SUCCESS)
    return EXIT_FAILURE;

  std::cout << " v1 -= element_prod(v1 + v2, v2);" << std::endl;
  for (std::size_t i=0; i < host_v1.size(); ++i)
    host_v1[i] -= (host_v1[i] + host_v2[i]) * host_v2[i];
  vcl_v1 -= viennacl::linalg::element_prod(vcl_v1 + vcl_v2, vcl_v2);

  if (check(host_v1, vcl_v1, epsilon) != EXIT_SUCCESS)
    return EXIT_FAILURE;

  ///////
  std::cout << " v1 = element_prod(v1, v2 + v1);" << std::endl;
  for (std::size_t i=0; i < host_v1.size(); ++i)
    host_v1[i] = host_v1[i] * (host_v2[i] + host_v1[i]);
  vcl_v1 = viennacl::linalg::element_prod(vcl_v1, vcl_v2 + vcl_v1);

  if (check(host_v1, vcl_v1, epsilon) != EXIT_SUCCESS)
    return EXIT_FAILURE;

  std::cout << " v1 += element_prod(v1, v2 + v1);" << std::endl;
  for (std::size_t i=0; i < host_v1.size(); ++i)
    host_v1[i] += host_v1[i] * (host_v2[i] + host_v1[i]);
  vcl_v1 += viennacl::linalg::element_prod(vcl_v1, vcl_v2 + vcl_v1);

  if (check(host_v1, vcl_v1, epsilon) != EXIT_SUCCESS)
    return EXIT_FAILURE;

  std::cout << " v1 -= element_prod(v1, v2 + v1);" << std::endl;
  for (std::size_t i=0; i < host_v1.size(); ++i)
    host_v1[i] -= host_v1[i] * (host_v2[i] + host_v1[i]);
  vcl_v1 -= viennacl::linalg::element_prod(vcl_v1, vcl_v2 + vcl_v1);

  if (check(host_v1, vcl_v1, epsilon) != EXIT_SUCCESS)
    return EXIT_FAILURE;

  ///////
  std::cout << " v1 = element_prod(v1 + v2, v2 + v1);" << std::endl;
  for (std::size_t i=0; i < host_v1.size(); ++i)
    host_v1[i] = (host_v1[i] + host_v2[i]) * (host_v2[i] + host_v1[i]);
  vcl_v1 = viennacl::linalg::element_prod(vcl_v1 + vcl_v2, vcl_v2 + vcl_v1);

  if (check(host_v1, vcl_v1, epsilon) != EXIT_SUCCESS)
    return EXIT_FAILURE;

  std::cout << " v1 += element_prod(v1 + v2, v2 + v1);" << std::endl;
  for (std::size_t i=0; i < host_v1.size(); ++i)
    host_v1[i] += (host_v1[i] + host_v2[i]) * (host_v2[i] + host_v1[i]);
  vcl_v1 += viennacl::linalg::element_prod(vcl_v1 + vcl_v2, vcl_v2 + vcl_v1);

  if (check(host_v1, vcl_v1, epsilon) != EXIT_SUCCESS)
    return EXIT_FAILURE;

  std::cout << " v1 -= element_prod(v1 + v2, v2 + v1);" << std::endl;
  for (std::size_t i=0; i < host_v1.size(); ++i)
    host_v1[i] -= (host_v1[i] + host_v2[i]) * (host_v2[i] + host_v1[i]);
  vcl_v1 -= viennacl::linalg::element_prod(vcl_v1 + vcl_v2, vcl_v2 + vcl_v1);

  if (check(host_v1, vcl_v1, epsilon) != EXIT_SUCCESS)
    return EXIT_FAILURE;


  std::cout << "Testing elementwise division..." << std::endl;
  for (std::size_t i=0; i<host_v1.size(); ++i)
  {
    host_v1[i] = NumericT(1.0) + randomNumber();
    host_v2[i] = NumericT(5.0) + randomNumber();
  }

  proxy_copy(host_v1, vcl_v1);
  proxy_copy(host_v2, vcl_v2);

  for (std::size_t i=0; i < host_v1.size(); ++i)
    host_v1[i] = host_v1[i] / host_v2[i];
  vcl_v1 = viennacl::linalg::element_div(vcl_v1, vcl_v2);

  if (check(host_v1, vcl_v1, epsilon) != EXIT_SUCCESS)
    return EXIT_FAILURE;

  for (std::size_t i=0; i < host_v1.size(); ++i)
    host_v1[i] += host_v1[i] / host_v2[i];
  vcl_v1 += viennacl::linalg::element_div(vcl_v1, vcl_v2);

  if (check(host_v1, vcl_v1, epsilon) != EXIT_SUCCESS)
    return EXIT_FAILURE;

  for (std::size_t i=0; i < host_v1.size(); ++i)
    host_v1[i] -= host_v1[i] / host_v2[i];
  vcl_v1 -= viennacl::linalg::element_div(vcl_v1, vcl_v2);

  if (check(host_v1, vcl_v1, epsilon) != EXIT_SUCCESS)
    return EXIT_FAILURE;

  ///////
  for (std::size_t i=0; i < host_v1.size(); ++i)
    host_v1[i] = (host_v1[i] + host_v2[i]) / host_v2[i];
  vcl_v1 = viennacl::linalg::element_div(vcl_v1 + vcl_v2, vcl_v2);

  if (check(host_v1, vcl_v1, epsilon) != EXIT_SUCCESS)
    return EXIT_FAILURE;

  for (std::size_t i=0; i < host_v1.size(); ++i)
    host_v1[i] += (host_v1[i] + host_v2[i]) / host_v2[i];
  vcl_v1 += viennacl::linalg::element_div(vcl_v1 + vcl_v2, vcl_v2);

  if (check(host_v1, vcl_v1, epsilon) != EXIT_SUCCESS)
    return EXIT_FAILURE;

  for (std::size_t i=0; i < host_v1.size(); ++i)
    host_v1[i] -= (host_v1[i] + host_v2[i]) / host_v2[i];
  vcl_v1 -= viennacl::linalg::element_div(vcl_v1 + vcl_v2, vcl_v2);

  if (check(host_v1, vcl_v1, epsilon) != EXIT_SUCCESS)
    return EXIT_FAILURE;

  ///////
  for (std::size_t i=0; i < host_v1.size(); ++i)
    host_v1[i] = host_v1[i] / (host_v2[i] + host_v1[i]);
  vcl_v1 = viennacl::linalg::element_div(vcl_v1, vcl_v2 + vcl_v1);

  if (check(host_v1, vcl_v1, epsilon) != EXIT_SUCCESS)
    return EXIT_FAILURE;

  for (std::size_t i=0; i < host_v1.size(); ++i)
    host_v1[i] += host_v1[i] / (host_v2[i] + host_v1[i]);
  vcl_v1 += viennacl::linalg::element_div(vcl_v1, vcl_v2 + vcl_v1);

  if (check(host_v1, vcl_v1, epsilon) != EXIT_SUCCESS)
    return EXIT_FAILURE;

  for (std::size_t i=0; i < host_v1.size(); ++i)
    host_v1[i] -= host_v1[i] / (host_v2[i] + host_v1[i]);
  vcl_v1 -= viennacl::linalg::element_div(vcl_v1, vcl_v2 + vcl_v1);

  if (check(host_v1, vcl_v1, epsilon) != EXIT_SUCCESS)
    return EXIT_FAILURE;

  ///////
  for (std::size_t i=0; i < host_v1.size(); ++i)
    host_v1[i] = (host_v1[i] + host_v2[i]) / (host_v2[i] + host_v1[i]);
  vcl_v1 = viennacl::linalg::element_div(vcl_v1 + vcl_v2, vcl_v2 + vcl_v1);

  if (check(host_v1, vcl_v1, epsilon) != EXIT_SUCCESS)
    return EXIT_FAILURE;

  for (std::size_t i=0; i < host_v1.size(); ++i)
    host_v1[i] += (host_v1[i] + host_v2[i]) / (host_v2[i] + host_v1[i]);
  vcl_v1 += viennacl::linalg::element_div(vcl_v1 + vcl_v2, vcl_v2 + vcl_v1);

  if (check(host_v1, vcl_v1, epsilon) != EXIT_SUCCESS)
    return EXIT_FAILURE;

  for (std::size_t i=0; i < host_v1.size(); ++i)
    host_v1[i] -= (host_v1[i] + host_v2[i]) / (host_v2[i] + host_v1[i]);
  vcl_v1 -= viennacl::linalg::element_div(vcl_v1 + vcl_v2, vcl_v2 + vcl_v1);

  if (check(host_v1, vcl_v1, epsilon) != EXIT_SUCCESS)
    return EXIT_FAILURE;


  std::cout << "Testing elementwise power function..." << std::endl;
  for (std::size_t i=0; i<host_v1.size(); ++i)
  {
    host_v1[i] = NumericT(1.1) + NumericT(0.5) * randomNumber();
    host_v2[i] = NumericT(1.1) + NumericT(0.5) * randomNumber();
  }
  std::vector<NumericT> std_v3(host_v1.size());
  vector_proxy<NumericT> host_v3(&std_v3[0], 0, 1, host_v1.size());

  proxy_copy(host_v1, vcl_v1);
  proxy_copy(host_v2, vcl_v2);

  for (std::size_t i=0; i<host_v3.size(); ++i)
    host_v3[i] = std::pow(host_v1[i], host_v2[i]);
  vcl_v1 = viennacl::linalg::element_pow(vcl_v1, vcl_v2);

  if (check(host_v3, vcl_v1, epsilon) != EXIT_SUCCESS)
  {
    std::cerr << "** Failure in v1 = pow(v1, v2);" << std::endl;
    return EXIT_FAILURE;
  }

  proxy_copy(host_v1, vcl_v1);
  for (std::size_t i=0; i<host_v3.size(); ++i)
    host_v3[i] = host_v1[i];
  for (std::size_t i=0; i<host_v3.size(); ++i)
    host_v3[i] += std::pow(host_v1[i], host_v2[i]);
  vcl_v1 += viennacl::linalg::element_pow(vcl_v1, vcl_v2);

  if (check(host_v3, vcl_v1, epsilon) != EXIT_SUCCESS)
  {
    std::cerr << "** Failure in v1 += pow(v1, v2);" << std::endl;
    return EXIT_FAILURE;
  }

  proxy_copy(host_v1, vcl_v1);
  for (std::size_t i=0; i<host_v3.size(); ++i)
    host_v3[i] = host_v1[i];
  for (std::size_t i=0; i<host_v3.size(); ++i)
    host_v3[i] -= std::pow(host_v1[i], host_v2[i]);
  vcl_v1 -= viennacl::linalg::element_pow(vcl_v1, vcl_v2);

  if (check(host_v3, vcl_v1, epsilon) != EXIT_SUCCESS)
  {
    std::cerr << "** Failure in v1 -= pow(v1, v2);" << std::endl;
    return EXIT_FAILURE;
  }

  ///////
  proxy_copy(host_v1, vcl_v1);
  for (std::size_t i=0; i<host_v3.size(); ++i)
    host_v3[i] = host_v1[i];
  for (std::size_t i=0; i<host_v3.size(); ++i)
    host_v3[i] = std::pow(host_v1[i] + host_v2[i], host_v2[i]);
  vcl_v1 = viennacl::linalg::element_pow(vcl_v1 + vcl_v2, vcl_v2);

  if (check(host_v3, vcl_v1, epsilon) != EXIT_SUCCESS)
  {
    std::cerr << "** Failure in v1 = pow(v1 + v2, v2);" << std::endl;
    return EXIT_FAILURE;
  }

  proxy_copy(host_v1, vcl_v1);
  for (std::size_t i=0; i<host_v3.size(); ++i)
    host_v3[i] = host_v1[i];
  for (std::size_t i=0; i<host_v3.size(); ++i)
    host_v3[i] += std::pow(host_v1[i] + host_v2[i], host_v2[i]);
  vcl_v1 += viennacl::linalg::element_pow(vcl_v1 + vcl_v2, vcl_v2);

  if (check(host_v3, vcl_v1, epsilon) != EXIT_SUCCESS)
  {
    std::cerr << "** Failure in v1 += pow(v1 + v2, v2);" << std::endl;
    return EXIT_FAILURE;
  }

  proxy_copy(host_v1, vcl_v1);
  for (std::size_t i=0; i<host_v3.size(); ++i)
    host_v3[i] = host_v1[i];
  for (std::size_t i=0; i<host_v3.size(); ++i)
    host_v3[i] -= std::pow(host_v1[i] + host_v2[i], host_v2[i]);
  vcl_v1 -= viennacl::linalg::element_pow(vcl_v1 + vcl_v2, vcl_v2);

  if (check(host_v3, vcl_v1, epsilon) != EXIT_SUCCESS)
  {
    std::cerr << "** Failure in v1 -= pow(v1 + v2, v2);" << std::endl;
    return EXIT_FAILURE;
  }

  ///////
  proxy_copy(host_v1, vcl_v1);
  for (std::size_t i=0; i<host_v3.size(); ++i)
    host_v3[i] = host_v1[i];
  for (std::size_t i=0; i<host_v3.size(); ++i)
    host_v3[i] = std::pow(host_v1[i], host_v2[i] + host_v1[i]);
  vcl_v1 = viennacl::linalg::element_pow(vcl_v1, vcl_v2 + vcl_v1);

  if (check(host_v3, vcl_v1, epsilon) != EXIT_SUCCESS)
  {
    std::cerr << "** Failure in v1 = pow(v1, v2 + v1);" << std::endl;
    return EXIT_FAILURE;
  }

  proxy_copy(host_v1, vcl_v1);
  for (std::size_t i=0; i<host_v3.size(); ++i)
    host_v3[i] = host_v1[i];
  for (std::size_t i=0; i<host_v3.size(); ++i)
    host_v3[i] += std::pow(host_v1[i], host_v2[i] + host_v1[i]);
  vcl_v1 += viennacl::linalg::element_pow(vcl_v1, vcl_v2 + vcl_v1);

  if (check(host_v3, vcl_v1, epsilon) != EXIT_SUCCESS)
  {
    std::cerr << "** Failure in v1 += pow(v1, v2 + v1);" << std::endl;
    return EXIT_FAILURE;
  }

  proxy_copy(host_v1, vcl_v1);
  for (std::size_t i=0; i<host_v3.size(); ++i)
    host_v3[i] = host_v1[i];
  for (std::size_t i=0; i<host_v3.size(); ++i)
    host_v3[i] -= std::pow(host_v1[i], host_v2[i] + host_v1[i]);
  vcl_v1 -= viennacl::linalg::element_pow(vcl_v1, vcl_v2 + vcl_v1);

  if (check(host_v3, vcl_v1, epsilon) != EXIT_SUCCESS)
  {
    std::cerr << "** Failure in v1 -= pow(v1, v2 + v1);" << std::endl;
    return EXIT_FAILURE;
  }

  ///////
  proxy_copy(host_v1, vcl_v1);
  for (std::size_t i=0; i<host_v3.size(); ++i)
    host_v3[i] = host_v1[i];
  for (std::size_t i=0; i<host_v3.size(); ++i)
    host_v3[i] = std::pow(host_v1[i] + host_v2[i], host_v2[i] + host_v1[i]);
  vcl_v1 = viennacl::linalg::element_pow(vcl_v1 + vcl_v2, vcl_v2 + vcl_v1);

  if (check(host_v3, vcl_v1, epsilon) != EXIT_SUCCESS)
  {
    std::cerr << "** Failure in v1 = pow(v1 + v2, v2 + v1);" << std::endl;
    return EXIT_FAILURE;
  }

  proxy_copy(host_v1, vcl_v1);
  for (std::size_t i=0; i<host_v3.size(); ++i)
    host_v3[i] = host_v1[i];
  for (std::size_t i=0; i<host_v3.size(); ++i)
    host_v3[i] += std::pow(host_v1[i] + host_v2[i], host_v2[i] + host_v1[i]);
  vcl_v1 += viennacl::linalg::element_pow(vcl_v1 + vcl_v2, vcl_v2 + vcl_v1);

  if (check(host_v3, vcl_v1, epsilon) != EXIT_SUCCESS)
  {
    std::cerr << "** Failure in v1 += pow(v1 + v2, v2 + v1);" << std::endl;
    return EXIT_FAILURE;
  }

  proxy_copy(host_v1, vcl_v1);
  for (std::size_t i=0; i<host_v3.size(); ++i)
    host_v3[i] = host_v1[i];
  for (std::size_t i=0; i<host_v3.size(); ++i)
    host_v3[i] -= std::pow(host_v1[i] + host_v2[i], host_v2[i] + host_v1[i]);
  vcl_v1 -= viennacl::linalg::element_pow(vcl_v1 + vcl_v2, vcl_v2 + vcl_v1);

  if (check(host_v3, vcl_v1, epsilon) != EXIT_SUCCESS)
  {
    std::cerr << "** Failure in v1 -= pow(v1 + v2, v2 + v1);" << std::endl;
    return EXIT_FAILURE;
  }

  std::cout << "Testing unary elementwise operations..." << std::endl;
  for (size_t i=0; i < host_v1.size(); ++i)
    host_v1[i] = randomNumber() / NumericT(4);

#define GENERATE_UNARY_OP_TEST(FUNCNAME) \
  for (std::size_t i=0; i<host_v1.size(); ++i) \
  host_v2[i] = NumericT(3.1415) * host_v1[i]; \
  proxy_copy(host_v1, vcl_v1); \
  proxy_copy(host_v2, vcl_v2); \
  \
  for (std::size_t i=0; i<host_v1.size(); ++i) \
    host_v1[i] = std::FUNCNAME(host_v2[i]); \
  vcl_v1 = viennacl::linalg::element_##FUNCNAME(vcl_v2); \
 \
  if (check(host_v1, vcl_v1, epsilon) != EXIT_SUCCESS) \
  { \
    std::cout << "Failure at v1 = " << #FUNCNAME << "(v2)" << std::endl; \
    return EXIT_FAILURE; \
  } \
 \
  for (std::size_t i=0; i<host_v1.size(); ++i) \
    host_v1[i] = std::FUNCNAME(host_v1[i] + host_v2[i]); \
  vcl_v1 = viennacl::linalg::element_##FUNCNAME(vcl_v1 + vcl_v2); \
 \
  if (check(host_v1, vcl_v1, epsilon) != EXIT_SUCCESS) \
  { \
    std::cout << "Failure at v1 = " << #FUNCNAME << "(v1 + v2)" << std::endl; \
    return EXIT_FAILURE; \
  } \
 \
  for (std::size_t i=0; i<host_v1.size(); ++i) \
    host_v1[i] += std::FUNCNAME(host_v1[i]); \
  vcl_v1 += viennacl::linalg::element_##FUNCNAME(vcl_v1); \
 \
  if (check(host_v1, vcl_v1, epsilon) != EXIT_SUCCESS) \
  { \
    std::cout << "Failure at v1 += " << #FUNCNAME << "(v2)" << std::endl; \
    return EXIT_FAILURE; \
  } \
 \
  for (std::size_t i=0; i<host_v1.size(); ++i) \
    host_v1[i] += std::FUNCNAME(host_v1[i] + host_v2[i]); \
  vcl_v1 += viennacl::linalg::element_##FUNCNAME(vcl_v1 + vcl_v2); \
 \
  if (check(host_v1, vcl_v1, epsilon) != EXIT_SUCCESS) \
  { \
    std::cout << "Failure at v1 += " << #FUNCNAME << "(v1 + v2)" << std::endl; \
    return EXIT_FAILURE; \
  } \
 \
  for (std::size_t i=0; i<host_v1.size(); ++i) \
    host_v1[i] -= std::FUNCNAME(host_v2[i]); \
  vcl_v1 -= viennacl::linalg::element_##FUNCNAME(vcl_v2); \
 \
  if (check(host_v1, vcl_v1, epsilon) != EXIT_SUCCESS) \
  { \
    std::cout << "Failure at v1 -= " << #FUNCNAME << "(v2)" << std::endl; \
    return EXIT_FAILURE; \
  } \
 \
  for (std::size_t i=0; i<host_v1.size(); ++i) \
    host_v1[i] -= std::FUNCNAME(host_v1[i] + host_v2[i]); \
  vcl_v1 -= viennacl::linalg::element_##FUNCNAME(vcl_v1 + vcl_v2); \
 \
  if (check(host_v1, vcl_v1, epsilon) != EXIT_SUCCESS) \
  { \
    std::cout << "Failure at v1 -= " << #FUNCNAME << "(v1 + v2)" << std::endl; \
    return EXIT_FAILURE; \
  } \

  GENERATE_UNARY_OP_TEST(cos);
  GENERATE_UNARY_OP_TEST(cosh);
  for (std::size_t i=0; i < host_v1.size(); ++i)
    host_v1[i] = randomNumber() / NumericT(4);
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

  // --------------------------------------------------------------------------
  for (std::size_t i=0; i<host_v1.size(); ++i)
    host_v2[i] = NumericT(3.1415) * host_v1[i];
  proxy_copy(host_v1, vcl_v1);
  proxy_copy(host_v2, vcl_v2);

  std::cout << "Testing another complicated vector expression with CPU scalars..." << std::endl;
  for (std::size_t i=0; i<host_v1.size(); ++i)
    host_v1[i] = host_v2[i] / alpha + beta * (host_v1[i] - alpha*host_v2[i]);
  vcl_v1   = vcl_v2 / alpha   + beta * (vcl_v1   - alpha*vcl_v2);

  if (check(host_v1, vcl_v1, epsilon) != EXIT_SUCCESS)
    return EXIT_FAILURE;

  std::cout << "Testing another complicated vector expression with GPU scalars..." << std::endl;
  for (std::size_t i=0; i<host_v1.size(); ++i)
    host_v2[i] = NumericT(3.1415) * host_v1[i];
  proxy_copy(host_v1, vcl_v1);
  proxy_copy(host_v2, vcl_v2);

  for (std::size_t i=0; i<host_v1.size(); ++i)
    host_v1[i] = host_v2[i] / alpha   +     beta * (host_v1[i] - alpha*host_v2[i]);
  vcl_v1   = vcl_v2 / gpu_alpha + gpu_beta * (vcl_v1   - gpu_alpha*vcl_v2);

  if (check(host_v1, vcl_v1, epsilon) != EXIT_SUCCESS)
    return EXIT_FAILURE;


  std::cout << "Testing lenghty sum of scaled vectors..." << std::endl;
  for (std::size_t i=0; i<host_v1.size(); ++i)
    host_v2[i] = NumericT(3.1415) * host_v1[i];
  proxy_copy(host_v1, vcl_v1);
  proxy_copy(host_v2, vcl_v2);

  for (std::size_t i=0; i<host_v1.size(); ++i)
    host_v1[i] = host_v2[i] / alpha   +     beta * host_v1[i] - alpha * host_v2[i] + beta * host_v1[i] - alpha * host_v1[i];
  vcl_v1   = vcl_v2 / gpu_alpha + gpu_beta *   vcl_v1 - alpha *   vcl_v2 + beta *   vcl_v1 - alpha *   vcl_v1;

  if (check(host_v1, vcl_v1, epsilon) != EXIT_SUCCESS)
    return EXIT_FAILURE;

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
  // Set up host objects
  //
  std::vector<NumericT> std_full_vec(size);
  std::vector<NumericT> std_full_vec2(std_full_vec.size());

  for (std::size_t i=0; i<std_full_vec.size(); ++i)
  {
    std_full_vec[i]  = NumericT(1.0) + randomNumber();
    std_full_vec2[i] = NumericT(1.0) + randomNumber();
  }

  std::size_t r1_start = std_full_vec.size() / 4;
  std::size_t r1_stop  = 2 * std_full_vec.size() / 4;
  std::size_t r2_start = 2 * std_full_vec2.size() / 4;
  std::size_t r2_stop  = 3 * std_full_vec2.size() / 4;
  vector_proxy<NumericT> host_range_vec (&std_full_vec[0],  r1_start, 1, r1_stop - r1_start);
  vector_proxy<NumericT> host_range_vec2(&std_full_vec2[0], r2_start, 1, r2_stop - r2_start);

  std::size_t s1_start = std_full_vec.size() / 4;
  std::size_t s1_inc   = 3;
  std::size_t s1_size  = std_full_vec.size() / 4;
  std::size_t s2_start = 2 * std_full_vec2.size() / 4;
  std::size_t s2_inc   = 2;
  std::size_t s2_size  = std_full_vec2.size() / 4;
  vector_proxy<NumericT> host_slice_vec (&std_full_vec[0],  s1_start, s1_inc, s1_size);
  vector_proxy<NumericT> host_slice_vec2(&std_full_vec2[0], s2_start, s2_inc, s2_size);

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

    std::vector<NumericT> std_short_vec(host_range_vec.size());
    for (std::size_t i=0; i<std_short_vec.size(); ++i)
      std_short_vec[i] = host_range_vec[i];
    vector_proxy<NumericT> host_short_vec(&std_short_vec[0], 0, 1, std_short_vec.size());

    std::vector<NumericT> std_short_vec2(host_range_vec2.size());
    for (std::size_t i=0; i<std_short_vec2.size(); ++i)
      std_short_vec2[i] = host_range_vec2[i];
    vector_proxy<NumericT> host_short_vec2(&std_short_vec2[0], 0, 1, std_short_vec.size());

    std::cout << "Testing creation of vectors from range..." << std::endl;
    if (check(host_short_vec, vcl_short_vec, epsilon) != EXIT_SUCCESS)
      return EXIT_FAILURE;
    if (check(host_short_vec2, vcl_short_vec2, epsilon) != EXIT_SUCCESS)
      return EXIT_FAILURE;
  }

  viennacl::slice vcl_s1(    vcl_full_vec.size() / 4, 3, vcl_full_vec.size() / 4);
  viennacl::slice vcl_s2(2 * vcl_full_vec2.size() / 4, 2, vcl_full_vec2.size() / 4);
  viennacl::vector_slice< viennacl::vector<NumericT> > vcl_slice_vec(vcl_full_vec, vcl_s1);
  viennacl::vector_slice< viennacl::vector<NumericT> > vcl_slice_vec2(vcl_full_vec2, vcl_s2);

  viennacl::vector<NumericT> vcl_short_vec(vcl_slice_vec);
  viennacl::vector<NumericT> vcl_short_vec2 = vcl_slice_vec2;

  std::vector<NumericT> std_short_vec(host_slice_vec.size());
  for (std::size_t i=0; i<std_short_vec.size(); ++i)
    std_short_vec[i] = host_slice_vec[i];
  vector_proxy<NumericT> host_short_vec(&std_short_vec[0], 0, 1, std_short_vec.size());

  std::vector<NumericT> std_short_vec2(host_slice_vec2.size());
  for (std::size_t i=0; i<std_short_vec2.size(); ++i)
    std_short_vec2[i] = host_slice_vec2[i];
  vector_proxy<NumericT> host_short_vec2(&std_short_vec2[0], 0, 1, std_short_vec.size());

  std::cout << "Testing creation of vectors from slice..." << std::endl;
  if (check(host_short_vec, vcl_short_vec, epsilon) != EXIT_SUCCESS)
    return EXIT_FAILURE;
  if (check(host_short_vec2, vcl_short_vec2, epsilon) != EXIT_SUCCESS)
    return EXIT_FAILURE;


  //
  // Now start running tests for vectors, ranges and slices:
  //

  std::cout << " ** vcl_v1 = vector, vcl_v2 = vector **" << std::endl;
  retval = test<NumericT>(epsilon,
                          host_short_vec, host_short_vec2,
                          vcl_short_vec, vcl_short_vec2);
  if (retval != EXIT_SUCCESS)
    return EXIT_FAILURE;

  std::cout << " ** vcl_v1 = vector, vcl_v2 = range **" << std::endl;
  retval = test<NumericT>(epsilon,
                          host_short_vec, host_short_vec2,
                          vcl_short_vec, vcl_range_vec2);
  if (retval != EXIT_SUCCESS)
    return EXIT_FAILURE;

  std::cout << " ** vcl_v1 = vector, vcl_v2 = slice **" << std::endl;
  retval = test<NumericT>(epsilon,
                          host_short_vec, host_short_vec2,
                          vcl_short_vec, vcl_slice_vec2);
  if (retval != EXIT_SUCCESS)
    return EXIT_FAILURE;

  ///////

  std::cout << " ** vcl_v1 = range, vcl_v2 = vector **" << std::endl;
  retval = test<NumericT>(epsilon,
                          host_short_vec, host_short_vec2,
                          vcl_range_vec, vcl_short_vec2);
  if (retval != EXIT_SUCCESS)
    return EXIT_FAILURE;

  std::cout << " ** vcl_v1 = range, vcl_v2 = range **" << std::endl;
  retval = test<NumericT>(epsilon,
                          host_short_vec, host_short_vec2,
                          vcl_range_vec, vcl_range_vec2);
  if (retval != EXIT_SUCCESS)
    return EXIT_FAILURE;

  std::cout << " ** vcl_v1 = range, vcl_v2 = slice **" << std::endl;
  retval = test<NumericT>(epsilon,
                          host_short_vec, host_short_vec2,
                          vcl_range_vec, vcl_slice_vec2);
  if (retval != EXIT_SUCCESS)
    return EXIT_FAILURE;

  ///////

  std::cout << " ** vcl_v1 = slice, vcl_v2 = vector **" << std::endl;
  retval = test<NumericT>(epsilon,
                          host_short_vec, host_short_vec2,
                          vcl_slice_vec, vcl_short_vec2);
  if (retval != EXIT_SUCCESS)
    return EXIT_FAILURE;

  std::cout << " ** vcl_v1 = slice, vcl_v2 = range **" << std::endl;
  retval = test<NumericT>(epsilon,
                          host_short_vec, host_short_vec2,
                          vcl_slice_vec, vcl_range_vec2);
  if (retval != EXIT_SUCCESS)
    return EXIT_FAILURE;

  std::cout << " ** vcl_v1 = slice, vcl_v2 = slice **" << std::endl;
  retval = test<NumericT>(epsilon,
                          host_short_vec, host_short_vec2,
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
    NumericT epsilon = static_cast<NumericT>(1.0E-2);
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
      NumericT epsilon = 1.0E-10;
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
