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



/** \file tests/src/vector_multi_inner_prod.cpp  Tests the performance of multiple inner products with a common vector.
*   \test   Tests the performance of multiple inner products with a common vector.
**/

//
// *** System
//
#include <iostream>
#include <iomanip>
#include <iterator>

//
// *** ViennaCL
//
#include "viennacl/vector.hpp"
#include "viennacl/vector_proxy.hpp"
#include "viennacl/linalg/inner_prod.hpp"
#include "viennacl/linalg/norm_1.hpp"
#include "viennacl/linalg/norm_2.hpp"
#include "viennacl/linalg/norm_inf.hpp"
#include "viennacl/tools/random.hpp"

//
// -------------------------------------------------------------
//
template<typename ScalarType>
ScalarType diff(ScalarType const & s1, ScalarType const & s2)
{
   viennacl::backend::finish();
   if (s1 != s2)
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
   if (s1 != s2)
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
   if (s1 != s2)
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

   for (unsigned int i=0;i<v1.size(); ++i)
   {
      if ( std::max( std::fabs(v2_cpu[i]), std::fabs(v1[i]) ) > 0 )
         v2_cpu[i] = std::fabs(v2_cpu[i] - v1[i]) / std::max( std::fabs(v2_cpu[i]), std::fabs(v1[i]) );
      else
         v2_cpu[i] = 0.0;
   }

   ScalarType norm_inf = 0;
   for (std::size_t i=0; i<v2_cpu.size(); ++i)
     norm_inf = std::max<ScalarType>(norm_inf, std::fabs(v2_cpu[i]));

   return norm_inf;
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
template< typename NumericT, typename Epsilon,
          typename STLVectorType1,      typename STLVectorType2,      typename STLVectorType3,      typename STLVectorType4,
          typename ViennaCLVectorType1, typename ViennaCLVectorType2, typename ViennaCLVectorType3, typename ViennaCLVectorType4 >
int test(Epsilon const& epsilon,
         STLVectorType1      & std_v1, STLVectorType2      & std_v2, STLVectorType3      & std_v3, STLVectorType4      & std_v4,
         ViennaCLVectorType1 & vcl_v1, ViennaCLVectorType2 & vcl_v2, ViennaCLVectorType3 & vcl_v3, ViennaCLVectorType4 & vcl_v4)
{
  int retval = EXIT_SUCCESS;

  viennacl::tools::uniform_random_numbers<NumericT> randomNumber;

  for (std::size_t i=0; i<std_v1.size(); ++i)
  {
    std_v1[i] = NumericT(1.0) + randomNumber();
    std_v2[i] = NumericT(1.0) + randomNumber();
    std_v3[i] = NumericT(1.0) + randomNumber();
    std_v4[i] = NumericT(1.0) + randomNumber();
  }

  viennacl::copy(std_v1.begin(), std_v1.end(), vcl_v1.begin());  //resync
  viennacl::copy(std_v2.begin(), std_v2.end(), vcl_v2.begin());
  viennacl::copy(std_v3.begin(), std_v3.end(), vcl_v3.begin());
  viennacl::copy(std_v4.begin(), std_v4.end(), vcl_v4.begin());

  std::cout << "Checking for successful copy..." << std::endl;
  if (check(std_v1, vcl_v1, epsilon) != EXIT_SUCCESS)
    return EXIT_FAILURE;
  if (check(std_v2, vcl_v2, epsilon) != EXIT_SUCCESS)
    return EXIT_FAILURE;
  if (check(std_v3, vcl_v3, epsilon) != EXIT_SUCCESS)
    return EXIT_FAILURE;
  if (check(std_v4, vcl_v4, epsilon) != EXIT_SUCCESS)
    return EXIT_FAILURE;

  std::vector<NumericT> ref_result(40, 0.0);
  viennacl::vector<NumericT> result = viennacl::scalar_vector<NumericT>(40, 0.0);

  std::cout << "Testing inner_prod with two vectors..." << std::endl;
  ref_result[2] = 0; for (std::size_t i=0; i<std_v1.size(); ++i) ref_result[2] += std_v1[i] * std_v1[i];
  ref_result[5] = 0; for (std::size_t i=0; i<std_v1.size(); ++i) ref_result[5] += std_v1[i] * std_v2[i];
  viennacl::project(result, viennacl::slice(2, 3, 2)) = viennacl::linalg::inner_prod(vcl_v1, viennacl::tie(vcl_v1, vcl_v2));
  if (check(ref_result, result, epsilon) != EXIT_SUCCESS)
  {
    std::copy(ref_result.begin(), ref_result.end(), std::ostream_iterator<NumericT>(std::cout, " ")); std::cout << std::endl;
    std::cout << result << std::endl;
    return EXIT_FAILURE;
  }

  ref_result[3] = 0; for (std::size_t i=0; i<std_v1.size(); ++i) ref_result[3] += std_v1[i] * std_v3[i];
  ref_result[7] = 0; for (std::size_t i=0; i<std_v1.size(); ++i) ref_result[7] += std_v1[i] * std_v4[i];
  viennacl::project(result, viennacl::slice(3, 4, 2)) = viennacl::linalg::inner_prod(vcl_v1, viennacl::tie(vcl_v3, vcl_v4));
  if (check(ref_result, result, epsilon) != EXIT_SUCCESS)
  {
    std::copy(ref_result.begin(), ref_result.end(), std::ostream_iterator<NumericT>(std::cout, " ")); std::cout << std::endl;
    std::cout << result << std::endl;
    return EXIT_FAILURE;
  }


  std::cout << "Testing inner_prod with three vectors..." << std::endl;
  ref_result[1] = 0; for (std::size_t i=0; i<std_v1.size(); ++i) ref_result[1] += std_v1[i] * std_v1[i];
  ref_result[3] = 0; for (std::size_t i=0; i<std_v1.size(); ++i) ref_result[3] += std_v1[i] * std_v2[i];
  ref_result[5] = 0; for (std::size_t i=0; i<std_v1.size(); ++i) ref_result[5] += std_v1[i] * std_v3[i];
  viennacl::project(result, viennacl::slice(1, 2, 3)) = viennacl::linalg::inner_prod(vcl_v1, viennacl::tie(vcl_v1, vcl_v2, vcl_v3));
  if (check(ref_result, result, epsilon) != EXIT_SUCCESS)
  {
    std::copy(ref_result.begin(), ref_result.end(), std::ostream_iterator<NumericT>(std::cout, " ")); std::cout << std::endl;
    std::cout << result << std::endl;
    return EXIT_FAILURE;
  }

  ref_result[2]  = 0; for (std::size_t i=0; i<std_v1.size(); ++i) ref_result[2]  += std_v1[i] * std_v3[i];
  ref_result[6]  = 0; for (std::size_t i=0; i<std_v1.size(); ++i) ref_result[6]  += std_v1[i] * std_v2[i];
  ref_result[10] = 0; for (std::size_t i=0; i<std_v1.size(); ++i) ref_result[10] += std_v1[i] * std_v4[i];
  viennacl::project(result, viennacl::slice(2, 4, 3)) = viennacl::linalg::inner_prod(vcl_v1, viennacl::tie(vcl_v3, vcl_v2, vcl_v4));
  if (check(ref_result, result, epsilon) != EXIT_SUCCESS)
  {
    std::copy(ref_result.begin(), ref_result.end(), std::ostream_iterator<NumericT>(std::cout, " ")); std::cout << std::endl;
    std::cout << result << std::endl;
    return EXIT_FAILURE;
  }

  std::cout << "Testing inner_prod with four vectors..." << std::endl;
  ref_result[4] = 0; for (std::size_t i=0; i<std_v1.size(); ++i) ref_result[4] += std_v1[i] * std_v1[i];
  ref_result[5] = 0; for (std::size_t i=0; i<std_v1.size(); ++i) ref_result[5] += std_v1[i] * std_v2[i];
  ref_result[6] = 0; for (std::size_t i=0; i<std_v1.size(); ++i) ref_result[6] += std_v1[i] * std_v3[i];
  ref_result[7] = 0; for (std::size_t i=0; i<std_v1.size(); ++i) ref_result[7] += std_v1[i] * std_v4[i];
  viennacl::project(result, viennacl::slice(4, 1, 4)) = viennacl::linalg::inner_prod(vcl_v1, viennacl::tie(vcl_v1, vcl_v2, vcl_v3, vcl_v4));
  if (check(ref_result, result, epsilon) != EXIT_SUCCESS)
  {
    std::copy(ref_result.begin(), ref_result.end(), std::ostream_iterator<NumericT>(std::cout, " ")); std::cout << std::endl;
    std::cout << result << std::endl;
    return EXIT_FAILURE;
  }

  ref_result[3]  = 0; for (std::size_t i=0; i<std_v1.size(); ++i) ref_result[3]  += std_v1[i] * std_v3[i];
  ref_result[6]  = 0; for (std::size_t i=0; i<std_v1.size(); ++i) ref_result[6]  += std_v1[i] * std_v2[i];
  ref_result[9]  = 0; for (std::size_t i=0; i<std_v1.size(); ++i) ref_result[9]  += std_v1[i] * std_v4[i];
  ref_result[12] = 0; for (std::size_t i=0; i<std_v1.size(); ++i) ref_result[12] += std_v1[i] * std_v1[i];
  viennacl::project(result, viennacl::slice(3, 3, 4)) = viennacl::linalg::inner_prod(vcl_v1, viennacl::tie(vcl_v3, vcl_v2, vcl_v4, vcl_v1));
  if (check(ref_result, result, epsilon) != EXIT_SUCCESS)
  {
    std::copy(ref_result.begin(), ref_result.end(), std::ostream_iterator<NumericT>(std::cout, " ")); std::cout << std::endl;
    std::cout << result << std::endl;
    return EXIT_FAILURE;
  }

  std::cout << "Testing inner_prod with five vectors..." << std::endl;
  ref_result[1] = 0; for (std::size_t i=0; i<std_v1.size(); ++i) ref_result[1] += std_v1[i] * std_v1[i];
  ref_result[3] = 0; for (std::size_t i=0; i<std_v1.size(); ++i) ref_result[3] += std_v1[i] * std_v2[i];
  ref_result[5] = 0; for (std::size_t i=0; i<std_v1.size(); ++i) ref_result[5] += std_v1[i] * std_v3[i];
  ref_result[7] = 0; for (std::size_t i=0; i<std_v1.size(); ++i) ref_result[7] += std_v1[i] * std_v4[i];
  ref_result[9] = 0; for (std::size_t i=0; i<std_v1.size(); ++i) ref_result[9] += std_v1[i] * std_v2[i];
  viennacl::project(result, viennacl::slice(1, 2, 5)) = viennacl::linalg::inner_prod(vcl_v1, viennacl::tie(vcl_v1, vcl_v2, vcl_v3, vcl_v4, vcl_v2));
  if (check(ref_result, result, epsilon) != EXIT_SUCCESS)
  {
    std::copy(ref_result.begin(), ref_result.end(), std::ostream_iterator<NumericT>(std::cout, " ")); std::cout << std::endl;
    std::cout << result << std::endl;
    return EXIT_FAILURE;
  }

  ref_result[2]  = 0; for (std::size_t i=0; i<std_v1.size(); ++i) ref_result[2]  += std_v1[i] * std_v3[i];
  ref_result[4]  = 0; for (std::size_t i=0; i<std_v1.size(); ++i) ref_result[4]  += std_v1[i] * std_v2[i];
  ref_result[6]  = 0; for (std::size_t i=0; i<std_v1.size(); ++i) ref_result[6]  += std_v1[i] * std_v4[i];
  ref_result[8]  = 0; for (std::size_t i=0; i<std_v1.size(); ++i) ref_result[8]  += std_v1[i] * std_v1[i];
  ref_result[10] = 0; for (std::size_t i=0; i<std_v1.size(); ++i) ref_result[10] += std_v1[i] * std_v2[i];
  viennacl::project(result, viennacl::slice(2, 2, 5)) = viennacl::linalg::inner_prod(vcl_v1, viennacl::tie(vcl_v3, vcl_v2, vcl_v4, vcl_v1, vcl_v2));
  if (check(ref_result, result, epsilon) != EXIT_SUCCESS)
  {
    std::copy(ref_result.begin(), ref_result.end(), std::ostream_iterator<NumericT>(std::cout, " ")); std::cout << std::endl;
    std::cout << result << std::endl;
    return EXIT_FAILURE;
  }


  std::cout << "Testing inner_prod with eight vectors..." << std::endl;
  ref_result[1]  = 0; for (std::size_t i=0; i<std_v1.size(); ++i) ref_result[1]  += std_v1[i] * std_v1[i];
  ref_result[5]  = 0; for (std::size_t i=0; i<std_v1.size(); ++i) ref_result[5]  += std_v1[i] * std_v2[i];
  ref_result[9]  = 0; for (std::size_t i=0; i<std_v1.size(); ++i) ref_result[9]  += std_v1[i] * std_v3[i];
  ref_result[13] = 0; for (std::size_t i=0; i<std_v1.size(); ++i) ref_result[13] += std_v1[i] * std_v4[i];
  ref_result[17] = 0; for (std::size_t i=0; i<std_v1.size(); ++i) ref_result[17] += std_v1[i] * std_v3[i];
  ref_result[21] = 0; for (std::size_t i=0; i<std_v1.size(); ++i) ref_result[21] += std_v1[i] * std_v2[i];
  ref_result[25] = 0; for (std::size_t i=0; i<std_v1.size(); ++i) ref_result[25] += std_v1[i] * std_v1[i];
  ref_result[29] = 0; for (std::size_t i=0; i<std_v1.size(); ++i) ref_result[29] += std_v1[i] * std_v2[i];
  std::vector<viennacl::vector_base<NumericT> const *> vecs1(8);
  vecs1[0] = &vcl_v1;
  vecs1[1] = &vcl_v2;
  vecs1[2] = &vcl_v3;
  vecs1[3] = &vcl_v4;
  vecs1[4] = &vcl_v3;
  vecs1[5] = &vcl_v2;
  vecs1[6] = &vcl_v1;
  vecs1[7] = &vcl_v2;
  viennacl::vector_tuple<NumericT> tuple1(vecs1);
  viennacl::project(result, viennacl::slice(1, 4, 8)) = viennacl::linalg::inner_prod(vcl_v1, tuple1);
  if (check(ref_result, result, epsilon) != EXIT_SUCCESS)
  {
    std::copy(ref_result.begin(), ref_result.end(), std::ostream_iterator<NumericT>(std::cout, " ")); std::cout << std::endl;
    std::cout << result << std::endl;
    return EXIT_FAILURE;
  }

  ref_result[3]  = 0; for (std::size_t i=0; i<std_v1.size(); ++i) ref_result[3]  += std_v1[i] * std_v2[i];
  ref_result[5]  = 0; for (std::size_t i=0; i<std_v1.size(); ++i) ref_result[5]  += std_v1[i] * std_v4[i];
  ref_result[7]  = 0; for (std::size_t i=0; i<std_v1.size(); ++i) ref_result[7]  += std_v1[i] * std_v1[i];
  ref_result[9]  = 0; for (std::size_t i=0; i<std_v1.size(); ++i) ref_result[9]  += std_v1[i] * std_v2[i];
  ref_result[11] = 0; for (std::size_t i=0; i<std_v1.size(); ++i) ref_result[11] += std_v1[i] * std_v2[i];
  ref_result[13] = 0; for (std::size_t i=0; i<std_v1.size(); ++i) ref_result[13] += std_v1[i] * std_v1[i];
  ref_result[15] = 0; for (std::size_t i=0; i<std_v1.size(); ++i) ref_result[15] += std_v1[i] * std_v4[i];
  ref_result[17] = 0; for (std::size_t i=0; i<std_v1.size(); ++i) ref_result[17] += std_v1[i] * std_v2[i];
  std::vector<viennacl::vector_base<NumericT> const *> vecs2(8);
  vecs2[0] = &vcl_v2;
  vecs2[1] = &vcl_v4;
  vecs2[2] = &vcl_v1;
  vecs2[3] = &vcl_v2;
  vecs2[4] = &vcl_v2;
  vecs2[5] = &vcl_v1;
  vecs2[6] = &vcl_v4;
  vecs2[7] = &vcl_v2;
  viennacl::vector_tuple<NumericT> tuple2(vecs2);
  viennacl::project(result, viennacl::slice(3, 2, 8)) = viennacl::linalg::inner_prod(vcl_v1, tuple2);
  if (check(ref_result, result, epsilon) != EXIT_SUCCESS)
  {
    std::copy(ref_result.begin(), ref_result.end(), std::ostream_iterator<NumericT>(std::cout, " ")); std::cout << std::endl;
    std::cout << result << std::endl;
    return EXIT_FAILURE;
  }


  // --------------------------------------------------------------------------
  return retval;
}


template< typename NumericT, typename Epsilon >
int test(Epsilon const& epsilon)
{
  viennacl::tools::uniform_random_numbers<NumericT> randomNumber;

  int retval = EXIT_SUCCESS;
  std::size_t size = 8 * 1337;

  std::cout << "Running tests for vector of size " << size << std::endl;

  //
  // Set up STL objects
  //
  std::vector<NumericT> std_full_vec1(size);
  std::vector<NumericT> std_full_vec2(std_full_vec1.size());

  for (std::size_t i=0; i<std_full_vec1.size(); ++i)
  {
    std_full_vec1[i] = NumericT(1.0) + randomNumber();
    std_full_vec2[i] = NumericT(1.0) + randomNumber();
  }

  std::vector<NumericT> std_slice_vec1(std_full_vec1.size() / 8); for (std::size_t i=0; i<std_slice_vec1.size(); ++i) std_slice_vec1[i] = std_full_vec1[    std_full_vec1.size() / 8 + i * 3];
  std::vector<NumericT> std_slice_vec2(std_full_vec2.size() / 8); for (std::size_t i=0; i<std_slice_vec2.size(); ++i) std_slice_vec2[i] = std_full_vec2[2 * std_full_vec2.size() / 8 + i * 1];
  std::vector<NumericT> std_slice_vec3(std_full_vec1.size() / 8); for (std::size_t i=0; i<std_slice_vec3.size(); ++i) std_slice_vec3[i] = std_full_vec1[4 * std_full_vec1.size() / 8 + i * 2];
  std::vector<NumericT> std_slice_vec4(std_full_vec2.size() / 8); for (std::size_t i=0; i<std_slice_vec4.size(); ++i) std_slice_vec4[i] = std_full_vec2[3 * std_full_vec2.size() / 8 + i * 4];

  //
  // Set up ViennaCL objects
  //
  viennacl::vector<NumericT> vcl_full_vec1(std_full_vec1.size());
  viennacl::vector<NumericT> vcl_full_vec2(std_full_vec2.size());

  viennacl::fast_copy(std_full_vec1.begin(), std_full_vec1.end(), vcl_full_vec1.begin());
  viennacl::copy     (std_full_vec2.begin(), std_full_vec2.end(), vcl_full_vec2.begin());

  viennacl::slice vcl_s1(    vcl_full_vec1.size() / 8, 3, vcl_full_vec1.size() / 8);
  viennacl::slice vcl_s2(2 * vcl_full_vec2.size() / 8, 1, vcl_full_vec2.size() / 8);
  viennacl::slice vcl_s3(4 * vcl_full_vec1.size() / 8, 2, vcl_full_vec1.size() / 8);
  viennacl::slice vcl_s4(3 * vcl_full_vec2.size() / 8, 4, vcl_full_vec2.size() / 8);
  viennacl::vector_slice< viennacl::vector<NumericT> > vcl_slice_vec1(vcl_full_vec1, vcl_s1);
  viennacl::vector_slice< viennacl::vector<NumericT> > vcl_slice_vec2(vcl_full_vec2, vcl_s2);
  viennacl::vector_slice< viennacl::vector<NumericT> > vcl_slice_vec3(vcl_full_vec1, vcl_s3);
  viennacl::vector_slice< viennacl::vector<NumericT> > vcl_slice_vec4(vcl_full_vec2, vcl_s4);

  viennacl::vector<NumericT> vcl_short_vec1(vcl_slice_vec1);
  viennacl::vector<NumericT> vcl_short_vec2 = vcl_slice_vec2;
  viennacl::vector<NumericT> vcl_short_vec3 = vcl_slice_vec2 + vcl_slice_vec1;
  viennacl::vector<NumericT> vcl_short_vec4 = vcl_short_vec1 + vcl_slice_vec2;

  std::vector<NumericT> std_short_vec1(std_slice_vec1);
  std::vector<NumericT> std_short_vec2(std_slice_vec2);
  std::vector<NumericT> std_short_vec3(std_slice_vec2.size()); for (std::size_t i=0; i<std_short_vec3.size(); ++i) std_short_vec3[i] = std_slice_vec2[i] + std_slice_vec1[i];
  std::vector<NumericT> std_short_vec4(std_slice_vec2.size()); for (std::size_t i=0; i<std_short_vec4.size(); ++i) std_short_vec4[i] = std_slice_vec1[i] + std_slice_vec2[i];

  std::cout << "Testing creation of vectors from slice..." << std::endl;
  if (check(std_short_vec1, vcl_short_vec1, epsilon) != EXIT_SUCCESS)
    return EXIT_FAILURE;
  if (check(std_short_vec2, vcl_short_vec2, epsilon) != EXIT_SUCCESS)
    return EXIT_FAILURE;
  if (check(std_short_vec3, vcl_short_vec3, epsilon) != EXIT_SUCCESS)
    return EXIT_FAILURE;
  if (check(std_short_vec4, vcl_short_vec4, epsilon) != EXIT_SUCCESS)
    return EXIT_FAILURE;


  //
  // Now start running tests for vectors, ranges and slices:
  //

  std::cout << " ** [vector|vector|vector|vector] **" << std::endl;
  retval = test<NumericT>(epsilon,
                          std_short_vec1, std_short_vec2, std_short_vec2, std_short_vec2,
                          vcl_short_vec1, vcl_short_vec2, vcl_short_vec3, vcl_short_vec4);
  if (retval != EXIT_SUCCESS)
    return EXIT_FAILURE;

  std::cout << " ** [vector|vector|vector|slice] **" << std::endl;
  retval = test<NumericT>(epsilon,
                          std_short_vec1, std_short_vec2, std_short_vec2, std_slice_vec2,
                          vcl_short_vec1, vcl_short_vec2, vcl_short_vec3, vcl_slice_vec4);
  if (retval != EXIT_SUCCESS)
    return EXIT_FAILURE;

  std::cout << " ** [vector|vector|slice|vector] **" << std::endl;
  retval = test<NumericT>(epsilon,
                          std_short_vec1, std_short_vec2, std_slice_vec2, std_short_vec2,
                          vcl_short_vec1, vcl_short_vec2, vcl_slice_vec3, vcl_short_vec4);
  if (retval != EXIT_SUCCESS)
    return EXIT_FAILURE;

  std::cout << " ** [vector|vector|slice|slice] **" << std::endl;
  retval = test<NumericT>(epsilon,
                          std_short_vec1, std_short_vec2, std_slice_vec2, std_slice_vec2,
                          vcl_short_vec1, vcl_short_vec2, vcl_slice_vec3, vcl_slice_vec4);
  if (retval != EXIT_SUCCESS)
    return EXIT_FAILURE;

  std::cout << " ** [vector|slice|vector|vector] **" << std::endl;
  retval = test<NumericT>(epsilon,
                          std_short_vec1, std_slice_vec2, std_short_vec2, std_short_vec2,
                          vcl_short_vec1, vcl_slice_vec2, vcl_short_vec3, vcl_short_vec4);
  if (retval != EXIT_SUCCESS)
    return EXIT_FAILURE;

  std::cout << " ** [vector|slice|vector|slice] **" << std::endl;
  retval = test<NumericT>(epsilon,
                          std_short_vec1, std_slice_vec2, std_short_vec2, std_slice_vec2,
                          vcl_short_vec1, vcl_slice_vec2, vcl_short_vec3, vcl_slice_vec4);
  if (retval != EXIT_SUCCESS)
    return EXIT_FAILURE;

  std::cout << " ** [vector|slice|slice|vector] **" << std::endl;
  retval = test<NumericT>(epsilon,
                          std_short_vec1, std_slice_vec2, std_slice_vec2, std_short_vec2,
                          vcl_short_vec1, vcl_slice_vec2, vcl_slice_vec3, vcl_short_vec4);
  if (retval != EXIT_SUCCESS)
    return EXIT_FAILURE;

  std::cout << " ** [vector|slice|slice|slice] **" << std::endl;
  retval = test<NumericT>(epsilon,
                          std_short_vec1, std_slice_vec2, std_slice_vec2, std_slice_vec2,
                          vcl_short_vec1, vcl_slice_vec2, vcl_slice_vec3, vcl_slice_vec4);
  if (retval != EXIT_SUCCESS)
    return EXIT_FAILURE;


  //////////////////


  std::cout << " ** [slice|vector|vector|vector] **" << std::endl;
  retval = test<NumericT>(epsilon,
                          std_slice_vec1, std_short_vec2, std_short_vec2, std_short_vec2,
                          vcl_slice_vec1, vcl_short_vec2, vcl_short_vec3, vcl_short_vec4);
  if (retval != EXIT_SUCCESS)
    return EXIT_FAILURE;

  std::cout << " ** [slice|vector|vector|slice] **" << std::endl;
  retval = test<NumericT>(epsilon,
                          std_slice_vec1, std_short_vec2, std_short_vec2, std_slice_vec2,
                          vcl_slice_vec1, vcl_short_vec2, vcl_short_vec3, vcl_slice_vec4);
  if (retval != EXIT_SUCCESS)
    return EXIT_FAILURE;

  std::cout << " ** [slice|vector|slice|vector] **" << std::endl;
  retval = test<NumericT>(epsilon,
                          std_slice_vec1, std_short_vec2, std_slice_vec2, std_short_vec2,
                          vcl_slice_vec1, vcl_short_vec2, vcl_slice_vec3, vcl_short_vec4);
  if (retval != EXIT_SUCCESS)
    return EXIT_FAILURE;

  std::cout << " ** [slice|vector|slice|slice] **" << std::endl;
  retval = test<NumericT>(epsilon,
                          std_slice_vec1, std_short_vec2, std_slice_vec2, std_slice_vec2,
                          vcl_slice_vec1, vcl_short_vec2, vcl_slice_vec3, vcl_slice_vec4);
  if (retval != EXIT_SUCCESS)
    return EXIT_FAILURE;

  std::cout << " ** [slice|slice|vector|vector] **" << std::endl;
  retval = test<NumericT>(epsilon,
                          std_slice_vec1, std_slice_vec2, std_short_vec2, std_short_vec2,
                          vcl_slice_vec1, vcl_slice_vec2, vcl_short_vec3, vcl_short_vec4);
  if (retval != EXIT_SUCCESS)
    return EXIT_FAILURE;

  std::cout << " ** [slice|slice|vector|slice] **" << std::endl;
  retval = test<NumericT>(epsilon,
                          std_slice_vec1, std_slice_vec2, std_short_vec2, std_slice_vec2,
                          vcl_slice_vec1, vcl_slice_vec2, vcl_short_vec3, vcl_slice_vec4);
  if (retval != EXIT_SUCCESS)
    return EXIT_FAILURE;

  std::cout << " ** [slice|slice|slice|vector] **" << std::endl;
  retval = test<NumericT>(epsilon,
                          std_slice_vec1, std_slice_vec2, std_slice_vec2, std_short_vec2,
                          vcl_slice_vec1, vcl_slice_vec2, vcl_slice_vec3, vcl_short_vec4);
  if (retval != EXIT_SUCCESS)
    return EXIT_FAILURE;

  std::cout << " ** [slice|slice|slice|slice] **" << std::endl;
  retval = test<NumericT>(epsilon,
                          std_slice_vec1, std_slice_vec2, std_slice_vec2, std_slice_vec2,
                          vcl_slice_vec1, vcl_slice_vec2, vcl_slice_vec3, vcl_slice_vec4);
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
   std::cout << "## Test :: Vector multiple inner products" << std::endl;
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
