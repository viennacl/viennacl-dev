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
#include "viennacl/vector_proxy.hpp"
#include "viennacl/linalg/inner_prod.hpp"
#include "viennacl/linalg/norm_1.hpp"
#include "viennacl/linalg/norm_2.hpp"
#include "viennacl/linalg/norm_inf.hpp"
#include "viennacl/tools/random.hpp"

using namespace boost::numeric;


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

template<typename ScalarType, typename ViennaCLVectorType>
ScalarType diff(ublas::vector_slice<ublas::vector<ScalarType> > const & v1, ViennaCLVectorType const & vcl_vec)
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
  return retval;
}


//
// -------------------------------------------------------------
//
template< typename NumericT, typename Epsilon,
          typename UblasVectorType1,    typename UblasVectorType2,    typename UblasVectorType3,    typename UblasVectorType4,
          typename ViennaCLVectorType1, typename ViennaCLVectorType2, typename ViennaCLVectorType3, typename ViennaCLVectorType4 >
int test(Epsilon const& epsilon,
         UblasVectorType1    & ublas_v1, UblasVectorType2    & ublas_v2, UblasVectorType3    & ublas_v3, UblasVectorType4    & ublas_v4,
         ViennaCLVectorType1 &   vcl_v1, ViennaCLVectorType2 &   vcl_v2, ViennaCLVectorType3 &   vcl_v3, ViennaCLVectorType4 &   vcl_v4)
{
  int retval = EXIT_SUCCESS;

  viennacl::tools::uniform_random_numbers<NumericT> randomNumber;

  for (std::size_t i=0; i<ublas_v1.size(); ++i)
  {
    ublas_v1[i] = NumericT(1.0) + randomNumber();
    ublas_v2[i] = NumericT(1.0) + randomNumber();
    ublas_v3[i] = NumericT(1.0) + randomNumber();
    ublas_v4[i] = NumericT(1.0) + randomNumber();
  }

  viennacl::copy(ublas_v1.begin(), ublas_v1.end(), vcl_v1.begin());  //resync
  viennacl::copy(ublas_v2.begin(), ublas_v2.end(), vcl_v2.begin());
  viennacl::copy(ublas_v3.begin(), ublas_v3.end(), vcl_v3.begin());
  viennacl::copy(ublas_v4.begin(), ublas_v4.end(), vcl_v4.begin());

  std::cout << "Checking for successful copy..." << std::endl;
  if (check(ublas_v1, vcl_v1, epsilon) != EXIT_SUCCESS)
    return EXIT_FAILURE;
  if (check(ublas_v2, vcl_v2, epsilon) != EXIT_SUCCESS)
    return EXIT_FAILURE;
  if (check(ublas_v3, vcl_v3, epsilon) != EXIT_SUCCESS)
    return EXIT_FAILURE;
  if (check(ublas_v4, vcl_v4, epsilon) != EXIT_SUCCESS)
    return EXIT_FAILURE;

  ublas::vector<NumericT> ref_result = ublas::scalar_vector<NumericT>(40, 0.0);
  viennacl::vector<NumericT> result = viennacl::scalar_vector<NumericT>(40, 0.0);

  std::cout << "Testing inner_prod with two vectors..." << std::endl;
  ref_result(2) = ublas::inner_prod(ublas_v1, ublas_v1);
  ref_result(5) = ublas::inner_prod(ublas_v1, ublas_v2);
  viennacl::project(result, viennacl::slice(2, 3, 2)) = viennacl::linalg::inner_prod(vcl_v1, viennacl::tie(vcl_v1, vcl_v2));
  if (check(ref_result, result, epsilon) != EXIT_SUCCESS)
  {
    std::cout << ref_result << std::endl;
    std::cout << result << std::endl;
    return EXIT_FAILURE;
  }

  ref_result(3) = ublas::inner_prod(ublas_v1, ublas_v3);
  ref_result(7) = ublas::inner_prod(ublas_v1, ublas_v4);
  viennacl::project(result, viennacl::slice(3, 4, 2)) = viennacl::linalg::inner_prod(vcl_v1, viennacl::tie(vcl_v3, vcl_v4));
  if (check(ref_result, result, epsilon) != EXIT_SUCCESS)
  {
    std::cout << ref_result << std::endl;
    std::cout << result << std::endl;
    return EXIT_FAILURE;
  }


  std::cout << "Testing inner_prod with three vectors..." << std::endl;
  ref_result(1) = ublas::inner_prod(ublas_v1, ublas_v1);
  ref_result(3) = ublas::inner_prod(ublas_v1, ublas_v2);
  ref_result(5) = ublas::inner_prod(ublas_v1, ublas_v3);
  viennacl::project(result, viennacl::slice(1, 2, 3)) = viennacl::linalg::inner_prod(vcl_v1, viennacl::tie(vcl_v1, vcl_v2, vcl_v3));
  if (check(ref_result, result, epsilon) != EXIT_SUCCESS)
  {
    std::cout << ref_result << std::endl;
    std::cout << result << std::endl;
    return EXIT_FAILURE;
  }

  ref_result(2)  = ublas::inner_prod(ublas_v1, ublas_v3);
  ref_result(6)  = ublas::inner_prod(ublas_v1, ublas_v2);
  ref_result(10) = ublas::inner_prod(ublas_v1, ublas_v4);
  viennacl::project(result, viennacl::slice(2, 4, 3)) = viennacl::linalg::inner_prod(vcl_v1, viennacl::tie(vcl_v3, vcl_v2, vcl_v4));
  if (check(ref_result, result, epsilon) != EXIT_SUCCESS)
  {
    std::cout << ref_result << std::endl;
    std::cout << result << std::endl;
    return EXIT_FAILURE;
  }

  std::cout << "Testing inner_prod with four vectors..." << std::endl;
  ref_result(4) = ublas::inner_prod(ublas_v1, ublas_v1);
  ref_result(5) = ublas::inner_prod(ublas_v1, ublas_v2);
  ref_result(6) = ublas::inner_prod(ublas_v1, ublas_v3);
  ref_result(7) = ublas::inner_prod(ublas_v1, ublas_v4);
  viennacl::project(result, viennacl::slice(4, 1, 4)) = viennacl::linalg::inner_prod(vcl_v1, viennacl::tie(vcl_v1, vcl_v2, vcl_v3, vcl_v4));
  if (check(ref_result, result, epsilon) != EXIT_SUCCESS)
  {
    std::cout << ref_result << std::endl;
    std::cout << result << std::endl;
    return EXIT_FAILURE;
  }

  ref_result(3)  = ublas::inner_prod(ublas_v1, ublas_v3);
  ref_result(6)  = ublas::inner_prod(ublas_v1, ublas_v2);
  ref_result(9)  = ublas::inner_prod(ublas_v1, ublas_v4);
  ref_result(12) = ublas::inner_prod(ublas_v1, ublas_v1);
  viennacl::project(result, viennacl::slice(3, 3, 4)) = viennacl::linalg::inner_prod(vcl_v1, viennacl::tie(vcl_v3, vcl_v2, vcl_v4, vcl_v1));
  if (check(ref_result, result, epsilon) != EXIT_SUCCESS)
  {
    std::cout << ref_result << std::endl;
    std::cout << result << std::endl;
    return EXIT_FAILURE;
  }

  std::cout << "Testing inner_prod with five vectors..." << std::endl;
  ref_result(1) = ublas::inner_prod(ublas_v1, ublas_v1);
  ref_result(3) = ublas::inner_prod(ublas_v1, ublas_v2);
  ref_result(5) = ublas::inner_prod(ublas_v1, ublas_v3);
  ref_result(7) = ublas::inner_prod(ublas_v1, ublas_v4);
  ref_result(9) = ublas::inner_prod(ublas_v1, ublas_v2);
  viennacl::project(result, viennacl::slice(1, 2, 5)) = viennacl::linalg::inner_prod(vcl_v1, viennacl::tie(vcl_v1, vcl_v2, vcl_v3, vcl_v4, vcl_v2));
  if (check(ref_result, result, epsilon) != EXIT_SUCCESS)
  {
    std::cout << ref_result << std::endl;
    std::cout << result << std::endl;
    return EXIT_FAILURE;
  }

  ref_result(2)  = ublas::inner_prod(ublas_v1, ublas_v3);
  ref_result(4)  = ublas::inner_prod(ublas_v1, ublas_v2);
  ref_result(6)  = ublas::inner_prod(ublas_v1, ublas_v4);
  ref_result(8)  = ublas::inner_prod(ublas_v1, ublas_v1);
  ref_result(10) = ublas::inner_prod(ublas_v1, ublas_v2);
  viennacl::project(result, viennacl::slice(2, 2, 5)) = viennacl::linalg::inner_prod(vcl_v1, viennacl::tie(vcl_v3, vcl_v2, vcl_v4, vcl_v1, vcl_v2));
  if (check(ref_result, result, epsilon) != EXIT_SUCCESS)
  {
    std::cout << ref_result << std::endl;
    std::cout << result << std::endl;
    return EXIT_FAILURE;
  }


  std::cout << "Testing inner_prod with eight vectors..." << std::endl;
  ref_result(1)  = ublas::inner_prod(ublas_v1, ublas_v1);
  ref_result(5)  = ublas::inner_prod(ublas_v1, ublas_v2);
  ref_result(9)  = ublas::inner_prod(ublas_v1, ublas_v3);
  ref_result(13) = ublas::inner_prod(ublas_v1, ublas_v4);
  ref_result(17) = ublas::inner_prod(ublas_v1, ublas_v3);
  ref_result(21) = ublas::inner_prod(ublas_v1, ublas_v2);
  ref_result(25) = ublas::inner_prod(ublas_v1, ublas_v1);
  ref_result(29) = ublas::inner_prod(ublas_v1, ublas_v2);
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
    std::cout << ref_result << std::endl;
    std::cout << result << std::endl;
    return EXIT_FAILURE;
  }

  ref_result(3)  = ublas::inner_prod(ublas_v1, ublas_v2);
  ref_result(5)  = ublas::inner_prod(ublas_v1, ublas_v4);
  ref_result(7)  = ublas::inner_prod(ublas_v1, ublas_v1);
  ref_result(9)  = ublas::inner_prod(ublas_v1, ublas_v2);
  ref_result(11) = ublas::inner_prod(ublas_v1, ublas_v2);
  ref_result(13) = ublas::inner_prod(ublas_v1, ublas_v1);
  ref_result(15) = ublas::inner_prod(ublas_v1, ublas_v4);
  ref_result(17) = ublas::inner_prod(ublas_v1, ublas_v2);
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
    std::cout << ref_result << std::endl;
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
  // Set up UBLAS objects
  //
  ublas::vector<NumericT> ublas_full_vec1(size);
  ublas::vector<NumericT> ublas_full_vec2(ublas_full_vec1.size());

  for (std::size_t i=0; i<ublas_full_vec1.size(); ++i)
  {
    ublas_full_vec1[i] = NumericT(1.0) + randomNumber();
    ublas_full_vec2[i] = NumericT(1.0) + randomNumber();
  }

  ublas::slice s1(    ublas_full_vec1.size() / 8, 3, ublas_full_vec1.size() / 8);
  ublas::slice s2(2 * ublas_full_vec2.size() / 8, 1, ublas_full_vec2.size() / 8);
  ublas::slice s3(4 * ublas_full_vec1.size() / 8, 2, ublas_full_vec1.size() / 8);
  ublas::slice s4(3 * ublas_full_vec2.size() / 8, 4, ublas_full_vec2.size() / 8);
  ublas::vector_slice< ublas::vector<NumericT> > ublas_slice_vec1(ublas_full_vec1, s1);
  ublas::vector_slice< ublas::vector<NumericT> > ublas_slice_vec2(ublas_full_vec2, s2);
  ublas::vector_slice< ublas::vector<NumericT> > ublas_slice_vec3(ublas_full_vec1, s3);
  ublas::vector_slice< ublas::vector<NumericT> > ublas_slice_vec4(ublas_full_vec2, s4);

  //
  // Set up ViennaCL objects
  //
  viennacl::vector<NumericT> vcl_full_vec1(ublas_full_vec1.size());
  viennacl::vector<NumericT> vcl_full_vec2(ublas_full_vec2.size());

  viennacl::fast_copy(ublas_full_vec1.begin(), ublas_full_vec1.end(), vcl_full_vec1.begin());
  viennacl::copy     (ublas_full_vec2.begin(), ublas_full_vec2.end(), vcl_full_vec2.begin());

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

  ublas::vector<NumericT> ublas_short_vec1(ublas_slice_vec1);
  ublas::vector<NumericT> ublas_short_vec2(ublas_slice_vec2);
  ublas::vector<NumericT> ublas_short_vec3 = ublas_slice_vec2 + ublas_slice_vec1;
  ublas::vector<NumericT> ublas_short_vec4 = ublas_short_vec1 + ublas_slice_vec2;

  std::cout << "Testing creation of vectors from slice..." << std::endl;
  if (check(ublas_short_vec1, vcl_short_vec1, epsilon) != EXIT_SUCCESS)
    return EXIT_FAILURE;
  if (check(ublas_short_vec2, vcl_short_vec2, epsilon) != EXIT_SUCCESS)
    return EXIT_FAILURE;
  if (check(ublas_short_vec3, vcl_short_vec3, epsilon) != EXIT_SUCCESS)
    return EXIT_FAILURE;
  if (check(ublas_short_vec4, vcl_short_vec4, epsilon) != EXIT_SUCCESS)
    return EXIT_FAILURE;


  //
  // Now start running tests for vectors, ranges and slices:
  //

  std::cout << " ** [vector|vector|vector|vector] **" << std::endl;
  retval = test<NumericT>(epsilon,
                          ublas_short_vec1, ublas_short_vec2, ublas_short_vec2, ublas_short_vec2,
                            vcl_short_vec1,   vcl_short_vec2,   vcl_short_vec3,   vcl_short_vec4);
  if (retval != EXIT_SUCCESS)
    return EXIT_FAILURE;

  std::cout << " ** [vector|vector|vector|slice] **" << std::endl;
  retval = test<NumericT>(epsilon,
                          ublas_short_vec1, ublas_short_vec2, ublas_short_vec2, ublas_slice_vec2,
                            vcl_short_vec1,   vcl_short_vec2,   vcl_short_vec3,   vcl_slice_vec4);
  if (retval != EXIT_SUCCESS)
    return EXIT_FAILURE;

  std::cout << " ** [vector|vector|slice|vector] **" << std::endl;
  retval = test<NumericT>(epsilon,
                          ublas_short_vec1, ublas_short_vec2, ublas_slice_vec2, ublas_short_vec2,
                            vcl_short_vec1,   vcl_short_vec2,   vcl_slice_vec3,   vcl_short_vec4);
  if (retval != EXIT_SUCCESS)
    return EXIT_FAILURE;

  std::cout << " ** [vector|vector|slice|slice] **" << std::endl;
  retval = test<NumericT>(epsilon,
                          ublas_short_vec1, ublas_short_vec2, ublas_slice_vec2, ublas_slice_vec2,
                            vcl_short_vec1,   vcl_short_vec2,   vcl_slice_vec3,   vcl_slice_vec4);
  if (retval != EXIT_SUCCESS)
    return EXIT_FAILURE;

  std::cout << " ** [vector|slice|vector|vector] **" << std::endl;
  retval = test<NumericT>(epsilon,
                          ublas_short_vec1, ublas_slice_vec2, ublas_short_vec2, ublas_short_vec2,
                            vcl_short_vec1,   vcl_slice_vec2,   vcl_short_vec3,   vcl_short_vec4);
  if (retval != EXIT_SUCCESS)
    return EXIT_FAILURE;

  std::cout << " ** [vector|slice|vector|slice] **" << std::endl;
  retval = test<NumericT>(epsilon,
                          ublas_short_vec1, ublas_slice_vec2, ublas_short_vec2, ublas_slice_vec2,
                            vcl_short_vec1,   vcl_slice_vec2,   vcl_short_vec3,   vcl_slice_vec4);
  if (retval != EXIT_SUCCESS)
    return EXIT_FAILURE;

  std::cout << " ** [vector|slice|slice|vector] **" << std::endl;
  retval = test<NumericT>(epsilon,
                          ublas_short_vec1, ublas_slice_vec2, ublas_slice_vec2, ublas_short_vec2,
                            vcl_short_vec1,   vcl_slice_vec2,   vcl_slice_vec3,   vcl_short_vec4);
  if (retval != EXIT_SUCCESS)
    return EXIT_FAILURE;

  std::cout << " ** [vector|slice|slice|slice] **" << std::endl;
  retval = test<NumericT>(epsilon,
                          ublas_short_vec1, ublas_slice_vec2, ublas_slice_vec2, ublas_slice_vec2,
                            vcl_short_vec1,   vcl_slice_vec2,   vcl_slice_vec3,   vcl_slice_vec4);
  if (retval != EXIT_SUCCESS)
    return EXIT_FAILURE;


  //////////////////


  std::cout << " ** [slice|vector|vector|vector] **" << std::endl;
  retval = test<NumericT>(epsilon,
                          ublas_slice_vec1, ublas_short_vec2, ublas_short_vec2, ublas_short_vec2,
                            vcl_slice_vec1,   vcl_short_vec2,   vcl_short_vec3,   vcl_short_vec4);
  if (retval != EXIT_SUCCESS)
    return EXIT_FAILURE;

  std::cout << " ** [slice|vector|vector|slice] **" << std::endl;
  retval = test<NumericT>(epsilon,
                          ublas_slice_vec1, ublas_short_vec2, ublas_short_vec2, ublas_slice_vec2,
                            vcl_slice_vec1,   vcl_short_vec2,   vcl_short_vec3,   vcl_slice_vec4);
  if (retval != EXIT_SUCCESS)
    return EXIT_FAILURE;

  std::cout << " ** [slice|vector|slice|vector] **" << std::endl;
  retval = test<NumericT>(epsilon,
                          ublas_slice_vec1, ublas_short_vec2, ublas_slice_vec2, ublas_short_vec2,
                            vcl_slice_vec1,   vcl_short_vec2,   vcl_slice_vec3,   vcl_short_vec4);
  if (retval != EXIT_SUCCESS)
    return EXIT_FAILURE;

  std::cout << " ** [slice|vector|slice|slice] **" << std::endl;
  retval = test<NumericT>(epsilon,
                          ublas_slice_vec1, ublas_short_vec2, ublas_slice_vec2, ublas_slice_vec2,
                            vcl_slice_vec1,   vcl_short_vec2,   vcl_slice_vec3,   vcl_slice_vec4);
  if (retval != EXIT_SUCCESS)
    return EXIT_FAILURE;

  std::cout << " ** [slice|slice|vector|vector] **" << std::endl;
  retval = test<NumericT>(epsilon,
                          ublas_slice_vec1, ublas_slice_vec2, ublas_short_vec2, ublas_short_vec2,
                            vcl_slice_vec1,   vcl_slice_vec2,   vcl_short_vec3,   vcl_short_vec4);
  if (retval != EXIT_SUCCESS)
    return EXIT_FAILURE;

  std::cout << " ** [slice|slice|vector|slice] **" << std::endl;
  retval = test<NumericT>(epsilon,
                          ublas_slice_vec1, ublas_slice_vec2, ublas_short_vec2, ublas_slice_vec2,
                            vcl_slice_vec1,   vcl_slice_vec2,   vcl_short_vec3,   vcl_slice_vec4);
  if (retval != EXIT_SUCCESS)
    return EXIT_FAILURE;

  std::cout << " ** [slice|slice|slice|vector] **" << std::endl;
  retval = test<NumericT>(epsilon,
                          ublas_slice_vec1, ublas_slice_vec2, ublas_slice_vec2, ublas_short_vec2,
                            vcl_slice_vec1,   vcl_slice_vec2,   vcl_slice_vec3,   vcl_short_vec4);
  if (retval != EXIT_SUCCESS)
    return EXIT_FAILURE;

  std::cout << " ** [slice|slice|slice|slice] **" << std::endl;
  retval = test<NumericT>(epsilon,
                          ublas_slice_vec1, ublas_slice_vec2, ublas_slice_vec2, ublas_slice_vec2,
                            vcl_slice_vec1,   vcl_slice_vec2,   vcl_slice_vec3,   vcl_slice_vec4);
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
