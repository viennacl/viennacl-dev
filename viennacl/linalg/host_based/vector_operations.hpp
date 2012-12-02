#ifndef VIENNACL_LINALG_HOST_BASED_VECTOR_OPERATIONS_HPP_
#define VIENNACL_LINALG_HOST_BASED_VECTOR_OPERATIONS_HPP_

/* =========================================================================
   Copyright (c) 2010-2012, Institute for Microelectronics,
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

/** @file viennacl/linalg/host_based/vector_operations.hpp
    @brief Implementations of vector operations using a plain single-threaded or OpenMP-enabled execution on CPU
*/

#include <cmath>
#include "viennacl/forwards.h"
#include "viennacl/scalar.hpp"
#include "viennacl/tools/tools.hpp"
#include "viennacl/meta/predicate.hpp"
#include "viennacl/meta/enable_if.hpp"
#include "viennacl/traits/size.hpp"
#include "viennacl/traits/start.hpp"
#include "viennacl/linalg/host_based/common.hpp"
#include "viennacl/traits/stride.hpp"

namespace viennacl
{
  namespace linalg
  {
    namespace host_based
    {
      
      //
      // Introductory note: By convention, all dimensions are already checked in the dispatcher frontend. No need to double-check again in here!
      //
      
      
      template <typename V1,
                typename V2, typename ScalarType1>
      typename viennacl::enable_if< viennacl::is_any_dense_nonstructured_vector<V1>::value
                                    && viennacl::is_any_dense_nonstructured_vector<V2>::value
                                    && viennacl::is_any_scalar<ScalarType1>::value
                                  >::type
      av(V1 & vec1, 
         V2 const & vec2, ScalarType1 const & alpha, std::size_t /*len_alpha*/, bool reciprocal_alpha, bool flip_sign_alpha) 
      {
        typedef typename viennacl::result_of::cpu_value_type<V1>::type        value_type;
        
        value_type       * data_vec1 = detail::extract_raw_pointer<value_type>(vec1);
        value_type const * data_vec2 = detail::extract_raw_pointer<value_type>(vec2);
        
        value_type data_alpha = alpha;
        if (flip_sign_alpha)
          data_alpha = -data_alpha;
        if (reciprocal_alpha)
          data_alpha = static_cast<value_type>(1) / data_alpha;
        
        std::size_t start1 = viennacl::traits::start(vec1);
        std::size_t inc1   = viennacl::traits::stride(vec1);
        std::size_t size1  = viennacl::traits::size(vec1);
        
        std::size_t start2 = viennacl::traits::start(vec2);
        std::size_t inc2   = viennacl::traits::stride(vec2);
        
#ifdef VIENNACL_WITH_OPENMP
        #pragma omp parallel for
#endif
        for (std::size_t i = 0; i < size1; ++i)
          data_vec1[i*inc1+start1] = data_vec2[i*inc2+start2] * data_alpha;
      }
      
      
      template <typename V1,
                typename V2, typename ScalarType1,
                typename V3, typename ScalarType2>
      typename viennacl::enable_if< viennacl::is_any_dense_nonstructured_vector<V1>::value
                                    && viennacl::is_any_dense_nonstructured_vector<V2>::value
                                    && viennacl::is_any_dense_nonstructured_vector<V3>::value
                                    && viennacl::is_any_scalar<ScalarType1>::value
                                    && viennacl::is_any_scalar<ScalarType2>::value
                                  >::type
      avbv(V1 & vec1, 
          V2 const & vec2, ScalarType1 const & alpha, std::size_t /* len_alpha */, bool reciprocal_alpha, bool flip_sign_alpha,
          V3 const & vec3, ScalarType2 const & beta,  std::size_t /* len_beta */,  bool reciprocal_beta,  bool flip_sign_beta) 
      {
        typedef typename viennacl::result_of::cpu_value_type<V1>::type        value_type;

        value_type       * data_vec1 = detail::extract_raw_pointer<value_type>(vec1);
        value_type const * data_vec2 = detail::extract_raw_pointer<value_type>(vec2);
        value_type const * data_vec3 = detail::extract_raw_pointer<value_type>(vec3);
        
        value_type data_alpha = alpha;
        if (flip_sign_alpha)
          data_alpha = -data_alpha;
        if (reciprocal_alpha)
          data_alpha = static_cast<value_type>(1) / data_alpha;

        value_type data_beta = beta;
        if (flip_sign_beta)
          data_beta = -data_beta;
        if (reciprocal_beta)
          data_beta = static_cast<value_type>(1) / data_beta;
        
        std::size_t start1 = viennacl::traits::start(vec1);
        std::size_t inc1   = viennacl::traits::stride(vec1);
        std::size_t size1  = viennacl::traits::size(vec1);
        
        std::size_t start2 = viennacl::traits::start(vec2);
        std::size_t inc2   = viennacl::traits::stride(vec2);
        
        std::size_t start3 = viennacl::traits::start(vec3);
        std::size_t inc3   = viennacl::traits::stride(vec3);
        
#ifdef VIENNACL_WITH_OPENMP
        #pragma omp parallel for
#endif
        for (std::size_t i = 0; i < size1; ++i)
          data_vec1[i*inc1+start1] = data_vec2[i*inc2+start2] * data_alpha + data_vec3[i*inc3+start3] * data_beta;
      }
      
      
      template <typename V1,
                typename V2, typename ScalarType1,
                typename V3, typename ScalarType2>
      typename viennacl::enable_if< viennacl::is_any_dense_nonstructured_vector<V1>::value
                                    && viennacl::is_any_dense_nonstructured_vector<V2>::value
                                    && viennacl::is_any_dense_nonstructured_vector<V3>::value
                                    && viennacl::is_any_scalar<ScalarType1>::value
                                    && viennacl::is_any_scalar<ScalarType2>::value
                                  >::type
      avbv_v(V1 & vec1,
             V2 const & vec2, ScalarType1 const & alpha, std::size_t /*len_alpha*/, bool reciprocal_alpha, bool flip_sign_alpha,
             V3 const & vec3, ScalarType2 const & beta,  std::size_t /*len_beta*/,  bool reciprocal_beta,  bool flip_sign_beta) 
      {
        typedef typename viennacl::result_of::cpu_value_type<V1>::type        value_type;

        value_type       * data_vec1 = detail::extract_raw_pointer<value_type>(vec1);
        value_type const * data_vec2 = detail::extract_raw_pointer<value_type>(vec2);
        value_type const * data_vec3 = detail::extract_raw_pointer<value_type>(vec3);
        
        value_type data_alpha = alpha;
        if (flip_sign_alpha)
          data_alpha = -data_alpha;
        if (reciprocal_alpha)
          data_alpha = static_cast<value_type>(1) / data_alpha;

        value_type data_beta = beta;
        if (flip_sign_beta)
          data_beta = -data_beta;
        if (reciprocal_beta)
          data_beta = static_cast<value_type>(1) / data_beta;
        
        std::size_t start1 = viennacl::traits::start(vec1);
        std::size_t inc1   = viennacl::traits::stride(vec1);
        std::size_t size1  = viennacl::traits::size(vec1);
        
        std::size_t start2 = viennacl::traits::start(vec2);
        std::size_t inc2   = viennacl::traits::stride(vec2);
        
        std::size_t start3 = viennacl::traits::start(vec3);
        std::size_t inc3   = viennacl::traits::stride(vec3);
        
#ifdef VIENNACL_WITH_OPENMP
        #pragma omp parallel for
#endif
        for (std::size_t i = 0; i < size1; ++i)
          data_vec1[i*inc1+start1] += data_vec2[i*inc2+start2] * data_alpha + data_vec3[i*inc3+start3] * data_beta;
      }
      
      
      
      
      /** @brief Assign a constant value to a vector (-range/-slice)
      *
      * @param vec1   The vector to which the value should be assigned
      * @param alpha  The value to be assigned
      */
      template <typename V1, typename S1>
      typename viennacl::enable_if< viennacl::is_any_dense_nonstructured_vector<V1>::value
                                    && viennacl::is_cpu_scalar<S1>::value
                                  >::type
      vector_assign(V1 & vec1, const S1 & alpha)
      {
        typedef typename viennacl::result_of::cpu_value_type<V1>::type        value_type;
        
        value_type * data_vec1 = detail::extract_raw_pointer<value_type>(vec1);
        
        std::size_t start1 = viennacl::traits::start(vec1);
        std::size_t inc1   = viennacl::traits::stride(vec1);
        std::size_t size1  = viennacl::traits::size(vec1);
        std::size_t internal_size1  = vec1.internal_size();  //Note: Do NOT use traits::internal_size() here, because vector proxies don't require padding.
        
        value_type data_alpha = static_cast<value_type>(alpha);
        
#ifdef VIENNACL_WITH_OPENMP
        #pragma omp parallel for
#endif
        for (std::size_t i = 0; i < internal_size1; ++i)
          data_vec1[i*inc1+start1] = (i < size1) ? data_alpha : 0;
      }

      
      /** @brief Swaps the contents of two vectors, data is copied
      *
      * @param vec1   The first vector (or -range, or -slice)
      * @param vec2   The second vector (or -range, or -slice)
      */
      template <typename V1, typename V2>
      typename viennacl::enable_if<    viennacl::is_any_dense_nonstructured_vector<V1>::value
                                    && viennacl::is_any_dense_nonstructured_vector<V2>::value
                                  >::type
      vector_swap(V1 & vec1, V2 & vec2)
      {
        typedef typename viennacl::result_of::cpu_value_type<V1>::type        value_type;
        
        value_type * data_vec1 = detail::extract_raw_pointer<value_type>(vec1);
        value_type * data_vec2 = detail::extract_raw_pointer<value_type>(vec2);
        
        std::size_t start1 = viennacl::traits::start(vec1);
        std::size_t inc1   = viennacl::traits::stride(vec1);
        std::size_t size1  = viennacl::traits::size(vec1);
        
        std::size_t start2 = viennacl::traits::start(vec2);
        std::size_t inc2   = viennacl::traits::stride(vec2);
        
#ifdef VIENNACL_WITH_OPENMP
        #pragma omp parallel for
#endif
        for (std::size_t i = 0; i < size1; ++i)
        {
          value_type temp = data_vec2[i*inc2+start2];
          data_vec2[i*inc2+start2] = data_vec1[i*inc1+start1];
          data_vec1[i*inc1+start1] = temp;
        }
      }
      
      
      ///////////////////////// Elementwise operations /////////////
      
      /** @brief Implementation of the element-wise operation v1 = v2 .* v3 and v1 = v2 ./ v3    (using MATLAB syntax)
      *
      * @param vec1   The result vector (or -range, or -slice)
      * @param proxy  The proxy object holding v2, v3 and the operation
      */
      template <typename V1, typename V2, typename V3, typename OP>
      typename viennacl::enable_if<    viennacl::is_any_dense_nonstructured_vector<V1>::value
                                    && viennacl::is_any_dense_nonstructured_vector<V2>::value
                                    && viennacl::is_any_dense_nonstructured_vector<V3>::value
                                  >::type
      element_op(V1 & vec1,
                vector_expression<const V2, const V3, OP> const & proxy)
      {
        typedef typename viennacl::result_of::cpu_value_type<V1>::type        value_type;
        
        value_type       * data_vec1 = detail::extract_raw_pointer<value_type>(vec1);
        value_type const * data_vec2 = detail::extract_raw_pointer<value_type>(proxy.lhs());
        value_type const * data_vec3 = detail::extract_raw_pointer<value_type>(proxy.rhs());
        
        std::size_t start1 = viennacl::traits::start(vec1);
        std::size_t inc1   = viennacl::traits::stride(vec1);
        std::size_t size1  = viennacl::traits::size(vec1);
        
        std::size_t start2 = viennacl::traits::start(proxy.lhs());
        std::size_t inc2   = viennacl::traits::stride(proxy.lhs());

        std::size_t start3 = viennacl::traits::start(proxy.rhs());
        std::size_t inc3   = viennacl::traits::stride(proxy.rhs());
        
        if (viennacl::is_product<OP>::value)
        {
#ifdef VIENNACL_WITH_OPENMP
          #pragma omp parallel for
#endif
          for (std::size_t i = 0; i < size1; ++i)
            data_vec1[i*inc1+start1] = data_vec2[i*inc2+start2] * data_vec3[i*inc3+start3];
        }
        else
        {
#ifdef VIENNACL_WITH_OPENMP
          #pragma omp parallel for
#endif
          for (std::size_t i = 0; i < size1; ++i)
            data_vec1[i*inc1+start1] = data_vec2[i*inc2+start2] / data_vec3[i*inc3+start3];
        }
      }



      ///////////////////////// Norms and inner product ///////////////////


      //implementation of inner product:
      //namespace {
      /** @brief Computes the inner product of two vectors - implementation. Library users should call inner_prod(vec1, vec2).
      *
      * @param vec1 The first vector
      * @param vec2 The second vector
      * @param result The result scalar (on the gpu)
      */
      template <typename V1, typename V2, typename S3>
      typename viennacl::enable_if<    viennacl::is_any_dense_nonstructured_vector<V1>::value
                                    && viennacl::is_any_dense_nonstructured_vector<V2>::value
                                    && viennacl::is_any_scalar<S3>::value
                                  >::type
      inner_prod_impl(V1 const & vec1,
                      V2 const & vec2,
                      S3 & result)
      {
        typedef typename viennacl::result_of::cpu_value_type<V1>::type        value_type;
        
        value_type const * data_vec1 = detail::extract_raw_pointer<value_type>(vec1);
        value_type const * data_vec2 = detail::extract_raw_pointer<value_type>(vec2);
        
        std::size_t start1 = viennacl::traits::start(vec1);
        std::size_t inc1   = viennacl::traits::stride(vec1);
        std::size_t size1  = viennacl::traits::size(vec1);
        
        std::size_t start2 = viennacl::traits::start(vec2);
        std::size_t inc2   = viennacl::traits::stride(vec2);
        
        value_type temp = 0;
        
        for (std::size_t i = 0; i < size1; ++i)
          temp += data_vec1[i*inc1+start1] * data_vec2[i*inc2+start2];
        
        result = temp;  //Note: Assignment to result might be expensive, thus 'temp' is used for accumulation
      }

      
      /** @brief Computes the l^1-norm of a vector
      *
      * @param vec1 The vector
      * @param result The result scalar
      */
      template <typename V1, typename S2>
      typename viennacl::enable_if<    viennacl::is_any_dense_nonstructured_vector<V1>::value
                                    && viennacl::is_any_scalar<S2>::value
                                  >::type
      norm_1_impl(V1 const & vec1,
                  S2 & result)
      {
        typedef typename viennacl::result_of::cpu_value_type<V1>::type        value_type;
        
        value_type const * data_vec1 = detail::extract_raw_pointer<value_type>(vec1);
        
        std::size_t start1 = viennacl::traits::start(vec1);
        std::size_t inc1   = viennacl::traits::stride(vec1);
        std::size_t size1  = viennacl::traits::size(vec1);
        
        value_type temp = 0;
        
        for (std::size_t i = 0; i < size1; ++i)
          temp += std::fabs(data_vec1[i*inc1+start1]);
        
        result = temp;  //Note: Assignment to result might be expensive, thus 'temp' is used for accumulation
      }

      /** @brief Computes the l^2-norm of a vector - implementation
      *
      * @param vec1 The vector
      * @param result The result scalar
      */
      template <typename V1, typename S2>
      typename viennacl::enable_if<    viennacl::is_any_dense_nonstructured_vector<V1>::value
                                    && viennacl::is_any_scalar<S2>::value
                                  >::type
      norm_2_impl(V1 const & vec1,
                  S2 & result)
      {
        typedef typename viennacl::result_of::cpu_value_type<V1>::type        value_type;
        
        value_type const * data_vec1 = detail::extract_raw_pointer<value_type>(vec1);
        
        std::size_t start1 = viennacl::traits::start(vec1);
        std::size_t inc1   = viennacl::traits::stride(vec1);
        std::size_t size1  = viennacl::traits::size(vec1);
        
        value_type temp = 0;
        value_type data = 0;
        
        for (std::size_t i = 0; i < size1; ++i)
        {
          data = data_vec1[i*inc1+start1];
          temp += data * data;
        }
        
        result = std::sqrt(temp);  //Note: Assignment to result might be expensive, thus 'temp' is used for accumulation
      }

      /** @brief Computes the supremum-norm of a vector
      *
      * @param vec1 The vector
      * @param result The result scalar
      */
      template <typename V1, typename S2>
      typename viennacl::enable_if<    viennacl::is_any_dense_nonstructured_vector<V1>::value
                                    && viennacl::is_any_scalar<S2>::value
                                  >::type
      norm_inf_impl(V1 const & vec1,
                    S2 & result)
      {
        typedef typename viennacl::result_of::cpu_value_type<V1>::type        value_type;
        
        value_type const * data_vec1 = detail::extract_raw_pointer<value_type>(vec1);
        
        std::size_t start1 = viennacl::traits::start(vec1);
        std::size_t inc1   = viennacl::traits::stride(vec1);
        std::size_t size1  = viennacl::traits::size(vec1);
        
        value_type temp = 0;
        
        for (std::size_t i = 0; i < size1; ++i)
          temp = std::max(temp, std::fabs(data_vec1[i*inc1+start1]));
        
        result = temp;  //Note: Assignment to result might be expensive, thus 'temp' is used for accumulation
      }

      //This function should return a CPU scalar, otherwise statements like 
      // vcl_rhs[index_norm_inf(vcl_rhs)] 
      // are ambiguous
      /** @brief Computes the index of the first entry that is equal to the supremum-norm in modulus.
      *
      * @param vec1 The vector
      * @return The result. Note that the result must be a CPU scalar (unsigned int), since gpu scalars are floating point types.
      */
      template <typename V1>
      typename viennacl::enable_if< viennacl::is_any_dense_nonstructured_vector<V1>::value,
                                    std::size_t
                                  >::type
      index_norm_inf(V1 const & vec1)
      {
        typedef typename viennacl::result_of::cpu_value_type<V1>::type        value_type;
        
        value_type const * data_vec1 = detail::extract_raw_pointer<value_type>(vec1);
        
        std::size_t start1 = viennacl::traits::start(vec1);
        std::size_t inc1   = viennacl::traits::stride(vec1);
        std::size_t size1  = viennacl::traits::size(vec1);
        
        value_type temp = 0;
        value_type data;
        std::size_t index = start1;
        
        for (std::size_t i = 0; i < size1; ++i)
        {
          data = std::fabs(data_vec1[i*inc1+start1]);
          if (data > temp)
          {
            index = i;
            temp = data;
          }
        }
        
        return index;
      }

      
      /** @brief Computes a plane rotation of two vectors.
      *
      * Computes (x,y) <- (alpha * x + beta * y, -beta * x + alpha * y)
      *
      * @param vec1   The first vector
      * @param vec2   The second vector
      * @param alpha  The first transformation coefficient
      * @param beta   The second transformation coefficient
      */
      template <typename V1, typename V2, typename SCALARTYPE>
      typename viennacl::enable_if< viennacl::is_any_dense_nonstructured_vector<V1>::value
                                    && viennacl::is_any_dense_nonstructured_vector<V2>::value
                                    && viennacl::is_any_scalar<SCALARTYPE>::value
                                  >::type
      plane_rotation(V1 & vec1,
                     V2 & vec2,
                     SCALARTYPE alpha,
                     SCALARTYPE beta)
      {
        typedef typename viennacl::result_of::cpu_value_type<V1>::type        value_type;
        
        value_type * data_vec1 = detail::extract_raw_pointer<value_type>(vec1);
        value_type * data_vec2 = detail::extract_raw_pointer<value_type>(vec2);
        
        std::size_t start1 = viennacl::traits::start(vec1);
        std::size_t inc1   = viennacl::traits::stride(vec1);
        std::size_t size1  = viennacl::traits::size(vec1);
        
        std::size_t start2 = viennacl::traits::start(vec2);
        std::size_t inc2   = viennacl::traits::stride(vec2);
        
        value_type temp1 = 0;
        value_type temp2 = 0;
        value_type data_alpha = alpha;
        value_type data_beta  = beta;
        
        for (std::size_t i = 0; i < size1; ++i)
        {
          temp1 = data_vec1[i*inc1+start1];
          temp2 = data_vec2[i*inc2+start2];
          
          data_vec1[i*inc1+start1] = data_alpha * temp1 + data_beta * temp2;
          data_vec2[i*inc2+start2] = data_alpha * temp2 - data_beta * temp1;
        }
      }

    } //namespace host_based
  } //namespace linalg
} //namespace viennacl


#endif
