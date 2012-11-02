#ifndef VIENNACL_LINALG_CUDA_VECTOR_OPERATIONS_HPP_
#define VIENNACL_LINALG_CUDA_VECTOR_OPERATIONS_HPP_

/* =========================================================================
   Copyright (c) 2010-2012, Institute for Microelectronics,
                            Institute for Analysis and Scientific Computing,
                            TU Wien.

                            -----------------
                  ViennaCL - The Vienna Computing Library
                            -----------------

   Project Head:    Karl Rupp                   rupp@iue.tuwien.ac.at
               
   (A list of authors and contributors can be found in the PDF manual)

   License:         MIT (X11), see file LICENSE in the base directory
============================================================================= */

/** @file viennacl/linalg/cuda/vector_operations.hpp
    @brief Implementations of vector operations using a plain single-threaded execution on CPU
*/

#include <cmath>
#include "viennacl/forwards.h"
#include "viennacl/scalar.hpp"
#include "viennacl/tools/tools.hpp"
#include "viennacl/meta/predicate.hpp"
#include "viennacl/meta/enable_if.hpp"
#include "viennacl/traits/size.hpp"
#include "viennacl/traits/start.hpp"
#include "viennacl/linalg/single_threaded/common.hpp"
#include "viennacl/traits/stride.hpp"

namespace viennacl
{
  namespace linalg
  {
    namespace cuda
    {
      
      //
      // Introductory note: By convention, all dimensions are already checked in the dispatcher frontend. No need to double-check again in here!
      //
      
      
      //////////////////////// as /////////////////////////////
      
      template <typename T>
      __global__ void av_kernel(T * vec1,
                                unsigned int start1,
                                unsigned int inc1,          
                                unsigned int size1,
                                
                                const T * fac2,
                                unsigned int options2,
                                const T * vec2,
                                unsigned int start2,
                                unsigned int inc2)
      { 
        T alpha = *fac2;
        if (options2 & (1 << 0))
          alpha = -alpha;
        if (options2 & (1 << 1))
          alpha = ((T)(1)) / alpha;

        for (unsigned int i = blockDim.x * blockIdx.x + threadIdx.x;
                          i < size1;
                          i += gridDim.x * blockDim.x)
          vec1[i*inc1+start1] = vec2[i*inc2+start2] * alpha;
      }

      template <typename T>
      __global__ void av_kernel(T * vec1,
                                unsigned int start1,
                                unsigned int inc1,          
                                unsigned int size1,
                                
                                T fac2,
                                unsigned int options2,
                                const T * vec2,
                                unsigned int start2,
                                unsigned int inc2)
      { 
        T alpha = fac2;
        if (options2 & (1 << 0))
          alpha = -alpha;
        if (options2 & (1 << 1))
          alpha = ((T)(1)) / alpha;

        for (unsigned int i = blockDim.x * blockIdx.x + threadIdx.x;
                          i < size1;
                          i += gridDim.x * blockDim.x)
          vec1[i*inc1+start1] = vec2[i*inc2+start2] * alpha;
      }

      
      
      template <typename V1,
                typename V2, typename ScalarType1>
      typename viennacl::enable_if< viennacl::is_any_dense_nonstructured_vector<V1>::value
                                    && viennacl::is_any_dense_nonstructured_vector<V2>::value
                                    && viennacl::is_any_scalar<ScalarType1>::value
                                  >::type
      av(V1 & vec1, 
         V2 const & vec2, ScalarType1 const & alpha, std::size_t len_alpha, bool reciprocal_alpha, bool flip_sign_alpha) 
      {
        typedef typename viennacl::result_of::cpu_value_type<V1>::type        value_type;
        
        unsigned int options_alpha =   ((len_alpha > 1) ? (len_alpha << 2) : 0)
                                      + (reciprocal_alpha ? 2 : 0)
                                      + (flip_sign_alpha ? 1 : 0);
        
        value_type data_alpha = alpha;
        if (flip_sign_alpha)
          data_alpha = -data_alpha;
        if (reciprocal_alpha)
          data_alpha = static_cast<value_type>(1) / data_alpha;

        value_type temporary_alpha;                             
        if (viennacl::is_cpu_scalar<ScalarType1>::value)
          temporary_alpha = alpha;
        
        av_kernel<<<128, 128>>>(detail::cuda_arg<value_type>(vec1),
                                static_cast<unsigned int>(viennacl::traits::start(vec1)),
                                static_cast<unsigned int>(viennacl::traits::stride(vec1)),
                                static_cast<unsigned int>(viennacl::traits::size(vec1)),
                                    
                                detail::cuda_arg<value_type>(detail::arg_reference(alpha, temporary_alpha)),
                                options_alpha,
                                detail::cuda_arg<value_type>(vec2),
                                static_cast<unsigned int>(viennacl::traits::start(vec2)),
                                static_cast<unsigned int>(viennacl::traits::stride(vec2)) );
      }
      
      
      ///////////////////// avbv //////////////////////////////////
      
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

        throw "todo";
      }
      
      
      ////////////////////////// avbv_v //////////////////////////////////////
      
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

        throw "todo";
      }
      
      
      //////////////////////////
      
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
        
        throw "todo";
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
        
        throw "todo";
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
        
        throw "todo";
      }

      
      /** @brief Computes the l^1-norm of a vector
      *
      * @param vec The vector
      * @param result The result scalar
      */
      template <typename V1, typename S2>
      typename viennacl::enable_if<    viennacl::is_any_dense_nonstructured_vector<V1>::value
                                    && viennacl::is_scalar<S2>::value
                                  >::type
      norm_1_impl(V1 const & vec1,
                  S2 & result)
      {
        typedef typename viennacl::result_of::cpu_value_type<V1>::type        value_type;
        
        throw "todo";
      }

      /** @brief Computes the l^2-norm of a vector - implementation
      *
      * @param vec The vector
      * @param result The result scalar
      * @param dummy  Dummy parameter used for SFINAE
      */
      template <typename V1, typename S2>
      typename viennacl::enable_if<    viennacl::is_any_dense_nonstructured_vector<V1>::value
                                    && viennacl::is_scalar<S2>::value
                                  >::type
      norm_2_impl(V1 const & vec1,
                  S2 & result)
      {
        typedef typename viennacl::result_of::cpu_value_type<V1>::type        value_type;
        
        throw "todo";
      }

      /** @brief Computes the supremum-norm of a vector
      *
      * @param vec The vector
      * @param result The result scalar
      */
      template <typename V1, typename S2>
      typename viennacl::enable_if<    viennacl::is_any_dense_nonstructured_vector<V1>::value
                                    && viennacl::is_scalar<S2>::value
                                  >::type
      norm_inf_impl(V1 const & vec1,
                    S2 & result)
      {
        typedef typename viennacl::result_of::cpu_value_type<V1>::type        value_type;
        
        throw "todo";
      }

      //This function should return a CPU scalar, otherwise statements like 
      // vcl_rhs[index_norm_inf(vcl_rhs)] 
      // are ambiguous
      /** @brief Computes the index of the first entry that is equal to the supremum-norm in modulus.
      *
      * @param vec The vector
      * @return The result. Note that the result must be a CPU scalar (unsigned int), since gpu scalars are floating point types.
      */
      template <typename V1>
      typename viennacl::enable_if< viennacl::is_any_dense_nonstructured_vector<V1>::value,
                                    std::size_t
                                  >::type
      index_norm_inf(V1 const & vec1)
      {
        typedef typename viennacl::result_of::cpu_value_type<V1>::type        value_type;
        
        throw "todo";
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
        
        throw "todo";
      }

    } //namespace opencl
  } //namespace linalg
} //namespace viennacl


#endif
