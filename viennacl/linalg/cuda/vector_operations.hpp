#ifndef VIENNACL_LINALG_CUDA_VECTOR_OPERATIONS_HPP_
#define VIENNACL_LINALG_CUDA_VECTOR_OPERATIONS_HPP_

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
      
      // gpu scalar
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

      // cpu scalar
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
        VIENNACL_CUDA_LAST_ERROR_CHECK("av_kernel");
      }
      
      
      ///////////////////// avbv //////////////////////////////////
      
      // alpha and beta on GPU
      template <typename T>
      __global__ void avbv_kernel(T * vec1,
                                  unsigned int start1,
                                  unsigned int inc1,          
                                  unsigned int size1,
                                  
                                  const T * fac2,
                                  unsigned int options2,
                                  const T * vec2,
                                  unsigned int start2,
                                  unsigned int inc2,
                                  
                                  const T * fac3,
                                  unsigned int options3,
                                  const T * vec3,
                                  unsigned int start3,
                                  unsigned int inc3)
      { 
        T alpha = *fac2;
        if (options2 & (1 << 0))
          alpha = -alpha;
        if (options2 & (1 << 1))
          alpha = ((T)(1)) / alpha;

        T beta = *fac3;
        if (options3 & (1 << 0))
          beta = -beta;
        if (options3 & (1 << 1))
          beta = ((T)(1)) / beta;
        
        for (unsigned int i = blockDim.x * blockIdx.x + threadIdx.x;
                          i < size1;
                          i += gridDim.x * blockDim.x)
          vec1[i*inc1+start1] = vec2[i*inc2+start2] * alpha + vec3[i*inc3+start3] * beta;
      }
      
      // alpha on CPU, beta on GPU
      template <typename T>
      __global__ void avbv_kernel(T * vec1,
                                  unsigned int start1,
                                  unsigned int inc1,          
                                  unsigned int size1,
                                  
                                  T fac2,
                                  unsigned int options2,
                                  const T * vec2,
                                  unsigned int start2,
                                  unsigned int inc2,
                                  
                                  const T * fac3,
                                  unsigned int options3,
                                  const T * vec3,
                                  unsigned int start3,
                                  unsigned int inc3)
      { 
        T alpha = fac2;
        if (options2 & (1 << 0))
          alpha = -alpha;
        if (options2 & (1 << 1))
          alpha = ((T)(1)) / alpha;

        T beta = *fac3;
        if (options3 & (1 << 0))
          beta = -beta;
        if (options3 & (1 << 1))
          beta = ((T)(1)) / beta;
        
        for (unsigned int i = blockDim.x * blockIdx.x + threadIdx.x;
                          i < size1;
                          i += gridDim.x * blockDim.x)
          vec1[i*inc1+start1] = vec2[i*inc2+start2] * alpha + vec3[i*inc3+start3] * beta;
      }
      
      // alpha on GPU, beta on CPU
      template <typename T>
      __global__ void avbv_kernel(T * vec1,
                                  unsigned int start1,
                                  unsigned int inc1,          
                                  unsigned int size1,
                                  
                                  const T * fac2,
                                  unsigned int options2,
                                  const T * vec2,
                                  unsigned int start2,
                                  unsigned int inc2,
                                  
                                  T fac3,
                                  unsigned int options3,
                                  const T * vec3,
                                  unsigned int start3,
                                  unsigned int inc3)
      { 
        T alpha = *fac2;
        if (options2 & (1 << 0))
          alpha = -alpha;
        if (options2 & (1 << 1))
          alpha = ((T)(1)) / alpha;

        T beta = fac3;
        if (options3 & (1 << 0))
          beta = -beta;
        if (options3 & (1 << 1))
          beta = ((T)(1)) / beta;
        
        for (unsigned int i = blockDim.x * blockIdx.x + threadIdx.x;
                          i < size1;
                          i += gridDim.x * blockDim.x)
          vec1[i*inc1+start1] = vec2[i*inc2+start2] * alpha + vec3[i*inc3+start3] * beta;
      }
      
      // alpha and beta on CPU
      template <typename T>
      __global__ void avbv_kernel(T * vec1,
                                  unsigned int start1,
                                  unsigned int inc1,          
                                  unsigned int size1,
                                  
                                  T fac2,
                                  unsigned int options2,
                                  const T * vec2,
                                  unsigned int start2,
                                  unsigned int inc2,
                                  
                                  T fac3,
                                  unsigned int options3,
                                  const T * vec3,
                                  unsigned int start3,
                                  unsigned int inc3)
      { 
        T alpha = fac2;
        if (options2 & (1 << 0))
          alpha = -alpha;
        if (options2 & (1 << 1))
          alpha = ((T)(1)) / alpha;

        T beta = fac3;
        if (options3 & (1 << 0))
          beta = -beta;
        if (options3 & (1 << 1))
          beta = ((T)(1)) / beta;
        
        for (unsigned int i = blockDim.x * blockIdx.x + threadIdx.x;
                          i < size1;
                          i += gridDim.x * blockDim.x)
          vec1[i*inc1+start1] = vec2[i*inc2+start2] * alpha + vec3[i*inc3+start3] * beta;
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
           V2 const & vec2, ScalarType1 const & alpha, std::size_t len_alpha, bool reciprocal_alpha, bool flip_sign_alpha,
           V3 const & vec3, ScalarType2 const & beta,  std::size_t len_beta,  bool reciprocal_beta,  bool flip_sign_beta) 
      {
        typedef typename viennacl::result_of::cpu_value_type<V1>::type        value_type;

        unsigned int options_alpha =   ((len_alpha > 1) ? (len_alpha << 2) : 0)
                                    + (reciprocal_alpha ?                2 : 0)
                                    + (flip_sign_alpha  ?                1 : 0);
        
        value_type data_alpha = alpha;
        if (flip_sign_alpha)
          data_alpha = -data_alpha;
        if (reciprocal_alpha)
          data_alpha = static_cast<value_type>(1) / data_alpha;

        value_type temporary_alpha;                             
        if (viennacl::is_cpu_scalar<ScalarType1>::value)
          temporary_alpha = alpha;
        
        unsigned int options_beta =    ((len_beta > 1) ? (len_beta << 2) : 0)
                                    + (reciprocal_beta ?               2 : 0)
                                    +  (flip_sign_beta ?               1 : 0);
        
        value_type temporary_beta;                             
        if (viennacl::is_cpu_scalar<ScalarType2>::value)
          temporary_beta = beta;
                                    
        
        avbv_kernel<<<128, 128>>>(detail::cuda_arg<value_type>(vec1),
                                  static_cast<unsigned int>(viennacl::traits::start(vec1)),
                                  static_cast<unsigned int>(viennacl::traits::stride(vec1)),
                                  static_cast<unsigned int>(viennacl::traits::size(vec1)),
                                      
                                  detail::cuda_arg<value_type>(detail::arg_reference(alpha, temporary_alpha)),
                                  options_alpha,
                                  detail::cuda_arg<value_type>(vec2),
                                  static_cast<unsigned int>(viennacl::traits::start(vec2)),
                                  static_cast<unsigned int>(viennacl::traits::stride(vec2)),
                                  
                                  detail::cuda_arg<value_type>(detail::arg_reference(beta, temporary_beta)),
                                  options_beta,
                                  detail::cuda_arg<value_type>(vec3),
                                  static_cast<unsigned int>(viennacl::traits::start(vec3)),
                                  static_cast<unsigned int>(viennacl::traits::stride(vec3)) );
        VIENNACL_CUDA_LAST_ERROR_CHECK("avbv_kernel");
      }
      
      
      ////////////////////////// avbv_v //////////////////////////////////////
      
      
      // alpha and beta on GPU
      template <typename T>
      __global__ void avbv_v_kernel(T * vec1,
                                    unsigned int start1,
                                    unsigned int inc1,          
                                    unsigned int size1,
                                    
                                    const T * fac2,
                                    unsigned int options2,
                                    const T * vec2,
                                    unsigned int start2,
                                    unsigned int inc2,
                                    
                                    const T * fac3,
                                    unsigned int options3,
                                    const T * vec3,
                                    unsigned int start3,
                                    unsigned int inc3)
      { 
        T alpha = *fac2;
        if (options2 & (1 << 0))
          alpha = -alpha;
        if (options2 & (1 << 1))
          alpha = ((T)(1)) / alpha;

        T beta = *fac3;
        if (options3 & (1 << 0))
          beta = -beta;
        if (options3 & (1 << 1))
          beta = ((T)(1)) / beta;
        
        for (unsigned int i = blockDim.x * blockIdx.x + threadIdx.x;
                          i < size1;
                          i += gridDim.x * blockDim.x)
          vec1[i*inc1+start1] += vec2[i*inc2+start2] * alpha + vec3[i*inc3+start3] * beta;
      }
      
      // alpha on CPU, beta on GPU
      template <typename T>
      __global__ void avbv_v_kernel(T * vec1,
                                    unsigned int start1,
                                    unsigned int inc1,          
                                    unsigned int size1,
                                    
                                    T fac2,
                                    unsigned int options2,
                                    const T * vec2,
                                    unsigned int start2,
                                    unsigned int inc2,
                                    
                                    const T * fac3,
                                    unsigned int options3,
                                    const T * vec3,
                                    unsigned int start3,
                                    unsigned int inc3)
      { 
        T alpha = fac2;
        if (options2 & (1 << 0))
          alpha = -alpha;
        if (options2 & (1 << 1))
          alpha = ((T)(1)) / alpha;

        T beta = *fac3;
        if (options3 & (1 << 0))
          beta = -beta;
        if (options3 & (1 << 1))
          beta = ((T)(1)) / beta;
        
        for (unsigned int i = blockDim.x * blockIdx.x + threadIdx.x;
                          i < size1;
                          i += gridDim.x * blockDim.x)
          vec1[i*inc1+start1] += vec2[i*inc2+start2] * alpha + vec3[i*inc3+start3] * beta;
      }
      
      // alpha on GPU, beta on CPU
      template <typename T>
      __global__ void avbv_v_kernel(T * vec1,
                                    unsigned int start1,
                                    unsigned int inc1,          
                                    unsigned int size1,
                                    
                                    const T * fac2,
                                    unsigned int options2,
                                    const T * vec2,
                                    unsigned int start2,
                                    unsigned int inc2,
                                    
                                    T fac3,
                                    unsigned int options3,
                                    const T * vec3,
                                    unsigned int start3,
                                    unsigned int inc3)
      { 
        T alpha = *fac2;
        if (options2 & (1 << 0))
          alpha = -alpha;
        if (options2 & (1 << 1))
          alpha = ((T)(1)) / alpha;

        T beta = fac3;
        if (options3 & (1 << 0))
          beta = -beta;
        if (options3 & (1 << 1))
          beta = ((T)(1)) / beta;
        
        for (unsigned int i = blockDim.x * blockIdx.x + threadIdx.x;
                          i < size1;
                          i += gridDim.x * blockDim.x)
          vec1[i*inc1+start1] = vec2[i*inc2+start2] * alpha + vec3[i*inc3+start3] * beta;
      }
      
      // alpha and beta on CPU
      template <typename T>
      __global__ void avbv_v_kernel(T * vec1,
                                    unsigned int start1,
                                    unsigned int inc1,          
                                    unsigned int size1,
                                    
                                    T fac2,
                                    unsigned int options2,
                                    const T * vec2,
                                    unsigned int start2,
                                    unsigned int inc2,
                                    
                                    T fac3,
                                    unsigned int options3,
                                    const T * vec3,
                                    unsigned int start3,
                                    unsigned int inc3)
      { 
        T alpha = fac2;
        if (options2 & (1 << 0))
          alpha = -alpha;
        if (options2 & (1 << 1))
          alpha = ((T)(1)) / alpha;

        T beta = fac3;
        if (options3 & (1 << 0))
          beta = -beta;
        if (options3 & (1 << 1))
          beta = ((T)(1)) / beta;
        
        for (unsigned int i = blockDim.x * blockIdx.x + threadIdx.x;
                          i < size1;
                          i += gridDim.x * blockDim.x)
          vec1[i*inc1+start1] += vec2[i*inc2+start2] * alpha + vec3[i*inc3+start3] * beta;
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
             V2 const & vec2, ScalarType1 const & alpha, std::size_t len_alpha, bool reciprocal_alpha, bool flip_sign_alpha,
             V3 const & vec3, ScalarType2 const & beta,  std::size_t len_beta,  bool reciprocal_beta,  bool flip_sign_beta) 
      {
        typedef typename viennacl::result_of::cpu_value_type<V1>::type        value_type;

        unsigned int options_alpha =   ((len_alpha > 1) ? (len_alpha << 2) : 0)
                                    + (reciprocal_alpha ?                2 : 0)
                                    + (flip_sign_alpha  ?                1 : 0);
        
        value_type data_alpha = alpha;
        if (flip_sign_alpha)
          data_alpha = -data_alpha;
        if (reciprocal_alpha)
          data_alpha = static_cast<value_type>(1) / data_alpha;

        value_type temporary_alpha;                             
        if (viennacl::is_cpu_scalar<ScalarType1>::value)
          temporary_alpha = alpha;
        
        unsigned int options_beta =    ((len_beta > 1) ? (len_beta << 2) : 0)
                                    + (reciprocal_beta ?               2 : 0)
                                    +  (flip_sign_beta ?               1 : 0);
        
        value_type temporary_beta;                             
        if (viennacl::is_cpu_scalar<ScalarType2>::value)
          temporary_beta = beta;
                                    
        
        avbv_v_kernel<<<128, 128>>>(detail::cuda_arg<value_type>(vec1),
                                    static_cast<unsigned int>(viennacl::traits::start(vec1)),
                                    static_cast<unsigned int>(viennacl::traits::stride(vec1)),
                                    static_cast<unsigned int>(viennacl::traits::size(vec1)),
                                        
                                    detail::cuda_arg<value_type>(detail::arg_reference(alpha, temporary_alpha)),
                                    options_alpha,
                                    detail::cuda_arg<value_type>(vec2),
                                    static_cast<unsigned int>(viennacl::traits::start(vec2)),
                                    static_cast<unsigned int>(viennacl::traits::stride(vec2)),
                                    
                                    detail::cuda_arg<value_type>(detail::arg_reference(beta, temporary_beta)),
                                    options_beta,
                                    detail::cuda_arg<value_type>(vec3),
                                    static_cast<unsigned int>(viennacl::traits::start(vec3)),
                                    static_cast<unsigned int>(viennacl::traits::stride(vec3)) );
      }
      
      
      //////////////////////////
      
      template <typename T>
      __global__ void vector_assign_kernel(T * vec1,
                                           unsigned int start1,
                                           unsigned int inc1,          
                                           unsigned int size1,
                                           unsigned int internal_size1,
                                            
                                           T alpha)
      { 
        for (unsigned int i = blockDim.x * blockIdx.x + threadIdx.x;
                          i < internal_size1;
                          i += gridDim.x * blockDim.x)
          vec1[i*inc1+start1] =  (i < size1) ? alpha : 0;
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
        
        value_type temporary_alpha;                             
        if (viennacl::is_cpu_scalar<S1>::value)
          temporary_alpha = alpha;
        
        vector_assign_kernel<<<128, 128>>>(detail::cuda_arg<value_type>(vec1),
                                           static_cast<unsigned int>(viennacl::traits::start(vec1)),
                                           static_cast<unsigned int>(viennacl::traits::stride(vec1)),
                                           static_cast<unsigned int>(viennacl::traits::size(vec1)),
                                           static_cast<unsigned int>(vec1.internal_size()),  //Note: Do NOT use traits::internal_size() here, because vector proxies don't require padding.
                                              
                                           detail::cuda_arg<value_type>(detail::arg_reference(alpha, temporary_alpha)) );
        VIENNACL_CUDA_LAST_ERROR_CHECK("avbv_v_kernel");
      }

      //////////////////////////
      
      template <typename T>
      __global__ void vector_swap_kernel(T * vec1,
                                         unsigned int start1,
                                         unsigned int inc1,
                                         unsigned int size1,
                                          
                                         T * vec2,
                                         unsigned int start2,
                                         unsigned int inc2)
      { 
        T tmp;
        for (unsigned int i = blockDim.x * blockIdx.x + threadIdx.x;
                          i < size1;
                          i += gridDim.x * blockDim.x)
        {
          tmp = vec2[i*inc2+start2];
          vec2[i*inc2+start2] = vec1[i*inc1+start1];
          vec1[i*inc1+start1] = tmp;
        }
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
        
        vector_swap_kernel<<<128, 128>>>(detail::cuda_arg<value_type>(vec1),
                                         static_cast<unsigned int>(viennacl::traits::start(vec1)),
                                         static_cast<unsigned int>(viennacl::traits::stride(vec1)),
                                         static_cast<unsigned int>(viennacl::traits::size(vec1)),
                                          
                                         detail::cuda_arg<value_type>(vec2),
                                         static_cast<unsigned int>(viennacl::traits::start(vec2)),
                                         static_cast<unsigned int>(viennacl::traits::stride(vec2)) );                                          
        VIENNACL_CUDA_LAST_ERROR_CHECK("vector_swap_kernel");
      }

      ///////////////////////// Elementwise operations /////////////
      
      template <typename T>
      __global__ void element_op_kernel(T * vec1,
                                         unsigned int start1,
                                         unsigned int inc1,
                                         unsigned int size1,
                                          
                                         T const * vec2,
                                         unsigned int start2,
                                         unsigned int inc2,
                                         
                                         T const * vec3,
                                         unsigned int start3,
                                         unsigned int inc3,
                                         
                                         unsigned int is_division
                                       )
      { 
        if (is_division)
        {
          for (unsigned int i = blockDim.x * blockIdx.x + threadIdx.x;
                            i < size1;
                            i += gridDim.x * blockDim.x)
          {
            vec1[i*inc1+start1] = vec2[i*inc2+start2] / vec3[i*inc3+start3];
          }
        }
        else
        {
          for (unsigned int i = blockDim.x * blockIdx.x + threadIdx.x;
                            i < size1;
                            i += gridDim.x * blockDim.x)
          {
            vec1[i*inc1+start1] = vec2[i*inc2+start2] * vec3[i*inc3+start3];
          }
        }
      }
 
      
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
        
        element_op_kernel<<<128, 128>>>(detail::cuda_arg<value_type>(vec1),
                                        static_cast<unsigned int>(viennacl::traits::start(vec1)),
                                        static_cast<unsigned int>(viennacl::traits::stride(vec1)),
                                        static_cast<unsigned int>(viennacl::traits::size(vec1)),
                                          
                                        detail::cuda_arg<value_type>(proxy.lhs()),
                                        static_cast<unsigned int>(viennacl::traits::start(proxy.lhs())),
                                        static_cast<unsigned int>(viennacl::traits::stride(proxy.lhs())),
                                        
                                        detail::cuda_arg<value_type>(proxy.rhs()),
                                        static_cast<unsigned int>(viennacl::traits::start(proxy.rhs())),
                                        static_cast<unsigned int>(viennacl::traits::stride(proxy.rhs())),
                                        
                                        static_cast<unsigned int>(viennacl::is_division<OP>::value)
                                       );                                          
        VIENNACL_CUDA_LAST_ERROR_CHECK("element_op_kernel");
      }


      ///////////////////////// Norms and inner product ///////////////////


      template <typename T>
      __global__ void inner_prod_kernel(const T * vec1,
                                        unsigned int start1,
                                        unsigned int inc1,
                                        unsigned int size1,
                                        const T * vec2,
                                        unsigned int start2,
                                        unsigned int inc2,
                                        unsigned int size2,
                                        T * group_buffer)
      {
        __shared__ T tmp_buffer[128]; 
        unsigned int group_start1 = (blockIdx.x * size1) / (gridDim.x) * inc1 + start1;
        unsigned int group_start2 = (blockIdx.x * size2) / (gridDim.x) * inc2 + start2;
        
        unsigned int group_size1 = ((blockIdx.x + 1) * size1) / (gridDim.x)
                                     - (  blockIdx.x * size1) / (gridDim.x);
                                     

        T tmp = 0;
        for (unsigned int i = threadIdx.x; i < group_size1; i += blockDim.x)
          tmp += vec1[i*inc1+group_start1] * vec2[i*inc2+group_start2];
        tmp_buffer[threadIdx.x] = tmp;

        // parallel reduction
        for (unsigned int stride = blockDim.x/2; stride > 0; stride /= 2)
        {
          __syncthreads();
          if (threadIdx.x < stride)
            tmp_buffer[threadIdx.x] += tmp_buffer[threadIdx.x+stride];
        }
        
        if (threadIdx.x == 0)
          group_buffer[blockIdx.x] = tmp_buffer[0];
        
      }

      
      
      // sums the array 'vec1' and writes to result. Makes use of a single work-group only. 
      template <typename T>
      __global__ void vector_sum_kernel(
                T * vec1,
                unsigned int start1,
                unsigned int inc1,
                unsigned int size1,
                unsigned int option, //0: use fmax, 1: just sum, 2: sum and return sqrt of sum
                T * result) 
      { 
        __shared__ T tmp_buffer[128]; 
        T thread_sum = 0;
        for (unsigned int i = threadIdx.x; i<size1; i += blockDim.x)
        {
          if (option > 0)
            thread_sum += vec1[i*inc1+start1];
          else
            thread_sum = fmax(thread_sum, fabs(vec1[i*inc1+start1]));
        }
        
        tmp_buffer[threadIdx.x] = thread_sum;

        for (unsigned int stride = blockDim.x/2; stride > 0; stride /= 2)
        {
          if (threadIdx.x < stride)
          {
            if (option > 0)
              tmp_buffer[threadIdx.x] += tmp_buffer[threadIdx.x + stride];
            else
              tmp_buffer[threadIdx.x] = fmax(tmp_buffer[threadIdx.x], tmp_buffer[threadIdx.x + stride]);
          }
          __syncthreads();
        }
        
        if (threadIdx.x == 0)
        {
          if (option == 2)
            *result = sqrt(tmp_buffer[0]);
          else
            *result = tmp_buffer[0];
        }
      }



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
        
        static const unsigned int work_groups = 128;
        static viennacl::vector<value_type> temp(work_groups);
        
        inner_prod_kernel<<<128, 128>>>(detail::cuda_arg<value_type>(vec1),
                                        static_cast<unsigned int>(viennacl::traits::start(vec1)),
                                        static_cast<unsigned int>(viennacl::traits::stride(vec1)),
                                        static_cast<unsigned int>(viennacl::traits::size(vec1)),
                                        detail::cuda_arg<value_type>(vec2),
                                        static_cast<unsigned int>(viennacl::traits::start(vec2)),
                                        static_cast<unsigned int>(viennacl::traits::stride(vec2)),
                                        static_cast<unsigned int>(viennacl::traits::size(vec2)),
                                        detail::cuda_arg<value_type>(temp)
                                       );
        VIENNACL_CUDA_LAST_ERROR_CHECK("inner_prod_kernel");
        
        vector_sum_kernel<<<1, 128>>>(detail::cuda_arg<value_type>(temp),
                                      static_cast<unsigned int>(viennacl::traits::start(temp)),
                                      static_cast<unsigned int>(viennacl::traits::stride(temp)),
                                      static_cast<unsigned int>(viennacl::traits::size(temp)),
                                      static_cast<unsigned int>(1),
                                      detail::cuda_arg<value_type>(result) );
        VIENNACL_CUDA_LAST_ERROR_CHECK("vector_sum_kernel");
      }

      
      /** @brief Computes the inner product of two vectors - implementation. Library users should call inner_prod(vec1, vec2).
      *
      * @param vec1 The first vector
      * @param vec2 The second vector
      * @param result The result scalar (on the host)
      */
      template <typename V1, typename V2, typename S3>
      typename viennacl::enable_if<    viennacl::is_any_dense_nonstructured_vector<V1>::value
                                    && viennacl::is_any_dense_nonstructured_vector<V2>::value
                                    && viennacl::is_cpu_scalar<S3>::value
                                  >::type
      inner_prod_cpu(V1 const & vec1,
                     V2 const & vec2,
                     S3 & result)
      {
        typedef typename viennacl::result_of::cpu_value_type<V1>::type        value_type;
        
        static const unsigned int work_groups = 128;
        static viennacl::vector<value_type> temp(work_groups);
                
        inner_prod_kernel<<<128, 128>>>(detail::cuda_arg<value_type>(vec1),
                                        static_cast<unsigned int>(viennacl::traits::start(vec1)),
                                        static_cast<unsigned int>(viennacl::traits::stride(vec1)),
                                        static_cast<unsigned int>(viennacl::traits::size(vec1)),
                                        detail::cuda_arg<value_type>(vec2),
                                        static_cast<unsigned int>(viennacl::traits::start(vec2)),
                                        static_cast<unsigned int>(viennacl::traits::stride(vec2)),
                                        static_cast<unsigned int>(viennacl::traits::size(vec2)),
                                        detail::cuda_arg<value_type>(temp)
                                       );        
        VIENNACL_CUDA_LAST_ERROR_CHECK("inner_prod_kernel");

        // Now copy partial results from GPU back to CPU and run reduction there:
        static std::vector<value_type> temp_cpu(work_groups);
        viennacl::fast_copy(temp.begin(), temp.end(), temp_cpu.begin());
        
        result = 0;
        for (typename std::vector<value_type>::const_iterator it = temp_cpu.begin(); it != temp_cpu.end(); ++it)
          result += *it;
      }
      
      ///////////////////////////////////
      
      
      template <typename T>
      __device__ T impl_norm_kernel( const T * vec,
                                     unsigned int start1,
                                     unsigned int inc1,
                                     unsigned int size1,
                                     unsigned int norm_selector,
                                     T * tmp_buffer)
      {
        T tmp = 0;
        if (norm_selector == 1) //norm_1
        {
          for (unsigned int i = threadIdx.x; i < size1; i += blockDim.x)
            tmp += fabs(vec[i*inc1 + start1]);
        }
        else if (norm_selector == 2) //norm_2
        {
          T vec_entry = 0;
          for (unsigned int i = threadIdx.x; i < size1; i += blockDim.x)
          {
            vec_entry = vec[i*inc1 + start1];
            tmp += vec_entry * vec_entry;
          }
        }
        else if (norm_selector == 0) //norm_inf
        {
          for (unsigned int i = threadIdx.x; i < size1; i += blockDim.x)
            tmp = fmax(fabs(vec[i*inc1 + start1]), tmp);
        }
        
        tmp_buffer[threadIdx.x] = tmp;

        if (norm_selector > 0) //parallel reduction for norm_1 or norm_2:
        {
          for (unsigned int stride = blockDim.x/2; stride > 0; stride /= 2)
          {
            __syncthreads();
            if (threadIdx.x < stride)
              tmp_buffer[threadIdx.x] += tmp_buffer[threadIdx.x+stride];
          }
          return tmp_buffer[0];
        }
        
        //norm_inf:
        for (unsigned int stride = blockDim.x/2; stride > 0; stride /= 2)
        {
          __syncthreads();
          if (threadIdx.x < stride)
            tmp_buffer[threadIdx.x] = fmax(tmp_buffer[threadIdx.x], tmp_buffer[threadIdx.x+stride]);
        }
        
        return tmp_buffer[0];
      };

      template <typename T>
      __global__ void norm_kernel(
                 const T * vec,
                unsigned int start1,
                unsigned int inc1,
                unsigned int size1,
                unsigned int norm_selector,
                T * group_buffer)
      {
        __shared__ T tmp_buffer[128];
        T tmp = impl_norm_kernel(vec,
                                 (        blockIdx.x  * size1) / gridDim.x * inc1 + start1,
                                 inc1,
                                 (   (1 + blockIdx.x) * size1) / gridDim.x
                                 - (      blockIdx.x  * size1) / gridDim.x,
                                 norm_selector,
                                 tmp_buffer);
        
        if (threadIdx.x == 0)
          group_buffer[blockIdx.x] = tmp;  
      }

      
      /** @brief Computes the l^1-norm of a vector
      *
      * @param vec1 The vector
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
        
        static std::size_t work_groups = 128;
        static viennacl::vector<value_type> temp(work_groups);
        
        norm_kernel<<<128, 128>>>(detail::cuda_arg<value_type>(vec1),
                                  static_cast<unsigned int>(viennacl::traits::start(vec1)),
                                  static_cast<unsigned int>(viennacl::traits::stride(vec1)),
                                  static_cast<unsigned int>(viennacl::traits::size(vec1)),
                                  static_cast<unsigned int>(1),
                                  detail::cuda_arg<value_type>(temp)
                                 );
        VIENNACL_CUDA_LAST_ERROR_CHECK("norm_kernel");
      
        vector_sum_kernel<<<1, 128>>>(detail::cuda_arg<value_type>(temp),
                                      static_cast<unsigned int>(viennacl::traits::start(temp)),
                                      static_cast<unsigned int>(viennacl::traits::stride(temp)),
                                      static_cast<unsigned int>(viennacl::traits::size(temp)),
                                      static_cast<unsigned int>(1),
                                      detail::cuda_arg<value_type>(result) );
        VIENNACL_CUDA_LAST_ERROR_CHECK("vector_sum_kernel");
      }

      /** @brief Computes the l^1-norm of a vector
      *
      * @param vec1 The vector
      * @param result The result scalar
      */
      template <typename V1, typename S2>
      typename viennacl::enable_if<    viennacl::is_any_dense_nonstructured_vector<V1>::value
                                    && viennacl::is_cpu_scalar<S2>::value
                                  >::type
      norm_1_cpu(V1 const & vec1,
                 S2 & result)
      {
        typedef typename viennacl::result_of::cpu_value_type<V1>::type        value_type;
        
        static std::size_t work_groups = 128;
        static viennacl::vector<value_type> temp(work_groups);
        
        norm_kernel<<<128, 128>>>(detail::cuda_arg<value_type>(vec1),
                                  static_cast<unsigned int>(viennacl::traits::start(vec1)),
                                  static_cast<unsigned int>(viennacl::traits::stride(vec1)),
                                  static_cast<unsigned int>(viennacl::traits::size(vec1)),
                                  static_cast<unsigned int>(1),
                                  detail::cuda_arg<value_type>(temp)
                                 );
        VIENNACL_CUDA_LAST_ERROR_CHECK("norm_kernel");
      
        // Now copy partial results from GPU back to CPU and run reduction there:
        static std::vector<value_type> temp_cpu(work_groups);
        viennacl::fast_copy(temp.begin(), temp.end(), temp_cpu.begin());
        
        result = 0;
        for (typename std::vector<value_type>::const_iterator it = temp_cpu.begin(); it != temp_cpu.end(); ++it)
          result += *it;
      }

      ///// norm_2
      
      /** @brief Computes the l^2-norm of a vector - implementation
      *
      * @param vec1 The vector
      * @param result The result scalar
      */
      template <typename V1, typename S2>
      typename viennacl::enable_if<    viennacl::is_any_dense_nonstructured_vector<V1>::value
                                    && viennacl::is_scalar<S2>::value
                                  >::type
      norm_2_impl(V1 const & vec1,
                  S2 & result)
      {
        typedef typename viennacl::result_of::cpu_value_type<V1>::type        value_type;
        
        static std::size_t work_groups = 128;
        static viennacl::vector<value_type> temp(work_groups);
        
        norm_kernel<<<128, 128>>>(detail::cuda_arg<value_type>(vec1),
                                  static_cast<unsigned int>(viennacl::traits::start(vec1)),
                                  static_cast<unsigned int>(viennacl::traits::stride(vec1)),
                                  static_cast<unsigned int>(viennacl::traits::size(vec1)),
                                  static_cast<unsigned int>(2),
                                  detail::cuda_arg<value_type>(temp)
                                 );
        VIENNACL_CUDA_LAST_ERROR_CHECK("norm_kernel");
      
        vector_sum_kernel<<<1, 128>>>(detail::cuda_arg<value_type>(temp),
                                      static_cast<unsigned int>(viennacl::traits::start(temp)),
                                      static_cast<unsigned int>(viennacl::traits::stride(temp)),
                                      static_cast<unsigned int>(viennacl::traits::size(temp)),
                                      static_cast<unsigned int>(2),
                                      detail::cuda_arg<value_type>(result) );
        VIENNACL_CUDA_LAST_ERROR_CHECK("vector_sum_kernel");
      }

      /** @brief Computes the l^2-norm of a vector - implementation
      *
      * @param vec1 The vector
      * @param result The result scalar
      */
      template <typename V1, typename S2>
      typename viennacl::enable_if<    viennacl::is_any_dense_nonstructured_vector<V1>::value
                                    && viennacl::is_cpu_scalar<S2>::value
                                  >::type
      norm_2_cpu(V1 const & vec1,
                  S2 & result)
      {
        typedef typename viennacl::result_of::cpu_value_type<V1>::type        value_type;
        
        static std::size_t work_groups = 128;
        static viennacl::vector<value_type> temp(work_groups);
        
        norm_kernel<<<128, 128>>>(detail::cuda_arg<value_type>(vec1),
                                  static_cast<unsigned int>(viennacl::traits::start(vec1)),
                                  static_cast<unsigned int>(viennacl::traits::stride(vec1)),
                                  static_cast<unsigned int>(viennacl::traits::size(vec1)),
                                  static_cast<unsigned int>(2),
                                  detail::cuda_arg<value_type>(temp)
                                 );
        VIENNACL_CUDA_LAST_ERROR_CHECK("norm_kernel");
      
        static std::vector<value_type> temp_cpu(work_groups);
        viennacl::fast_copy(temp.begin(), temp.end(), temp_cpu.begin());
        
        result = 0;
        for (typename std::vector<value_type>::const_iterator it = temp_cpu.begin(); it != temp_cpu.end(); ++it)
          result += *it;
        result = std::sqrt(result);
      }

      
      ////// norm_inf
      
      /** @brief Computes the supremum-norm of a vector
      *
      * @param vec1 The vector
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
        
        static std::size_t work_groups = 128;
        static viennacl::vector<value_type> temp(work_groups);
        
        norm_kernel<<<128, 128>>>(detail::cuda_arg<value_type>(vec1),
                                  static_cast<unsigned int>(viennacl::traits::start(vec1)),
                                  static_cast<unsigned int>(viennacl::traits::stride(vec1)),
                                  static_cast<unsigned int>(viennacl::traits::size(vec1)),
                                  static_cast<unsigned int>(0),
                                  detail::cuda_arg<value_type>(temp)
                                 );
        VIENNACL_CUDA_LAST_ERROR_CHECK("norm_kernel");
      
        vector_sum_kernel<<<1, 128>>>(detail::cuda_arg<value_type>(temp),
                                      static_cast<unsigned int>(viennacl::traits::start(temp)),
                                      static_cast<unsigned int>(viennacl::traits::stride(temp)),
                                      static_cast<unsigned int>(viennacl::traits::size(temp)),
                                      static_cast<unsigned int>(0),
                                      detail::cuda_arg<value_type>(result) );
        VIENNACL_CUDA_LAST_ERROR_CHECK("vector_sum_kernel");
      }

      
      
      /** @brief Computes the supremum-norm of a vector
      *
      * @param vec1 The vector
      * @param result The result scalar
      */
      template <typename V1, typename S2>
      typename viennacl::enable_if<    viennacl::is_any_dense_nonstructured_vector<V1>::value
                                    && viennacl::is_cpu_scalar<S2>::value
                                  >::type
      norm_inf_cpu(V1 const & vec1,
                    S2 & result)
      {
        typedef typename viennacl::result_of::cpu_value_type<V1>::type        value_type;
        
        static std::size_t work_groups = 128;
        static viennacl::vector<value_type> temp(work_groups);
        
        norm_kernel<<<128, 128>>>(detail::cuda_arg<value_type>(vec1),
                                  static_cast<unsigned int>(viennacl::traits::start(vec1)),
                                  static_cast<unsigned int>(viennacl::traits::stride(vec1)),
                                  static_cast<unsigned int>(viennacl::traits::size(vec1)),
                                  static_cast<unsigned int>(0),
                                  detail::cuda_arg<value_type>(temp)
                                 );
        VIENNACL_CUDA_LAST_ERROR_CHECK("norm_kernel");
      
        static std::vector<value_type> temp_cpu(work_groups);
        viennacl::fast_copy(temp.begin(), temp.end(), temp_cpu.begin());
        
        result = 0;
        for (typename std::vector<value_type>::const_iterator it = temp_cpu.begin(); it != temp_cpu.end(); ++it)
          result = std::max(result, *it);
      }
      
      
      //////////////////////////////////////
      
      

      //index_norm_inf:
      template <typename T>
      __device__ unsigned int index_norm_inf_impl_kernel( const T * vec,
                                                          unsigned int start1,
                                                          unsigned int inc1,
                                                          unsigned int size1,
                                                          T * float_buffer,
                                                          unsigned int * index_buffer)
      {
        //step 1: fill buffer:
        T cur_max = (T)0;
        T tmp;
        for (unsigned int i = threadIdx.x; i < size1; i += blockDim.x)
        {
          tmp = fabs(vec[i*inc1+start1]);
          if (cur_max < tmp)
          {
            float_buffer[threadIdx.x] = tmp;
            index_buffer[threadIdx.x] = i;
            cur_max = tmp;
          }
        }
        
        //step 2: parallel reduction:
        for (unsigned int stride = blockDim.x/2; stride > 0; stride /= 2)
        {
          __syncthreads();
          if (threadIdx.x < stride)
          {
            //find the first occurring index
            if (float_buffer[threadIdx.x] < float_buffer[threadIdx.x+stride])
            {
              index_buffer[threadIdx.x] = index_buffer[threadIdx.x+stride];
              float_buffer[threadIdx.x] = float_buffer[threadIdx.x+stride];
            }
          }
        }
        
        return index_buffer[0];
      }

      template <typename T>
      __global__ void index_norm_inf_kernel(const T * vec,
                                            unsigned int start1,
                                            unsigned int inc1,
                                            unsigned int size1,
                                            unsigned int * result) 
      { 
        __shared__ T float_buffer[128];
        __shared__ unsigned int index_buffer[128];
        
        unsigned int tmp = index_norm_inf_impl_kernel(vec, start1, inc1, size1, float_buffer, index_buffer);
        
        if (threadIdx.x == 0) 
          *result = tmp;
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

        viennacl::backend::mem_handle h;
        viennacl::backend::memory_create(h, sizeof(unsigned int));
        
        index_norm_inf_kernel<<<1, 128>>>(detail::cuda_arg<value_type>(vec1),
                                          static_cast<unsigned int>(viennacl::traits::start(vec1)),
                                          static_cast<unsigned int>(viennacl::traits::stride(vec1)),
                                          static_cast<unsigned int>(viennacl::traits::size(vec1)),
                                          detail::cuda_arg<unsigned int>(h.cuda_handle())
                                        );
        VIENNACL_CUDA_LAST_ERROR_CHECK("index_norm_inf_kernel");
        
        unsigned int ret = 0;
        viennacl::backend::memory_read(h, 0, sizeof(unsigned int), &ret);
        return static_cast<std::size_t>(ret);
      }

      ///////////////////////////////////////////
      
      template <typename T>
      __global__ void plane_rotation_kernel(
                T * vec1,
                unsigned int start1,
                unsigned int inc1,
                unsigned int size1,
                T * vec2, 
                unsigned int start2,
                unsigned int inc2,
                unsigned int size2,
                T alpha,
                T beta) 
      { 
        T tmp1 = 0;
        T tmp2 = 0;

        for (unsigned int i = blockDim.x * blockIdx.x + threadIdx.x; i < size1; i += blockDim.x * gridDim.x)
        {
          tmp1 = vec1[i*inc1+start1];
          tmp2 = vec2[i*inc2+start2];
          
          vec1[i*inc1+start1] = alpha * tmp1 + beta * tmp2;
          vec2[i*inc2+start2] = alpha * tmp2 - beta * tmp1;
        }

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

        value_type temporary_alpha;                             
        if (viennacl::is_cpu_scalar<SCALARTYPE>::value)
          temporary_alpha = alpha;
        
        value_type temporary_beta;                             
        if (viennacl::is_cpu_scalar<SCALARTYPE>::value)
          temporary_beta = beta;
        
        plane_rotation_kernel<<<128, 128>>>(detail::cuda_arg<value_type>(vec1),
                                            static_cast<unsigned int>(viennacl::traits::start(vec1)),
                                            static_cast<unsigned int>(viennacl::traits::stride(vec1)),                                 
                                            static_cast<unsigned int>(viennacl::traits::size(vec1)),                                 
                                            detail::cuda_arg<value_type>(vec2),
                                            static_cast<unsigned int>(viennacl::traits::start(vec2)),
                                            static_cast<unsigned int>(viennacl::traits::stride(vec2)),                                 
                                            static_cast<unsigned int>(viennacl::traits::size(vec2)),                                 
                                            detail::cuda_arg<value_type>(detail::arg_reference(alpha, temporary_alpha)),
                                            detail::cuda_arg<value_type>(detail::arg_reference(beta, temporary_beta)) );
        VIENNACL_CUDA_LAST_ERROR_CHECK("plane_rotation_kernel");
      }

    } //namespace opencl
  } //namespace linalg
} //namespace viennacl


#endif
