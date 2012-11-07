#ifndef VIENNACL_LINALG_SCALAR_OPERATIONS_HPP
#define VIENNACL_LINALG_SCALAR_OPERATIONS_HPP

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

/** @file viennacl/linalg/scalar_operations.hpp
    @brief Implementations of scalar operations.
*/

#include "viennacl/forwards.h"
#include "viennacl/scalar.hpp"
#include "viennacl/tools/tools.hpp"
#include "viennacl/meta/predicate.hpp"
#include "viennacl/meta/enable_if.hpp"
#include "viennacl/traits/size.hpp"
#include "viennacl/traits/start.hpp"
#include "viennacl/traits/handle.hpp"
#include "viennacl/traits/stride.hpp"
#include "viennacl/linalg/single_threaded/scalar_operations.hpp"

#ifdef VIENNACL_WITH_OPENCL
  #include "viennacl/linalg/opencl/scalar_operations.hpp"
#endif

#ifdef VIENNACL_WITH_CUDA
  #include "viennacl/linalg/cuda/scalar_operations.hpp"
#endif



namespace viennacl
{
  namespace linalg
  {
    
    /** @brief Interface for the generic operation s1 = s2 @ alpha, where s1 and s2 are GPU scalars, @ denotes multiplication or division, and alpha is either a GPU or a CPU scalar 
     * 
     * @param s1                The first  (GPU) scalar
     * @param s2                The second (GPU) scalar
     * @param len_alpha         If alpha is obtained from summing over a small GPU vector (e.g. the final summation after a multi-group reduction), then supply the length of the array here
     * @param reciprocal_alpha  If true, then s2 / alpha instead of s2 * alpha is computed
     * @param flip_sign_alpha   If true, then (-alpha) is used instead of alpha
     */
    template <typename S1,
              typename S2, typename ScalarType1>
    typename viennacl::enable_if< viennacl::is_scalar<S1>::value
                                  && viennacl::is_scalar<S2>::value
                                  && viennacl::is_any_scalar<ScalarType1>::value
                                >::type
    as(S1 & s1, 
       S2 const & s2, ScalarType1 const & alpha, std::size_t len_alpha, bool reciprocal_alpha, bool flip_sign_alpha) 
    {
      switch (viennacl::traits::handle(s1).get_active_handle_id())
      {
        case viennacl::backend::MAIN_MEMORY:
          viennacl::linalg::single_threaded::as(s1, s2, alpha, len_alpha, reciprocal_alpha, flip_sign_alpha);
          break;
#ifdef VIENNACL_WITH_OPENCL          
        case viennacl::backend::OPENCL_MEMORY:
          viennacl::linalg::opencl::as(s1, s2, alpha, len_alpha, reciprocal_alpha, flip_sign_alpha);
          break;
#endif          
#ifdef VIENNACL_WITH_CUDA
        case viennacl::backend::CUDA_MEMORY:
          viennacl::linalg::cuda::as(s1, s2, alpha, len_alpha, reciprocal_alpha, flip_sign_alpha);
          break;
#endif          
        default:
          throw "not implemented";
      }
    }
    
    
    /** @brief Interface for the generic operation s1 = s2 @ alpha + s3 @ beta, where s1, s2 and s3 are GPU scalars, @ denotes multiplication or division, and alpha, beta are either a GPU or a CPU scalar
     * 
     * @param s1                The first  (GPU) scalar
     * @param s2                The second (GPU) scalar
     * @param len_alpha         If alpha is a small GPU vector, which needs to be summed in order to obtain the final scalar, then supply the length of the array here
     * @param reciprocal_alpha  If true, then s2 / alpha instead of s2 * alpha is computed
     * @param flip_sign_alpha   If true, then (-alpha) is used instead of alpha
     * @param s3                The third (GPU) scalar
     * @param len_beta          If beta is obtained from summing over a small GPU vector (e.g. the final summation after a multi-group reduction), then supply the length of the array here
     * @param reciprocal_beta   If true, then s2 / beta instead of s2 * beta is computed
     * @param flip_sign_beta    If true, then (-beta) is used instead of beta
     */
    template <typename S1,
              typename S2, typename ScalarType1,
              typename S3, typename ScalarType2>
    typename viennacl::enable_if< viennacl::is_scalar<S1>::value
                                  && viennacl::is_scalar<S2>::value
                                  && viennacl::is_scalar<S3>::value
                                  && viennacl::is_any_scalar<ScalarType1>::value
                                  && viennacl::is_any_scalar<ScalarType2>::value
                                >::type
    asbs(S1 & vec1, 
         S2 const & vec2, ScalarType1 const & alpha, std::size_t len_alpha, bool reciprocal_alpha, bool flip_sign_alpha,
         S3 const & vec3, ScalarType2 const & beta,  std::size_t len_beta,  bool reciprocal_beta,  bool flip_sign_beta) 
    {
      switch (viennacl::traits::handle(vec1).get_active_handle_id())
      {
        case viennacl::backend::MAIN_MEMORY:
          viennacl::linalg::single_threaded::asbs(vec1,
                                                  vec2, alpha, len_alpha, reciprocal_alpha, flip_sign_alpha,
                                                  vec3,  beta, len_beta,  reciprocal_beta,  flip_sign_beta);
          break;
#ifdef VIENNACL_WITH_OPENCL          
        case viennacl::backend::OPENCL_MEMORY:
          viennacl::linalg::opencl::asbs(vec1,
                                         vec2, alpha, len_alpha, reciprocal_alpha, flip_sign_alpha,
                                         vec3,  beta, len_beta,  reciprocal_beta,  flip_sign_beta);
          break;
#endif
#ifdef VIENNACL_WITH_CUDA
        case viennacl::backend::CUDA_MEMORY:
          viennacl::linalg::cuda::asbs(vec1,
                                       vec2, alpha, len_alpha, reciprocal_alpha, flip_sign_alpha,
                                       vec3,  beta, len_beta,  reciprocal_beta,  flip_sign_beta);
          break;
#endif
        default:
          throw "not implemented";
      }
    }
    
    
    /** @brief Interface for the generic operation s1 += s2 @ alpha + s3 @ beta, where s1, s2 and s3 are GPU scalars, @ denotes multiplication or division, and alpha, beta are either a GPU or a CPU scalar
     * 
     * @param s1                The first  (GPU) scalar
     * @param s2                The second (GPU) scalar
     * @param len_alpha         If alpha is a small GPU vector, which needs to be summed in order to obtain the final scalar, then supply the length of the array here
     * @param reciprocal_alpha  If true, then s2 / alpha instead of s2 * alpha is computed
     * @param flip_sign_alpha   If true, then (-alpha) is used instead of alpha
     * @param s3                The third (GPU) scalar
     * @param len_beta          If beta is obtained from summing over a small GPU vector (e.g. the final summation after a multi-group reduction), then supply the length of the array here
     * @param reciprocal_beta   If true, then s2 / beta instead of s2 * beta is computed
     * @param flip_sign_beta    If true, then (-beta) is used instead of beta
     */
    template <typename S1,
              typename S2, typename ScalarType1,
              typename S3, typename ScalarType2>
    typename viennacl::enable_if< viennacl::is_scalar<S1>::value
                                  && viennacl::is_scalar<S2>::value
                                  && viennacl::is_scalar<S3>::value
                                  && viennacl::is_any_scalar<ScalarType1>::value
                                  && viennacl::is_any_scalar<ScalarType2>::value
                                >::type
    asbs_s(S1 & vec1,
           S2 const & vec2, ScalarType1 const & alpha, std::size_t len_alpha, bool reciprocal_alpha, bool flip_sign_alpha,
           S3 const & vec3, ScalarType2 const & beta,  std::size_t len_beta,  bool reciprocal_beta,  bool flip_sign_beta) 
    {
      switch (viennacl::traits::handle(vec1).get_active_handle_id())
      {
        case viennacl::backend::MAIN_MEMORY:
          viennacl::linalg::single_threaded::asbs_s(vec1,
                                                    vec2, alpha, len_alpha, reciprocal_alpha, flip_sign_alpha,
                                                    vec3,  beta, len_beta,  reciprocal_beta,  flip_sign_beta);
          break;
#ifdef VIENNACL_WITH_OPENCL          
        case viennacl::backend::OPENCL_MEMORY:
          viennacl::linalg::opencl::asbs_s(vec1,
                                           vec2, alpha, len_alpha, reciprocal_alpha, flip_sign_alpha,
                                           vec3,  beta, len_beta,  reciprocal_beta,  flip_sign_beta);
          break;
#endif          
#ifdef VIENNACL_WITH_CUDA
        case viennacl::backend::CUDA_MEMORY:
          viennacl::linalg::cuda::asbs_s(vec1,
                                         vec2, alpha, len_alpha, reciprocal_alpha, flip_sign_alpha,
                                         vec3,  beta, len_beta,  reciprocal_beta,  flip_sign_beta);
          break;
#endif          
        default:
          throw "not implemented";
      }
    }
    
    
    
    /** @brief Swaps the contents of two scalars
    *
    * @param vec1   The first scalar
    * @param vec2   The second scalar
    */
    template <typename S1, typename S2>
    typename viennacl::enable_if<    viennacl::is_scalar<S1>::value
                                  && viennacl::is_scalar<S2>::value
                                >::type
    swap(S1 & s1, S2 & s2)
    {
      switch (viennacl::traits::handle(s1).get_active_handle_id())
      {
        case viennacl::backend::MAIN_MEMORY:
          viennacl::linalg::single_threaded::swap(s1, s2);
          break;
#ifdef VIENNACL_WITH_OPENCL          
        case viennacl::backend::OPENCL_MEMORY:
          viennacl::linalg::opencl::swap(s1, s2);
          break;
#endif
#ifdef VIENNACL_WITH_CUDA
        case viennacl::backend::CUDA_MEMORY:
          viennacl::linalg::cuda::swap(s1, s2);
          break;
#endif
        default:
          throw "not implemented";
      }
    }

    
  } //namespace linalg
} //namespace viennacl


#endif
