#ifndef VIENNACL_VECTOR_OPERATIONS_HPP_
#define VIENNACL_VECTOR_OPERATIONS_HPP_

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

/** @file viennacl/linalg/vector_operations.hpp
    @brief Implementations of vector operations.
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
#include "viennacl/linalg/single_threaded/vector_operations.hpp"

#ifdef VIENNACL_WITH_OPENCL
  #include "viennacl/linalg/opencl/vector_operations.hpp"
#endif

#ifdef VIENNACL_WITH_CUDA
  #include "viennacl/linalg/cuda/vector_operations.hpp"
#endif

namespace viennacl
{
  namespace linalg
  {
    template <typename V1,
              typename V2, typename ScalarType1>
    typename viennacl::enable_if<    viennacl::is_any_dense_nonstructured_vector<V1>::value
                                  && viennacl::is_any_dense_nonstructured_vector<V2>::value
                                  && viennacl::is_any_scalar<ScalarType1>::value
                                >::type
    av(V1 & vec1, 
       V2 const & vec2, ScalarType1 const & alpha, std::size_t len_alpha, bool reciprocal_alpha, bool flip_sign_alpha) 
    {
      assert(viennacl::traits::size(vec1) == viennacl::traits::size(vec2) && bool("Incompatible vector sizes in v1 = v2 @ alpha: size(v1) != size(v2)"));
      
      switch (viennacl::traits::handle(vec1).get_active_handle_id())
      {
        case viennacl::backend::MAIN_MEMORY:
          viennacl::linalg::single_threaded::av(vec1, vec2, alpha, len_alpha, reciprocal_alpha, flip_sign_alpha);
          break;
#ifdef VIENNACL_WITH_OPENCL          
        case viennacl::backend::OPENCL_MEMORY:
          viennacl::linalg::opencl::av(vec1, vec2, alpha, len_alpha, reciprocal_alpha, flip_sign_alpha);
          break;
#endif
#ifdef VIENNACL_WITH_CUDA
        case viennacl::backend::CUDA_MEMORY:
          viennacl::linalg::cuda::av(vec1, vec2, alpha, len_alpha, reciprocal_alpha, flip_sign_alpha);
          break;
#endif
        default:
          throw "not implemented";
      }
    }
    
    
    template <typename V1,
              typename V2, typename ScalarType1,
              typename V3, typename ScalarType2>
    typename viennacl::enable_if<    viennacl::is_any_dense_nonstructured_vector<V1>::value
                                  && viennacl::is_any_dense_nonstructured_vector<V2>::value
                                  && viennacl::is_any_dense_nonstructured_vector<V3>::value
                                  && viennacl::is_any_scalar<ScalarType1>::value
                                  && viennacl::is_any_scalar<ScalarType2>::value
                                >::type
    avbv(V1 & vec1, 
         V2 const & vec2, ScalarType1 const & alpha, std::size_t len_alpha, bool reciprocal_alpha, bool flip_sign_alpha,
         V3 const & vec3, ScalarType2 const & beta, std::size_t len_beta, bool reciprocal_beta, bool flip_sign_beta) 
    {
      assert(viennacl::traits::size(vec1) == viennacl::traits::size(vec2) && bool("Incompatible vector sizes in v1 = v2 @ alpha + v3 @ beta: size(v1) != size(v2)"));
      assert(viennacl::traits::size(vec2) == viennacl::traits::size(vec3) && bool("Incompatible vector sizes in v1 = v2 @ alpha + v3 @ beta: size(v2) != size(v3)"));
      
      switch (viennacl::traits::handle(vec1).get_active_handle_id())
      {
        case viennacl::backend::MAIN_MEMORY:
          viennacl::linalg::single_threaded::avbv(vec1,
                                                  vec2, alpha, len_alpha, reciprocal_alpha, flip_sign_alpha,
                                                  vec3,  beta, len_beta,  reciprocal_beta,  flip_sign_beta);
          break;
#ifdef VIENNACL_WITH_OPENCL          
        case viennacl::backend::OPENCL_MEMORY:
          viennacl::linalg::opencl::avbv(vec1,
                                         vec2, alpha, len_alpha, reciprocal_alpha, flip_sign_alpha,
                                         vec3,  beta, len_beta,  reciprocal_beta,  flip_sign_beta);
          break;
#endif
#ifdef VIENNACL_WITH_CUDA
        case viennacl::backend::CUDA_MEMORY:
          viennacl::linalg::cuda::avbv(vec1,
                                       vec2, alpha, len_alpha, reciprocal_alpha, flip_sign_alpha,
                                       vec3,  beta, len_beta,  reciprocal_beta,  flip_sign_beta);
          break;
#endif
        default:
          throw "not implemented";
      }
    }
    
    
    template <typename V1,
              typename V2, typename ScalarType1,
              typename V3, typename ScalarType2>
    typename viennacl::enable_if<    viennacl::is_any_dense_nonstructured_vector<V1>::value
                                  && viennacl::is_any_dense_nonstructured_vector<V2>::value
                                  && viennacl::is_any_dense_nonstructured_vector<V3>::value
                                  && viennacl::is_any_scalar<ScalarType1>::value
                                  && viennacl::is_any_scalar<ScalarType2>::value
                                >::type
    avbv_v(V1 & vec1,
           V2 const & vec2, ScalarType1 const & alpha, std::size_t len_alpha, bool reciprocal_alpha, bool flip_sign_alpha,
           V3 const & vec3, ScalarType2 const & beta,  std::size_t len_beta,  bool reciprocal_beta,  bool flip_sign_beta) 
    {
      assert(viennacl::traits::size(vec1) == viennacl::traits::size(vec2) && bool("Incompatible vector sizes in v1 += v2 @ alpha + v3 @ beta: size(v1) != size(v2)"));
      assert(viennacl::traits::size(vec2) == viennacl::traits::size(vec3) && bool("Incompatible vector sizes in v1 += v2 @ alpha + v3 @ beta: size(v2) != size(v3)"));
      
      switch (viennacl::traits::handle(vec1).get_active_handle_id())
      {
        case viennacl::backend::MAIN_MEMORY:
          viennacl::linalg::single_threaded::avbv_v(vec1,
                                                    vec2, alpha, len_alpha, reciprocal_alpha, flip_sign_alpha,
                                                    vec3,  beta, len_beta,  reciprocal_beta,  flip_sign_beta);
          break;
#ifdef VIENNACL_WITH_OPENCL          
        case viennacl::backend::OPENCL_MEMORY:
          viennacl::linalg::opencl::avbv_v(vec1,
                                           vec2, alpha, len_alpha, reciprocal_alpha, flip_sign_alpha,
                                           vec3,  beta, len_beta,  reciprocal_beta,  flip_sign_beta);
          break;
#endif
#ifdef VIENNACL_WITH_CUDA
        case viennacl::backend::CUDA_MEMORY:
          viennacl::linalg::cuda::avbv_v(vec1,
                                         vec2, alpha, len_alpha, reciprocal_alpha, flip_sign_alpha,
                                         vec3,  beta, len_beta,  reciprocal_beta,  flip_sign_beta);
          break;
#endif
        default:
          throw "not implemented";
      }
    }
    
    
    /** @brief Assign a constant value to a vector (-range/-slice)
    *
    * @param vec1   The vector to which the value should be assigned
    * @param alpha  The value to be assigned
    */
    template <typename V1, typename S1>
    typename viennacl::enable_if<    viennacl::is_any_dense_nonstructured_vector<V1>::value
                                  && viennacl::is_cpu_scalar<S1>::value
                                >::type
    vector_assign(V1 & vec1, const S1 & alpha)
    {
      switch (viennacl::traits::handle(vec1).get_active_handle_id())
      {
        case viennacl::backend::MAIN_MEMORY:
          viennacl::linalg::single_threaded::vector_assign(vec1, alpha);
          break;
#ifdef VIENNACL_WITH_OPENCL          
        case viennacl::backend::OPENCL_MEMORY:
          viennacl::linalg::opencl::vector_assign(vec1, alpha);
          break;
#endif
#ifdef VIENNACL_WITH_CUDA
        case viennacl::backend::CUDA_MEMORY:
          viennacl::linalg::cuda::vector_assign(vec1, alpha);
          break;
#endif
        default:
          throw "not implemented";
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
      assert(viennacl::traits::size(vec1) == viennacl::traits::size(vec2) && bool("Incompatible vector sizes in vector_swap()"));

      switch (viennacl::traits::handle(vec1).get_active_handle_id())
      {
        case viennacl::backend::MAIN_MEMORY:
          viennacl::linalg::single_threaded::vector_swap(vec1, vec2);
          break;
#ifdef VIENNACL_WITH_OPENCL          
        case viennacl::backend::OPENCL_MEMORY:
          viennacl::linalg::opencl::vector_swap(vec1, vec2);
          break;
#endif
#ifdef VIENNACL_WITH_CUDA
        case viennacl::backend::CUDA_MEMORY:
          viennacl::linalg::cuda::vector_swap(vec1, vec2);
          break;
#endif
        default:
          throw "not implemented";
      }
    }


    ///////////////////////// Norms and inner product ///////////////////


    //implementation of inner product:
    //namespace {
    /** @brief Computes the inner product of two vectors - dispatcher interface
     *
     * @param vec1 The first vector
     * @param vec2 The second vector
     * @param result The result scalar (on the gpu)
     */
    template <typename V1, typename V2, typename S3>
    typename viennacl::enable_if<    viennacl::is_any_dense_nonstructured_vector<V1>::value
                                  && viennacl::is_any_dense_nonstructured_vector<V2>::value
                                  && viennacl::is_scalar<S3>::value
                                >::type
    inner_prod_impl(V1 const & vec1,
                    V2 const & vec2,
                    S3 & result)
    {
      assert( vec1.size() == vec2.size() && bool("Size mismatch") );
      
      switch (viennacl::traits::handle(vec1).get_active_handle_id())
      {
        case viennacl::backend::MAIN_MEMORY:
          viennacl::linalg::single_threaded::inner_prod_impl(vec1, vec2, result);
          break;
#ifdef VIENNACL_WITH_OPENCL          
        case viennacl::backend::OPENCL_MEMORY:
          viennacl::linalg::opencl::inner_prod_impl(vec1, vec2, result);
          break;
#endif
#ifdef VIENNACL_WITH_CUDA          
        case viennacl::backend::CUDA_MEMORY:
          viennacl::linalg::cuda::inner_prod_impl(vec1, vec2, result);
          break;
#endif
        default:
          throw "not implemented";
      }
    }
    

    /** @brief Computes the inner product of two vectors with the final reduction step on the CPU - dispatcher interface
     *
     * @param vec1 The first vector
     * @param vec2 The second vector
     * @param result The result scalar (on the gpu)
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
      assert( vec1.size() == vec2.size() && bool("Size mismatch") );
      
      switch (viennacl::traits::handle(vec1).get_active_handle_id())
      {
        case viennacl::backend::MAIN_MEMORY:
          viennacl::linalg::single_threaded::inner_prod_impl(vec1, vec2, result);
          break;
#ifdef VIENNACL_WITH_OPENCL          
        case viennacl::backend::OPENCL_MEMORY:
          viennacl::linalg::opencl::inner_prod_cpu(vec1, vec2, result);
          break;
#endif
#ifdef VIENNACL_WITH_CUDA
        case viennacl::backend::CUDA_MEMORY:
          viennacl::linalg::cuda::inner_prod_cpu(vec1, vec2, result);
          break;
#endif
        default:
          throw "not implemented";
      }
    }
    

    //public interface of inner product
    /** @brief Computes the inner product of two vectors - returns an expression template for lazy evaluation.
    *
    * @param vec1 The first vector
    * @param vec2 The second vector
    * @return The result
    */
    template <typename V1, typename V2>
    typename viennacl::enable_if<    viennacl::is_any_dense_nonstructured_vector<V1>::value
                                  && viennacl::is_any_dense_nonstructured_vector<V2>::value,
                                  viennacl::scalar_expression< const V1, const V2, viennacl::op_inner_prod >
                                >::type
    inner_prod_impl(V1 const & vec1,
                    V2 const & vec2)
    {
      return viennacl::scalar_expression< const V1, 
                                          const V2,
                                          viennacl::op_inner_prod >(vec1, vec2);
    }


    
    /** @brief Computes the l^1-norm of a vector - dispatcher interface
    *
    * @param vec The vector
    * @param result The result scalar
    */
    template <typename V1, typename S2>
    typename viennacl::enable_if<    viennacl::is_any_dense_nonstructured_vector<V1>::value
                                  && viennacl::is_scalar<S2>::value
                                >::type
    norm_1_impl(V1 const & vec,
                S2 & result)
    {
      switch (viennacl::traits::handle(vec).get_active_handle_id())
      {
        case viennacl::backend::MAIN_MEMORY:
          viennacl::linalg::single_threaded::norm_1_impl(vec, result);
          break;
#ifdef VIENNACL_WITH_OPENCL          
        case viennacl::backend::OPENCL_MEMORY:
          viennacl::linalg::opencl::norm_1_impl(vec, result);
          break;
#endif
#ifdef VIENNACL_WITH_CUDA
        case viennacl::backend::CUDA_MEMORY:
          viennacl::linalg::cuda::norm_1_impl(vec, result);
          break;
#endif
        default:
          throw "not implemented";
      }
    }

    /** @brief Computes the l^2-norm of a vector - dispatcher interface
    *
    * @param vec The vector
    * @param result The result scalar
    */
    template <typename V1, typename S2>
    typename viennacl::enable_if<    viennacl::is_any_dense_nonstructured_vector<V1>::value
                                  && viennacl::is_scalar<S2>::value
                                >::type
    norm_2_impl(V1 const & vec,
                S2 & result)
    {
      switch (viennacl::traits::handle(vec).get_active_handle_id())
      {
        case viennacl::backend::MAIN_MEMORY:
          viennacl::linalg::single_threaded::norm_2_impl(vec, result);
          break;
#ifdef VIENNACL_WITH_OPENCL          
        case viennacl::backend::OPENCL_MEMORY:
          viennacl::linalg::opencl::norm_2_impl(vec, result);
          break;
#endif
#ifdef VIENNACL_WITH_CUDA
        case viennacl::backend::CUDA_MEMORY:
          viennacl::linalg::cuda::norm_2_impl(vec, result);
          break;
#endif
        default:
          throw "not implemented";
      }
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
    norm_inf_impl(V1 const & vec,
                  S2 & result)
    {
      switch (viennacl::traits::handle(vec).get_active_handle_id())
      {
        case viennacl::backend::MAIN_MEMORY:
          viennacl::linalg::single_threaded::norm_inf_impl(vec, result);
          break;
#ifdef VIENNACL_WITH_OPENCL          
        case viennacl::backend::OPENCL_MEMORY:
          viennacl::linalg::opencl::norm_inf_impl(vec, result);
          break;
#endif
#ifdef VIENNACL_WITH_CUDA
        case viennacl::backend::CUDA_MEMORY:
          viennacl::linalg::cuda::norm_inf_impl(vec, result);
          break;
#endif
        default:
          throw "not implemented";
      }
    }

    //This function should return a CPU scalar, otherwise statements like 
    // vcl_rhs[index_norm_inf(vcl_rhs)] 
    // are ambiguous
    /** @brief Computes the index of the first entry that is equal to the supremum-norm in modulus.
    *
    * @param vec The vector
    * @return The result. Note that the result must be a CPU scalar
    */
    template <typename V1>
    typename viennacl::enable_if< viennacl::is_any_dense_nonstructured_vector<V1>::value,
                                  std::size_t
                                >::type
    index_norm_inf(V1 const & vec)
    {
      switch (viennacl::traits::handle(vec).get_active_handle_id())
      {
        case viennacl::backend::MAIN_MEMORY:
          return viennacl::linalg::single_threaded::index_norm_inf(vec);
#ifdef VIENNACL_WITH_OPENCL          
        case viennacl::backend::OPENCL_MEMORY:
          return viennacl::linalg::opencl::index_norm_inf(vec);
#endif
#ifdef VIENNACL_WITH_CUDA
        case viennacl::backend::CUDA_MEMORY:
          return viennacl::linalg::cuda::index_norm_inf(vec);
#endif
        default:
          throw "not implemented";
      }
    }
    

    /** @brief Computes a plane rotation of two vectors.
    *
    * Computes (x,y) <- (alpha * x + beta * y, -beta * x + alpha * y)
    *
    * @param vec1   The first vector
    * @param vec2   The second vector
    * @param alpha  The first transformation coefficient (CPU scalar)
    * @param beta   The second transformation coefficient (CPU scalar)
    */
    template <typename V1, typename V2, typename SCALARTYPE>
    typename viennacl::enable_if<    viennacl::is_any_dense_nonstructured_vector<V1>::value
                                  && viennacl::is_any_dense_nonstructured_vector<V2>::value
                                  && viennacl::is_cpu_scalar<SCALARTYPE>::value
                                >::type
    plane_rotation(V1 & vec1,
                   V2 & vec2,
                   SCALARTYPE alpha,
                   SCALARTYPE beta)
    {
      switch (viennacl::traits::handle(vec1).get_active_handle_id())
      {
        case viennacl::backend::MAIN_MEMORY:
          viennacl::linalg::single_threaded::plane_rotation(vec1, vec2, alpha, beta);
          break;
#ifdef VIENNACL_WITH_OPENCL          
        case viennacl::backend::OPENCL_MEMORY:
          viennacl::linalg::opencl::plane_rotation(vec1, vec2, alpha, beta);
          break;
#endif
#ifdef VIENNACL_WITH_CUDA
        case viennacl::backend::CUDA_MEMORY:
          viennacl::linalg::cuda::plane_rotation(vec1, vec2, alpha, beta);
          break;
#endif
        default:
          throw "not implemented";
      }
    }
    
  } //namespace linalg
} //namespace viennacl


#endif
