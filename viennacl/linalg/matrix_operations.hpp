#ifndef VIENNACL_LINALG_MATRIX_OPERATIONS_HPP_
#define VIENNACL_LINALG_MATRIX_OPERATIONS_HPP_

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

/** @file viennacl/linalg/matrix_operations.hpp
    @brief Implementations of dense matrix related operations including matrix-vector products.
*/

#include "viennacl/forwards.h"
#include "viennacl/scalar.hpp"
#include "viennacl/vector.hpp"
#include "viennacl/vector_proxy.hpp"
#include "viennacl/tools/tools.hpp"
#include "viennacl/meta/enable_if.hpp"
#include "viennacl/meta/predicate.hpp"
#include "viennacl/meta/result_of.hpp"
#include "viennacl/traits/size.hpp"
#include "viennacl/traits/start.hpp"
#include "viennacl/traits/handle.hpp"
#include "viennacl/traits/stride.hpp"
#include "viennacl/linalg/host_based/matrix_operations.hpp"

#ifdef VIENNACL_WITH_OPENCL
  #include "viennacl/linalg/opencl/matrix_operations.hpp"
#endif

#ifdef VIENNACL_WITH_CUDA
  #include "viennacl/linalg/cuda/matrix_operations.hpp"
#endif

namespace viennacl
{
  namespace linalg
  {
    
    template <typename M1,
              typename M2, typename ScalarType1>
    typename viennacl::enable_if< viennacl::is_any_dense_nonstructured_matrix<M1>::value
                                  && viennacl::is_any_dense_nonstructured_matrix<M2>::value
                                  && viennacl::is_any_scalar<ScalarType1>::value
                                >::type
    am(M1 & mat1, 
       M2 const & mat2, ScalarType1 const & alpha, std::size_t len_alpha, bool reciprocal_alpha, bool flip_sign_alpha) 
    {
      switch (viennacl::traits::handle(mat1).get_active_handle_id())
      {
        case viennacl::MAIN_MEMORY:
          viennacl::linalg::host_based::am(mat1, mat2, alpha, len_alpha, reciprocal_alpha, flip_sign_alpha);
          break;
#ifdef VIENNACL_WITH_OPENCL          
        case viennacl::OPENCL_MEMORY:
          viennacl::linalg::opencl::am(mat1, mat2, alpha, len_alpha, reciprocal_alpha, flip_sign_alpha);
          break;
#endif
#ifdef VIENNACL_WITH_CUDA
        case viennacl::CUDA_MEMORY:
          viennacl::linalg::cuda::am(mat1, mat2, alpha, len_alpha, reciprocal_alpha, flip_sign_alpha);
          break;
#endif
        default:
          throw "not implemented";
      }
    }
    
    
    template <typename M1,
              typename M2, typename ScalarType1,
              typename M3, typename ScalarType2>
    typename viennacl::enable_if< viennacl::is_any_dense_nonstructured_matrix<M1>::value
                                  && viennacl::is_any_dense_nonstructured_matrix<M2>::value
                                  && viennacl::is_any_dense_nonstructured_matrix<M3>::value
                                  && viennacl::is_any_scalar<ScalarType1>::value
                                  && viennacl::is_any_scalar<ScalarType2>::value
                                >::type
    ambm(M1 & mat1, 
         M2 const & mat2, ScalarType1 const & alpha, std::size_t len_alpha, bool reciprocal_alpha, bool flip_sign_alpha,
         M3 const & mat3, ScalarType2 const & beta, std::size_t len_beta, bool reciprocal_beta, bool flip_sign_beta) 
    {
      switch (viennacl::traits::handle(mat1).get_active_handle_id())
      {
        case viennacl::MAIN_MEMORY:
          viennacl::linalg::host_based::ambm(mat1,
                                             mat2, alpha, len_alpha, reciprocal_alpha, flip_sign_alpha,
                                             mat3,  beta, len_beta,  reciprocal_beta,  flip_sign_beta);
          break;
#ifdef VIENNACL_WITH_OPENCL          
        case viennacl::OPENCL_MEMORY:
          viennacl::linalg::opencl::ambm(mat1,
                                         mat2, alpha, len_alpha, reciprocal_alpha, flip_sign_alpha,
                                         mat3,  beta, len_beta,  reciprocal_beta,  flip_sign_beta);
          break;
#endif
#ifdef VIENNACL_WITH_CUDA
        case viennacl::CUDA_MEMORY:
          viennacl::linalg::cuda::ambm(mat1,
                                       mat2, alpha, len_alpha, reciprocal_alpha, flip_sign_alpha,
                                       mat3,  beta, len_beta,  reciprocal_beta,  flip_sign_beta);
          break;
#endif
        default:
          throw "not implemented";
      }
    }
    
    
    template <typename M1,
              typename M2, typename ScalarType1,
              typename M3, typename ScalarType2>
    typename viennacl::enable_if< viennacl::is_any_dense_nonstructured_matrix<M1>::value
                                  && viennacl::is_any_dense_nonstructured_matrix<M2>::value
                                  && viennacl::is_any_dense_nonstructured_matrix<M3>::value
                                  && viennacl::is_any_scalar<ScalarType1>::value
                                  && viennacl::is_any_scalar<ScalarType2>::value
                                >::type
    ambm_m(M1 & mat1,
           M2 const & mat2, ScalarType1 const & alpha, std::size_t len_alpha, bool reciprocal_alpha, bool flip_sign_alpha,
           M3 const & mat3, ScalarType2 const & beta,  std::size_t len_beta,  bool reciprocal_beta,  bool flip_sign_beta) 
    {
      switch (viennacl::traits::handle(mat1).get_active_handle_id())
      {
        case viennacl::MAIN_MEMORY:
          viennacl::linalg::host_based::ambm_m(mat1,
                                               mat2, alpha, len_alpha, reciprocal_alpha, flip_sign_alpha,
                                               mat3,  beta, len_beta,  reciprocal_beta,  flip_sign_beta);
          break;
#ifdef VIENNACL_WITH_OPENCL          
        case viennacl::OPENCL_MEMORY:
          viennacl::linalg::opencl::ambm_m(mat1,
                                           mat2, alpha, len_alpha, reciprocal_alpha, flip_sign_alpha,
                                           mat3,  beta, len_beta,  reciprocal_beta,  flip_sign_beta);
          break;
#endif
#ifdef VIENNACL_WITH_CUDA
        case viennacl::CUDA_MEMORY:
          viennacl::linalg::cuda::ambm_m(mat1,
                                         mat2, alpha, len_alpha, reciprocal_alpha, flip_sign_alpha,
                                         mat3,  beta, len_beta,  reciprocal_beta,  flip_sign_beta);
          break;
#endif
        default:
          throw "not implemented";
      }
    }


    template <typename M1, typename ScalarType>
    typename viennacl::enable_if<    viennacl::is_any_dense_nonstructured_matrix<M1>::value
                                  && viennacl::is_cpu_scalar<ScalarType>::value
                                >::type    
    matrix_assign(M1 & mat, ScalarType s)
    {
      switch (viennacl::traits::handle(mat).get_active_handle_id())
      {
        case viennacl::MAIN_MEMORY:
          viennacl::linalg::host_based::matrix_assign(mat, s);
          break;
#ifdef VIENNACL_WITH_OPENCL          
        case viennacl::OPENCL_MEMORY:
          viennacl::linalg::opencl::matrix_assign(mat, s);
          break;
#endif
#ifdef VIENNACL_WITH_CUDA
        case viennacl::CUDA_MEMORY:
          viennacl::linalg::cuda::matrix_assign(mat, s);
          break;
#endif
        default:
          throw "not implemented";
      }
    }

    
    template <typename M1, typename ScalarType>
    typename viennacl::enable_if<    viennacl::is_any_dense_nonstructured_matrix<M1>::value
                                  && viennacl::is_cpu_scalar<ScalarType>::value
                                >::type    
    matrix_diagonal_assign(M1 & mat, ScalarType s)
    {
      switch (viennacl::traits::handle(mat).get_active_handle_id())
      {
        case viennacl::MAIN_MEMORY:
          viennacl::linalg::host_based::matrix_diagonal_assign(mat, s);
          break;
#ifdef VIENNACL_WITH_OPENCL          
        case viennacl::OPENCL_MEMORY:
          viennacl::linalg::opencl::matrix_diagonal_assign(mat, s);
          break;
#endif
#ifdef VIENNACL_WITH_CUDA
        case viennacl::CUDA_MEMORY:
          viennacl::linalg::cuda::matrix_diagonal_assign(mat, s);
          break;
#endif
        default:
          throw "not implemented";
      }
    }
    
    
    //
    /////////////////////////   matrix-vector products /////////////////////////////////
    //



    // A * x

    /** @brief Carries out matrix-vector multiplication
    *
    * Implementation of the convenience expression result = prod(mat, vec);
    *
    * @param mat    The matrix
    * @param vec    The vector
    * @param result The result vector
    */
    template <typename MatrixType, typename VectorType1, typename VectorType2>
    typename viennacl::enable_if<   viennacl::is_any_dense_nonstructured_matrix<MatrixType>::value 
                                  && viennacl::is_any_dense_nonstructured_vector<VectorType1>::value 
                                  && viennacl::is_any_dense_nonstructured_vector<VectorType2>::value >::type
    prod_impl(const MatrixType & mat, 
              const VectorType1 & vec, 
                    VectorType2 & result)
    {
      assert( (viennacl::traits::size1(mat) == viennacl::traits::size(result)) && bool("Size check failed at v1 = prod(A, v2): size1(A) != size(v1)"));
      assert( (viennacl::traits::size2(mat) == viennacl::traits::size(vec))    && bool("Size check failed at v1 = prod(A, v2): size2(A) != size(v2)"));
      
      switch (viennacl::traits::handle(mat).get_active_handle_id())
      {
        case viennacl::MAIN_MEMORY:
          viennacl::linalg::host_based::prod_impl(mat, vec, result);
          break;
#ifdef VIENNACL_WITH_OPENCL          
        case viennacl::OPENCL_MEMORY:
          viennacl::linalg::opencl::prod_impl(mat, vec, result);
          break;
#endif
#ifdef VIENNACL_WITH_CUDA
        case viennacl::CUDA_MEMORY:
          viennacl::linalg::cuda::prod_impl(mat, vec, result);
          break;
#endif
        default:
          throw "not implemented";
      }
    }


    // trans(A) * x
    /** @brief Returns a proxy class that represents matrix-vector multiplication with a transposed matrix
    *
    * This is used for the convenience expression result = trans(mat) * vec;
    *
    * @param proxy  The transposed matrix proxy
    * @param vec    The vector
    */
    template<class SCALARTYPE, typename F, unsigned int ALIGNMENT, unsigned int VECTOR_ALIGNMENT>
    viennacl::vector_expression<const viennacl::matrix_expression< const matrix<SCALARTYPE, F, ALIGNMENT>,
                                                                   const matrix<SCALARTYPE, F, ALIGNMENT>,
                                                                   op_trans>,
                                const viennacl::vector<SCALARTYPE, VECTOR_ALIGNMENT>, 
                                op_prod > prod_impl(const viennacl::matrix_expression< const matrix<SCALARTYPE, F, ALIGNMENT>,
                                                                                       const matrix<SCALARTYPE, F, ALIGNMENT>,
                                                                                       op_trans> & proxy, 
                                                    const viennacl::vector<SCALARTYPE, VECTOR_ALIGNMENT> & vec)
    {
      return viennacl::vector_expression<const viennacl::matrix_expression< const matrix<SCALARTYPE, F, ALIGNMENT>,
                                                                            const matrix<SCALARTYPE, F, ALIGNMENT>,
                                                                            op_trans>,
                                         const viennacl::vector<SCALARTYPE, VECTOR_ALIGNMENT>, 
                                         op_prod >(proxy, vec);
    }

    
    /** @brief Carries out matrix-vector multiplication with a transposed matrix
    *
    * Implementation of the convenience expression result = trans(mat) * vec;
    *
    * @param mat_trans  The transposed matrix proxy
    * @param vec        The vector
    * @param result     The result vector
    */
    template <typename M1, typename V1, typename V2>
    typename viennacl::enable_if<    viennacl::is_any_dense_nonstructured_matrix<M1>::value
                                  && viennacl::is_any_dense_nonstructured_vector<V1>::value
                                  && viennacl::is_any_dense_nonstructured_vector<V2>::value
                                >::type
    prod_impl(const viennacl::matrix_expression< const M1, const M1, op_trans> & mat_trans,
              const V1 & vec, 
                    V2 & result)
    {
      assert( (viennacl::traits::size1(mat_trans.lhs()) == viennacl::traits::size(vec))    && bool("Size check failed at v1 = trans(A) * v2: size1(A) != size(v2)"));
      assert( (viennacl::traits::size2(mat_trans.lhs()) == viennacl::traits::size(result)) && bool("Size check failed at v1 = trans(A) * v2: size2(A) != size(v1)"));
      
      switch (viennacl::traits::handle(mat_trans.lhs()).get_active_handle_id())
      {
        case viennacl::MAIN_MEMORY:
          viennacl::linalg::host_based::prod_impl(mat_trans, vec, result);
          break;
#ifdef VIENNACL_WITH_OPENCL          
        case viennacl::OPENCL_MEMORY:
          viennacl::linalg::opencl::prod_impl(mat_trans, vec, result);
          break;
#endif
#ifdef VIENNACL_WITH_CUDA
        case viennacl::CUDA_MEMORY:
          viennacl::linalg::cuda::prod_impl(mat_trans, vec, result);
          break;
#endif
        default:
          throw "not implemented";
      }
    }


    //
    /////////////////////////   matrix-matrix products /////////////////////////////////
    //
    
    /** @brief Carries out matrix-matrix multiplication
    *
    * Implementation of C = prod(A, B);
    *
    */
    template <typename T1, typename T2, typename T3, typename ScalarType >
    typename viennacl::enable_if<    viennacl::is_any_dense_nonstructured_matrix<T1>::value
                                  && viennacl::is_any_dense_nonstructured_matrix<T2>::value
                                  && viennacl::is_any_dense_nonstructured_matrix<T3>::value
                                >::type
    prod_impl(const T1 & A, 
              const T2 & B, 
                    T3 & C,
              ScalarType alpha,
              ScalarType beta)
    {
      assert( (viennacl::traits::size1(A) == viennacl::traits::size1(C)) && bool("Size check failed at C = prod(A, B): size1(A) != size1(C)"));
      assert( (viennacl::traits::size2(A) == viennacl::traits::size1(B)) && bool("Size check failed at C = prod(A, B): size2(A) != size1(B)"));
      assert( (viennacl::traits::size2(B) == viennacl::traits::size2(C)) && bool("Size check failed at C = prod(A, B): size2(B) != size2(C)"));

      
      switch (viennacl::traits::handle(A).get_active_handle_id())
      {
        case viennacl::MAIN_MEMORY:
          viennacl::linalg::host_based::prod_impl(A, B, C, alpha, beta);
          break;
#ifdef VIENNACL_WITH_OPENCL          
        case viennacl::OPENCL_MEMORY:
          viennacl::linalg::opencl::prod_impl(A, B, C, alpha, beta);
          break;
#endif
#ifdef VIENNACL_WITH_CUDA
        case viennacl::CUDA_MEMORY:
          viennacl::linalg::cuda::prod_impl(A, B, C, alpha, beta);
          break;
#endif
        default:
          throw "not implemented";
      }
    }



    /** @brief Carries out matrix-matrix multiplication
    *
    * Implementation of C = prod(trans(A), B);
    *
    */
    template <typename T1, typename T2, typename T3, typename ScalarType >
    typename viennacl::enable_if<    viennacl::is_any_dense_nonstructured_matrix<T1>::value
                                  && viennacl::is_any_dense_nonstructured_matrix<T2>::value
                                  && viennacl::is_any_dense_nonstructured_matrix<T3>::value
                                >::type
    prod_impl(const viennacl::matrix_expression< const T1,
                                                 const T1,
                                                 op_trans> & A, 
              const T2 & B, 
                    T3 & C,
              ScalarType alpha,
              ScalarType beta)
    {
      assert(viennacl::traits::size2(A.lhs()) == viennacl::traits::size1(C) && bool("Size check failed at C = prod(trans(A), B): size2(A) != size1(C)"));
      assert(viennacl::traits::size1(A.lhs()) == viennacl::traits::size1(B) && bool("Size check failed at C = prod(trans(A), B): size1(A) != size1(B)"));
      assert(viennacl::traits::size2(B)       == viennacl::traits::size2(C) && bool("Size check failed at C = prod(trans(A), B): size2(B) != size2(C)"));
      
      switch (viennacl::traits::handle(A.lhs()).get_active_handle_id())
      {
        case viennacl::MAIN_MEMORY:
          viennacl::linalg::host_based::prod_impl(A, B, C, alpha, beta);
          break;
#ifdef VIENNACL_WITH_OPENCL          
        case viennacl::OPENCL_MEMORY:
          viennacl::linalg::opencl::prod_impl(A, B, C, alpha, beta);
          break;
#endif
#ifdef VIENNACL_WITH_CUDA
        case viennacl::CUDA_MEMORY:
          viennacl::linalg::cuda::prod_impl(A, B, C, alpha, beta);
          break;
#endif
        default:
          throw "not implemented";
      }
    }




    /** @brief Carries out matrix-matrix multiplication
    *
    * Implementation of C = prod(A, trans(B));
    *
    */
    template <typename T1, typename T2, typename T3, typename ScalarType >
    typename viennacl::enable_if<    viennacl::is_any_dense_nonstructured_matrix<T1>::value
                                  && viennacl::is_any_dense_nonstructured_matrix<T2>::value
                                  && viennacl::is_any_dense_nonstructured_matrix<T3>::value
                                >::type
    prod_impl(const T1 & A, 
              const viennacl::matrix_expression< const T2, const T2, op_trans> & B,
                    T3 & C,
              ScalarType alpha,
              ScalarType beta)
    {
      assert(viennacl::traits::size1(A)       == viennacl::traits::size1(C)       && bool("Size check failed at C = prod(A, trans(B)): size1(A) != size1(C)"));
      assert(viennacl::traits::size2(A)       == viennacl::traits::size2(B.lhs()) && bool("Size check failed at C = prod(A, trans(B)): size2(A) != size2(B)"));
      assert(viennacl::traits::size1(B.lhs()) == viennacl::traits::size2(C)       && bool("Size check failed at C = prod(A, trans(B)): size1(B) != size2(C)"));
      
      switch (viennacl::traits::handle(A).get_active_handle_id())
      {
        case viennacl::MAIN_MEMORY:
          viennacl::linalg::host_based::prod_impl(A, B, C, alpha, beta);
          break;
#ifdef VIENNACL_WITH_OPENCL          
        case viennacl::OPENCL_MEMORY:
          viennacl::linalg::opencl::prod_impl(A, B, C, alpha, beta);
          break;
#endif
#ifdef VIENNACL_WITH_CUDA
        case viennacl::CUDA_MEMORY:
          viennacl::linalg::cuda::prod_impl(A, B, C, alpha, beta);
          break;
#endif
        default:
          throw "not implemented";
      }
    }



    /** @brief Carries out matrix-matrix multiplication
    *
    * Implementation of C = prod(trans(A), trans(B));
    *
    */
    template <typename T1, typename T2, typename T3, typename ScalarType >
    typename viennacl::enable_if<    viennacl::is_any_dense_nonstructured_matrix<T1>::value
                                  && viennacl::is_any_dense_nonstructured_matrix<T2>::value
                                  && viennacl::is_any_dense_nonstructured_matrix<T3>::value
                                >::type
    prod_impl(const viennacl::matrix_expression< const T1, const T1, op_trans> & A,
              const viennacl::matrix_expression< const T2, const T2, op_trans> & B,
              T3 & C,
              ScalarType alpha,
              ScalarType beta)
    {
      assert(viennacl::traits::size2(A.lhs()) == viennacl::traits::size1(C)       && bool("Size check failed at C = prod(trans(A), trans(B)): size2(A) != size1(C)"));
      assert(viennacl::traits::size1(A.lhs()) == viennacl::traits::size2(B.lhs()) && bool("Size check failed at C = prod(trans(A), trans(B)): size1(A) != size2(B)"));
      assert(viennacl::traits::size1(B.lhs()) == viennacl::traits::size2(C)       && bool("Size check failed at C = prod(trans(A), trans(B)): size1(B) != size2(C)"));
      
      switch (viennacl::traits::handle(A.lhs()).get_active_handle_id())
      {
        case viennacl::MAIN_MEMORY:
          viennacl::linalg::host_based::prod_impl(A, B, C, alpha, beta);
          break;
#ifdef VIENNACL_WITH_OPENCL          
        case viennacl::OPENCL_MEMORY:
          viennacl::linalg::opencl::prod_impl(A, B, C, alpha, beta);
          break;
#endif
#ifdef VIENNACL_WITH_CUDA
        case viennacl::CUDA_MEMORY:
          viennacl::linalg::cuda::prod_impl(A, B, C, alpha, beta);
          break;
#endif
        default:
          throw "not implemented";
      }
    }




    //
    /////////////////////////   miscellaneous operations /////////////////////////////////
    //


    /** @brief Returns a proxy class for the operation mat += vec1 * vec2^T, i.e. a rank 1 update
    *
    * @param vec1    The first vector
    * @param vec2    The second vector
    */
    template <typename V1, typename V2>
    typename viennacl::enable_if<    viennacl::is_any_dense_nonstructured_vector<V1>::value
                                  && viennacl::is_any_dense_nonstructured_vector<V2>::value,
                                  viennacl::matrix_expression<const V1, const V2, op_prod>
                                >::type
    outer_prod(const V1 & vec1, const V2 & vec2)
    {
      return viennacl::matrix_expression< const V1, const V2, op_prod>(vec1, vec2);
    }
    
    
    /** @brief The implementation of the operation mat += alpha * vec1 * vec2^T, i.e. a scaled rank 1 update
    *
    * Implementation of the convenience expression result += alpha * outer_prod(vec1, vec2);
    *
    * @param mat1             The matrix to be updated
    * @param alpha            The scaling factor (either a viennacl::scalar<>, float, or double)
    * @param len_alpha        Length of the buffer for an eventual final reduction step (currently always '1')
    * @param reciprocal_alpha Use 1/alpha instead of alpha
    * @param flip_sign_alpha  Use -alpha instead of alpha
    * @param vec1             The first vector
    * @param vec2             The second vector
    */
    template <typename M1, typename S1, typename V1, typename V2>
    typename viennacl::enable_if<    viennacl::is_any_dense_nonstructured_matrix<M1>::value
                                  && viennacl::is_any_scalar<S1>::value
                                  && viennacl::is_any_dense_nonstructured_vector<V1>::value
                                  && viennacl::is_any_dense_nonstructured_vector<V2>::value
                                >::type
    scaled_rank_1_update(M1 & mat1,
                         S1 const & alpha, std::size_t len_alpha, bool reciprocal_alpha, bool flip_sign_alpha,
                         const V1 & vec1, 
                         const V2 & vec2)
    {
      switch (viennacl::traits::handle(mat1).get_active_handle_id())
      {
        case viennacl::MAIN_MEMORY:
          viennacl::linalg::host_based::scaled_rank_1_update(mat1,
                                                             alpha, len_alpha, reciprocal_alpha, flip_sign_alpha,
                                                             vec1, vec2);
          break;
#ifdef VIENNACL_WITH_OPENCL          
        case viennacl::OPENCL_MEMORY:
          viennacl::linalg::opencl::scaled_rank_1_update(mat1,
                                                         alpha, len_alpha, reciprocal_alpha, flip_sign_alpha,
                                                         vec1, vec2);
          break;
#endif
#ifdef VIENNACL_WITH_CUDA
        case viennacl::CUDA_MEMORY:
          viennacl::linalg::cuda::scaled_rank_1_update(mat1,
                                                       alpha, len_alpha, reciprocal_alpha, flip_sign_alpha,
                                                       vec1, vec2);
          break;
#endif
        default:
          throw "not implemented";
      }
    }
    
  } //namespace linalg




  //
  /////////////////////////  Operator overloads /////////////////////////////////
  //





  //v = A * x
//  /** @brief Implementation of the operation v1 = A * v2, where A is a matrix
//  *
//  * @param proxy  An expression template proxy class.
//  */
//   template <typename SCALARTYPE, unsigned int ALIGNMENT>
//   template <typename M1, typename V1>
//   typename viennacl::enable_if<    viennacl::is_any_dense_nonstructured_matrix<M1>::value
//                                 && viennacl::is_any_dense_nonstructured_vector<V1>::value,
//                                 viennacl::vector<SCALARTYPE, ALIGNMENT> & 
//                                >::type
//   viennacl::vector<SCALARTYPE, ALIGNMENT>::operator=(const viennacl::vector_expression< const M1,
//                                                                                         const V1,
//                                                                                         viennacl::op_prod> & proxy) 
//   {
//     assert(viennacl::traits::size1(proxy.lhs()) == size() && bool("Size check failed for v1 = A * v2: size1(A) != size(v1)"));
//     
//     // check for the special case x = A * x
//     if (viennacl::traits::handle(proxy.rhs()) == viennacl::traits::handle(*this))
//     {
//       viennacl::vector<SCALARTYPE, ALIGNMENT> result(viennacl::traits::size1(proxy.lhs()));
//       viennacl::linalg::prod_impl(proxy.lhs(), proxy.rhs(), result);
//       *this = result;
//     }
//     else
//     {
//       viennacl::linalg::prod_impl(proxy.lhs(), proxy.rhs(), *this);
//     }
//     return *this;
//   }

  //v += A * x
  /** @brief Implementation of the operation v1 += A * v2, where A is a matrix
  *
  * @param v1     The result vector v1 where A * v2 is added to
  * @param proxy  An expression template proxy class.
  */
  template <typename V1, typename M2, typename V2>
  typename viennacl::enable_if<    viennacl::is_any_dense_nonstructured_vector<V1>::value
                                && viennacl::is_any_dense_nonstructured_matrix<M2>::value
                                && viennacl::is_any_dense_nonstructured_vector<V2>::value,
                                V1 &
                               >::type
  operator+=(V1 & v1,
             const viennacl::vector_expression< const M2, const V2, viennacl::op_prod> & proxy) 
  {
    typedef typename viennacl::result_of::cpu_value_type<V1>::type   cpu_value_type;
    
    assert(viennacl::traits::size1(proxy.lhs()) == v1.size() && bool("Size check failed for v1 += A * v2: size1(A) != size(v1)"));
    
    vector<cpu_value_type> result(viennacl::traits::size1(proxy.lhs()));
    viennacl::linalg::prod_impl(proxy.lhs(), proxy.rhs(), result);
    v1 += result;
    return v1;
  }

  /** @brief Implementation of the operation v1 -= A * v2, where A is a matrix
  *
  * @param v1     The result vector v1 where A * v2 is subtracted from
  * @param proxy  An expression template proxy class.
  */
  template <typename V1, typename M2, typename V2>
  typename viennacl::enable_if<    viennacl::is_any_dense_nonstructured_vector<V1>::value
                                && viennacl::is_any_dense_nonstructured_matrix<M2>::value
                                && viennacl::is_any_dense_nonstructured_vector<V2>::value,
                                V1 &
                               >::type
  operator-=(V1 & v1,
             const viennacl::vector_expression< const M2, const V2, viennacl::op_prod> & proxy) 
  {
    typedef typename viennacl::result_of::cpu_value_type<V1>::type   cpu_value_type;
    
    assert(viennacl::traits::size1(proxy.lhs()) == v1.size() && bool("Size check failed for v1 -= A * v2: size1(A) != size(v1)"));
    
    vector<cpu_value_type> result(viennacl::traits::size1(proxy.lhs()));
    viennacl::linalg::prod_impl(proxy.lhs(), proxy.rhs(), result);
    v1 -= result;
    return v1;
  }
  
  
  
  
  
  //free functions:
  /** @brief Implementation of the operation 'result = v1 + A * v2', where A is a matrix
  *
  * @param v1     The addend vector.
  * @param proxy  An expression template proxy class.
  */
  template <typename M1, typename V1, typename V2>
  typename viennacl::enable_if<    viennacl::is_any_dense_nonstructured_matrix<M1>::value
                                && viennacl::is_any_dense_nonstructured_vector<V1>::value
                                && viennacl::is_any_dense_nonstructured_vector<V2>::value,
                                viennacl::vector<typename viennacl::result_of::cpu_value_type<V1>::type,
                                                 viennacl::result_of::alignment<V1>::value> 
                              >::type
  operator+(const V1 & v1,
            const vector_expression< const M1, const V2, op_prod> & proxy) 
  {
    assert(viennacl::traits::size1(proxy.lhs()) == viennacl::traits::size(v1) && bool("Size check failed for v1 + A * v2: size1(A) != size(v1)"));
    
    vector<typename viennacl::result_of::cpu_value_type<V1>::type,
           viennacl::result_of::alignment<V1>::value> result(viennacl::traits::size(v1));
    viennacl::linalg::prod_impl(proxy.lhs(), proxy.rhs(), result);
    result += v1;
    return result;
  }

  /** @brief Implementation of the operation 'result = v1 - A * v2', where A is a matrix
  *
  * @param v1     The addend vector.
  * @param proxy  An expression template proxy class.
  */
  template <typename M1, typename V1, typename V2>
  typename viennacl::enable_if<    viennacl::is_any_dense_nonstructured_matrix<M1>::value
                                && viennacl::is_any_dense_nonstructured_vector<V1>::value
                                && viennacl::is_any_dense_nonstructured_vector<V2>::value,
                                viennacl::vector<typename viennacl::result_of::cpu_value_type<V1>::type,
                                                 viennacl::result_of::alignment<V1>::value> 
                              >::type
  operator-(const V1 & v1,
            const vector_expression< const M1, const V2, op_prod> & proxy) 
  {
    assert(viennacl::traits::size1(proxy.lhs()) == viennacl::traits::size(v1) && bool("Size check failed for v1 - A * v2: size1(A) != size(v1)"));
    
    vector<typename viennacl::result_of::cpu_value_type<V1>::type,
           viennacl::result_of::alignment<V1>::value> result(viennacl::traits::size(v1));
    viennacl::linalg::prod_impl(proxy.lhs(), proxy.rhs(), result);
    result = v1 - result;
    return result;
  }


  ////////// transposed_matrix_proxy


  //v = trans(A) * x
//   /** @brief Implementation of the operation v1 = A * v2, where A is a matrix
//   *
//   * @param proxy  An expression template proxy class.
//   */
//   template <typename SCALARTYPE, unsigned int ALIGNMENT>
//   template <typename M1, typename V1>
//   typename viennacl::enable_if<    viennacl::is_any_dense_nonstructured_matrix<M1>::value
//                                 && viennacl::is_any_dense_nonstructured_vector<V1>::value,
//                                 viennacl::vector<SCALARTYPE, ALIGNMENT> &
//                               >::type
//   viennacl::vector<SCALARTYPE, ALIGNMENT>::operator=(const viennacl::vector_expression< const matrix_expression< const M1, const M1, op_trans>,
//                                                                                         const V1,
//                                                                                         viennacl::op_prod> & proxy) 
//   {
//     assert(viennacl::traits::size1(proxy.lhs()) == size() && bool("Size check failed in v1 = trans(A) * v2: size2(A) != size(v1)"));
// 
//     // check for the special case x = trans(A) * x
//     if (viennacl::traits::handle(proxy.rhs()) == viennacl::traits::handle(*this))
//     {
//       viennacl::vector<SCALARTYPE, ALIGNMENT> result(viennacl::traits::size1(proxy.lhs()));
//       viennacl::linalg::prod_impl(proxy.lhs(), proxy.rhs(), result);
//       *this = result;
//     }
//     else
//     {
//       viennacl::linalg::prod_impl(proxy.lhs(), proxy.rhs(), *this);
//     }
//     return *this;
//   }

  //v += A * x
  /** @brief Implementation of the operation v1 += A * v2, where A is a matrix
  *
  * @param v1     The addend vector where the result is written to.
  * @param proxy  An expression template proxy class.
  */
  template <typename V1, typename M2, typename V2>
  typename viennacl::enable_if<    viennacl::is_any_dense_nonstructured_matrix<M2>::value
                                && viennacl::is_any_dense_nonstructured_vector<V2>::value,
                                V1 &
                              >::type
  operator+=(V1 & v1, 
             const vector_expression< const matrix_expression<const M2, const M2, op_trans>,
                                                              const V2,
                                                              op_prod> & proxy) 
  {
    typedef typename viennacl::result_of::cpu_value_type<V1>::type   cpu_value_type;
    
    assert(viennacl::traits::size2(proxy.lhs()) == v1.size() && bool("Size check failed in v1 += trans(A) * v2: size2(A) != size(v1)"));
    
    vector<cpu_value_type> result(viennacl::traits::size2(proxy.lhs()));
    viennacl::linalg::prod_impl(proxy.lhs(), proxy.rhs(), result);
    v1 += result;
    return v1;
  }

  /** @brief Implementation of the operation v1 -= A * v2, where A is a matrix
  *
  * @param v1     The addend vector where the result is written to.
  * @param proxy  An expression template proxy class.
  */
  template <typename V1, typename M2, typename V2>
  typename viennacl::enable_if<    viennacl::is_any_dense_nonstructured_matrix<M2>::value
                                && viennacl::is_any_dense_nonstructured_vector<V2>::value,
                                V1 &
                              >::type
  operator-=(V1 & v1, 
             const vector_expression< const matrix_expression<const M2, const M2, op_trans>,
                                                              const V2,
                                                              op_prod> & proxy) 
  {
    typedef typename viennacl::result_of::cpu_value_type<V1>::type   cpu_value_type;
    
    assert(viennacl::traits::size2(proxy.lhs()) == v1.size() && bool("Size check failed in v1 += trans(A) * v2: size2(A) != size(v1)"));
    
    vector<cpu_value_type> result(viennacl::traits::size2(proxy.lhs()));
    viennacl::linalg::prod_impl(proxy.lhs(), proxy.rhs(), result);
    v1 -= result;
    return v1;
  }
  
  
  //free functions:
  /** @brief Implementation of the operation 'result = v1 + A * v2', where A is a matrix
  *
  * @param v1     The addend vector.
  * @param proxy  An expression template proxy class.
  */
  template <typename M1, typename V1, typename V2>
  typename viennacl::enable_if<    viennacl::is_any_dense_nonstructured_matrix<M1>::value
                                && viennacl::is_any_dense_nonstructured_vector<V1>::value
                                && viennacl::is_any_dense_nonstructured_vector<V2>::value,
                                viennacl::vector<typename viennacl::result_of::cpu_value_type<V1>::type,
                                                 viennacl::result_of::alignment<V1>::value> 
                              >::type
  operator+(const V1 & v1,
            const vector_expression< const matrix_expression<const M1, const M1, op_trans>,
                                    const V2,
                                    op_prod> & proxy) 
  {
    assert(viennacl::traits::size2(proxy.lhs()) == viennacl::traits::size(v1) && bool("Size check failed in v1 + trans(A) * v2: size2(A) != size(v1)"));
    
    vector<typename viennacl::result_of::cpu_value_type<V1>::type,
           viennacl::result_of::alignment<V1>::value> result(viennacl::traits::size(v1));
    viennacl::linalg::prod_impl(proxy.lhs(), proxy.rhs(), result);
    result += v1;
    return result;
  }

  /** @brief Implementation of the operation 'result = v1 - A * v2', where A is a matrix
  *
  * @param v1     The addend vector.
  * @param proxy  An expression template proxy class.
  */
  template <typename M1, typename V1, typename V2>
  typename viennacl::enable_if<    viennacl::is_any_dense_nonstructured_matrix<M1>::value
                                && viennacl::is_any_dense_nonstructured_vector<V1>::value
                                && viennacl::is_any_dense_nonstructured_vector<V2>::value,
                                viennacl::vector<typename viennacl::result_of::cpu_value_type<V1>::type,
                                                 viennacl::result_of::alignment<V1>::value> 
                              >::type
  operator-(const V1 & v1,
            const vector_expression< const matrix_expression<const M1, const M1, op_trans>,
                                     const V2,
                                     op_prod> & proxy) 
  {
    assert(viennacl::traits::size2(proxy.lhs()) == viennacl::traits::size(v1) && bool("Size check failed in v1 - trans(A) * v2: size2(A) != size(v1)"));
    
    vector<typename viennacl::result_of::cpu_value_type<V1>::type,
           viennacl::result_of::alignment<V1>::value> result(viennacl::traits::size(v1));
    viennacl::linalg::prod_impl(proxy.lhs(), proxy.rhs(), result);
    result = v1 - result;
    return result;
  }


} //namespace viennacl


#endif
