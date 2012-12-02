#ifndef VIENNACL_LINALG_HOST_BASED_MATRIX_OPERATIONS_HPP_
#define VIENNACL_LINALG_HOST_BASED_MATRIX_OPERATIONS_HPP_

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

/** @file  viennacl/linalg/host_based/matrix_operations.hpp
    @brief Implementations of dense matrix related operations, including matrix-vector products, using a plain single-threaded or OpenMP-enabled execution on CPU.
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
#include "viennacl/linalg/host_based/common.hpp"

namespace viennacl
{
  namespace linalg
  {
    namespace host_based
    {
      
      //
      // Introductory note: By convention, all dimensions are already checked in the dispatcher frontend. No need to double-check again in here!
      //
      
      template <typename M1,
                typename M2, typename ScalarType1>
      typename viennacl::enable_if< viennacl::is_any_dense_nonstructured_matrix<M1>::value
                                    && viennacl::is_any_dense_nonstructured_matrix<M2>::value
                                    && viennacl::is_any_scalar<ScalarType1>::value
                                  >::type
      am(M1 & mat1, 
         M2 const & mat2, ScalarType1 const & alpha, std::size_t /*len_alpha*/, bool reciprocal_alpha, bool flip_sign_alpha) 
      {
        typedef typename viennacl::result_of::cpu_value_type<M1>::type        value_type;
        
        value_type       * data_A = detail::extract_raw_pointer<value_type>(mat1);
        value_type const * data_B = detail::extract_raw_pointer<value_type>(mat2);
        
        value_type data_alpha = alpha;
        if (flip_sign_alpha)
          data_alpha = -data_alpha;
        if (reciprocal_alpha)
          data_alpha = static_cast<value_type>(1) / data_alpha;
        
        std::size_t A_start1 = viennacl::traits::start1(mat1);
        std::size_t A_start2 = viennacl::traits::start2(mat1);
        std::size_t A_inc1   = viennacl::traits::stride1(mat1);
        std::size_t A_inc2   = viennacl::traits::stride2(mat1);
        std::size_t A_size1  = viennacl::traits::size1(mat1);
        std::size_t A_size2  = viennacl::traits::size2(mat1);
        std::size_t A_internal_size1  = viennacl::traits::internal_size1(mat1);
        std::size_t A_internal_size2  = viennacl::traits::internal_size2(mat1);
        
        std::size_t B_start1 = viennacl::traits::start1(mat2);
        std::size_t B_start2 = viennacl::traits::start2(mat2);
        std::size_t B_inc1   = viennacl::traits::stride1(mat2);
        std::size_t B_inc2   = viennacl::traits::stride2(mat2);
        std::size_t B_internal_size1  = viennacl::traits::internal_size1(mat2);
        std::size_t B_internal_size2  = viennacl::traits::internal_size2(mat2);
        
        detail::matrix_array_wrapper<value_type,       typename M1::orientation_category, false> wrapper_A(data_A, A_start1, A_start2, A_inc1, A_inc2, A_internal_size1, A_internal_size2);
        detail::matrix_array_wrapper<value_type const, typename M2::orientation_category, false> wrapper_B(data_B, B_start1, B_start2, B_inc1, B_inc2, B_internal_size1, B_internal_size2);
        //typedef typename detail::majority_struct_for_orientation<typename M1::orientation_category>::type index_generator_A;
        //typedef typename detail::majority_struct_for_orientation<typename M2::orientation_category>::type index_generator_B;
        
        if (detail::is_row_major(typename M1::orientation_category()))
        {
#ifdef VIENNACL_WITH_OPENMP
          #pragma omp parallel for
#endif
          for (std::size_t row = 0; row < A_size1; ++row)
            for (std::size_t col = 0; col < A_size2; ++col)
              wrapper_A(row, col) = wrapper_B(row, col) * data_alpha;
              //data_A[index_generator_A::mem_index(row * A_inc1 + A_start1, col * A_inc2 + A_start2, A_internal_size1, A_internal_size2)] 
              // = data_B[index_generator_B::mem_index(row * B_inc1 + B_start1, col * B_inc2 + B_start2, B_internal_size1, B_internal_size2)] * alpha;
        }
        else
        {
#ifdef VIENNACL_WITH_OPENMP
          #pragma omp parallel for
#endif
          for (std::size_t col = 0; col < A_size2; ++col)
            for (std::size_t row = 0; row < A_size1; ++row)
              wrapper_A(row, col) = wrapper_B(row, col) * data_alpha;
              //data_A[index_generator_A::mem_index(row * A_inc1 + A_start1, col * A_inc2 + A_start2, A_internal_size1, A_internal_size2)] 
              // = data_B[index_generator_B::mem_index(row * B_inc1 + B_start1, col * B_inc2 + B_start2, B_internal_size1, B_internal_size2)] * alpha;
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
           M2 const & mat2, ScalarType1 const & alpha, std::size_t /*len_alpha*/, bool reciprocal_alpha, bool flip_sign_alpha,
           M3 const & mat3, ScalarType2 const & beta, std::size_t /*len_beta*/, bool reciprocal_beta, bool flip_sign_beta) 
      {
        typedef typename viennacl::result_of::cpu_value_type<M1>::type        value_type;
       
        value_type       * data_A = detail::extract_raw_pointer<value_type>(mat1);
        value_type const * data_B = detail::extract_raw_pointer<value_type>(mat2);
        value_type const * data_C = detail::extract_raw_pointer<value_type>(mat3);
        
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
        
        std::size_t A_start1 = viennacl::traits::start1(mat1);
        std::size_t A_start2 = viennacl::traits::start2(mat1);
        std::size_t A_inc1   = viennacl::traits::stride1(mat1);
        std::size_t A_inc2   = viennacl::traits::stride2(mat1);
        std::size_t A_size1  = viennacl::traits::size1(mat1);
        std::size_t A_size2  = viennacl::traits::size2(mat1);
        std::size_t A_internal_size1  = viennacl::traits::internal_size1(mat1);
        std::size_t A_internal_size2  = viennacl::traits::internal_size2(mat1);
        
        std::size_t B_start1 = viennacl::traits::start1(mat2);
        std::size_t B_start2 = viennacl::traits::start2(mat2);
        std::size_t B_inc1   = viennacl::traits::stride1(mat2);
        std::size_t B_inc2   = viennacl::traits::stride2(mat2);
        std::size_t B_internal_size1  = viennacl::traits::internal_size1(mat2);
        std::size_t B_internal_size2  = viennacl::traits::internal_size2(mat2);
        
        std::size_t C_start1 = viennacl::traits::start1(mat3);
        std::size_t C_start2 = viennacl::traits::start2(mat3);
        std::size_t C_inc1   = viennacl::traits::stride1(mat3);
        std::size_t C_inc2   = viennacl::traits::stride2(mat3);
        std::size_t C_internal_size1  = viennacl::traits::internal_size1(mat3);
        std::size_t C_internal_size2  = viennacl::traits::internal_size2(mat3);
        
        detail::matrix_array_wrapper<value_type,       typename M1::orientation_category, false> wrapper_A(data_A, A_start1, A_start2, A_inc1, A_inc2, A_internal_size1, A_internal_size2);
        detail::matrix_array_wrapper<value_type const, typename M2::orientation_category, false> wrapper_B(data_B, B_start1, B_start2, B_inc1, B_inc2, B_internal_size1, B_internal_size2);
        detail::matrix_array_wrapper<value_type const, typename M3::orientation_category, false> wrapper_C(data_C, C_start1, C_start2, C_inc1, C_inc2, C_internal_size1, C_internal_size2);
        
        if (detail::is_row_major(typename M1::orientation_category()))
        {
#ifdef VIENNACL_WITH_OPENMP
          #pragma omp parallel for
#endif
          for (std::size_t row = 0; row < A_size1; ++row)
            for (std::size_t col = 0; col < A_size2; ++col)
              wrapper_A(row, col) = wrapper_B(row, col) * data_alpha + wrapper_C(row, col) * data_beta;
              //data_A[index_generator_A::mem_index(row * A_inc1 + A_start1, col * A_inc2 + A_start2, A_internal_size1, A_internal_size2)] 
              // =   data_B[index_generator_B::mem_index(row * B_inc1 + B_start1, col * B_inc2 + B_start2, B_internal_size1, B_internal_size2)] * alpha
              //   + data_C[index_generator_C::mem_index(row * C_inc1 + C_start1, col * C_inc2 + C_start2, C_internal_size1, C_internal_size2)] * beta;
        }
        else
        {
#ifdef VIENNACL_WITH_OPENMP
          #pragma omp parallel for
#endif
          for (std::size_t col = 0; col < A_size2; ++col)
            for (std::size_t row = 0; row < A_size1; ++row)
              wrapper_A(row, col) = wrapper_B(row, col) * data_alpha + wrapper_C(row, col) * data_beta;
            
              //data_A[index_generator_A::mem_index(row * A_inc1 + A_start1, col * A_inc2 + A_start2, A_internal_size1, A_internal_size2)] 
              // =   data_B[index_generator_B::mem_index(row * B_inc1 + B_start1, col * B_inc2 + B_start2, B_internal_size1, B_internal_size2)] * alpha
              //   + data_C[index_generator_C::mem_index(row * C_inc1 + C_start1, col * C_inc2 + C_start2, C_internal_size1, C_internal_size2)] * beta;
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
             M2 const & mat2, ScalarType1 const & alpha, std::size_t /*len_alpha*/, bool reciprocal_alpha, bool flip_sign_alpha,
             M3 const & mat3, ScalarType2 const & beta,  std::size_t /*len_beta*/,  bool reciprocal_beta,  bool flip_sign_beta) 
      {
        typedef typename viennacl::result_of::cpu_value_type<M1>::type        value_type;
       
        value_type       * data_A = detail::extract_raw_pointer<value_type>(mat1);
        value_type const * data_B = detail::extract_raw_pointer<value_type>(mat2);
        value_type const * data_C = detail::extract_raw_pointer<value_type>(mat3);
        
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
        
        std::size_t A_start1 = viennacl::traits::start1(mat1);
        std::size_t A_start2 = viennacl::traits::start2(mat1);
        std::size_t A_inc1   = viennacl::traits::stride1(mat1);
        std::size_t A_inc2   = viennacl::traits::stride2(mat1);
        std::size_t A_size1  = viennacl::traits::size1(mat1);
        std::size_t A_size2  = viennacl::traits::size2(mat1);
        std::size_t A_internal_size1  = viennacl::traits::internal_size1(mat1);
        std::size_t A_internal_size2  = viennacl::traits::internal_size2(mat1);
        
        std::size_t B_start1 = viennacl::traits::start1(mat2);
        std::size_t B_start2 = viennacl::traits::start2(mat2);
        std::size_t B_inc1   = viennacl::traits::stride1(mat2);
        std::size_t B_inc2   = viennacl::traits::stride2(mat2);
        std::size_t B_internal_size1  = viennacl::traits::internal_size1(mat2);
        std::size_t B_internal_size2  = viennacl::traits::internal_size2(mat2);
        
        std::size_t C_start1 = viennacl::traits::start1(mat3);
        std::size_t C_start2 = viennacl::traits::start2(mat3);
        std::size_t C_inc1   = viennacl::traits::stride1(mat3);
        std::size_t C_inc2   = viennacl::traits::stride2(mat3);
        std::size_t C_internal_size1  = viennacl::traits::internal_size1(mat3);
        std::size_t C_internal_size2  = viennacl::traits::internal_size2(mat3);
        
        detail::matrix_array_wrapper<value_type,       typename M1::orientation_category, false> wrapper_A(data_A, A_start1, A_start2, A_inc1, A_inc2, A_internal_size1, A_internal_size2);
        detail::matrix_array_wrapper<value_type const, typename M2::orientation_category, false> wrapper_B(data_B, B_start1, B_start2, B_inc1, B_inc2, B_internal_size1, B_internal_size2);
        detail::matrix_array_wrapper<value_type const, typename M3::orientation_category, false> wrapper_C(data_C, C_start1, C_start2, C_inc1, C_inc2, C_internal_size1, C_internal_size2);
        
        //typedef typename detail::majority_struct_for_orientation<typename M1::orientation_category>::type index_generator_A;
        //typedef typename detail::majority_struct_for_orientation<typename M2::orientation_category>::type index_generator_B;
        //typedef typename detail::majority_struct_for_orientation<typename M3::orientation_category>::type index_generator_C;
        
        if (detail::is_row_major(typename M1::orientation_category()))
        {
#ifdef VIENNACL_WITH_OPENMP
          #pragma omp parallel for
#endif
          for (std::size_t row = 0; row < A_size1; ++row)
            for (std::size_t col = 0; col < A_size2; ++col)
              wrapper_A(row, col) += wrapper_B(row, col) * data_alpha + wrapper_C(row, col) * data_beta;
              //data_A[index_generator_A::mem_index(row * A_inc1 + A_start1, col * A_inc2 + A_start2, A_internal_size1, A_internal_size2)] 
              // +=   data_B[index_generator_B::mem_index(row * B_inc1 + B_start1, col * B_inc2 + B_start2, B_internal_size1, B_internal_size2)] * alpha
              //    + data_C[index_generator_C::mem_index(row * C_inc1 + C_start1, col * C_inc2 + C_start2, C_internal_size1, C_internal_size2)] * beta;
        }
        else
        {
#ifdef VIENNACL_WITH_OPENMP
          #pragma omp parallel for
#endif
          for (std::size_t col = 0; col < A_size2; ++col)
            for (std::size_t row = 0; row < A_size1; ++row)
              wrapper_A(row, col) += wrapper_B(row, col) * data_alpha + wrapper_C(row, col) * data_beta;
              //data_A[index_generator_A::mem_index(row * A_inc1 + A_start1, col * A_inc2 + A_start2, A_internal_size1, A_internal_size2)] 
              // +=   data_B[index_generator_B::mem_index(row * B_inc1 + B_start1, col * B_inc2 + B_start2, B_internal_size1, B_internal_size2)] * alpha
              //    + data_C[index_generator_C::mem_index(row * C_inc1 + C_start1, col * C_inc2 + C_start2, C_internal_size1, C_internal_size2)] * beta;
        }
        
      }


      
      
      template <typename M1, typename ScalarType>
      typename viennacl::enable_if<    viennacl::is_any_dense_nonstructured_matrix<M1>::value
                                    && viennacl::is_cpu_scalar<ScalarType>::value
                                  >::type    
      matrix_assign(M1 & mat, ScalarType s)
      {
        typedef typename viennacl::result_of::cpu_value_type<M1>::type        value_type;
        
        value_type       * data_A = detail::extract_raw_pointer<value_type>(mat);
        value_type alpha = static_cast<value_type>(s);
        
        std::size_t A_start1 = viennacl::traits::start1(mat);
        std::size_t A_start2 = viennacl::traits::start2(mat);
        std::size_t A_inc1   = viennacl::traits::stride1(mat);
        std::size_t A_inc2   = viennacl::traits::stride2(mat);
        std::size_t A_size1  = viennacl::traits::size1(mat);
        std::size_t A_size2  = viennacl::traits::size2(mat);
        std::size_t A_internal_size1  = viennacl::traits::internal_size1(mat);
        std::size_t A_internal_size2  = viennacl::traits::internal_size2(mat);
        
        detail::matrix_array_wrapper<value_type,       typename M1::orientation_category, false> wrapper_A(data_A, A_start1, A_start2, A_inc1, A_inc2, A_internal_size1, A_internal_size2);
        
        if (detail::is_row_major(typename M1::orientation_category()))
        {
#ifdef VIENNACL_WITH_OPENMP
          #pragma omp parallel for
#endif
          for (std::size_t row = 0; row < A_size1; ++row)
            for (std::size_t col = 0; col < A_size2; ++col)
              wrapper_A(row, col) = alpha;
              //data_A[index_generator_A::mem_index(row * A_inc1 + A_start1, col * A_inc2 + A_start2, A_internal_size1, A_internal_size2)] 
              // = data_B[index_generator_B::mem_index(row * B_inc1 + B_start1, col * B_inc2 + B_start2, B_internal_size1, B_internal_size2)] * alpha;
        }
        else
        {
#ifdef VIENNACL_WITH_OPENMP
          #pragma omp parallel for
#endif
          for (std::size_t col = 0; col < A_size2; ++col)
            for (std::size_t row = 0; row < A_size1; ++row)
              wrapper_A(row, col) = alpha;
              //data_A[index_generator_A::mem_index(row * A_inc1 + A_start1, col * A_inc2 + A_start2, A_internal_size1, A_internal_size2)] 
              // = data_B[index_generator_B::mem_index(row * B_inc1 + B_start1, col * B_inc2 + B_start2, B_internal_size1, B_internal_size2)] * alpha;
        }
      }
      
      
      
      template <typename M1, typename ScalarType>
      typename viennacl::enable_if<    viennacl::is_any_dense_nonstructured_matrix<M1>::value
                                    && viennacl::is_cpu_scalar<ScalarType>::value
                                  >::type    
      matrix_diagonal_assign(M1 & mat, ScalarType s)
      {
        typedef typename viennacl::result_of::cpu_value_type<M1>::type        value_type;
        
        value_type       * data_A = detail::extract_raw_pointer<value_type>(mat);
        value_type alpha = static_cast<value_type>(s);
        
        std::size_t A_start1 = viennacl::traits::start1(mat);
        std::size_t A_start2 = viennacl::traits::start2(mat);
        std::size_t A_inc1   = viennacl::traits::stride1(mat);
        std::size_t A_inc2   = viennacl::traits::stride2(mat);
        std::size_t A_size1  = viennacl::traits::size1(mat);
        //std::size_t A_size2  = viennacl::traits::size2(mat);
        std::size_t A_internal_size1  = viennacl::traits::internal_size1(mat);
        std::size_t A_internal_size2  = viennacl::traits::internal_size2(mat);
        
        detail::matrix_array_wrapper<value_type, typename M1::orientation_category, false> wrapper_A(data_A, A_start1, A_start2, A_inc1, A_inc2, A_internal_size1, A_internal_size2);
        
#ifdef VIENNACL_WITH_OPENMP
        #pragma omp parallel for
#endif
        for (std::size_t row = 0; row < A_size1; ++row)
          wrapper_A(row, row) = alpha;
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
        typedef typename viennacl::result_of::cpu_value_type<VectorType1>::type        value_type;
        
        value_type const * data_A = detail::extract_raw_pointer<value_type>(mat);
        value_type const * data_x = detail::extract_raw_pointer<value_type>(vec);
        value_type       * data_result = detail::extract_raw_pointer<value_type>(result);
        
        std::size_t A_start1 = viennacl::traits::start1(mat);
        std::size_t A_start2 = viennacl::traits::start2(mat);
        std::size_t A_inc1   = viennacl::traits::stride1(mat);
        std::size_t A_inc2   = viennacl::traits::stride2(mat);
        std::size_t A_size1  = viennacl::traits::size1(mat);
        std::size_t A_size2  = viennacl::traits::size2(mat);
        std::size_t A_internal_size1  = viennacl::traits::internal_size1(mat);
        std::size_t A_internal_size2  = viennacl::traits::internal_size2(mat);
        
        std::size_t start1 = viennacl::traits::start(vec);
        std::size_t inc1   = viennacl::traits::stride(vec);
        
        std::size_t start2 = viennacl::traits::start(result);
        std::size_t inc2   = viennacl::traits::stride(result);
        
        if (detail::is_row_major(typename MatrixType::orientation_category()))
        {
#ifdef VIENNACL_WITH_OPENMP
          #pragma omp parallel for
#endif
          for (std::size_t row = 0; row < A_size1; ++row)
          {
            value_type temp = 0;
            for (std::size_t col = 0; col < A_size2; ++col)
              temp += data_A[viennacl::row_major::mem_index(row * A_inc1 + A_start1, col * A_inc2 + A_start2, A_internal_size1, A_internal_size2)] * data_x[col * inc1 + start1];
              
            data_result[row * inc2 + start2] = temp;
          }
        }
        else
        {
          {
            value_type temp = data_x[start1];
            for (std::size_t row = 0; row < A_size1; ++row)
              data_result[row * inc2 + start2] = data_A[viennacl::column_major::mem_index(row * A_inc1 + A_start1, A_start2, A_internal_size1, A_internal_size2)] * temp;
          }
          for (std::size_t col = 1; col < A_size2; ++col)  //run through matrix sequentially
          {
            value_type temp = data_x[col * inc1 + start1];
            for (std::size_t row = 0; row < A_size1; ++row)
              data_result[row * inc2 + start2] += data_A[viennacl::column_major::mem_index(row * A_inc1 + A_start1, col * A_inc2 + A_start2, A_internal_size1, A_internal_size2)] * temp;
          }
        }
      }


      // trans(A) * x
      
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
      prod_impl(const viennacl::matrix_expression< const M1,
                                                   const M1,
                                                   op_trans> & mat_trans,
                const V1 & vec, 
                      V2 & result)
      {
        typedef typename viennacl::result_of::cpu_value_type<V1>::type        value_type;
        
        value_type const * data_A = detail::extract_raw_pointer<value_type>(mat_trans.lhs());
        value_type const * data_x = detail::extract_raw_pointer<value_type>(vec);
        value_type       * data_result = detail::extract_raw_pointer<value_type>(result);
        
        std::size_t A_start1 = viennacl::traits::start1(mat_trans.lhs());
        std::size_t A_start2 = viennacl::traits::start2(mat_trans.lhs());
        std::size_t A_inc1   = viennacl::traits::stride1(mat_trans.lhs());
        std::size_t A_inc2   = viennacl::traits::stride2(mat_trans.lhs());
        std::size_t A_size1  = viennacl::traits::size1(mat_trans.lhs());
        std::size_t A_size2  = viennacl::traits::size2(mat_trans.lhs());
        std::size_t A_internal_size1  = viennacl::traits::internal_size1(mat_trans.lhs());
        std::size_t A_internal_size2  = viennacl::traits::internal_size2(mat_trans.lhs());
        
        std::size_t start1 = viennacl::traits::start(vec);
        std::size_t inc1   = viennacl::traits::stride(vec);
        
        std::size_t start2 = viennacl::traits::start(result);
        std::size_t inc2   = viennacl::traits::stride(result);
        
        if (detail::is_row_major(typename M1::orientation_category()))
        {
          {
            value_type temp = data_x[start1];
            for (std::size_t row = 0; row < A_size2; ++row)
              data_result[row * inc2 + start2] = data_A[viennacl::row_major::mem_index(A_start2, row * A_inc1 + A_start1, A_internal_size1, A_internal_size2)] * temp;
          }
            
          for (std::size_t col = 1; col < A_size1; ++col)  //run through matrix sequentially
          {
            value_type temp = data_x[col * inc1 + start1];
            for (std::size_t row = 0; row < A_size2; ++row)
              data_result[row * inc2 + start2] += data_A[viennacl::row_major::mem_index(col * A_inc2 + A_start2, row * A_inc1 + A_start1, A_internal_size1, A_internal_size2)] * temp;
          }
        }
        else
        {
#ifdef VIENNACL_WITH_OPENMP
          #pragma omp parallel for
#endif
          for (std::size_t row = 0; row < A_size2; ++row)
          {
            value_type temp = 0;
            for (std::size_t col = 0; col < A_size1; ++col)
              temp += data_A[viennacl::column_major::mem_index(col * A_inc2 + A_start2, row * A_inc1 + A_start1, A_internal_size1, A_internal_size2)] * data_x[col * inc1 + start1];
              
            data_result[row * inc2 + start2] = temp;
          }
        }
      }


      //
      /////////////////////////   matrix-matrix products /////////////////////////////////
      //

      namespace detail
      {
        template <typename A, typename B, typename C, typename NumericT>
        void prod(A & a, B & b, C & c,
                  std::size_t C_size1, std::size_t C_size2, std::size_t A_size2,
                  NumericT alpha, NumericT beta)
        {
          for (std::size_t i=0; i<C_size1; ++i)
          {
            for (std::size_t j=0; j<C_size2; ++j)
            {
              NumericT temp = 0;
              for (std::size_t k=0; k<A_size2; ++k)
                temp += a(i, k) * b(k, j);
                
              temp *= alpha;
              if (beta != 0)
                temp += beta * c(i,j);
              c(i,j) = temp;
            }
          }
        }
        
      }
      
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
        typedef typename viennacl::result_of::cpu_value_type<T1>::type        value_type;
       
        value_type const * data_A = detail::extract_raw_pointer<value_type>(A);
        value_type const * data_B = detail::extract_raw_pointer<value_type>(B);
        value_type       * data_C = detail::extract_raw_pointer<value_type>(C);
        
        std::size_t A_start1 = viennacl::traits::start1(A);
        std::size_t A_start2 = viennacl::traits::start2(A);
        std::size_t A_inc1   = viennacl::traits::stride1(A);
        std::size_t A_inc2   = viennacl::traits::stride2(A);
        std::size_t A_size2  = viennacl::traits::size2(A);
        std::size_t A_internal_size1  = viennacl::traits::internal_size1(A);
        std::size_t A_internal_size2  = viennacl::traits::internal_size2(A);
        
        std::size_t B_start1 = viennacl::traits::start1(B);
        std::size_t B_start2 = viennacl::traits::start2(B);
        std::size_t B_inc1   = viennacl::traits::stride1(B);
        std::size_t B_inc2   = viennacl::traits::stride2(B);
        std::size_t B_internal_size1  = viennacl::traits::internal_size1(B);
        std::size_t B_internal_size2  = viennacl::traits::internal_size2(B);
        
        std::size_t C_start1 = viennacl::traits::start1(C);
        std::size_t C_start2 = viennacl::traits::start2(C);
        std::size_t C_inc1   = viennacl::traits::stride1(C);
        std::size_t C_inc2   = viennacl::traits::stride2(C);
        std::size_t C_size1  = viennacl::traits::size1(C);
        std::size_t C_size2  = viennacl::traits::size2(C);
        std::size_t C_internal_size1  = viennacl::traits::internal_size1(C);
        std::size_t C_internal_size2  = viennacl::traits::internal_size2(C);
        
        detail::matrix_array_wrapper<value_type const, typename T1::orientation_category, false>   wrapper_A(data_A, A_start1, A_start2, A_inc1, A_inc2, A_internal_size1, A_internal_size2);
        detail::matrix_array_wrapper<value_type const, typename T2::orientation_category, false>   wrapper_B(data_B, B_start1, B_start2, B_inc1, B_inc2, B_internal_size1, B_internal_size2);
        detail::matrix_array_wrapper<value_type,       typename T3::orientation_category, false>   wrapper_C(data_C, C_start1, C_start2, C_inc1, C_inc2, C_internal_size1, C_internal_size2);
        
        detail::prod(wrapper_A, wrapper_B, wrapper_C, C_size1, C_size2, A_size2, static_cast<value_type>(alpha), static_cast<value_type>(beta));
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
        typedef typename viennacl::result_of::cpu_value_type<T1>::type        value_type;
       
        value_type const * data_A = detail::extract_raw_pointer<value_type>(A.lhs());
        value_type const * data_B = detail::extract_raw_pointer<value_type>(B);
        value_type       * data_C = detail::extract_raw_pointer<value_type>(C);
        
        std::size_t A_start1 = viennacl::traits::start1(A.lhs());
        std::size_t A_start2 = viennacl::traits::start2(A.lhs());
        std::size_t A_inc1   = viennacl::traits::stride1(A.lhs());
        std::size_t A_inc2   = viennacl::traits::stride2(A.lhs());
        std::size_t A_size1  = viennacl::traits::size1(A.lhs());
        std::size_t A_internal_size1  = viennacl::traits::internal_size1(A.lhs());
        std::size_t A_internal_size2  = viennacl::traits::internal_size2(A.lhs());
        
        std::size_t B_start1 = viennacl::traits::start1(B);
        std::size_t B_start2 = viennacl::traits::start2(B);
        std::size_t B_inc1   = viennacl::traits::stride1(B);
        std::size_t B_inc2   = viennacl::traits::stride2(B);
        std::size_t B_internal_size1  = viennacl::traits::internal_size1(B);
        std::size_t B_internal_size2  = viennacl::traits::internal_size2(B);
        
        std::size_t C_start1 = viennacl::traits::start1(C);
        std::size_t C_start2 = viennacl::traits::start2(C);
        std::size_t C_inc1   = viennacl::traits::stride1(C);
        std::size_t C_inc2   = viennacl::traits::stride2(C);
        std::size_t C_size1  = viennacl::traits::size1(C);
        std::size_t C_size2  = viennacl::traits::size2(C);
        std::size_t C_internal_size1  = viennacl::traits::internal_size1(C);
        std::size_t C_internal_size2  = viennacl::traits::internal_size2(C);
        
        detail::matrix_array_wrapper<value_type const, typename T1::orientation_category, true>    wrapper_A(data_A, A_start1, A_start2, A_inc1, A_inc2, A_internal_size1, A_internal_size2);
        detail::matrix_array_wrapper<value_type const, typename T2::orientation_category, false>   wrapper_B(data_B, B_start1, B_start2, B_inc1, B_inc2, B_internal_size1, B_internal_size2);
        detail::matrix_array_wrapper<value_type,       typename T3::orientation_category, false>   wrapper_C(data_C, C_start1, C_start2, C_inc1, C_inc2, C_internal_size1, C_internal_size2);
        
        detail::prod(wrapper_A, wrapper_B, wrapper_C, C_size1, C_size2, A_size1, static_cast<value_type>(alpha), static_cast<value_type>(beta));
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
                const viennacl::matrix_expression< const T2,
                                                  const T2,
                                                  op_trans> & B,
                      T3 & C,
                ScalarType alpha,
                ScalarType beta)
      {
        typedef typename viennacl::result_of::cpu_value_type<T1>::type        value_type;
       
        value_type const * data_A = detail::extract_raw_pointer<value_type>(A);
        value_type const * data_B = detail::extract_raw_pointer<value_type>(B.lhs());
        value_type       * data_C = detail::extract_raw_pointer<value_type>(C);
        
        std::size_t A_start1 = viennacl::traits::start1(A);
        std::size_t A_start2 = viennacl::traits::start2(A);
        std::size_t A_inc1   = viennacl::traits::stride1(A);
        std::size_t A_inc2   = viennacl::traits::stride2(A);
        std::size_t A_size2  = viennacl::traits::size2(A);
        std::size_t A_internal_size1  = viennacl::traits::internal_size1(A);
        std::size_t A_internal_size2  = viennacl::traits::internal_size2(A);
        
        std::size_t B_start1 = viennacl::traits::start1(B.lhs());
        std::size_t B_start2 = viennacl::traits::start2(B.lhs());
        std::size_t B_inc1   = viennacl::traits::stride1(B.lhs());
        std::size_t B_inc2   = viennacl::traits::stride2(B.lhs());
        std::size_t B_internal_size1  = viennacl::traits::internal_size1(B.lhs());
        std::size_t B_internal_size2  = viennacl::traits::internal_size2(B.lhs());
        
        std::size_t C_start1 = viennacl::traits::start1(C);
        std::size_t C_start2 = viennacl::traits::start2(C);
        std::size_t C_inc1   = viennacl::traits::stride1(C);
        std::size_t C_inc2   = viennacl::traits::stride2(C);
        std::size_t C_size1  = viennacl::traits::size1(C);
        std::size_t C_size2  = viennacl::traits::size2(C);
        std::size_t C_internal_size1  = viennacl::traits::internal_size1(C);
        std::size_t C_internal_size2  = viennacl::traits::internal_size2(C);
        
        detail::matrix_array_wrapper<value_type const, typename T1::orientation_category, false>   wrapper_A(data_A, A_start1, A_start2, A_inc1, A_inc2, A_internal_size1, A_internal_size2);
        detail::matrix_array_wrapper<value_type const, typename T2::orientation_category, true>    wrapper_B(data_B, B_start1, B_start2, B_inc1, B_inc2, B_internal_size1, B_internal_size2);
        detail::matrix_array_wrapper<value_type,       typename T3::orientation_category, false>   wrapper_C(data_C, C_start1, C_start2, C_inc1, C_inc2, C_internal_size1, C_internal_size2);
        
        detail::prod(wrapper_A, wrapper_B, wrapper_C, C_size1, C_size2, A_size2, static_cast<value_type>(alpha), static_cast<value_type>(beta));
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
        typedef typename viennacl::result_of::cpu_value_type<T1>::type        value_type;
       
        value_type const * data_A = detail::extract_raw_pointer<value_type>(A.lhs());
        value_type const * data_B = detail::extract_raw_pointer<value_type>(B.lhs());
        value_type       * data_C = detail::extract_raw_pointer<value_type>(C);
        
        std::size_t A_start1 = viennacl::traits::start1(A.lhs());
        std::size_t A_start2 = viennacl::traits::start2(A.lhs());
        std::size_t A_inc1   = viennacl::traits::stride1(A.lhs());
        std::size_t A_inc2   = viennacl::traits::stride2(A.lhs());
        std::size_t A_size1  = viennacl::traits::size1(A.lhs());
        std::size_t A_internal_size1  = viennacl::traits::internal_size1(A.lhs());
        std::size_t A_internal_size2  = viennacl::traits::internal_size2(A.lhs());
        
        std::size_t B_start1 = viennacl::traits::start1(B.lhs());
        std::size_t B_start2 = viennacl::traits::start2(B.lhs());
        std::size_t B_inc1   = viennacl::traits::stride1(B.lhs());
        std::size_t B_inc2   = viennacl::traits::stride2(B.lhs());
        std::size_t B_internal_size1  = viennacl::traits::internal_size1(B.lhs());
        std::size_t B_internal_size2  = viennacl::traits::internal_size2(B.lhs());
        
        std::size_t C_start1 = viennacl::traits::start1(C);
        std::size_t C_start2 = viennacl::traits::start2(C);
        std::size_t C_inc1   = viennacl::traits::stride1(C);
        std::size_t C_inc2   = viennacl::traits::stride2(C);
        std::size_t C_size1  = viennacl::traits::size1(C);
        std::size_t C_size2  = viennacl::traits::size2(C);
        std::size_t C_internal_size1  = viennacl::traits::internal_size1(C);
        std::size_t C_internal_size2  = viennacl::traits::internal_size2(C);
        
        detail::matrix_array_wrapper<value_type const, typename T1::orientation_category, true>    wrapper_A(data_A, A_start1, A_start2, A_inc1, A_inc2, A_internal_size1, A_internal_size2);
        detail::matrix_array_wrapper<value_type const, typename T2::orientation_category, true>    wrapper_B(data_B, B_start1, B_start2, B_inc1, B_inc2, B_internal_size1, B_internal_size2);
        detail::matrix_array_wrapper<value_type,       typename T3::orientation_category, false>   wrapper_C(data_C, C_start1, C_start2, C_inc1, C_inc2, C_internal_size1, C_internal_size2);
        
        detail::prod(wrapper_A, wrapper_B, wrapper_C, C_size1, C_size2, A_size1, static_cast<value_type>(alpha), static_cast<value_type>(beta));
      }




      //
      /////////////////////////   miscellaneous operations /////////////////////////////////
      //

      
      /** @brief The implementation of the operation mat += alpha * vec1 * vec2^T, i.e. a scaled rank 1 update
      *
      * Implementation of the convenience expression result += alpha * outer_prod(vec1, vec2);
      *
      * @param mat1    The matrix to be updated
      * @param alpha            The scaling factor (either a viennacl::scalar<>, float, or double)
      * @param reciprocal_alpha Use 1/alpha instead of alpha
      * @param flip_sign_alpha  Use -alpha instead of alpha
      * @param vec1    The first vector
      * @param vec2    The second vector
      */
      template <typename M1, typename S1, typename V1, typename V2>
      typename viennacl::enable_if<    viennacl::is_any_dense_nonstructured_matrix<M1>::value
                                    && viennacl::is_any_scalar<S1>::value
                                    && viennacl::is_any_dense_nonstructured_vector<V1>::value
                                    && viennacl::is_any_dense_nonstructured_vector<V2>::value
                                  >::type
      scaled_rank_1_update(M1 & mat1,
                    S1 const & alpha, std::size_t /*len_alpha*/, bool reciprocal_alpha, bool flip_sign_alpha,
                    const V1 & vec1, 
                    const V2 & vec2)
      {
        typedef typename viennacl::result_of::cpu_value_type<V1>::type        value_type;
       
        value_type       * data_A  = detail::extract_raw_pointer<value_type>(mat1);
        value_type const * data_v1 = detail::extract_raw_pointer<value_type>(vec1);
        value_type const * data_v2 = detail::extract_raw_pointer<value_type>(vec2);
        
        std::size_t A_start1 = viennacl::traits::start1(mat1);
        std::size_t A_start2 = viennacl::traits::start2(mat1);
        std::size_t A_inc1   = viennacl::traits::stride1(mat1);
        std::size_t A_inc2   = viennacl::traits::stride2(mat1);
        std::size_t A_size1  = viennacl::traits::size1(mat1);
        std::size_t A_size2  = viennacl::traits::size2(mat1);
        std::size_t A_internal_size1  = viennacl::traits::internal_size1(mat1);
        std::size_t A_internal_size2  = viennacl::traits::internal_size2(mat1);
        
        std::size_t start1 = viennacl::traits::start(vec1);
        std::size_t inc1   = viennacl::traits::stride(vec1);
        
        std::size_t start2 = viennacl::traits::start(vec2);
        std::size_t inc2   = viennacl::traits::stride(vec2);
        
        value_type data_alpha = alpha;
        if (flip_sign_alpha)
          data_alpha = -data_alpha;
        if (reciprocal_alpha)
          data_alpha = static_cast<value_type>(1) / data_alpha;
        
        if (detail::is_row_major(typename M1::orientation_category()))
        {
          for (std::size_t row = 0; row < A_size1; ++row)
          {
            value_type value_v1 = data_alpha * data_v1[row * inc1 + start1];
            for (std::size_t col = 0; col < A_size2; ++col)
              data_A[viennacl::row_major::mem_index(row * A_inc1 + A_start1, col * A_inc2 + A_start2, A_internal_size1, A_internal_size2)] += value_v1 * data_v2[col * inc2 + start2];
          }
        }
        else
        {
          for (std::size_t col = 0; col < A_size2; ++col)  //run through matrix sequentially
          {
            value_type value_v2 = data_alpha * data_v2[col * inc2 + start2];
            for (std::size_t row = 0; row < A_size1; ++row)
              data_A[viennacl::column_major::mem_index(row * A_inc1 + A_start1, col * A_inc2 + A_start2, A_internal_size1, A_internal_size2)] += data_v1[row * inc1 + start1] * value_v2;
          }
        }
      }

    } // namespace host_based
  } //namespace linalg
} //namespace viennacl


#endif
