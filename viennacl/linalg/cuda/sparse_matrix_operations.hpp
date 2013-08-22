#ifndef VIENNACL_LINALG_CUDA_SPARSE_MATRIX_OPERATIONS_HPP_
#define VIENNACL_LINALG_CUDA_SPARSE_MATRIX_OPERATIONS_HPP_

/* =========================================================================
   Copyright (c) 2010-2013, Institute for Microelectronics,
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

/** @file viennacl/linalg/cuda/sparse_matrix_operations.hpp
    @brief Implementations of operations using sparse matrices using CUDA
*/

#include "viennacl/forwards.h"
#include "viennacl/scalar.hpp"
#include "viennacl/vector.hpp"
#include "viennacl/tools/tools.hpp"
#include "viennacl/linalg/cuda/common.hpp"

#include "viennacl/linalg/cuda/sparse_matrix_operations_solve.hpp"

namespace viennacl
{
  namespace linalg
  {
    namespace cuda
    {
      //
      // Compressed matrix
      //

      namespace detail
      {

        template <typename T>
        __global__ void csr_row_info_extractor_kernel(
                  const unsigned int * row_indices,
                  const unsigned int * column_indices,
                  const T * elements,
                  T * result,
                  unsigned int size,
                  unsigned int option)
        {
          for (unsigned int row  = blockDim.x * blockIdx.x + threadIdx.x;
                            row  < size;
                            row += gridDim.x * blockDim.x)
          {
            T value = 0;
            unsigned int row_end = row_indices[row+1];

            switch (option)
            {
              case 0: //inf-norm
                for (unsigned int i = row_indices[row]; i < row_end; ++i)
                  value = max(value, fabs(elements[i]));
                break;

              case 1: //1-norm
                for (unsigned int i = row_indices[row]; i < row_end; ++i)
                  value += fabs(elements[i]);
                break;

              case 2: //2-norm
                for (unsigned int i = row_indices[row]; i < row_end; ++i)
                  value += elements[i] * elements[i];
                value = sqrt(value);
                break;

              case 3: //diagonal entry
                for (unsigned int i = row_indices[row]; i < row_end; ++i)
                {
                  if (column_indices[i] == row)
                  {
                    value = elements[i];
                    break;
                  }
                }
                break;

              default:
                break;
            }
            result[row] = value;
          }
        }


        template<typename ScalarType, unsigned int MAT_ALIGNMENT>
        void row_info(compressed_matrix<ScalarType, MAT_ALIGNMENT> const & mat,
                      vector_base<ScalarType> & vec,
                      viennacl::linalg::detail::row_info_types info_selector)
        {
          ScalarType const * ptr1 = detail::cuda_arg<ScalarType>(mat.handle().cuda_handle());
          ScalarType       * ptr2 = detail::cuda_arg<ScalarType>(vec);
          //csr_row_info_extractor_kernel<<<128, 128>>>(detail::cuda_arg<unsigned int>(mat.handle1().cuda_handle()),
          //                                            detail::cuda_arg<unsigned int>(mat.handle2().cuda_handle()),
          //                                            detail::cuda_arg<ScalarType>(mat.handle().cuda_handle()),
          //                                            detail::cuda_arg<ScalarType>(vec),
          //                                            static_cast<unsigned int>(mat.size1()),
          //                                           static_cast<unsigned int>(info_selector)
          //                                           );
          VIENNACL_CUDA_LAST_ERROR_CHECK("csr_row_info_extractor_kernel");
        }

      } //namespace detail


      template <typename T>
      __global__ void compressed_matrix_vec_mul_kernel(
                const unsigned int * row_indices,
                const unsigned int * column_indices,
                const T * elements,
                const T * x,
                unsigned int start_x,
                unsigned int inc_x,
                T * result,
                unsigned int start_result,
                unsigned int inc_result,
                unsigned int size_result)
      {
        for (unsigned int row  = blockDim.x * blockIdx.x + threadIdx.x;
                          row  < size_result;
                          row += gridDim.x * blockDim.x)
        {
          T dot_prod = (T)0;
          unsigned int row_end = row_indices[row+1];
          for (unsigned int i = row_indices[row]; i < row_end; ++i)
            dot_prod += elements[i] * x[column_indices[i] * inc_x + start_x];
          result[row * inc_result + start_result] = dot_prod;
        }
      }





      /** @brief Carries out matrix-vector multiplication with a compressed_matrix
      *
      * Implementation of the convenience expression result = prod(mat, vec);
      *
      * @param mat    The matrix
      * @param vec    The vector
      * @param result The result vector
      */
      template<class ScalarType, unsigned int ALIGNMENT>
      void prod_impl(const viennacl::compressed_matrix<ScalarType, ALIGNMENT> & mat,
                     const viennacl::vector_base<ScalarType> & vec,
                           viennacl::vector_base<ScalarType> & result)
      {
        compressed_matrix_vec_mul_kernel<<<128, 128>>>(detail::cuda_arg<unsigned int>(mat.handle1().cuda_handle()),
                                                       detail::cuda_arg<unsigned int>(mat.handle2().cuda_handle()),
                                                       detail::cuda_arg<ScalarType>(mat.handle().cuda_handle()),
                                                       detail::cuda_arg<ScalarType>(vec),
                                                       static_cast<unsigned int>(vec.start()),
                                                       static_cast<unsigned int>(vec.stride()),
                                                       detail::cuda_arg<ScalarType>(result),
                                                       static_cast<unsigned int>(result.start()),
                                                       static_cast<unsigned int>(result.stride()),
                                                       static_cast<unsigned int>(result.size())
                                                      );
        VIENNACL_CUDA_LAST_ERROR_CHECK("compressed_matrix_vec_mul_kernel");
      }

      template <typename T>
      __global__ void compressed_matrix_d_mat_mul_kernel(
                const unsigned int * sp_mat_row_indices,
                const unsigned int * sp_mat_col_indices,
                const T * sp_mat_elements,
                const T * d_mat,
                unsigned int d_mat_row_start,
                unsigned int d_mat_col_start,
                unsigned int d_mat_row_inc,
                unsigned int d_mat_col_inc,
                unsigned int d_mat_row_size,
                unsigned int d_mat_col_size,
                unsigned int d_mat_internal_rows,
                unsigned int d_mat_internal_cols,
                T * result,
                unsigned int result_row_start,
                unsigned int result_col_start,
                unsigned int result_row_inc,
                unsigned int result_col_inc,
                unsigned int result_row_size,
                unsigned int result_col_size,
                unsigned int result_internal_rows,
                unsigned int result_internal_cols) {

        for (unsigned int row  = blockIdx.x; row  < result_row_size; row += gridDim.x) {

          unsigned int row_start = sp_mat_row_indices[row];
          unsigned int row_end = sp_mat_row_indices[row+1];

          for ( unsigned int col = threadIdx.x; col < result_col_size; col += blockDim.x) {

            float r = 0;

            for (unsigned int k = row_start; k < row_end; k++) {

              unsigned int j = sp_mat_col_indices[k];
              float x = sp_mat_elements[k];

              float y = d_mat[ (d_mat_row_start + j * d_mat_row_inc) * d_mat_internal_cols +
                                d_mat_col_start + col * d_mat_col_inc ];

              r += x * y;
            }

            result [ (result_row_start + row * result_row_inc) * result_internal_cols +
                      result_col_start + col * result_col_inc ] = r;
          }

        }

      }


      /** @brief Carries out sparse_matrix-dense_matrix multiplication first matrix being compressed
      *
      * Implementation of the convenience expression result = prod(mat, vec);
      *
      * @param sp_mat   The sparse matrix
      * @param d_mat    The dense matrix
      * @param result   The result matrix
      */
      template< typename TYPE, unsigned int ALIGNMENT, typename F>
      void prod_impl(const viennacl::compressed_matrix<TYPE, ALIGNMENT> & sp_mat,
                     const viennacl::matrix_base<TYPE, F> & d_mat,
                           viennacl::matrix_base<TYPE, F> & result) {
        compressed_matrix_d_mat_mul_kernel<<<128, 128>>>
                                                      (detail::cuda_arg<unsigned int>(sp_mat.handle1().cuda_handle()),
                                                       detail::cuda_arg<unsigned int>(sp_mat.handle2().cuda_handle()),
                                                       detail::cuda_arg<TYPE>(sp_mat.handle().cuda_handle()),

                                                       detail::cuda_arg<TYPE>(d_mat),
                                                       static_cast<unsigned int>(viennacl::traits::start1(d_mat)),         static_cast<unsigned int>(viennacl::traits::start2(d_mat)),
                                                       static_cast<unsigned int>(viennacl::traits::stride1(d_mat)),        static_cast<unsigned int>(viennacl::traits::stride2(d_mat)),
                                                       static_cast<unsigned int>(viennacl::traits::size1(d_mat)),          static_cast<unsigned int>(viennacl::traits::size2(d_mat)),
                                                       static_cast<unsigned int>(viennacl::traits::internal_size1(d_mat)), static_cast<unsigned int>(viennacl::traits::internal_size2(d_mat)),

                                                       detail::cuda_arg<TYPE>(result),
                                                       static_cast<unsigned int>(viennacl::traits::start1(result)),         static_cast<unsigned int>(viennacl::traits::start2(result)),
                                                       static_cast<unsigned int>(viennacl::traits::stride1(result)),        static_cast<unsigned int>(viennacl::traits::stride2(result)),
                                                       static_cast<unsigned int>(viennacl::traits::size1(result)),          static_cast<unsigned int>(viennacl::traits::size2(result)),
                                                       static_cast<unsigned int>(viennacl::traits::internal_size1(result)), static_cast<unsigned int>(viennacl::traits::internal_size2(result))
                                                      );
        VIENNACL_CUDA_LAST_ERROR_CHECK("compressed_matrix_d_mat_mul_kernel");
      }


      template <typename T>
      __global__ void compressed_matrix_d_tr_mat_mul_kernel(
                const unsigned int * sp_mat_row_indices,
                const unsigned int * sp_mat_col_indices,
                const T * sp_mat_elements,
                const T * d_mat,
                unsigned int d_mat_row_start,
                unsigned int d_mat_col_start,
                unsigned int d_mat_row_inc,
                unsigned int d_mat_col_inc,
                unsigned int d_mat_row_size,
                unsigned int d_mat_col_size,
                unsigned int d_mat_internal_rows,
                unsigned int d_mat_internal_cols,
                T * result,
                unsigned int result_row_start,
                unsigned int result_col_start,
                unsigned int result_row_inc,
                unsigned int result_col_inc,
                unsigned int result_row_size,
                unsigned int result_col_size,
                unsigned int result_internal_rows,
                unsigned int result_internal_cols) {

        for (unsigned int row  = blockIdx.x; row  < result_row_size; row += gridDim.x) {

          unsigned int row_start = sp_mat_row_indices[row];
          unsigned int row_end = sp_mat_row_indices[row+1];

          for ( unsigned int col = threadIdx.x; col < result_col_size; col += blockDim.x) {

            float r = 0;

            for (unsigned int k = row_start; k < row_end; k++) {

              unsigned int j = sp_mat_col_indices[k];
              float x = sp_mat_elements[k];

              float y = d_mat[ (d_mat_row_start + col * d_mat_row_inc) * d_mat_internal_cols +
                                d_mat_col_start + j * d_mat_col_inc ];

              r += x * y;
            }

            result [ (result_row_start + row * result_row_inc) * result_internal_cols +
                      result_col_start + col * result_col_inc ] = r;
          }
        }

      }

      /** @brief Carries out matrix-trans(matrix) multiplication first matrix being compressed
      *          and the second transposed
      *
      * Implementation of the convenience expression result = prod(sp_mat, d_mat);
      *
      * @param sp_mat             The sparse matrix
      * @param trans(d_mat)       The transposed dense matrix
      * @param result             The result matrix
      */
      template< typename TYPE, unsigned int ALIGNMENT, typename F>
      void prod_impl(const viennacl::compressed_matrix<TYPE, ALIGNMENT> & sp_mat,
                     const viennacl::matrix_expression< const viennacl::matrix_base<TYPE, F>,
                                                        const viennacl::matrix_base<TYPE, F>,
                                                        viennacl::op_trans > & d_mat,
                      viennacl::matrix_base<TYPE, F> & result) {

        compressed_matrix_d_tr_mat_mul_kernel<<<128, 128>>>
                                                      (detail::cuda_arg<unsigned int>(sp_mat.handle1().cuda_handle()),
                                                       detail::cuda_arg<unsigned int>(sp_mat.handle2().cuda_handle()),
                                                       detail::cuda_arg<TYPE>(sp_mat.handle().cuda_handle()),

                                                       detail::cuda_arg<TYPE>(d_mat.lhs()),
                                                       static_cast<unsigned int>(viennacl::traits::start1(d_mat.lhs())),         static_cast<unsigned int>(viennacl::traits::start2(d_mat.lhs())),
                                                       static_cast<unsigned int>(viennacl::traits::stride1(d_mat.lhs())),        static_cast<unsigned int>(viennacl::traits::stride2(d_mat.lhs())),
                                                       static_cast<unsigned int>(viennacl::traits::size1(d_mat.lhs())),          static_cast<unsigned int>(viennacl::traits::size2(d_mat.lhs())),
                                                       static_cast<unsigned int>(viennacl::traits::internal_size1(d_mat.lhs())), static_cast<unsigned int>(viennacl::traits::internal_size2(d_mat.lhs())),

                                                       detail::cuda_arg<TYPE>(result),
                                                       static_cast<unsigned int>(viennacl::traits::start1(result)),         static_cast<unsigned int>(viennacl::traits::start2(result)),
                                                       static_cast<unsigned int>(viennacl::traits::stride1(result)),        static_cast<unsigned int>(viennacl::traits::stride2(result)),
                                                       static_cast<unsigned int>(viennacl::traits::size1(result)),          static_cast<unsigned int>(viennacl::traits::size2(result)),
                                                       static_cast<unsigned int>(viennacl::traits::internal_size1(result)), static_cast<unsigned int>(viennacl::traits::internal_size2(result))
                                                      );
        VIENNACL_CUDA_LAST_ERROR_CHECK("compressed_matrix_d_tr_mat_mul_kernel");
      }


      //
      // triangular solves for compressed_matrix
      //

      template <typename T>
      __global__ void compressed_matrix_diagonal_kernel(
                const unsigned int * row_indices,
                const unsigned int * column_indices,
                const T * elements,
                T * result,
                unsigned int size)
      {
        for (unsigned int row  = blockDim.x * blockIdx.x + threadIdx.x;
                          row  < size;
                          row += gridDim.x * blockDim.x)
        {
          T diag = (T)0;
          unsigned int row_end = row_indices[row+1];
          for (unsigned int i = row_indices[row]; i < row_end; ++i)
          {
            unsigned int col_index = column_indices[i];
            if (col_index == row)
            {
              diag = elements[i];
              break;
            }
          }
          result[row] = diag;
        }
      }


      /** @brief Carries out triangular inplace solves
      *
      * @param mat    The matrix
      * @param vec    The vector holding the right hand side. Is overwritten by the solution.
      */
      template<typename SparseMatrixType, class ScalarType>
      typename viennacl::enable_if< viennacl::is_any_sparse_matrix<SparseMatrixType>::value>::type
      inplace_solve(const SparseMatrixType & mat,
                    viennacl::vector_base<ScalarType> & vec,
                    viennacl::linalg::unit_lower_tag)
      {
        csr_unit_lu_forward_kernel<<<1, 128>>>(detail::cuda_arg<unsigned int>(mat.handle1().cuda_handle()),
                                          detail::cuda_arg<unsigned int>(mat.handle2().cuda_handle()),
                                          detail::cuda_arg<ScalarType>(mat.handle().cuda_handle()),
                                          detail::cuda_arg<ScalarType>(vec),
                                          static_cast<unsigned int>(mat.size1())
                                         );
        VIENNACL_CUDA_LAST_ERROR_CHECK("csr_unit_lu_forward_kernel");
      }


      /** @brief Carries out triangular inplace solves
      *
      * @param mat    The matrix
      * @param vec    The vector holding the right hand side. Is overwritten by the solution.
      */
      template<typename SparseMatrixType, class ScalarType>
      typename viennacl::enable_if< viennacl::is_any_sparse_matrix<SparseMatrixType>::value>::type
      inplace_solve(const SparseMatrixType & mat,
                    viennacl::vector_base<ScalarType> & vec,
                    viennacl::linalg::lower_tag)
      {
        csr_lu_forward_kernel<<<1, 128>>>(detail::cuda_arg<unsigned int>(mat.handle1().cuda_handle()),
                                          detail::cuda_arg<unsigned int>(mat.handle2().cuda_handle()),
                                          detail::cuda_arg<ScalarType>(mat.handle().cuda_handle()),
                                          detail::cuda_arg<ScalarType>(vec),
                                          static_cast<unsigned int>(mat.size1())
                                         );
        VIENNACL_CUDA_LAST_ERROR_CHECK("csr_lu_forward_kernel");
      }



      /** @brief Carries out triangular inplace solves
      *
      * @param mat    The matrix
      * @param vec    The vector holding the right hand side. Is overwritten by the solution.
      */
      template<typename SparseMatrixType, class ScalarType>
      typename viennacl::enable_if< viennacl::is_any_sparse_matrix<SparseMatrixType>::value>::type
      inplace_solve(const SparseMatrixType & mat,
                    viennacl::vector_base<ScalarType> & vec,
                    viennacl::linalg::unit_upper_tag)
      {
        csr_unit_lu_backward_kernel<<<1, 128>>>(detail::cuda_arg<unsigned int>(mat.handle1().cuda_handle()),
                                          detail::cuda_arg<unsigned int>(mat.handle2().cuda_handle()),
                                          detail::cuda_arg<ScalarType>(mat.handle().cuda_handle()),
                                          detail::cuda_arg<ScalarType>(vec),
                                          static_cast<unsigned int>(mat.size1())
                                         );
        VIENNACL_CUDA_LAST_ERROR_CHECK("csr_unit_lu_backward_kernel");
      }


      /** @brief Carries out triangular inplace solves
      *
      * @param mat    The matrix
      * @param vec    The vector holding the right hand side. Is overwritten by the solution.
      */
      template<typename SparseMatrixType, class ScalarType>
      typename viennacl::enable_if< viennacl::is_any_sparse_matrix<SparseMatrixType>::value>::type
      inplace_solve(const SparseMatrixType & mat,
                    viennacl::vector_base<ScalarType> & vec,
                    viennacl::linalg::upper_tag)
      {
        csr_lu_backward_kernel<<<1, 128>>>(detail::cuda_arg<unsigned int>(mat.handle1().cuda_handle()),
                                          detail::cuda_arg<unsigned int>(mat.handle2().cuda_handle()),
                                          detail::cuda_arg<ScalarType>(mat.handle().cuda_handle()),
                                          detail::cuda_arg<ScalarType>(vec),
                                          static_cast<unsigned int>(mat.size1())
                                         );
        VIENNACL_CUDA_LAST_ERROR_CHECK("csr_lu_backward_kernel");
      }



      // transposed

      /** @brief Carries out triangular inplace solves
      *
      * @param mat    The matrix
      * @param vec    The vector holding the right hand side. Is overwritten by the solution.
      */
      template<typename SparseMatrixType, class ScalarType>
      typename viennacl::enable_if< viennacl::is_any_sparse_matrix<SparseMatrixType>::value>::type
      inplace_solve(const matrix_expression<const SparseMatrixType, const SparseMatrixType, op_trans> & mat,
                    viennacl::vector_base<ScalarType> & vec,
                    viennacl::linalg::unit_lower_tag)
      {
        csr_trans_unit_lu_forward_kernel<<<1, 128>>>(detail::cuda_arg<unsigned int>(mat.lhs().handle1().cuda_handle()),
                                                detail::cuda_arg<unsigned int>(mat.lhs().handle2().cuda_handle()),
                                                detail::cuda_arg<ScalarType>(mat.lhs().handle().cuda_handle()),
                                                detail::cuda_arg<ScalarType>(vec),
                                                static_cast<unsigned int>(mat.lhs().size1())
                                               );
        VIENNACL_CUDA_LAST_ERROR_CHECK("csr_trans_unit_lu_forward_kernel");
      }


      /** @brief Carries out triangular inplace solves
      *
      * @param mat    The matrix
      * @param vec    The vector holding the right hand side. Is overwritten by the solution.
      */
      template<typename SparseMatrixType, class ScalarType>
      typename viennacl::enable_if< viennacl::is_any_sparse_matrix<SparseMatrixType>::value>::type
      inplace_solve(const matrix_expression<const SparseMatrixType, const SparseMatrixType, op_trans> & mat,
                    viennacl::vector_base<ScalarType> & vec,
                    viennacl::linalg::lower_tag)
      {
        viennacl::vector<ScalarType> diagonal(vec.size());

        compressed_matrix_diagonal_kernel<<<1, 128>>>(detail::cuda_arg<unsigned int>(mat.lhs().handle1().cuda_handle()),
                                                      detail::cuda_arg<unsigned int>(mat.lhs().handle2().cuda_handle()),
                                                      detail::cuda_arg<ScalarType>(mat.lhs().handle().cuda_handle()),
                                                      detail::cuda_arg<ScalarType>(diagonal),
                                                      static_cast<unsigned int>(mat.size1())
                                                     );

        csr_trans_lu_forward_kernel<<<1, 128>>>(detail::cuda_arg<unsigned int>(mat.lhs().handle1().cuda_handle()),
                                                detail::cuda_arg<unsigned int>(mat.lhs().handle2().cuda_handle()),
                                                detail::cuda_arg<ScalarType>(mat.lhs().handle().cuda_handle()),
                                                detail::cuda_arg<ScalarType>(diagonal),
                                                detail::cuda_arg<ScalarType>(vec),
                                                static_cast<unsigned int>(mat.lhs().size1())
                                               );
        VIENNACL_CUDA_LAST_ERROR_CHECK("csr_trans_lu_forward_kernel");
      }


      /** @brief Carries out triangular inplace solves
      *
      * @param mat    The matrix
      * @param vec    The vector holding the right hand side. Is overwritten by the solution.
      */
      template<typename SparseMatrixType, class ScalarType>
      typename viennacl::enable_if< viennacl::is_any_sparse_matrix<SparseMatrixType>::value>::type
      inplace_solve(const matrix_expression<const SparseMatrixType, const SparseMatrixType, op_trans> & mat,
                    viennacl::vector_base<ScalarType> & vec,
                    viennacl::linalg::unit_upper_tag)
      {
        csr_trans_unit_lu_backward_kernel<<<1, 128>>>(detail::cuda_arg<unsigned int>(mat.lhs().handle1().cuda_handle()),
                                                      detail::cuda_arg<unsigned int>(mat.lhs().handle2().cuda_handle()),
                                                      detail::cuda_arg<ScalarType>(mat.lhs().handle().cuda_handle()),
                                                      detail::cuda_arg<ScalarType>(vec),
                                                      static_cast<unsigned int>(mat.lhs().size1())
                                                    );
        VIENNACL_CUDA_LAST_ERROR_CHECK("csr_trans_unit_lu_backward_kernel");
      }


      /** @brief Carries out triangular inplace solves
      *
      * @param mat    The matrix
      * @param vec    The vector holding the right hand side. Is overwritten by the solution.
      */
      template<typename SparseMatrixType, class ScalarType>
      typename viennacl::enable_if< viennacl::is_any_sparse_matrix<SparseMatrixType>::value>::type
      inplace_solve(const matrix_expression<const SparseMatrixType, const SparseMatrixType, op_trans> & mat,
                    viennacl::vector_base<ScalarType> & vec,
                    viennacl::linalg::upper_tag)
      {
        viennacl::vector<ScalarType> diagonal(vec.size());

        compressed_matrix_diagonal_kernel<<<1, 128>>>(detail::cuda_arg<unsigned int>(mat.lhs().handle1().cuda_handle()),
                                                      detail::cuda_arg<unsigned int>(mat.lhs().handle2().cuda_handle()),
                                                      detail::cuda_arg<ScalarType>(mat.lhs().handle().cuda_handle()),
                                                      detail::cuda_arg<ScalarType>(diagonal),
                                                      static_cast<unsigned int>(mat.size1())
                                                     );

        csr_trans_lu_backward_kernel<<<1, 128>>>(detail::cuda_arg<unsigned int>(mat.lhs().handle1().cuda_handle()),
                                                 detail::cuda_arg<unsigned int>(mat.lhs().handle2().cuda_handle()),
                                                 detail::cuda_arg<ScalarType>(mat.lhs().handle().cuda_handle()),
                                                 detail::cuda_arg<ScalarType>(diagonal),
                                                 detail::cuda_arg<ScalarType>(vec),
                                                 static_cast<unsigned int>(mat.lhs().size1())
                                                );
        VIENNACL_CUDA_LAST_ERROR_CHECK("csr_trans_lu_backward_kernel");
      }

      namespace detail
      {
        //
        // block solves
        //
        template<typename ScalarType, unsigned int MAT_ALIGNMENT>
        void block_inplace_solve(const matrix_expression<const compressed_matrix<ScalarType, MAT_ALIGNMENT>,
                                                         const compressed_matrix<ScalarType, MAT_ALIGNMENT>,
                                                         op_trans> & L,
                                 viennacl::backend::mem_handle const & block_indices, std::size_t num_blocks,
                                 vector_base<ScalarType> const & /* L_diagonal */,  //ignored
                                 vector_base<ScalarType> & vec,
                                 viennacl::linalg::unit_lower_tag)
        {
          csr_block_trans_unit_lu_forward<<<num_blocks, 128>>>(detail::cuda_arg<unsigned int>(L.lhs().handle1().cuda_handle()),
                                                               detail::cuda_arg<unsigned int>(L.lhs().handle2().cuda_handle()),
                                                               detail::cuda_arg<ScalarType>(L.lhs().handle().cuda_handle()),
                                                               detail::cuda_arg<unsigned int>(block_indices.cuda_handle()),
                                                               detail::cuda_arg<ScalarType>(vec),
                                                               static_cast<unsigned int>(L.lhs().size1())
                                                              );
        }


        template<typename ScalarType, unsigned int MAT_ALIGNMENT>
        void block_inplace_solve(const matrix_expression<const compressed_matrix<ScalarType, MAT_ALIGNMENT>,
                                                         const compressed_matrix<ScalarType, MAT_ALIGNMENT>,
                                                         op_trans> & U,
                                 viennacl::backend::mem_handle const & block_indices, std::size_t num_blocks,
                                 vector_base<ScalarType> const & U_diagonal,
                                 vector_base<ScalarType> & vec,
                                 viennacl::linalg::upper_tag)
        {
          csr_block_trans_lu_backward<<<num_blocks, 128>>>(detail::cuda_arg<unsigned int>(U.lhs().handle1().cuda_handle()),
                                                           detail::cuda_arg<unsigned int>(U.lhs().handle2().cuda_handle()),
                                                           detail::cuda_arg<ScalarType>(U.lhs().handle().cuda_handle()),
                                                           detail::cuda_arg<ScalarType>(U_diagonal.handle().cuda_handle()),
                                                           detail::cuda_arg<unsigned int>(block_indices.cuda_handle()),
                                                           detail::cuda_arg<ScalarType>(vec),
                                                           static_cast<unsigned int>(U.lhs().size1())
                                                          );
        }


      }




      //
      // Coordinate Matrix
      //


      namespace detail
      {

        template <typename T>
        __global__ void coo_row_info_extractor( const unsigned int * coords, //(row_index, column_index)
                                                const T * elements,
                                                const uint  * group_boundaries,
                                                T * result,
                                                unsigned int option)
        {
          __shared__ unsigned int shared_rows[128];
          __shared__ T inter_results[128];

          uint2 tmp;
          T val;
          uint last_index  = blockDim.x - 1;
          uint group_start = group_boundaries[blockIdx.x];
          uint group_end   = group_boundaries[blockIdx.x + 1];
          uint k_end = (group_end > group_start) ? 1 + (group_end - group_start - 1) / blockDim.x : 0;   // -1 in order to have correct behavior if group_end - group_start == j * blockDim.x

          uint local_index = 0;

          for (uint k = 0; k < k_end; ++k)
          {
            local_index = group_start + k * blockDim.x + threadIdx.x;

            tmp = (local_index < group_end) ? ((const uint2 *)coords)[local_index] : ::make_uint2(0, 0);
            val = (local_index < group_end && (option != 3 || tmp.x == tmp.y) ) ? elements[local_index] : 0;

            //check for carry from previous loop run:
            if (threadIdx.x == 0 && k > 0)
            {
              if (tmp.x == shared_rows[last_index])
              {
                switch (option)
                {
                  case 0: //inf-norm
                  case 3: //diagonal entry
                    val = max(val, fabs(inter_results[last_index]));
                    break;

                  case 1: //1-norm
                    val = fabs(val) + inter_results[last_index];
                    break;

                  case 2: //2-norm
                    val = sqrt(val * val + inter_results[last_index]);
                    break;

                  default:
                    break;
                }
              }
              else
              {
                switch (option)
                {
                  case 0: //inf-norm
                  case 1: //1-norm
                  case 3: //diagonal entry
                    result[shared_rows[last_index]] = inter_results[last_index];
                    break;

                  case 2: //2-norm
                    result[shared_rows[last_index]] = sqrt(inter_results[last_index]);
                  default:
                    break;
                }
              }
            }

            //segmented parallel reduction begin
            __syncthreads();
            shared_rows[threadIdx.x] = tmp.x;
            switch (option)
            {
              case 0:
              case 3:
                inter_results[threadIdx.x] = val;
                break;
              case 1:
                inter_results[threadIdx.x] = fabs(val);
                break;
              case 2:
                inter_results[threadIdx.x] = val * val;
              default:
                break;
            }
            T left = 0;
            __syncthreads();

            for (unsigned int stride = 1; stride < blockDim.x; stride *= 2)
            {
              left = (threadIdx.x >= stride && tmp.x == shared_rows[threadIdx.x - stride]) ? inter_results[threadIdx.x - stride] : 0;
              __syncthreads();
              switch (option)
              {
                case 0: //inf-norm
                case 3: //diagonal entry
                  inter_results[threadIdx.x] = max(inter_results[threadIdx.x], left);
                  break;

                case 1: //1-norm
                  inter_results[threadIdx.x] += left;
                  break;

                case 2: //2-norm
                  inter_results[threadIdx.x] += left;
                  break;

                default:
                  break;
              }
              __syncthreads();
            }
            //segmented parallel reduction end

            if (threadIdx.x != last_index &&
                shared_rows[threadIdx.x] != shared_rows[threadIdx.x + 1] &&
                inter_results[threadIdx.x] != 0)
            {
              result[tmp.x] = (option == 2) ? sqrt(inter_results[threadIdx.x]) : inter_results[threadIdx.x];
            }

            __syncthreads();
          } //for k

          if (threadIdx.x == last_index && inter_results[last_index] != 0)
            result[tmp.x] = (option == 2) ? sqrt(inter_results[last_index]) : inter_results[last_index];
        }

        template<typename ScalarType, unsigned int MAT_ALIGNMENT>
        void row_info(coordinate_matrix<ScalarType, MAT_ALIGNMENT> const & mat,
                      vector_base<ScalarType> & vec,
                      viennacl::linalg::detail::row_info_types info_selector)
        {
          coo_row_info_extractor<<<64, 128>>>(detail::cuda_arg<unsigned int>(mat.handle12().cuda_handle()),
                                               detail::cuda_arg<ScalarType>(mat.handle().cuda_handle()),
                                               detail::cuda_arg<unsigned int>(mat.handle3().cuda_handle()),
                                               detail::cuda_arg<ScalarType>(vec),
                                               static_cast<unsigned int>(info_selector)
                                              );
          VIENNACL_CUDA_LAST_ERROR_CHECK("coo_row_info_extractor");
        }

      } //namespace detail


      template <typename T>
      __global__ void coordinate_matrix_vec_mul_kernel(const unsigned int * coords, //(row_index, column_index)
                                                       const T * elements,
                                                       const unsigned int * group_boundaries,
                                                       const T * x,
                                                       unsigned int start_x,
                                                       unsigned int inc_x,
                                                             T * result,
                                                       unsigned int start_result,
                                                       unsigned int inc_result
                                                       )
      {
        __shared__ unsigned int shared_rows[128];
        __shared__ T inter_results[128];

        uint2 tmp;
        T val;
        uint last_index  = blockDim.x - 1;
        uint group_start = group_boundaries[blockIdx.x];
        uint group_end   = group_boundaries[blockIdx.x + 1];
        uint k_end = (group_end > group_start) ? 1 + (group_end - group_start - 1) / blockDim.x : 0;   // -1 in order to have correct behavior if group_end - group_start == j * blockDim.x

        uint local_index = 0;

        for (uint k = 0; k < k_end; ++k)
        {
          local_index = group_start + k * blockDim.x + threadIdx.x;

          tmp = (local_index < group_end) ? ((const uint2 *)coords)[local_index] : ::make_uint2(0, 0);
          val = (local_index < group_end) ? elements[local_index] * x[tmp.y * inc_x + start_x] : 0;

          //check for carry from previous loop run:
          if (threadIdx.x == 0 && k > 0)
          {
            if (tmp.x == shared_rows[last_index])
              val += inter_results[last_index];
            else
              result[shared_rows[last_index] * inc_result + start_result] += inter_results[last_index];
          }

          //segmented parallel reduction begin
          __syncthreads();
          shared_rows[threadIdx.x] = tmp.x;
          inter_results[threadIdx.x] = val;
          T left = 0;
          __syncthreads();

          for (unsigned int stride = 1; stride < blockDim.x; stride *= 2)
          {
            left = (threadIdx.x >= stride && tmp.x == shared_rows[threadIdx.x - stride]) ? inter_results[threadIdx.x - stride] : 0;
            __syncthreads();
            inter_results[threadIdx.x] += left;
            __syncthreads();
          }
          //segmented parallel reduction end

          if (threadIdx.x != last_index &&
              shared_rows[threadIdx.x] != shared_rows[threadIdx.x + 1] &&
              inter_results[threadIdx.x] != 0)
          {
            result[tmp.x * inc_result + start_result] += inter_results[threadIdx.x];
          }

          __syncthreads();
        } //for k

        if (threadIdx.x == last_index && inter_results[last_index] != 0)
          result[tmp.x * inc_result + start_result] += inter_results[last_index];
      }


      /** @brief Carries out matrix-vector multiplication with a coordinate_matrix
      *
      * Implementation of the convenience expression result = prod(mat, vec);
      *
      * @param mat    The matrix
      * @param vec    The vector
      * @param result The result vector
      */
      template<class ScalarType, unsigned int ALIGNMENT>
      void prod_impl(const viennacl::coordinate_matrix<ScalarType, ALIGNMENT> & mat,
                     const viennacl::vector_base<ScalarType> & vec,
                           viennacl::vector_base<ScalarType> & result)
      {
        result.clear();

        assert(vec.start() == 0 && vec.stride() == 1 && result.start() == 0 && result.stride() == 1 && bool("Vector strides unsupported for COO using CUDA"));

        coordinate_matrix_vec_mul_kernel<<<64, 128>>>(detail::cuda_arg<unsigned int>(mat.handle12().cuda_handle()),
                                                      detail::cuda_arg<ScalarType>(mat.handle().cuda_handle()),
                                                      detail::cuda_arg<unsigned int>(mat.handle3().cuda_handle()),
                                                      detail::cuda_arg<ScalarType>(vec),
                                                      static_cast<unsigned int>(vec.start()),
                                                      static_cast<unsigned int>(vec.stride()),
                                                      detail::cuda_arg<ScalarType>(result),
                                                      static_cast<unsigned int>(result.start()),
                                                      static_cast<unsigned int>(result.stride())
                                                     );
        VIENNACL_CUDA_LAST_ERROR_CHECK("coordinate_matrix_vec_mul_kernel");
      }

      template <typename ScalarType, typename NumericT>
      __global__ void coordinate_matrix_d_mat_mul_kernel(const unsigned int * coords, //(row_index, column_index)
                                                         const ScalarType * elements,
                                                         const unsigned int * group_boundaries,
                                                         const NumericT * d_mat,
                                                         unsigned int d_mat_row_start,
                                                         unsigned int d_mat_col_start,
                                                         unsigned int d_mat_row_inc,
                                                         unsigned int d_mat_col_inc,
                                                         unsigned int d_mat_row_size,
                                                         unsigned int d_mat_col_size,
                                                         unsigned int d_mat_internal_rows,
                                                         unsigned int d_mat_internal_cols,
                                                         NumericT * result,
                                                         unsigned int result_row_start,
                                                         unsigned int result_col_start,
                                                         unsigned int result_row_inc,
                                                         unsigned int result_col_inc,
                                                         unsigned int result_row_size,
                                                         unsigned int result_col_size,
                                                         unsigned int result_internal_rows,
                                                         unsigned int result_internal_cols) {

      }


      /** @brief Carries out Compressed Matrix(COO)-Dense Matrix multiplication
      *
      * Implementation of the convenience expression result = prod(sp_mat, d_mat);
      *
      * @param sp_mat     The Sparse Matrix (Coordinate format)
      * @param d_mat      The Dense Matrix
      * @param result     The Result Matrix
      */
      template<class ScalarType, unsigned int ALIGNMENT, class NumericT, typename F>
      void prod_impl(const viennacl::coordinate_matrix<ScalarType, ALIGNMENT> & sp_mat,
                     const viennacl::matrix_base<NumericT, F> & d_mat,
                           viennacl::matrix_base<NumericT, F> & result) {

        coordinate_matrix_d_mat_mul_kernel<<<128, 128>>>(detail::cuda_arg<unsigned int>(sp_mat.handle12().cuda_handle()),
                                                         detail::cuda_arg<ScalarType>(sp_mat.handle().cuda_handle()),
                                                         detail::cuda_arg<unsigned int>(sp_mat.handle3().cuda_handle()),

                                                         detail::cuda_arg<NumericT>(d_mat),
                                                         static_cast<unsigned int>(viennacl::traits::start1(d_mat)),         static_cast<unsigned int>(viennacl::traits::start2(d_mat)),
                                                         static_cast<unsigned int>(viennacl::traits::stride1(d_mat)),        static_cast<unsigned int>(viennacl::traits::stride2(d_mat)),
                                                         static_cast<unsigned int>(viennacl::traits::size1(d_mat)),          static_cast<unsigned int>(viennacl::traits::size2(d_mat)),
                                                         static_cast<unsigned int>(viennacl::traits::internal_size1(d_mat)), static_cast<unsigned int>(viennacl::traits::internal_size2(d_mat)),

                                                         detail::cuda_arg<NumericT>(result),
                                                         static_cast<unsigned int>(viennacl::traits::start1(result)),         static_cast<unsigned int>(viennacl::traits::start2(result)),
                                                         static_cast<unsigned int>(viennacl::traits::stride1(result)),        static_cast<unsigned int>(viennacl::traits::stride2(result)),
                                                         static_cast<unsigned int>(viennacl::traits::size1(result)),          static_cast<unsigned int>(viennacl::traits::size2(result)),
                                                         static_cast<unsigned int>(viennacl::traits::internal_size1(result)), static_cast<unsigned int>(viennacl::traits::internal_size2(result))
                                                         );

      }

      template <typename ScalarType, typename NumericT>
      __global__ void coordinate_matrix_d_tr_mat_mul_kernel(const unsigned int * coords, //(row_index, column_index)
                                                           const ScalarType * elements,
                                                           const unsigned int * group_boundaries,
                                                           const NumericT * d_mat,
                                                           unsigned int d_mat_row_start,
                                                           unsigned int d_mat_col_start,
                                                           unsigned int d_mat_row_inc,
                                                           unsigned int d_mat_col_inc,
                                                           unsigned int d_mat_row_size,
                                                           unsigned int d_mat_col_size,
                                                           unsigned int d_mat_internal_rows,
                                                           unsigned int d_mat_internal_cols,
                                                           NumericT * result,
                                                           unsigned int result_row_start,
                                                           unsigned int result_col_start,
                                                           unsigned int result_row_inc,
                                                           unsigned int result_col_inc,
                                                           unsigned int result_row_size,
                                                           unsigned int result_col_size,
                                                           unsigned int result_internal_rows,
                                                           unsigned int result_internal_cols) {

        VIENNACL_CUDA_LAST_ERROR_CHECK("coordinate_matrix_d_mat_mul_kernel");
      }

      /** @brief Carries out Compressed Matrix(COO)-Dense Transposed Matrix multiplication
      *
      * Implementation of the convenience expression result = prod(sp_mat, trans(d_mat));
      *
      * @param sp_mat     The Sparse Matrix (Coordinate format)
      * @param d_mat      The Dense Transposed Matrix
      * @param result     The Result Matrix
      */
      template<class ScalarType, unsigned int ALIGNMENT, class NumericT, typename F>
      void prod_impl(const viennacl::coordinate_matrix<ScalarType, ALIGNMENT> & sp_mat,
                     const viennacl::matrix_expression< const viennacl::matrix_base<NumericT, F>,
                                                        const viennacl::matrix_base<NumericT, F>,
                                                        viennacl::op_trans > & d_mat,
                           viennacl::matrix_base<NumericT, F> & result) {

        coordinate_matrix_d_mat_mul_kernel<<<128, 128>>>(detail::cuda_arg<unsigned int>(sp_mat.handle12().cuda_handle()),
                                                         detail::cuda_arg<ScalarType>(sp_mat.handle().cuda_handle()),
                                                         detail::cuda_arg<unsigned int>(sp_mat.handle3().cuda_handle()),

                                                         detail::cuda_arg<NumericT>(d_mat.lhs()),
                                                         static_cast<unsigned int>(viennacl::traits::start1(d_mat.lhs())),         static_cast<unsigned int>(viennacl::traits::start2(d_mat.lhs())),
                                                         static_cast<unsigned int>(viennacl::traits::stride1(d_mat.lhs())),        static_cast<unsigned int>(viennacl::traits::stride2(d_mat.lhs())),
                                                         static_cast<unsigned int>(viennacl::traits::size1(d_mat.lhs())),          static_cast<unsigned int>(viennacl::traits::size2(d_mat.lhs())),
                                                         static_cast<unsigned int>(viennacl::traits::internal_size1(d_mat.lhs())), static_cast<unsigned int>(viennacl::traits::internal_size2(d_mat.lhs())),

                                                         detail::cuda_arg<NumericT>(result),
                                                         static_cast<unsigned int>(viennacl::traits::start1(result)),         static_cast<unsigned int>(viennacl::traits::start2(result)),
                                                         static_cast<unsigned int>(viennacl::traits::stride1(result)),        static_cast<unsigned int>(viennacl::traits::stride2(result)),
                                                         static_cast<unsigned int>(viennacl::traits::size1(result)),          static_cast<unsigned int>(viennacl::traits::size2(result)),
                                                         static_cast<unsigned int>(viennacl::traits::internal_size1(result)), static_cast<unsigned int>(viennacl::traits::internal_size2(result))
                                                        );

        VIENNACL_CUDA_LAST_ERROR_CHECK("coordinate_matrix_d_tr_mat_mul_kernel");
      }


      //
      // ELL Matrix
      //

      template <typename T>
      __global__ void ell_matrix_vec_mul_kernel(const unsigned int * coords,
                                                const T * elements,
                                                const T * x,
                                                unsigned int start_x,
                                                unsigned int inc_x,
                                                      T * result,
                                                unsigned int start_result,
                                                unsigned int inc_result,
                                                unsigned int row_num,
                                                unsigned int col_num,
                                                unsigned int internal_row_num,
                                                unsigned int items_per_row,
                                                unsigned int aligned_items_per_row
                                               )
      {
        uint glb_id = blockDim.x * blockIdx.x + threadIdx.x;
        uint glb_sz = gridDim.x * blockDim.x;

        for(uint row_id = glb_id; row_id < row_num; row_id += glb_sz)
        {
          T sum = 0;

          uint offset = row_id;
          for(uint item_id = 0; item_id < items_per_row; item_id++, offset += internal_row_num)
          {
            T val = elements[offset];

            if(val != (T)0)
            {
              int col = coords[offset];
              sum += (x[col * inc_x + start_x] * val);
            }
          }

          result[row_id * inc_result + start_result] = sum;
        }
      }


      /** @brief Carries out matrix-vector multiplication with a ell_matrix
      *
      * Implementation of the convenience expression result = prod(mat, vec);
      *
      * @param mat    The matrix
      * @param vec    The vector
      * @param result The result vector
      */
      template<class ScalarType, unsigned int ALIGNMENT>
      void prod_impl(const viennacl::ell_matrix<ScalarType, ALIGNMENT> & mat,
                     const viennacl::vector_base<ScalarType> & vec,
                           viennacl::vector_base<ScalarType> & result)
      {
        ell_matrix_vec_mul_kernel<<<256, 128>>>(detail::cuda_arg<unsigned int>(mat.handle2().cuda_handle()),
                                                detail::cuda_arg<ScalarType>(mat.handle().cuda_handle()),
                                                detail::cuda_arg<ScalarType>(vec),
                                                static_cast<unsigned int>(vec.start()),
                                                static_cast<unsigned int>(vec.stride()),
                                                detail::cuda_arg<ScalarType>(result),
                                                static_cast<unsigned int>(result.start()),
                                                static_cast<unsigned int>(result.stride()),
                                                static_cast<unsigned int>(mat.size1()),
                                                static_cast<unsigned int>(mat.size2()),
                                                static_cast<unsigned int>(mat.internal_size1()),
                                                static_cast<unsigned int>(mat.maxnnz()),
                                                static_cast<unsigned int>(mat.internal_maxnnz())
                                               );
        VIENNACL_CUDA_LAST_ERROR_CHECK("ell_matrix_vec_mul_kernel");
      }

      template <typename ScalarType, typename NumericT >
      __global__ void ell_matrix_d_mat_mul_kernel(const unsigned int * sp_mat_coords,
                                                  const ScalarType * sp_mat_elements,
                                                  unsigned int sp_mat_row_num,
                                                  unsigned int sp_mat_col_num,
                                                  unsigned int sp_mat_internal_row_num,
                                                  unsigned int sp_mat_items_per_row,
                                                  unsigned int sp_mat_aligned_items_per_row,
                                                  const NumericT * d_mat,
                                                  unsigned int d_mat_row_start,
                                                  unsigned int d_mat_col_start,
                                                  unsigned int d_mat_row_inc,
                                                  unsigned int d_mat_col_inc,
                                                  unsigned int d_mat_row_size,
                                                  unsigned int d_mat_col_size,
                                                  unsigned int d_mat_internal_rows,
                                                  unsigned int d_mat_internal_cols,
                                                  NumericT * result,
                                                  unsigned int result_row_start,
                                                  unsigned int result_col_start,
                                                  unsigned int result_row_inc,
                                                  unsigned int result_col_inc,
                                                  unsigned int result_row_size,
                                                  unsigned int result_col_size,
                                                  unsigned int result_internal_rows,
                                                  unsigned int result_internal_cols) {


        uint glb_id = blockDim.x * blockIdx.x + threadIdx.x;
        uint glb_sz = gridDim.x * blockDim.x;
        for( uint rc = glb_id; rc < (sp_mat_row_num * d_mat_col_size); rc += glb_sz) {
          uint row = rc % sp_mat_row_num;
          uint col = rc / sp_mat_row_num;

          uint offset = row;
          NumericT r = (NumericT)0;

          for(uint k = 0; k < sp_mat_items_per_row; k++, offset += sp_mat_internal_row_num) {

            unsigned int j = sp_mat_coords[offset];
            NumericT x = static_cast<NumericT>(sp_mat_elements[offset]);

            if(x != (NumericT)0) {

                NumericT y = d_mat[ (d_mat_row_start + j * d_mat_row_inc) * d_mat_internal_cols +
                                     d_mat_col_start + col * d_mat_col_inc ];

                r += x*y;
              }
            }
          result [ (result_row_start + row * result_row_inc) * result_internal_cols +
                    result_col_start + col * result_col_inc ] = r;
        }

      }

      /** @brief Carries out Sparse Matrix(ELL)-Dense Matrix multiplication
      *
      * Implementation of the convenience expression result = prod(sp_mat, d_mat);
      * sp_mat being in ELL format
      *
      * @param sp_mat     The sparse matrix (ELL)
      * @param d_mat      The dense matrix
      * @param result     The result matrix
      */
      template<class ScalarType, unsigned int ALIGNMENT, class NumericT, typename F >
      void prod_impl(const viennacl::ell_matrix<ScalarType, ALIGNMENT> & sp_mat,
                     const viennacl::matrix_base<NumericT, F> & d_mat,
                           viennacl::matrix_base<NumericT, F> & result) {

        ell_matrix_d_mat_mul_kernel<<<128, 128>>>(detail::cuda_arg<unsigned int>(sp_mat.handle2().cuda_handle()),
                                                  detail::cuda_arg<ScalarType>(sp_mat.handle().cuda_handle()),
                                                  static_cast<unsigned int>(sp_mat.size1()),
                                                  static_cast<unsigned int>(sp_mat.size2()),
                                                  static_cast<unsigned int>(sp_mat.internal_size1()),
                                                  static_cast<unsigned int>(sp_mat.maxnnz()),
                                                  static_cast<unsigned int>(sp_mat.internal_maxnnz()),
                                                  detail::cuda_arg<NumericT>(d_mat),
                                                  static_cast<unsigned int>(viennacl::traits::start1(d_mat)),         static_cast<unsigned int>(viennacl::traits::start2(d_mat)),
                                                  static_cast<unsigned int>(viennacl::traits::stride1(d_mat)),        static_cast<unsigned int>(viennacl::traits::stride2(d_mat)),
                                                  static_cast<unsigned int>(viennacl::traits::size1(d_mat)),          static_cast<unsigned int>(viennacl::traits::size2(d_mat)),
                                                  static_cast<unsigned int>(viennacl::traits::internal_size1(d_mat)), static_cast<unsigned int>(viennacl::traits::internal_size2(d_mat)),

                                                  detail::cuda_arg<NumericT>(result),
                                                  static_cast<unsigned int>(viennacl::traits::start1(result)),         static_cast<unsigned int>(viennacl::traits::start2(result)),
                                                  static_cast<unsigned int>(viennacl::traits::stride1(result)),        static_cast<unsigned int>(viennacl::traits::stride2(result)),
                                                  static_cast<unsigned int>(viennacl::traits::size1(result)),          static_cast<unsigned int>(viennacl::traits::size2(result)),
                                                  static_cast<unsigned int>(viennacl::traits::internal_size1(result)), static_cast<unsigned int>(viennacl::traits::internal_size2(result))
                                               );
        VIENNACL_CUDA_LAST_ERROR_CHECK("ell_matrix_d_mat_mul_kernel");
      }

      template <typename ScalarType, typename NumericT >
      __global__ void ell_matrix_d_tr_mat_mul_kernel(const unsigned int * sp_mat_coords,
                                                  const ScalarType * sp_mat_elements,
                                                  unsigned int sp_mat_row_num,
                                                  unsigned int sp_mat_col_num,
                                                  unsigned int sp_mat_internal_row_num,
                                                  unsigned int sp_mat_items_per_row,
                                                  unsigned int sp_mat_aligned_items_per_row,
                                                  const NumericT * d_mat,
                                                  unsigned int d_mat_row_start,
                                                  unsigned int d_mat_col_start,
                                                  unsigned int d_mat_row_inc,
                                                  unsigned int d_mat_col_inc,
                                                  unsigned int d_mat_row_size,
                                                  unsigned int d_mat_col_size,
                                                  unsigned int d_mat_internal_rows,
                                                  unsigned int d_mat_internal_cols,
                                                  NumericT * result,
                                                  unsigned int result_row_start,
                                                  unsigned int result_col_start,
                                                  unsigned int result_row_inc,
                                                  unsigned int result_col_inc,
                                                  unsigned int result_row_size,
                                                  unsigned int result_col_size,
                                                  unsigned int result_internal_rows,
                                                  unsigned int result_internal_cols) {


        uint glb_id = blockDim.x * blockIdx.x + threadIdx.x;
        uint glb_sz = gridDim.x * blockDim.x;
        for( uint rc = glb_id; rc < (sp_mat_row_num * d_mat_row_size); rc += glb_sz) {
          uint row = rc % sp_mat_row_num;
          uint col = rc / sp_mat_row_num;

          uint offset = row;
          NumericT r = (NumericT)0;

          for(uint k = 0; k < sp_mat_items_per_row; k++, offset += sp_mat_internal_row_num) {

            unsigned int j = sp_mat_coords[offset];
            NumericT x = static_cast<NumericT>(sp_mat_elements[offset]);

            if(x != (NumericT)0) {

                NumericT y = d_mat[ (d_mat_row_start + col * d_mat_row_inc) * d_mat_internal_cols +
                                     d_mat_col_start + j * d_mat_col_inc ];

                r += x*y;
              }
            }
          result [ (result_row_start + row * result_row_inc) * result_internal_cols +
                    result_col_start + col * result_col_inc ] = r;
        }

      }

      /** @brief Carries out Sparse Matrix(ELL)-Dense Transposed Matrix multiplication
      *
      * Implementation of the convenience expression result = prod(sp_mat, trans(d_mat));
      * sp_mat being in ELL format
      *
      * @param sp_mat     The sparse matrix (ELL)
      * @param d_mat      The dense matrix
      * @param result     The result matrix
      */
      template<class ScalarType, unsigned int ALIGNMENT, class NumericT, typename F >
      void prod_impl(const viennacl::ell_matrix<ScalarType, ALIGNMENT> & sp_mat,
                     const viennacl::matrix_expression< const viennacl::matrix_base<NumericT, F>,
                                                        const viennacl::matrix_base<NumericT, F>,
                                                        viennacl::op_trans > & d_mat,
                           viennacl::matrix_base<NumericT, F> & result) {

        ell_matrix_d_tr_mat_mul_kernel<<<128, 128>>>(detail::cuda_arg<unsigned int>(sp_mat.handle2().cuda_handle()),
                                                     detail::cuda_arg<ScalarType>(sp_mat.handle().cuda_handle()),
                                                     static_cast<unsigned int>(sp_mat.size1()),
                                                     static_cast<unsigned int>(sp_mat.size2()),
                                                     static_cast<unsigned int>(sp_mat.internal_size1()),
                                                     static_cast<unsigned int>(sp_mat.maxnnz()),
                                                     static_cast<unsigned int>(sp_mat.internal_maxnnz()),

                                                     detail::cuda_arg<NumericT>(d_mat.lhs()),
                                                     static_cast<unsigned int>(viennacl::traits::start1(d_mat.lhs())),         static_cast<unsigned int>(viennacl::traits::start2(d_mat.lhs())),
                                                     static_cast<unsigned int>(viennacl::traits::stride1(d_mat.lhs())),        static_cast<unsigned int>(viennacl::traits::stride2(d_mat.lhs())),
                                                     static_cast<unsigned int>(viennacl::traits::size1(d_mat.lhs())),          static_cast<unsigned int>(viennacl::traits::size2(d_mat.lhs())),
                                                     static_cast<unsigned int>(viennacl::traits::internal_size1(d_mat.lhs())), static_cast<unsigned int>(viennacl::traits::internal_size2(d_mat.lhs())),

                                                     detail::cuda_arg<NumericT>(result),
                                                     static_cast<unsigned int>(viennacl::traits::start1(result)),         static_cast<unsigned int>(viennacl::traits::start2(result)),
                                                     static_cast<unsigned int>(viennacl::traits::stride1(result)),        static_cast<unsigned int>(viennacl::traits::stride2(result)),
                                                     static_cast<unsigned int>(viennacl::traits::size1(result)),          static_cast<unsigned int>(viennacl::traits::size2(result)),
                                                     static_cast<unsigned int>(viennacl::traits::internal_size1(result)), static_cast<unsigned int>(viennacl::traits::internal_size2(result))
                                               );

        VIENNACL_CUDA_LAST_ERROR_CHECK("ell_matrix_d_tr_mat_mul_kernel");
      }

      //
      // Hybrid Matrix
      //


      template <typename T>
      __global__ void hyb_matrix_vec_mul_kernel(const unsigned int * ell_coords,
                                                const T * ell_elements,
                                                const unsigned int * csr_rows,
                                                const unsigned int * csr_cols,
                                                const T * csr_elements,
                                                const T * x,
                                                unsigned int start_x,
                                                unsigned int inc_x,
                                                      T * result,
                                                unsigned int start_result,
                                                unsigned int inc_result,
                                                unsigned int row_num,
                                                unsigned int internal_row_num,
                                                unsigned int items_per_row,
                                                unsigned int aligned_items_per_row
                                               )
      {
        uint glb_id = blockDim.x * blockIdx.x + threadIdx.x;
        uint glb_sz = gridDim.x * blockDim.x;

        for(uint row_id = glb_id; row_id < row_num; row_id += glb_sz)
        {
          T sum = 0;

          uint offset = row_id;
          for(uint item_id = 0; item_id < items_per_row; item_id++, offset += internal_row_num)
          {
            T val = ell_elements[offset];


            if(val != 0.0f)
            {
              int col = ell_coords[offset];
              sum += (x[col * inc_x + start_x] * val);
            }
          }

          uint col_begin = csr_rows[row_id];
          uint col_end   = csr_rows[row_id + 1];

          for(uint item_id = col_begin; item_id < col_end; item_id++)
          {
            sum += (x[csr_cols[item_id] * inc_x + start_x] * csr_elements[item_id]);
          }

          result[row_id * inc_result + start_result] = sum;
        }
      }



      /** @brief Carries out matrix-vector multiplication with a hyb_matrix
      *
      * Implementation of the convenience expression result = prod(mat, vec);
      *
      * @param mat    The matrix
      * @param vec    The vector
      * @param result The result vector
      */
      template<class ScalarType, unsigned int ALIGNMENT>
      void prod_impl(const viennacl::hyb_matrix<ScalarType, ALIGNMENT> & mat,
                     const viennacl::vector_base<ScalarType> & vec,
                           viennacl::vector_base<ScalarType> & result)
      {
        hyb_matrix_vec_mul_kernel<<<256, 128>>>(detail::cuda_arg<unsigned int>(mat.handle2().cuda_handle()),
                                                detail::cuda_arg<ScalarType>(mat.handle().cuda_handle()),
                                                detail::cuda_arg<unsigned int>(mat.handle3().cuda_handle()),
                                                detail::cuda_arg<unsigned int>(mat.handle4().cuda_handle()),
                                                detail::cuda_arg<ScalarType>(mat.handle5().cuda_handle()),
                                                detail::cuda_arg<ScalarType>(vec),
                                                static_cast<unsigned int>(vec.start()),
                                                static_cast<unsigned int>(vec.stride()),
                                                detail::cuda_arg<ScalarType>(result),
                                                static_cast<unsigned int>(result.start()),
                                                static_cast<unsigned int>(result.stride()),
                                                static_cast<unsigned int>(mat.size1()),
                                                static_cast<unsigned int>(mat.internal_size1()),
                                                static_cast<unsigned int>(mat.ell_nnz()),
                                                static_cast<unsigned int>(mat.internal_ellnnz())
                                               );
        VIENNACL_CUDA_LAST_ERROR_CHECK("hyb_matrix_vec_mul_kernel");
      }


    } // namespace opencl
  } //namespace linalg
} //namespace viennacl


#endif
