#ifndef VIENNACL_LINALG_CUDA_SPARSE_MATRIX_OPERATIONS_HPP_
#define VIENNACL_LINALG_CUDA_SPARSE_MATRIX_OPERATIONS_HPP_

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

/** @file viennacl/linalg/cuda/sparse_matrix_operations.hpp
    @brief Implementations of operations using sparse matrices using CUDA
*/

#include "viennacl/forwards.h"
#include "viennacl/scalar.hpp"
#include "viennacl/vector.hpp"
#include "viennacl/tools/tools.hpp"
#include "viennacl/linalg/cuda/common.hpp"

namespace viennacl
{
  namespace linalg
  {
    namespace cuda
    {
      //
      // Compressed matrix
      //
      
      template <typename T>
      __global__ void compressed_matrix_vec_mul_kernel(
                const unsigned int * row_indices,
                const unsigned int * column_indices, 
                const T * elements,
                const T * vector,  
                T * result,
                unsigned int size) 
      { 
        for (unsigned int row  = blockDim.x * blockIdx.x + threadIdx.x;
                          row  < size;
                          row += gridDim.x * blockDim.x)
        {
          T dot_prod = (T)0;
          unsigned int row_end = row_indices[row+1];
          for (unsigned int i = row_indices[row]; i < row_end; ++i)
            dot_prod += elements[i] * vector[column_indices[i]];
          result[row] = dot_prod;
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
      template<class ScalarType, unsigned int ALIGNMENT, unsigned int VECTOR_ALIGNMENT>
      void prod_impl(const viennacl::compressed_matrix<ScalarType, ALIGNMENT> & mat, 
                     const viennacl::vector<ScalarType, VECTOR_ALIGNMENT> & vec,
                           viennacl::vector<ScalarType, VECTOR_ALIGNMENT> & result)
      {
        compressed_matrix_vec_mul_kernel<<<128, 128>>>(detail::cuda_arg<unsigned int>(mat.handle1().cuda_handle()),
                                                       detail::cuda_arg<unsigned int>(mat.handle2().cuda_handle()),
                                                       detail::cuda_arg<ScalarType>(mat.handle().cuda_handle()),
                                                       detail::cuda_arg<ScalarType>(vec),
                                                       detail::cuda_arg<ScalarType>(result),
                                                       static_cast<unsigned int>(mat.size1())
                                                      );
        VIENNACL_CUDA_LAST_ERROR_CHECK("compressed_matrix_vec_mul_kernel");
      }

      
      
      template <typename T>
      __global__ void csr_lu_forward_kernel(
                const unsigned int * row_indices,
                const unsigned int * column_indices, 
                const T * elements,
                      T * vector,
                unsigned int size) 
      {
        __shared__  unsigned int col_index_buffer[128];
        __shared__  T element_buffer[128];
        __shared__  T vector_buffer[128];
        
        unsigned int nnz = row_indices[size];
        unsigned int current_row = 0;
        unsigned int row_at_window_start = 0;
        T current_vector_entry = vector[0];
        unsigned int loop_end = (nnz / blockDim.x + 1) * blockDim.x;
        unsigned int next_row = row_indices[1];
        
        for (unsigned int i = threadIdx.x; i < loop_end; i += blockDim.x)
        {
          //load into shared memory (coalesced access):
          if (i < nnz)
          {
            element_buffer[threadIdx.x] = elements[i];
            unsigned int tmp = column_indices[i];
            col_index_buffer[threadIdx.x] = tmp;
            vector_buffer[threadIdx.x] = vector[tmp];
          }
          
          __syncthreads();
          
          //now a single thread does the remaining work in shared memory:
          if (threadIdx.x == 0)
          {
            // traverse through all the loaded data:
            for (unsigned int k=0; k<blockDim.x; ++k)
            {
              if (current_row < size && i+k == next_row) //current row is finished. Write back result
              {
                vector[current_row] = current_vector_entry;
                ++current_row;
                if (current_row < size) //load next row's data
                {
                  next_row = row_indices[current_row+1];
                  current_vector_entry = vector[current_row];
                }
              }
              
              if (current_row < size && col_index_buffer[k] < current_row) //substitute
              {
                if (col_index_buffer[k] < row_at_window_start) //use recently computed results
                  current_vector_entry -= element_buffer[k] * vector_buffer[k];
                else if (col_index_buffer[k] < current_row) //use buffered data
                  current_vector_entry -= element_buffer[k] * vector[col_index_buffer[k]];
              }

            } // for k
            
            row_at_window_start = current_row;
          } // if (get_local_id(0) == 0)
          
          __syncthreads();
        } //for i
      }


      /** @brief Carries out triangular inplace solves
      *
      * @param mat    The matrix
      * @param vec    The vector
      * @param result The result vector
      */
      template<typename SparseMatrixType, class ScalarType, unsigned int ALIGNMENT, typename SOLVERTAG>
      typename viennacl::enable_if< viennacl::is_any_sparse_matrix<SparseMatrixType>::value>::type
      inplace_solve(const SparseMatrixType & mat, 
                    viennacl::vector<ScalarType, ALIGNMENT> & vec,
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


      template <typename T>
      __global__ void csr_trans_lu_forward_kernel2(
                const unsigned int * row_indices,
                const unsigned int * column_indices, 
                const T * elements,
                      T * vector,
                unsigned int size) 
      {
        for (unsigned int row = 0; row < size; ++row) 
        { 
          T result_entry = vector[row]; 
          
          unsigned int row_start = row_indices[row]; 
          unsigned int row_stop  = row_indices[row + 1];
          for (unsigned int entry_index = row_start + threadIdx.x; entry_index < row_stop; entry_index += blockDim.x) 
          {
            unsigned int col_index = column_indices[entry_index];
            if (col_index > row)
              vector[col_index] -= result_entry * elements[entry_index]; 
          }
          
          __syncthreads();
        } 
      }      
      
      template <typename T>
      __global__ void csr_trans_lu_forward_kernel(
                const unsigned int * row_indices,
                const unsigned int * column_indices, 
                const T * elements,
                      T * vector,
                unsigned int size) 
      {
        __shared__  unsigned int row_index_lookahead[256];
        __shared__  unsigned int row_index_buffer[256];
        
        unsigned int row_index;
        unsigned int col_index;
        T matrix_entry;
        unsigned int nnz = row_indices[size];
        unsigned int row_at_window_start = 0;
        unsigned int row_at_window_end = 0;
        unsigned int loop_end = ( (nnz - 1) / blockDim.x + 1) * blockDim.x;
        
        for (unsigned int i = threadIdx.x; i < loop_end; i += blockDim.x)
        {
          col_index    = (i < nnz) ? column_indices[i] : 0;
          matrix_entry = (i < nnz) ? elements[i]       : 0;
          row_index_lookahead[threadIdx.x] = (row_at_window_start + threadIdx.x < size) ? row_indices[row_at_window_start + threadIdx.x] : size - 1;

          __syncthreads();
          
          if (i < nnz)
          {
            unsigned int row_index_inc = 0;
            while (i >= row_index_lookahead[row_index_inc + 1])
              ++row_index_inc;
            row_index = row_at_window_start + row_index_inc;
            row_index_buffer[threadIdx.x] = row_index;
          }
          else
          {
            row_index = size+1;
            row_index_buffer[threadIdx.x] = size - 1;
          }
          
          __syncthreads();
          
          row_at_window_start = row_index_buffer[0];
          row_at_window_end   = row_index_buffer[blockDim.x - 1];
          
          //forward elimination
          for (unsigned int row = row_at_window_start; row <= row_at_window_end; ++row) 
          { 
            T result_entry = vector[row];
            
            if ( (row_index == row) && (col_index > row) )
              vector[col_index] -= result_entry * matrix_entry; 

            __syncthreads();
          }
          
          row_at_window_start = row_at_window_end;
        }
          
      }
      
      
      /** @brief Carries out triangular inplace solves
      *
      * @param mat    The matrix
      * @param vec    The vector
      * @param result The result vector
      */
      template<typename SparseMatrixType, class ScalarType, unsigned int ALIGNMENT>
      typename viennacl::enable_if< viennacl::is_any_sparse_matrix<SparseMatrixType>::value>::type
      inplace_solve(const matrix_expression<const SparseMatrixType, const SparseMatrixType, op_trans> & mat, 
                    viennacl::vector<ScalarType, ALIGNMENT> & vec,
                    viennacl::linalg::unit_lower_tag)
      {
        std::cout << "size: " << mat.lhs().size1() << std::endl;
        csr_trans_lu_forward_kernel<<<1, 128>>>(detail::cuda_arg<unsigned int>(mat.lhs().handle1().cuda_handle()),
                                                detail::cuda_arg<unsigned int>(mat.lhs().handle2().cuda_handle()),
                                                detail::cuda_arg<ScalarType>(mat.lhs().handle().cuda_handle()),
                                                detail::cuda_arg<ScalarType>(vec),
                                                static_cast<unsigned int>(mat.lhs().size1())
                                               );
        VIENNACL_CUDA_LAST_ERROR_CHECK("csr_trans_lu_forward_kernel");
      }
      
      
      
      
      
      
      
      
      
      
      //////////////////////// backward solve ///////////////////////////
      
      template <typename T>
      __global__ void csr_lu_backward_kernel(
                const unsigned int * row_indices,
                const unsigned int * column_indices, 
                const T * elements,
                      T * vector,
                unsigned int size) 
      {
        __shared__  unsigned int col_index_buffer[128];
        __shared__  T element_buffer[128];
        __shared__  T vector_buffer[128];
        
        unsigned int nnz = row_indices[size];
        unsigned int current_row = size-1;
        unsigned int row_at_window_start = size-1;
        T current_vector_entry = vector[size-1];
        unsigned int loop_end = ( (nnz - 1) / blockDim.x) * blockDim.x;
        unsigned int next_row = row_indices[size-1];
        
        unsigned int i = loop_end + threadIdx.x;
        while (1)
        {
          //load into shared memory (coalesced access):
          if (i < nnz)
          {
            element_buffer[threadIdx.x] = elements[i];
            unsigned int tmp = column_indices[i];
            col_index_buffer[threadIdx.x] = tmp;
            vector_buffer[threadIdx.x] = vector[tmp];
          }
          
          __syncthreads();
          
          //now a single thread does the remaining work in shared memory:
          if (threadIdx.x == 0)
          {
            // traverse through all the loaded data from back to front:
            for (unsigned int k2=0; k2<blockDim.x; ++k2)
            {
              unsigned int k = (blockDim.x - k2) - 1;
              
              if (i+k >= nnz)
                continue;
              
              if (col_index_buffer[k] > row_at_window_start) //use recently computed results
                current_vector_entry -= element_buffer[k] * vector_buffer[k];
              else if (col_index_buffer[k] > current_row) //use buffered data
                current_vector_entry -= element_buffer[k] * vector[col_index_buffer[k]];
              else if (col_index_buffer[k] == current_row)
                current_vector_entry /= element_buffer[k];
              
              if (i+k == next_row) //current row is finished. Write back result
              {
                vector[current_row] = current_vector_entry;
                if (current_row > 0) //load next row's data
                {
                  --current_row;
                  next_row = row_indices[current_row];
                  current_vector_entry = vector[current_row];
                }
              }
              
              
            } // for k
            
            row_at_window_start = current_row;
          } // if (get_local_id(0) == 0)
          
          __syncthreads();
          
          if (i < blockDim.x)
            break;
          
          i -= blockDim.x;
        } //for i
      }


      /** @brief Carries out triangular inplace solves
      *
      * @param mat    The matrix
      * @param vec    The vector
      * @param result The result vector
      */
      template<typename SparseMatrixType, class ScalarType, unsigned int ALIGNMENT>
      typename viennacl::enable_if< viennacl::is_any_sparse_matrix<SparseMatrixType>::value>::type
      inplace_solve(const SparseMatrixType & mat, 
                    viennacl::vector<ScalarType, ALIGNMENT> & vec,
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
      

      
      template <typename T>
      __global__ void csr_trans_lu_backward_kernel(
                const unsigned int * row_indices,
                const unsigned int * column_indices, 
                const T * elements,
                const T * diagonal_entries,
                      T * vector,
                unsigned int size) 
      {
        T result_entry = 0;
        
        //backward elimination, using U and D: 
        for (unsigned int row2 = 0; row2 < size; ++row2) 
        { 
          unsigned int row = (size - row2) - 1;
          result_entry = vector[row] / diagonal_entries[row]; 
          
          unsigned int row_start = row_indices[row]; 
          unsigned int row_stop  = row_indices[row + 1];
          for (unsigned int entry_index = row_start + threadIdx.x; entry_index < row_stop; ++entry_index) 
          {
            unsigned int col_index = column_indices[entry_index];
            if (col_index < row)
              vector[col_index] -= result_entry * elements[entry_index]; 
          }
          
          __syncthreads();
          
          if (threadIdx.x == 0)
            vector[row] = result_entry;
        } 
      }
      
      
      /** @brief Carries out triangular inplace solves
      *
      * @param mat    The matrix
      * @param vec    The vector
      * @param result The result vector
      */
      template<typename SparseMatrixType, class ScalarType, unsigned int ALIGNMENT>
      typename viennacl::enable_if< viennacl::is_any_sparse_matrix<SparseMatrixType>::value>::type
      inplace_solve(const matrix_expression<const SparseMatrixType, const SparseMatrixType, op_trans> & mat, 
                    viennacl::vector<ScalarType, ALIGNMENT> & vec,
                    viennacl::linalg::upper_tag)
      {
        csr_trans_lu_backward_kernel<<<1, 128>>>(detail::cuda_arg<unsigned int>(mat.handle1().cuda_handle()),
                                                 detail::cuda_arg<unsigned int>(mat.handle2().cuda_handle()),
                                                 detail::cuda_arg<ScalarType>(mat.handle().cuda_handle()),
                                                 detail::cuda_arg<ScalarType>(vec),
                                                 static_cast<unsigned int>(mat.size1())
                                                );
        VIENNACL_CUDA_LAST_ERROR_CHECK("csr_trans_lu_backward_kernel");
      }
      
      
      
      //
      // Coordinate Matrix
      //
      
      template <typename T>
      __device__ void coordinate_matrix_segmented_parallel_reduction(unsigned int row, 
                                                                     T val, 
                                                                     unsigned int * shared_rows, 
                                                                     T * inter_results) 
      { 
        shared_rows[threadIdx.x] = row; 
        inter_results[threadIdx.x] = val; 
        T left = 0;
      
        __syncthreads();
        if( threadIdx.x >=  1 && row == shared_rows[threadIdx.x -  1] ) { left = inter_results[threadIdx.x -  1]; }  
        __syncthreads(); 
        inter_results[threadIdx.x] += left; left = 0;
        __syncthreads();

        if( threadIdx.x >=  2 && row == shared_rows[threadIdx.x -  2] ) { left = inter_results[threadIdx.x -  2]; } 
        __syncthreads();
        inter_results[threadIdx.x] += left; left = 0;
        __syncthreads();

        if( threadIdx.x >=  4 && row == shared_rows[threadIdx.x -  4] ) { left = inter_results[threadIdx.x -  4]; } 
        __syncthreads();
        inter_results[threadIdx.x] += left; left = 0;
        __syncthreads();

        if( threadIdx.x >=  8 && row == shared_rows[threadIdx.x -  8] ) { left = inter_results[threadIdx.x -  8]; } 
        __syncthreads();
        inter_results[threadIdx.x] += left; left = 0;
        __syncthreads();

        if( threadIdx.x >= 16 && row == shared_rows[threadIdx.x - 16] ) { left = inter_results[threadIdx.x - 16]; } 
        __syncthreads();
        inter_results[threadIdx.x] += left; left = 0;
        __syncthreads();

        if( threadIdx.x >= 32 && row == shared_rows[threadIdx.x - 32] ) { left = inter_results[threadIdx.x - 32]; } 
        __syncthreads();
        inter_results[threadIdx.x] += left; left = 0;
        __syncthreads();

        if( threadIdx.x >= 64 && row == shared_rows[threadIdx.x - 64] ) { left = inter_results[threadIdx.x - 64]; } 
        __syncthreads();
        inter_results[threadIdx.x] += left; left = 0;
        __syncthreads();

        if( threadIdx.x >= 128 && row == shared_rows[threadIdx.x - 128] ) { left = inter_results[threadIdx.x - 128]; } 
        __syncthreads();
        inter_results[threadIdx.x] += left; left = 0;
        __syncthreads();

      }


      template <typename T>
      __global__ void coordinate_matrix_vec_mul_kernel(const unsigned int * coords, //(row_index, column_index) 
                                                       const T * elements, 
                                                       const unsigned int * group_boundaries,
                                                       const T * vector,  
                                                             T * result) 
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
          __syncthreads();
          
          local_index = group_start + k * blockDim.x + threadIdx.x; 
        
          if (local_index < group_end)
          {
            tmp.x = coords[2*local_index]; 
            tmp.y = coords[2*local_index+1]; 
            val = elements[local_index] * vector[tmp.y]; 
          }
          else
          {
            tmp.x = 0;
            tmp.y = 0;
            val = 0;
          }

          __syncthreads();

          //check for carry from previous loop run: 
          if (threadIdx.x == 0 && k > 0)
          { 
            if (tmp.x == shared_rows[last_index]) 
              val += inter_results[last_index]; 
            else 
              result[shared_rows[last_index]] += inter_results[last_index]; 
          } 

          __syncthreads();

          coordinate_matrix_segmented_parallel_reduction(tmp.x, val, shared_rows, inter_results); //all threads have to enter this function

          __syncthreads();

          if (threadIdx.x != last_index &&
              shared_rows[threadIdx.x] != shared_rows[threadIdx.x + 1] &&
              inter_results[threadIdx.x] != 0) 
          { 
            result[tmp.x] += inter_results[threadIdx.x]; 
          }
        
          __syncthreads();
        } //for k
        
        if (threadIdx.x == last_index && inter_results[last_index] != 0) 
          result[tmp.x] += inter_results[last_index]; 
      }      

      
      /** @brief Carries out matrix-vector multiplication with a coordinate_matrix
      *
      * Implementation of the convenience expression result = prod(mat, vec);
      *
      * @param mat    The matrix
      * @param vec    The vector
      * @param result The result vector
      */
      template<class ScalarType, unsigned int ALIGNMENT, unsigned int VECTOR_ALIGNMENT>
      void prod_impl(const viennacl::coordinate_matrix<ScalarType, ALIGNMENT> & mat, 
                     const viennacl::vector<ScalarType, VECTOR_ALIGNMENT> & vec,
                           viennacl::vector<ScalarType, VECTOR_ALIGNMENT> & result)
      {
        result.clear();
        
        coordinate_matrix_vec_mul_kernel<<<64, 128>>>(detail::cuda_arg<unsigned int>(mat.handle12().cuda_handle()),
                                                      detail::cuda_arg<ScalarType>(mat.handle().cuda_handle()),
                                                      detail::cuda_arg<unsigned int>(mat.handle3().cuda_handle()),
                                                      detail::cuda_arg<ScalarType>(vec),
                                                      detail::cuda_arg<ScalarType>(result)
                                                     );
        VIENNACL_CUDA_LAST_ERROR_CHECK("coordinate_matrix_vec_mul_kernel");
      }
      

      
      
      
      //
      // ELL Matrix
      //
      
      template <typename T>
      __global__ void ell_matrix_vec_mul_kernel(const unsigned int * coords,
                                                const T * elements,
                                                const T * vector,
                                                      T * result,
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
              sum += (vector[col] * val);
            }
          }

          result[row_id] = sum;
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
      template<class ScalarType, unsigned int ALIGNMENT, unsigned int VECTOR_ALIGNMENT>
      void prod_impl(const viennacl::ell_matrix<ScalarType, ALIGNMENT> & mat, 
                     const viennacl::vector<ScalarType, VECTOR_ALIGNMENT> & vec,
                           viennacl::vector<ScalarType, VECTOR_ALIGNMENT> & result)
      {
        ell_matrix_vec_mul_kernel<<<256, 128>>>(detail::cuda_arg<unsigned int>(mat.handle2().cuda_handle()),
                                                detail::cuda_arg<ScalarType>(mat.handle().cuda_handle()),
                                                detail::cuda_arg<ScalarType>(vec),
                                                detail::cuda_arg<ScalarType>(result),
                                                static_cast<unsigned int>(mat.size1()),
                                                static_cast<unsigned int>(mat.size2()),
                                                static_cast<unsigned int>(mat.internal_size1()),
                                                static_cast<unsigned int>(mat.maxnnz()),
                                                static_cast<unsigned int>(mat.internal_maxnnz())
                                               );
        VIENNACL_CUDA_LAST_ERROR_CHECK("ell_matrix_vec_mul_kernel");
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
                                                const T * vector,
                                                      T * result,
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
              sum += (vector[col] * val);
            }
          }

          uint col_begin = csr_rows[row_id];
          uint col_end   = csr_rows[row_id + 1];

          for(uint item_id = col_begin; item_id < col_end; item_id++)
          {
            sum += (vector[csr_cols[item_id]] * csr_elements[item_id]);
          }

          result[row_id] = sum;
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
      template<class ScalarType, unsigned int ALIGNMENT, unsigned int VECTOR_ALIGNMENT>
      void prod_impl(const viennacl::hyb_matrix<ScalarType, ALIGNMENT> & mat, 
                     const viennacl::vector<ScalarType, VECTOR_ALIGNMENT> & vec,
                           viennacl::vector<ScalarType, VECTOR_ALIGNMENT> & result)
      {
        hyb_matrix_vec_mul_kernel<<<256, 128>>>(detail::cuda_arg<unsigned int>(mat.handle2().cuda_handle()),
                                                detail::cuda_arg<ScalarType>(mat.handle().cuda_handle()),
                                                detail::cuda_arg<unsigned int>(mat.handle3().cuda_handle()),
                                                detail::cuda_arg<unsigned int>(mat.handle4().cuda_handle()),
                                                detail::cuda_arg<ScalarType>(mat.handle5().cuda_handle()),
                                                detail::cuda_arg<ScalarType>(vec),
                                                detail::cuda_arg<ScalarType>(result),
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
