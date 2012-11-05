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
