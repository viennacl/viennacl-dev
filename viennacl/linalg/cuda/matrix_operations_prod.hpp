#ifndef VIENNACL_LINALG_CUDA_MATRIX_OPERATIONS_PROD_HPP_
#define VIENNACL_LINALG_CUDA_MATRIX_OPERATIONS_PROD_HPP_

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

/** @file  viennacl/linalg/cuda/matrix_operations_prod.hpp
    @brief Dense matrix-matrix product CUDA kernels reside here.
    
    Note: File created semi-automatically from OpenCL kernels.
*/


namespace viennacl
{
  namespace linalg
  {
    namespace cuda
    {
      
      template <typename T>
      __global__ void matrix_matrix_prod_kernel(
                T alpha,
                const T * A,
                unsigned int A_row_start,
                unsigned int A_col_start,
                unsigned int A_row_inc,
                unsigned int A_col_inc,
                unsigned int A_row_size,
                unsigned int A_col_size,
                unsigned int A_internal_rows,
                unsigned int A_internal_cols,
                bool row_major_A,
                bool transpose_A,
                const T * B,  
                unsigned int B_row_start,
                unsigned int B_col_start,
                unsigned int B_row_inc,
                unsigned int B_col_inc,
                unsigned int B_row_size,
                unsigned int B_col_size,
                unsigned int B_internal_rows,
                unsigned int B_internal_cols,
                bool row_major_B,
                bool transpose_B,
                T beta,
                T * C,
                unsigned int C_row_start,
                unsigned int C_col_start,
                unsigned int C_row_inc,
                unsigned int C_col_inc,
                unsigned int C_row_size,
                unsigned int C_col_size,
                unsigned int C_internal_rows,
                unsigned int C_internal_cols,
                bool row_major_C) 
      { 

        __shared__ T bufA[272];
        __shared__ T bufB[272];

        const size_t block_size = 16;
        
        size_t aBegin;
        size_t aStep;
        if (row_major_A && transpose_A)
        {
          aBegin = (blockIdx.x * block_size * A_col_inc + A_col_start) + A_row_start * A_internal_cols;
          aStep = block_size * A_row_inc * A_internal_cols;
        }
        else if (row_major_A && !transpose_A)
        {
          aBegin = (blockIdx.x * block_size * A_row_inc + A_row_start) * A_internal_cols + A_col_start;
          aStep = block_size * A_col_inc;
        }
        else if (!row_major_A && transpose_A)
        {
          aBegin = (blockIdx.x * block_size * A_col_inc + A_col_start) * A_internal_rows + A_row_start;
          aStep = block_size * A_row_inc;
        }
        else if (!row_major_A && !transpose_A)
        {
          aBegin = (blockIdx.x * block_size * A_row_inc + A_row_start) + A_col_start * A_internal_rows;
          aStep = block_size * A_col_inc * A_internal_rows;
        }
        
        size_t bBegin;
        size_t bStep;
        if (row_major_B && transpose_B)
        {
          bBegin = (blockIdx.y * block_size * B_row_inc + B_row_start) * B_internal_cols + B_col_start;
          bStep = block_size * B_col_inc;
        }
        else if (row_major_B && !transpose_B)
        {
          bBegin = (blockIdx.y * block_size * B_col_inc + B_col_start) + B_row_start * B_internal_cols;
          bStep = block_size * B_internal_cols * B_row_inc;
        }
        else if (!row_major_B && transpose_B)
        {
          bBegin = (blockIdx.y * block_size * B_row_inc + B_row_start) + B_col_start * B_internal_rows;
          bStep = block_size * B_internal_rows * B_col_inc;
        }
        else if (!row_major_B && !transpose_B)
        {
          bBegin = (blockIdx.y * block_size * B_col_inc + B_col_start) * B_internal_rows + B_row_start;
          bStep = block_size * B_row_inc;
        }
        
        size_t block_num = (transpose_A) ? (A_row_size + block_size - 1) / block_size
                                         : (A_col_size + block_size - 1) / block_size;
        
        T Csub = 0;
        
        size_t aOffset = (row_major_A) ? threadIdx.x * A_col_inc + threadIdx.y * A_row_inc * A_internal_cols
                                       : threadIdx.x * A_row_inc + threadIdx.y * A_col_inc * A_internal_rows;
        
        size_t bOffset = (row_major_B) ? threadIdx.x * B_col_inc + threadIdx.y * B_row_inc * B_internal_cols
                                       : threadIdx.x * B_row_inc + threadIdx.y * B_col_inc *  B_internal_rows;

        size_t row_thread_id_times_block_size = threadIdx.x * (block_size + 1);
        size_t col_thread_id_times_block_size = threadIdx.y * (block_size + 1);
        for (size_t block = 0;
                block < block_num;
                ++block)
        {
          if (transpose_A && row_major_A)
            bufA[row_thread_id_times_block_size + threadIdx.y] = ((block * block_size + threadIdx.y < A_row_size) && (blockIdx.x * block_size + threadIdx.x < A_col_size)) ? A[aBegin + aOffset] : 0;
          else if (transpose_A && !row_major_A)
            bufA[col_thread_id_times_block_size + threadIdx.x] = ((block * block_size + threadIdx.x < A_row_size) && (blockIdx.x * block_size + threadIdx.y < A_col_size)) ? A[aBegin + aOffset] : 0;
          else if (!transpose_A && row_major_A)
            bufA[col_thread_id_times_block_size + threadIdx.x] = ((block * block_size + threadIdx.x < A_col_size) && (blockIdx.x * block_size + threadIdx.y < A_row_size)) ? A[aBegin + aOffset] : 0;
          else if (!transpose_A && !row_major_A)
            bufA[row_thread_id_times_block_size + threadIdx.y] = ((block * block_size + threadIdx.y < A_col_size) && (blockIdx.x * block_size + threadIdx.x < A_row_size)) ? A[aBegin + aOffset] : 0;


          if (transpose_B && row_major_B)
            bufB[col_thread_id_times_block_size + threadIdx.x] = ((block * block_size + threadIdx.x < B_col_size) && (blockIdx.y * block_size + threadIdx.y < B_row_size)) ? B[bBegin + bOffset] : 0;
          else if (transpose_B && !row_major_B)
            bufB[row_thread_id_times_block_size + threadIdx.y] = ((block * block_size + threadIdx.y < B_col_size) && (blockIdx.y * block_size + threadIdx.x < B_row_size)) ? B[bBegin + bOffset] : 0;
          else if (!transpose_B && row_major_B)
            bufB[row_thread_id_times_block_size + threadIdx.y] = ((block * block_size + threadIdx.y < B_row_size) && (blockIdx.y * block_size + threadIdx.x < B_col_size)) ? B[bBegin + bOffset] : 0;
          else if (!transpose_B && !row_major_B)
            bufB[col_thread_id_times_block_size + threadIdx.x] = ((block * block_size + threadIdx.x < B_row_size) && (blockIdx.y * block_size + threadIdx.y < B_col_size)) ? B[bBegin + bOffset] : 0;
          
          __syncthreads();
          T * bufAptr = bufA + row_thread_id_times_block_size;
          T * bufBptr = bufB + col_thread_id_times_block_size;
          Csub += (*bufAptr) * (*bufBptr); ++bufAptr; ++bufBptr;
          Csub += (*bufAptr) * (*bufBptr); ++bufAptr; ++bufBptr;
          Csub += (*bufAptr) * (*bufBptr); ++bufAptr; ++bufBptr;
          Csub += (*bufAptr) * (*bufBptr); ++bufAptr; ++bufBptr;
          Csub += (*bufAptr) * (*bufBptr); ++bufAptr; ++bufBptr;
          Csub += (*bufAptr) * (*bufBptr); ++bufAptr; ++bufBptr;
          Csub += (*bufAptr) * (*bufBptr); ++bufAptr; ++bufBptr;
          Csub += (*bufAptr) * (*bufBptr); ++bufAptr; ++bufBptr;
          Csub += (*bufAptr) * (*bufBptr); ++bufAptr; ++bufBptr;
          Csub += (*bufAptr) * (*bufBptr); ++bufAptr; ++bufBptr;
          Csub += (*bufAptr) * (*bufBptr); ++bufAptr; ++bufBptr;
          Csub += (*bufAptr) * (*bufBptr); ++bufAptr; ++bufBptr;
          Csub += (*bufAptr) * (*bufBptr); ++bufAptr; ++bufBptr;
          Csub += (*bufAptr) * (*bufBptr); ++bufAptr; ++bufBptr;
          Csub += (*bufAptr) * (*bufBptr); ++bufAptr; ++bufBptr;
          Csub += (*bufAptr) * (*bufBptr); ++bufAptr; ++bufBptr;

          aBegin += aStep;
          bBegin += bStep;
        }
        
        if (transpose_A && transpose_B)
        {
          if ((blockIdx.x * blockDim.x + threadIdx.x) >= A_col_size || (blockIdx.y * blockDim.y + threadIdx.y) >= B_row_size)
            return;
        }
        else if ( transpose_A && !transpose_B)
        {
          if ((blockIdx.x * blockDim.x + threadIdx.x) >= A_col_size || (blockIdx.y * blockDim.y + threadIdx.y) >= B_col_size)
            return;
        }
        else if (!transpose_A &&  transpose_B)
        {
          if ((blockIdx.x * blockDim.x + threadIdx.x) >= A_row_size || (blockIdx.y * blockDim.y + threadIdx.y) >= B_row_size)
            return;
        }
        else
        {
          if ((blockIdx.x * blockDim.x + threadIdx.x) >= A_row_size || (blockIdx.y * blockDim.y + threadIdx.y) >= B_col_size)
            return;
        }
          
          
        if (row_major_C)
        {
          C[((blockIdx.x * blockDim.x + threadIdx.x) * C_row_inc + C_row_start) * C_internal_cols + (blockIdx.y * blockDim.y + threadIdx.y) * C_col_inc + C_col_start] = 
            (beta == 0) ? alpha * Csub 
                        : alpha * Csub + beta * C[((blockIdx.x * blockDim.x + threadIdx.x) * C_row_inc + C_row_start) * C_internal_cols + (blockIdx.y * blockDim.y + threadIdx.y) * C_col_inc + C_col_start];
        }
        else
        {
          C[(blockIdx.x * blockDim.x + threadIdx.x) * C_row_inc + C_row_start + ((blockIdx.y * blockDim.y + threadIdx.y) * C_col_inc + C_col_start) * C_internal_rows] = 
            (beta == 0) ? alpha * Csub
                        : alpha * Csub + beta * C[(blockIdx.x * blockDim.x + threadIdx.x) * C_row_inc + C_row_start + ((blockIdx.y * blockDim.y + threadIdx.y) * C_col_inc + C_col_start) * C_internal_rows];
        }
        
      }

      
    } // namespace cuda
  } //namespace linalg
} //namespace viennacl


#endif
