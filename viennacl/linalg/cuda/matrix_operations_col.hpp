#ifndef VIENNACL_LINALG_CUDA_MATRIX_OPERATIONS_COL_HPP_
#define VIENNACL_LINALG_CUDA_MATRIX_OPERATIONS_COL_HPP_

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

/** @file  viennacl/linalg/cuda/matrix_operations_col.hpp
    @brief Implementations of column-major dense matrix related operations, including matrix-vector products, using CUDA.
*/


namespace viennacl
{
  namespace linalg
  {
    namespace cuda
    {
      //
      // am
      //
      
      // alpha on CPU
      template <typename T>
      __global__ void am_col_kernel(
                T * A,
                unsigned int A_start1, unsigned int A_start2,
                unsigned int A_inc1,   unsigned int A_inc2,
                unsigned int A_size1,  unsigned int A_size2,
                unsigned int A_internal_size1,  unsigned int A_internal_size2,
                
                T fac2,
                unsigned int options2,
                const T * B,
                unsigned int B_start1, unsigned int B_start2,
                unsigned int B_inc1,   unsigned int B_inc2,
                unsigned int B_internal_size1,  unsigned int B_internal_size2)
      { 
        T alpha = fac2;
        if (options2 & (1 << 0))
          alpha = -alpha;
        if (options2 & (1 << 1))
          alpha = ((T)(1)) / alpha;

        unsigned int row_gid = (blockIdx.x * blockDim.x + threadIdx.x) / blockDim.x;
        unsigned int col_gid = (blockIdx.x * blockDim.x + threadIdx.x) % blockDim.x;
        
        for (unsigned int col = col_gid; col < A_size2; col += gridDim.x)
          for (unsigned int row = row_gid; row < A_size1; row += blockDim.x)
            A[(row * A_inc1 + A_start1) + (col * A_inc2 + A_start2) * A_internal_size1] = B[(row * B_inc1 + B_start1) + (col * B_inc2 + B_start2) * B_internal_size1] * alpha;
      }

      // alpha on GPU
      template <typename T>
      __global__ void am_col_kernel(
                T * A,
                unsigned int A_start1, unsigned int A_start2,
                unsigned int A_inc1,   unsigned int A_inc2,
                unsigned int A_size1,  unsigned int A_size2,
                unsigned int A_internal_size1,  unsigned int A_internal_size2,
                
                const T * fac2,
                unsigned int options2,
                const T * B,
                unsigned int B_start1, unsigned int B_start2,
                unsigned int B_inc1,   unsigned int B_inc2,
                unsigned int B_internal_size1,  unsigned int B_internal_size2)
      { 
        T alpha = *fac2;
        if (options2 & (1 << 0))
          alpha = -alpha;
        if (options2 & (1 << 1))
          alpha = ((T)(1)) / alpha;

        unsigned int row_gid = (blockIdx.x * blockDim.x + threadIdx.x) % blockDim.x;
        unsigned int col_gid = (blockIdx.x * blockDim.x + threadIdx.x) / blockDim.x;
        
        for (unsigned int col = col_gid; col < A_size2; col += gridDim.x)
          for (unsigned int row = row_gid; row < A_size1; row += blockDim.x)
            A[(row * A_inc1 + A_start1) + (col * A_inc2 + A_start2) * A_internal_size1] = B[(row * B_inc1 + B_start1) + (col * B_inc2 + B_start2) * B_internal_size1] * alpha;
      }
      
      
      //
      // ambm
      //
      
      // alpha and beta on CPU
      template <typename T>
      __global__ void ambm_col_kernel(
                T * A,
                unsigned int A_start1, unsigned int A_start2,
                unsigned int A_inc1,   unsigned int A_inc2,
                unsigned int A_size1,  unsigned int A_size2,
                unsigned int A_internal_size1,  unsigned int A_internal_size2,
                
                T fac2,
                unsigned int options2,
                const T * B,
                unsigned int B_start1, unsigned int B_start2,
                unsigned int B_inc1,   unsigned int B_inc2,
                unsigned int B_internal_size1,  unsigned int B_internal_size2,
                
                T fac3,
                unsigned int options3,
                const T * C,
                unsigned int C_start1, unsigned int C_start2,
                unsigned int C_inc1,   unsigned int C_inc2,
                unsigned int C_internal_size1,  unsigned int C_internal_size2)
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
          
        unsigned int row_gid = (blockIdx.x * blockDim.x + threadIdx.x) % blockDim.x;
        unsigned int col_gid = (blockIdx.x * blockDim.x + threadIdx.x) / blockDim.x;
        
        for (unsigned int col = col_gid; col < A_size2; col += gridDim.x)
          for (unsigned int row = row_gid; row < A_size1; row += blockDim.x)
            A[(row * A_inc1 + A_start1) + (col * A_inc2 + A_start2) * A_internal_size1] 
          = B[(row * B_inc1 + B_start1) + (col * B_inc2 + B_start2) * B_internal_size1] * alpha
          + C[(row * C_inc1 + C_start1) + (col * C_inc2 + C_start2) * C_internal_size1] * beta;
      }
      
      
      // alpha on CPU, beta on GPU
      template <typename T>
      __global__ void ambm_col_kernel(
                T * A,
                unsigned int A_start1, unsigned int A_start2,
                unsigned int A_inc1,   unsigned int A_inc2,
                unsigned int A_size1,  unsigned int A_size2,
                unsigned int A_internal_size1,  unsigned int A_internal_size2,
                
                T fac2,
                unsigned int options2,
                const T * B,
                unsigned int B_start1, unsigned int B_start2,
                unsigned int B_inc1,   unsigned int B_inc2,
                unsigned int B_internal_size1,  unsigned int B_internal_size2,
                
                const T * fac3,
                unsigned int options3,
                const T * C,
                unsigned int C_start1, unsigned int C_start2,
                unsigned int C_inc1,   unsigned int C_inc2,
                unsigned int C_internal_size1,  unsigned int C_internal_size2)
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
          
        unsigned int row_gid = (blockIdx.x * blockDim.x + threadIdx.x) % blockDim.x;
        unsigned int col_gid = (blockIdx.x * blockDim.x + threadIdx.x) / blockDim.x;
        
        for (unsigned int col = col_gid; col < A_size2; col += gridDim.x)
          for (unsigned int row = row_gid; row < A_size1; row += blockDim.x)
            A[(row * A_inc1 + A_start1) + (col * A_inc2 + A_start2) * A_internal_size1] 
          = B[(row * B_inc1 + B_start1) + (col * B_inc2 + B_start2) * B_internal_size1] * alpha
          + C[(row * C_inc1 + C_start1) + (col * C_inc2 + C_start2) * C_internal_size1] * beta;
      }
      
      // alpha on GPU, beta on CPU
      template <typename T>
      __global__ void ambm_col_kernel(
                T * A,
                unsigned int A_start1, unsigned int A_start2,
                unsigned int A_inc1,   unsigned int A_inc2,
                unsigned int A_size1,  unsigned int A_size2,
                unsigned int A_internal_size1,  unsigned int A_internal_size2,
                
                const T * fac2,
                unsigned int options2,
                const T * B,
                unsigned int B_start1, unsigned int B_start2,
                unsigned int B_inc1,   unsigned int B_inc2,
                unsigned int B_internal_size1,  unsigned int B_internal_size2,
                
                T fac3,
                unsigned int options3,
                const T * C,
                unsigned int C_start1, unsigned int C_start2,
                unsigned int C_inc1,   unsigned int C_inc2,
                unsigned int C_internal_size1,  unsigned int C_internal_size2)
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
          
        unsigned int row_gid = (blockIdx.x * blockDim.x + threadIdx.x) % blockDim.x;
        unsigned int col_gid = (blockIdx.x * blockDim.x + threadIdx.x) / blockDim.x;
        
        for (unsigned int col = col_gid; col < A_size2; col += gridDim.x)
          for (unsigned int row = row_gid; row < A_size1; row += blockDim.x)
            A[(row * A_inc1 + A_start1) + (col * A_inc2 + A_start2) * A_internal_size1] 
          = B[(row * B_inc1 + B_start1) + (col * B_inc2 + B_start2) * B_internal_size1] * alpha
          + C[(row * C_inc1 + C_start1) + (col * C_inc2 + C_start2) * C_internal_size1] * beta;
      }
      
      
      // alpha and beta on GPU
      template <typename T>
      __global__ void ambm_col_kernel(
                T * A,
                unsigned int A_start1, unsigned int A_start2,
                unsigned int A_inc1,   unsigned int A_inc2,
                unsigned int A_size1,  unsigned int A_size2,
                unsigned int A_internal_size1,  unsigned int A_internal_size2,
                
                const T * fac2,
                unsigned int options2,
                const T * B,
                unsigned int B_start1, unsigned int B_start2,
                unsigned int B_inc1,   unsigned int B_inc2,
                unsigned int B_internal_size1,  unsigned int B_internal_size2,
                
                const T * fac3,
                unsigned int options3,
                const T * C,
                unsigned int C_start1, unsigned int C_start2,
                unsigned int C_inc1,   unsigned int C_inc2,
                unsigned int C_internal_size1,  unsigned int C_internal_size2)
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
          
        unsigned int row_gid = (blockIdx.x * blockDim.x + threadIdx.x) % blockDim.x;
        unsigned int col_gid = (blockIdx.x * blockDim.x + threadIdx.x) / blockDim.x;
        
        for (unsigned int col = col_gid; col < A_size2; col += gridDim.x)
          for (unsigned int row = row_gid; row < A_size1; row += blockDim.x)
            A[(row * A_inc1 + A_start1) + (col * A_inc2 + A_start2) * A_internal_size1] 
          = B[(row * B_inc1 + B_start1) + (col * B_inc2 + B_start2) * B_internal_size1] * alpha
          + C[(row * C_inc1 + C_start1) + (col * C_inc2 + C_start2) * C_internal_size1] * beta;
      }
      
      
      //
      // ambm_m
      //
      
      // alpha and beta on CPU
      template <typename T>
      __global__ void ambm_m_col_kernel(
                T * A,
                unsigned int A_start1, unsigned int A_start2,
                unsigned int A_inc1,   unsigned int A_inc2,
                unsigned int A_size1,  unsigned int A_size2,
                unsigned int A_internal_size1,  unsigned int A_internal_size2,
                
                T fac2,
                unsigned int options2,
                const T * B,
                unsigned int B_start1, unsigned int B_start2,
                unsigned int B_inc1,   unsigned int B_inc2,
                unsigned int B_internal_size1,  unsigned int B_internal_size2,
                
                T fac3,
                unsigned int options3,
                const T * C,
                unsigned int C_start1, unsigned int C_start2,
                unsigned int C_inc1,   unsigned int C_inc2,
                unsigned int C_internal_size1,  unsigned int C_internal_size2)
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
          
        unsigned int row_gid = (blockIdx.x * blockDim.x + threadIdx.x) % blockDim.x;
        unsigned int col_gid = (blockIdx.x * blockDim.x + threadIdx.x) / blockDim.x;
        
        for (unsigned int col = col_gid; col < A_size2; col += gridDim.x)
          for (unsigned int row = row_gid; row < A_size1; row += blockDim.x)
            A[(row * A_inc1 + A_start1) + (col * A_inc2 + A_start2) * A_internal_size1] 
          += B[(row * B_inc1 + B_start1) + (col * B_inc2 + B_start2) * B_internal_size1] * alpha
           + C[(row * C_inc1 + C_start1) + (col * C_inc2 + C_start2) * C_internal_size1] * beta;
      }
      
      
      // alpha on CPU, beta on GPU
      template <typename T>
      __global__ void ambm_m_col_kernel(
                T * A,
                unsigned int A_start1, unsigned int A_start2,
                unsigned int A_inc1,   unsigned int A_inc2,
                unsigned int A_size1,  unsigned int A_size2,
                unsigned int A_internal_size1,  unsigned int A_internal_size2,
                
                T fac2,
                unsigned int options2,
                const T * B,
                unsigned int B_start1, unsigned int B_start2,
                unsigned int B_inc1,   unsigned int B_inc2,
                unsigned int B_internal_size1,  unsigned int B_internal_size2,
                
                const T * fac3,
                unsigned int options3,
                const T * C,
                unsigned int C_start1, unsigned int C_start2,
                unsigned int C_inc1,   unsigned int C_inc2,
                unsigned int C_internal_size1,  unsigned int C_internal_size2)
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
          
        unsigned int row_gid = (blockIdx.x * blockDim.x + threadIdx.x) % blockDim.x;
        unsigned int col_gid = (blockIdx.x * blockDim.x + threadIdx.x) / blockDim.x;
        
        for (unsigned int col = col_gid; col < A_size2; col += gridDim.x)
          for (unsigned int row = row_gid; row < A_size1; row += blockDim.x)
            A[(row * A_inc1 + A_start1) + (col * A_inc2 + A_start2) * A_internal_size1] 
          += B[(row * B_inc1 + B_start1) + (col * B_inc2 + B_start2) * B_internal_size1] * alpha
           + C[(row * C_inc1 + C_start1) + (col * C_inc2 + C_start2) * C_internal_size1] * beta;
      }
      
      // alpha on GPU, beta on CPU
      template <typename T>
      __global__ void ambm_m_col_kernel(
                T * A,
                unsigned int A_start1, unsigned int A_start2,
                unsigned int A_inc1,   unsigned int A_inc2,
                unsigned int A_size1,  unsigned int A_size2,
                unsigned int A_internal_size1,  unsigned int A_internal_size2,
                
                const T * fac2,
                unsigned int options2,
                const T * B,
                unsigned int B_start1, unsigned int B_start2,
                unsigned int B_inc1,   unsigned int B_inc2,
                unsigned int B_internal_size1,  unsigned int B_internal_size2,
                
                T fac3,
                unsigned int options3,
                const T * C,
                unsigned int C_start1, unsigned int C_start2,
                unsigned int C_inc1,   unsigned int C_inc2,
                unsigned int C_internal_size1,  unsigned int C_internal_size2)
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
          
        unsigned int row_gid = (blockIdx.x * blockDim.x + threadIdx.x) % blockDim.x;
        unsigned int col_gid = (blockIdx.x * blockDim.x + threadIdx.x) / blockDim.x;
        
        for (unsigned int col = col_gid; col < A_size2; col += gridDim.x)
          for (unsigned int row = row_gid; row < A_size1; row += blockDim.x)
            A[(row * A_inc1 + A_start1) + (col * A_inc2 + A_start2) * A_internal_size1] 
          += B[(row * B_inc1 + B_start1) + (col * B_inc2 + B_start2) * B_internal_size1] * alpha
           + C[(row * C_inc1 + C_start1) + (col * C_inc2 + C_start2) * C_internal_size1] * beta;
      }
      
      
      // alpha and beta on GPU
      template <typename T>
      __global__ void ambm_m_col_kernel(
                T * A,
                unsigned int A_start1, unsigned int A_start2,
                unsigned int A_inc1,   unsigned int A_inc2,
                unsigned int A_size1,  unsigned int A_size2,
                unsigned int A_internal_size1,  unsigned int A_internal_size2,
                
                const T * fac2,
                unsigned int options2,
                const T * B,
                unsigned int B_start1, unsigned int B_start2,
                unsigned int B_inc1,   unsigned int B_inc2,
                unsigned int B_internal_size1,  unsigned int B_internal_size2,
                
                const T * fac3,
                unsigned int options3,
                const T * C,
                unsigned int C_start1, unsigned int C_start2,
                unsigned int C_inc1,   unsigned int C_inc2,
                unsigned int C_internal_size1,  unsigned int C_internal_size2)
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
          
        unsigned int row_gid = (blockIdx.x * blockDim.x + threadIdx.x) % blockDim.x;
        unsigned int col_gid = (blockIdx.x * blockDim.x + threadIdx.x) / blockDim.x;
        
        for (unsigned int col = col_gid; col < A_size2; col += gridDim.x)
          for (unsigned int row = row_gid; row < A_size1; row += blockDim.x)
            A[(row * A_inc1 + A_start1) + (col * A_inc2 + A_start2) * A_internal_size1] 
          += B[(row * B_inc1 + B_start1) + (col * B_inc2 + B_start2) * B_internal_size1] * alpha
           + C[(row * C_inc1 + C_start1) + (col * C_inc2 + C_start2) * C_internal_size1] * beta;
      }
      
      
      
      //
      // assignments
      //
      
      template <typename T>
      __global__ void matrix_col_assign_kernel(
                T * A,
                unsigned int A_start1, unsigned int A_start2,
                unsigned int A_inc1,   unsigned int A_inc2,
                unsigned int A_size1,  unsigned int A_size2,
                unsigned int A_internal_size1,  unsigned int A_internal_size2,
                T alpha)
      { 
        unsigned int row_gid = (blockIdx.x * blockDim.x + threadIdx.x) % blockDim.x;
        unsigned int col_gid = (blockIdx.x * blockDim.x + threadIdx.x) / blockDim.x;
        
        for (unsigned int col = col_gid; col < A_size2; col += gridDim.x)
          for (unsigned int row = row_gid; row < A_size1; row += blockDim.x)
            A[(row * A_inc1 + A_start1) + (col * A_inc2 + A_start2) * A_internal_size1] = alpha;
      }
      
      
      template <typename T>
      __global__ void matrix_col_diagonal_assign_kernel(
                T * A,
                unsigned int A_start1, unsigned int A_start2,
                unsigned int A_inc1,   unsigned int A_inc2,
                unsigned int A_size1,  unsigned int A_size2,
                unsigned int A_internal_size1,  unsigned int A_internal_size2,
                T alpha)
      { 
        unsigned int gid = (blockIdx.x * blockDim.x + threadIdx.x);
        
        for (unsigned int row = gid; row < A_size1; row += blockDim.x * gridDim.x)
          A[(row * A_inc1 + A_start1) + (row * A_inc2 + A_start2) * A_internal_size1] = alpha;
      }

      
      
      //
      // matrix-vector product
      //

      template <typename T>
      __global__ void vec_mul_col_kernel(
                const T * A,
                unsigned int A_row_start,
                unsigned int A_col_start,
                unsigned int A_row_inc,
                unsigned int A_col_inc,
                unsigned int A_row_size,
                unsigned int A_col_size,
                unsigned int A_internal_rows,
                unsigned int A_internal_cols,
                const T * v,
                unsigned int v_start,
                unsigned int v_inc,
                unsigned int v_size,
                T * result,
                unsigned int result_start,
                unsigned int result_inc,
                unsigned int result_size) 
      { 
        
        for (unsigned int row = blockIdx.x * blockDim.x + threadIdx.x; row < A_row_size; row += gridDim.x * blockDim.x)
        {
          T dot_prod = 0;
          for (unsigned int col = 0; col < A_col_size; ++col)
            dot_prod += A[(row * A_row_inc + A_row_start) + (col * A_col_inc + A_col_start) * A_internal_rows] * v[v_start + v_inc * col];
          result[row * result_inc + result_start] = dot_prod;
        }
      }
      
      
      template <typename T>
      __global__ void trans_vec_mul_col_kernel(
                const T * A,
                unsigned int A_row_start,
                unsigned int A_col_start,
                unsigned int A_row_inc,
                unsigned int A_col_inc,
                unsigned int A_row_size,
                unsigned int A_col_size,
                unsigned int A_internal_rows,
                unsigned int A_internal_cols,
                const T * v,
                unsigned int v_start,
                unsigned int v_inc,
                unsigned int v_size,
                T * result,
                unsigned int result_start,
                unsigned int result_inc,
                unsigned int result_size) 
      { 
        __shared__ T work[128];
        
        unsigned int row_gid = (blockIdx.x * blockDim.x + threadIdx.x) / blockDim.x;
        unsigned int col_gid = (blockIdx.x * blockDim.x + threadIdx.x) % blockDim.x;
        unsigned int lid = threadIdx.x;
        
        for (unsigned int row = row_gid; row < A_col_size; row += gridDim.x)
        {
          T dot_prod = 0;
          for (unsigned int col = col_gid; col < A_row_size; col += blockDim.x)
            dot_prod += A[(row * A_col_inc + A_col_start) * A_internal_rows + col * A_row_inc + A_row_start] * v[v_start + v_inc * col];
          work[lid] = dot_prod;
          
          for(unsigned int stride = blockDim.x/2 ; stride>0 ; stride>>=1){
            __syncthreads();
            if(lid < stride)
              work[lid] += work[lid+stride];
          }
          
          if(lid == 0)
            result[row * result_inc + result_start] = work[0];
        }
      }
      
      
      //
      // matrix-matrix products
      //
      
      
      
      
      //
      // scaled rank-1-update
      //
      
      // alpha on CPU
      template <typename T>
      __global__ void scaled_rank1_update_col_kernel(
                T * A,
                unsigned int A_start1, unsigned int A_start2,
                unsigned int A_inc1,   unsigned int A_inc2,
                unsigned int A_size1,  unsigned int A_size2,
                unsigned int A_internal_size1,  unsigned int A_internal_size2,

                T val,
                unsigned int options2,
                
                const T * vec1,
                unsigned int start1,
                unsigned int inc1,          
                unsigned int size1,

                const T * vec2,
                unsigned int start2,
                unsigned int inc2,          
                unsigned int size2) 
      { 
        T alpha = val;
        if (options2 & (1 << 0))
          alpha = -alpha;
        if (options2 & (1 << 1))
          alpha = ((T)(1)) / alpha;

        unsigned int row_gid = (blockIdx.x * blockDim.x + threadIdx.x) / blockDim.x;
        unsigned int col_gid = (blockIdx.x * blockDim.x + threadIdx.x) % blockDim.x;
        
        for (unsigned int row = row_gid; row < A_size1; row += gridDim.x)
        {
          T tmp = alpha * vec1[row * inc1 + start1];
          for (unsigned int col = col_gid; col < A_size2; col += blockDim.x)
            A[(row * A_inc1 + A_start1) + (col * A_inc2 + A_start2) * A_internal_size1] += tmp * vec2[col * inc2 + start2];
        }
      }


      // alpha on GPU
      template <typename T>
      __global__ void scaled_rank1_update_col_kernel(
                T * A,
                unsigned int A_start1, unsigned int A_start2,
                unsigned int A_inc1,   unsigned int A_inc2,
                unsigned int A_size1,  unsigned int A_size2,
                unsigned int A_internal_size1,  unsigned int A_internal_size2,

                T * val,
                unsigned int options2,
                
                const T * vec1,
                unsigned int start1,
                unsigned int inc1,          
                unsigned int size1,

                const T * vec2,
                unsigned int start2,
                unsigned int inc2,          
                unsigned int size2) 
      { 
        T alpha = *val;
        if (options2 & (1 << 0))
          alpha = -alpha;
        if (options2 & (1 << 1))
          alpha = ((T)(1)) / alpha;

        unsigned int row_gid = (blockIdx.x * blockDim.x + threadIdx.x) / blockDim.x;
        unsigned int col_gid = (blockIdx.x * blockDim.x + threadIdx.x) % blockDim.x;
        
        for (unsigned int row = row_gid; row < A_size1; row += gridDim.x)
        {
          T tmp = alpha * vec1[row * inc1 + start1];
          for (unsigned int col = col_gid; col < A_size2; col += blockDim.x)
            A[(row * A_inc1 + A_start1) + (col * A_inc2 + A_start2) * A_internal_size1] += tmp * vec2[col * inc2 + start2];
        }
      }

      
      
    } // namespace cuda
  } //namespace linalg
} //namespace viennacl


#endif
