#ifndef VIENNACL_LINALG_CUDA_MATRIX_OPERATIONS_ROW_HPP_
#define VIENNACL_LINALG_CUDA_MATRIX_OPERATIONS_ROW_HPP_

/* =========================================================================
   Copyright (c) 2010-2014, Institute for Microelectronics,
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

/** @file  viennacl/linalg/cuda/matrix_operations_row.hpp
    @brief Implementations of row-major dense matrix related operations, including matrix-vector products, using CUDA.
*/


namespace viennacl
{
namespace linalg
{
namespace cuda
{

inline  __device__ unsigned int row_major_index(unsigned int row,unsigned int col, unsigned int /*num_rows*/,unsigned int num_cols)
{
  return row*num_cols+col;
}
inline  __device__ unsigned int col_major_index(unsigned int row,unsigned int col, unsigned int num_rows, unsigned int /*num_cols*/)
{
  return row+col*num_rows;
}
//Matrix transpose kernel
template<typename NumericT>
__global__ void trans_kernel(
          const NumericT * A,
          unsigned int A_start1,          unsigned int A_start2,
          unsigned int A_internal_size1,  unsigned int A_internal_size2,
          unsigned int A_size1,           unsigned int A_size2,
          unsigned int A_stride1,         unsigned int A_stride2,

          NumericT * B,
          unsigned int B_start1,          unsigned int B_start2,
          unsigned int B_internal_size1,  unsigned int B_internal_size2,
          unsigned int B_stride1,         unsigned int B_stride2,
          bool data_major)
{
  unsigned int size = A_internal_size2*A_internal_size1;
  for(unsigned int i = blockIdx.x; i<size/gridDim.x; i+=gridDim.x)
  {
    unsigned int matrix_index = i * blockDim.x + threadIdx.x;

    unsigned int row = matrix_index / A_internal_size2;
    unsigned int col = matrix_index % A_internal_size2;
    if (row < A_size1 && col < A_size2)
    {
      if(data_major)
      {
        unsigned int pos = row_major_index(A_start1 + A_stride1 * row,
                                           A_start2 + A_stride2 * col,
                                           A_internal_size1, A_internal_size2);
        unsigned int new_pos = row_major_index(B_start2 + B_stride2 * col,
                                           B_start1 + B_stride1 * row,
                                           B_internal_size1, B_internal_size2);
        B[new_pos] = A[pos];

      } else
      {
        unsigned int pos = col_major_index(A_start1 + A_stride1 * row,
                                           A_start2 + A_stride2 * col,
                                           A_internal_size1, A_internal_size2);
        unsigned int new_pos = col_major_index(B_start2 + B_stride2 * col,
                                           B_start1 + B_stride1 * row,
                                           B_internal_size1, B_internal_size2);
        B[new_pos] = A[pos];

      }
     }
  }
}

//
// am
//

// alpha on CPU
template<typename NumericT>
__global__ void am_row_kernel(
          NumericT * A,
          unsigned int A_start1, unsigned int A_start2,
          unsigned int A_inc1,   unsigned int A_inc2,
          unsigned int A_size1,  unsigned int A_size2,
          unsigned int A_internal_size1,  unsigned int A_internal_size2,

          NumericT fac2,
          unsigned int options2,
          const NumericT * B,
          unsigned int B_start1, unsigned int B_start2,
          unsigned int B_inc1,   unsigned int B_inc2,
          unsigned int B_internal_size1,  unsigned int B_internal_size2)
{
  NumericT alpha = fac2;
  if (options2 & (1 << 0))
    alpha = -alpha;

  unsigned int row_gid = (blockIdx.x * blockDim.x + threadIdx.x) / blockDim.x;
  unsigned int col_gid = (blockIdx.x * blockDim.x + threadIdx.x) % blockDim.x;

  if (options2 & (1 << 1))
  {
    for (unsigned int row = row_gid; row < A_size1; row += gridDim.x)
      for (unsigned int col = col_gid; col < A_size2; col += blockDim.x)
        A[(row * A_inc1 + A_start1) * A_internal_size2 + col * A_inc2 + A_start2] = B[(row * B_inc1 + B_start1) * B_internal_size2 + col * B_inc2 + B_start2] / alpha;
  }
  else
  {
    for (unsigned int row = row_gid; row < A_size1; row += gridDim.x)
      for (unsigned int col = col_gid; col < A_size2; col += blockDim.x)
        A[(row * A_inc1 + A_start1) * A_internal_size2 + col * A_inc2 + A_start2] = B[(row * B_inc1 + B_start1) * B_internal_size2 + col * B_inc2 + B_start2] * alpha;
  }
}

// alpha on GPU
template<typename NumericT>
__global__ void am_row_kernel(
          NumericT * A,
          unsigned int A_start1, unsigned int A_start2,
          unsigned int A_inc1,   unsigned int A_inc2,
          unsigned int A_size1,  unsigned int A_size2,
          unsigned int A_internal_size1,  unsigned int A_internal_size2,

          const NumericT * fac2,
          unsigned int options2,
          const NumericT * B,
          unsigned int B_start1, unsigned int B_start2,
          unsigned int B_inc1,   unsigned int B_inc2,
          unsigned int B_internal_size1,  unsigned int B_internal_size2)
{
  NumericT alpha = *fac2;
  if (options2 & (1 << 0))
    alpha = -alpha;

  unsigned int row_gid = (blockIdx.x * blockDim.x + threadIdx.x) / blockDim.x;
  unsigned int col_gid = (blockIdx.x * blockDim.x + threadIdx.x) % blockDim.x;

  if (options2 & (1 << 1))
  {
    for (unsigned int row = row_gid; row < A_size1; row += gridDim.x)
      for (unsigned int col = col_gid; col < A_size2; col += blockDim.x)
        A[(row * A_inc1 + A_start1) * A_internal_size2 + col * A_inc2 + A_start2] = B[(row * B_inc1 + B_start1) * B_internal_size2 + col * B_inc2 + B_start2] / alpha;
  }
  else
  {
    for (unsigned int row = row_gid; row < A_size1; row += gridDim.x)
      for (unsigned int col = col_gid; col < A_size2; col += blockDim.x)
        A[(row * A_inc1 + A_start1) * A_internal_size2 + col * A_inc2 + A_start2] = B[(row * B_inc1 + B_start1) * B_internal_size2 + col * B_inc2 + B_start2] * alpha;
  }
}


//
// ambm
//

// alpha and beta on CPU
template<typename NumericT>
__global__ void ambm_row_kernel(
          NumericT * A,
          unsigned int A_start1, unsigned int A_start2,
          unsigned int A_inc1,   unsigned int A_inc2,
          unsigned int A_size1,  unsigned int A_size2,
          unsigned int A_internal_size1,  unsigned int A_internal_size2,

          NumericT fac2,
          unsigned int options2,
          const NumericT * B,
          unsigned int B_start1, unsigned int B_start2,
          unsigned int B_inc1,   unsigned int B_inc2,
          unsigned int B_internal_size1,  unsigned int B_internal_size2,

          NumericT fac3,
          unsigned int options3,
          const NumericT * C,
          unsigned int C_start1, unsigned int C_start2,
          unsigned int C_inc1,   unsigned int C_inc2,
          unsigned int C_internal_size1,  unsigned int C_internal_size2)
{
  NumericT alpha = fac2;
  if (options2 & (1 << 0))
    alpha = -alpha;

  NumericT beta = fac3;
  if (options3 & (1 << 0))
    beta = -beta;

  unsigned int row_gid = (blockIdx.x * blockDim.x + threadIdx.x) / blockDim.x;
  unsigned int col_gid = (blockIdx.x * blockDim.x + threadIdx.x) % blockDim.x;

  if (options2 & (1 << 1))
  {
    if (options3 & (1 << 1))
    {
      for (unsigned int row = row_gid; row < A_size1; row += gridDim.x)
        for (unsigned int col = col_gid; col < A_size2; col += blockDim.x)
          A[(row * A_inc1 + A_start1) * A_internal_size2 + col * A_inc2 + A_start2]
        = B[(row * B_inc1 + B_start1) * B_internal_size2 + col * B_inc2 + B_start2] / alpha
        + C[(row * C_inc1 + C_start1) * C_internal_size2 + col * C_inc2 + C_start2] / beta;
    }
    else
    {
      for (unsigned int row = row_gid; row < A_size1; row += gridDim.x)
        for (unsigned int col = col_gid; col < A_size2; col += blockDim.x)
          A[(row * A_inc1 + A_start1) * A_internal_size2 + col * A_inc2 + A_start2]
        = B[(row * B_inc1 + B_start1) * B_internal_size2 + col * B_inc2 + B_start2] / alpha
        + C[(row * C_inc1 + C_start1) * C_internal_size2 + col * C_inc2 + C_start2] * beta;
    }
  }
  else
  {
    if (options3 & (1 << 1))
    {
      for (unsigned int row = row_gid; row < A_size1; row += gridDim.x)
        for (unsigned int col = col_gid; col < A_size2; col += blockDim.x)
          A[(row * A_inc1 + A_start1) * A_internal_size2 + col * A_inc2 + A_start2]
        = B[(row * B_inc1 + B_start1) * B_internal_size2 + col * B_inc2 + B_start2] * alpha
        + C[(row * C_inc1 + C_start1) * C_internal_size2 + col * C_inc2 + C_start2] / beta;
    }
    else
    {
      for (unsigned int row = row_gid; row < A_size1; row += gridDim.x)
        for (unsigned int col = col_gid; col < A_size2; col += blockDim.x)
          A[(row * A_inc1 + A_start1) * A_internal_size2 + col * A_inc2 + A_start2]
        = B[(row * B_inc1 + B_start1) * B_internal_size2 + col * B_inc2 + B_start2] * alpha
        + C[(row * C_inc1 + C_start1) * C_internal_size2 + col * C_inc2 + C_start2] * beta;
    }
  }
}


// alpha on CPU, beta on GPU
template<typename NumericT>
__global__ void ambm_row_kernel(
          NumericT * A,
          unsigned int A_start1, unsigned int A_start2,
          unsigned int A_inc1,   unsigned int A_inc2,
          unsigned int A_size1,  unsigned int A_size2,
          unsigned int A_internal_size1,  unsigned int A_internal_size2,

          NumericT fac2,
          unsigned int options2,
          const NumericT * B,
          unsigned int B_start1, unsigned int B_start2,
          unsigned int B_inc1,   unsigned int B_inc2,
          unsigned int B_internal_size1,  unsigned int B_internal_size2,

          const NumericT * fac3,
          unsigned int options3,
          const NumericT * C,
          unsigned int C_start1, unsigned int C_start2,
          unsigned int C_inc1,   unsigned int C_inc2,
          unsigned int C_internal_size1,  unsigned int C_internal_size2)
{
  NumericT alpha = fac2;
  if (options2 & (1 << 0))
    alpha = -alpha;

  NumericT beta = *fac3;
  if (options3 & (1 << 0))
    beta = -beta;

  unsigned int row_gid = (blockIdx.x * blockDim.x + threadIdx.x) / blockDim.x;
  unsigned int col_gid = (blockIdx.x * blockDim.x + threadIdx.x) % blockDim.x;

  if (options2 & (1 << 1))
  {
    if (options3 & (1 << 1))
    {
      for (unsigned int row = row_gid; row < A_size1; row += gridDim.x)
        for (unsigned int col = col_gid; col < A_size2; col += blockDim.x)
          A[(row * A_inc1 + A_start1) * A_internal_size2 + col * A_inc2 + A_start2]
        = B[(row * B_inc1 + B_start1) * B_internal_size2 + col * B_inc2 + B_start2] / alpha
        + C[(row * C_inc1 + C_start1) * C_internal_size2 + col * C_inc2 + C_start2] / beta;
    }
    else
    {
      for (unsigned int row = row_gid; row < A_size1; row += gridDim.x)
        for (unsigned int col = col_gid; col < A_size2; col += blockDim.x)
          A[(row * A_inc1 + A_start1) * A_internal_size2 + col * A_inc2 + A_start2]
        = B[(row * B_inc1 + B_start1) * B_internal_size2 + col * B_inc2 + B_start2] / alpha
        + C[(row * C_inc1 + C_start1) * C_internal_size2 + col * C_inc2 + C_start2] * beta;
    }
  }
  else
  {
    if (options3 & (1 << 1))
    {
      for (unsigned int row = row_gid; row < A_size1; row += gridDim.x)
        for (unsigned int col = col_gid; col < A_size2; col += blockDim.x)
          A[(row * A_inc1 + A_start1) * A_internal_size2 + col * A_inc2 + A_start2]
        = B[(row * B_inc1 + B_start1) * B_internal_size2 + col * B_inc2 + B_start2] * alpha
        + C[(row * C_inc1 + C_start1) * C_internal_size2 + col * C_inc2 + C_start2] / beta;
    }
    else
    {
      for (unsigned int row = row_gid; row < A_size1; row += gridDim.x)
        for (unsigned int col = col_gid; col < A_size2; col += blockDim.x)
          A[(row * A_inc1 + A_start1) * A_internal_size2 + col * A_inc2 + A_start2]
        = B[(row * B_inc1 + B_start1) * B_internal_size2 + col * B_inc2 + B_start2] * alpha
        + C[(row * C_inc1 + C_start1) * C_internal_size2 + col * C_inc2 + C_start2] * beta;
    }
  }
}

// alpha on GPU, beta on CPU
template<typename NumericT>
__global__ void ambm_row_kernel(
          NumericT * A,
          unsigned int A_start1, unsigned int A_start2,
          unsigned int A_inc1,   unsigned int A_inc2,
          unsigned int A_size1,  unsigned int A_size2,
          unsigned int A_internal_size1,  unsigned int A_internal_size2,

          const NumericT * fac2,
          unsigned int options2,
          const NumericT * B,
          unsigned int B_start1, unsigned int B_start2,
          unsigned int B_inc1,   unsigned int B_inc2,
          unsigned int B_internal_size1,  unsigned int B_internal_size2,

          NumericT fac3,
          unsigned int options3,
          const NumericT * C,
          unsigned int C_start1, unsigned int C_start2,
          unsigned int C_inc1,   unsigned int C_inc2,
          unsigned int C_internal_size1,  unsigned int C_internal_size2)
{
  NumericT alpha = *fac2;
  if (options2 & (1 << 0))
    alpha = -alpha;

  NumericT beta = fac3;
  if (options3 & (1 << 0))
    beta = -beta;

  unsigned int row_gid = (blockIdx.x * blockDim.x + threadIdx.x) / blockDim.x;
  unsigned int col_gid = (blockIdx.x * blockDim.x + threadIdx.x) % blockDim.x;

  if (options2 & (1 << 1))
  {
    if (options3 & (1 << 1))
    {
      for (unsigned int row = row_gid; row < A_size1; row += gridDim.x)
        for (unsigned int col = col_gid; col < A_size2; col += blockDim.x)
          A[(row * A_inc1 + A_start1) * A_internal_size2 + col * A_inc2 + A_start2]
        = B[(row * B_inc1 + B_start1) * B_internal_size2 + col * B_inc2 + B_start2] / alpha
        + C[(row * C_inc1 + C_start1) * C_internal_size2 + col * C_inc2 + C_start2] / beta;
    }
    else
    {
      for (unsigned int row = row_gid; row < A_size1; row += gridDim.x)
        for (unsigned int col = col_gid; col < A_size2; col += blockDim.x)
          A[(row * A_inc1 + A_start1) * A_internal_size2 + col * A_inc2 + A_start2]
        = B[(row * B_inc1 + B_start1) * B_internal_size2 + col * B_inc2 + B_start2] / alpha
        + C[(row * C_inc1 + C_start1) * C_internal_size2 + col * C_inc2 + C_start2] * beta;
    }
  }
  else
  {
    if (options3 & (1 << 1))
    {
      for (unsigned int row = row_gid; row < A_size1; row += gridDim.x)
        for (unsigned int col = col_gid; col < A_size2; col += blockDim.x)
          A[(row * A_inc1 + A_start1) * A_internal_size2 + col * A_inc2 + A_start2]
        = B[(row * B_inc1 + B_start1) * B_internal_size2 + col * B_inc2 + B_start2] * alpha
        + C[(row * C_inc1 + C_start1) * C_internal_size2 + col * C_inc2 + C_start2] / beta;
    }
    else
    {
      for (unsigned int row = row_gid; row < A_size1; row += gridDim.x)
        for (unsigned int col = col_gid; col < A_size2; col += blockDim.x)
          A[(row * A_inc1 + A_start1) * A_internal_size2 + col * A_inc2 + A_start2]
        = B[(row * B_inc1 + B_start1) * B_internal_size2 + col * B_inc2 + B_start2] * alpha
        + C[(row * C_inc1 + C_start1) * C_internal_size2 + col * C_inc2 + C_start2] * beta;
    }
  }
}


// alpha and beta on GPU
template<typename NumericT>
__global__ void ambm_row_kernel(
          NumericT * A,
          unsigned int A_start1, unsigned int A_start2,
          unsigned int A_inc1,   unsigned int A_inc2,
          unsigned int A_size1,  unsigned int A_size2,
          unsigned int A_internal_size1,  unsigned int A_internal_size2,

          const NumericT * fac2,
          unsigned int options2,
          const NumericT * B,
          unsigned int B_start1, unsigned int B_start2,
          unsigned int B_inc1,   unsigned int B_inc2,
          unsigned int B_internal_size1,  unsigned int B_internal_size2,

          const NumericT * fac3,
          unsigned int options3,
          const NumericT * C,
          unsigned int C_start1, unsigned int C_start2,
          unsigned int C_inc1,   unsigned int C_inc2,
          unsigned int C_internal_size1,  unsigned int C_internal_size2)
{
  NumericT alpha = *fac2;
  if (options2 & (1 << 0))
    alpha = -alpha;

  NumericT beta = *fac3;
  if (options3 & (1 << 0))
    beta = -beta;

  unsigned int row_gid = (blockIdx.x * blockDim.x + threadIdx.x) / blockDim.x;
  unsigned int col_gid = (blockIdx.x * blockDim.x + threadIdx.x) % blockDim.x;

  if (options2 & (1 << 1))
  {
    if (options3 & (1 << 1))
    {
      for (unsigned int row = row_gid; row < A_size1; row += gridDim.x)
        for (unsigned int col = col_gid; col < A_size2; col += blockDim.x)
          A[(row * A_inc1 + A_start1) * A_internal_size2 + col * A_inc2 + A_start2]
        = B[(row * B_inc1 + B_start1) * B_internal_size2 + col * B_inc2 + B_start2] / alpha
        + C[(row * C_inc1 + C_start1) * C_internal_size2 + col * C_inc2 + C_start2] / beta;
    }
    else
    {
      for (unsigned int row = row_gid; row < A_size1; row += gridDim.x)
        for (unsigned int col = col_gid; col < A_size2; col += blockDim.x)
          A[(row * A_inc1 + A_start1) * A_internal_size2 + col * A_inc2 + A_start2]
        = B[(row * B_inc1 + B_start1) * B_internal_size2 + col * B_inc2 + B_start2] / alpha
        + C[(row * C_inc1 + C_start1) * C_internal_size2 + col * C_inc2 + C_start2] * beta;
    }
  }
  else
  {
    if (options3 & (1 << 1))
    {
      for (unsigned int row = row_gid; row < A_size1; row += gridDim.x)
        for (unsigned int col = col_gid; col < A_size2; col += blockDim.x)
          A[(row * A_inc1 + A_start1) * A_internal_size2 + col * A_inc2 + A_start2]
        = B[(row * B_inc1 + B_start1) * B_internal_size2 + col * B_inc2 + B_start2] * alpha
        + C[(row * C_inc1 + C_start1) * C_internal_size2 + col * C_inc2 + C_start2] / beta;
    }
    else
    {
      for (unsigned int row = row_gid; row < A_size1; row += gridDim.x)
        for (unsigned int col = col_gid; col < A_size2; col += blockDim.x)
          A[(row * A_inc1 + A_start1) * A_internal_size2 + col * A_inc2 + A_start2]
        = B[(row * B_inc1 + B_start1) * B_internal_size2 + col * B_inc2 + B_start2] * alpha
        + C[(row * C_inc1 + C_start1) * C_internal_size2 + col * C_inc2 + C_start2] * beta;
    }
  }
}


//
// ambm_m
//

// alpha and beta on CPU
template<typename NumericT>
__global__ void ambm_m_row_kernel(
          NumericT * A,
          unsigned int A_start1, unsigned int A_start2,
          unsigned int A_inc1,   unsigned int A_inc2,
          unsigned int A_size1,  unsigned int A_size2,
          unsigned int A_internal_size1,  unsigned int A_internal_size2,

          NumericT fac2,
          unsigned int options2,
          const NumericT * B,
          unsigned int B_start1, unsigned int B_start2,
          unsigned int B_inc1,   unsigned int B_inc2,
          unsigned int B_internal_size1,  unsigned int B_internal_size2,

          NumericT fac3,
          unsigned int options3,
          const NumericT * C,
          unsigned int C_start1, unsigned int C_start2,
          unsigned int C_inc1,   unsigned int C_inc2,
          unsigned int C_internal_size1,  unsigned int C_internal_size2)
{
  NumericT alpha = fac2;
  if (options2 & (1 << 0))
    alpha = -alpha;

  NumericT beta = fac3;
  if (options3 & (1 << 0))
    beta = -beta;

  unsigned int row_gid = (blockIdx.x * blockDim.x + threadIdx.x) / blockDim.x;
  unsigned int col_gid = (blockIdx.x * blockDim.x + threadIdx.x) % blockDim.x;

  if (options2 & (1 << 1))
  {
    if (options3 & (1 << 1))
    {
      for (unsigned int row = row_gid; row < A_size1; row += gridDim.x)
        for (unsigned int col = col_gid; col < A_size2; col += blockDim.x)
          A[(row * A_inc1 + A_start1) * A_internal_size2 + col * A_inc2 + A_start2]
       += B[(row * B_inc1 + B_start1) * B_internal_size2 + col * B_inc2 + B_start2] / alpha
        + C[(row * C_inc1 + C_start1) * C_internal_size2 + col * C_inc2 + C_start2] / beta;
    }
    else
    {
      for (unsigned int row = row_gid; row < A_size1; row += gridDim.x)
        for (unsigned int col = col_gid; col < A_size2; col += blockDim.x)
          A[(row * A_inc1 + A_start1) * A_internal_size2 + col * A_inc2 + A_start2]
       += B[(row * B_inc1 + B_start1) * B_internal_size2 + col * B_inc2 + B_start2] / alpha
        + C[(row * C_inc1 + C_start1) * C_internal_size2 + col * C_inc2 + C_start2] * beta;
    }
  }
  else
  {
    if (options3 & (1 << 1))
    {
      for (unsigned int row = row_gid; row < A_size1; row += gridDim.x)
        for (unsigned int col = col_gid; col < A_size2; col += blockDim.x)
          A[(row * A_inc1 + A_start1) * A_internal_size2 + col * A_inc2 + A_start2]
       += B[(row * B_inc1 + B_start1) * B_internal_size2 + col * B_inc2 + B_start2] * alpha
        + C[(row * C_inc1 + C_start1) * C_internal_size2 + col * C_inc2 + C_start2] / beta;
    }
    else
    {
      for (unsigned int row = row_gid; row < A_size1; row += gridDim.x)
        for (unsigned int col = col_gid; col < A_size2; col += blockDim.x)
          A[(row * A_inc1 + A_start1) * A_internal_size2 + col * A_inc2 + A_start2]
       += B[(row * B_inc1 + B_start1) * B_internal_size2 + col * B_inc2 + B_start2] * alpha
        + C[(row * C_inc1 + C_start1) * C_internal_size2 + col * C_inc2 + C_start2] * beta;
    }
  }
}


// alpha on CPU, beta on GPU
template<typename NumericT>
__global__ void ambm_m_row_kernel(
          NumericT * A,
          unsigned int A_start1, unsigned int A_start2,
          unsigned int A_inc1,   unsigned int A_inc2,
          unsigned int A_size1,  unsigned int A_size2,
          unsigned int A_internal_size1,  unsigned int A_internal_size2,

          NumericT fac2,
          unsigned int options2,
          const NumericT * B,
          unsigned int B_start1, unsigned int B_start2,
          unsigned int B_inc1,   unsigned int B_inc2,
          unsigned int B_internal_size1,  unsigned int B_internal_size2,

          const NumericT * fac3,
          unsigned int options3,
          const NumericT * C,
          unsigned int C_start1, unsigned int C_start2,
          unsigned int C_inc1,   unsigned int C_inc2,
          unsigned int C_internal_size1,  unsigned int C_internal_size2)
{
  NumericT alpha = fac2;
  if (options2 & (1 << 0))
    alpha = -alpha;

  NumericT beta = *fac3;
  if (options3 & (1 << 0))
    beta = -beta;

  unsigned int row_gid = (blockIdx.x * blockDim.x + threadIdx.x) / blockDim.x;
  unsigned int col_gid = (blockIdx.x * blockDim.x + threadIdx.x) % blockDim.x;

  if (options2 & (1 << 1))
  {
    if (options3 & (1 << 1))
    {
      for (unsigned int row = row_gid; row < A_size1; row += gridDim.x)
        for (unsigned int col = col_gid; col < A_size2; col += blockDim.x)
          A[(row * A_inc1 + A_start1) * A_internal_size2 + col * A_inc2 + A_start2]
       += B[(row * B_inc1 + B_start1) * B_internal_size2 + col * B_inc2 + B_start2] / alpha
        + C[(row * C_inc1 + C_start1) * C_internal_size2 + col * C_inc2 + C_start2] / beta;
    }
    else
    {
      for (unsigned int row = row_gid; row < A_size1; row += gridDim.x)
        for (unsigned int col = col_gid; col < A_size2; col += blockDim.x)
          A[(row * A_inc1 + A_start1) * A_internal_size2 + col * A_inc2 + A_start2]
       += B[(row * B_inc1 + B_start1) * B_internal_size2 + col * B_inc2 + B_start2] / alpha
        + C[(row * C_inc1 + C_start1) * C_internal_size2 + col * C_inc2 + C_start2] * beta;
    }
  }
  else
  {
    if (options3 & (1 << 1))
    {
      for (unsigned int row = row_gid; row < A_size1; row += gridDim.x)
        for (unsigned int col = col_gid; col < A_size2; col += blockDim.x)
          A[(row * A_inc1 + A_start1) * A_internal_size2 + col * A_inc2 + A_start2]
       += B[(row * B_inc1 + B_start1) * B_internal_size2 + col * B_inc2 + B_start2] * alpha
        + C[(row * C_inc1 + C_start1) * C_internal_size2 + col * C_inc2 + C_start2] / beta;
    }
    else
    {
      for (unsigned int row = row_gid; row < A_size1; row += gridDim.x)
        for (unsigned int col = col_gid; col < A_size2; col += blockDim.x)
          A[(row * A_inc1 + A_start1) * A_internal_size2 + col * A_inc2 + A_start2]
       += B[(row * B_inc1 + B_start1) * B_internal_size2 + col * B_inc2 + B_start2] * alpha
        + C[(row * C_inc1 + C_start1) * C_internal_size2 + col * C_inc2 + C_start2] * beta;
    }
  }
}

// alpha on GPU, beta on CPU
template<typename NumericT>
__global__ void ambm_m_row_kernel(
          NumericT * A,
          unsigned int A_start1, unsigned int A_start2,
          unsigned int A_inc1,   unsigned int A_inc2,
          unsigned int A_size1,  unsigned int A_size2,
          unsigned int A_internal_size1,  unsigned int A_internal_size2,

          const NumericT * fac2,
          unsigned int options2,
          const NumericT * B,
          unsigned int B_start1, unsigned int B_start2,
          unsigned int B_inc1,   unsigned int B_inc2,
          unsigned int B_internal_size1,  unsigned int B_internal_size2,

          NumericT fac3,
          unsigned int options3,
          const NumericT * C,
          unsigned int C_start1, unsigned int C_start2,
          unsigned int C_inc1,   unsigned int C_inc2,
          unsigned int C_internal_size1,  unsigned int C_internal_size2)
{
  NumericT alpha = *fac2;
  if (options2 & (1 << 0))
    alpha = -alpha;

  NumericT beta = fac3;
  if (options3 & (1 << 0))
    beta = -beta;

  unsigned int row_gid = (blockIdx.x * blockDim.x + threadIdx.x) / blockDim.x;
  unsigned int col_gid = (blockIdx.x * blockDim.x + threadIdx.x) % blockDim.x;

  if (options2 & (1 << 1))
  {
    if (options3 & (1 << 1))
    {
      for (unsigned int row = row_gid; row < A_size1; row += gridDim.x)
        for (unsigned int col = col_gid; col < A_size2; col += blockDim.x)
          A[(row * A_inc1 + A_start1) * A_internal_size2 + col * A_inc2 + A_start2]
       += B[(row * B_inc1 + B_start1) * B_internal_size2 + col * B_inc2 + B_start2] / alpha
        + C[(row * C_inc1 + C_start1) * C_internal_size2 + col * C_inc2 + C_start2] / beta;
    }
    else
    {
      for (unsigned int row = row_gid; row < A_size1; row += gridDim.x)
        for (unsigned int col = col_gid; col < A_size2; col += blockDim.x)
          A[(row * A_inc1 + A_start1) * A_internal_size2 + col * A_inc2 + A_start2]
       += B[(row * B_inc1 + B_start1) * B_internal_size2 + col * B_inc2 + B_start2] / alpha
        + C[(row * C_inc1 + C_start1) * C_internal_size2 + col * C_inc2 + C_start2] * beta;
    }
  }
  else
  {
    if (options3 & (1 << 1))
    {
      for (unsigned int row = row_gid; row < A_size1; row += gridDim.x)
        for (unsigned int col = col_gid; col < A_size2; col += blockDim.x)
          A[(row * A_inc1 + A_start1) * A_internal_size2 + col * A_inc2 + A_start2]
       += B[(row * B_inc1 + B_start1) * B_internal_size2 + col * B_inc2 + B_start2] * alpha
        + C[(row * C_inc1 + C_start1) * C_internal_size2 + col * C_inc2 + C_start2] / beta;
    }
    else
    {
      for (unsigned int row = row_gid; row < A_size1; row += gridDim.x)
        for (unsigned int col = col_gid; col < A_size2; col += blockDim.x)
          A[(row * A_inc1 + A_start1) * A_internal_size2 + col * A_inc2 + A_start2]
       += B[(row * B_inc1 + B_start1) * B_internal_size2 + col * B_inc2 + B_start2] * alpha
        + C[(row * C_inc1 + C_start1) * C_internal_size2 + col * C_inc2 + C_start2] * beta;
    }
  }
}


// alpha and beta on GPU
template<typename NumericT>
__global__ void ambm_m_row_kernel(
          NumericT * A,
          unsigned int A_start1, unsigned int A_start2,
          unsigned int A_inc1,   unsigned int A_inc2,
          unsigned int A_size1,  unsigned int A_size2,
          unsigned int A_internal_size1,  unsigned int A_internal_size2,

          const NumericT * fac2,
          unsigned int options2,
          const NumericT * B,
          unsigned int B_start1, unsigned int B_start2,
          unsigned int B_inc1,   unsigned int B_inc2,
          unsigned int B_internal_size1,  unsigned int B_internal_size2,

          const NumericT * fac3,
          unsigned int options3,
          const NumericT * C,
          unsigned int C_start1, unsigned int C_start2,
          unsigned int C_inc1,   unsigned int C_inc2,
          unsigned int C_internal_size1,  unsigned int C_internal_size2)
{
  NumericT alpha = *fac2;
  if (options2 & (1 << 0))
    alpha = -alpha;

  NumericT beta = *fac3;
  if (options3 & (1 << 0))
    beta = -beta;

  unsigned int row_gid = (blockIdx.x * blockDim.x + threadIdx.x) / blockDim.x;
  unsigned int col_gid = (blockIdx.x * blockDim.x + threadIdx.x) % blockDim.x;

  if (options2 & (1 << 1))
  {
    if (options3 & (1 << 1))
    {
      for (unsigned int row = row_gid; row < A_size1; row += gridDim.x)
        for (unsigned int col = col_gid; col < A_size2; col += blockDim.x)
          A[(row * A_inc1 + A_start1) * A_internal_size2 + col * A_inc2 + A_start2]
       += B[(row * B_inc1 + B_start1) * B_internal_size2 + col * B_inc2 + B_start2] / alpha
        + C[(row * C_inc1 + C_start1) * C_internal_size2 + col * C_inc2 + C_start2] / beta;
    }
    else
    {
      for (unsigned int row = row_gid; row < A_size1; row += gridDim.x)
        for (unsigned int col = col_gid; col < A_size2; col += blockDim.x)
          A[(row * A_inc1 + A_start1) * A_internal_size2 + col * A_inc2 + A_start2]
       += B[(row * B_inc1 + B_start1) * B_internal_size2 + col * B_inc2 + B_start2] / alpha
        + C[(row * C_inc1 + C_start1) * C_internal_size2 + col * C_inc2 + C_start2] * beta;
    }
  }
  else
  {
    if (options3 & (1 << 1))
    {
      for (unsigned int row = row_gid; row < A_size1; row += gridDim.x)
        for (unsigned int col = col_gid; col < A_size2; col += blockDim.x)
          A[(row * A_inc1 + A_start1) * A_internal_size2 + col * A_inc2 + A_start2]
       += B[(row * B_inc1 + B_start1) * B_internal_size2 + col * B_inc2 + B_start2] * alpha
        + C[(row * C_inc1 + C_start1) * C_internal_size2 + col * C_inc2 + C_start2] / beta;
    }
    else
    {
      for (unsigned int row = row_gid; row < A_size1; row += gridDim.x)
        for (unsigned int col = col_gid; col < A_size2; col += blockDim.x)
          A[(row * A_inc1 + A_start1) * A_internal_size2 + col * A_inc2 + A_start2]
       += B[(row * B_inc1 + B_start1) * B_internal_size2 + col * B_inc2 + B_start2] * alpha
        + C[(row * C_inc1 + C_start1) * C_internal_size2 + col * C_inc2 + C_start2] * beta;
    }
  }
}

//
// assignments
//

template<typename NumericT>
__global__ void matrix_row_assign_kernel(
          NumericT * A,
          unsigned int A_start1, unsigned int A_start2,
          unsigned int A_inc1,   unsigned int A_inc2,
          unsigned int A_size1,  unsigned int A_size2,
          unsigned int A_internal_size1,  unsigned int A_internal_size2,
          NumericT alpha)
{
  unsigned int row_gid = (blockIdx.x * blockDim.x + threadIdx.x) / blockDim.x;
  unsigned int col_gid = (blockIdx.x * blockDim.x + threadIdx.x) % blockDim.x;

  for (unsigned int row = row_gid; row < A_size1; row += gridDim.x)
    for (unsigned int col = col_gid; col < A_size2; col += blockDim.x)
      A[(row * A_inc1 + A_start1) * A_internal_size2 + col * A_inc2 + A_start2] = alpha;
}


template<typename NumericT>
__global__ void matrix_row_diagonal_assign_kernel(
          NumericT * A,
          unsigned int A_start1, unsigned int A_start2,
          unsigned int A_inc1,   unsigned int A_inc2,
          unsigned int A_size1,  unsigned int A_size2,
          unsigned int A_internal_size1,  unsigned int A_internal_size2,
          NumericT alpha)
{
  unsigned int gid = (blockIdx.x * blockDim.x + threadIdx.x);

  for (unsigned int row = gid; row < A_size1; row += blockDim.x * gridDim.x)
    A[(row * A_inc1 + A_start1) * A_internal_size2 + row * A_inc2 + A_start2] = alpha;
}

//
// binary element-wise operations
//

template<typename NumericT>
__global__ void element_op_row_kernel(
          NumericT * A,
          unsigned int A_start1, unsigned int A_start2,
          unsigned int A_inc1,   unsigned int A_inc2,
          unsigned int A_size1,  unsigned int A_size2,
          unsigned int A_internal_size1,  unsigned int A_internal_size2,

          const NumericT * B,
          unsigned int B_start1, unsigned int B_start2,
          unsigned int B_inc1,   unsigned int B_inc2,
          unsigned int B_internal_size1,  unsigned int B_internal_size2,

          const NumericT * C,
          unsigned int C_start1, unsigned int C_start2,
          unsigned int C_inc1,   unsigned int C_inc2,
          unsigned int C_internal_size1,  unsigned int C_internal_size2,

          unsigned int op_type) //0: product, 1: division, 2: pow
{
  unsigned int row_gid = (blockIdx.x * blockDim.x + threadIdx.x) / blockDim.x;
  unsigned int col_gid = (blockIdx.x * blockDim.x + threadIdx.x) % blockDim.x;

  if (op_type == 2)
  {
    for (unsigned int row = row_gid; row < A_size1; row += gridDim.x)
      for (unsigned int col = col_gid; col < A_size2; col += blockDim.x)
        A[(row * A_inc1 + A_start1) * A_internal_size2 + col * A_inc2 + A_start2]
      = pow(B[(row * B_inc1 + B_start1) * B_internal_size2 + col * B_inc2 + B_start2],
            C[(row * C_inc1 + C_start1) * C_internal_size2 + col * C_inc2 + C_start2]);
  }
  else if (op_type == 1)
  {
    for (unsigned int row = row_gid; row < A_size1; row += gridDim.x)
      for (unsigned int col = col_gid; col < A_size2; col += blockDim.x)
        A[(row * A_inc1 + A_start1) * A_internal_size2 + col * A_inc2 + A_start2]
      = B[(row * B_inc1 + B_start1) * B_internal_size2 + col * B_inc2 + B_start2]
      / C[(row * C_inc1 + C_start1) * C_internal_size2 + col * C_inc2 + C_start2];
  }
  else if (op_type == 0)
  {
    for (unsigned int row = row_gid; row < A_size1; row += gridDim.x)
      for (unsigned int col = col_gid; col < A_size2; col += blockDim.x)
        A[(row * A_inc1 + A_start1) * A_internal_size2 + col * A_inc2 + A_start2]
      = B[(row * B_inc1 + B_start1) * B_internal_size2 + col * B_inc2 + B_start2]
      * C[(row * C_inc1 + C_start1) * C_internal_size2 + col * C_inc2 + C_start2];
  }
}

template<typename NumericT>
__global__ void element_op_int_row_kernel(
          NumericT * A,
          unsigned int A_start1, unsigned int A_start2,
          unsigned int A_inc1,   unsigned int A_inc2,
          unsigned int A_size1,  unsigned int A_size2,
          unsigned int A_internal_size1,  unsigned int A_internal_size2,

          const NumericT * B,
          unsigned int B_start1, unsigned int B_start2,
          unsigned int B_inc1,   unsigned int B_inc2,
          unsigned int B_internal_size1,  unsigned int B_internal_size2,

          const NumericT * C,
          unsigned int C_start1, unsigned int C_start2,
          unsigned int C_inc1,   unsigned int C_inc2,
          unsigned int C_internal_size1,  unsigned int C_internal_size2,

          unsigned int op_type) //0: product, 1: division, 2: pow
{
  unsigned int row_gid = (blockIdx.x * blockDim.x + threadIdx.x) / blockDim.x;
  unsigned int col_gid = (blockIdx.x * blockDim.x + threadIdx.x) % blockDim.x;

  if (op_type == 1)
  {
    for (unsigned int row = row_gid; row < A_size1; row += gridDim.x)
      for (unsigned int col = col_gid; col < A_size2; col += blockDim.x)
        A[(row * A_inc1 + A_start1) * A_internal_size2 + col * A_inc2 + A_start2]
      = B[(row * B_inc1 + B_start1) * B_internal_size2 + col * B_inc2 + B_start2]
      / C[(row * C_inc1 + C_start1) * C_internal_size2 + col * C_inc2 + C_start2];
  }
  else if (op_type == 0)
  {
    for (unsigned int row = row_gid; row < A_size1; row += gridDim.x)
      for (unsigned int col = col_gid; col < A_size2; col += blockDim.x)
        A[(row * A_inc1 + A_start1) * A_internal_size2 + col * A_inc2 + A_start2]
      = B[(row * B_inc1 + B_start1) * B_internal_size2 + col * B_inc2 + B_start2]
      * C[(row * C_inc1 + C_start1) * C_internal_size2 + col * C_inc2 + C_start2];
  }
}

//
// unary element-wise operations
//

// abs
template<typename NumericT>
__global__ void matrix_row_element_abs_kernel(
          NumericT * A,
          unsigned int A_start1, unsigned int A_start2,
          unsigned int A_inc1,   unsigned int A_inc2,
          unsigned int A_size1,  unsigned int A_size2,
          unsigned int A_internal_size1,  unsigned int A_internal_size2,

          const NumericT * B,
          unsigned int B_start1, unsigned int B_start2,
          unsigned int B_inc1,   unsigned int B_inc2,
          unsigned int B_internal_size1,  unsigned int B_internal_size2)
{
  unsigned int row_gid = (blockIdx.x * blockDim.x + threadIdx.x) / blockDim.x;
  unsigned int col_gid = (blockIdx.x * blockDim.x + threadIdx.x) % blockDim.x;

  for (unsigned int row = row_gid; row < A_size1; row += gridDim.x)
    for (unsigned int col = col_gid; col < A_size2; col += blockDim.x)
      A[(row * A_inc1 + A_start1) * A_internal_size2 + col * A_inc2 + A_start2] = abs(B[(row * B_inc1 + B_start1) * B_internal_size2 + col * B_inc2 + B_start2]);
}


// acos
template<typename NumericT>
__global__ void matrix_row_element_acos_kernel(
          NumericT * A,
          unsigned int A_start1, unsigned int A_start2,
          unsigned int A_inc1,   unsigned int A_inc2,
          unsigned int A_size1,  unsigned int A_size2,
          unsigned int A_internal_size1,  unsigned int A_internal_size2,

          const NumericT * B,
          unsigned int B_start1, unsigned int B_start2,
          unsigned int B_inc1,   unsigned int B_inc2,
          unsigned int B_internal_size1,  unsigned int B_internal_size2)
{
  unsigned int row_gid = (blockIdx.x * blockDim.x + threadIdx.x) / blockDim.x;
  unsigned int col_gid = (blockIdx.x * blockDim.x + threadIdx.x) % blockDim.x;

  for (unsigned int row = row_gid; row < A_size1; row += gridDim.x)
    for (unsigned int col = col_gid; col < A_size2; col += blockDim.x)
      A[(row * A_inc1 + A_start1) * A_internal_size2 + col * A_inc2 + A_start2] = acos(B[(row * B_inc1 + B_start1) * B_internal_size2 + col * B_inc2 + B_start2]);
}


// asin
template<typename NumericT>
__global__ void matrix_row_element_asin_kernel(
          NumericT * A,
          unsigned int A_start1, unsigned int A_start2,
          unsigned int A_inc1,   unsigned int A_inc2,
          unsigned int A_size1,  unsigned int A_size2,
          unsigned int A_internal_size1,  unsigned int A_internal_size2,

          const NumericT * B,
          unsigned int B_start1, unsigned int B_start2,
          unsigned int B_inc1,   unsigned int B_inc2,
          unsigned int B_internal_size1,  unsigned int B_internal_size2)
{
  unsigned int row_gid = (blockIdx.x * blockDim.x + threadIdx.x) / blockDim.x;
  unsigned int col_gid = (blockIdx.x * blockDim.x + threadIdx.x) % blockDim.x;

  for (unsigned int row = row_gid; row < A_size1; row += gridDim.x)
    for (unsigned int col = col_gid; col < A_size2; col += blockDim.x)
      A[(row * A_inc1 + A_start1) * A_internal_size2 + col * A_inc2 + A_start2] = asin(B[(row * B_inc1 + B_start1) * B_internal_size2 + col * B_inc2 + B_start2]);
}


// atan
template<typename NumericT>
__global__ void matrix_row_element_atan_kernel(
          NumericT * A,
          unsigned int A_start1, unsigned int A_start2,
          unsigned int A_inc1,   unsigned int A_inc2,
          unsigned int A_size1,  unsigned int A_size2,
          unsigned int A_internal_size1,  unsigned int A_internal_size2,

          const NumericT * B,
          unsigned int B_start1, unsigned int B_start2,
          unsigned int B_inc1,   unsigned int B_inc2,
          unsigned int B_internal_size1,  unsigned int B_internal_size2)
{
  unsigned int row_gid = (blockIdx.x * blockDim.x + threadIdx.x) / blockDim.x;
  unsigned int col_gid = (blockIdx.x * blockDim.x + threadIdx.x) % blockDim.x;

  for (unsigned int row = row_gid; row < A_size1; row += gridDim.x)
    for (unsigned int col = col_gid; col < A_size2; col += blockDim.x)
      A[(row * A_inc1 + A_start1) * A_internal_size2 + col * A_inc2 + A_start2] = atan(B[(row * B_inc1 + B_start1) * B_internal_size2 + col * B_inc2 + B_start2]);
}


// ceil
template<typename NumericT>
__global__ void matrix_row_element_ceil_kernel(
          NumericT * A,
          unsigned int A_start1, unsigned int A_start2,
          unsigned int A_inc1,   unsigned int A_inc2,
          unsigned int A_size1,  unsigned int A_size2,
          unsigned int A_internal_size1,  unsigned int A_internal_size2,

          const NumericT * B,
          unsigned int B_start1, unsigned int B_start2,
          unsigned int B_inc1,   unsigned int B_inc2,
          unsigned int B_internal_size1,  unsigned int B_internal_size2)
{
  unsigned int row_gid = (blockIdx.x * blockDim.x + threadIdx.x) / blockDim.x;
  unsigned int col_gid = (blockIdx.x * blockDim.x + threadIdx.x) % blockDim.x;

  for (unsigned int row = row_gid; row < A_size1; row += gridDim.x)
    for (unsigned int col = col_gid; col < A_size2; col += blockDim.x)
      A[(row * A_inc1 + A_start1) * A_internal_size2 + col * A_inc2 + A_start2] = ceil(B[(row * B_inc1 + B_start1) * B_internal_size2 + col * B_inc2 + B_start2]);
}


// cos
template<typename NumericT>
__global__ void matrix_row_element_cos_kernel(
          NumericT * A,
          unsigned int A_start1, unsigned int A_start2,
          unsigned int A_inc1,   unsigned int A_inc2,
          unsigned int A_size1,  unsigned int A_size2,
          unsigned int A_internal_size1,  unsigned int A_internal_size2,

          const NumericT * B,
          unsigned int B_start1, unsigned int B_start2,
          unsigned int B_inc1,   unsigned int B_inc2,
          unsigned int B_internal_size1,  unsigned int B_internal_size2)
{
  unsigned int row_gid = (blockIdx.x * blockDim.x + threadIdx.x) / blockDim.x;
  unsigned int col_gid = (blockIdx.x * blockDim.x + threadIdx.x) % blockDim.x;

  for (unsigned int row = row_gid; row < A_size1; row += gridDim.x)
    for (unsigned int col = col_gid; col < A_size2; col += blockDim.x)
      A[(row * A_inc1 + A_start1) * A_internal_size2 + col * A_inc2 + A_start2] = cos(B[(row * B_inc1 + B_start1) * B_internal_size2 + col * B_inc2 + B_start2]);
}


// cosh
template<typename NumericT>
__global__ void matrix_row_element_cosh_kernel(
          NumericT * A,
          unsigned int A_start1, unsigned int A_start2,
          unsigned int A_inc1,   unsigned int A_inc2,
          unsigned int A_size1,  unsigned int A_size2,
          unsigned int A_internal_size1,  unsigned int A_internal_size2,

          const NumericT * B,
          unsigned int B_start1, unsigned int B_start2,
          unsigned int B_inc1,   unsigned int B_inc2,
          unsigned int B_internal_size1,  unsigned int B_internal_size2)
{
  unsigned int row_gid = (blockIdx.x * blockDim.x + threadIdx.x) / blockDim.x;
  unsigned int col_gid = (blockIdx.x * blockDim.x + threadIdx.x) % blockDim.x;

  for (unsigned int row = row_gid; row < A_size1; row += gridDim.x)
    for (unsigned int col = col_gid; col < A_size2; col += blockDim.x)
      A[(row * A_inc1 + A_start1) * A_internal_size2 + col * A_inc2 + A_start2] = cosh(B[(row * B_inc1 + B_start1) * B_internal_size2 + col * B_inc2 + B_start2]);
}


// exp
template<typename NumericT>
__global__ void matrix_row_element_exp_kernel(
          NumericT * A,
          unsigned int A_start1, unsigned int A_start2,
          unsigned int A_inc1,   unsigned int A_inc2,
          unsigned int A_size1,  unsigned int A_size2,
          unsigned int A_internal_size1,  unsigned int A_internal_size2,

          const NumericT * B,
          unsigned int B_start1, unsigned int B_start2,
          unsigned int B_inc1,   unsigned int B_inc2,
          unsigned int B_internal_size1,  unsigned int B_internal_size2)
{
  unsigned int row_gid = (blockIdx.x * blockDim.x + threadIdx.x) / blockDim.x;
  unsigned int col_gid = (blockIdx.x * blockDim.x + threadIdx.x) % blockDim.x;

  for (unsigned int row = row_gid; row < A_size1; row += gridDim.x)
    for (unsigned int col = col_gid; col < A_size2; col += blockDim.x)
      A[(row * A_inc1 + A_start1) * A_internal_size2 + col * A_inc2 + A_start2] = exp(B[(row * B_inc1 + B_start1) * B_internal_size2 + col * B_inc2 + B_start2]);
}


// fabs
template<typename NumericT>
__global__ void matrix_row_element_fabs_kernel(
          NumericT * A,
          unsigned int A_start1, unsigned int A_start2,
          unsigned int A_inc1,   unsigned int A_inc2,
          unsigned int A_size1,  unsigned int A_size2,
          unsigned int A_internal_size1,  unsigned int A_internal_size2,

          const NumericT * B,
          unsigned int B_start1, unsigned int B_start2,
          unsigned int B_inc1,   unsigned int B_inc2,
          unsigned int B_internal_size1,  unsigned int B_internal_size2)
{
  unsigned int row_gid = (blockIdx.x * blockDim.x + threadIdx.x) / blockDim.x;
  unsigned int col_gid = (blockIdx.x * blockDim.x + threadIdx.x) % blockDim.x;

  for (unsigned int row = row_gid; row < A_size1; row += gridDim.x)
    for (unsigned int col = col_gid; col < A_size2; col += blockDim.x)
      A[(row * A_inc1 + A_start1) * A_internal_size2 + col * A_inc2 + A_start2] = fabs(B[(row * B_inc1 + B_start1) * B_internal_size2 + col * B_inc2 + B_start2]);
}


// floor
template<typename NumericT>
__global__ void matrix_row_element_floor_kernel(
          NumericT * A,
          unsigned int A_start1, unsigned int A_start2,
          unsigned int A_inc1,   unsigned int A_inc2,
          unsigned int A_size1,  unsigned int A_size2,
          unsigned int A_internal_size1,  unsigned int A_internal_size2,

          const NumericT * B,
          unsigned int B_start1, unsigned int B_start2,
          unsigned int B_inc1,   unsigned int B_inc2,
          unsigned int B_internal_size1,  unsigned int B_internal_size2)
{
  unsigned int row_gid = (blockIdx.x * blockDim.x + threadIdx.x) / blockDim.x;
  unsigned int col_gid = (blockIdx.x * blockDim.x + threadIdx.x) % blockDim.x;

  for (unsigned int row = row_gid; row < A_size1; row += gridDim.x)
    for (unsigned int col = col_gid; col < A_size2; col += blockDim.x)
      A[(row * A_inc1 + A_start1) * A_internal_size2 + col * A_inc2 + A_start2] = floor(B[(row * B_inc1 + B_start1) * B_internal_size2 + col * B_inc2 + B_start2]);
}


// log
template<typename NumericT>
__global__ void matrix_row_element_log_kernel(
          NumericT * A,
          unsigned int A_start1, unsigned int A_start2,
          unsigned int A_inc1,   unsigned int A_inc2,
          unsigned int A_size1,  unsigned int A_size2,
          unsigned int A_internal_size1,  unsigned int A_internal_size2,

          const NumericT * B,
          unsigned int B_start1, unsigned int B_start2,
          unsigned int B_inc1,   unsigned int B_inc2,
          unsigned int B_internal_size1,  unsigned int B_internal_size2)
{
  unsigned int row_gid = (blockIdx.x * blockDim.x + threadIdx.x) / blockDim.x;
  unsigned int col_gid = (blockIdx.x * blockDim.x + threadIdx.x) % blockDim.x;

  for (unsigned int row = row_gid; row < A_size1; row += gridDim.x)
    for (unsigned int col = col_gid; col < A_size2; col += blockDim.x)
      A[(row * A_inc1 + A_start1) * A_internal_size2 + col * A_inc2 + A_start2] = log(B[(row * B_inc1 + B_start1) * B_internal_size2 + col * B_inc2 + B_start2]);
}


// log10
template<typename NumericT>
__global__ void matrix_row_element_log10_kernel(
          NumericT * A,
          unsigned int A_start1, unsigned int A_start2,
          unsigned int A_inc1,   unsigned int A_inc2,
          unsigned int A_size1,  unsigned int A_size2,
          unsigned int A_internal_size1,  unsigned int A_internal_size2,

          const NumericT * B,
          unsigned int B_start1, unsigned int B_start2,
          unsigned int B_inc1,   unsigned int B_inc2,
          unsigned int B_internal_size1,  unsigned int B_internal_size2)
{
  unsigned int row_gid = (blockIdx.x * blockDim.x + threadIdx.x) / blockDim.x;
  unsigned int col_gid = (blockIdx.x * blockDim.x + threadIdx.x) % blockDim.x;

  for (unsigned int row = row_gid; row < A_size1; row += gridDim.x)
    for (unsigned int col = col_gid; col < A_size2; col += blockDim.x)
      A[(row * A_inc1 + A_start1) * A_internal_size2 + col * A_inc2 + A_start2] = log10(B[(row * B_inc1 + B_start1) * B_internal_size2 + col * B_inc2 + B_start2]);
}


// sin
template<typename NumericT>
__global__ void matrix_row_element_sin_kernel(
          NumericT * A,
          unsigned int A_start1, unsigned int A_start2,
          unsigned int A_inc1,   unsigned int A_inc2,
          unsigned int A_size1,  unsigned int A_size2,
          unsigned int A_internal_size1,  unsigned int A_internal_size2,

          const NumericT * B,
          unsigned int B_start1, unsigned int B_start2,
          unsigned int B_inc1,   unsigned int B_inc2,
          unsigned int B_internal_size1,  unsigned int B_internal_size2)
{
  unsigned int row_gid = (blockIdx.x * blockDim.x + threadIdx.x) / blockDim.x;
  unsigned int col_gid = (blockIdx.x * blockDim.x + threadIdx.x) % blockDim.x;

  for (unsigned int row = row_gid; row < A_size1; row += gridDim.x)
    for (unsigned int col = col_gid; col < A_size2; col += blockDim.x)
      A[(row * A_inc1 + A_start1) * A_internal_size2 + col * A_inc2 + A_start2] = sin(B[(row * B_inc1 + B_start1) * B_internal_size2 + col * B_inc2 + B_start2]);
}


// sinh
template<typename NumericT>
__global__ void matrix_row_element_sinh_kernel(
          NumericT * A,
          unsigned int A_start1, unsigned int A_start2,
          unsigned int A_inc1,   unsigned int A_inc2,
          unsigned int A_size1,  unsigned int A_size2,
          unsigned int A_internal_size1,  unsigned int A_internal_size2,

          const NumericT * B,
          unsigned int B_start1, unsigned int B_start2,
          unsigned int B_inc1,   unsigned int B_inc2,
          unsigned int B_internal_size1,  unsigned int B_internal_size2)
{
  unsigned int row_gid = (blockIdx.x * blockDim.x + threadIdx.x) / blockDim.x;
  unsigned int col_gid = (blockIdx.x * blockDim.x + threadIdx.x) % blockDim.x;

  for (unsigned int row = row_gid; row < A_size1; row += gridDim.x)
    for (unsigned int col = col_gid; col < A_size2; col += blockDim.x)
      A[(row * A_inc1 + A_start1) * A_internal_size2 + col * A_inc2 + A_start2] = sinh(B[(row * B_inc1 + B_start1) * B_internal_size2 + col * B_inc2 + B_start2]);
}


// sqrt
template<typename NumericT>
__global__ void matrix_row_element_sqrt_kernel(
          NumericT * A,
          unsigned int A_start1, unsigned int A_start2,
          unsigned int A_inc1,   unsigned int A_inc2,
          unsigned int A_size1,  unsigned int A_size2,
          unsigned int A_internal_size1,  unsigned int A_internal_size2,

          const NumericT * B,
          unsigned int B_start1, unsigned int B_start2,
          unsigned int B_inc1,   unsigned int B_inc2,
          unsigned int B_internal_size1,  unsigned int B_internal_size2)
{
  unsigned int row_gid = (blockIdx.x * blockDim.x + threadIdx.x) / blockDim.x;
  unsigned int col_gid = (blockIdx.x * blockDim.x + threadIdx.x) % blockDim.x;

  for (unsigned int row = row_gid; row < A_size1; row += gridDim.x)
    for (unsigned int col = col_gid; col < A_size2; col += blockDim.x)
      A[(row * A_inc1 + A_start1) * A_internal_size2 + col * A_inc2 + A_start2] = sqrt(B[(row * B_inc1 + B_start1) * B_internal_size2 + col * B_inc2 + B_start2]);
}


// tan
template<typename NumericT>
__global__ void matrix_row_element_tan_kernel(
          NumericT * A,
          unsigned int A_start1, unsigned int A_start2,
          unsigned int A_inc1,   unsigned int A_inc2,
          unsigned int A_size1,  unsigned int A_size2,
          unsigned int A_internal_size1,  unsigned int A_internal_size2,

          const NumericT * B,
          unsigned int B_start1, unsigned int B_start2,
          unsigned int B_inc1,   unsigned int B_inc2,
          unsigned int B_internal_size1,  unsigned int B_internal_size2)
{
  unsigned int row_gid = (blockIdx.x * blockDim.x + threadIdx.x) / blockDim.x;
  unsigned int col_gid = (blockIdx.x * blockDim.x + threadIdx.x) % blockDim.x;

  for (unsigned int row = row_gid; row < A_size1; row += gridDim.x)
    for (unsigned int col = col_gid; col < A_size2; col += blockDim.x)
      A[(row * A_inc1 + A_start1) * A_internal_size2 + col * A_inc2 + A_start2] = tan(B[(row * B_inc1 + B_start1) * B_internal_size2 + col * B_inc2 + B_start2]);
}


// tanh
template<typename NumericT>
__global__ void matrix_row_element_tanh_kernel(
          NumericT * A,
          unsigned int A_start1, unsigned int A_start2,
          unsigned int A_inc1,   unsigned int A_inc2,
          unsigned int A_size1,  unsigned int A_size2,
          unsigned int A_internal_size1,  unsigned int A_internal_size2,

          const NumericT * B,
          unsigned int B_start1, unsigned int B_start2,
          unsigned int B_inc1,   unsigned int B_inc2,
          unsigned int B_internal_size1,  unsigned int B_internal_size2)
{
  unsigned int row_gid = (blockIdx.x * blockDim.x + threadIdx.x) / blockDim.x;
  unsigned int col_gid = (blockIdx.x * blockDim.x + threadIdx.x) % blockDim.x;

  for (unsigned int row = row_gid; row < A_size1; row += gridDim.x)
    for (unsigned int col = col_gid; col < A_size2; col += blockDim.x)
      A[(row * A_inc1 + A_start1) * A_internal_size2 + col * A_inc2 + A_start2] = tanh(B[(row * B_inc1 + B_start1) * B_internal_size2 + col * B_inc2 + B_start2]);
}



//
// matrix-vector product
//

template<typename NumericT>
__global__ void vec_mul_row_kernel(
          const NumericT * A,
          unsigned int A_row_start,
          unsigned int A_col_start,
          unsigned int A_row_inc,
          unsigned int A_col_inc,
          unsigned int A_row_size,
          unsigned int A_col_size,
          unsigned int A_internal_rows,
          unsigned int A_internal_cols,
          const NumericT * v,
          unsigned int v_start,
          unsigned int v_inc,
          unsigned int v_size,
          NumericT * result,
          unsigned int result_start,
          unsigned int result_inc,
          unsigned int result_size)
{
  __shared__ NumericT work[128];

  unsigned int row_gid = (blockIdx.x * blockDim.x + threadIdx.x) / blockDim.x;
  unsigned int col_gid = (blockIdx.x * blockDim.x + threadIdx.x) % blockDim.x;
  unsigned int lid = threadIdx.x;

  for (unsigned int row = row_gid; row < A_row_size; row += gridDim.x)
  {
    NumericT dot_prod = 0;
    for (unsigned int col = col_gid; col < A_col_size; col += blockDim.x)
      dot_prod += A[(row * A_row_inc + A_row_start) * A_internal_cols + col * A_col_inc + A_col_start] * v[v_start + v_inc * col];
    work[lid] = dot_prod;

    for (unsigned int stride = blockDim.x/2; stride>0; stride>>=1){
      __syncthreads();
      if (lid < stride)
        work[lid] += work[lid+stride];
    }

    if (lid == 0)
      result[row * result_inc + result_start] = work[0];
  }
}


template<typename NumericT>
__global__ void trans_vec_mul_row_kernel(
          const NumericT * A,
          unsigned int A_row_start,
          unsigned int A_col_start,
          unsigned int A_row_inc,
          unsigned int A_col_inc,
          unsigned int A_row_size,
          unsigned int A_col_size,
          unsigned int A_internal_rows,
          unsigned int A_internal_cols,
          const NumericT * v,
          unsigned int v_start,
          unsigned int v_inc,
          unsigned int v_size,
          NumericT * result,
          unsigned int result_start,
          unsigned int result_inc,
          unsigned int result_size)
{
  for (unsigned int row = blockIdx.x * blockDim.x + threadIdx.x; row < A_col_size; row += gridDim.x * blockDim.x)
  {
    NumericT dot_prod = 0;
    for (unsigned int col = 0; col < A_row_size; ++col)
      dot_prod += A[(row * A_col_inc + A_col_start) + (col * A_row_inc + A_row_start) * A_internal_cols] * v[v_start + v_inc * col];
    result[row * result_inc + result_start] = dot_prod;
  }
}


//
// matrix-matrix products
//




//
// scaled rank-1-update
//

// alpha on CPU
template<typename NumericT>
__global__ void scaled_rank1_update_row_kernel(
          NumericT * A,
          unsigned int A_start1, unsigned int A_start2,
          unsigned int A_inc1,   unsigned int A_inc2,
          unsigned int A_size1,  unsigned int A_size2,
          unsigned int A_internal_size1,  unsigned int A_internal_size2,

          NumericT val,
          unsigned int options2,

          const NumericT * vec1,
          unsigned int start1,
          unsigned int inc1,
          unsigned int size1,

          const NumericT * vec2,
          unsigned int start2,
          unsigned int inc2,
          unsigned int size2)
{
  NumericT alpha = val;
  if (options2 & (1 << 0))
    alpha = -alpha;
  if (options2 & (1 << 1))
    alpha = NumericT(1) / alpha;

  unsigned int row_gid = (blockIdx.x * blockDim.x + threadIdx.x) / blockDim.x;
  unsigned int col_gid = (blockIdx.x * blockDim.x + threadIdx.x) % blockDim.x;

  for (unsigned int row = row_gid; row < A_size1; row += gridDim.x)
  {
    NumericT tmp = alpha * vec1[row * inc1 + start1];
    for (unsigned int col = col_gid; col < A_size2; col += blockDim.x)
      A[(row * A_inc1 + A_start1) * A_internal_size2 + col * A_inc2 + A_start2] += tmp * vec2[col * inc2 + start2];
  }
}


// alpha on GPU
template<typename NumericT>
__global__ void scaled_rank1_update_row_kernel(
          NumericT * A,
          unsigned int A_start1, unsigned int A_start2,
          unsigned int A_inc1,   unsigned int A_inc2,
          unsigned int A_size1,  unsigned int A_size2,
          unsigned int A_internal_size1,  unsigned int A_internal_size2,

          const NumericT * val,
          unsigned int options2,

          const NumericT * vec1,
          unsigned int start1,
          unsigned int inc1,
          unsigned int size1,

          const NumericT * vec2,
          unsigned int start2,
          unsigned int inc2,
          unsigned int size2)
{
  NumericT alpha = *val;
  if (options2 & (1 << 0))
    alpha = -alpha;
  if (options2 & (1 << 1))
    alpha = NumericT(1) / alpha;

  unsigned int row_gid = (blockIdx.x * blockDim.x + threadIdx.x) / blockDim.x;
  unsigned int col_gid = (blockIdx.x * blockDim.x + threadIdx.x) % blockDim.x;

  for (unsigned int row = row_gid; row < A_size1; row += gridDim.x)
  {
    NumericT tmp = alpha * vec1[row * inc1 + start1];
    for (unsigned int col = col_gid; col < A_size2; col += blockDim.x)
      A[(row * A_inc1 + A_start1) * A_internal_size2 + col * A_inc2 + A_start2] += tmp * vec2[col * inc2 + start2];
  }
}



} // namespace cuda
} //namespace linalg
} //namespace viennacl


#endif
