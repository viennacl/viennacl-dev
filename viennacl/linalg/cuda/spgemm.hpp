#ifndef VIENNACL_LINALG_CUDA_SPGEMM_HPP_
#define VIENNACL_LINALG_CUDA_SPGEMM_HPP_

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
// Stage 1: Obtain upper bound for number of elements per row in C:
//
template<typename IndexT>
__global__ void compressed_matrix_gemm_stage_1(
          const IndexT * A_row_indices,
          const IndexT * A_col_indices,
          IndexT A_size1,
          const IndexT * B_row_indices,
          IndexT * upper_bound_per_group)
{
  unsigned int current_upper_bound = 0;

  for (unsigned int row = blockDim.x * blockIdx.x + threadIdx.x;
                    row < A_size1;
                    row += gridDim.x * blockDim.x)
  {
    unsigned int upper_bound_for_row = 0;
    unsigned int A_row_end = A_row_indices[row+1];
    for (unsigned int j = A_row_indices[row]; j < A_row_end; ++j)
    {
      unsigned int col = A_col_indices[j];
      upper_bound_for_row += B_row_indices[col + 1] - B_row_indices[col];
    }
    current_upper_bound = (upper_bound_for_row > current_upper_bound) ? upper_bound_for_row : current_upper_bound;
  }

  // reduction to obtain maximum in thread block
  __shared__ unsigned int shared_upper_bounds[256];

  shared_upper_bounds[threadIdx.x] = current_upper_bound;
  for (unsigned int stride = blockDim.x/2; stride > 0; stride /= 2)
  {
    __syncthreads();
    if (threadIdx.x < stride)
      shared_upper_bounds[threadIdx.x] = max(shared_upper_bounds[threadIdx.x], shared_upper_bounds[threadIdx.x + stride]);
  }

  if (threadIdx.x == 0)
    upper_bound_per_group[blockIdx.x] = shared_upper_bounds[0];
}

//
// Stage 2: Determine sparsity pattern of C
//

template<typename NumericT>
__device__ NumericT minimum_in_buffer(NumericT *shared_buffer)
{
  for (unsigned int stride = blockDim.x/2; stride > 0; stride /= 2)
  {
    __syncthreads();
    if (threadIdx.x < stride)
      shared_buffer[threadIdx.x] = min(shared_buffer[threadIdx.x], shared_buffer[threadIdx.x + stride]);
  }
  __syncthreads();
  return shared_buffer[0];
}


template<typename IndexT>
__global__ void compressed_matrix_gemm_stage_2(
          const IndexT * A_row_indices,
          const IndexT * A_col_indices,
          IndexT A_size1,
          IndexT A_nnz,
          const IndexT * B_row_indices,
          const IndexT * B_col_indices,
          IndexT B_size1,
          IndexT B_size2,
          IndexT * C_row_indices,
          IndexT * scratchpad_for_groups,
          IndexT scratchpad_size_per_group)
{
  //unsigned int warp_id = threadIdx.x / 32;
  //unsigned int * scratchpad = scratchpad_for_groups + (8 * blockIdx.x + warp_id) * scratchpad_size_per_group;

  unsigned int rows_per_group = (A_size1 - 1) / gridDim.x + 1;
  unsigned int row_per_group_end = min(A_size1, rows_per_group * (blockIdx.x + 1));

  for (unsigned int row = rows_per_group * blockIdx.x; row < row_per_group_end; ++row)
  {
    unsigned int row_A_start = A_row_indices[row];
    unsigned int row_A_end   = A_row_indices[row+1];

    unsigned int my_row_B = row_A_start + threadIdx.x;
    unsigned int row_B_index = (my_row_B < row_A_end) ? A_col_indices[my_row_B] : 0;
    unsigned int row_B_start = (my_row_B < row_A_end) ? B_row_indices[row_B_index] : 0;
    unsigned int row_B_end   = (my_row_B < row_A_end) ? B_row_indices[row_B_index + 1] : 0;

    // merge loop to determine the number of nonzeros in C:
    __shared__ unsigned int shared_fronts[256];

    unsigned int current_front_index = (row_B_start < row_B_end) ? B_col_indices[row_B_start] : B_size2;

    unsigned int num_nnz = 0;
    while (1)
    {
      // determine current minimum:
      shared_fronts[threadIdx.x] = current_front_index;
      unsigned int min_index = minimum_in_buffer(shared_fronts);

      if (min_index == B_size2)
        break;

      // update front:
      if (current_front_index == min_index)
      {
        ++row_B_start;
        current_front_index = (row_B_start < row_B_end) ? B_col_indices[row_B_start] : B_size2;
      }

      __syncthreads(); // prevent divergence

      ++num_nnz;
    }

    if (threadIdx.x == 0)
      C_row_indices[row] = num_nnz;
  }

}


//
// Stage 3: Fill C with values
//

template<typename NumericT>
__device__ NumericT sum_buffer(NumericT *shared_buffer)
{
  for (unsigned int stride = blockDim.x/2; stride > 0; stride /= 2)
  {
    __syncthreads();
    if (threadIdx.x < stride)
      shared_buffer[threadIdx.x] += shared_buffer[threadIdx.x + stride];
  }
  __syncthreads();
  return shared_buffer[0];
}


template<typename IndexT, typename NumericT>
__global__ void compressed_matrix_gemm_stage_3(
          const IndexT * A_row_indices,
          const IndexT * A_col_indices,
          const NumericT * A_elements,
          IndexT A_size1,
          const IndexT * B_row_indices,
          const IndexT * B_col_indices,
          const NumericT * B_elements,
          IndexT B_size2,
          IndexT * C_row_indices,
          IndexT * C_col_indices,
          NumericT * C_elements
    )
{
  //unsigned int warp_id = threadIdx.x / 32;
  //unsigned int * scratchpad = scratchpad_for_groups + (8 * blockIdx.x + warp_id) * scratchpad_size_per_group;

  unsigned int rows_per_group = (A_size1 - 1) / gridDim.x + 1;
  unsigned int row_per_group_end = min(A_size1, rows_per_group * (blockIdx.x + 1));

  for (unsigned int row = rows_per_group * blockIdx.x; row < row_per_group_end; ++row)
  {
    unsigned int row_A_start = A_row_indices[row];
    unsigned int row_A_end   = A_row_indices[row+1];

    unsigned int my_row_B = row_A_start + threadIdx.x;
    unsigned int row_B_index = (my_row_B < row_A_end) ? A_col_indices[my_row_B] : 0;
    unsigned int row_B_start = (my_row_B < row_A_end) ? B_row_indices[row_B_index] : 0;
    unsigned int row_B_end   = (my_row_B < row_A_end) ? B_row_indices[row_B_index + 1] : 0;
    NumericT val_A = (my_row_B < row_A_end) ? A_elements[my_row_B] : 0;

    // merge loop to determine the number of nonzeros in C:
    __shared__ unsigned int shared_indices[256];
    __shared__ NumericT shared_values[256];

    unsigned int current_front_index = (row_B_start < row_B_end) ? B_col_indices[row_B_start] : B_size2;
    NumericT current_front_value     = (row_B_start < row_B_end) ? B_elements[row_B_start]    : 0;

    unsigned int index_in_C = C_row_indices[row];
    while (1)
    {
      // determine current minimum:
      shared_indices[threadIdx.x] = current_front_index;
      unsigned int min_index = minimum_in_buffer(shared_indices);

      if (min_index == B_size2)
        break;

      // compute entry in C:
      shared_values[threadIdx.x] = (current_front_index == min_index) ? val_A * current_front_value : 0;
      NumericT C_value = sum_buffer(shared_values);

      // update front:
      if (current_front_index == min_index)
      {
        ++row_B_start;
        current_front_index = (row_B_start < row_B_end) ? B_col_indices[row_B_start] : B_size2;
        current_front_value = (row_B_start < row_B_end) ? B_elements[row_B_start]    : 0;
      }

      if (threadIdx.x == blockDim.x - 1) // this thread is most likely not busy
      {
        C_col_indices[index_in_C] = min_index;
        C_elements[index_in_C] = C_value;
      }
      __syncthreads();

      ++index_in_C;
    }
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
template<class NumericT, unsigned int AlignmentV>
void prod_impl(viennacl::compressed_matrix<NumericT, AlignmentV> const & A,
               viennacl::compressed_matrix<NumericT, AlignmentV> const & B,
               viennacl::compressed_matrix<NumericT, AlignmentV> & C)
{
  C.resize(A.size1(), B.size2(), false);

  viennacl::vector<unsigned int> upper_bound_nonzeros_per_row_C(256, viennacl::traits::context(A)); // upper bound for the nonzeros per row encountered for each work group

  //
  // Stage 1: Determine upper bound for number of nonzeros
  //
  compressed_matrix_gemm_stage_1<<<256, 256>>>(detail::cuda_arg<unsigned int>(A.handle1().cuda_handle()),
                                               detail::cuda_arg<unsigned int>(A.handle2().cuda_handle()),
                                               static_cast<unsigned int>(A.size1()),
                                               detail::cuda_arg<unsigned int>(B.handle1().cuda_handle()),
                                               detail::cuda_arg<unsigned int>(upper_bound_nonzeros_per_row_C)
                                              );
  VIENNACL_CUDA_LAST_ERROR_CHECK("compressed_matrix_gemm_stage_1");

  upper_bound_nonzeros_per_row_C.switch_memory_context(viennacl::context(MAIN_MEMORY));
  unsigned int * upper_bound_nonzeros_per_row_C_ptr = viennacl::linalg::host_based::detail::extract_raw_pointer<unsigned int>(upper_bound_nonzeros_per_row_C.handle());

  unsigned int max_nnz_per_row_C = 0;
  for (std::size_t i=0; i<upper_bound_nonzeros_per_row_C.size(); ++i)
    max_nnz_per_row_C = (max_nnz_per_row_C < upper_bound_nonzeros_per_row_C_ptr[i]) ? upper_bound_nonzeros_per_row_C_ptr[i] : max_nnz_per_row_C;
  max_nnz_per_row_C = std::max(max_nnz_per_row_C, static_cast<unsigned int>(B.size2()));

  //
  // Stage 2: Determine pattern of C
  //
  //viennacl::vector<unsigned int> scratchpad_memory(8 * 256 * max_nnz_per_row_C, viennacl::traits::context(A)); // 8 warps max per group
  viennacl::vector<unsigned int> scratchpad_memory(2, viennacl::traits::context(A)); // 8 warps max per group

  compressed_matrix_gemm_stage_2<<<256, 256>>>(detail::cuda_arg<unsigned int>(A.handle1().cuda_handle()),
                                               detail::cuda_arg<unsigned int>(A.handle2().cuda_handle()),
                                               static_cast<unsigned int>(A.size1()),
                                               static_cast<unsigned int>(A.nnz()),
                                               detail::cuda_arg<unsigned int>(B.handle1().cuda_handle()),
                                               detail::cuda_arg<unsigned int>(B.handle2().cuda_handle()),
                                               static_cast<unsigned int>(B.size1()),
                                               static_cast<unsigned int>(B.size2()),
                                               detail::cuda_arg<unsigned int>(C.handle1().cuda_handle()),
                                               detail::cuda_arg<unsigned int>(scratchpad_memory),
                                               max_nnz_per_row_C
                                              );
  VIENNACL_CUDA_LAST_ERROR_CHECK("compressed_matrix_gemm_stage_2");

  // exclusive scan on C.handle1(), ultimately allowing to allocate remaining memory for C
  viennacl::backend::typesafe_host_array<unsigned int> row_buffer(C.handle1(), C.size1() + 1);
  viennacl::backend::memory_read(C.handle1(), 0, row_buffer.raw_size(), row_buffer.get());
  unsigned int current_offset = 0;
  for (std::size_t i=0; i<C.size1(); ++i)
  {
    unsigned int tmp = row_buffer[i];
    row_buffer.set(i, current_offset);
    current_offset += tmp;
  }
  row_buffer.set(C.size1(), current_offset);
  viennacl::backend::memory_write(C.handle1(), 0, row_buffer.raw_size(), row_buffer.get());

  //
  // Stage 3: Compute entries in C
  //
  C.reserve(current_offset, false);

  compressed_matrix_gemm_stage_3<<<256, 256>>>(detail::cuda_arg<unsigned int>(A.handle1().cuda_handle()),
                                               detail::cuda_arg<unsigned int>(A.handle2().cuda_handle()),
                                               detail::cuda_arg<NumericT>(A.handle().cuda_handle()),
                                               static_cast<unsigned int>(A.size1()),
                                               detail::cuda_arg<unsigned int>(B.handle1().cuda_handle()),
                                               detail::cuda_arg<unsigned int>(B.handle2().cuda_handle()),
                                               detail::cuda_arg<NumericT>(B.handle().cuda_handle()),
                                               static_cast<unsigned int>(B.size2()),
                                               detail::cuda_arg<unsigned int>(C.handle1().cuda_handle()),
                                               detail::cuda_arg<unsigned int>(C.handle2().cuda_handle()),
                                               detail::cuda_arg<NumericT>(C.handle().cuda_handle())
                                              );
  VIENNACL_CUDA_LAST_ERROR_CHECK("compressed_matrix_gemm_stage_3");

}

} // namespace cuda
} //namespace linalg
} //namespace viennacl


#endif
