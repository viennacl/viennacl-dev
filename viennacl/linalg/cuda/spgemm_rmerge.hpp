#ifndef VIENNACL_LINALG_CUDA_SPGEMM_RMERGE_HPP_
#define VIENNACL_LINALG_CUDA_SPGEMM_RMERGE_HPP_

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

#include <stdexcept>

#include "viennacl/forwards.h"
#include "viennacl/scalar.hpp"
#include "viennacl/vector.hpp"
#include "viennacl/tools/tools.hpp"
#include "viennacl/linalg/cuda/common.hpp"

#include "viennacl/tools/timer.hpp"

#include "viennacl/linalg/cuda/sparse_matrix_operations_solve.hpp"

namespace viennacl
{
namespace linalg
{
namespace cuda
{

/** @brief Loads a value from the specified address. With CUDA arch 3.5 and above the value is also stored in global constant memory for later reuse */
template<typename NumericT>
static inline __device__ NumericT load_and_cache(const NumericT *address)
{
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 350
  return __ldg(address);
#else
  return *address;
#endif
}


//
// Stage 1: Obtain upper bound for number of elements per row in C:
//
template<typename IndexT>
__device__ IndexT round_to_next_power_of_2(IndexT val)
{
  if (val > 32)
    return 64; // just to indicate that we need to split/factor the matrix!
  else if (val > 16)
    return 32;
  else if (val > 8)
    return 16;
  else if (val > 4)
    return 8;
  else if (val > 2)
    return 4;
  else if (val > 1)
    return 2;
  else
    return 1;
}

template<typename IndexT>
__global__ void compressed_matrix_gemm_stage_1(
          const IndexT * A_row_indices,
          const IndexT * A_col_indices,
          IndexT A_size1,
          const IndexT * B_row_indices,
          IndexT *subwarpsize_per_group,
          IndexT *max_nnz_row_A_per_group,
          IndexT *max_nnz_row_B_per_group)
{
  unsigned int subwarpsize_in_thread = 0;
  unsigned int max_nnz_row_A = 0;
  unsigned int max_nnz_row_B = 0;

  unsigned int rows_per_group = (A_size1 - 1) / gridDim.x + 1;
  unsigned int row_per_group_end = min(A_size1, rows_per_group * (blockIdx.x + 1));

  for (unsigned int row = rows_per_group * blockIdx.x + threadIdx.x; row < row_per_group_end; row += blockDim.x)
  {
    unsigned int A_row_start = A_row_indices[row];
    unsigned int A_row_end   = A_row_indices[row+1];
    unsigned int row_num = A_row_end - A_row_start;
    subwarpsize_in_thread = max(A_row_end - A_row_start, subwarpsize_in_thread);
    max_nnz_row_A = max(max_nnz_row_A, row_num);
    for (unsigned int j = A_row_start; j < A_row_end; ++j)
    {
      unsigned int col = A_col_indices[j];
      unsigned int row_len_B = B_row_indices[col + 1] - B_row_indices[col];
      max_nnz_row_B = max(row_len_B, max_nnz_row_B);
    }
  }

  // reduction to obtain maximum in thread block
  __shared__ unsigned int shared_subwarpsize[256];
  __shared__ unsigned int shared_max_nnz_row_A[256];
  __shared__ unsigned int shared_max_nnz_row_B[256];

    shared_subwarpsize[threadIdx.x] = subwarpsize_in_thread;
  shared_max_nnz_row_A[threadIdx.x] = max_nnz_row_A;
  shared_max_nnz_row_B[threadIdx.x] = max_nnz_row_B;
  for (unsigned int stride = blockDim.x/2; stride > 0; stride /= 2)
  {
    __syncthreads();
    if (threadIdx.x < stride)
    {
        shared_subwarpsize[threadIdx.x] = max(  shared_subwarpsize[threadIdx.x],   shared_subwarpsize[threadIdx.x + stride]);
      shared_max_nnz_row_A[threadIdx.x] = max(shared_max_nnz_row_A[threadIdx.x], shared_max_nnz_row_A[threadIdx.x + stride]);
      shared_max_nnz_row_B[threadIdx.x] = max(shared_max_nnz_row_B[threadIdx.x], shared_max_nnz_row_B[threadIdx.x + stride]);
    }
  }

  if (threadIdx.x == 0)
  {
      subwarpsize_per_group[blockIdx.x] = round_to_next_power_of_2(shared_subwarpsize[0]);
    max_nnz_row_A_per_group[blockIdx.x] = shared_max_nnz_row_A[0];
    max_nnz_row_B_per_group[blockIdx.x] = shared_max_nnz_row_B[0];
  }
}

//
// Stage 2: Determine sparsity pattern of C
//
template<unsigned int SubWarpSizeV, typename IndexT>
__global__ void compressed_matrix_gemm_stage_2(
          const IndexT * A_row_indices,
          const IndexT * A_col_indices,
          IndexT A_size1,
          const IndexT * B_row_indices,
          const IndexT * B_col_indices,
          IndexT B_size2,
          IndexT * C_row_indices)
{
  unsigned int num_warps  =  blockDim.x / SubWarpSizeV;
  unsigned int warp_id    = threadIdx.x / SubWarpSizeV;
  unsigned int id_in_warp = threadIdx.x % SubWarpSizeV;

  unsigned int rows_per_group = (A_size1 - 1) / gridDim.x + 1;
  unsigned int row_per_group_end = min(A_size1, rows_per_group * (blockIdx.x + 1));

  for (unsigned int row = rows_per_group * blockIdx.x + warp_id; row < row_per_group_end; row += num_warps)
  {
    unsigned int row_A_start = A_row_indices[row];
    unsigned int row_A_end   = A_row_indices[row+1];

    // single merge
    unsigned int my_row_B = row_A_start + id_in_warp;
    unsigned int row_B_index = (my_row_B < row_A_end) ? A_col_indices[my_row_B] : 0;
    unsigned int row_B_start = (my_row_B < row_A_end) ? load_and_cache(B_row_indices + row_B_index) : 0;
    unsigned int row_B_end   = (my_row_B < row_A_end) ? load_and_cache(B_row_indices + row_B_index + 1) : 0;

    unsigned int current_front_index = (row_B_start < row_B_end) ? load_and_cache(B_col_indices + row_B_start) : B_size2;

    unsigned int num_nnz = 0;
    while (1)
    {
      // determine current minimum (warp shuffle)
      unsigned int min_index = current_front_index;
      for (unsigned int i = SubWarpSizeV/2; i >= 1; i /= 2)
        min_index = min(min_index, __shfl_xor(min_index, i));

      if (min_index == B_size2)
        break;

      // update front:
      if (current_front_index == min_index)
      {
        ++row_B_start;
        current_front_index = (row_B_start < row_B_end) ? load_and_cache(B_col_indices + row_B_start) : B_size2;
      }

      ++num_nnz;
    }

    if (id_in_warp == 0)
      C_row_indices[row] = num_nnz;
  }

}


//
// Stage 3: Fill C with values
//


template<unsigned int SubWarpSizeV, typename IndexT, typename NumericT>
__global__ void compressed_matrix_gemm_stage_3(
          const IndexT * A_row_indices,
          const IndexT * A_col_indices,
          const NumericT * A_elements,
          IndexT A_size1,
          const IndexT * B_row_indices,
          const IndexT * B_col_indices,
          const NumericT * B_elements,
          IndexT B_size2,
          IndexT const * C_row_indices,
          IndexT * C_col_indices,
          NumericT * C_elements)
{
  unsigned int num_warps  =  blockDim.x / SubWarpSizeV;
  unsigned int warp_id    = threadIdx.x / SubWarpSizeV;
  unsigned int id_in_warp = threadIdx.x % SubWarpSizeV;

  unsigned int rows_per_group = (A_size1 - 1) / gridDim.x + 1;
  unsigned int row_per_group_end = min(A_size1, rows_per_group * (blockIdx.x + 1));

  for (unsigned int row = rows_per_group * blockIdx.x + warp_id; row < row_per_group_end; row += num_warps)
  {
    unsigned int row_A_start = A_row_indices[row];
    unsigned int row_A_end   = A_row_indices[row+1];

    unsigned int my_row_B = row_A_start + id_in_warp;
    unsigned int row_B_index = (my_row_B < row_A_end) ? A_col_indices[my_row_B] : 0;
    unsigned int row_B_start = (my_row_B < row_A_end) ? load_and_cache(B_row_indices + row_B_index)     : 0;
    unsigned int row_B_end   = (my_row_B < row_A_end) ? load_and_cache(B_row_indices + row_B_index + 1) : 0;
    NumericT val_A = (my_row_B < row_A_end) ? A_elements[my_row_B] : 0;

    unsigned int index_in_C = C_row_indices[row];

    unsigned int current_front_index = (row_B_start < row_B_end) ? load_and_cache(B_col_indices + row_B_start) : B_size2;
    NumericT     current_front_value = (row_B_start < row_B_end) ? load_and_cache(B_elements    + row_B_start) : 0;

    unsigned int index_buffer = 0;
    NumericT     value_buffer = 0;
    unsigned int buffer_size = 0;
    while (1)
    {
      // determine current minimum:
      unsigned int min_index = current_front_index;
      for (unsigned int i = SubWarpSizeV/2; i >= 1; i /= 2)
        min_index = min(min_index, __shfl_xor(min_index, i));

      if (min_index == B_size2) // done
        break;

      // compute entry in C:
      NumericT output_value = (current_front_index == min_index) ? val_A * current_front_value : 0;
      for (unsigned int i = SubWarpSizeV/2; i >= 1; i /= 2)
        output_value += __shfl_xor(output_value, i);

      // update front:
      if (current_front_index == min_index)
      {
        ++row_B_start;
        current_front_index = (row_B_start < row_B_end) ? load_and_cache(B_col_indices + row_B_start) : B_size2;
        current_front_value = (row_B_start < row_B_end) ? load_and_cache(B_elements    + row_B_start) : 0;
      }

      // write current front to register buffer:
      index_buffer = (id_in_warp == buffer_size) ? min_index    : index_buffer;
      value_buffer = (id_in_warp == buffer_size) ? output_value : value_buffer;
      ++buffer_size;

      // flush register buffer via a coalesced write once full:
      if (buffer_size == SubWarpSizeV)
      {
        C_col_indices[index_in_C + id_in_warp] = index_buffer;
        C_elements[index_in_C + id_in_warp]    = value_buffer;
        buffer_size = 0;
        index_in_C += SubWarpSizeV;
      }
    }

    // write remaining entries in register buffer to C:
    if (id_in_warp < buffer_size)
    {
      C_col_indices[index_in_C + id_in_warp] = index_buffer;
      C_elements[index_in_C + id_in_warp]  = value_buffer;
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

  unsigned int blocknum = 256;
  unsigned int threadnum = 128;

  viennacl::vector<unsigned int> subwarp_sizes(blocknum, viennacl::traits::context(A)); // upper bound for the nonzeros per row encountered for each work group
  viennacl::vector<unsigned int> max_nnz_row_A(blocknum, viennacl::traits::context(A)); // upper bound for the nonzeros per row encountered for each work group
  viennacl::vector<unsigned int> max_nnz_row_B(blocknum, viennacl::traits::context(A)); // upper bound for the nonzeros per row encountered for each work group

  //
  // Stage 1: Determine upper bound for number of nonzeros
  //
  compressed_matrix_gemm_stage_1<<<blocknum, threadnum>>>(detail::cuda_arg<unsigned int>(A.handle1().cuda_handle()),
                                                          detail::cuda_arg<unsigned int>(A.handle2().cuda_handle()),
                                                          static_cast<unsigned int>(A.size1()),
                                                          detail::cuda_arg<unsigned int>(B.handle1().cuda_handle()),
                                                          detail::cuda_arg<unsigned int>(subwarp_sizes),
                                                          detail::cuda_arg<unsigned int>(max_nnz_row_A),
                                                          detail::cuda_arg<unsigned int>(max_nnz_row_B)
                                                         );
  VIENNACL_CUDA_LAST_ERROR_CHECK("compressed_matrix_gemm_stage_1");

  subwarp_sizes.switch_memory_context(viennacl::context(MAIN_MEMORY));
  unsigned int * subwarp_sizes_ptr = viennacl::linalg::host_based::detail::extract_raw_pointer<unsigned int>(subwarp_sizes.handle());

  max_nnz_row_A.switch_memory_context(viennacl::context(MAIN_MEMORY));
  unsigned int const * max_nnz_row_A_ptr = viennacl::linalg::host_based::detail::extract_raw_pointer<unsigned int>(max_nnz_row_A.handle());

  max_nnz_row_B.switch_memory_context(viennacl::context(MAIN_MEMORY));
  unsigned int const * max_nnz_row_B_ptr = viennacl::linalg::host_based::detail::extract_raw_pointer<unsigned int>(max_nnz_row_B.handle());

  unsigned int max_subwarp_size = 0;
  //std::cout << "Scratchpad offsets: " << std::endl;
  for (std::size_t i=0; i<subwarp_sizes.size(); ++i)
    max_subwarp_size = std::max(max_subwarp_size, subwarp_sizes_ptr[i]);

  if (max_subwarp_size > 32)
    throw std::runtime_error("Subwarp size too large!");

  std::cout << "Running RMerge with subwarp size " << max_subwarp_size << std::endl;

  subwarp_sizes.switch_memory_context(viennacl::traits::context(A));
  max_nnz_row_A.switch_memory_context(viennacl::traits::context(A));
  max_nnz_row_B.switch_memory_context(viennacl::traits::context(A));

  //
  // Stage 2: Determine pattern of C
  //

  if (max_subwarp_size == 32)
  {
    compressed_matrix_gemm_stage_2<32><<<blocknum, threadnum>>>(detail::cuda_arg<unsigned int>(A.handle1().cuda_handle()),
                                                           detail::cuda_arg<unsigned int>(A.handle2().cuda_handle()),
                                                           static_cast<unsigned int>(A.size1()),
                                                           detail::cuda_arg<unsigned int>(B.handle1().cuda_handle()),
                                                           detail::cuda_arg<unsigned int>(B.handle2().cuda_handle()),
                                                           static_cast<unsigned int>(B.size2()),
                                                           detail::cuda_arg<unsigned int>(C.handle1().cuda_handle())
                                                          );
    VIENNACL_CUDA_LAST_ERROR_CHECK("compressed_matrix_gemm_stage_2");
  }
  else if (max_subwarp_size == 16)
  {
    compressed_matrix_gemm_stage_2<16><<<blocknum, threadnum>>>(detail::cuda_arg<unsigned int>(A.handle1().cuda_handle()),
                                                           detail::cuda_arg<unsigned int>(A.handle2().cuda_handle()),
                                                           static_cast<unsigned int>(A.size1()),
                                                           detail::cuda_arg<unsigned int>(B.handle1().cuda_handle()),
                                                           detail::cuda_arg<unsigned int>(B.handle2().cuda_handle()),
                                                           static_cast<unsigned int>(B.size2()),
                                                           detail::cuda_arg<unsigned int>(C.handle1().cuda_handle())
                                                          );
    VIENNACL_CUDA_LAST_ERROR_CHECK("compressed_matrix_gemm_stage_2");
  }
  else
  {
    compressed_matrix_gemm_stage_2<8><<<blocknum, threadnum>>>(detail::cuda_arg<unsigned int>(A.handle1().cuda_handle()),
                                                           detail::cuda_arg<unsigned int>(A.handle2().cuda_handle()),
                                                           static_cast<unsigned int>(A.size1()),
                                                           detail::cuda_arg<unsigned int>(B.handle1().cuda_handle()),
                                                           detail::cuda_arg<unsigned int>(B.handle2().cuda_handle()),
                                                           static_cast<unsigned int>(B.size2()),
                                                           detail::cuda_arg<unsigned int>(C.handle1().cuda_handle())
                                                          );
    VIENNACL_CUDA_LAST_ERROR_CHECK("compressed_matrix_gemm_stage_2");
  }

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

  if (max_subwarp_size == 32)
  {
    compressed_matrix_gemm_stage_3<32><<<blocknum, threadnum>>>(detail::cuda_arg<unsigned int>(A.handle1().cuda_handle()),
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
  else if (max_subwarp_size == 16)
  {
    compressed_matrix_gemm_stage_3<16><<<blocknum, threadnum>>>(detail::cuda_arg<unsigned int>(A.handle1().cuda_handle()),
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
  else
  {
    compressed_matrix_gemm_stage_3<8><<<blocknum, threadnum>>>(detail::cuda_arg<unsigned int>(A.handle1().cuda_handle()),
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

}

} // namespace cuda
} //namespace linalg
} //namespace viennacl


#endif
