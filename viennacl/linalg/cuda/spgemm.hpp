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

#include <stdexcept>

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
          IndexT *upper_bound_per_group)
{
  unsigned int subwarpsize_in_thread = 0;
  unsigned int max_nnz_row_A = 0;

  unsigned int rows_per_group = (A_size1 - 1) / gridDim.x + 1;
  unsigned int row_per_group_end = min(A_size1, rows_per_group * (blockIdx.x + 1));

  unsigned current_upper_bound = 0;
  for (unsigned int row = rows_per_group * blockIdx.x + threadIdx.x; row < row_per_group_end; row += blockDim.x)
  {
    unsigned int upper_bound_for_row = 0;
    unsigned int A_row_start = A_row_indices[row];
    unsigned int A_row_end   = A_row_indices[row+1];
    unsigned int row_num = A_row_end - A_row_start;
    if (row_num > 32) // too many rows in B need to be merged for a single pass
    {
      unsigned int subwarp_sqrt = (unsigned int)sqrt(double(row_num)) + 1;
      subwarpsize_in_thread = max(subwarp_sqrt, subwarpsize_in_thread);
    }
    else
      subwarpsize_in_thread = max(A_row_end - A_row_start, subwarpsize_in_thread);
    max_nnz_row_A = max(max_nnz_row_A, row_num);
    for (unsigned int j = A_row_start; j < A_row_end; ++j)
    {
      unsigned int col = A_col_indices[j];
      upper_bound_for_row += B_row_indices[col + 1] - B_row_indices[col];
    }
    current_upper_bound = max(upper_bound_for_row, current_upper_bound);
  }

  // reduction to obtain maximum in thread block
  __shared__ unsigned int shared_subwarpsize[256];
  __shared__ unsigned int shared_max_nnz_row_A[256];
  __shared__ unsigned int shared_upper_bounds[256];

    shared_subwarpsize[threadIdx.x] = subwarpsize_in_thread;
  shared_max_nnz_row_A[threadIdx.x] = max_nnz_row_A;
   shared_upper_bounds[threadIdx.x] = current_upper_bound;
  for (unsigned int stride = blockDim.x/2; stride > 0; stride /= 2)
  {
    __syncthreads();
    if (threadIdx.x < stride)
    {
        shared_subwarpsize[threadIdx.x] = max(  shared_subwarpsize[threadIdx.x],   shared_subwarpsize[threadIdx.x + stride]);
      shared_max_nnz_row_A[threadIdx.x] = max(shared_max_nnz_row_A[threadIdx.x], shared_max_nnz_row_A[threadIdx.x + stride]);
       shared_upper_bounds[threadIdx.x] = max( shared_upper_bounds[threadIdx.x],  shared_upper_bounds[threadIdx.x + stride]);
    }
  }

  if (threadIdx.x == 0)
  {
      subwarpsize_per_group[blockIdx.x] = round_to_next_power_of_2(shared_subwarpsize[0]);
    max_nnz_row_A_per_group[blockIdx.x] = shared_max_nnz_row_A[0];
      upper_bound_per_group[blockIdx.x] = shared_upper_bounds[0];
  }
}

//
// Stage 2: Determine sparsity pattern of C
//
__device__ unsigned int merge_subwarp_symbolic(unsigned int row_B_start, unsigned int row_B_end, unsigned int const *B_col_indices, unsigned int B_size2, unsigned int subwarpsize)
{
  unsigned int current_front_index = (row_B_start < row_B_end) ? B_col_indices[row_B_start] : B_size2;

  unsigned int num_nnz = 0;
  while (1)
  {
    // determine current minimum (warp shuffle)
    unsigned int min_index = current_front_index;
    for (unsigned int i = subwarpsize/2; i >= 1; i /= 2)
      min_index = min(min_index, __shfl_xor(min_index, i));

    if (min_index == B_size2)
      break;

    // update front:
    current_front_index = (current_front_index == min_index) ? ((++row_B_start < row_B_end) ? B_col_indices[row_B_start] : B_size2)
                                                             : current_front_index;
    ++num_nnz;
  }

  return num_nnz;
}

__device__ unsigned int merge_subwarp_symbolic_double(unsigned int row_B_start, unsigned int row_B_end, unsigned int const *B_col_indices, unsigned int B_size2,
                                                      unsigned int *output_array, unsigned int id_in_warp, unsigned int subwarpsize)
{
  unsigned int current_front_index = (row_B_start < row_B_end) ? B_col_indices[row_B_start] : B_size2;

  unsigned int num_nnz = 0;
  unsigned int index_buffer = 0;
  unsigned int buffer_size = 0;
  while (1)
  {
    // determine current minimum (warp shuffle)
    unsigned int min_index = current_front_index;
    for (unsigned int i = subwarpsize/2; i >= 1; i /= 2)
      min_index = min(min_index, __shfl_xor(min_index, i));

    if (min_index == B_size2)
      break;

    // update front:
    current_front_index = (current_front_index == min_index) ? ((++row_B_start < row_B_end) ? B_col_indices[row_B_start] : B_size2)
                                                             : current_front_index;

    // write output
    index_buffer = (id_in_warp == buffer_size) ? min_index : index_buffer;
    ++buffer_size;

    if (buffer_size == subwarpsize) // register buffer full?
    {
      output_array[id_in_warp] = index_buffer;
      output_array += subwarpsize;
      buffer_size = 0;
    }

    ++num_nnz;
  }

  // write remaining entries from register buffer:
  if (id_in_warp < buffer_size)
    output_array[id_in_warp] = index_buffer;

  return num_nnz;
}

template<typename IndexT>
__global__ void compressed_matrix_gemm_stage_2(
          const IndexT * A_row_indices,
          const IndexT * A_col_indices,
          IndexT A_size1,
          const IndexT * B_row_indices,
          const IndexT * B_col_indices,
          IndexT B_size2,
          IndexT * C_row_indices,
          unsigned int *upper_bound_row_C_len,
          unsigned int *subwarpsize_array,
          unsigned int *max_row_size,
          unsigned int *scratchpad_offsets,
          unsigned int *scratchpad_indices)
{
  unsigned int subwarpsize = subwarpsize_array[blockIdx.x];

  unsigned int num_warps  =  blockDim.x / subwarpsize;
  unsigned int warp_id    = threadIdx.x / subwarpsize;
  unsigned int id_in_warp = threadIdx.x % subwarpsize;

  unsigned int scratchpad_rowlength = upper_bound_row_C_len[blockIdx.x];
  unsigned int scratchpad_rows_per_warp = max_row_size[blockIdx.x] / subwarpsize + 1;
  unsigned int *subwarp_scratchpad_start = scratchpad_indices + scratchpad_offsets[blockIdx.x] + warp_id * scratchpad_rows_per_warp * scratchpad_rowlength;

  unsigned int rows_per_group = (A_size1 - 1) / gridDim.x + 1;
  unsigned int row_per_group_end = min(A_size1, rows_per_group * (blockIdx.x + 1));

  for (unsigned int row = rows_per_group * blockIdx.x + warp_id; row < row_per_group_end; row += num_warps)
  {
    unsigned int row_A_start = A_row_indices[row];
    unsigned int row_A_end   = A_row_indices[row+1];

    if (row_A_end - row_A_start > subwarpsize)
    {
      unsigned int final_merge_start = 0;
      unsigned int final_merge_end   = 0;

      // merge to temporary scratchpad memory:
      unsigned int *subwarp_scratchpad = subwarp_scratchpad_start;
      unsigned int iter = 0;
      while (row_A_end > row_A_start)
      {
        unsigned int my_row_B = row_A_start + id_in_warp;
        unsigned int row_B_index = (my_row_B < row_A_end) ? A_col_indices[my_row_B] : 0;
        unsigned int row_B_start = (my_row_B < row_A_end) ? B_row_indices[row_B_index] : 0;
        unsigned int row_B_end   = (my_row_B < row_A_end) ? B_row_indices[row_B_index + 1] : 0;

        unsigned int nnz_in_merge = merge_subwarp_symbolic_double(row_B_start, row_B_end, B_col_indices, B_size2,
                                                                  subwarp_scratchpad, id_in_warp, subwarpsize);

        final_merge_start = (iter == id_in_warp) ? subwarp_scratchpad - scratchpad_indices : final_merge_start;
        final_merge_end   = (iter == id_in_warp) ? final_merge_start + nnz_in_merge        : final_merge_end;
        ++iter;

        row_A_start += subwarpsize;
        subwarp_scratchpad += scratchpad_rowlength; // write to next row in scratchpad
      }

      // final merge:
      unsigned int num_nnz = merge_subwarp_symbolic(final_merge_start, final_merge_end, scratchpad_indices, B_size2, subwarpsize);

      if (id_in_warp == 0)
        C_row_indices[row] = num_nnz;
    }
    else
    {
      // single merge
      unsigned int my_row_B = row_A_start + id_in_warp;
      unsigned int row_B_index = (my_row_B < row_A_end) ? A_col_indices[my_row_B] : 0;
      unsigned int row_B_start = (my_row_B < row_A_end) ? B_row_indices[row_B_index] : 0;
      unsigned int row_B_end   = (my_row_B < row_A_end) ? B_row_indices[row_B_index + 1] : 0;

      unsigned int num_nnz = merge_subwarp_symbolic(row_B_start, row_B_end, B_col_indices, B_size2, subwarpsize);

      if (id_in_warp == 0)
        C_row_indices[row] = num_nnz;
    }
  }

}


//
// Stage 3: Fill C with values
//
template<typename NumericT>
__device__ unsigned int merge_subwarp_numeric(NumericT scaling_factor,
                                              unsigned int input_start, unsigned int input_end, const unsigned int *input_indices, const NumericT *input_values, unsigned int invalid_token,
                                              unsigned int *output_indices, NumericT *output_values,
                                              unsigned int id_in_warp, unsigned int subwarpsize)
{
  unsigned int current_front_index = (input_start < input_end) ? input_indices[input_start] : invalid_token;
  NumericT     current_front_value = (input_start < input_end) ? input_values[input_start]  : 0;

  unsigned int index_buffer = 0;
  NumericT     value_buffer = 0;
  unsigned int buffer_size = 0;
  unsigned int nnz_written = 0;
  while (1)
  {
    // determine current minimum:
    unsigned int min_index = current_front_index;
    for (unsigned int i = subwarpsize/2; i >= 1; i /= 2)
      min_index = min(min_index, __shfl_xor(min_index, i));

    if (min_index == invalid_token) // done
      break;

    // compute entry in C:
    NumericT output_value = (current_front_index == min_index) ? scaling_factor * current_front_value : 0;
    for (unsigned int i = subwarpsize/2; i >= 1; i /= 2)
      output_value += __shfl_xor(output_value, i);

    // update front:
    if (current_front_index == min_index)
    {
      ++input_start;
      current_front_index = (input_start < input_end) ? input_indices[input_start] : invalid_token;
      current_front_value = (input_start < input_end) ? input_values[input_start]  : 0;
    }

    // write current front to register buffer:
    index_buffer = (id_in_warp == buffer_size) ? min_index    : index_buffer;
    value_buffer = (id_in_warp == buffer_size) ? output_value : value_buffer;
    ++buffer_size;

    // flush register buffer via a coalesced write once full:
    if (buffer_size == subwarpsize)
    {
      output_indices[id_in_warp] = index_buffer; output_indices += subwarpsize;
      output_values[id_in_warp]  = value_buffer; output_values  += subwarpsize;
      buffer_size = 0;
    }

    ++nnz_written;
  }

  // write remaining entries in register buffer to C:
  if (id_in_warp < buffer_size)
  {
    output_indices[id_in_warp] = index_buffer;
    output_values[id_in_warp]  = value_buffer;
  }

  return nnz_written;
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
          IndexT const * C_row_indices,
          IndexT * C_col_indices,
          NumericT * C_elements,
          unsigned int *upper_bound_row_C_len,
          unsigned int *subwarpsize_array,
          unsigned int *max_row_size,
          unsigned int *scratchpad_offsets,
          unsigned int *scratchpad_indices,
          NumericT *scratchpad_values)
{
  unsigned int subwarpsize = subwarpsize_array[blockIdx.x];

  unsigned int num_warps  =  blockDim.x / subwarpsize;
  unsigned int warp_id    = threadIdx.x / subwarpsize;
  unsigned int id_in_warp = threadIdx.x % subwarpsize;

  unsigned int scratchpad_rowlength = upper_bound_row_C_len[blockIdx.x];
  unsigned int scratchpad_rows_per_warp = max_row_size[blockIdx.x] / subwarpsize + 1;
  unsigned int subwarp_scratchpad_shift = scratchpad_offsets[blockIdx.x] + warp_id * scratchpad_rows_per_warp * scratchpad_rowlength;

  unsigned int rows_per_group = (A_size1 - 1) / gridDim.x + 1;
  unsigned int row_per_group_end = min(A_size1, rows_per_group * (blockIdx.x + 1));

  for (unsigned int row = rows_per_group * blockIdx.x + warp_id; row < row_per_group_end; row += num_warps)
  {
    unsigned int row_A_start = A_row_indices[row];
    unsigned int row_A_end   = A_row_indices[row+1];

    if (row_A_end - row_A_start > subwarpsize)
    {
      // first merge stage:
      unsigned int final_merge_start = 0;
      unsigned int final_merge_end = 0;
      unsigned int iter = 0;
      unsigned int *scratchpad_indices_ptr = scratchpad_indices + subwarp_scratchpad_shift;
      NumericT     *scratchpad_values_ptr  = scratchpad_values  + subwarp_scratchpad_shift;

      while (row_A_start < row_A_end)
      {
        unsigned int my_row_B = row_A_start + id_in_warp;
        unsigned int row_B_index = (my_row_B < row_A_end) ? A_col_indices[my_row_B] : 0;
        unsigned int row_B_start = (my_row_B < row_A_end) ? B_row_indices[row_B_index] : 0;
        unsigned int row_B_end   = (my_row_B < row_A_end) ? B_row_indices[row_B_index + 1] : 0;
        NumericT val_A = (my_row_B < row_A_end) ? A_elements[my_row_B] : 0;

        unsigned int nnz_written = merge_subwarp_numeric(val_A,
                                                         row_B_start, row_B_end, B_col_indices, B_elements, B_size2,
                                                         scratchpad_indices_ptr, scratchpad_values_ptr,
                                                         id_in_warp, subwarpsize);

        if (iter == id_in_warp)
        {
          final_merge_start = scratchpad_indices_ptr - scratchpad_indices;
          final_merge_end   = final_merge_start + nnz_written;
        }
        ++iter;

        row_A_start += subwarpsize;
        scratchpad_indices_ptr += scratchpad_rowlength;
        scratchpad_values_ptr  += scratchpad_rowlength;
      }

      // second merge stage:
      unsigned int index_in_C = C_row_indices[row];
      merge_subwarp_numeric(NumericT(1),
                            final_merge_start, final_merge_end, scratchpad_indices, scratchpad_values, B_size2,
                            C_col_indices + index_in_C, C_elements + index_in_C,
                            id_in_warp, subwarpsize);
    }
    else
    {
      unsigned int my_row_B = row_A_start + id_in_warp;
      unsigned int row_B_index = (my_row_B < row_A_end) ? A_col_indices[my_row_B] : 0;
      unsigned int row_B_start = (my_row_B < row_A_end) ? B_row_indices[row_B_index] : 0;
      unsigned int row_B_end   = (my_row_B < row_A_end) ? B_row_indices[row_B_index + 1] : 0;
      NumericT val_A = (my_row_B < row_A_end) ? A_elements[my_row_B] : 0;

      unsigned int index_in_C = C_row_indices[row];

      merge_subwarp_numeric(val_A,
                            row_B_start, row_B_end, B_col_indices, B_elements, B_size2,
                            C_col_indices + index_in_C, C_elements + index_in_C,
                            id_in_warp, subwarpsize);
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

  viennacl::vector<unsigned int> subwarp_sizes(256, viennacl::traits::context(A)); // upper bound for the nonzeros per row encountered for each work group
  viennacl::vector<unsigned int> max_nnz_row_A(256, viennacl::traits::context(A)); // upper bound for the nonzeros per row encountered for each work group
  viennacl::vector<unsigned int> upper_bound_nonzeros_per_row_C(256, viennacl::traits::context(A)); // upper bound for the nonzeros per row encountered for each work group

  //
  // Stage 1: Determine upper bound for number of nonzeros
  //
  compressed_matrix_gemm_stage_1<<<256, 256>>>(detail::cuda_arg<unsigned int>(A.handle1().cuda_handle()),
                                               detail::cuda_arg<unsigned int>(A.handle2().cuda_handle()),
                                               static_cast<unsigned int>(A.size1()),
                                               detail::cuda_arg<unsigned int>(B.handle1().cuda_handle()),
                                               detail::cuda_arg<unsigned int>(subwarp_sizes),
                                               detail::cuda_arg<unsigned int>(max_nnz_row_A),
                                               detail::cuda_arg<unsigned int>(upper_bound_nonzeros_per_row_C)
                                              );
  VIENNACL_CUDA_LAST_ERROR_CHECK("compressed_matrix_gemm_stage_1");

  upper_bound_nonzeros_per_row_C.switch_memory_context(viennacl::context(MAIN_MEMORY));
  unsigned int * upper_bound_nonzeros_per_row_C_ptr = viennacl::linalg::host_based::detail::extract_raw_pointer<unsigned int>(upper_bound_nonzeros_per_row_C.handle());

  unsigned int max_nnz_per_row_C = 0;
  for (std::size_t i=0; i<upper_bound_nonzeros_per_row_C.size(); ++i)
    max_nnz_per_row_C = (max_nnz_per_row_C < upper_bound_nonzeros_per_row_C_ptr[i]) ? upper_bound_nonzeros_per_row_C_ptr[i] : max_nnz_per_row_C;
  max_nnz_per_row_C = std::max(max_nnz_per_row_C, static_cast<unsigned int>(B.size2()));


  subwarp_sizes.switch_memory_context(viennacl::context(MAIN_MEMORY));
  unsigned int * subwarp_sizes_ptr = viennacl::linalg::host_based::detail::extract_raw_pointer<unsigned int>(subwarp_sizes.handle());

  max_nnz_row_A.switch_memory_context(viennacl::context(MAIN_MEMORY));
  unsigned int const * max_nnz_row_A_ptr = viennacl::linalg::host_based::detail::extract_raw_pointer<unsigned int>(max_nnz_row_A.handle());

  //std::cout << "Subwarp sizes: " << subwarp_sizes << std::endl;

  viennacl::vector<unsigned int> scratchpad_offsets(256, viennacl::context(MAIN_MEMORY)); // upper bound for the nonzeros per row encountered for each work group
  unsigned int * scratchpad_offsets_ptr = viennacl::linalg::host_based::detail::extract_raw_pointer<unsigned int>(scratchpad_offsets.handle());

  unsigned int max_subwarp_size = 0;
  unsigned int scratchpad_offset = 0;
  //std::cout << "Scratchpad offsets: " << std::endl;
  for (std::size_t i=0; i<subwarp_sizes.size(); ++i)
  {
    max_subwarp_size = std::max(max_subwarp_size, subwarp_sizes_ptr[i]);

    scratchpad_offsets_ptr[i] = scratchpad_offset;
    //std::cout << scratchpad_offset << " (with " << (max_nnz_row_A_ptr[i] / subwarp_sizes_ptr[i] + 1) << " warp reloads per group at " << max_nnz_row_A_ptr[i] << " max rows, "
    //                                            << upper_bound_nonzeros_per_row_C_ptr[i] << " row length, "
    //                                            << (256 / subwarp_sizes_ptr[i]) << " warps per group " << std::endl;
    scratchpad_offset += (max_nnz_row_A_ptr[i] / subwarp_sizes_ptr[i] + 1) // maximum number of warp reloads in group
                        * upper_bound_nonzeros_per_row_C_ptr[i]            // row length
                        * (256 / subwarp_sizes_ptr[i]);                    // number of warps in group
  }
  //std::cout << "Scratchpad memory for indices: " << scratchpad_offset << " entries (" << scratchpad_offset * sizeof(unsigned int) * 1e-6 << " MB)" << std::endl;

  if (max_subwarp_size > 32)
    throw std::runtime_error("Subwarp size too large!");

  upper_bound_nonzeros_per_row_C.switch_memory_context(viennacl::traits::context(A));
  subwarp_sizes.switch_memory_context(viennacl::traits::context(A));
  max_nnz_row_A.switch_memory_context(viennacl::traits::context(A));
  scratchpad_offsets.switch_memory_context(viennacl::traits::context(A));

  viennacl::vector<unsigned int> scratchpad_indices(scratchpad_offset, viennacl::traits::context(A)); // upper bound for the nonzeros per row encountered for each work group

  //
  // Stage 2: Determine pattern of C
  //

  compressed_matrix_gemm_stage_2<<<256, 256>>>(detail::cuda_arg<unsigned int>(A.handle1().cuda_handle()),
                                               detail::cuda_arg<unsigned int>(A.handle2().cuda_handle()),
                                               static_cast<unsigned int>(A.size1()),
                                               detail::cuda_arg<unsigned int>(B.handle1().cuda_handle()),
                                               detail::cuda_arg<unsigned int>(B.handle2().cuda_handle()),
                                               static_cast<unsigned int>(B.size2()),
                                               detail::cuda_arg<unsigned int>(C.handle1().cuda_handle()),
                                               detail::cuda_arg<unsigned int>(upper_bound_nonzeros_per_row_C),
                                               detail::cuda_arg<unsigned int>(subwarp_sizes),
                                               detail::cuda_arg<unsigned int>(max_nnz_row_A),
                                               detail::cuda_arg<unsigned int>(scratchpad_offsets),
                                               detail::cuda_arg<unsigned int>(scratchpad_indices)
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

  viennacl::vector<NumericT> scratchpad_values(scratchpad_offset, viennacl::traits::context(A)); // upper bound for the nonzeros per row encountered for each work group

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
                                               detail::cuda_arg<NumericT>(C.handle().cuda_handle()),
                                               detail::cuda_arg<unsigned int>(upper_bound_nonzeros_per_row_C),
                                               detail::cuda_arg<unsigned int>(subwarp_sizes),
                                               detail::cuda_arg<unsigned int>(max_nnz_row_A),
                                               detail::cuda_arg<unsigned int>(scratchpad_offsets),
                                               detail::cuda_arg<unsigned int>(scratchpad_indices),
                                               detail::cuda_arg<NumericT>(scratchpad_values)
                                              );
  VIENNACL_CUDA_LAST_ERROR_CHECK("compressed_matrix_gemm_stage_3");

}

} // namespace cuda
} //namespace linalg
} //namespace viennacl


#endif
