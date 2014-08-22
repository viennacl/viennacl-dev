#ifndef VIENNACL_LINALG_OPENCL_KERNELS_ITERATIVE_HPP
#define VIENNACL_LINALG_OPENCL_KERNELS_ITERATIVE_HPP

#include "viennacl/tools/tools.hpp"

#include "viennacl/vector_proxy.hpp"

#include "viennacl/scheduler/forwards.h"
#include "viennacl/scheduler/io.hpp"
#include "viennacl/scheduler/preset.hpp"

#include "viennacl/ocl/kernel.hpp"
#include "viennacl/ocl/platform.hpp"
#include "viennacl/ocl/utils.hpp"

#include "viennacl/device_specific/builtin_database/vector_axpy.hpp"
#include "viennacl/device_specific/builtin_database/reduction.hpp"

/** @file viennacl/linalg/opencl/kernels/iterative.hpp
 *  @brief OpenCL kernel file for specialized iterative solver kernels */
namespace viennacl
{
namespace linalg
{
namespace opencl
{
namespace kernels
{
//////////////////////////// Part 1: Kernel generation routines ////////////////////////////////////

template<typename StringT>
void generate_pipelined_cg_vector_update(StringT & source, std::string const & numeric_string)
{
  source.append("__kernel void cg_vector_update( \n");
  source.append("  __global "); source.append(numeric_string); source.append(" * result, \n");
  source.append("  "); source.append(numeric_string); source.append(" alpha, \n");
  source.append("  __global "); source.append(numeric_string); source.append(" * p, \n");
  source.append("  __global "); source.append(numeric_string); source.append(" * r, \n");
  source.append("  __global "); source.append(numeric_string); source.append(" const * Ap, \n");
  source.append("  "); source.append(numeric_string); source.append(" beta, \n");
  source.append("  __global "); source.append(numeric_string); source.append(" * inner_prod_buffer, \n");
  source.append("  unsigned int size, \n");
  source.append("  __local "); source.append(numeric_string); source.append(" * shared_array) \n");
  source.append("{ \n");
  source.append("  "); source.append(numeric_string); source.append(" inner_prod_contrib = 0; \n");
  source.append("  for (unsigned int i = get_global_id(0); i < size; i += get_global_size(0)) { \n");
  source.append("    "); source.append(numeric_string); source.append(" value_p = p[i]; \n");
  source.append("    "); source.append(numeric_string); source.append(" value_r = r[i]; \n");
  source.append("     \n");
  source.append("    result[i] += alpha * value_p; \n");
  source.append("    value_r   -= alpha * Ap[i]; \n");
  source.append("    value_p    = value_r + beta * value_p; \n");
  source.append("     \n");
  source.append("    p[i] = value_p; \n");
  source.append("    r[i] = value_r; \n");
  source.append("    inner_prod_contrib += value_r * value_r; \n");
  source.append("  }  \n");

  // parallel reduction in work group
  source.append("  shared_array[get_local_id(0)] = inner_prod_contrib; \n");
  source.append("  for (uint stride=get_local_size(0)/2; stride > 0; stride /= 2) \n");
  source.append("  { \n");
  source.append("    barrier(CLK_LOCAL_MEM_FENCE); \n");
  source.append("    if (get_local_id(0) < stride)  \n");
  source.append("      shared_array[get_local_id(0)] += shared_array[get_local_id(0) + stride];  \n");
  source.append("  } ");

  // write results to result array
  source.append(" if (get_local_id(0) == 0) \n ");
  source.append("   inner_prod_buffer[get_group_id(0)] = shared_array[0]; ");

  source.append("} \n");
}

template<typename StringT>
void generate_compressed_matrix_pipelined_cg_prod(StringT & source, std::string const & numeric_string)
{
  source.append("__kernel void cg_csr_prod( \n");
  source.append("  __global const unsigned int * row_indices, \n");
  source.append("  __global const unsigned int * column_indices, \n");
  source.append("  __global const "); source.append(numeric_string); source.append(" * elements, \n");
  source.append("  __global const "); source.append(numeric_string); source.append(" * p, \n");
  source.append("  __global "); source.append(numeric_string); source.append(" * Ap, \n");
  source.append("  unsigned int size, \n");
  source.append("  __global "); source.append(numeric_string); source.append(" * inner_prod_buffer, \n");
  source.append("  unsigned int buffer_size, \n");
  source.append("  __local "); source.append(numeric_string); source.append(" * shared_array_ApAp, \n");
  source.append("  __local "); source.append(numeric_string); source.append(" * shared_array_pAp) \n");
  source.append("{ \n");
  source.append("  "); source.append(numeric_string); source.append(" inner_prod_ApAp = 0; \n");
  source.append("  "); source.append(numeric_string); source.append(" inner_prod_pAp = 0; \n");
  source.append("  for (unsigned int row = get_global_id(0); row < size; row += get_global_size(0)) \n");
  source.append("  { \n");
  source.append("    "); source.append(numeric_string); source.append(" dot_prod = ("); source.append(numeric_string); source.append(")0; \n");
  source.append("    unsigned int row_end = row_indices[row+1]; \n");
  source.append("    for (unsigned int i = row_indices[row]; i < row_end; ++i) \n");
  source.append("      dot_prod += elements[i] * p[column_indices[i]]; \n");
  source.append("    Ap[row] = dot_prod; \n");
  source.append("    inner_prod_ApAp += dot_prod * dot_prod; \n");
  source.append("    inner_prod_pAp  +=   p[row] * dot_prod; \n");
  source.append("  } \n");

  // parallel reduction in work group
  source.append("  shared_array_ApAp[get_local_id(0)] = inner_prod_ApAp; \n");
  source.append("  shared_array_pAp[get_local_id(0)]  = inner_prod_pAp; \n");
  source.append("  for (uint stride=get_local_size(0)/2; stride > 0; stride /= 2) \n");
  source.append("  { \n");
  source.append("    barrier(CLK_LOCAL_MEM_FENCE); \n");
  source.append("    if (get_local_id(0) < stride) { \n");
  source.append("      shared_array_ApAp[get_local_id(0)] += shared_array_ApAp[get_local_id(0) + stride];  \n");
  source.append("      shared_array_pAp[get_local_id(0)]  += shared_array_pAp[get_local_id(0) + stride];  \n");
  source.append("    } ");
  source.append("  } ");

  // write results to result array
  source.append("  if (get_local_id(0) == 0) { \n ");
  source.append("    inner_prod_buffer[  buffer_size + get_group_id(0)] = shared_array_ApAp[0]; \n");
  source.append("    inner_prod_buffer[2*buffer_size + get_group_id(0)] = shared_array_pAp[0]; \n");
  source.append("  } \n");

  source.append("} \n \n");

}

template<typename StringT>
void generate_coordinate_matrix_pipelined_cg_prod(StringT & source, std::string const & numeric_string)
{
  source.append("__kernel void cg_coo_prod( \n");
  source.append("  __global const uint2 * coords,  \n");//(row_index, column_index)
  source.append("  __global const "); source.append(numeric_string); source.append(" * elements, \n");
  source.append("  __global const uint  * group_boundaries, \n");
  source.append("  __global const "); source.append(numeric_string); source.append(" * p, \n");
  source.append("  __global "); source.append(numeric_string); source.append(" * Ap, \n");
  source.append("  unsigned int size, \n");
  source.append("  __local unsigned int * shared_rows, \n");
  source.append("  __local "); source.append(numeric_string); source.append(" * inter_results, \n");
  source.append("  __global "); source.append(numeric_string); source.append(" * inner_prod_buffer, \n");
  source.append("  unsigned int buffer_size, \n");
  source.append("  __local "); source.append(numeric_string); source.append(" * shared_array_ApAp, \n");
  source.append("  __local "); source.append(numeric_string); source.append(" * shared_array_pAp) \n");
  source.append("{ \n");
  source.append("  "); source.append(numeric_string); source.append(" inner_prod_ApAp = 0; \n");
  source.append("  "); source.append(numeric_string); source.append(" inner_prod_pAp = 0; \n");

  ///////////// Sparse matrix-vector multiplication part /////////////
  source.append("  uint2 tmp; \n");
  source.append("  "); source.append(numeric_string); source.append(" val; \n");
  source.append("  uint group_start = group_boundaries[get_group_id(0)]; \n");
  source.append("  uint group_end   = group_boundaries[get_group_id(0) + 1]; \n");
  source.append("  uint k_end = (group_end > group_start) ? 1 + (group_end - group_start - 1) / get_local_size(0) : 0; \n");   // -1 in order to have correct behavior if group_end - group_start == j * get_local_size(0)

  source.append("  uint local_index = 0; \n");

  source.append("  for (uint k = 0; k < k_end; ++k) { \n");
  source.append("    local_index = group_start + k * get_local_size(0) + get_local_id(0); \n");

  source.append("    tmp = (local_index < group_end) ? coords[local_index] : (uint2) 0; \n");
  source.append("    val = (local_index < group_end) ? elements[local_index] * p[tmp.y] : 0; \n");

  //check for carry from previous loop run:
  source.append("    if (get_local_id(0) == 0 && k > 0) { \n");
  source.append("      if (tmp.x == shared_rows[get_local_size(0)-1]) \n");
  source.append("        val += inter_results[get_local_size(0)-1]; \n");
  source.append("      else {\n");
  source.append("        "); source.append(numeric_string); source.append(" Ap_entry = inter_results[get_local_size(0)-1]; \n");
  source.append("        Ap[shared_rows[get_local_size(0)-1]] = Ap_entry; \n");
  source.append("        inner_prod_ApAp += Ap_entry * Ap_entry; \n");
  source.append("        inner_prod_pAp  += p[shared_rows[get_local_size(0)-1]] * Ap_entry; \n");
  source.append("      } \n");
  source.append("    } \n");

  //segmented parallel reduction begin
  source.append("    barrier(CLK_LOCAL_MEM_FENCE); \n");
  source.append("    shared_rows[get_local_id(0)] = tmp.x; \n");
  source.append("    inter_results[get_local_id(0)] = val; \n");
  source.append("    "); source.append(numeric_string); source.append(" left = 0; \n");
  source.append("    barrier(CLK_LOCAL_MEM_FENCE); \n");

  source.append("    for (unsigned int stride = 1; stride < get_local_size(0); stride *= 2) { \n");
  source.append("      left = (get_local_id(0) >= stride && tmp.x == shared_rows[get_local_id(0) - stride]) ? inter_results[get_local_id(0) - stride] : 0; \n");
  source.append("      barrier(CLK_LOCAL_MEM_FENCE); \n");
  source.append("      inter_results[get_local_id(0)] += left; \n");
  source.append("      barrier(CLK_LOCAL_MEM_FENCE); \n");
  source.append("    } \n");
  //segmented parallel reduction end

  source.append("    if (local_index < group_end && get_local_id(0) < get_local_size(0) - 1 && \n");
  source.append("      shared_rows[get_local_id(0)] != shared_rows[get_local_id(0) + 1]) { \n");
  source.append("      "); source.append(numeric_string); source.append(" Ap_entry = inter_results[get_local_id(0)]; \n");
  source.append("      Ap[tmp.x] = Ap_entry; \n");
  source.append("      inner_prod_ApAp += Ap_entry * Ap_entry; \n");
  source.append("      inner_prod_pAp  += p[tmp.x] * Ap_entry; \n");
  source.append("    } \n");

  source.append("    barrier(CLK_LOCAL_MEM_FENCE); \n");
  source.append("  }  \n"); //for k

  source.append("  if (local_index + 1 == group_end) {\n");  //write results of last active entry (this may not necessarily be the case already)
  source.append("    "); source.append(numeric_string); source.append(" Ap_entry = inter_results[get_local_id(0)]; \n");
  source.append("    Ap[tmp.x] = Ap_entry; \n");
  source.append("    inner_prod_ApAp += Ap_entry * Ap_entry; \n");
  source.append("    inner_prod_pAp  += p[tmp.x] * Ap_entry; \n");
  source.append("  }  \n");

  //////////// parallel reduction of inner product contributions within work group ///////////////
  source.append("  shared_array_ApAp[get_local_id(0)] = inner_prod_ApAp; \n");
  source.append("  shared_array_pAp[get_local_id(0)]  = inner_prod_pAp; \n");
  source.append("  for (uint stride=get_local_size(0)/2; stride > 0; stride /= 2) \n");
  source.append("  { \n");
  source.append("    barrier(CLK_LOCAL_MEM_FENCE); \n");
  source.append("    if (get_local_id(0) < stride) { \n");
  source.append("      shared_array_ApAp[get_local_id(0)] += shared_array_ApAp[get_local_id(0) + stride];  \n");
  source.append("      shared_array_pAp[get_local_id(0)]  += shared_array_pAp[get_local_id(0) + stride];  \n");
  source.append("    } ");
  source.append("  } ");

  // write results to result array
  source.append("  if (get_local_id(0) == 0) { \n ");
  source.append("    inner_prod_buffer[  buffer_size + get_group_id(0)] = shared_array_ApAp[0]; \n");
  source.append("    inner_prod_buffer[2*buffer_size + get_group_id(0)] = shared_array_pAp[0]; \n");
  source.append("  } \n");

  source.append("} \n \n");

}


template<typename StringT>
void generate_ell_matrix_pipelined_cg_prod(StringT & source, std::string const & numeric_string)
{
  source.append("__kernel void cg_ell_prod( \n");
  source.append("  __global const unsigned int * coords, \n");
  source.append("  __global const "); source.append(numeric_string); source.append(" * elements, \n");
  source.append("  unsigned int internal_row_num, \n");
  source.append("  unsigned int items_per_row, \n");
  source.append("  unsigned int aligned_items_per_row, \n");
  source.append("  __global const "); source.append(numeric_string); source.append(" * p, \n");
  source.append("  __global "); source.append(numeric_string); source.append(" * Ap, \n");
  source.append("  unsigned int size, \n");
  source.append("  __global "); source.append(numeric_string); source.append(" * inner_prod_buffer, \n");
  source.append("  unsigned int buffer_size, \n");
  source.append("  __local "); source.append(numeric_string); source.append(" * shared_array_ApAp, \n");
  source.append("  __local "); source.append(numeric_string); source.append(" * shared_array_pAp) \n");
  source.append("{ \n");
  source.append("  "); source.append(numeric_string); source.append(" inner_prod_ApAp = 0; \n");
  source.append("  "); source.append(numeric_string); source.append(" inner_prod_pAp = 0; \n");
  source.append("  uint glb_id = get_global_id(0); \n");
  source.append("  uint glb_sz = get_global_size(0); \n");

  source.append("  for (uint row = glb_id; row < size; row += glb_sz) { \n");
  source.append("    "); source.append(numeric_string); source.append(" sum = 0; \n");

  source.append("    uint offset = row; \n");
  source.append("    for (uint item_id = 0; item_id < items_per_row; item_id++, offset += internal_row_num) { \n");
  source.append("      "); source.append(numeric_string); source.append(" val = elements[offset]; \n");
  source.append("      sum += val ? p[coords[offset]] * val : ("); source.append(numeric_string); source.append(")0; \n");
  source.append("    } \n");

  source.append("    Ap[row] = sum; \n");
  source.append("    inner_prod_ApAp += sum * sum; \n");
  source.append("    inner_prod_pAp  += p[row] * sum; \n");
  source.append("  }  \n");

  //////////// parallel reduction of inner product contributions within work group ///////////////
  source.append("  shared_array_ApAp[get_local_id(0)] = inner_prod_ApAp; \n");
  source.append("  shared_array_pAp[get_local_id(0)]  = inner_prod_pAp; \n");
  source.append("  for (uint stride=get_local_size(0)/2; stride > 0; stride /= 2) \n");
  source.append("  { \n");
  source.append("    barrier(CLK_LOCAL_MEM_FENCE); \n");
  source.append("    if (get_local_id(0) < stride) { \n");
  source.append("      shared_array_ApAp[get_local_id(0)] += shared_array_ApAp[get_local_id(0) + stride];  \n");
  source.append("      shared_array_pAp[get_local_id(0)]  += shared_array_pAp[get_local_id(0) + stride];  \n");
  source.append("    } ");
  source.append("  } ");

  // write results to result array
  source.append("  if (get_local_id(0) == 0) { \n ");
  source.append("    inner_prod_buffer[  buffer_size + get_group_id(0)] = shared_array_ApAp[0]; \n");
  source.append("    inner_prod_buffer[2*buffer_size + get_group_id(0)] = shared_array_pAp[0]; \n");
  source.append("  } \n");
  source.append("} \n \n");
}

template<typename StringT>
void generate_sliced_ell_matrix_pipelined_cg_prod(StringT & source, std::string const & numeric_string)
{
  source.append("__kernel void cg_sliced_ell_prod( \n");
  source.append("  __global const unsigned int * columns_per_block, \n");
  source.append("  __global const unsigned int * column_indices, \n");
  source.append("  __global const unsigned int * block_start, \n");
  source.append("  __global const "); source.append(numeric_string); source.append(" * elements, \n");
  source.append("  __global const "); source.append(numeric_string); source.append(" * p, \n");
  source.append("  __global "); source.append(numeric_string); source.append(" * Ap, \n");
  source.append("  unsigned int size, \n");
  source.append("  __global "); source.append(numeric_string); source.append(" * inner_prod_buffer, \n");
  source.append("  unsigned int buffer_size, \n");
  source.append("  __local "); source.append(numeric_string); source.append(" * shared_array_ApAp, \n");
  source.append("  __local "); source.append(numeric_string); source.append(" * shared_array_pAp) \n");
  source.append("{ \n");
  source.append("  "); source.append(numeric_string); source.append(" inner_prod_ApAp = 0; \n");
  source.append("  "); source.append(numeric_string); source.append(" inner_prod_pAp = 0; \n");
  source.append("  uint local_id   = get_local_id(0); \n");
  source.append("  uint local_size = get_local_size(0); \n");

  source.append("  for (uint block_idx = get_group_id(0); block_idx <= size / local_size; block_idx += get_num_groups(0)) { \n");
  source.append("    "); source.append(numeric_string); source.append(" sum = 0; \n");

  source.append("    uint row    = block_idx * local_size + local_id; \n");
  source.append("    uint offset = block_start[block_idx]; \n");
  source.append("    uint num_columns = columns_per_block[block_idx]; \n");
  source.append("    for (uint item_id = 0; item_id < num_columns; item_id++) { \n");
  source.append("      uint index = offset + item_id * local_size + local_id; \n");
  source.append("      "); source.append(numeric_string); source.append(" val = elements[index]; \n");
  source.append("      sum += val ? (p[column_indices[index]] * val) : 0; \n");
  source.append("    } \n");

  source.append("    if (row < size) {\n");
  source.append("      Ap[row] = sum; \n");
  source.append("      inner_prod_ApAp += sum * sum; \n");
  source.append("      inner_prod_pAp  += p[row] * sum; \n");
  source.append("    }  \n");
  source.append("  }  \n");

  //////////// parallel reduction of inner product contributions within work group ///////////////
  source.append("  shared_array_ApAp[get_local_id(0)] = inner_prod_ApAp; \n");
  source.append("  shared_array_pAp[get_local_id(0)]  = inner_prod_pAp; \n");
  source.append("  for (uint stride=get_local_size(0)/2; stride > 0; stride /= 2) \n");
  source.append("  { \n");
  source.append("    barrier(CLK_LOCAL_MEM_FENCE); \n");
  source.append("    if (get_local_id(0) < stride) { \n");
  source.append("      shared_array_ApAp[get_local_id(0)] += shared_array_ApAp[get_local_id(0) + stride];  \n");
  source.append("      shared_array_pAp[get_local_id(0)]  += shared_array_pAp[get_local_id(0) + stride];  \n");
  source.append("    } ");
  source.append("  } ");

  // write results to result array
  source.append("  if (get_local_id(0) == 0) { \n ");
  source.append("    inner_prod_buffer[  buffer_size + get_group_id(0)] = shared_array_ApAp[0]; \n");
  source.append("    inner_prod_buffer[2*buffer_size + get_group_id(0)] = shared_array_pAp[0]; \n");
  source.append("  } \n");
  source.append("} \n \n");
}

template<typename StringT>
void generate_hyb_matrix_pipelined_cg_prod(StringT & source, std::string const & numeric_string)
{
  source.append("__kernel void cg_hyb_prod( \n");
  source.append("  const __global int* ell_coords, \n");
  source.append("  const __global "); source.append(numeric_string); source.append("* ell_elements, \n");
  source.append("  const __global uint* csr_rows, \n");
  source.append("  const __global uint* csr_cols, \n");
  source.append("  const __global "); source.append(numeric_string); source.append("* csr_elements, \n");
  source.append("  unsigned int internal_row_num, \n");
  source.append("  unsigned int items_per_row, \n");
  source.append("  unsigned int aligned_items_per_row, \n");
  source.append("  __global const "); source.append(numeric_string); source.append(" * p, \n");
  source.append("  __global "); source.append(numeric_string); source.append(" * Ap, \n");
  source.append("  unsigned int size, \n");
  source.append("  __global "); source.append(numeric_string); source.append(" * inner_prod_buffer, \n");
  source.append("  unsigned int buffer_size, \n");
  source.append("  __local "); source.append(numeric_string); source.append(" * shared_array_ApAp, \n");
  source.append("  __local "); source.append(numeric_string); source.append(" * shared_array_pAp) \n");
  source.append("{ \n");
  source.append("  "); source.append(numeric_string); source.append(" inner_prod_ApAp = 0; \n");
  source.append("  "); source.append(numeric_string); source.append(" inner_prod_pAp = 0; \n");
  source.append("  uint glb_id = get_global_id(0); \n");
  source.append("  uint glb_sz = get_global_size(0); \n");

  source.append("  for (uint row = glb_id; row < size; row += glb_sz) { \n");
  source.append("    "); source.append(numeric_string); source.append(" sum = 0; \n");

  source.append("    uint offset = row; \n");
  source.append("    for (uint item_id = 0; item_id < items_per_row; item_id++, offset += internal_row_num) { \n");
  source.append("      "); source.append(numeric_string); source.append(" val = ell_elements[offset]; \n");
  source.append("      sum += val ? (p[ell_coords[offset]] * val) : 0; \n");
  source.append("    } \n");

  source.append("    uint col_begin = csr_rows[row]; \n");
  source.append("    uint col_end   = csr_rows[row + 1]; \n");

  source.append("    for (uint item_id = col_begin; item_id < col_end; item_id++) {  \n");
  source.append("      sum += (p[csr_cols[item_id]] * csr_elements[item_id]); \n");
  source.append("    } \n");

  source.append("    Ap[row] = sum; \n");
  source.append("    inner_prod_ApAp += sum * sum; \n");
  source.append("    inner_prod_pAp  += p[row] * sum; \n");
  source.append("  }  \n");

  //////////// parallel reduction of inner product contributions within work group ///////////////
  source.append("  shared_array_ApAp[get_local_id(0)] = inner_prod_ApAp; \n");
  source.append("  shared_array_pAp[get_local_id(0)]  = inner_prod_pAp; \n");
  source.append("  for (uint stride=get_local_size(0)/2; stride > 0; stride /= 2) \n");
  source.append("  { \n");
  source.append("    barrier(CLK_LOCAL_MEM_FENCE); \n");
  source.append("    if (get_local_id(0) < stride) { \n");
  source.append("      shared_array_ApAp[get_local_id(0)] += shared_array_ApAp[get_local_id(0) + stride];  \n");
  source.append("      shared_array_pAp[get_local_id(0)]  += shared_array_pAp[get_local_id(0) + stride];  \n");
  source.append("    } ");
  source.append("  } ");

  // write results to result array
  source.append("  if (get_local_id(0) == 0) { \n ");
  source.append("    inner_prod_buffer[  buffer_size + get_group_id(0)] = shared_array_ApAp[0]; \n");
  source.append("    inner_prod_buffer[2*buffer_size + get_group_id(0)] = shared_array_pAp[0]; \n");
  source.append("  } \n");
  source.append("} \n \n");
}


//////////////////////////////////////////////////////


template<typename StringT>
void generate_pipelined_bicgstab_update_s(StringT & source, std::string const & numeric_string)
{
  source.append("__kernel void bicgstab_update_s( \n");
  source.append("  __global "); source.append(numeric_string); source.append(" * s, \n");
  source.append("  __global "); source.append(numeric_string); source.append(" const * r, \n");
  source.append("  __global "); source.append(numeric_string); source.append(" const * Ap, \n");
  source.append("  __global "); source.append(numeric_string); source.append(" * inner_prod_buffer, \n");
  source.append("  unsigned int chunk_size, \n");
  source.append("  unsigned int chunk_offset, \n");
  source.append("  unsigned int size, \n");
  source.append("  __local "); source.append(numeric_string); source.append(" * shared_array, \n");
  source.append("  __local "); source.append(numeric_string); source.append(" * shared_array_Ap_in_r0) \n");
  source.append("{ \n");

  source.append("  "); source.append(numeric_string); source.append(" alpha = 0; \n");

  // parallel reduction in work group to compute <r, r0> / <Ap, r0>
  source.append("  shared_array[get_local_id(0)]  = inner_prod_buffer[get_local_id(0)]; \n");
  source.append("  shared_array_Ap_in_r0[get_local_id(0)] = inner_prod_buffer[get_local_id(0) + 3 * chunk_size]; \n");
  source.append("  for (uint stride=get_local_size(0)/2; stride > 0; stride /= 2) \n");
  source.append("  { \n");
  source.append("    barrier(CLK_LOCAL_MEM_FENCE); \n");
  source.append("    if (get_local_id(0) < stride) { \n");
  source.append("      shared_array[get_local_id(0)]  += shared_array[get_local_id(0) + stride];  \n");
  source.append("      shared_array_Ap_in_r0[get_local_id(0)] += shared_array_Ap_in_r0[get_local_id(0) + stride];  \n");
  source.append("    } ");
  source.append("  } ");

  // compute alpha from reduced values:
  source.append("  barrier(CLK_LOCAL_MEM_FENCE); \n");
  source.append("  alpha = shared_array[0] / shared_array_Ap_in_r0[0]; ");

  source.append("  "); source.append(numeric_string); source.append(" inner_prod_contrib = 0; \n");
  source.append("  for (unsigned int i = get_global_id(0); i < size; i += get_global_size(0)) { \n");
  source.append("    "); source.append(numeric_string); source.append(" value_s = s[i]; \n");
  source.append("     \n");
  source.append("    value_s = r[i] - alpha * Ap[i]; \n");
  source.append("    inner_prod_contrib += value_s * value_s; \n");
  source.append("     \n");
  source.append("    s[i] = value_s; \n");
  source.append("  }  \n");
  source.append("  barrier(CLK_LOCAL_MEM_FENCE); \n");

  // parallel reduction in work group
  source.append("  shared_array[get_local_id(0)] = inner_prod_contrib; \n");
  source.append("  for (uint stride=get_local_size(0)/2; stride > 0; stride /= 2) \n");
  source.append("  { \n");
  source.append("    barrier(CLK_LOCAL_MEM_FENCE); \n");
  source.append("    if (get_local_id(0) < stride)  \n");
  source.append("      shared_array[get_local_id(0)] += shared_array[get_local_id(0) + stride];  \n");
  source.append("  } ");

  // write results to result array
  source.append(" if (get_local_id(0) == 0) \n ");
  source.append("   inner_prod_buffer[get_group_id(0) + chunk_offset] = shared_array[0]; ");

  source.append("} \n");

}



template<typename StringT>
void generate_pipelined_bicgstab_vector_update(StringT & source, std::string const & numeric_string)
{
  source.append("__kernel void bicgstab_vector_update( \n");
  source.append("  __global "); source.append(numeric_string); source.append(" * result, \n");
  source.append("  "); source.append(numeric_string); source.append(" alpha, \n");
  source.append("  __global "); source.append(numeric_string); source.append(" * p, \n");
  source.append("  "); source.append(numeric_string); source.append(" omega, \n");
  source.append("  __global "); source.append(numeric_string); source.append(" const * s, \n");
  source.append("  __global "); source.append(numeric_string); source.append(" * residual, \n");
  source.append("  __global "); source.append(numeric_string); source.append(" const * As, \n");
  source.append("  "); source.append(numeric_string); source.append(" beta, \n");
  source.append("  __global "); source.append(numeric_string); source.append(" const * Ap, \n");
  source.append("  __global "); source.append(numeric_string); source.append(" const * r0star, \n");
  source.append("  __global "); source.append(numeric_string); source.append(" * inner_prod_buffer, \n");
  source.append("  unsigned int size, \n");
  source.append("  __local "); source.append(numeric_string); source.append(" * shared_array) \n");
  source.append("{ \n");
  source.append("  "); source.append(numeric_string); source.append(" inner_prod_r_r0star = 0; \n");
  source.append("  for (unsigned int i = get_global_id(0); i < size; i += get_global_size(0)) { \n");
  source.append("    "); source.append(numeric_string); source.append(" value_result = result[i]; \n");
  source.append("    "); source.append(numeric_string); source.append(" value_p = p[i]; \n");
  source.append("    "); source.append(numeric_string); source.append(" value_s = s[i]; \n");
  source.append("    "); source.append(numeric_string); source.append(" value_residual = residual[i]; \n");
  source.append("    "); source.append(numeric_string); source.append(" value_As = As[i]; \n");
  source.append("    "); source.append(numeric_string); source.append(" value_Ap = Ap[i]; \n");
  source.append("    "); source.append(numeric_string); source.append(" value_r0star = r0star[i]; \n");
  source.append("     \n");
  source.append("    value_result += alpha * value_p + omega * value_s; \n");
  source.append("    value_residual  = value_s - omega * value_As; \n");
  source.append("    value_p         = value_residual + beta * (value_p - omega * value_Ap); \n");
  source.append("     \n");
  source.append("    result[i]   = value_result; \n");
  source.append("    residual[i] = value_residual; \n");
  source.append("    p[i]        = value_p; \n");
  source.append("    inner_prod_r_r0star += value_residual * value_r0star; \n");
  source.append("  }  \n");

  // parallel reduction in work group
  source.append("  shared_array[get_local_id(0)] = inner_prod_r_r0star; \n");
  source.append("  for (uint stride=get_local_size(0)/2; stride > 0; stride /= 2) \n");
  source.append("  { \n");
  source.append("    barrier(CLK_LOCAL_MEM_FENCE); \n");
  source.append("    if (get_local_id(0) < stride)  \n");
  source.append("      shared_array[get_local_id(0)] += shared_array[get_local_id(0) + stride];  \n");
  source.append("  } ");

  // write results to result array
  source.append(" if (get_local_id(0) == 0) \n ");
  source.append("   inner_prod_buffer[get_group_id(0)] = shared_array[0]; ");

  source.append("} \n");
}


template<typename StringT>
void generate_compressed_matrix_pipelined_bicgstab_prod(StringT & source, std::string const & numeric_string)
{
  source.append("__kernel void bicgstab_csr_prod( \n");
  source.append("  __global const unsigned int * row_indices, \n");
  source.append("  __global const unsigned int * column_indices, \n");
  source.append("  __global const "); source.append(numeric_string); source.append(" * elements, \n");
  source.append("  __global const "); source.append(numeric_string); source.append(" * p, \n");
  source.append("  __global "); source.append(numeric_string); source.append(" * Ap, \n");
  source.append("  __global const "); source.append(numeric_string); source.append(" * r0star, \n");
  source.append("  unsigned int size, \n");
  source.append("  __global "); source.append(numeric_string); source.append(" * inner_prod_buffer, \n");
  source.append("  unsigned int buffer_size, \n");
  source.append("  unsigned int buffer_offset, \n");
  source.append("  __local "); source.append(numeric_string); source.append(" * shared_array_ApAp, \n");
  source.append("  __local "); source.append(numeric_string); source.append(" * shared_array_pAp, \n");
  source.append("  __local "); source.append(numeric_string); source.append(" * shared_array_r0Ap) \n");
  source.append("{ \n");
  source.append("  "); source.append(numeric_string); source.append(" inner_prod_ApAp = 0; \n");
  source.append("  "); source.append(numeric_string); source.append(" inner_prod_pAp = 0; \n");
  source.append("  "); source.append(numeric_string); source.append(" inner_prod_r0Ap = 0; \n");
  source.append("  for (unsigned int row = get_global_id(0); row < size; row += get_global_size(0)) \n");
  source.append("  { \n");
  source.append("    "); source.append(numeric_string); source.append(" dot_prod = ("); source.append(numeric_string); source.append(")0; \n");
  source.append("    unsigned int row_end = row_indices[row+1]; \n");
  source.append("    for (unsigned int i = row_indices[row]; i < row_end; ++i) \n");
  source.append("      dot_prod += elements[i] * p[column_indices[i]]; \n");
  source.append("    Ap[row] = dot_prod; \n");
  source.append("    inner_prod_ApAp  +=    dot_prod * dot_prod; \n");
  source.append("    inner_prod_pAp   +=      p[row] * dot_prod; \n");
  source.append("    inner_prod_r0Ap  += r0star[row] * dot_prod; \n");
  source.append("  } \n");

  // parallel reduction in work group
  source.append("  shared_array_ApAp[get_local_id(0)] = inner_prod_ApAp; \n");
  source.append("  shared_array_pAp[get_local_id(0)]  = inner_prod_pAp; \n");
  source.append("  shared_array_r0Ap[get_local_id(0)] = inner_prod_r0Ap; \n");
  source.append("  for (uint stride=get_local_size(0)/2; stride > 0; stride /= 2) \n");
  source.append("  { \n");
  source.append("    barrier(CLK_LOCAL_MEM_FENCE); \n");
  source.append("    if (get_local_id(0) < stride) { \n");
  source.append("      shared_array_ApAp[get_local_id(0)] += shared_array_ApAp[get_local_id(0) + stride];  \n");
  source.append("      shared_array_pAp[get_local_id(0)]  += shared_array_pAp[get_local_id(0) + stride];  \n");
  source.append("      shared_array_r0Ap[get_local_id(0)]  += shared_array_r0Ap[get_local_id(0) + stride];  \n");
  source.append("    } ");
  source.append("  } ");

  // write results to result array
  source.append("  if (get_local_id(0) == 0) { \n ");
  source.append("    inner_prod_buffer[  buffer_size + get_group_id(0)] = shared_array_ApAp[0]; \n");
  source.append("    inner_prod_buffer[2*buffer_size + get_group_id(0)] = shared_array_pAp[0]; \n");
  source.append("    inner_prod_buffer[buffer_offset + get_group_id(0)] = shared_array_r0Ap[0]; \n");
  source.append("  } \n");

  source.append("} \n \n");

}

template<typename StringT>
void generate_coordinate_matrix_pipelined_bicgstab_prod(StringT & source, std::string const & numeric_string)
{
  source.append("__kernel void bicgstab_coo_prod( \n");
  source.append("  __global const uint2 * coords,  \n");//(row_index, column_index)
  source.append("  __global const "); source.append(numeric_string); source.append(" * elements, \n");
  source.append("  __global const uint  * group_boundaries, \n");
  source.append("  __global const "); source.append(numeric_string); source.append(" * p, \n");
  source.append("  __global "); source.append(numeric_string); source.append(" * Ap, \n");
  source.append("  __global const "); source.append(numeric_string); source.append(" * r0star, \n");
  source.append("  unsigned int size, \n");
  source.append("  __local unsigned int * shared_rows, \n");
  source.append("  __local "); source.append(numeric_string); source.append(" * inter_results, \n");
  source.append("  __global "); source.append(numeric_string); source.append(" * inner_prod_buffer, \n");
  source.append("  unsigned int buffer_size, \n");
  source.append("  unsigned int buffer_offset, \n");
  source.append("  __local "); source.append(numeric_string); source.append(" * shared_array_ApAp, \n");
  source.append("  __local "); source.append(numeric_string); source.append(" * shared_array_pAp, \n");
  source.append("  __local "); source.append(numeric_string); source.append(" * shared_array_r0Ap) \n");
  source.append("{ \n");
  source.append("  "); source.append(numeric_string); source.append(" inner_prod_ApAp = 0; \n");
  source.append("  "); source.append(numeric_string); source.append(" inner_prod_pAp = 0; \n");
  source.append("  "); source.append(numeric_string); source.append(" inner_prod_r0Ap = 0; \n");

  ///////////// Sparse matrix-vector multiplication part /////////////
  source.append("  uint2 tmp; \n");
  source.append("  "); source.append(numeric_string); source.append(" val; \n");
  source.append("  uint group_start = group_boundaries[get_group_id(0)]; \n");
  source.append("  uint group_end   = group_boundaries[get_group_id(0) + 1]; \n");
  source.append("  uint k_end = (group_end > group_start) ? 1 + (group_end - group_start - 1) / get_local_size(0) : 0; \n");   // -1 in order to have correct behavior if group_end - group_start == j * get_local_size(0)

  source.append("  uint local_index = 0; \n");

  source.append("  for (uint k = 0; k < k_end; ++k) { \n");
  source.append("    local_index = group_start + k * get_local_size(0) + get_local_id(0); \n");

  source.append("    tmp = (local_index < group_end) ? coords[local_index] : (uint2) 0; \n");
  source.append("    val = (local_index < group_end) ? elements[local_index] * p[tmp.y] : 0; \n");

  //check for carry from previous loop run:
  source.append("    if (get_local_id(0) == 0 && k > 0) { \n");
  source.append("      if (tmp.x == shared_rows[get_local_size(0)-1]) \n");
  source.append("        val += inter_results[get_local_size(0)-1]; \n");
  source.append("      else {\n");
  source.append("        "); source.append(numeric_string); source.append(" Ap_entry = inter_results[get_local_size(0)-1]; \n");
  source.append("        Ap[shared_rows[get_local_size(0)-1]] = Ap_entry; \n");
  source.append("        inner_prod_ApAp += Ap_entry * Ap_entry; \n");
  source.append("        inner_prod_pAp  += p[shared_rows[get_local_size(0)-1]] * Ap_entry; \n");
  source.append("        inner_prod_r0Ap  += r0star[shared_rows[get_local_size(0)-1]] * Ap_entry; \n");
  source.append("      } \n");
  source.append("    } \n");

  //segmented parallel reduction begin
  source.append("    barrier(CLK_LOCAL_MEM_FENCE); \n");
  source.append("    shared_rows[get_local_id(0)] = tmp.x; \n");
  source.append("    inter_results[get_local_id(0)] = val; \n");
  source.append("    "); source.append(numeric_string); source.append(" left = 0; \n");
  source.append("    barrier(CLK_LOCAL_MEM_FENCE); \n");

  source.append("    for (unsigned int stride = 1; stride < get_local_size(0); stride *= 2) { \n");
  source.append("      left = (get_local_id(0) >= stride && tmp.x == shared_rows[get_local_id(0) - stride]) ? inter_results[get_local_id(0) - stride] : 0; \n");
  source.append("      barrier(CLK_LOCAL_MEM_FENCE); \n");
  source.append("      inter_results[get_local_id(0)] += left; \n");
  source.append("      barrier(CLK_LOCAL_MEM_FENCE); \n");
  source.append("    } \n");
  //segmented parallel reduction end

  source.append("    if (local_index < group_end && get_local_id(0) < get_local_size(0) - 1 && \n");
  source.append("      shared_rows[get_local_id(0)] != shared_rows[get_local_id(0) + 1]) { \n");
  source.append("      "); source.append(numeric_string); source.append(" Ap_entry = inter_results[get_local_id(0)]; \n");
  source.append("      Ap[tmp.x] = Ap_entry; \n");
  source.append("      inner_prod_ApAp += Ap_entry * Ap_entry; \n");
  source.append("      inner_prod_pAp  += p[tmp.x] * Ap_entry; \n");
  source.append("      inner_prod_r0Ap += r0star[tmp.x] * Ap_entry; \n");
  source.append("    } \n");

  source.append("    barrier(CLK_LOCAL_MEM_FENCE); \n");
  source.append("  }  \n"); //for k

  source.append("  if (local_index + 1 == group_end) {\n");  //write results of last active entry (this may not necessarily be the case already)
  source.append("    "); source.append(numeric_string); source.append(" Ap_entry = inter_results[get_local_id(0)]; \n");
  source.append("    Ap[tmp.x] = Ap_entry; \n");
  source.append("    inner_prod_ApAp += Ap_entry * Ap_entry; \n");
  source.append("    inner_prod_pAp  += p[tmp.x] * Ap_entry; \n");
  source.append("    inner_prod_r0Ap += r0star[tmp.x] * Ap_entry; \n");
  source.append("  }  \n");

  // parallel reduction in work group
  source.append("  shared_array_ApAp[get_local_id(0)] = inner_prod_ApAp; \n");
  source.append("  shared_array_pAp[get_local_id(0)]  = inner_prod_pAp; \n");
  source.append("  shared_array_r0Ap[get_local_id(0)] = inner_prod_r0Ap; \n");
  source.append("  for (uint stride=get_local_size(0)/2; stride > 0; stride /= 2) \n");
  source.append("  { \n");
  source.append("    barrier(CLK_LOCAL_MEM_FENCE); \n");
  source.append("    if (get_local_id(0) < stride) { \n");
  source.append("      shared_array_ApAp[get_local_id(0)] += shared_array_ApAp[get_local_id(0) + stride];  \n");
  source.append("      shared_array_pAp[get_local_id(0)]  += shared_array_pAp[get_local_id(0) + stride];  \n");
  source.append("      shared_array_r0Ap[get_local_id(0)]  += shared_array_r0Ap[get_local_id(0) + stride];  \n");
  source.append("    } ");
  source.append("  } ");

  // write results to result array
  source.append("  if (get_local_id(0) == 0) { \n ");
  source.append("    inner_prod_buffer[  buffer_size + get_group_id(0)] = shared_array_ApAp[0]; \n");
  source.append("    inner_prod_buffer[2*buffer_size + get_group_id(0)] = shared_array_pAp[0]; \n");
  source.append("    inner_prod_buffer[buffer_offset + get_group_id(0)] = shared_array_r0Ap[0]; \n");
  source.append("  } \n");

  source.append("} \n \n");

}


template<typename StringT>
void generate_ell_matrix_pipelined_bicgstab_prod(StringT & source, std::string const & numeric_string)
{
  source.append("__kernel void bicgstab_ell_prod( \n");
  source.append("  __global const unsigned int * coords, \n");
  source.append("  __global const "); source.append(numeric_string); source.append(" * elements, \n");
  source.append("  unsigned int internal_row_num, \n");
  source.append("  unsigned int items_per_row, \n");
  source.append("  unsigned int aligned_items_per_row, \n");
  source.append("  __global const "); source.append(numeric_string); source.append(" * p, \n");
  source.append("  __global "); source.append(numeric_string); source.append(" * Ap, \n");
  source.append("  __global const "); source.append(numeric_string); source.append(" * r0star, \n");
  source.append("  unsigned int size, \n");
  source.append("  __global "); source.append(numeric_string); source.append(" * inner_prod_buffer, \n");
  source.append("  unsigned int buffer_size, \n");
  source.append("  unsigned int buffer_offset, \n");
  source.append("  __local "); source.append(numeric_string); source.append(" * shared_array_ApAp, \n");
  source.append("  __local "); source.append(numeric_string); source.append(" * shared_array_pAp, \n");
  source.append("  __local "); source.append(numeric_string); source.append(" * shared_array_r0Ap) \n");
  source.append("{ \n");
  source.append("  "); source.append(numeric_string); source.append(" inner_prod_ApAp = 0; \n");
  source.append("  "); source.append(numeric_string); source.append(" inner_prod_pAp = 0; \n");
  source.append("  "); source.append(numeric_string); source.append(" inner_prod_r0Ap = 0; \n");
  source.append("  uint glb_id = get_global_id(0); \n");
  source.append("  uint glb_sz = get_global_size(0); \n");

  source.append("  for (uint row = glb_id; row < size; row += glb_sz) { \n");
  source.append("    "); source.append(numeric_string); source.append(" sum = 0; \n");

  source.append("    uint offset = row; \n");
  source.append("    for (uint item_id = 0; item_id < items_per_row; item_id++, offset += internal_row_num) { \n");
  source.append("      "); source.append(numeric_string); source.append(" val = elements[offset]; \n");
  source.append("      sum += val ? p[coords[offset]] * val : ("); source.append(numeric_string); source.append(")0; \n");
  source.append("    } \n");

  source.append("    Ap[row] = sum; \n");
  source.append("    inner_prod_ApAp += sum * sum; \n");
  source.append("    inner_prod_pAp  += p[row] * sum; \n");
  source.append("    inner_prod_r0Ap += r0star[row] * sum; \n");
  source.append("  }  \n");

  // parallel reduction in work group
  source.append("  shared_array_ApAp[get_local_id(0)] = inner_prod_ApAp; \n");
  source.append("  shared_array_pAp[get_local_id(0)]  = inner_prod_pAp; \n");
  source.append("  shared_array_r0Ap[get_local_id(0)] = inner_prod_r0Ap; \n");
  source.append("  for (uint stride=get_local_size(0)/2; stride > 0; stride /= 2) \n");
  source.append("  { \n");
  source.append("    barrier(CLK_LOCAL_MEM_FENCE); \n");
  source.append("    if (get_local_id(0) < stride) { \n");
  source.append("      shared_array_ApAp[get_local_id(0)] += shared_array_ApAp[get_local_id(0) + stride];  \n");
  source.append("      shared_array_pAp[get_local_id(0)]  += shared_array_pAp[get_local_id(0) + stride];  \n");
  source.append("      shared_array_r0Ap[get_local_id(0)] += shared_array_r0Ap[get_local_id(0) + stride];  \n");
  source.append("    } ");
  source.append("  } ");

  // write results to result array
  source.append("  if (get_local_id(0) == 0) { \n ");
  source.append("    inner_prod_buffer[  buffer_size + get_group_id(0)] = shared_array_ApAp[0]; \n");
  source.append("    inner_prod_buffer[2*buffer_size + get_group_id(0)] = shared_array_pAp[0]; \n");
  source.append("    inner_prod_buffer[buffer_offset + get_group_id(0)] = shared_array_r0Ap[0]; \n");
  source.append("  } \n");
  source.append("} \n \n");
}

template<typename StringT>
void generate_sliced_ell_matrix_pipelined_bicgstab_prod(StringT & source, std::string const & numeric_string)
{
  source.append("__kernel void bicgstab_sliced_ell_prod( \n");
  source.append("  __global const unsigned int * columns_per_block, \n");
  source.append("  __global const unsigned int * column_indices, \n");
  source.append("  __global const unsigned int * block_start, \n");
  source.append("  __global const "); source.append(numeric_string); source.append(" * elements, \n");
  source.append("  __global const "); source.append(numeric_string); source.append(" * p, \n");
  source.append("  __global "); source.append(numeric_string); source.append(" * Ap, \n");
  source.append("  __global const "); source.append(numeric_string); source.append(" * r0star, \n");
  source.append("  unsigned int size, \n");
  source.append("  __global "); source.append(numeric_string); source.append(" * inner_prod_buffer, \n");
  source.append("  unsigned int buffer_size, \n");
  source.append("  unsigned int buffer_offset, \n");
  source.append("  __local "); source.append(numeric_string); source.append(" * shared_array_ApAp, \n");
  source.append("  __local "); source.append(numeric_string); source.append(" * shared_array_pAp, \n");
  source.append("  __local "); source.append(numeric_string); source.append(" * shared_array_r0Ap) \n");
  source.append("{ \n");
  source.append("  "); source.append(numeric_string); source.append(" inner_prod_ApAp = 0; \n");
  source.append("  "); source.append(numeric_string); source.append(" inner_prod_pAp = 0; \n");
  source.append("  "); source.append(numeric_string); source.append(" inner_prod_r0Ap = 0; \n");
  source.append("  uint local_id   = get_local_id(0); \n");
  source.append("  uint local_size = get_local_size(0); \n");

  source.append("  for (uint block_idx = get_group_id(0); block_idx <= size / local_size; block_idx += get_num_groups(0)) { \n");
  source.append("    "); source.append(numeric_string); source.append(" sum = 0; \n");

  source.append("    uint row    = block_idx * local_size + local_id; \n");
  source.append("    uint offset = block_start[block_idx]; \n");
  source.append("    uint num_columns = columns_per_block[block_idx]; \n");
  source.append("    for (uint item_id = 0; item_id < num_columns; item_id++) { \n");
  source.append("      uint index = offset + item_id * local_size + local_id; \n");
  source.append("      "); source.append(numeric_string); source.append(" val = elements[index]; \n");
  source.append("      sum += val ? (p[column_indices[index]] * val) : 0; \n");
  source.append("    } \n");

  source.append("    if (row < size) {\n");
  source.append("      Ap[row] = sum; \n");
  source.append("      inner_prod_ApAp += sum * sum; \n");
  source.append("      inner_prod_pAp  += p[row] * sum; \n");
  source.append("      inner_prod_r0Ap += r0star[row] * sum; \n");
  source.append("    }  \n");
  source.append("  }  \n");

  // parallel reduction in work group
  source.append("  shared_array_ApAp[get_local_id(0)] = inner_prod_ApAp; \n");
  source.append("  shared_array_pAp[get_local_id(0)]  = inner_prod_pAp; \n");
  source.append("  shared_array_r0Ap[get_local_id(0)] = inner_prod_r0Ap; \n");
  source.append("  for (uint stride=get_local_size(0)/2; stride > 0; stride /= 2) \n");
  source.append("  { \n");
  source.append("    barrier(CLK_LOCAL_MEM_FENCE); \n");
  source.append("    if (get_local_id(0) < stride) { \n");
  source.append("      shared_array_ApAp[get_local_id(0)] += shared_array_ApAp[get_local_id(0) + stride];  \n");
  source.append("      shared_array_pAp[get_local_id(0)]  += shared_array_pAp[get_local_id(0) + stride];  \n");
  source.append("      shared_array_r0Ap[get_local_id(0)] += shared_array_r0Ap[get_local_id(0) + stride];  \n");
  source.append("    } ");
  source.append("  } ");

  // write results to result array
  source.append("  if (get_local_id(0) == 0) { \n ");
  source.append("    inner_prod_buffer[  buffer_size + get_group_id(0)] = shared_array_ApAp[0]; \n");
  source.append("    inner_prod_buffer[2*buffer_size + get_group_id(0)] = shared_array_pAp[0]; \n");
  source.append("    inner_prod_buffer[buffer_offset + get_group_id(0)] = shared_array_r0Ap[0]; \n");
  source.append("  } \n");
  source.append("} \n \n");
}

template<typename StringT>
void generate_hyb_matrix_pipelined_bicgstab_prod(StringT & source, std::string const & numeric_string)
{
  source.append("__kernel void bicgstab_hyb_prod( \n");
  source.append("  const __global int* ell_coords, \n");
  source.append("  const __global "); source.append(numeric_string); source.append("* ell_elements, \n");
  source.append("  const __global uint* csr_rows, \n");
  source.append("  const __global uint* csr_cols, \n");
  source.append("  const __global "); source.append(numeric_string); source.append("* csr_elements, \n");
  source.append("  unsigned int internal_row_num, \n");
  source.append("  unsigned int items_per_row, \n");
  source.append("  unsigned int aligned_items_per_row, \n");
  source.append("  __global const "); source.append(numeric_string); source.append(" * p, \n");
  source.append("  __global "); source.append(numeric_string); source.append(" * Ap, \n");
  source.append("  __global const "); source.append(numeric_string); source.append(" * r0star, \n");
  source.append("  unsigned int size, \n");
  source.append("  __global "); source.append(numeric_string); source.append(" * inner_prod_buffer, \n");
  source.append("  unsigned int buffer_size, \n");
  source.append("  unsigned int buffer_offset, \n");
  source.append("  __local "); source.append(numeric_string); source.append(" * shared_array_ApAp, \n");
  source.append("  __local "); source.append(numeric_string); source.append(" * shared_array_pAp, \n");
  source.append("   __local "); source.append(numeric_string); source.append(" * shared_array_r0Ap) \n");
  source.append("{ \n");
  source.append("  "); source.append(numeric_string); source.append(" inner_prod_ApAp = 0; \n");
  source.append("  "); source.append(numeric_string); source.append(" inner_prod_pAp = 0; \n");
  source.append("  "); source.append(numeric_string); source.append(" inner_prod_r0Ap = 0; \n");
  source.append("  uint glb_id = get_global_id(0); \n");
  source.append("  uint glb_sz = get_global_size(0); \n");

  source.append("  for (uint row = glb_id; row < size; row += glb_sz) { \n");
  source.append("    "); source.append(numeric_string); source.append(" sum = 0; \n");

  source.append("    uint offset = row; \n");
  source.append("    for (uint item_id = 0; item_id < items_per_row; item_id++, offset += internal_row_num) { \n");
  source.append("      "); source.append(numeric_string); source.append(" val = ell_elements[offset]; \n");
  source.append("      sum += val ? (p[ell_coords[offset]] * val) : 0; \n");
  source.append("    } \n");

  source.append("    uint col_begin = csr_rows[row]; \n");
  source.append("    uint col_end   = csr_rows[row + 1]; \n");

  source.append("    for (uint item_id = col_begin; item_id < col_end; item_id++) {  \n");
  source.append("      sum += (p[csr_cols[item_id]] * csr_elements[item_id]); \n");
  source.append("    } \n");

  source.append("    Ap[row] = sum; \n");
  source.append("    inner_prod_ApAp += sum * sum; \n");
  source.append("    inner_prod_pAp  += p[row] * sum; \n");
  source.append("    inner_prod_r0Ap += r0star[row] * sum; \n");
  source.append("  }  \n");

  // parallel reduction in work group
  source.append("  shared_array_ApAp[get_local_id(0)] = inner_prod_ApAp; \n");
  source.append("  shared_array_pAp[get_local_id(0)]  = inner_prod_pAp; \n");
  source.append("  shared_array_r0Ap[get_local_id(0)] = inner_prod_r0Ap; \n");
  source.append("  for (uint stride=get_local_size(0)/2; stride > 0; stride /= 2) \n");
  source.append("  { \n");
  source.append("    barrier(CLK_LOCAL_MEM_FENCE); \n");
  source.append("    if (get_local_id(0) < stride) { \n");
  source.append("      shared_array_ApAp[get_local_id(0)] += shared_array_ApAp[get_local_id(0) + stride];  \n");
  source.append("      shared_array_pAp[get_local_id(0)]  += shared_array_pAp[get_local_id(0) + stride];  \n");
  source.append("      shared_array_r0Ap[get_local_id(0)]  += shared_array_r0Ap[get_local_id(0) + stride];  \n");
  source.append("    } ");
  source.append("  } ");

  // write results to result array
  source.append("  if (get_local_id(0) == 0) { \n ");
  source.append("    inner_prod_buffer[  buffer_size + get_group_id(0)] = shared_array_ApAp[0]; \n");
  source.append("    inner_prod_buffer[2*buffer_size + get_group_id(0)] = shared_array_pAp[0]; \n");
  source.append("    inner_prod_buffer[buffer_offset + get_group_id(0)] = shared_array_r0Ap[0]; \n");
  source.append("  } \n");
  source.append("} \n \n");
}



//////////////////////////// Part 2: Main kernel class ////////////////////////////////////

// main kernel class
/** @brief Main kernel class for generating specialized OpenCL kernels for fast iterative solvers. */
template<typename NumericT>
struct iterative
{
  static std::string program_name()
  {
    return viennacl::ocl::type_to_string<NumericT>::apply() + "_iterative";
  }

  static void init(viennacl::ocl::context & ctx)
  {
    static std::map<cl_context, bool> init_done;
    if (!init_done[ctx.handle().get()])
    {
      viennacl::ocl::DOUBLE_PRECISION_CHECKER<NumericT>::apply(ctx);
      std::string numeric_string = viennacl::ocl::type_to_string<NumericT>::apply();

      std::string source;
      source.reserve(1024);

      viennacl::ocl::append_double_precision_pragma<NumericT>(ctx, source);

      generate_pipelined_cg_vector_update(source, numeric_string);
      generate_compressed_matrix_pipelined_cg_prod(source, numeric_string);
      generate_coordinate_matrix_pipelined_cg_prod(source, numeric_string);
      generate_ell_matrix_pipelined_cg_prod(source, numeric_string);
      generate_sliced_ell_matrix_pipelined_cg_prod(source, numeric_string);
      generate_hyb_matrix_pipelined_cg_prod(source, numeric_string);

      generate_pipelined_bicgstab_update_s(source, numeric_string);
      generate_pipelined_bicgstab_vector_update(source, numeric_string);
      generate_compressed_matrix_pipelined_bicgstab_prod(source, numeric_string);
      generate_coordinate_matrix_pipelined_bicgstab_prod(source, numeric_string);
      generate_ell_matrix_pipelined_bicgstab_prod(source, numeric_string);
      generate_sliced_ell_matrix_pipelined_bicgstab_prod(source, numeric_string);
      generate_hyb_matrix_pipelined_bicgstab_prod(source, numeric_string);

      std::string prog_name = program_name();
      #ifdef VIENNACL_BUILD_INFO
      std::cout << "Creating program " << prog_name << std::endl;
      #endif
      ctx.add_program(source, prog_name);
      init_done[ctx.handle().get()] = true;
    } //if
  } //init
};

}  // namespace kernels
}  // namespace opencl
}  // namespace linalg
}  // namespace viennacl
#endif

