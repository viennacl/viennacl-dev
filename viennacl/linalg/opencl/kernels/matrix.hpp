#ifndef VIENNACL_LINALG_OPENCL_KERNELS_MATRIX_HPP
#define VIENNACL_LINALG_OPENCL_KERNELS_MATRIX_HPP

#include "viennacl/scheduler/preset.hpp"
#include "viennacl/tools/tools.hpp"
#include "viennacl/ocl/kernel.hpp"
#include "viennacl/ocl/platform.hpp"
#include "viennacl/ocl/utils.hpp"

#include "viennacl/device_specific/execution_handler.hpp"
#include "viennacl/device_specific/builtin_database/vector_axpy.hpp"
#include "viennacl/device_specific/builtin_database/matrix_axpy.hpp"
#include "viennacl/device_specific/builtin_database/row_wise_reduction.hpp"
#include "viennacl/device_specific/builtin_database/matrix_product.hpp"

/** @file viennacl/linalg/opencl/kernels/matrix.hpp
 *  @brief Runtime generation of OpenCL kernels for matrix operations */
namespace viennacl
{
namespace linalg
{
namespace opencl
{
namespace kernels
{

//////////////////////////// Part 1: Kernel generation routines ////////////////////////////////////

/** @brief Enumeration for the scalar type in ambm-like operations */
enum ambm_scalar_type
{
  VIENNACL_AMBM_NONE = 0, // matrix does not exist/contribute
  VIENNACL_AMBM_CPU,
  VIENNACL_AMBM_GPU
};

/** @brief Configuration struct for generating OpenCL kernels for linear combinations of matrices */
struct ambm_config
{
  ambm_config() : with_stride_and_range(true), is_row_major(true), a(VIENNACL_AMBM_CPU), b(VIENNACL_AMBM_NONE) {}

  bool with_stride_and_range;
  bool is_row_major;
  std::string      assign_op;
  ambm_scalar_type a;
  ambm_scalar_type b;
};




template<typename StringT>
void generate_fft(StringT & source, std::string const & numeric_string, bool is_row_major)
{
  // naive fourier transform (quadratic complexity, use for reference only)
  source.append("__kernel void fft_direct(__global "); source.append(numeric_string); source.append("2 *input, \n");
  source.append("                         __global "); source.append(numeric_string); source.append("2 *output, \n");
  source.append("                         unsigned int size, \n");
  source.append("                         unsigned int stride, \n");
  source.append("                         unsigned int batch_num, \n");
  source.append("                         "); source.append(numeric_string); source.append(" sign) { \n");
  source.append("    const "); source.append(numeric_string); source.append(" NUM_PI = 3.14159265358979323846; \n");
  source.append(" \n");
  source.append("    for (unsigned int batch_id = 0; batch_id < batch_num; batch_id++) { \n");
  source.append("        for (unsigned int k = get_global_id(0); k < size; k += get_global_size(0)) { \n");
  source.append("            "); source.append(numeric_string); source.append("2 f = 0.0f; \n");
  source.append(" \n");
  source.append("            for (unsigned int n = 0; n < size; n++) { \n");
  source.append("                "); source.append(numeric_string); source.append("2 in = ");
  if (is_row_major)
    source.append("input[batch_id * stride + n]; \n"); //input index here
  else
    source.append("input[n * stride + batch_id]; \n"); //input index here
  source.append(" \n");
  source.append("                "); source.append(numeric_string); source.append(" sn, cs; \n");
  source.append("                "); source.append(numeric_string); source.append(" arg = sign * 2 * NUM_PI * k / size * n; \n");
  source.append("                sn = sincos(arg, &cs); \n");
  source.append(" \n");
  source.append("                "); source.append(numeric_string); source.append("2 ex = ("); source.append(numeric_string); source.append("2)(cs, sn); \n");
  source.append("                f = f + ("); source.append(numeric_string); source.append("2)(in.x * ex.x - in.y * ex.y, in.x * ex.y + in.y * ex.x); \n");
  source.append("            } \n");
  source.append(" \n");
  if (is_row_major)
    source.append("            output[batch_id * stride + k] = f; \n"); // output index here
  else
    source.append("            output[k * stride + batch_id] = f; \n"); // output index here
  source.append("        } \n");
  source.append("    } \n");
  source.append("} \n");

  source.append(" \n"); //////////////////////////////

  source.append("__kernel void fft_radix2(__global "); source.append(numeric_string); source.append("2* input, \n");
  source.append("                         unsigned int s, \n");
  source.append("                         unsigned int bit_size, \n");
  source.append("                         unsigned int size, \n");
  source.append("                         unsigned int stride, \n");
  source.append("                         unsigned int batch_num, \n");
  source.append("                         "); source.append(numeric_string); source.append(" sign) { \n");
  source.append(" \n");
  source.append("    unsigned int ss = 1 << s; \n");
  source.append("    unsigned int half_size = size >> 1; \n");
  source.append(" \n");
  source.append("    "); source.append(numeric_string); source.append(" cs, sn; \n");
  source.append("    const "); source.append(numeric_string); source.append(" NUM_PI = 3.14159265358979323846; \n");
  source.append(" \n");
  source.append("    unsigned int glb_id = get_global_id(0); \n");
  source.append("    unsigned int glb_sz = get_global_size(0); \n");

  source.append("    for (unsigned int batch_id = 0; batch_id < batch_num; batch_id++) { \n");
  source.append("        for (unsigned int tid = glb_id; tid < half_size; tid += glb_sz) { \n");
  source.append("            unsigned int group = (tid & (ss - 1)); \n");
  source.append("            unsigned int pos = ((tid >> s) << (s + 1)) + group; \n");

  if (is_row_major)
  {
    source.append("            unsigned int offset = batch_id * stride + pos; \n");
    source.append("            "); source.append(numeric_string); source.append("2 in1 = input[offset]; \n"); //index
    source.append("            "); source.append(numeric_string); source.append("2 in2 = input[offset + ss]; \n");//index
  }
  else
  {
    source.append("            unsigned int offset = pos * stride + batch_id; \n");
    source.append("            "); source.append(numeric_string); source.append("2 in1 = input[offset]; \n"); //index
    source.append("            "); source.append(numeric_string); source.append("2 in2 = input[offset + ss * stride]; \n");//index
  }

  source.append("            "); source.append(numeric_string); source.append(" arg = group * sign * NUM_PI / ss; \n");

  source.append("            sn = sincos(arg, &cs); \n");

  source.append("            "); source.append(numeric_string); source.append("2 ex = ("); source.append(numeric_string); source.append("2)(cs, sn); \n");

  source.append("            "); source.append(numeric_string); source.append("2 tmp = ("); source.append(numeric_string); source.append("2)(in2.x * ex.x - in2.y * ex.y, in2.x * ex.y + in2.y * ex.x); \n");

  if (is_row_major)
    source.append("            input[offset + ss] = in1 - tmp; \n");//index
  else
    source.append("            input[offset + ss * stride] = in1 - tmp; \n");//index
  source.append("            input[offset] = in1 + tmp; \n");//index
  source.append("        } \n");
  source.append("    } \n");
  source.append("} \n");

  source.append(" \n"); //////////////////////////////

  source.append(" unsigned int get_reorder_num(unsigned int v, unsigned int bit_size) { \n");
  source.append("     v = ((v >> 1) & 0x55555555) | ((v & 0x55555555) << 1); \n");
  source.append("     v = ((v >> 2) & 0x33333333) | ((v & 0x33333333) << 2); \n");
  source.append("     v = ((v >> 4) & 0x0F0F0F0F) | ((v & 0x0F0F0F0F) << 4); \n");
  source.append("     v = ((v >> 8) & 0x00FF00FF) | ((v & 0x00FF00FF) << 8); \n");
  source.append("     v = (v >> 16) | (v << 16); \n");
  source.append("  \n");
  source.append("     v = v >> (32 - bit_size); \n");
  source.append("  \n");
  source.append("     return v; \n");
  source.append(" } \n");

  source.append(" __kernel void fft_radix2_local(__global "); source.append(numeric_string); source.append("2* input, \n");
  source.append("                                 __local "); source.append(numeric_string); source.append("2* lcl_input, \n");
  source.append("                                 unsigned int bit_size, \n");
  source.append("                                 unsigned int size, \n");
  source.append("                                 unsigned int stride, \n");
  source.append("                                 unsigned int batch_num, \n");
  source.append("                                 "); source.append(numeric_string); source.append(" sign) { \n");

  source.append("     unsigned int grp_id = get_group_id(0); \n");
  source.append("     unsigned int grp_num = get_num_groups(0); \n");

  source.append("     unsigned int lcl_sz = get_local_size(0); \n");
  source.append("     unsigned int lcl_id = get_local_id(0); \n");
  source.append("     const "); source.append(numeric_string); source.append(" NUM_PI = 3.14159265358979323846; \n");

  source.append("     for (unsigned int batch_id = grp_id; batch_id < batch_num; batch_id += grp_num) { \n");
          //unsigned int base_offset = stride * batch_id; \n");
          //copy chunk of global memory to local \n");
  source.append("         for (unsigned int p = lcl_id; p < size; p += lcl_sz) { \n");
  source.append("             unsigned int v = get_reorder_num(p, bit_size); \n");
  if (is_row_major)
    source.append("             lcl_input[v] = input[batch_id * stride + p]; \n"); //index
  else
    source.append("             lcl_input[v] = input[p * stride + batch_id]; \n"); //index
  source.append("         } \n");

  source.append("         barrier(CLK_LOCAL_MEM_FENCE); \n");

          //performs Cooley-Tukey FFT on local array
  source.append("         for (unsigned int s = 0; s < bit_size; s++) { \n");
  source.append("             unsigned int ss = 1 << s; \n");

  source.append("             "); source.append(numeric_string); source.append(" cs, sn; \n");

  source.append("             for (unsigned int tid = lcl_id; tid < size; tid += lcl_sz) { \n");
  source.append("                 unsigned int group = (tid & (ss - 1)); \n");
  source.append("                 unsigned int pos = ((tid >> s) << (s + 1)) + group; \n");

  source.append("                 "); source.append(numeric_string); source.append("2 in1 = lcl_input[pos]; \n");
  source.append("                 "); source.append(numeric_string); source.append("2 in2 = lcl_input[pos + ss]; \n");

  source.append("                 "); source.append(numeric_string); source.append(" arg = group * sign * NUM_PI / ss; \n");

  source.append("                 sn = sincos(arg, &cs); \n");
  source.append("                 "); source.append(numeric_string); source.append("2 ex = ("); source.append(numeric_string); source.append("2)(cs, sn); \n");

  source.append("                 "); source.append(numeric_string); source.append("2 tmp = ("); source.append(numeric_string); source.append("2)(in2.x * ex.x - in2.y * ex.y, in2.x * ex.y + in2.y * ex.x); \n");

  source.append("                 lcl_input[pos + ss] = in1 - tmp; \n");
  source.append("                 lcl_input[pos] = in1 + tmp; \n");
  source.append("             } \n");

  source.append("             barrier(CLK_LOCAL_MEM_FENCE); \n");
  source.append("         } \n");

          //copy local array back to global memory
  source.append("         for (unsigned int p = lcl_id; p < size; p += lcl_sz) { \n");
  if (is_row_major)
    source.append("             input[batch_id * stride + p] = lcl_input[p]; \n");//index
  else
    source.append("             input[p * stride + batch_id] = lcl_input[p]; \n");//index
  source.append("         } \n");
  source.append("     } \n");
  source.append(" } \n");

  source.append(" \n"); //////////////////////////////

  //
  // Performs reordering of input data in bit-reversal order
  // Probably it's better to do in host side,
  //
  source.append("unsigned int get_reorder_num_2(unsigned int v, unsigned int bit_size) { \n");
  source.append("    v = ((v >> 1) & 0x55555555) | ((v & 0x55555555) << 1); \n");
  source.append("    v = ((v >> 2) & 0x33333333) | ((v & 0x33333333) << 2); \n");
  source.append("    v = ((v >> 4) & 0x0F0F0F0F) | ((v & 0x0F0F0F0F) << 4); \n");
  source.append("    v = ((v >> 8) & 0x00FF00FF) | ((v & 0x00FF00FF) << 8); \n");
  source.append("    v = (v >> 16) | (v << 16); \n");

  source.append("    v = v >> (32 - bit_size); \n");

  source.append("    return v; \n");
  source.append("} \n");

  source.append("__kernel void fft_reorder(__global "); source.append(numeric_string); source.append("2* input, \n");
  source.append("                          unsigned int bit_size, \n");
  source.append("                          unsigned int size, \n");
  source.append("                          unsigned int stride, \n");
  source.append("                          int batch_num) { \n");

  source.append("    unsigned int glb_id = get_global_id(0); \n");
  source.append("    unsigned int glb_sz = get_global_size(0); \n");

  source.append("    for (unsigned int batch_id = 0; batch_id < batch_num; batch_id++) { \n");
  source.append("        for (unsigned int i = glb_id; i < size; i += glb_sz) { \n");
  source.append("            unsigned int v = get_reorder_num_2(i, bit_size); \n");

  source.append("            if (i < v) {\n");
  if (is_row_major)
  {
    source.append("                "); source.append(numeric_string); source.append("2 tmp = input[batch_id * stride + i]; \n"); // index
    source.append("                input[batch_id * stride + i] = input[batch_id * stride + v]; \n"); //index
    source.append("                input[batch_id * stride + v] = tmp; \n"); //index
  }
  else
  {
    source.append("                "); source.append(numeric_string); source.append("2 tmp = input[i * stride + batch_id]; \n"); // index
    source.append("                input[i * stride + batch_id] = input[v * stride + batch_id]; \n"); //index
    source.append("                input[v * stride + batch_id] = tmp; \n"); //index
  }
  source.append("            } \n");
  source.append("        } \n");
  source.append("    } \n");
  source.append("} \n");
}

template<typename StringT>
void generate_lu(StringT & source, std::string const & numeric_string, bool is_row_major)
{
  source.append("__kernel void lu_factorize( \n");
  source.append("          __global "); source.append(numeric_string); source.append(" * matrix, \n");
  source.append("          unsigned int matrix_rows, \n");
  source.append("          unsigned int matrix_cols, \n");
  source.append("          unsigned int matrix_internal_rows, \n");
  source.append("          unsigned int matrix_internal_cols) \n");
  source.append("{ \n");
  source.append("  "); source.append(numeric_string); source.append(" temp; \n");

  if (is_row_major)
  {
    source.append("  unsigned rowi; \n");
    source.append("  unsigned rowk; \n");
    source.append("  for (unsigned int i=1; i<matrix_rows; ++i) \n");
    source.append("  { \n");
    source.append("    rowi = i * matrix_internal_cols; \n");
    source.append("    for (unsigned int k=0; k<i; ++k) \n");
    source.append("    { \n");
    source.append("      rowk = k * matrix_internal_cols; \n");
    source.append("      if (get_global_id(0) == 0) \n");
    source.append("        matrix[rowi + k] /= matrix[rowk + k]; \n");

    source.append("      barrier(CLK_GLOBAL_MEM_FENCE); \n");
    source.append("      temp = matrix[rowi + k]; \n");

    //parallel subtraction:
    source.append("      for (unsigned int j=k+1 + get_global_id(0); j<matrix_rows; j += get_global_size(0)) \n");
    source.append("        matrix[rowi + j] -= temp * matrix[rowk + j]; \n");
  }
  else
  {
    source.append("      for (unsigned int i=1; i<matrix_rows; ++i) \n");
    source.append("      { \n");
    source.append("        for (unsigned int k=0; k<i; ++k) \n");
    source.append("        { \n");

    source.append("          if (get_global_id(0) == 0) \n");
    source.append("            matrix[i + k*matrix_internal_rows] /= matrix[k + k*matrix_internal_rows]; \n");

    source.append("          barrier(CLK_GLOBAL_MEM_FENCE); \n");
    source.append("          temp = matrix[i + k*matrix_internal_rows]; \n");

    //parallel subtraction:
    source.append("          for (unsigned int j=k+1 + get_global_id(0); j<matrix_cols; j += get_global_size(0)) \n");
    source.append("            matrix[i + j*matrix_internal_rows] -= temp * matrix[k + j*matrix_internal_rows]; \n");
  }
  source.append("   }");
  source.append("  }");
  source.append("}");
}


template<typename StringT>
void generate_scaled_rank1_update(StringT & source, std::string const & numeric_string, bool is_row_major, bool alpha_on_cpu)
{
  source.append("__kernel void scaled_rank1_update_"); alpha_on_cpu ? source.append("cpu") : source.append("gpu"); source.append("( \n");
  source.append("  __global "); source.append(numeric_string); source.append(" * A, \n");
  source.append("  unsigned int A_start1, unsigned int A_start2, \n");
  source.append("  unsigned int A_inc1,   unsigned int A_inc2, \n");
  source.append("  unsigned int A_size1,  unsigned int A_size2, \n");
  source.append("  unsigned int A_internal_size1,  unsigned int A_internal_size2, \n");

  if (alpha_on_cpu) {
    source.append("  "); source.append(numeric_string); source.append(" val, \n");
  } else {
    source.append("  __global const "); source.append(numeric_string); source.append(" *val, \n");
  }
  source.append("  unsigned int options2, \n");

  source.append("  __global const "); source.append(numeric_string); source.append(" * vec1, \n");
  source.append("  unsigned int start1, \n");
  source.append("  unsigned int inc1, \n");
  source.append("  unsigned int size1, \n");

  source.append("  __global const "); source.append(numeric_string); source.append(" * vec2, \n");
  source.append("  unsigned int start2, \n");
  source.append("  unsigned int inc2, \n");
  source.append("  unsigned int size2) \n");
  source.append("{ \n");

  if (alpha_on_cpu) {
    source.append("  "); source.append(numeric_string); source.append(" alpha = val; \n");
  } else {
    source.append("  "); source.append(numeric_string); source.append(" alpha = val[0]; \n");
  }
  source.append("  if (options2 & (1 << 0)) \n");
  source.append("    alpha = -alpha; \n");

  source.append("  unsigned int row_gid = get_global_id(0) / get_local_size(0); \n");
  source.append("  unsigned int col_gid = get_global_id(0) % get_local_size(0); \n");

  source.append("  for (unsigned int row = row_gid; row < A_size1; row += get_num_groups(0)) \n");
  source.append("  { \n");
  source.append("    "); source.append(numeric_string); source.append(" tmp = vec1[row * inc1 + start1];");
  source.append("    tmp = (options2 & (1 << 1)) ? tmp / alpha : tmp * alpha;");
  source.append("    for (unsigned int col = col_gid; col < A_size2; col += get_local_size(0)) \n");
  if (is_row_major)
    source.append("      A[(row * A_inc1 + A_start1) * A_internal_size2 + col * A_inc2 + A_start2] += tmp * vec2[col * inc2 + start2]; \n");
  else
    source.append("      A[(row * A_inc1 + A_start1) + (col * A_inc2 + A_start2) * A_internal_size1] += tmp * vec2[col * inc2 + start2]; \n");
  source.append("  } \n");
  source.append("} \n");
}

template<typename StringT>
void generate_triangular_substitute_inplace(StringT & source, std::string const & numeric_string, bool is_row_major)
{
  source.append("__kernel void triangular_substitute_inplace( \n");
  source.append("          __global "); source.append(numeric_string); source.append(" * A, \n");
  source.append("          unsigned int A_start1, unsigned int A_start2, \n");
  source.append("          unsigned int A_inc1,   unsigned int A_inc2, \n");
  source.append("          unsigned int A_size1,  unsigned int A_size2, \n");
  source.append("          unsigned int A_internal_size1,  unsigned int A_internal_size2, \n");
  source.append("          __global "); source.append(numeric_string); source.append(" * v, \n");
  source.append("          unsigned int v_start, \n");
  source.append("          unsigned int v_inc, \n");
  source.append("          unsigned int v_size, \n");
  source.append("          unsigned int options) \n");
  source.append("{ \n");
  source.append("  "); source.append(numeric_string); source.append(" temp; \n");
  source.append("  unsigned int unit_diagonal_flag  = (options & (1 << 0)); \n");
  source.append("  unsigned int transposed_access_A = (options & (1 << 1)); \n");
  source.append("  unsigned int is_lower_solve      = (options & (1 << 2)); \n");
  source.append("  unsigned int row; \n");
  source.append("  for (unsigned int rows_processed = 0; rows_processed < A_size1; ++rows_processed)  \n");   //Note: A required to be square
  source.append("  { \n");
  source.append("    row = is_lower_solve ? rows_processed : ((A_size1 - rows_processed) - 1); \n");
  source.append("    barrier(CLK_GLOBAL_MEM_FENCE); \n");
  source.append("    if (!unit_diagonal_flag) \n");
  source.append("    { \n");
  source.append("      if (get_global_id(0) == 0) \n");
  if (is_row_major)
    source.append("        v[row * v_inc + v_start] /= A[(row * A_inc1 + A_start1) * A_internal_size2 + (row * A_inc2 + A_start2)]; \n");
  else
    source.append("        v[row * v_inc + v_start] /= A[(row * A_inc1 + A_start1) + (row * A_inc2 + A_start2) * A_internal_size1]; \n");
  source.append("   } \n");

  source.append("    barrier(CLK_GLOBAL_MEM_FENCE); \n");

  source.append("    temp = v[row * v_inc + v_start]; \n");

  source.append("    for (int elim = (is_lower_solve ? (row + get_global_id(0) + 1) : get_global_id(0)); \n");
  source.append("             elim < (is_lower_solve ? A_size1 : row); \n");
  source.append("             elim += get_global_size(0)) \n");
  if (is_row_major)
  {
    source.append("      v[elim * v_inc + v_start] -= temp * A[transposed_access_A ? ((row  * A_inc1 + A_start1) * A_internal_size2 + (elim * A_inc2 + A_start2)) \n");
    source.append("                                                                : ((elim * A_inc1 + A_start1) * A_internal_size2 + (row  * A_inc2 + A_start2))]; \n");
  }
  else
  {
    source.append("      v[elim * v_inc + v_start] -= temp * A[transposed_access_A ? ((row  * A_inc1 + A_start1) + (elim * A_inc2 + A_start2) * A_internal_size1) \n");
    source.append("                                                                : ((elim * A_inc1 + A_start1) + (row  * A_inc2 + A_start2) * A_internal_size1)]; \n");
  }
  source.append("  } \n");
  source.append("} \n");
}

template <typename StringT>
void generate_trans_kernel(StringT & source, std::string const & numeric_string, bool is_row_major)
{
  source.append("__kernel void trans_kernel(\n");
  source.append("           __global const ");source.append(numeric_string);source.append(" * A, \n");
  source.append("           unsigned int A_start1,          unsigned int A_start2, \n");
  source.append("           unsigned int A_internal_size1,  unsigned int A_internal_size2, \n");
  source.append("           unsigned int A_size1,           unsigned int A_size2, \n");
  source.append("           unsigned int A_stride1,         unsigned int A_stride2, \n");
  source.append("           __global ");source.append(numeric_string);source.append(" * B, \n");
  source.append("           unsigned int B_start1,          unsigned int B_start2, \n");
  source.append("           unsigned int B_internal_size1,  unsigned int B_internal_size2, \n");
  source.append("           unsigned int B_stride1,         unsigned int B_stride2) \n");
  source.append("{ \n");
  source.append("  for(unsigned int row = get_group_id(0); row < A_size1; row += get_num_groups(0))\n");
  source.append("  {  \n");
  source.append("    for(unsigned int col = get_local_id(0); col < A_size2; col += get_local_size(0))\n");
  source.append("    {  \n");
  if(is_row_major)
    source.append("      B[(B_start1 + B_stride1 * col) * B_internal_size2 + (B_start2 + B_stride2 * row)] = A[(A_start1 + A_stride1 * row) * A_internal_size2 + (A_start2 + A_stride2 * col)];  \n");
  else
    source.append("      B[(B_start1 + B_stride1 * col) + (B_start2 + B_stride2 * row) * B_internal_size1] = A[(A_start1 + A_stride1 * row) + (A_start2 + A_stride2 * col) * A_internal_size1];  \n");
  source.append("    } \n");
  source.append("  } \n");
  source.append("}  \n");
}

namespace detail
{
  inline std::string type_to_string(viennacl::row_major)    { return "row"; }
  inline std::string type_to_string(viennacl::column_major) { return "col"; }
}

//////////////////////////// Part 2: Main kernel class ////////////////////////////////////

/** @brief Main kernel class for generating OpenCL kernels for operations on/with viennacl::vector<> without involving matrices, multiple inner products, or element-wise operations other than addition or subtraction. */
template<typename NumericT>
class matrix
{
private:

  template<typename ScalarT1, typename ScalarT2>
  static void generate_ambm_impl2(device_specific::execution_handler & handler, std::string const & prefix, device_specific::matrix_axpy_template::parameters_type const & parameters, scheduler::operation_node_type ASSIGN_OP,
                                 viennacl::matrix_base<NumericT> const * x, viennacl::matrix_base<NumericT> const * y, ScalarT1 const * a,
                                 viennacl::matrix_base<NumericT> const * z, ScalarT2 const * b)
  {
    namespace ds = viennacl::device_specific;

    handler.add(prefix + "0000", ds::matrix_axpy_template(parameters), scheduler::preset::avbv(ASSIGN_OP, x, y, a, false, false, z, b, false, false));
    handler.add(prefix + "1000", ds::matrix_axpy_template(parameters), scheduler::preset::avbv(ASSIGN_OP, x, y, a, true, false, z, b, false, false));
    handler.add(prefix + "0100", ds::matrix_axpy_template(parameters), scheduler::preset::avbv(ASSIGN_OP, x, y, a, false, true, z, b, false, false));
    handler.add(prefix + "1100", ds::matrix_axpy_template(parameters), scheduler::preset::avbv(ASSIGN_OP, x, y, a, true, true, z, b, false, false));
    if (b)
    {
      handler.add(prefix + "0010", ds::matrix_axpy_template(parameters), scheduler::preset::avbv(ASSIGN_OP, x, y, a, false, false, z, b, true, false));
      handler.add(prefix + "1010", ds::matrix_axpy_template(parameters), scheduler::preset::avbv(ASSIGN_OP, x, y, a, true, false, z, b, true, false));
      handler.add(prefix + "0110", ds::matrix_axpy_template(parameters), scheduler::preset::avbv(ASSIGN_OP, x, y, a, false, true, z, b, true, false));
      handler.add(prefix + "1110", ds::matrix_axpy_template(parameters), scheduler::preset::avbv(ASSIGN_OP, x, y, a, true, true, z, b, true, false));

      handler.add(prefix + "0001", ds::matrix_axpy_template(parameters), scheduler::preset::avbv(ASSIGN_OP, x, y, a, false, false, z, b, false, true));
      handler.add(prefix + "1001", ds::matrix_axpy_template(parameters), scheduler::preset::avbv(ASSIGN_OP, x, y, a, true, false, z, b, false, true));
      handler.add(prefix + "0101", ds::matrix_axpy_template(parameters), scheduler::preset::avbv(ASSIGN_OP, x, y, a, false, true, z, b, false, true));
      handler.add(prefix + "1101", ds::matrix_axpy_template(parameters), scheduler::preset::avbv(ASSIGN_OP, x, y, a, true, true, z, b, false, true));

      handler.add(prefix + "0011", ds::matrix_axpy_template(parameters), scheduler::preset::avbv(ASSIGN_OP, x, y, a, false, false, z, b, true, true));
      handler.add(prefix + "1011", ds::matrix_axpy_template(parameters), scheduler::preset::avbv(ASSIGN_OP, x, y, a, true, false, z, b, true, true));
      handler.add(prefix + "0111", ds::matrix_axpy_template(parameters), scheduler::preset::avbv(ASSIGN_OP, x, y, a, false, true, z, b, true, true));
      handler.add(prefix + "1111", ds::matrix_axpy_template(parameters), scheduler::preset::avbv(ASSIGN_OP, x, y, a, true, true, z, b, true, true));
    }
  }

  template<typename ScalarT>
  static void generate_ambm_impl(device_specific::execution_handler & handler, std::string const & prefix, device_specific::matrix_axpy_template::parameters_type const & parameters, scheduler::operation_node_type ASSIGN_OP,
                                 viennacl::matrix_base<NumericT> const * x, viennacl::matrix_base<NumericT> const * y, ScalarT const * ha, viennacl::scalar<ScalarT> const * da,
                                 viennacl::matrix_base<NumericT> const * z, ScalarT const * hb, viennacl::scalar<ScalarT> const * db)
  {
    //x ASSIGN_OP a*y
    generate_ambm_impl2(handler, prefix + "hm_", parameters, ASSIGN_OP, x, y, ha, (viennacl::matrix_base<NumericT>*)NULL, (NumericT*)NULL);
    generate_ambm_impl2(handler, prefix + "dm_", parameters, ASSIGN_OP, x, y, da, (viennacl::matrix_base<NumericT>*)NULL, (NumericT*)NULL);

    //x ASSIGN_OP a*y + b*z
    generate_ambm_impl2(handler, prefix + "hmhm_", parameters, ASSIGN_OP, x, y, ha, z, hb);
    generate_ambm_impl2(handler, prefix + "dmhm_", parameters, ASSIGN_OP, x, y, da, z, hb);
    generate_ambm_impl2(handler, prefix + "hmdm_", parameters, ASSIGN_OP, x, y, ha, z, db);
    generate_ambm_impl2(handler, prefix + "dmdm_", parameters, ASSIGN_OP, x, y, da, z, db);
  }


public:
  static device_specific::execution_handler & execution_handler(bool is_row_major, viennacl::ocl::context & ctx)
  {
    static std::map<std::pair<bool, cl_context>, device_specific::execution_handler> handlers_map;
    cl_context h = ctx.handle().get();
    std::pair<bool, cl_context> key(is_row_major, h);
    if (handlers_map.find(key) == handlers_map.end())
    {
      viennacl::ocl::DOUBLE_PRECISION_CHECKER<NumericT>::apply(ctx);

      namespace ds = viennacl::device_specific;
      viennacl::ocl::device const & device = ctx.current_device();
      std::string program_name = viennacl::ocl::type_to_string<NumericT>::apply() + (is_row_major?"matrix_row":"matrix_col");
      handlers_map.insert(std::make_pair(key, ds::execution_handler(program_name, ctx, device)));
      ds::execution_handler & handler = viennacl::device_specific::at(handlers_map, key);

      ds::matrix_axpy_template::parameters_type matrix_axpy_params = ds::builtin_database::matrix_axpy_params<NumericT>(device);
      ds::vector_axpy_template::parameters_type vector_axpy_params = ds::builtin_database::vector_axpy_params<NumericT>(device);

      tools::shared_ptr<viennacl::matrix_base<NumericT> > pA, pB, pC;
      if (is_row_major)
      {
        pA.reset(new viennacl::matrix<NumericT, viennacl::row_major>());
        pB.reset(new viennacl::matrix<NumericT, viennacl::row_major>());
        pC.reset(new viennacl::matrix<NumericT, viennacl::row_major>());
      }
      else
      {
        pA.reset(new viennacl::matrix<NumericT, viennacl::column_major>());
        pB.reset(new viennacl::matrix<NumericT, viennacl::column_major>());
        pC.reset(new viennacl::matrix<NumericT, viennacl::column_major>());
      }

      viennacl::matrix_base<NumericT> & A = *pA;
      viennacl::matrix_base<NumericT> & B = *pB;
      viennacl::matrix_base<NumericT> & C = *pC;
      viennacl::vector<NumericT> x;
      viennacl::vector<NumericT> y;
      viennacl::scalar_matrix<NumericT> M(0,0,0,viennacl::context(ctx));
      viennacl::scalar_vector<NumericT> sx(0,0,viennacl::context(ctx));
      viennacl::scalar<NumericT> da;
      viennacl::scalar<NumericT> db;
      NumericT ha;
      NumericT hb;
      int hi = 0;
      unsigned int hui = 0;

      // fully parametrized kernels:
      generate_ambm_impl(handler, "assign_", matrix_axpy_params, scheduler::OPERATION_BINARY_ASSIGN_TYPE, &A, &B, &ha, &da, &C, &hb, &db);
      generate_ambm_impl(handler, "ip_add_", matrix_axpy_params, scheduler::OPERATION_BINARY_INPLACE_ADD_TYPE, &A, &B, &ha, &da, &C, &hb, &db);

      handler.add("assign_cpu", ds::matrix_axpy_template(matrix_axpy_params), scheduler::preset::assign_cpu(&A, &M));
      handler.add("matrix_diag_from_vector", ds::matrix_axpy_template(matrix_axpy_params), scheduler::preset::matrix_diag_from_vector(&x, &A, hi));
      handler.add("matrix_row", ds::vector_axpy_template(vector_axpy_params), scheduler::preset::matrix_row(&x, &A, hui));
      handler.add("matrix_column", ds::vector_axpy_template(vector_axpy_params), scheduler::preset::matrix_column(&x, &A, hui));
      handler.add("matrix_diag_to_vector", ds::vector_axpy_template(vector_axpy_params), scheduler::preset::matrix_diag_to_vector(&x, &A, hi));
      handler.add("diagonal_assign_cpu", ds::vector_axpy_template(vector_axpy_params), scheduler::preset::diagonal_assign_cpu(&A, &sx));
    }
    return viennacl::device_specific::at(handlers_map, key);
  }
};

// main kernel class
/** @brief Main kernel class for generating OpenCL kernels for elementwise operations other than addition and subtraction on/with viennacl::vector<>. */
template<typename NumericT>
struct matrix_element
{

public:
  static device_specific::execution_handler & execution_handler(bool is_row_major, viennacl::ocl::context & ctx)
  {
    static std::map<std::pair<bool, cl_context>, device_specific::execution_handler> handlers_map;
    cl_context h = ctx.handle().get();
    std::pair<bool, cl_context> key(is_row_major, h);
    if (handlers_map.find(key) == handlers_map.end())
    {
      viennacl::ocl::DOUBLE_PRECISION_CHECKER<NumericT>::apply(ctx);

      namespace ds = viennacl::device_specific;
      using namespace scheduler;
      using device_specific::tree_parsing::operator_string;

      std::string numeric_string = viennacl::ocl::type_to_string<NumericT>::apply();
      viennacl::ocl::device const & device = ctx.current_device();
      std::string program_name = viennacl::ocl::type_to_string<NumericT>::apply() + (is_row_major?"matrix_element_row":"matrix_element_col");
      handlers_map.insert(std::make_pair(key, ds::execution_handler(program_name, ctx, device)));
      ds::execution_handler & handler = viennacl::device_specific::at(handlers_map, key);
      ds::matrix_axpy_template::parameters_type matrix_axpy_params = ds::builtin_database::matrix_axpy_params<NumericT>(device);

      tools::shared_ptr<viennacl::matrix_base<NumericT> > pA, pB, pC;
      if (is_row_major)
      {
        pA.reset(new viennacl::matrix<NumericT, viennacl::row_major>());
        pB.reset(new viennacl::matrix<NumericT, viennacl::row_major>());
        pC.reset(new viennacl::matrix<NumericT, viennacl::row_major>());
      }
      else
      {
        pA.reset(new viennacl::matrix<NumericT, viennacl::column_major>());
        pB.reset(new viennacl::matrix<NumericT, viennacl::column_major>());
        pC.reset(new viennacl::matrix<NumericT, viennacl::column_major>());
      }

      viennacl::matrix_base<NumericT> & A = *pA;
      viennacl::matrix_base<NumericT> & B = *pB;
      viennacl::matrix_base<NumericT> & C = *pC;


      // unary operations
#define VIENNACL_ADD_UNARY(OPTYPE) handler.add(operator_string(OPTYPE), ds::matrix_axpy_template(matrix_axpy_params),scheduler::preset::unary_element_op(&A, &B, OPTYPE))
      if (numeric_string == "float" || numeric_string == "double")
      {
        VIENNACL_ADD_UNARY(OPERATION_UNARY_ACOS_TYPE);
        VIENNACL_ADD_UNARY(OPERATION_UNARY_ASIN_TYPE);
        VIENNACL_ADD_UNARY(OPERATION_UNARY_ATAN_TYPE);
        VIENNACL_ADD_UNARY(OPERATION_UNARY_CEIL_TYPE);
        VIENNACL_ADD_UNARY(OPERATION_UNARY_COS_TYPE);
        VIENNACL_ADD_UNARY(OPERATION_UNARY_COSH_TYPE);
        VIENNACL_ADD_UNARY(OPERATION_UNARY_EXP_TYPE);
        VIENNACL_ADD_UNARY(OPERATION_UNARY_FABS_TYPE);
        VIENNACL_ADD_UNARY(OPERATION_UNARY_FLOOR_TYPE);
        VIENNACL_ADD_UNARY(OPERATION_UNARY_LOG_TYPE);
        VIENNACL_ADD_UNARY(OPERATION_UNARY_LOG10_TYPE);
        VIENNACL_ADD_UNARY(OPERATION_UNARY_SIN_TYPE);
        VIENNACL_ADD_UNARY(OPERATION_UNARY_SINH_TYPE);
        VIENNACL_ADD_UNARY(OPERATION_UNARY_SQRT_TYPE);
        VIENNACL_ADD_UNARY(OPERATION_UNARY_TAN_TYPE);
        VIENNACL_ADD_UNARY(OPERATION_UNARY_TANH_TYPE);
      }
      else
      {
        VIENNACL_ADD_UNARY(OPERATION_UNARY_ABS_TYPE);
      }
#undef VIENNACL_ADD_UNARY

      // binary operations
#define VIENNACL_ADD_BINARY(OPTYPE) handler.add(operator_string(OPTYPE), ds::matrix_axpy_template(matrix_axpy_params),scheduler::preset::binary_element_op(&A, &B, &C, OPTYPE))
      VIENNACL_ADD_BINARY(OPERATION_BINARY_ELEMENT_DIV_TYPE);
      VIENNACL_ADD_BINARY(OPERATION_BINARY_ELEMENT_PROD_TYPE);
      if (numeric_string == "float" || numeric_string == "double")
      {
        VIENNACL_ADD_BINARY(OPERATION_BINARY_ELEMENT_POW_TYPE);
      }
#undef VIENNACL_ADD_BINARY

    }
    return viennacl::device_specific::at(handlers_map, key);
  }
};


/** @brief Main kernel class for generating OpenCL kernels for operations on/with viennacl::vector<> without involving matrices, multiple inner products, or element-wise operations other than addition or subtraction. */
template<typename NumericT>
class row_wise_reduction
{
public:
  static device_specific::execution_handler & execution_handler(viennacl::ocl::context & ctx)
  {
    static std::map<cl_context, device_specific::execution_handler> handlers_map;
    cl_context key = ctx.handle().get();
    if (handlers_map.find(key) == handlers_map.end())
    {
      viennacl::ocl::DOUBLE_PRECISION_CHECKER<NumericT>::apply(ctx);

      namespace ds = viennacl::device_specific;
      viennacl::ocl::device const & device = ctx.current_device();
      std::string program_name = viennacl::ocl::type_to_string<NumericT>::apply() + "_matrix_row_wise";
      handlers_map.insert(std::make_pair(key, ds::execution_handler(program_name, ctx, device)));
      ds::execution_handler & handler = viennacl::device_specific::at(handlers_map, key);

      viennacl::matrix<NumericT, viennacl::column_major> A;
      viennacl::vector<NumericT> x;
      viennacl::vector<NumericT> y;
      handler.add("mat_vec_T", ds::row_wise_reduction_template(ds::builtin_database::row_wise_reduction_params<NumericT>(device, 'T'), 'T'), scheduler::preset::mat_vec_prod(&A, true, &x, &y));
      handler.add("mat_vec_N", ds::row_wise_reduction_template(ds::builtin_database::row_wise_reduction_params<NumericT>(device, 'N'), 'N'), scheduler::preset::mat_vec_prod(&A, false, &x, &y));

    }
    return viennacl::device_specific::at(handlers_map, key);
  }
};

/** @brief Main kernel class for generating OpenCL kernels for operations on/with viennacl::vector<> without involving matrices, multiple inner products, or element-wise operations other than addition or subtraction. */
template<typename NumericT>
class matrix_prod
{
public:
  static device_specific::execution_handler & execution_handler(bool is_row_major, viennacl::ocl::context & ctx)
  {
    static std::map<std::pair<bool, cl_context>, device_specific::execution_handler> handlers_map;
    cl_context h = ctx.handle().get();
    std::pair<bool, cl_context> key(is_row_major, h);
    if (handlers_map.find(key) == handlers_map.end())
    {
      viennacl::ocl::DOUBLE_PRECISION_CHECKER<NumericT>::apply(ctx);

      namespace ds = viennacl::device_specific;
      viennacl::ocl::device const & device = ctx.current_device();
      std::string program_name = viennacl::ocl::type_to_string<NumericT>::apply() + (is_row_major?"_matrix_prod_row":"_matrix_prod_col");
      handlers_map.insert(std::make_pair(key, ds::execution_handler(program_name, ctx, device)));
      ds::execution_handler & handler = viennacl::device_specific::at(handlers_map, key);

      ds::matrix_product_template::parameters_type matrix_product_params_NN = ds::builtin_database::matrix_product_params<NumericT>(device, 'N', 'N');
      ds::matrix_product_template::parameters_type matrix_product_params_TN = ds::builtin_database::matrix_product_params<NumericT>(device, 'T', 'N');
      ds::matrix_product_template::parameters_type matrix_product_params_NT = ds::builtin_database::matrix_product_params<NumericT>(device, 'N', 'T');
      ds::matrix_product_template::parameters_type matrix_product_params_TT = ds::builtin_database::matrix_product_params<NumericT>(device, 'T', 'T');

      tools::shared_ptr<viennacl::matrix_base<NumericT> > pC;
      if (is_row_major)
        pC.reset(new viennacl::matrix<NumericT, viennacl::row_major>());
      else
        pC.reset(new viennacl::matrix<NumericT, viennacl::column_major>());

      //Dummy types. The values don't matter for the kernel generation.
      viennacl::matrix_base<NumericT>& C = *pC;
      viennacl::matrix<NumericT, viennacl::column_major> A;
      viennacl::matrix<NumericT, viennacl::column_major> B;
      NumericT alpha = 1;
      NumericT beta = 0;

      handler.add("prod_NN", ds::matrix_product_template(matrix_product_params_NN, 'N', 'N'), scheduler::preset::mat_mat_prod(alpha, &A, false, &B, false, beta, &C));
      handler.add("prod_TN", ds::matrix_product_template(matrix_product_params_TN, 'T', 'N'), scheduler::preset::mat_mat_prod(alpha, &A, true, &B, false, beta, &C));
      handler.add("prod_NT", ds::matrix_product_template(matrix_product_params_NT, 'N', 'T'), scheduler::preset::mat_mat_prod(alpha, &A, false, &B, true, beta, &C));
      handler.add("prod_TT", ds::matrix_product_template(matrix_product_params_TT, 'T', 'T'), scheduler::preset::mat_mat_prod(alpha, &A, true, &B, true, beta, &C));

    }
	return viennacl::device_specific::at(handlers_map, key);
  }
};

// main kernel class
/** @brief Main kernel class for generating OpenCL kernels for operations on/with dense matrix objects of type viennacl::matrix<>. */
template<typename NumericT, typename LayoutT>
struct matrix_legacy
{
  static std::string program_name()
  {
    return viennacl::ocl::type_to_string<NumericT>::apply() + "_matrix_" + detail::type_to_string(LayoutT());
  }

  static void init(viennacl::ocl::context & ctx)
  {
    static std::map<cl_context, bool> init_done;
    if (!init_done[ctx.handle().get()])
    {
      viennacl::ocl::DOUBLE_PRECISION_CHECKER<NumericT>::apply(ctx);
      std::string numeric_string = viennacl::ocl::type_to_string<NumericT>::apply();
      bool is_row_major = viennacl::is_row_major<LayoutT>::value;

      std::string source;
      source.reserve(8192);

      viennacl::ocl::append_double_precision_pragma<NumericT>(ctx, source);

      // kernels with mostly predetermined skeleton:
      generate_scaled_rank1_update(source, numeric_string, is_row_major, true);
      generate_scaled_rank1_update(source, numeric_string, is_row_major, false);

      if (numeric_string == "float" || numeric_string == "double")
      {
        generate_fft(source, numeric_string, is_row_major);
        generate_lu(source, numeric_string, is_row_major);
        generate_triangular_substitute_inplace(source, numeric_string, is_row_major);
        generate_trans_kernel(source, numeric_string, is_row_major);
      }

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

