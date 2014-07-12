#ifndef VIENNACL_LINALG_OPENCL_KERNELS_MATRIX_HPP
#define VIENNACL_LINALG_OPENCL_KERNELS_MATRIX_HPP

#include "viennacl/scheduler/preset.hpp"
#include "viennacl/tools/tools.hpp"
#include "viennacl/ocl/kernel.hpp"
#include "viennacl/ocl/platform.hpp"
#include "viennacl/ocl/utils.hpp"

#include "viennacl/device_specific/builtin_database/vector_axpy.hpp"
#include "viennacl/device_specific/builtin_database/matrix_axpy.hpp"
#include "viennacl/device_specific/builtin_database/row_wise_reduction.hpp"

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
        using namespace viennacl::device_specific;

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

        template<typename T, typename ScalarType1, typename ScalarType2>
        inline void generate_ambm_impl2(std::string & source, matrix_axpy_template::parameters const & parameters, scheduler::operation_node_type ASSIGN_OP,
                                       viennacl::matrix_base<T> const * x, viennacl::matrix_base<T> const * y, ScalarType1 const * a,
                                       viennacl::matrix_base<T> const * z, ScalarType2 const * b,
                                        std::string const & prefix, viennacl::ocl::device const & device)
        {
          source.append(matrix_axpy_template(parameters, prefix + "0000").generate(scheduler::preset::avbv(ASSIGN_OP, x, y, a, false, false, z, b, false, false),device));
          source.append(matrix_axpy_template(parameters, prefix + "1000").generate(scheduler::preset::avbv(ASSIGN_OP, x, y, a, true, false, z, b, false, false),device));
          source.append(matrix_axpy_template(parameters, prefix + "0100").generate(scheduler::preset::avbv(ASSIGN_OP, x, y, a, false, true, z, b, false, false),device));
          source.append(matrix_axpy_template(parameters, prefix + "1100").generate(scheduler::preset::avbv(ASSIGN_OP, x, y, a, true, true, z, b, false, false),device));
          if(b)
          {
            source.append(matrix_axpy_template(parameters, prefix + "0010").generate(scheduler::preset::avbv(ASSIGN_OP, x, y, a, false, false, z, b, true, false),device));
            source.append(matrix_axpy_template(parameters, prefix + "1010").generate(scheduler::preset::avbv(ASSIGN_OP, x, y, a, true, false, z, b, true, false),device));
            source.append(matrix_axpy_template(parameters, prefix + "0110").generate(scheduler::preset::avbv(ASSIGN_OP, x, y, a, false, true, z, b, true, false),device));
            source.append(matrix_axpy_template(parameters, prefix + "1110").generate(scheduler::preset::avbv(ASSIGN_OP, x, y, a, true, true, z, b, true, false),device));

            source.append(matrix_axpy_template(parameters, prefix + "0001").generate(scheduler::preset::avbv(ASSIGN_OP, x, y, a, false, false, z, b, false, true),device));
            source.append(matrix_axpy_template(parameters, prefix + "1001").generate(scheduler::preset::avbv(ASSIGN_OP, x, y, a, true, false, z, b, false, true),device));
            source.append(matrix_axpy_template(parameters, prefix + "0101").generate(scheduler::preset::avbv(ASSIGN_OP, x, y, a, false, true, z, b, false, true),device));
            source.append(matrix_axpy_template(parameters, prefix + "1101").generate(scheduler::preset::avbv(ASSIGN_OP, x, y, a, true, true, z, b, false, true),device));

            source.append(matrix_axpy_template(parameters, prefix + "0011").generate(scheduler::preset::avbv(ASSIGN_OP, x, y, a, false, false, z, b, true, true),device));
            source.append(matrix_axpy_template(parameters, prefix + "1011").generate(scheduler::preset::avbv(ASSIGN_OP, x, y, a, true, false, z, b, true, true),device));
            source.append(matrix_axpy_template(parameters, prefix + "0111").generate(scheduler::preset::avbv(ASSIGN_OP, x, y, a, false, true, z, b, true, true),device));
            source.append(matrix_axpy_template(parameters, prefix + "1111").generate(scheduler::preset::avbv(ASSIGN_OP, x, y, a, true, true, z, b, true, true),device));
          }
        }

        template<typename T, typename ScalarType>
        inline void generate_ambm_impl(std::string & source, matrix_axpy_template::parameters const & parameters, scheduler::operation_node_type ASSIGN_OP,
                                       viennacl::matrix_base<T> const * x, viennacl::matrix_base<T> const * y, ScalarType const * ha, viennacl::scalar<ScalarType> const * da,
                                       viennacl::matrix_base<T> const * z, ScalarType const * hb, viennacl::scalar<ScalarType> const * db,
                                       std::string const & prefix, viennacl::ocl::device const & device)
        {
          //x ASSIGN_OP a*y
          generate_ambm_impl2(source, parameters, ASSIGN_OP, x, y, ha, (viennacl::matrix_base<T>*)NULL, (T*)NULL, prefix + "hm_", device);
          generate_ambm_impl2(source, parameters, ASSIGN_OP, x, y, da, (viennacl::matrix_base<T>*)NULL, (T*)NULL, prefix + "dm_", device);

          //x ASSIGN_OP a*y + b*z
          generate_ambm_impl2(source, parameters, ASSIGN_OP, x, y, ha, z, hb, prefix + "hmhm_", device);
          generate_ambm_impl2(source, parameters, ASSIGN_OP, x, y, da, z, hb, prefix + "dmhm_", device);
          generate_ambm_impl2(source, parameters, ASSIGN_OP, x, y, ha, z, db, prefix + "hmdm_", device);
          generate_ambm_impl2(source, parameters, ASSIGN_OP, x, y, da, z, db, prefix + "dmdm_", device);
        }


        template <typename StringType>
        void generate_fft(StringType & source, std::string const & numeric_string, bool is_row_major)
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
          source.append("    for(unsigned int batch_id = 0; batch_id < batch_num; batch_id++) { \n");
          source.append("        for(unsigned int k = get_global_id(0); k < size; k += get_global_size(0)) { \n");
          source.append("            "); source.append(numeric_string); source.append("2 f = 0.0f; \n");
          source.append(" \n");
          source.append("            for(unsigned int n = 0; n < size; n++) { \n");
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

          source.append("    for(unsigned int batch_id = 0; batch_id < batch_num; batch_id++) { \n");
          source.append("        for(unsigned int tid = glb_id; tid < half_size; tid += glb_sz) { \n");
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

          source.append("     for(unsigned int batch_id = grp_id; batch_id < batch_num; batch_id += grp_num) { \n");
                  //unsigned int base_offset = stride * batch_id; \n");
                  //copy chunk of global memory to local \n");
          source.append("         for(unsigned int p = lcl_id; p < size; p += lcl_sz) { \n");
          source.append("             unsigned int v = get_reorder_num(p, bit_size); \n");
          if (is_row_major)
            source.append("             lcl_input[v] = input[batch_id * stride + p]; \n"); //index
          else
            source.append("             lcl_input[v] = input[p * stride + batch_id]; \n"); //index
          source.append("         } \n");

          source.append("         barrier(CLK_LOCAL_MEM_FENCE); \n");

                  //performs Cooley-Tukey FFT on local array
          source.append("         for(unsigned int s = 0; s < bit_size; s++) { \n");
          source.append("             unsigned int ss = 1 << s; \n");

          source.append("             "); source.append(numeric_string); source.append(" cs, sn; \n");

          source.append("             for(unsigned int tid = lcl_id; tid < size; tid += lcl_sz) { \n");
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
          source.append("         for(unsigned int p = lcl_id; p < size; p += lcl_sz) { \n");
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

          source.append("    for(unsigned int batch_id = 0; batch_id < batch_num; batch_id++) { \n");
          source.append("        for(unsigned int i = glb_id; i < size; i += glb_sz) { \n");
          source.append("            unsigned int v = get_reorder_num_2(i, bit_size); \n");

          source.append("            if(i < v) {\n");
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

        template <typename StringType>
        void generate_lu(StringType & source, std::string const & numeric_string, bool is_row_major)
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


        template <typename StringType>
        void generate_scaled_rank1_update(StringType & source, std::string const & numeric_string, bool is_row_major, bool alpha_on_cpu)
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

        template <typename StringType>
        void generate_triangular_substitute_inplace(StringType & source, std::string const & numeric_string, bool is_row_major)
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
          source.append("    if (!unit_diagonal_flag) \n");
          source.append("    { \n");
          source.append("      barrier(CLK_GLOBAL_MEM_FENCE); \n");
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

        namespace detail
        {
          inline std::string type_to_string(viennacl::row_major)    { return "row"; }
          inline std::string type_to_string(viennacl::column_major) { return "col"; }
        }

        //////////////////////////// Part 2: Main kernel class ////////////////////////////////////

        // main kernel class
        /** @brief Main kernel class for generating OpenCL kernels for operations on/with dense matrix objects of type viennacl::matrix<>. */
        template <typename NumericT, typename F>
        struct matrix
        {
          static std::string program_name()
          {
            return viennacl::ocl::type_to_string<NumericT>::apply() + "_matrix_" + detail::type_to_string(F());
          }

          static void init(viennacl::ocl::context & ctx)
          {
            using namespace device_specific::builtin_database;

            viennacl::ocl::DOUBLE_PRECISION_CHECKER<NumericT>::apply(ctx);
            std::string numeric_string = viennacl::ocl::type_to_string<NumericT>::apply();
            bool is_row_major = viennacl::is_row_major<F>::value;

            static std::map<cl_context, bool> init_done;
            if (!init_done[ctx.handle().get()])
            {
              viennacl::ocl::device const & device = ctx.current_device();

              std::string source;
              source.reserve(8192);

              viennacl::ocl::append_double_precision_pragma<NumericT>(ctx, source);

              matrix_axpy_template::parameters matrix_axpy_params = device_specific::builtin_database::matrix_axpy_params<NumericT>(device);
              row_wise_reduction_template::parameters row_wise_reduction_params_N = device_specific::builtin_database::row_wise_reduction_params<NumericT>(device, 'N');
              row_wise_reduction_template::parameters row_wise_reduction_params_T = device_specific::builtin_database::row_wise_reduction_params<NumericT>(device, 'T');
              vector_axpy_template::parameters vector_axpy_params = device_specific::builtin_database::vector_axpy_params<NumericT>(device);

              viennacl::vector<NumericT> x;
              viennacl::vector<NumericT> y;
              viennacl::matrix<NumericT, F> A;
              viennacl::matrix<NumericT, F> B;
              viennacl::matrix<NumericT, F> C;
              viennacl::scalar_matrix<NumericT> M(0,0,0);
              viennacl::scalar_vector<NumericT> sx(0,0);
              viennacl::scalar<NumericT> da;
              viennacl::scalar<NumericT> db;
              NumericT ha;
              NumericT hb;
              int hi = 0;
              unsigned int hui = 0;

              // fully parametrized kernels:
              generate_ambm_impl(source, matrix_axpy_params, scheduler::OPERATION_BINARY_ASSIGN_TYPE, &A, &B, &ha, &da, &C, &hb, &db, "assign_",device);
              generate_ambm_impl(source, matrix_axpy_params, scheduler::OPERATION_BINARY_INPLACE_ADD_TYPE, &A, &B, &ha, &da, &C, &hb, &db, "ip_add_",device);

              source.append(matrix_axpy_template(matrix_axpy_params, "assign_cpu").generate(scheduler::preset::assign_cpu(&A, &M),device));
              source.append(matrix_axpy_template(matrix_axpy_params, "matrix_diag_from_vector").generate(scheduler::preset::matrix_diag_from_vector(&x, &A, hi),device));
              source.append(vector_axpy_template(vector_axpy_params, "matrix_row").generate(scheduler::preset::matrix_row(&x, &A, hui),device));
              source.append(vector_axpy_template(vector_axpy_params, "matrix_column").generate(scheduler::preset::matrix_column(&x, &A, hui),device));
              source.append(vector_axpy_template(vector_axpy_params, "matrix_diag_to_vector").generate(scheduler::preset::matrix_diag_to_vector(&x, &A, hi),device));
              source.append(vector_axpy_template(vector_axpy_params, "diagonal_assign_cpu").generate(scheduler::preset::diagonal_assign_cpu(&A, &sx),device));

              // kernels with mostly predetermined skeleton:
              generate_scaled_rank1_update(source, numeric_string, is_row_major, true);
              generate_scaled_rank1_update(source, numeric_string, is_row_major, false);

              source.append(row_wise_reduction_template(row_wise_reduction_params_T, 'N', "mat_vec_T").generate(scheduler::preset::mat_vec_prod(&A, true, &x, &y), device));
              source.append(row_wise_reduction_template(row_wise_reduction_params_N, 'T', "mat_vec_N").generate(scheduler::preset::mat_vec_prod(&A, false, &x, &y), device));

              if (numeric_string == "float" || numeric_string == "double")
              {
                generate_fft(source, numeric_string, is_row_major);
                generate_lu(source, numeric_string, is_row_major);
                generate_triangular_substitute_inplace(source, numeric_string, is_row_major);
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

