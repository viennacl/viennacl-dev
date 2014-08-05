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

        template <typename StringType>
        void generate_pipelined_cg_vector_update(StringType & source, std::string const & numeric_string)
        {
          source.append("__kernel void cg_vector_update( \n");
          source.append("          __global "); source.append(numeric_string); source.append(" * result, \n");
          source.append("          "); source.append(numeric_string); source.append(" alpha, \n");
          source.append("          __global "); source.append(numeric_string); source.append(" * p, \n");
          source.append("          __global "); source.append(numeric_string); source.append(" * r, \n");
          source.append("          __global "); source.append(numeric_string); source.append(" * Ap, \n");
          source.append("          "); source.append(numeric_string); source.append(" beta, \n");
          source.append("          __global "); source.append(numeric_string); source.append(" * inner_prod_buffer, \n");
          source.append("          unsigned int size, \n");
          source.append("         __local "); source.append(numeric_string); source.append(" * shared_array) \n");
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

        template <typename StringType>
        void generate_compressed_matrix_pipelined_cg_prod(StringType & source, std::string const & numeric_string)
        {
          source.append("__kernel void cg_csr_prod( \n");
          source.append("          __global const unsigned int * row_indices, \n");
          source.append("          __global const unsigned int * column_indices, \n");
          source.append("          __global const "); source.append(numeric_string); source.append(" * elements, \n");
          source.append("          __global const "); source.append(numeric_string); source.append(" * p, \n");
          source.append("          __global "); source.append(numeric_string); source.append(" * Ap, \n");
          source.append("          unsigned int size, \n");
          source.append("          __global "); source.append(numeric_string); source.append(" * inner_prod_buffer, \n");
          source.append("          unsigned int buffer_size, \n");
          source.append("         __local "); source.append(numeric_string); source.append(" * shared_array_ApAp, \n");
          source.append("         __local "); source.append(numeric_string); source.append(" * shared_array_pAp) \n");
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
          source.append("    inner_prod_buffer[  buffer_size + get_group_id(0)] = shared_array_ApAp[0]; ");
          source.append("    inner_prod_buffer[2*buffer_size + get_group_id(0)] = shared_array_pAp[0]; ");
          source.append("  } ");

          source.append("} \n");

        }


        //////////////////////////// Part 2: Main kernel class ////////////////////////////////////

        // main kernel class
        /** @brief Main kernel class for generating specialized OpenCL kernels for fast iterative solvers. */
        template <typename NumericT>
        struct iterative
        {
          static std::string program_name()
          {
            return viennacl::ocl::type_to_string<NumericT>::apply() + "_iterative";
          }

          static void init(viennacl::ocl::context & ctx)
          {
            viennacl::ocl::DOUBLE_PRECISION_CHECKER<NumericT>::apply(ctx);
            std::string numeric_string = viennacl::ocl::type_to_string<NumericT>::apply();

            static std::map<cl_context, bool> init_done;
            if (!init_done[ctx.handle().get()])
            {
              std::string source;
              source.reserve(1024);

              viennacl::ocl::append_double_precision_pragma<NumericT>(ctx, source);

              generate_pipelined_cg_vector_update(source, numeric_string);
              generate_compressed_matrix_pipelined_cg_prod(source, numeric_string);

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

