#ifndef VIENNACL_LINALG_OPENCL_KERNELS_COORDINATE_MATRIX_HPP
#define VIENNACL_LINALG_OPENCL_KERNELS_COORDINATE_MATRIX_HPP

#include "viennacl/tools/tools.hpp"
#include "viennacl/ocl/kernel.hpp"
#include "viennacl/ocl/platform.hpp"
#include "viennacl/ocl/utils.hpp"

/** @file viennacl/linalg/opencl/kernels/coordinate_matrix.hpp
 *  @brief OpenCL kernel file for coordinate_matrix operations */
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
        void generate_coordinate_matrix_vec_mul(StringType & source, std::string const & numeric_string)
        {
          source.append("__kernel void vec_mul( \n");
          source.append("  __global const uint2 * coords,  \n");//(row_index, column_index)
          source.append("  __global const "); source.append(numeric_string); source.append(" * elements, \n");
          source.append("  __global const uint  * group_boundaries, \n");
          source.append("  __global const "); source.append(numeric_string); source.append(" * x, \n");
          source.append("  uint4 layout_x, \n");
          source.append("  __global "); source.append(numeric_string); source.append(" * result, \n");
          source.append("  uint4 layout_result, \n");
          source.append("  __local unsigned int * shared_rows, \n");
          source.append("  __local "); source.append(numeric_string); source.append(" * inter_results) \n");
          source.append("{ \n");
          source.append("  uint2 tmp; \n");
          source.append("  "); source.append(numeric_string); source.append(" val; \n");
          source.append("  uint last_index  = get_local_size(0) - 1; \n");
          source.append("  uint group_start = group_boundaries[get_group_id(0)]; \n");
          source.append("  uint group_end   = group_boundaries[get_group_id(0) + 1]; \n");
          source.append("  uint k_end = (group_end > group_start) ? 1 + (group_end - group_start - 1) / get_local_size(0) : 0;   // -1 in order to have correct behavior if group_end - group_start == j * get_local_size(0) \n");

          source.append("  uint local_index = 0; \n");

          source.append("  for (uint k = 0; k < k_end; ++k) { \n");
          source.append("    local_index = group_start + k * get_local_size(0) + get_local_id(0); \n");

          source.append("    tmp = (local_index < group_end) ? coords[local_index] : (uint2) 0; \n");
          source.append("    val = (local_index < group_end) ? elements[local_index] * x[tmp.y * layout_x.y + layout_x.x] : 0; \n");

          //check for carry from previous loop run:
          source.append("    if (get_local_id(0) == 0 && k > 0) { \n");
          source.append("      if (tmp.x == shared_rows[last_index]) \n");
          source.append("        val += inter_results[last_index]; \n");
          source.append("      else \n");
          source.append("        result[shared_rows[last_index] * layout_result.y + layout_result.x] = inter_results[last_index]; \n");
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

          source.append("    if (get_local_id(0) != last_index && \n");
          source.append("      shared_rows[get_local_id(0)] != shared_rows[get_local_id(0) + 1] && \n");
          source.append("      inter_results[get_local_id(0)] != 0) { \n");
          source.append("      result[tmp.x * layout_result.y + layout_result.x] = inter_results[get_local_id(0)]; \n");
          source.append("    } \n");

          source.append("    barrier(CLK_LOCAL_MEM_FENCE); \n");
          source.append("  }  \n"); //for k

          source.append("  if (get_local_id(0) == last_index && inter_results[last_index] != 0) \n");
          source.append("    result[tmp.x * layout_result.y + layout_result.x] = inter_results[last_index]; \n");
          source.append("} \n");

        }

        template <typename StringType>
        void generate_coordinate_matrix_row_info_extractor(StringType & source, std::string const & numeric_string)
        {
          source.append("__kernel void row_info_extractor( \n");
          source.append("          __global const uint2 * coords,  \n");//(row_index, column_index)
          source.append("          __global const "); source.append(numeric_string); source.append(" * elements, \n");
          source.append("          __global const uint  * group_boundaries, \n");
          source.append("          __global "); source.append(numeric_string); source.append(" * result, \n");
          source.append("          unsigned int option, \n");
          source.append("          __local unsigned int * shared_rows, \n");
          source.append("          __local "); source.append(numeric_string); source.append(" * inter_results) \n");
          source.append("{ \n");
          source.append("  uint2 tmp; \n");
          source.append("  "); source.append(numeric_string); source.append(" val; \n");
          source.append("  uint last_index  = get_local_size(0) - 1; \n");
          source.append("  uint group_start = group_boundaries[get_group_id(0)]; \n");
          source.append("  uint group_end   = group_boundaries[get_group_id(0) + 1]; \n");
          source.append("  uint k_end = (group_end > group_start) ? 1 + (group_end - group_start - 1) / get_local_size(0) : ("); source.append(numeric_string); source.append(")0; \n");   // -1 in order to have correct behavior if group_end - group_start == j * get_local_size(0)

          source.append("  uint local_index = 0; \n");

          source.append("  for (uint k = 0; k < k_end; ++k) \n");
          source.append("  { \n");
          source.append("    local_index = group_start + k * get_local_size(0) + get_local_id(0); \n");

          source.append("    tmp = (local_index < group_end) ? coords[local_index] : (uint2) 0; \n");
          source.append("    val = (local_index < group_end && (option != 3 || tmp.x == tmp.y) ) ? elements[local_index] : 0; \n");

              //check for carry from previous loop run:
          source.append("    if (get_local_id(0) == 0 && k > 0) \n");
          source.append("    { \n");
          source.append("      if (tmp.x == shared_rows[last_index]) \n");
          source.append("      { \n");
          source.append("        switch (option) \n");
          source.append("        { \n");
          source.append("          case 0: \n"); //inf-norm
          source.append("          case 3: \n"); //diagonal entry
          source.append("            val = max(val, fabs(inter_results[last_index])); \n");
          source.append("            break; \n");

          source.append("          case 1: \n"); //1-norm
          source.append("            val = fabs(val) + inter_results[last_index]; \n");
          source.append("            break; \n");

          source.append("          case 2: \n"); //2-norm
          source.append("            val = sqrt(val * val + inter_results[last_index]); \n");
          source.append("            break; \n");

          source.append("          default: \n");
          source.append("            break; \n");
          source.append("        } \n");
          source.append("      } \n");
          source.append("      else \n");
          source.append("      { \n");
          source.append("        switch (option) \n");
          source.append("        { \n");
          source.append("          case 0: \n"); //inf-norm
          source.append("          case 1: \n"); //1-norm
          source.append("          case 3: \n"); //diagonal entry
          source.append("            result[shared_rows[last_index]] = inter_results[last_index]; \n");
          source.append("            break; \n");

          source.append("          case 2: \n"); //2-norm
          source.append("            result[shared_rows[last_index]] = sqrt(inter_results[last_index]); \n");
          source.append("          default: \n");
          source.append("            break; \n");
          source.append("        } \n");
          source.append("      } \n");
          source.append("    } \n");

              //segmented parallel reduction begin
          source.append("    barrier(CLK_LOCAL_MEM_FENCE); \n");
          source.append("    shared_rows[get_local_id(0)] = tmp.x; \n");
          source.append("    switch (option) \n");
          source.append("    { \n");
          source.append("      case 0: \n");
          source.append("      case 3: \n");
          source.append("        inter_results[get_local_id(0)] = val; \n");
          source.append("        break; \n");
          source.append("      case 1: \n");
          source.append("        inter_results[get_local_id(0)] = fabs(val); \n");
          source.append("        break; \n");
          source.append("      case 2: \n");
          source.append("        inter_results[get_local_id(0)] = val * val; \n");
          source.append("      default: \n");
          source.append("        break; \n");
          source.append("    } \n");
          source.append("    "); source.append(numeric_string); source.append(" left = 0; \n");
          source.append("    barrier(CLK_LOCAL_MEM_FENCE); \n");

          source.append("    for (unsigned int stride = 1; stride < get_local_size(0); stride *= 2) \n");
          source.append("    { \n");
          source.append("      left = (get_local_id(0) >= stride && tmp.x == shared_rows[get_local_id(0) - stride]) ? inter_results[get_local_id(0) - stride] : ("); source.append(numeric_string); source.append(")0; \n");
          source.append("      barrier(CLK_LOCAL_MEM_FENCE); \n");
          source.append("      switch (option) \n");
          source.append("      { \n");
          source.append("        case 0: \n"); //inf-norm
          source.append("        case 3: \n"); //diagonal entry
          source.append("          inter_results[get_local_id(0)] = max(inter_results[get_local_id(0)], left); \n");
          source.append("          break; \n");

          source.append("        case 1: \n"); //1-norm
          source.append("          inter_results[get_local_id(0)] += left; \n");
          source.append("          break; \n");

          source.append("        case 2: \n"); //2-norm
          source.append("          inter_results[get_local_id(0)] += left; \n");
          source.append("          break; \n");

          source.append("        default: \n");
          source.append("          break; \n");
          source.append("      } \n");
          source.append("      barrier(CLK_LOCAL_MEM_FENCE); \n");
          source.append("    } \n");
              //segmented parallel reduction end

          source.append("    if (get_local_id(0) != last_index && \n");
          source.append("        shared_rows[get_local_id(0)] != shared_rows[get_local_id(0) + 1] && \n");
          source.append("        inter_results[get_local_id(0)] != 0) \n");
          source.append("    { \n");
          source.append("      result[tmp.x] = (option == 2) ? sqrt(inter_results[get_local_id(0)]) : inter_results[get_local_id(0)]; \n");
          source.append("    } \n");

          source.append("    barrier(CLK_LOCAL_MEM_FENCE); \n");
          source.append("  } \n"); //for k

          source.append("  if (get_local_id(0) == last_index && inter_results[last_index] != 0) \n");
          source.append("    result[tmp.x] = (option == 2) ? sqrt(inter_results[last_index]) : inter_results[last_index]; \n");
          source.append("} \n");
        }

        //////////////////////////// Part 2: Main kernel class ////////////////////////////////////

        // main kernel class
        template <typename NumericT>
        struct coordinate_matrix
        {
          static std::string program_name()
          {
            return viennacl::ocl::type_to_string<NumericT>::apply() + "_coordinate_matrix";
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

              generate_coordinate_matrix_vec_mul(source, numeric_string);

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

