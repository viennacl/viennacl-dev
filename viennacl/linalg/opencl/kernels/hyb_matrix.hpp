#ifndef VIENNACL_LINALG_OPENCL_KERNELS_HYB_MATRIX_HPP
#define VIENNACL_LINALG_OPENCL_KERNELS_HYB_MATRIX_HPP

#include "viennacl/tools/tools.hpp"
#include "viennacl/ocl/kernel.hpp"
#include "viennacl/ocl/platform.hpp"
#include "viennacl/ocl/utils.hpp"

/** @file viennacl/linalg/opencl/kernels/hyb_matrix.hpp
 *  @brief OpenCL kernel file for hyb_matrix operations */
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
        void generate_hyb_vec_mul(StringType & source, std::string const & numeric_string)
        {
          source.append("__kernel void vec_mul( \n");
          source.append("  const __global int* ell_coords, \n");
          source.append("  const __global "); source.append(numeric_string); source.append("* ell_elements, \n");
          source.append("  const __global uint* csr_rows, \n");
          source.append("  const __global uint* csr_cols, \n");
          source.append("  const __global "); source.append(numeric_string); source.append("* csr_elements, \n");
          source.append("  const __global "); source.append(numeric_string); source.append(" * x, \n");
          source.append("  uint4 layout_x, \n");
          source.append("  __global "); source.append(numeric_string); source.append(" * result, \n");
          source.append("  uint4 layout_result, \n");
          source.append("  unsigned int row_num, \n");
          source.append("  unsigned int internal_row_num, \n");
          source.append("  unsigned int items_per_row, \n");
          source.append("  unsigned int aligned_items_per_row) \n");
          source.append("{ \n");
          source.append("  uint glb_id = get_global_id(0); \n");
          source.append("  uint glb_sz = get_global_size(0); \n");

          source.append("  for(uint row_id = glb_id; row_id < row_num; row_id += glb_sz) { \n");
          source.append("    "); source.append(numeric_string); source.append(" sum = 0; \n");

          source.append("    uint offset = row_id; \n");
          source.append("    for(uint item_id = 0; item_id < items_per_row; item_id++, offset += internal_row_num) { \n");
          source.append("      "); source.append(numeric_string); source.append(" val = ell_elements[offset]; \n");

          source.append("      if(val != ("); source.append(numeric_string); source.append(")0) { \n");
          source.append("        int col = ell_coords[offset]; \n");
          source.append("        sum += (x[col * layout_x.y + layout_x.x] * val); \n");
          source.append("      } \n");

          source.append("    } \n");

          source.append("    uint col_begin = csr_rows[row_id]; \n");
          source.append("    uint col_end   = csr_rows[row_id + 1]; \n");

          source.append("    for(uint item_id = col_begin; item_id < col_end; item_id++) {  \n");
          source.append("      sum += (x[csr_cols[item_id] * layout_x.y + layout_x.x] * csr_elements[item_id]); \n");
          source.append("    } \n");

          source.append("    result[row_id * layout_result.y + layout_result.x] = sum; \n");
          source.append("  } \n");
          source.append("} \n");
        }


        //////////////////////////// Part 2: Main kernel class ////////////////////////////////////

        // main kernel class
        template <typename NumericT>
        struct hyb_matrix
        {
          static std::string program_name()
          {
            return viennacl::ocl::type_to_string<NumericT>::apply() + "_hyb_matrix";
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

              generate_hyb_vec_mul(source, numeric_string);

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

