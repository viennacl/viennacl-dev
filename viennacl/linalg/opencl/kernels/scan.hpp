#ifndef VIENNACL_LINALG_OPENCL_KERNELS_SCAN_HPP
#define VIENNACL_LINALG_OPENCL_KERNELS_SCAN_HPP

#include "viennacl/tools/tools.hpp"
#include "viennacl/ocl/kernel.hpp"
#include "viennacl/ocl/platform.hpp"
#include "viennacl/ocl/utils.hpp"

/** @file viennacl/linalg/opencl/kernels/scan.hpp
 *  @brief OpenCL kernel file for scan operations. To be merged back to vector operations. */
namespace viennacl
{
namespace linalg
{
namespace opencl
{
namespace kernels
{


template <typename StringType>
void generate_svd_inclusive_scan_kernel_1(StringType & source, std::string const & numeric_string)
{
  source.append("#define SECTION_SIZE 256\n");
  source.append("__kernel void inclusive_scan_1(__global "); source.append(numeric_string); source.append("* X, \n");
  source.append("                               uint startX, \n");
  source.append("                               uint incX, \n");
  source.append("                               uint InputSize, \n");

  source.append("                               __global "); source.append(numeric_string); source.append("* Y, \n");
  source.append("                               uint startY, \n");
  source.append("                               uint incY, \n");

  source.append("                               __global "); source.append(numeric_string); source.append("* S, \n");
  source.append("                               uint startS, \n");
  source.append("                               uint incS) \n");

  source.append("{ \n");
  source.append("    uint glb_id = get_global_id(0); \n");

  source.append("    uint grp_id = get_group_id(0); \n");
  source.append("    uint grp_nm = get_num_groups(0); \n");

  source.append("    uint lcl_id = get_local_id(0); \n");
  source.append("    uint lcl_sz = get_local_size(0); \n");
  source.append("    __local "); source.append(numeric_string); source.append(" XY[SECTION_SIZE]; \n");  //section size

  source.append("    if(glb_id < InputSize) \n");
  source.append("       XY[lcl_id] = X[glb_id * incX + startX]; \n");
  source.append(" \n");

  source.append("    for(uint stride = 1; stride < lcl_sz; stride *= 2) \n");
  source.append("    { \n");
  source.append("         barrier(CLK_LOCAL_MEM_FENCE);   \n");
  source.append("         int index = (lcl_id + 1) * 2 * stride - 1;  \n");
  source.append("         if(index < lcl_sz)      \n");
  source.append("             XY[index] += XY[index - stride];     \n");
  source.append("    } \n");

  source.append("     for(int stride = SECTION_SIZE / 4; stride > 0; stride /= 2) \n");             //Section size = 512
  source.append("     { \n");
  source.append("         barrier(CLK_LOCAL_MEM_FENCE); \n");
  source.append("         int index = (lcl_id + 1) * 2 * stride - 1; \n");
  source.append("         if(index + stride < lcl_sz)  \n");
  source.append("             XY[index + stride] += XY[index];  \n");
  source.append("     } \n");

  source.append("     barrier(CLK_LOCAL_MEM_FENCE);       \n");
  source.append("     if(glb_id < InputSize) \n");
  source.append("       Y[glb_id * incY + startY] = XY[lcl_id];  \n");
  source.append("     barrier(CLK_LOCAL_MEM_FENCE);       \n");
  source.append("     if(lcl_id == 0)     \n");
  source.append("     { \n");
  source.append("         S[grp_id * incS + startS] = XY[SECTION_SIZE - 1]; \n");                    //Section size = 512
  source.append("     } \n");
  source.append("} \n");
}

template <typename StringType>
void generate_svd_exclusive_scan_kernel_1(StringType & source, std::string const & numeric_string)
{
  source.append("__kernel void exclusive_scan_1(__global "); source.append(numeric_string); source.append("* X, \n");
  source.append("                               uint startX, \n");
  source.append("                               uint incX, \n");
  source.append("                               uint InputSize, \n");

  source.append("                               __global "); source.append(numeric_string); source.append("* Y, \n");
  source.append("                               uint startY, \n");
  source.append("                               uint incY, \n");

  source.append("                               __global "); source.append(numeric_string); source.append("* S, \n");
  source.append("                               uint startS, \n");
  source.append("                               uint incS) \n");

  source.append("{ \n");
  source.append("    uint glb_id = get_global_id(0); \n");

  source.append("    uint grp_id = get_group_id(0); \n");
  source.append("    uint grp_nm = get_num_groups(0); \n");

  source.append("    uint lcl_id = get_local_id(0); \n");
  source.append("    uint lcl_sz = get_local_size(0); \n");
  source.append("    __local "); source.append(numeric_string); source.append(" XY[SECTION_SIZE]; \n");           //section size

  source.append("    if(glb_id < InputSize + 1 && glb_id != 0) \n");
  source.append("       XY[lcl_id] = X[(glb_id - 1) * incX + startX]; \n");
  source.append("     if(glb_id == 0)     \n");
  source.append("         XY[0] = 0;      \n");
  source.append(" \n");

  source.append("    for(uint stride = 1; stride < lcl_sz; stride *= 2) \n");
  source.append("    { \n");
  source.append("         barrier(CLK_LOCAL_MEM_FENCE);   \n");
  source.append("         int index = (lcl_id + 1) * 2 * stride - 1;  \n");
  source.append("         if(index < lcl_sz)      \n");
  source.append("             XY[index] += XY[index - stride];     \n");
  source.append("    } \n");

  source.append("     for(int stride = SECTION_SIZE / 4; stride > 0; stride /= 2) \n");             //Section size = 512
  source.append("     { \n");
  source.append("         barrier(CLK_LOCAL_MEM_FENCE); \n");
  source.append("         int index = (lcl_id + 1) * 2 * stride - 1; \n");
  source.append("         if(index + stride < lcl_sz)  \n");
  source.append("             XY[index + stride] += XY[index];  \n");
  source.append("     } \n");
  source.append("     barrier(CLK_LOCAL_MEM_FENCE);       \n");

  source.append("     if(glb_id < InputSize) \n");
  source.append("       Y[glb_id * incY + startY] = XY[lcl_id];  \n");
  source.append("     barrier(CLK_LOCAL_MEM_FENCE);       \n");
  source.append("     if(lcl_id == 0)     \n");
  source.append("     { \n");
  source.append("         S[grp_id * incS + startS] = XY[SECTION_SIZE - 1]; \n");                    //Section size = 512
  source.append("     } \n");
  source.append("} \n");
}

template <typename StringType>
void generate_svd_scan_kernel_2(StringType & source, std::string const & numeric_string)
{
  source.append("__kernel void scan_kernel_2(__global "); source.append(numeric_string); source.append("* S_ref, \n");
  source.append("                            uint startS_ref, \n");
  source.append("                            uint incS_ref, \n");

  source.append("                            __global "); source.append(numeric_string); source.append("* S, \n");
  source.append("                            uint startS, \n");
  source.append("                            uint incS, \n");
  source.append("                            uint InputSize) \n");

  source.append(" { \n");
  source.append("    uint glb_id = get_global_id(0); \n");

  source.append("    uint grp_id = get_group_id(0); \n");
  source.append("    uint grp_nm = get_num_groups(0); \n");

  source.append("    uint lcl_id = get_local_id(0); \n");
  source.append("    uint lcl_sz = get_local_size(0); \n");
  source.append("    __local "); source.append(numeric_string); source.append(" XY[SECTION_SIZE]; \n");       //section size

  source.append("     if(glb_id < InputSize)           \n");
  source.append("         XY[lcl_id] = S[glb_id * incS + startS];     \n");

  source.append("     for(uint stride = 1; stride < lcl_sz; stride *= 2)  \n");
  source.append("     {   \n");
  source.append("         barrier(CLK_LOCAL_MEM_FENCE);       \n");
  source.append("         int index = (lcl_id + 1) * 2 * stride - 1;   \n");
  source.append("         if(index < lcl_sz)  \n");
  source.append("             XY[index] += XY[index - stride]; \n");
  source.append("     }   \n");

  source.append("     for(int stride = SECTION_SIZE / 4; stride > 0; stride /= 2)  \n");
  source.append("     {   \n");
  source.append("         barrier(CLK_LOCAL_MEM_FENCE);                   \n");
  source.append("         int index = (lcl_id + 1) * 2 * stride - 1;      \n");
  source.append("         if(index + stride < lcl_sz)                     \n");
  source.append("             XY[index + stride] += XY[index];            \n");
  source.append("     }   \n");

  source.append("     barrier(CLK_LOCAL_MEM_FENCE);                       \n");
  source.append("     if(glb_id < InputSize)           \n");
  source.append("     {           \n");
  source.append("         S[glb_id * incS + startS] = XY[lcl_id];              \n");
  source.append("         S_ref[glb_id * incS_ref + startS_ref] = XY[lcl_id];  \n");
  source.append("     }           \n");
  source.append(" } \n");
}

template <typename StringType>
void generate_svd_scan_kernel_3(StringType & source, std::string const & numeric_string)
{
  source.append("__kernel void scan_kernel_3(__global "); source.append(numeric_string); source.append("* S_ref, \n");
  source.append("                            uint startS_ref, \n");
  source.append("                            uint incS_ref, \n");

  source.append("                            __global "); source.append(numeric_string); source.append("* S, \n");
  source.append("                            uint startS, \n");
  source.append("                            uint incS) \n");

  source.append(" { \n");
  source.append("    uint glb_id = get_global_id(0); \n");

  source.append("    uint grp_id = get_group_id(0); \n");
  source.append("    uint grp_nm = get_num_groups(0); \n");

  source.append("    uint lcl_id = get_local_id(0); \n");
  source.append("    uint lcl_sz = get_local_size(0); \n");


  source.append("     for(int j = 1; j <= grp_id; j++)  \n");
  source.append("         S[glb_id * incS + startS] += S_ref[(j * lcl_sz - 1) * incS_ref + startS_ref];    \n");
  source.append(" } \n");
}

template <typename StringType>
void generate_svd_scan_kernel_4(StringType & source, std::string const & numeric_string)
{
  source.append("__kernel void scan_kernel_4(__global "); source.append(numeric_string); source.append("* S, \n");
  source.append("                            uint startS, \n");
  source.append("                            uint incS, \n");

  source.append("                            __global "); source.append(numeric_string); source.append("* Y, \n");
  source.append("                            uint startY, \n");
  source.append("                            uint incY, \n");
  source.append("                            uint OutputSize) \n");

  source.append(" { \n");
  source.append("    barrier(CLK_LOCAL_MEM_FENCE);                   \n");
  source.append("    uint glb_id = get_global_id(0); \n");

  source.append("    uint grp_id = get_group_id(0); \n");
  source.append("    uint grp_nm = get_num_groups(0); \n");

  source.append("    uint lcl_id = get_local_id(0); \n");
  source.append("    uint lcl_sz = get_local_size(0); \n");


  source.append("    uint var = (grp_id + 1) * lcl_sz + lcl_id;          \n");
  source.append("    if(var < OutputSize)         \n");
  source.append("         Y[var * incY + startY] += S[grp_id * incS + startS]; \n");
  source.append(" } \n");
}




// main kernel class
/** @brief Main kernel class for generating OpenCL kernels for singular value decomposition of dense matrices. */
template<typename NumericT>
struct scan
{
  static std::string program_name()
  {
    return viennacl::ocl::type_to_string<NumericT>::apply() + "_scan";
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

      generate_svd_inclusive_scan_kernel_1(source, numeric_string);
      generate_svd_exclusive_scan_kernel_1(source, numeric_string);
      generate_svd_scan_kernel_2(source, numeric_string);
      generate_svd_scan_kernel_3(source, numeric_string);
      generate_svd_scan_kernel_4(source, numeric_string);

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

