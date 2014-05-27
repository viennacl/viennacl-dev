#ifndef VIENNACL_LINALG_OPENCL_KERNELS_VECTOR_HPP
#define VIENNACL_LINALG_OPENCL_KERNELS_VECTOR_HPP

#include "viennacl/tools/tools.hpp"

#include "viennacl/device_specific/database.hpp"
#include "viennacl/device_specific/generate.hpp"

#include "viennacl/scheduler/forwards.h"
#include "viennacl/scheduler/io.hpp"
#include "viennacl/scheduler/preset.hpp"

#include "viennacl/ocl/kernel.hpp"
#include "viennacl/ocl/platform.hpp"
#include "viennacl/ocl/utils.hpp"

/** @file viennacl/linalg/opencl/kernels/vector.hpp
 *  @brief OpenCL kernel file for vector operations */
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
        void generate_plane_rotation(StringType & source, std::string const & numeric_string)
        {
          source.append("__kernel void plane_rotation( \n");
          source.append("          __global "); source.append(numeric_string); source.append(" * vec1, \n");
          source.append("          unsigned int start1, \n");
          source.append("          unsigned int inc1, \n");
          source.append("          unsigned int size1, \n");
          source.append("          __global "); source.append(numeric_string); source.append(" * vec2, \n");
          source.append("          unsigned int start2, \n");
          source.append("          unsigned int inc2, \n");
          source.append("          unsigned int size2, \n");
          source.append("          "); source.append(numeric_string); source.append(" alpha, \n");
          source.append("          "); source.append(numeric_string); source.append(" beta) \n");
          source.append("{ \n");
          source.append("  "); source.append(numeric_string); source.append(" tmp1 = 0; \n");
          source.append("  "); source.append(numeric_string); source.append(" tmp2 = 0; \n");
          source.append(" \n");
          source.append("  for (unsigned int i = get_global_id(0); i < size1; i += get_global_size(0)) \n");
          source.append(" { \n");
          source.append("    tmp1 = vec1[i*inc1+start1]; \n");
          source.append("    tmp2 = vec2[i*inc2+start2]; \n");
          source.append(" \n");
          source.append("    vec1[i*inc1+start1] = alpha * tmp1 + beta * tmp2; \n");
          source.append("    vec2[i*inc2+start2] = alpha * tmp2 - beta * tmp1; \n");
          source.append("  } \n");
          source.append(" \n");
          source.append("} \n");
        }

        template <typename StringType>
        void generate_vector_swap(StringType & source, std::string const & numeric_string)
        {
          source.append("__kernel void swap( \n");
          source.append("          __global "); source.append(numeric_string); source.append(" * vec1, \n");
          source.append("          unsigned int start1, \n");
          source.append("          unsigned int inc1, \n");
          source.append("          unsigned int size1, \n");
          source.append("          __global "); source.append(numeric_string); source.append(" * vec2, \n");
          source.append("          unsigned int start2, \n");
          source.append("          unsigned int inc2, \n");
          source.append("          unsigned int size2 \n");
          source.append("          ) \n");
          source.append("{ \n");
          source.append("  "); source.append(numeric_string); source.append(" tmp; \n");
          source.append("  for (unsigned int i = get_global_id(0); i < size1; i += get_global_size(0)) \n");
          source.append("  { \n");
          source.append("    tmp = vec2[i*inc2+start2]; \n");
          source.append("    vec2[i*inc2+start2] = vec1[i*inc1+start1]; \n");
          source.append("    vec1[i*inc1+start1] = tmp; \n");
          source.append("  } \n");
          source.append("} \n");
        }

        template <typename StringType>
        void generate_assign_cpu(StringType & source, std::string const & numeric_string)
        {
          source.append("__kernel void assign_cpu( \n");
          source.append("          __global "); source.append(numeric_string); source.append(" * vec1, \n");
          source.append("          unsigned int start1, \n");
          source.append("          unsigned int inc1, \n");
          source.append("          unsigned int size1, \n");
          source.append("          unsigned int internal_size1, \n");
          source.append("          "); source.append(numeric_string); source.append(" alpha) \n");
          source.append("{ \n");
          source.append("  for (unsigned int i = get_global_id(0); i < internal_size1; i += get_global_size(0)) \n");
          source.append("    vec1[i*inc1+start1] = (i < size1) ? alpha : 0; \n");
          source.append("} \n");

        }

        template <typename StringType>
        void generate_inner_prod(StringType & source, std::string const & numeric_string, vcl_size_t vector_num)
        {
          std::stringstream ss;
          ss << vector_num;
          std::string vector_num_string = ss.str();

          source.append("__kernel void inner_prod"); source.append(vector_num_string); source.append("( \n");
          source.append("          __global const "); source.append(numeric_string); source.append(" * x, \n");
          source.append("          uint4 params_x, \n");
          for (vcl_size_t i=0; i<vector_num; ++i)
          {
            ss.str("");
            ss << i;
            source.append("          __global const "); source.append(numeric_string); source.append(" * y"); source.append(ss.str()); source.append(", \n");
            source.append("          uint4 params_y"); source.append(ss.str()); source.append(", \n");
          }
          source.append("          __local "); source.append(numeric_string); source.append(" * tmp_buffer, \n");
          source.append("          __global "); source.append(numeric_string); source.append(" * group_buffer) \n");
          source.append("{ \n");
          source.append("  unsigned int entries_per_thread = (params_x.z - 1) / get_global_size(0) + 1; \n");
          source.append("  unsigned int vec_start_index = get_group_id(0) * get_local_size(0) * entries_per_thread; \n");
          source.append("  unsigned int vec_stop_index  = min((unsigned int)((get_group_id(0) + 1) * get_local_size(0) * entries_per_thread), params_x.z); \n");

          // compute partial results within group:
          for (vcl_size_t i=0; i<vector_num; ++i)
          {
            ss.str("");
            ss << i;
            source.append("  "); source.append(numeric_string); source.append(" tmp"); source.append(ss.str()); source.append(" = 0; \n");
          }
          source.append("  for (unsigned int i = vec_start_index + get_local_id(0); i < vec_stop_index; i += get_local_size(0)) { \n");
          source.append("    ");  source.append(numeric_string); source.append(" val_x = x[i*params_x.y + params_x.x]; \n");
          for (vcl_size_t i=0; i<vector_num; ++i)
          {
            ss.str("");
            ss << i;
            source.append("    tmp"); source.append(ss.str()); source.append(" += val_x * y"); source.append(ss.str()); source.append("[i * params_y"); source.append(ss.str()); source.append(".y + params_y"); source.append(ss.str()); source.append(".x]; \n");
          }
          source.append("  } \n");
          for (vcl_size_t i=0; i<vector_num; ++i)
          {
            ss.str("");
            ss << i;
            source.append("  tmp_buffer[get_local_id(0) + "); source.append(ss.str()); source.append(" * get_local_size(0)] = tmp"); source.append(ss.str()); source.append("; \n");
          }

          // now run reduction:
          source.append("  for (unsigned int stride = get_local_size(0)/2; stride > 0; stride /= 2) \n");
          source.append("  { \n");
          source.append("    barrier(CLK_LOCAL_MEM_FENCE); \n");
          source.append("    if (get_local_id(0) < stride) { \n");
          for (vcl_size_t i=0; i<vector_num; ++i)
          {
            ss.str("");
            ss << i;
            source.append("      tmp_buffer[get_local_id(0) + "); source.append(ss.str()); source.append(" * get_local_size(0)] += tmp_buffer[get_local_id(0) + "); source.append(ss.str()); source.append(" * get_local_size(0) + stride]; \n");
          }
          source.append("    } \n");
          source.append("  } \n");
          source.append("  barrier(CLK_LOCAL_MEM_FENCE); \n");

          source.append("  if (get_local_id(0) == 0) { \n");
          for (vcl_size_t i=0; i<vector_num; ++i)
          {
            ss.str("");
            ss << i;
            source.append("    group_buffer[get_group_id(0) + "); source.append(ss.str()); source.append(" * get_num_groups(0)] = tmp_buffer["); source.append(ss.str()); source.append(" * get_local_size(0)]; \n");
          }
          source.append("  } \n");
          source.append("} \n");

        }

        template <typename StringType>
        void generate_norm(StringType & source, std::string const & numeric_string)
        {
          bool is_float_or_double = (numeric_string == "float" || numeric_string == "double");

          source.append(numeric_string); source.append(" impl_norm( \n");
          source.append("          __global const "); source.append(numeric_string); source.append(" * vec, \n");
          source.append("          unsigned int start1, \n");
          source.append("          unsigned int inc1, \n");
          source.append("          unsigned int size1, \n");
          source.append("          unsigned int norm_selector, \n");
          source.append("          __local "); source.append(numeric_string); source.append(" * tmp_buffer) \n");
          source.append("{ \n");
          source.append("  "); source.append(numeric_string); source.append(" tmp = 0; \n");
          source.append("  if (norm_selector == 1) \n"); //norm_1
          source.append("  { \n");
          source.append("    for (unsigned int i = get_local_id(0); i < size1; i += get_local_size(0)) \n");
          if (is_float_or_double)
            source.append("      tmp += fabs(vec[i*inc1 + start1]); \n");
          else
            source.append("      tmp += abs(vec[i*inc1 + start1]); \n");
          source.append("  } \n");
          source.append("  else if (norm_selector == 2) \n"); //norm_2
          source.append("  { \n");
          source.append("    "); source.append(numeric_string); source.append(" vec_entry = 0; \n");
          source.append("    for (unsigned int i = get_local_id(0); i < size1; i += get_local_size(0)) \n");
          source.append("    { \n");
          source.append("      vec_entry = vec[i*inc1 + start1]; \n");
          source.append("      tmp += vec_entry * vec_entry; \n");
          source.append("    } \n");
          source.append("  } \n");
          source.append("  else if (norm_selector == 0) \n"); //norm_inf
          source.append("  { \n");
          source.append("    for (unsigned int i = get_local_id(0); i < size1; i += get_local_size(0)) \n");
          if (is_float_or_double)
            source.append("      tmp = fmax(fabs(vec[i*inc1 + start1]), tmp); \n");
          else
          {
            source.append("      tmp = max(("); source.append(numeric_string); source.append(")abs(vec[i*inc1 + start1]), tmp); \n");
          }
          source.append("  } \n");

          source.append("  tmp_buffer[get_local_id(0)] = tmp; \n");

          source.append("  if (norm_selector > 0) \n"); //norm_1 or norm_2:
          source.append("  { \n");
          source.append("    for (unsigned int stride = get_local_size(0)/2; stride > 0; stride /= 2) \n");
          source.append("    { \n");
          source.append("      barrier(CLK_LOCAL_MEM_FENCE); \n");
          source.append("      if (get_local_id(0) < stride) \n");
          source.append("        tmp_buffer[get_local_id(0)] += tmp_buffer[get_local_id(0)+stride]; \n");
          source.append("    } \n");
          source.append("    return tmp_buffer[0]; \n");
          source.append("  } \n");

          //norm_inf:
          source.append("  for (unsigned int stride = get_local_size(0)/2; stride > 0; stride /= 2) \n");
          source.append("  { \n");
          source.append("    barrier(CLK_LOCAL_MEM_FENCE); \n");
          source.append("    if (get_local_id(0) < stride) \n");
          if (is_float_or_double)
            source.append("      tmp_buffer[get_local_id(0)] = fmax(tmp_buffer[get_local_id(0)], tmp_buffer[get_local_id(0)+stride]); \n");
          else
            source.append("      tmp_buffer[get_local_id(0)] = max(tmp_buffer[get_local_id(0)], tmp_buffer[get_local_id(0)+stride]); \n");
          source.append("  } \n");

          source.append("  return tmp_buffer[0]; \n");
          source.append("}; \n");

          source.append("__kernel void norm( \n");
          source.append("          __global const "); source.append(numeric_string); source.append(" * vec, \n");
          source.append("          unsigned int start1, \n");
          source.append("          unsigned int inc1, \n");
          source.append("          unsigned int size1, \n");
          source.append("          unsigned int norm_selector, \n");
          source.append("          __local "); source.append(numeric_string); source.append(" * tmp_buffer, \n");
          source.append("          __global "); source.append(numeric_string); source.append(" * group_buffer) \n");
          source.append("{ \n");
          source.append("  "); source.append(numeric_string); source.append(" tmp = impl_norm(vec, \n");
          source.append("                        (        get_group_id(0)  * size1) / get_num_groups(0) * inc1 + start1, \n");
          source.append("                        inc1, \n");
          source.append("                        (   (1 + get_group_id(0)) * size1) / get_num_groups(0) \n");
          source.append("                      - (        get_group_id(0)  * size1) / get_num_groups(0), \n");
          source.append("                        norm_selector, \n");
          source.append("                        tmp_buffer); \n");

          source.append("  if (get_local_id(0) == 0) \n");
          source.append("    group_buffer[get_group_id(0)] = tmp; \n");
          source.append("} \n");

        }

        template <typename StringType>
        void generate_inner_prod_sum(StringType & source, std::string const & numeric_string)
        {
          // sums the array 'vec1' and writes to result. Makes use of a single work-group only.
          source.append("__kernel void sum_inner_prod( \n");
          source.append("          __global "); source.append(numeric_string); source.append(" * vec1, \n");
          source.append("          __local "); source.append(numeric_string); source.append(" * tmp_buffer, \n");
          source.append("          __global "); source.append(numeric_string); source.append(" * result, \n");
          source.append("          unsigned int start_result, \n");
          source.append("          unsigned int inc_result) \n");
          source.append("{ \n");
          source.append("  tmp_buffer[get_local_id(0)] = vec1[get_global_id(0)]; \n");

          source.append("  for (unsigned int stride = get_local_size(0)/2; stride > 0; stride /= 2) \n");
          source.append("  { \n");
          source.append("    barrier(CLK_LOCAL_MEM_FENCE); \n");
          source.append("    if (get_local_id(0) < stride) \n");
          source.append("      tmp_buffer[get_local_id(0)] += tmp_buffer[get_local_id(0) + stride]; \n");
          source.append("  } \n");
          source.append("  barrier(CLK_LOCAL_MEM_FENCE); \n");

          source.append("  if (get_local_id(0) == 0) \n");
          source.append("    result[start_result + inc_result * get_group_id(0)] = tmp_buffer[0]; \n");
          source.append("} \n");

        }

        template <typename StringType>
        void generate_sum(StringType & source, std::string const & numeric_string)
        {
          // sums the array 'vec1' and writes to result. Makes use of a single work-group only.
          source.append("__kernel void sum( \n");
          source.append("          __global "); source.append(numeric_string); source.append(" * vec1, \n");
          source.append("          unsigned int start1, \n");
          source.append("          unsigned int inc1, \n");
          source.append("          unsigned int size1, \n");
          source.append("          unsigned int option,  \n"); //0: use fmax, 1: just sum, 2: sum and return sqrt of sum
          source.append("          __local "); source.append(numeric_string); source.append(" * tmp_buffer, \n");
          source.append("          __global "); source.append(numeric_string); source.append(" * result) \n");
          source.append("{ \n");
          source.append("  "); source.append(numeric_string); source.append(" thread_sum = 0; \n");
          source.append("  "); source.append(numeric_string); source.append(" tmp = 0; \n");
          source.append("  for (unsigned int i = get_local_id(0); i<size1; i += get_local_size(0)) \n");
          source.append("  { \n");
          source.append("    if (option > 0) \n");
          source.append("      thread_sum += vec1[i*inc1+start1]; \n");
          source.append("    else \n");
          source.append("    { \n");
          source.append("      tmp = vec1[i*inc1+start1]; \n");
          source.append("      tmp = (tmp < 0) ? -tmp : tmp; \n");
          source.append("      thread_sum = (thread_sum > tmp) ? thread_sum : tmp; \n");
          source.append("    } \n");
          source.append("  } \n");

          source.append("  tmp_buffer[get_local_id(0)] = thread_sum; \n");

          source.append("  for (unsigned int stride = get_local_size(0)/2; stride > 0; stride /= 2) \n");
          source.append("  { \n");
          source.append("    barrier(CLK_LOCAL_MEM_FENCE); \n");
          source.append("    if (get_local_id(0) < stride) \n");
          source.append("    { \n");
          source.append("      if (option > 0) \n");
          source.append("        tmp_buffer[get_local_id(0)] += tmp_buffer[get_local_id(0) + stride]; \n");
          source.append("      else \n");
          source.append("        tmp_buffer[get_local_id(0)] = (tmp_buffer[get_local_id(0)] > tmp_buffer[get_local_id(0) + stride]) ? tmp_buffer[get_local_id(0)] : tmp_buffer[get_local_id(0) + stride]; \n");
          source.append("    } \n");
          source.append("  } \n");
          source.append("  barrier(CLK_LOCAL_MEM_FENCE); \n");

          source.append("  if (get_global_id(0) == 0) \n");
          source.append("  { \n");
          if (numeric_string == "float" || numeric_string == "double")
          {
            source.append("    if (option == 2) \n");
            source.append("      *result = sqrt(tmp_buffer[0]); \n");
            source.append("    else \n");
          }
          source.append("      *result = tmp_buffer[0]; \n");
          source.append("  } \n");
          source.append("} \n");

        }

        template <typename StringType>
        void generate_index_norm_inf(StringType & source, std::string const & numeric_string)
        {
          //index_norm_inf:
          source.append("unsigned int index_norm_inf_impl( \n");
          source.append("          __global const "); source.append(numeric_string); source.append(" * vec, \n");
          source.append("          unsigned int start1, \n");
          source.append("          unsigned int inc1, \n");
          source.append("          unsigned int size1, \n");
          source.append("          __local "); source.append(numeric_string); source.append(" * entry_buffer, \n");
          source.append("          __local unsigned int * index_buffer) \n");
          source.append("{ \n");
          //step 1: fill buffer:
          source.append("  "); source.append(numeric_string); source.append(" cur_max = 0; \n");
          source.append("  "); source.append(numeric_string); source.append(" tmp; \n");
          source.append("  for (unsigned int i = get_global_id(0); i < size1; i += get_global_size(0)) \n");
          source.append("  { \n");
          if (numeric_string == "float" || numeric_string == "double")
            source.append("    tmp = fabs(vec[i*inc1+start1]); \n");
          else
            source.append("    tmp = abs(vec[i*inc1+start1]); \n");
          source.append("    if (cur_max < tmp) \n");
          source.append("    { \n");
          source.append("      entry_buffer[get_global_id(0)] = tmp; \n");
          source.append("      index_buffer[get_global_id(0)] = i; \n");
          source.append("      cur_max = tmp; \n");
          source.append("    } \n");
          source.append("  } \n");

          //step 2: parallel reduction:
          source.append("  for (unsigned int stride = get_global_size(0)/2; stride > 0; stride /= 2) \n");
          source.append("  { \n");
          source.append("    barrier(CLK_LOCAL_MEM_FENCE); \n");
          source.append("    if (get_global_id(0) < stride) \n");
          source.append("   { \n");
          //find the first occurring index
          source.append("      if (entry_buffer[get_global_id(0)] < entry_buffer[get_global_id(0)+stride]) \n");
          source.append("      { \n");
          source.append("        index_buffer[get_global_id(0)] = index_buffer[get_global_id(0)+stride]; \n");
          source.append("        entry_buffer[get_global_id(0)] = entry_buffer[get_global_id(0)+stride]; \n");
          source.append("      } \n");
          source.append("    } \n");
          source.append("  } \n");
          source.append(" \n");
          source.append("  return index_buffer[0]; \n");
          source.append("} \n");

          source.append("__kernel void index_norm_inf( \n");
          source.append("          __global "); source.append(numeric_string); source.append(" * vec, \n");
          source.append("          unsigned int start1, \n");
          source.append("          unsigned int inc1, \n");
          source.append("          unsigned int size1, \n");
          source.append("          __local "); source.append(numeric_string); source.append(" * entry_buffer, \n");
          source.append("          __local unsigned int * index_buffer, \n");
          source.append("          __global unsigned int * result) \n");
          source.append("{ \n");
          source.append("  entry_buffer[get_global_id(0)] = 0; \n");
          source.append("  index_buffer[get_global_id(0)] = 0; \n");
          source.append("  unsigned int tmp = index_norm_inf_impl(vec, start1, inc1, size1, entry_buffer, index_buffer); \n");
          source.append("  if (get_global_id(0) == 0) *result = tmp; \n");
          source.append("} \n");

        }


        inline void generate_avbv(std::string & source, device_specific::template_base & axpy, scheduler::statement_node_numeric_type NUMERIC_TYPE)
        {
          using device_specific::generate::opencl_source;
          using scheduler::HOST_SCALAR_TYPE;
          using scheduler::DEVICE_SCALAR_TYPE;
          using scheduler::INVALID_SUBTYPE;

          //av
          source.append(opencl_source(axpy, scheduler::preset::avbv(NUMERIC_TYPE, HOST_SCALAR_TYPE, false, false, INVALID_SUBTYPE, false, false)));
          source.append(opencl_source(axpy, scheduler::preset::avbv(NUMERIC_TYPE, HOST_SCALAR_TYPE, true, false, INVALID_SUBTYPE, false, false)));
          source.append(opencl_source(axpy, scheduler::preset::avbv(NUMERIC_TYPE, HOST_SCALAR_TYPE, false, true, INVALID_SUBTYPE, false, false)));
          source.append(opencl_source(axpy, scheduler::preset::avbv(NUMERIC_TYPE, HOST_SCALAR_TYPE, true, true, INVALID_SUBTYPE, false, false)));
          source.append(opencl_source(axpy, scheduler::preset::avbv(NUMERIC_TYPE, DEVICE_SCALAR_TYPE, false, false, INVALID_SUBTYPE, false, false)));
          source.append(opencl_source(axpy, scheduler::preset::avbv(NUMERIC_TYPE, DEVICE_SCALAR_TYPE, true, false, INVALID_SUBTYPE, false, false)));
          source.append(opencl_source(axpy, scheduler::preset::avbv(NUMERIC_TYPE, DEVICE_SCALAR_TYPE, false, true, INVALID_SUBTYPE, false, false)));
          source.append(opencl_source(axpy, scheduler::preset::avbv(NUMERIC_TYPE, DEVICE_SCALAR_TYPE, true, true, INVALID_SUBTYPE, false, false)));

          //avbv

          // b = HOST

          // b = no flip, no reciprocal
          source.append(opencl_source(axpy, scheduler::preset::avbv(NUMERIC_TYPE, HOST_SCALAR_TYPE, false, false, HOST_SCALAR_TYPE, false, false)));
          source.append(opencl_source(axpy, scheduler::preset::avbv(NUMERIC_TYPE, HOST_SCALAR_TYPE, true, false, HOST_SCALAR_TYPE, false, false)));
          source.append(opencl_source(axpy, scheduler::preset::avbv(NUMERIC_TYPE, HOST_SCALAR_TYPE, false, true, HOST_SCALAR_TYPE, false, false)));
          source.append(opencl_source(axpy, scheduler::preset::avbv(NUMERIC_TYPE, HOST_SCALAR_TYPE, true, true, HOST_SCALAR_TYPE, false, false)));
          source.append(opencl_source(axpy, scheduler::preset::avbv(NUMERIC_TYPE, DEVICE_SCALAR_TYPE, false, false, HOST_SCALAR_TYPE, false, false)));
          source.append(opencl_source(axpy, scheduler::preset::avbv(NUMERIC_TYPE, DEVICE_SCALAR_TYPE, true, false, HOST_SCALAR_TYPE, false, false)));
          source.append(opencl_source(axpy, scheduler::preset::avbv(NUMERIC_TYPE, DEVICE_SCALAR_TYPE, false, true, HOST_SCALAR_TYPE, false, false)));
          source.append(opencl_source(axpy, scheduler::preset::avbv(NUMERIC_TYPE, DEVICE_SCALAR_TYPE, true, true, HOST_SCALAR_TYPE, false, false)));
          // b = flip, no reciprocal
          source.append(opencl_source(axpy, scheduler::preset::avbv(NUMERIC_TYPE, HOST_SCALAR_TYPE, false, false, HOST_SCALAR_TYPE, true, false)));
          source.append(opencl_source(axpy, scheduler::preset::avbv(NUMERIC_TYPE, HOST_SCALAR_TYPE, true, false, HOST_SCALAR_TYPE, true, false)));
          source.append(opencl_source(axpy, scheduler::preset::avbv(NUMERIC_TYPE, HOST_SCALAR_TYPE, false, true, HOST_SCALAR_TYPE, true, false)));
          source.append(opencl_source(axpy, scheduler::preset::avbv(NUMERIC_TYPE, HOST_SCALAR_TYPE, true, true, HOST_SCALAR_TYPE, true, false)));
          source.append(opencl_source(axpy, scheduler::preset::avbv(NUMERIC_TYPE, DEVICE_SCALAR_TYPE, false, false, HOST_SCALAR_TYPE, true, false)));
          source.append(opencl_source(axpy, scheduler::preset::avbv(NUMERIC_TYPE, DEVICE_SCALAR_TYPE, true, false, HOST_SCALAR_TYPE, true, false)));
          source.append(opencl_source(axpy, scheduler::preset::avbv(NUMERIC_TYPE, DEVICE_SCALAR_TYPE, false, true, HOST_SCALAR_TYPE, true, false)));
          source.append(opencl_source(axpy, scheduler::preset::avbv(NUMERIC_TYPE, DEVICE_SCALAR_TYPE, true, true, HOST_SCALAR_TYPE, true, false)));
          // b = no flip, reciprocal
          source.append(opencl_source(axpy, scheduler::preset::avbv(NUMERIC_TYPE, HOST_SCALAR_TYPE, false, false, HOST_SCALAR_TYPE, false, true)));
          source.append(opencl_source(axpy, scheduler::preset::avbv(NUMERIC_TYPE, HOST_SCALAR_TYPE, true, false, HOST_SCALAR_TYPE, false, true)));
          source.append(opencl_source(axpy, scheduler::preset::avbv(NUMERIC_TYPE, HOST_SCALAR_TYPE, false, true, HOST_SCALAR_TYPE, false, true)));
          source.append(opencl_source(axpy, scheduler::preset::avbv(NUMERIC_TYPE, HOST_SCALAR_TYPE, true, true, HOST_SCALAR_TYPE, false, true)));
          source.append(opencl_source(axpy, scheduler::preset::avbv(NUMERIC_TYPE, DEVICE_SCALAR_TYPE, false, false, HOST_SCALAR_TYPE, false, true)));
          source.append(opencl_source(axpy, scheduler::preset::avbv(NUMERIC_TYPE, DEVICE_SCALAR_TYPE, true, false, HOST_SCALAR_TYPE, false, true)));
          source.append(opencl_source(axpy, scheduler::preset::avbv(NUMERIC_TYPE, DEVICE_SCALAR_TYPE, false, true, HOST_SCALAR_TYPE, false, true)));
          source.append(opencl_source(axpy, scheduler::preset::avbv(NUMERIC_TYPE, DEVICE_SCALAR_TYPE, true, true, HOST_SCALAR_TYPE, false, true)));
          // b = flip, reciprocal
          source.append(opencl_source(axpy, scheduler::preset::avbv(NUMERIC_TYPE, HOST_SCALAR_TYPE, false, false, HOST_SCALAR_TYPE, true, true)));
          source.append(opencl_source(axpy, scheduler::preset::avbv(NUMERIC_TYPE, HOST_SCALAR_TYPE, true, false, HOST_SCALAR_TYPE, true, true)));
          source.append(opencl_source(axpy, scheduler::preset::avbv(NUMERIC_TYPE, HOST_SCALAR_TYPE, false, true, HOST_SCALAR_TYPE, true, true)));
          source.append(opencl_source(axpy, scheduler::preset::avbv(NUMERIC_TYPE, HOST_SCALAR_TYPE, true, true, HOST_SCALAR_TYPE, true, true)));
          source.append(opencl_source(axpy, scheduler::preset::avbv(NUMERIC_TYPE, DEVICE_SCALAR_TYPE, false, false, HOST_SCALAR_TYPE, true, true)));
          source.append(opencl_source(axpy, scheduler::preset::avbv(NUMERIC_TYPE, DEVICE_SCALAR_TYPE, true, false, HOST_SCALAR_TYPE, true, true)));
          source.append(opencl_source(axpy, scheduler::preset::avbv(NUMERIC_TYPE, DEVICE_SCALAR_TYPE, false, true, HOST_SCALAR_TYPE, true, true)));
          source.append(opencl_source(axpy, scheduler::preset::avbv(NUMERIC_TYPE, DEVICE_SCALAR_TYPE, true, true, HOST_SCALAR_TYPE, true, true)));

          // b = DEVICE

          // b = no flip, no reciprocal
          source.append(opencl_source(axpy, scheduler::preset::avbv(NUMERIC_TYPE, HOST_SCALAR_TYPE, false, false, DEVICE_SCALAR_TYPE, false, false)));
          source.append(opencl_source(axpy, scheduler::preset::avbv(NUMERIC_TYPE, HOST_SCALAR_TYPE, true, false, DEVICE_SCALAR_TYPE, false, false)));
          source.append(opencl_source(axpy, scheduler::preset::avbv(NUMERIC_TYPE, HOST_SCALAR_TYPE, false, true, DEVICE_SCALAR_TYPE, false, false)));
          source.append(opencl_source(axpy, scheduler::preset::avbv(NUMERIC_TYPE, HOST_SCALAR_TYPE, true, true, DEVICE_SCALAR_TYPE, false, false)));
          source.append(opencl_source(axpy, scheduler::preset::avbv(NUMERIC_TYPE, DEVICE_SCALAR_TYPE, false, false, DEVICE_SCALAR_TYPE, false, false)));
          source.append(opencl_source(axpy, scheduler::preset::avbv(NUMERIC_TYPE, DEVICE_SCALAR_TYPE, true, false, DEVICE_SCALAR_TYPE, false, false)));
          source.append(opencl_source(axpy, scheduler::preset::avbv(NUMERIC_TYPE, DEVICE_SCALAR_TYPE, false, true, DEVICE_SCALAR_TYPE, false, false)));
          source.append(opencl_source(axpy, scheduler::preset::avbv(NUMERIC_TYPE, DEVICE_SCALAR_TYPE, true, true, DEVICE_SCALAR_TYPE, false, false)));
          // b = flip, no reciprocal
          source.append(opencl_source(axpy, scheduler::preset::avbv(NUMERIC_TYPE, HOST_SCALAR_TYPE, false, false, DEVICE_SCALAR_TYPE, true, false)));
          source.append(opencl_source(axpy, scheduler::preset::avbv(NUMERIC_TYPE, HOST_SCALAR_TYPE, true, false, DEVICE_SCALAR_TYPE, true, false)));
          source.append(opencl_source(axpy, scheduler::preset::avbv(NUMERIC_TYPE, HOST_SCALAR_TYPE, false, true, DEVICE_SCALAR_TYPE, true, false)));
          source.append(opencl_source(axpy, scheduler::preset::avbv(NUMERIC_TYPE, HOST_SCALAR_TYPE, true, true, DEVICE_SCALAR_TYPE, true, false)));
          source.append(opencl_source(axpy, scheduler::preset::avbv(NUMERIC_TYPE, DEVICE_SCALAR_TYPE, false, false, DEVICE_SCALAR_TYPE, true, false)));
          source.append(opencl_source(axpy, scheduler::preset::avbv(NUMERIC_TYPE, DEVICE_SCALAR_TYPE, true, false, DEVICE_SCALAR_TYPE, true, false)));
          source.append(opencl_source(axpy, scheduler::preset::avbv(NUMERIC_TYPE, DEVICE_SCALAR_TYPE, false, true, DEVICE_SCALAR_TYPE, true, false)));
          source.append(opencl_source(axpy, scheduler::preset::avbv(NUMERIC_TYPE, DEVICE_SCALAR_TYPE, true, true, DEVICE_SCALAR_TYPE, true, false)));
          // b = no flip, reciprocal
          source.append(opencl_source(axpy, scheduler::preset::avbv(NUMERIC_TYPE, HOST_SCALAR_TYPE, false, false, DEVICE_SCALAR_TYPE, false, true)));
          source.append(opencl_source(axpy, scheduler::preset::avbv(NUMERIC_TYPE, HOST_SCALAR_TYPE, true, false, DEVICE_SCALAR_TYPE, false, true)));
          source.append(opencl_source(axpy, scheduler::preset::avbv(NUMERIC_TYPE, HOST_SCALAR_TYPE, false, true, DEVICE_SCALAR_TYPE, false, true)));
          source.append(opencl_source(axpy, scheduler::preset::avbv(NUMERIC_TYPE, HOST_SCALAR_TYPE, true, true, DEVICE_SCALAR_TYPE, false, true)));
          source.append(opencl_source(axpy, scheduler::preset::avbv(NUMERIC_TYPE, DEVICE_SCALAR_TYPE, false, false, DEVICE_SCALAR_TYPE, false, true)));
          source.append(opencl_source(axpy, scheduler::preset::avbv(NUMERIC_TYPE, DEVICE_SCALAR_TYPE, true, false, DEVICE_SCALAR_TYPE, false, true)));
          source.append(opencl_source(axpy, scheduler::preset::avbv(NUMERIC_TYPE, DEVICE_SCALAR_TYPE, false, true, DEVICE_SCALAR_TYPE, false, true)));
          source.append(opencl_source(axpy, scheduler::preset::avbv(NUMERIC_TYPE, DEVICE_SCALAR_TYPE, true, true, DEVICE_SCALAR_TYPE, false, true)));
          // b = flip, reciprocal
          source.append(opencl_source(axpy, scheduler::preset::avbv(NUMERIC_TYPE, HOST_SCALAR_TYPE, false, false, DEVICE_SCALAR_TYPE, true, true)));
          source.append(opencl_source(axpy, scheduler::preset::avbv(NUMERIC_TYPE, HOST_SCALAR_TYPE, true, false, DEVICE_SCALAR_TYPE, true, true)));
          source.append(opencl_source(axpy, scheduler::preset::avbv(NUMERIC_TYPE, HOST_SCALAR_TYPE, false, true, DEVICE_SCALAR_TYPE, true, true)));
          source.append(opencl_source(axpy, scheduler::preset::avbv(NUMERIC_TYPE, HOST_SCALAR_TYPE, true, true, DEVICE_SCALAR_TYPE, true, true)));
          source.append(opencl_source(axpy, scheduler::preset::avbv(NUMERIC_TYPE, DEVICE_SCALAR_TYPE, false, false, DEVICE_SCALAR_TYPE, true, true)));
          source.append(opencl_source(axpy, scheduler::preset::avbv(NUMERIC_TYPE, DEVICE_SCALAR_TYPE, true, false, DEVICE_SCALAR_TYPE, true, true)));
          source.append(opencl_source(axpy, scheduler::preset::avbv(NUMERIC_TYPE, DEVICE_SCALAR_TYPE, false, true, DEVICE_SCALAR_TYPE, true, true)));
          source.append(opencl_source(axpy, scheduler::preset::avbv(NUMERIC_TYPE, DEVICE_SCALAR_TYPE, true, true, DEVICE_SCALAR_TYPE, true, true)));
        }

        //////////////////////////// Part 2: Main kernel class ////////////////////////////////////

        // main kernel class
        /** @brief Main kernel class for generating OpenCL kernels for operations on/with viennacl::vector<> without involving matrices, multiple inner products, or element-wise operations other than addition or subtraction. */
        template <class TYPE>
        struct vector
        {
          static std::string program_name()
          {
            return viennacl::ocl::type_to_string<TYPE>::apply() + "_vector";
          }

          static void init(viennacl::ocl::context & ctx)
          {
            viennacl::ocl::DOUBLE_PRECISION_CHECKER<TYPE>::apply(ctx);
            std::string numeric_string = viennacl::ocl::type_to_string<TYPE>::apply();
            scheduler::statement_node_numeric_type NUMERIC_TYPE = scheduler::statement_node_numeric_type(scheduler::result_of::numeric_type_id<TYPE>::value);

            static std::map<cl_context, bool> init_done;
            if (!init_done[ctx.handle().get()])
            {
              std::string source;
              source.reserve(8192);

              viennacl::ocl::append_double_precision_pragma<TYPE>(ctx, source);

              generate_avbv(source, device_specific::database::get(device_specific::database::axpy, NUMERIC_TYPE), NUMERIC_TYPE);

              // kernels with mostly predetermined skeleton:
              generate_plane_rotation(source, numeric_string);
              generate_vector_swap(source, numeric_string);
              generate_assign_cpu(source, numeric_string);

              generate_inner_prod(source, numeric_string, 1);
              generate_norm(source, numeric_string);
              generate_sum(source, numeric_string);
              generate_index_norm_inf(source, numeric_string);

              std::string prog_name = program_name();
              #ifdef VIENNACL_BUILD_INFO
              std::cout << "Creating program " << prog_name << std::endl;
              #endif
              ctx.add_program(source, prog_name);
              init_done[ctx.handle().get()] = true;
            } //if
          } //init
        };

        // class with kernels for multiple inner products.
        /** @brief Main kernel class for generating OpenCL kernels for multiple inner products on/with viennacl::vector<>. */
        template <class TYPE>
        struct vector_multi_inner_prod
        {
          static std::string program_name()
          {
            return viennacl::ocl::type_to_string<TYPE>::apply() + "_vector_multi";
          }

          static void init(viennacl::ocl::context & ctx)
          {
            viennacl::ocl::DOUBLE_PRECISION_CHECKER<TYPE>::apply(ctx);
            std::string numeric_string = viennacl::ocl::type_to_string<TYPE>::apply();

            static std::map<cl_context, bool> init_done;
            if (!init_done[ctx.handle().get()])
            {
              std::string source;
              source.reserve(8192);

              viennacl::ocl::append_double_precision_pragma<TYPE>(ctx, source);

              generate_inner_prod(source, numeric_string, 2);
              generate_inner_prod(source, numeric_string, 3);
              generate_inner_prod(source, numeric_string, 4);
              generate_inner_prod(source, numeric_string, 8);

              generate_inner_prod_sum(source, numeric_string);

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

