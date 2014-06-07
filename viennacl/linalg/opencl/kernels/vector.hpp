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

        using namespace viennacl::device_specific;

        //////////////////////////// Part 1: Kernel generation routines ////////////////////////////////////

        template<class T>
        void generate_plane_rotation(std::string & source, template_base & generator)
        {
          viennacl::vector<T> x; viennacl::vector<T> y;
          T a; T b;
          source.append(generate::opencl_source(generator, scheduler::preset::plane_rotation(&x, &y, &a, &b)));
        }

        template<class T>
        void generate_vector_swap(std::string & source, template_base & generator)
        {
          viennacl::vector<T> x; viennacl::vector<T> y;
          source.append(generate::opencl_source(generator, scheduler::preset::swap(&x, &y)));
        }

        template<class T>
        void generate_assign_cpu(std::string & source, template_base & generator)
        {
          viennacl::vector<T> x; viennacl::scalar_vector<T> y(0,0);
          source.append(generate::opencl_source(generator, scheduler::preset::assign_cpu(&x, &y)));
        }

        template<typename T>
        void generate_inner_prod(std::string & source, template_base & generator, vcl_size_t vector_num)
        {
          viennacl::vector<T> x;
          viennacl::vector<T> y;
          viennacl::scalar<T> s;

          statements_container::data_type statements;
          for(unsigned int i = 0 ; i < vector_num ; ++i)
            statements.push_back(scheduler::preset::inner_prod(&s, &x, &y));

          source.append(generate::opencl_source(generator, statements_container(statements,statements_container::INDEPENDENT),BIND_ALL_UNIQUE));
        }


        template <typename T>
        void generate_norms_sum(std::string & source, template_base & generator)
        {
          viennacl::vector<T> x;
          viennacl::scalar<T> s;

          source.append(generate::opencl_source(generator, scheduler::preset::norm_1(&s, &x)));
          source.append(generate::opencl_source(generator, scheduler::preset::norm_2(&s, &x)));
          source.append(generate::opencl_source(generator, scheduler::preset::norm_inf(&s, &x)));
          source.append(generate::opencl_source(generator, scheduler::preset::index_norm_inf(&s, &x)));
          source.append(generate::opencl_source(generator, scheduler::preset::sum(&s, &x)));
        }

//        template <typename StringType>
//        void generate_norm(StringType & source, std::string const & numeric_string)
//        {
//          bool is_float_or_double = (numeric_string == "float" || numeric_string == "double");

//          source.append(numeric_string); source.append(" impl_norm( \n");
//          source.append("          __global const "); source.append(numeric_string); source.append(" * vec, \n");
//          source.append("          unsigned int start1, \n");
//          source.append("          unsigned int inc1, \n");
//          source.append("          unsigned int size1, \n");
//          source.append("          unsigned int norm_selector, \n");
//          source.append("          __local "); source.append(numeric_string); source.append(" * tmp_buffer) \n");
//          source.append("{ \n");
//          source.append("  "); source.append(numeric_string); source.append(" tmp = 0; \n");
//          source.append("  if (norm_selector == 1) \n"); //norm_1
//          source.append("  { \n");
//          source.append("    for (unsigned int i = get_local_id(0); i < size1; i += get_local_size(0)) \n");
//          if (is_float_or_double)
//            source.append("      tmp += fabs(vec[i*inc1 + start1]); \n");
//          else
//            source.append("      tmp += abs(vec[i*inc1 + start1]); \n");
//          source.append("  } \n");
//          source.append("  else if (norm_selector == 2) \n"); //norm_2
//          source.append("  { \n");
//          source.append("    "); source.append(numeric_string); source.append(" vec_entry = 0; \n");
//          source.append("    for (unsigned int i = get_local_id(0); i < size1; i += get_local_size(0)) \n");
//          source.append("    { \n");
//          source.append("      vec_entry = vec[i*inc1 + start1]; \n");
//          source.append("      tmp += vec_entry * vec_entry; \n");
//          source.append("    } \n");
//          source.append("  } \n");
//          source.append("  else if (norm_selector == 0) \n"); //norm_inf
//          source.append("  { \n");
//          source.append("    for (unsigned int i = get_local_id(0); i < size1; i += get_local_size(0)) \n");
//          if (is_float_or_double)
//            source.append("      tmp = fmax(fabs(vec[i*inc1 + start1]), tmp); \n");
//          else
//          {
//            source.append("      tmp = max(("); source.append(numeric_string); source.append(")abs(vec[i*inc1 + start1]), tmp); \n");
//          }
//          source.append("  } \n");

//          source.append("  tmp_buffer[get_local_id(0)] = tmp; \n");

//          source.append("  if (norm_selector > 0) \n"); //norm_1 or norm_2:
//          source.append("  { \n");
//          source.append("    for (unsigned int stride = get_local_size(0)/2; stride > 0; stride /= 2) \n");
//          source.append("    { \n");
//          source.append("      barrier(CLK_LOCAL_MEM_FENCE); \n");
//          source.append("      if (get_local_id(0) < stride) \n");
//          source.append("        tmp_buffer[get_local_id(0)] += tmp_buffer[get_local_id(0)+stride]; \n");
//          source.append("    } \n");
//          source.append("    return tmp_buffer[0]; \n");
//          source.append("  } \n");

//          //norm_inf:
//          source.append("  for (unsigned int stride = get_local_size(0)/2; stride > 0; stride /= 2) \n");
//          source.append("  { \n");
//          source.append("    barrier(CLK_LOCAL_MEM_FENCE); \n");
//          source.append("    if (get_local_id(0) < stride) \n");
//          if (is_float_or_double)
//            source.append("      tmp_buffer[get_local_id(0)] = fmax(tmp_buffer[get_local_id(0)], tmp_buffer[get_local_id(0)+stride]); \n");
//          else
//            source.append("      tmp_buffer[get_local_id(0)] = max(tmp_buffer[get_local_id(0)], tmp_buffer[get_local_id(0)+stride]); \n");
//          source.append("  } \n");

//          source.append("  return tmp_buffer[0]; \n");
//          source.append("}; \n");

//          source.append("__kernel void norm( \n");
//          source.append("          __global const "); source.append(numeric_string); source.append(" * vec, \n");
//          source.append("          unsigned int start1, \n");
//          source.append("          unsigned int inc1, \n");
//          source.append("          unsigned int size1, \n");
//          source.append("          unsigned int norm_selector, \n");
//          source.append("          __local "); source.append(numeric_string); source.append(" * tmp_buffer, \n");
//          source.append("          __global "); source.append(numeric_string); source.append(" * group_buffer) \n");
//          source.append("{ \n");
//          source.append("  "); source.append(numeric_string); source.append(" tmp = impl_norm(vec, \n");
//          source.append("                        (        get_group_id(0)  * size1) / get_num_groups(0) * inc1 + start1, \n");
//          source.append("                        inc1, \n");
//          source.append("                        (   (1 + get_group_id(0)) * size1) / get_num_groups(0) \n");
//          source.append("                      - (        get_group_id(0)  * size1) / get_num_groups(0), \n");
//          source.append("                        norm_selector, \n");
//          source.append("                        tmp_buffer); \n");

//          source.append("  if (get_local_id(0) == 0) \n");
//          source.append("    group_buffer[get_group_id(0)] = tmp; \n");
//          source.append("} \n");

//        }

//        template <typename StringType>
//        void generate_inner_prod_sum(StringType & source, std::string const & numeric_string)
//        {
//          // sums the array 'vec1' and writes to result. Makes use of a single work-group only.
//          source.append("__kernel void sum_inner_prod( \n");
//          source.append("          __global "); source.append(numeric_string); source.append(" * vec1, \n");
//          source.append("          __local "); source.append(numeric_string); source.append(" * tmp_buffer, \n");
//          source.append("          __global "); source.append(numeric_string); source.append(" * result, \n");
//          source.append("          unsigned int start_result, \n");
//          source.append("          unsigned int inc_result) \n");
//          source.append("{ \n");
//          source.append("  tmp_buffer[get_local_id(0)] = vec1[get_global_id(0)]; \n");

//          source.append("  for (unsigned int stride = get_local_size(0)/2; stride > 0; stride /= 2) \n");
//          source.append("  { \n");
//          source.append("    barrier(CLK_LOCAL_MEM_FENCE); \n");
//          source.append("    if (get_local_id(0) < stride) \n");
//          source.append("      tmp_buffer[get_local_id(0)] += tmp_buffer[get_local_id(0) + stride]; \n");
//          source.append("  } \n");
//          source.append("  barrier(CLK_LOCAL_MEM_FENCE); \n");

//          source.append("  if (get_local_id(0) == 0) \n");
//          source.append("    result[start_result + inc_result * get_group_id(0)] = tmp_buffer[0]; \n");
//          source.append("} \n");

//        }

//        template <typename StringType>
//        void generate_sum(StringType & source, std::string const & numeric_string)
//        {
//          // sums the array 'vec1' and writes to result. Makes use of a single work-group only.
//          source.append("__kernel void sum( \n");
//          source.append("          __global "); source.append(numeric_string); source.append(" * vec1, \n");
//          source.append("          unsigned int start1, \n");
//          source.append("          unsigned int inc1, \n");
//          source.append("          unsigned int size1, \n");
//          source.append("          unsigned int option,  \n"); //0: use fmax, 1: just sum, 2: sum and return sqrt of sum
//          source.append("          __local "); source.append(numeric_string); source.append(" * tmp_buffer, \n");
//          source.append("          __global "); source.append(numeric_string); source.append(" * result) \n");
//          source.append("{ \n");
//          source.append("  "); source.append(numeric_string); source.append(" thread_sum = 0; \n");
//          source.append("  "); source.append(numeric_string); source.append(" tmp = 0; \n");
//          source.append("  for (unsigned int i = get_local_id(0); i<size1; i += get_local_size(0)) \n");
//          source.append("  { \n");
//          source.append("    if (option > 0) \n");
//          source.append("      thread_sum += vec1[i*inc1+start1]; \n");
//          source.append("    else \n");
//          source.append("    { \n");
//          source.append("      tmp = vec1[i*inc1+start1]; \n");
//          source.append("      tmp = (tmp < 0) ? -tmp : tmp; \n");
//          source.append("      thread_sum = (thread_sum > tmp) ? thread_sum : tmp; \n");
//          source.append("    } \n");
//          source.append("  } \n");

//          source.append("  tmp_buffer[get_local_id(0)] = thread_sum; \n");

//          source.append("  for (unsigned int stride = get_local_size(0)/2; stride > 0; stride /= 2) \n");
//          source.append("  { \n");
//          source.append("    barrier(CLK_LOCAL_MEM_FENCE); \n");
//          source.append("    if (get_local_id(0) < stride) \n");
//          source.append("    { \n");
//          source.append("      if (option > 0) \n");
//          source.append("        tmp_buffer[get_local_id(0)] += tmp_buffer[get_local_id(0) + stride]; \n");
//          source.append("      else \n");
//          source.append("        tmp_buffer[get_local_id(0)] = (tmp_buffer[get_local_id(0)] > tmp_buffer[get_local_id(0) + stride]) ? tmp_buffer[get_local_id(0)] : tmp_buffer[get_local_id(0) + stride]; \n");
//          source.append("    } \n");
//          source.append("  } \n");
//          source.append("  barrier(CLK_LOCAL_MEM_FENCE); \n");

//          source.append("  if (get_global_id(0) == 0) \n");
//          source.append("  { \n");
//          if (numeric_string == "float" || numeric_string == "double")
//          {
//            source.append("    if (option == 2) \n");
//            source.append("      *result = sqrt(tmp_buffer[0]); \n");
//            source.append("    else \n");
//          }
//          source.append("      *result = tmp_buffer[0]; \n");
//          source.append("  } \n");
//          source.append("} \n");

//        }

//        template <typename StringType>
//        void generate_index_norm_inf(StringType & source, std::string const & numeric_string)
//        {
//          //index_norm_inf:
//          source.append("unsigned int index_norm_inf_impl( \n");
//          source.append("          __global const "); source.append(numeric_string); source.append(" * vec, \n");
//          source.append("          unsigned int start1, \n");
//          source.append("          unsigned int inc1, \n");
//          source.append("          unsigned int size1, \n");
//          source.append("          __local "); source.append(numeric_string); source.append(" * entry_buffer, \n");
//          source.append("          __local unsigned int * index_buffer) \n");
//          source.append("{ \n");
//          //step 1: fill buffer:
//          source.append("  "); source.append(numeric_string); source.append(" cur_max = 0; \n");
//          source.append("  "); source.append(numeric_string); source.append(" tmp; \n");
//          source.append("  for (unsigned int i = get_global_id(0); i < size1; i += get_global_size(0)) \n");
//          source.append("  { \n");
//          if (numeric_string == "float" || numeric_string == "double")
//            source.append("    tmp = fabs(vec[i*inc1+start1]); \n");
//          else
//            source.append("    tmp = abs(vec[i*inc1+start1]); \n");
//          source.append("    if (cur_max < tmp) \n");
//          source.append("    { \n");
//          source.append("      entry_buffer[get_global_id(0)] = tmp; \n");
//          source.append("      index_buffer[get_global_id(0)] = i; \n");
//          source.append("      cur_max = tmp; \n");
//          source.append("    } \n");
//          source.append("  } \n");

//          //step 2: parallel reduction:
//          source.append("  for (unsigned int stride = get_global_size(0)/2; stride > 0; stride /= 2) \n");
//          source.append("  { \n");
//          source.append("    barrier(CLK_LOCAL_MEM_FENCE); \n");
//          source.append("    if (get_global_id(0) < stride) \n");
//          source.append("   { \n");
//          //find the first occurring index
//          source.append("      if (entry_buffer[get_global_id(0)] < entry_buffer[get_global_id(0)+stride]) \n");
//          source.append("      { \n");
//          source.append("        index_buffer[get_global_id(0)] = index_buffer[get_global_id(0)+stride]; \n");
//          source.append("        entry_buffer[get_global_id(0)] = entry_buffer[get_global_id(0)+stride]; \n");
//          source.append("      } \n");
//          source.append("    } \n");
//          source.append("  } \n");
//          source.append(" \n");
//          source.append("  return index_buffer[0]; \n");
//          source.append("} \n");

//          source.append("__kernel void index_norm_inf( \n");
//          source.append("          __global "); source.append(numeric_string); source.append(" * vec, \n");
//          source.append("          unsigned int start1, \n");
//          source.append("          unsigned int inc1, \n");
//          source.append("          unsigned int size1, \n");
//          source.append("          __local "); source.append(numeric_string); source.append(" * entry_buffer, \n");
//          source.append("          __local unsigned int * index_buffer, \n");
//          source.append("          __global unsigned int * result) \n");
//          source.append("{ \n");
//          source.append("  entry_buffer[get_global_id(0)] = 0; \n");
//          source.append("  index_buffer[get_global_id(0)] = 0; \n");
//          source.append("  unsigned int tmp = index_norm_inf_impl(vec, start1, inc1, size1, entry_buffer, index_buffer); \n");
//          source.append("  if (get_global_id(0) == 0) *result = tmp; \n");
//          source.append("} \n");

//        }

        template<typename T, typename ScalarType1, typename ScalarType2>
        inline void generate_avbv_impl(std::string & source, template_base & generator, scheduler::operation_node_type ASSIGN_OP,
                                       viennacl::vector_base<T> const * x, viennacl::vector_base<T> const * y, ScalarType1 const * a,
                                       viennacl::vector_base<T> const * z, ScalarType2 const * b)
        {
          using generate::opencl_source;

          source.append(opencl_source(generator, scheduler::preset::avbv(ASSIGN_OP, x, y, a, false, false, z, b, false, false), BIND_ALL_UNIQUE));
          source.append(opencl_source(generator, scheduler::preset::avbv(ASSIGN_OP, x, y, a, true, false, z, b, false, false), BIND_ALL_UNIQUE));
          source.append(opencl_source(generator, scheduler::preset::avbv(ASSIGN_OP, x, y, a, false, true, z, b, false, false), BIND_ALL_UNIQUE));
          source.append(opencl_source(generator, scheduler::preset::avbv(ASSIGN_OP, x, y, a, true, true, z, b, false, false), BIND_ALL_UNIQUE));
          if(b)
          {
            source.append(opencl_source(generator, scheduler::preset::avbv(ASSIGN_OP, x, y, a, false, false, z, b, true, false), BIND_ALL_UNIQUE));
            source.append(opencl_source(generator, scheduler::preset::avbv(ASSIGN_OP, x, y, a, true, false, z, b, true, false), BIND_ALL_UNIQUE));
            source.append(opencl_source(generator, scheduler::preset::avbv(ASSIGN_OP, x, y, a, false, true, z, b, true, false), BIND_ALL_UNIQUE));
            source.append(opencl_source(generator, scheduler::preset::avbv(ASSIGN_OP, x, y, a, true, true, z, b, true, false), BIND_ALL_UNIQUE));

            source.append(opencl_source(generator, scheduler::preset::avbv(ASSIGN_OP, x, y, a, false, false, z, b, false, true), BIND_ALL_UNIQUE));
            source.append(opencl_source(generator, scheduler::preset::avbv(ASSIGN_OP, x, y, a, true, false, z, b, false, true), BIND_ALL_UNIQUE));
            source.append(opencl_source(generator, scheduler::preset::avbv(ASSIGN_OP, x, y, a, false, true, z, b, false, true), BIND_ALL_UNIQUE));
            source.append(opencl_source(generator, scheduler::preset::avbv(ASSIGN_OP, x, y, a, true, true, z, b, false, true), BIND_ALL_UNIQUE));

            source.append(opencl_source(generator, scheduler::preset::avbv(ASSIGN_OP, x, y, a, false, false, z, b, true, true), BIND_ALL_UNIQUE));
            source.append(opencl_source(generator, scheduler::preset::avbv(ASSIGN_OP, x, y, a, true, false, z, b, true, true), BIND_ALL_UNIQUE));
            source.append(opencl_source(generator, scheduler::preset::avbv(ASSIGN_OP, x, y, a, false, true, z, b, true, true), BIND_ALL_UNIQUE));
            source.append(opencl_source(generator, scheduler::preset::avbv(ASSIGN_OP, x, y, a, true, true, z, b, true, true), BIND_ALL_UNIQUE));
          }
        }

        template<class T>
        inline void generate_avbv(std::string & source, template_base & generator, scheduler::operation_node_type ASSIGN_TYPE)
        {
          viennacl::vector<T> x;
          viennacl::vector<T> y;
          viennacl::vector<T> z;

          viennacl::scalar<T> da;
          viennacl::scalar<T> db;

          T ha;
          T hb;

          //x = a*y
          generate_avbv_impl(source, generator, ASSIGN_TYPE, &x, &y, &ha, (viennacl::vector<T>*)NULL, (T*)NULL);
          generate_avbv_impl(source, generator, ASSIGN_TYPE, &x, &y, &da, (viennacl::vector<T>*)NULL, (T*)NULL);

          //x = a*y + b*z
          generate_avbv_impl(source, generator, ASSIGN_TYPE, &x, &y, &ha, &z, &hb);
          generate_avbv_impl(source, generator, ASSIGN_TYPE, &x, &y, &da, &z, &hb);
          generate_avbv_impl(source, generator, ASSIGN_TYPE, &x, &y, &ha, &z, &db);
          generate_avbv_impl(source, generator, ASSIGN_TYPE, &x, &y, &da, &z, &db);
        }

        template<class T>
        inline void generate_avbv(std::string & source, template_base & generator)
        {
          generate_avbv<T>(source, generator, scheduler::OPERATION_BINARY_ASSIGN_TYPE);
          generate_avbv<T>(source, generator, scheduler::OPERATION_BINARY_INPLACE_ADD_TYPE);
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
            static std::map<cl_context, bool> init_done;
            if (!init_done[ctx.handle().get()])
            {
              std::string numeric_string = viennacl::ocl::type_to_string<TYPE>::apply();
              template_base & axpy = database::get<TYPE>(database::axpy);
              template_base & reduction = database::get<TYPE>(database::reduction);

              std::string source;
              source.reserve(8192);

              viennacl::ocl::append_double_precision_pragma<TYPE>(ctx, source);

              generate_avbv<TYPE>(source, axpy);
              generate_plane_rotation<TYPE>(source, axpy);
              generate_vector_swap<TYPE>(source, axpy);
              generate_assign_cpu<TYPE>(source, axpy);
              generate_inner_prod<TYPE>(source, reduction, 1);
              generate_norms_sum<TYPE>(source, reduction);
              // kernels with mostly predetermined skeleton:
//              generate_norm(source, numeric_string);
//              generate_sum(source, numeric_string);
//              generate_index_norm_inf(source, numeric_string);

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
            static std::map<cl_context, bool> init_done;
            if (!init_done[ctx.handle().get()])
            {
              template_base & reduction = database::get<TYPE>(database::reduction);

              std::string source;
              source.reserve(8192);

              viennacl::ocl::append_double_precision_pragma<TYPE>(ctx, source);

              generate_inner_prod<TYPE>(source, reduction, 2);
              generate_inner_prod<TYPE>(source, reduction, 3);
              generate_inner_prod<TYPE>(source, reduction, 4);
              generate_inner_prod<TYPE>(source, reduction, 8);

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

