#ifndef VIENNACL_LINALG_OPENCL_KERNELS_VECTOR_HPP
#define VIENNACL_LINALG_OPENCL_KERNELS_VECTOR_HPP

#include "viennacl/tools/tools.hpp"

#include "viennacl/vector_proxy.hpp"

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

        template<typename T, typename ScalarType>
        void generate_inner_prod_impl(std::string & source, reduction_template::parameters const & parameters, vcl_size_t vector_num,
                                       viennacl::vector<T> const * x, viennacl::vector<T> const * y, ScalarType const* s,
                                      std::string const & prefix, viennacl::ocl::device const & device)
        {
          statements_container::data_type statements;
          for(unsigned int i = 0 ; i < vector_num ; ++i)
            statements.push_back(scheduler::preset::inner_prod(s, x, y));
          source.append(reduction_template(parameters, prefix).generate(statements_container(statements,statements_container::INDEPENDENT), device));
        }

        template<typename T, typename ScalarType1, typename ScalarType2>
        inline void generate_avbv_impl2(std::string & source, vector_axpy_template::parameters const & parameters, scheduler::operation_node_type ASSIGN_OP,
                                       viennacl::vector_base<T> const * x, viennacl::vector_base<T> const * y, ScalarType1 const * a,
                                       viennacl::vector_base<T> const * z, ScalarType2 const * b,
                                        std::string const & prefix, viennacl::ocl::device const & device)
        {
          source.append(vector_axpy_template(parameters, prefix + "0000").generate(scheduler::preset::avbv(ASSIGN_OP, x, y, a, false, false, z, b, false, false), device));
          source.append(vector_axpy_template(parameters, prefix + "1000").generate(scheduler::preset::avbv(ASSIGN_OP, x, y, a, true, false, z, b, false, false), device));
          source.append(vector_axpy_template(parameters ,prefix + "0100").generate(scheduler::preset::avbv(ASSIGN_OP, x, y, a, false, true, z, b, false, false), device));
          source.append(vector_axpy_template(parameters, prefix + "1100").generate(scheduler::preset::avbv(ASSIGN_OP, x, y, a, true, true, z, b, false, false), device));
          if(b)
          {
            source.append(vector_axpy_template(parameters, prefix + "0010").generate(scheduler::preset::avbv(ASSIGN_OP, x, y, a, false, false, z, b, true, false), device));
            source.append(vector_axpy_template(parameters, prefix + "1010").generate(scheduler::preset::avbv(ASSIGN_OP, x, y, a, true, false, z, b, true, false), device));
            source.append(vector_axpy_template(parameters ,prefix + "0110").generate(scheduler::preset::avbv(ASSIGN_OP, x, y, a, false, true, z, b, true, false), device));
            source.append(vector_axpy_template(parameters, prefix + "1110").generate(scheduler::preset::avbv(ASSIGN_OP, x, y, a, true, true, z, b, true, false), device));

            source.append(vector_axpy_template(parameters, prefix + "0001").generate(scheduler::preset::avbv(ASSIGN_OP, x, y, a, false, false, z, b, false, true), device));
            source.append(vector_axpy_template(parameters, prefix + "1001").generate(scheduler::preset::avbv(ASSIGN_OP, x, y, a, true, false, z, b, false, true), device));
            source.append(vector_axpy_template(parameters ,prefix + "0101").generate(scheduler::preset::avbv(ASSIGN_OP, x, y, a, false, true, z, b, false, true), device));
            source.append(vector_axpy_template(parameters, prefix + "1101").generate(scheduler::preset::avbv(ASSIGN_OP, x, y, a, true, true, z, b, false, true), device));

            source.append(vector_axpy_template(parameters, prefix + "0011").generate(scheduler::preset::avbv(ASSIGN_OP, x, y, a, false, false, z, b, true, true), device));
            source.append(vector_axpy_template(parameters, prefix + "1011").generate(scheduler::preset::avbv(ASSIGN_OP, x, y, a, true, false, z, b, true, true), device));
            source.append(vector_axpy_template(parameters ,prefix + "0111").generate(scheduler::preset::avbv(ASSIGN_OP, x, y, a, false, true, z, b, true, true), device));
            source.append(vector_axpy_template(parameters, prefix + "1111").generate(scheduler::preset::avbv(ASSIGN_OP, x, y, a, true, true, z, b, true, true), device));
          }
        }

        template<typename T, typename ScalarType>
        inline void generate_avbv_impl(std::string & source, vector_axpy_template::parameters const & parameters, scheduler::operation_node_type ASSIGN_OP,
                                       viennacl::vector_base<T> const * x, viennacl::vector_base<T> const * y, ScalarType const * ha, viennacl::scalar<ScalarType> const * da,
                                       viennacl::vector_base<T> const * z, ScalarType const * hb, viennacl::scalar<ScalarType> const * db,
                                       std::string const & prefix, viennacl::ocl::device const & device)
        {
          //x ASSIGN_OP a*y
          generate_avbv_impl2(source, parameters, ASSIGN_OP, x, y, ha, (viennacl::vector<T>*)NULL, (T*)NULL, prefix + "hv_", device);
          generate_avbv_impl2(source, parameters, ASSIGN_OP, x, y, da, (viennacl::vector<T>*)NULL, (T*)NULL, prefix + "dv_", device);

          //x ASSIGN_OP a*y + b*z
          generate_avbv_impl2(source, parameters, ASSIGN_OP, x, y, ha, z, hb, prefix + "hvhv_", device);
          generate_avbv_impl2(source, parameters, ASSIGN_OP, x, y, da, z, hb, prefix + "dvhv_", device);
          generate_avbv_impl2(source, parameters, ASSIGN_OP, x, y, ha, z, db, prefix + "hvdv_", device);
          generate_avbv_impl2(source, parameters, ASSIGN_OP, x, y, da, z, db, prefix + "dvdv_", device);
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
              viennacl::ocl::device const & device = ctx.current_device();
              vector_axpy_template::parameters const & axpy_parameters = database::get<TYPE>(database::vector_axpy, device);
              reduction_template::parameters const & reduction_parameters = database::get<TYPE>(database::reduction, device);

              std::string source;
              source.reserve(8192);

              viennacl::ocl::append_double_precision_pragma<TYPE>(ctx, source);

              viennacl::vector<TYPE> x;
              viennacl::vector<TYPE> y;
              viennacl::scalar_vector<TYPE> scalary(0,0);
              viennacl::vector<TYPE> z;
              viennacl::scalar<TYPE> da;
              viennacl::scalar<TYPE> db;
              TYPE ha;
              TYPE hb;

              generate_avbv_impl(source, axpy_parameters, scheduler::OPERATION_BINARY_ASSIGN_TYPE, &x, &y, &ha, &da, &z, &hb, &db, "assign_", device);
              generate_avbv_impl(source, axpy_parameters, scheduler::OPERATION_BINARY_INPLACE_ADD_TYPE, &x, &y, &ha, &da, &z, &hb, &db, "ip_add_", device);

              source.append(vector_axpy_template(axpy_parameters, "plane_rotation").generate(scheduler::preset::plane_rotation(&x, &y, &ha, &hb), device));
              source.append(vector_axpy_template(axpy_parameters, "swap").generate(scheduler::preset::swap(&x, &y), device));
              source.append(vector_axpy_template(axpy_parameters, "assign_cpu").generate(scheduler::preset::assign_cpu(&x, &scalary), device));

              generate_inner_prod_impl(source, reduction_parameters, 1, &x, &y, &da, "inner_prod", device);

              source.append(reduction_template(reduction_parameters, "norm_1").generate(scheduler::preset::norm_1(&da, &x), device));
              if(is_floating_point<TYPE>::value)
                source.append(reduction_template(reduction_parameters, "norm_2", BIND_TO_HANDLE).generate(scheduler::preset::norm_2(&da, &x), device)); //BIND_TO_HANDLE for optimization (will load x once in the internal inner product)
              source.append(reduction_template(reduction_parameters, "norm_inf").generate(scheduler::preset::norm_inf(&da, &x), device));
              source.append(reduction_template(reduction_parameters, "index_norm_inf").generate(scheduler::preset::index_norm_inf(&da, &x), device));
              source.append(reduction_template(reduction_parameters, "sum").generate(scheduler::preset::sum(&da, &x), device));

              std::string prog_name = program_name();
              #ifdef VIENNACL_BUILD_INFO
              std::cerr << "Creating program " << prog_name << std::endl;
              #endif
              ctx.add_program(source, prog_name);
              #ifdef VIENNACL_BUILD_INFO
              std::cerr << "Done creating program " << prog_name << std::endl;
              #endif
              init_done[ctx.handle().get()] = true;
            } //if
            #ifdef VIENNACL_BUILD_INFO
            else {
            std::cerr << "init done for context " << ctx.handle().get() << std::endl;
            }
            #endif
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
              viennacl::ocl::device const & device = ctx.current_device();

              reduction_template::parameters const & reduction_parameters = database::get<TYPE>(database::reduction, device);

              std::string source;
              source.reserve(8192);

              viennacl::ocl::append_double_precision_pragma<TYPE>(ctx, source);

              //Dummy holders for the statements
              viennacl::vector<TYPE> x;
              viennacl::vector<TYPE> y;
              viennacl::vector<TYPE> res;
              viennacl::vector_range< viennacl::vector_base<TYPE> > da(res, viennacl::range(0,1));

              generate_inner_prod_impl(source, reduction_parameters, 1, &x, &y, &da, "inner_prod_1", device);
              generate_inner_prod_impl(source, reduction_parameters, 2, &x, &y, &da, "inner_prod_2", device);
              generate_inner_prod_impl(source, reduction_parameters, 3, &x, &y, &da, "inner_prod_3", device);
              generate_inner_prod_impl(source, reduction_parameters, 4, &x, &y, &da, "inner_prod_4", device);
              generate_inner_prod_impl(source, reduction_parameters, 8, &x, &y, &da, "inner_prod_8", device);

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

