#ifndef VIENNACL_LINALG_OPENCL_KERNELS_VECTOR_HPP
#define VIENNACL_LINALG_OPENCL_KERNELS_VECTOR_HPP

#include "viennacl/tools/tools.hpp"

#include "viennacl/vector_proxy.hpp"

#include "viennacl/scheduler/forwards.h"
#include "viennacl/scheduler/io.hpp"
#include "viennacl/scheduler/preset.hpp"

#include "viennacl/ocl/kernel.hpp"
#include "viennacl/ocl/platform.hpp"
#include "viennacl/ocl/utils.hpp"

#include "viennacl/device_specific/execution_handler.hpp"

#include "viennacl/device_specific/builtin_database/vector_axpy.hpp"
#include "viennacl/device_specific/builtin_database/reduction.hpp"

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

        template<typename T, typename ScalarType>
        void generate_inner_prod_impl(std::string & source, device_specific::reduction_template::parameters_type const & parameters, vcl_size_t vector_num,
                                       viennacl::vector<T> const * x, viennacl::vector<T> const * y, ScalarType const* s,
                                      std::string const & prefix, viennacl::ocl::device const & device)
        {
          using namespace device_specific;
          statements_container::data_type statements;
          for(unsigned int i = 0 ; i < vector_num ; ++i)
            statements.push_back(scheduler::preset::inner_prod(s, x, y));
          source.append(reduction_template(parameters, prefix).generate(statements_container(statements,statements_container::INDEPENDENT), device));
        }

        template<typename T, typename ScalarType1, typename ScalarType2>
        inline void generate_avbv_impl2(std::string & source, device_specific::vector_axpy_template::parameters_type const & parameters, scheduler::operation_node_type ASSIGN_OP,
                                       viennacl::vector_base<T> const * x, viennacl::vector_base<T> const * y, ScalarType1 const * a,
                                       viennacl::vector_base<T> const * z, ScalarType2 const * b,
                                        std::string const & prefix, viennacl::ocl::device const & device)
        {
          using device_specific::vector_axpy_template;

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
        inline void generate_avbv_impl(std::string & source, device_specific::vector_axpy_template::parameters_type const & parameters, scheduler::operation_node_type ASSIGN_OP,
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
              using namespace device_specific;

              viennacl::ocl::device const & device = ctx.current_device();

              vector_axpy_template::parameters_type vector_axpy_params = builtin_database::vector_axpy_params<TYPE>(device);
              reduction_template::parameters_type reduction_params = builtin_database::reduction_params<TYPE>(device);


              std::string source;
              source.reserve(8192);

              viennacl::ocl::append_double_precision_pragma<TYPE>(ctx, source);

              viennacl::vector<TYPE> x;
              viennacl::vector<TYPE> y;
              viennacl::scalar_vector<TYPE> scalary(0,0,viennacl::context(ctx));
              viennacl::vector<TYPE> z;
              viennacl::scalar<TYPE> da;
              viennacl::scalar<TYPE> db;
              TYPE ha;
              TYPE hb;

              generate_avbv_impl(source, vector_axpy_params, scheduler::OPERATION_BINARY_ASSIGN_TYPE, &x, &y, &ha, &da, &z, &hb, &db, "assign_", device);
              generate_avbv_impl(source, vector_axpy_params, scheduler::OPERATION_BINARY_INPLACE_ADD_TYPE, &x, &y, &ha, &da, &z, &hb, &db, "ip_add_", device);

              source.append(vector_axpy_template(vector_axpy_params, "plane_rotation").generate(scheduler::preset::plane_rotation(&x, &y, &ha, &hb), device));
              source.append(vector_axpy_template(vector_axpy_params, "swap").generate(scheduler::preset::swap(&x, &y), device));
              source.append(vector_axpy_template(vector_axpy_params, "assign_cpu").generate(scheduler::preset::assign_cpu(&x, &scalary), device));

              generate_inner_prod_impl(source, reduction_params, 1, &x, &y, &da, "inner_prod", device);

              source.append(reduction_template(reduction_params, "norm_1").generate(scheduler::preset::norm_1(&da, &x), device));
              if(is_floating_point<TYPE>::value)
                source.append(reduction_template(reduction_params, "norm_2", BIND_TO_HANDLE).generate(scheduler::preset::norm_2(&da, &x), device)); //BIND_TO_HANDLE for optimization (will load x once in the internal inner product)
              source.append(reduction_template(reduction_params, "norm_inf").generate(scheduler::preset::norm_inf(&da, &x), device));
              source.append(reduction_template(reduction_params, "index_norm_inf").generate(scheduler::preset::index_norm_inf(&da, &x), device));
              source.append(reduction_template(reduction_params, "sum").generate(scheduler::preset::sum(&da, &x), device));

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
              using namespace device_specific;

              viennacl::ocl::device const & device = ctx.current_device();

              reduction_template::parameters_type reduction_params = builtin_database::reduction_params<TYPE>(device);

              std::string source;
              source.reserve(8192);

              viennacl::ocl::append_double_precision_pragma<TYPE>(ctx, source);

              //Dummy holders for the statements
              viennacl::vector<TYPE> x;
              viennacl::vector<TYPE> y;
              viennacl::vector<TYPE> res;
              viennacl::vector_range< viennacl::vector_base<TYPE> > da(res, viennacl::range(0,1));

              generate_inner_prod_impl(source, reduction_params, 1, &x, &y, &da, "inner_prod_1", device);
              generate_inner_prod_impl(source, reduction_params, 2, &x, &y, &da, "inner_prod_2", device);
              generate_inner_prod_impl(source, reduction_params, 3, &x, &y, &da, "inner_prod_3", device);
              generate_inner_prod_impl(source, reduction_params, 4, &x, &y, &da, "inner_prod_4", device);
              generate_inner_prod_impl(source, reduction_params, 8, &x, &y, &da, "inner_prod_8", device);

              std::string prog_name = program_name();
              #ifdef VIENNACL_BUILD_INFO
              std::cout << "Creating program " << prog_name << std::endl;
              #endif
              ctx.add_program(source, prog_name);
              init_done[ctx.handle().get()] = true;
            } //if
          } //init
        };





        // main kernel class
        /** @brief Main kernel class for generating OpenCL kernels for operations on/with viennacl::vector<> without involving matrices, multiple inner products, or element-wise operations other than addition or subtraction. */
        template <class TYPE>
        class vector_test
        {
        private:
          template<typename T, typename ScalarType>
          static void generate_inner_prod_impl(device_specific::execution_handler & handler, std::string const & prefix, device_specific::reduction_template::parameters_type const & parameters, vcl_size_t vector_num,
                                         viennacl::vector<T> const * x, viennacl::vector<T> const * y, ScalarType const* s)
          {
            using device_specific::reduction_template;
            using device_specific::statements_container;

            statements_container::data_type statements;
            for(unsigned int i = 0 ; i < vector_num ; ++i)
              statements.push_back(scheduler::preset::inner_prod(s, x, y));
            handler.add(prefix, new reduction_template(parameters, prefix), statements_container(statements,statements_container::INDEPENDENT));
          }

          template<typename T, typename ScalarType1, typename ScalarType2>
          static void generate_avbv_impl2(device_specific::execution_handler & handler, std::string const & prefix, device_specific::vector_axpy_template::parameters_type const & parameters, scheduler::operation_node_type ASSIGN_OP,
                                         viennacl::vector_base<T> const * x, viennacl::vector_base<T> const * y, ScalarType1 const * a,
                                         viennacl::vector_base<T> const * z, ScalarType2 const * b)
          {
            using device_specific::vector_axpy_template;

            handler.add(prefix + "0000", new vector_axpy_template(parameters, prefix + "0000"), scheduler::preset::avbv(ASSIGN_OP, x, y, a, false, false, z, b, false, false));
            handler.add(prefix + "0000", new vector_axpy_template(parameters, prefix + "1000"), scheduler::preset::avbv(ASSIGN_OP, x, y, a, true, false, z, b, false, false));
            handler.add(prefix + "0000", new vector_axpy_template(parameters ,prefix + "0100"), scheduler::preset::avbv(ASSIGN_OP, x, y, a, false, true, z, b, false, false));
            handler.add(prefix + "0000", new vector_axpy_template(parameters, prefix + "1100"), scheduler::preset::avbv(ASSIGN_OP, x, y, a, true, true, z, b, false, false));
            if(b)
            {
              handler.add(prefix + "0000", new vector_axpy_template(parameters, prefix + "0010"), scheduler::preset::avbv(ASSIGN_OP, x, y, a, false, false, z, b, true, false));
              handler.add(prefix + "0000", new vector_axpy_template(parameters, prefix + "1010"), scheduler::preset::avbv(ASSIGN_OP, x, y, a, true, false, z, b, true, false));
              handler.add(prefix + "0000", new vector_axpy_template(parameters ,prefix + "0110"), scheduler::preset::avbv(ASSIGN_OP, x, y, a, false, true, z, b, true, false));
              handler.add(prefix + "0000", new vector_axpy_template(parameters, prefix + "1110"), scheduler::preset::avbv(ASSIGN_OP, x, y, a, true, true, z, b, true, false));

              handler.add(prefix + "0000", new vector_axpy_template(parameters, prefix + "0001"), scheduler::preset::avbv(ASSIGN_OP, x, y, a, false, false, z, b, false, true));
              handler.add(prefix + "0000", new vector_axpy_template(parameters, prefix + "1001"), scheduler::preset::avbv(ASSIGN_OP, x, y, a, true, false, z, b, false, true));
              handler.add(prefix + "0000", new vector_axpy_template(parameters ,prefix + "0101"), scheduler::preset::avbv(ASSIGN_OP, x, y, a, false, true, z, b, false, true));
              handler.add(prefix + "0000", new vector_axpy_template(parameters, prefix + "1101"), scheduler::preset::avbv(ASSIGN_OP, x, y, a, true, true, z, b, false, true));

              handler.add(prefix + "0000", new vector_axpy_template(parameters, prefix + "0011"), scheduler::preset::avbv(ASSIGN_OP, x, y, a, false, false, z, b, true, true));
              handler.add(prefix + "0000", new vector_axpy_template(parameters, prefix + "1011"), scheduler::preset::avbv(ASSIGN_OP, x, y, a, true, false, z, b, true, true));
              handler.add(prefix + "0000", new vector_axpy_template(parameters ,prefix + "0111"), scheduler::preset::avbv(ASSIGN_OP, x, y, a, false, true, z, b, true, true));
              handler.add(prefix + "0000", new vector_axpy_template(parameters, prefix + "1111"), scheduler::preset::avbv(ASSIGN_OP, x, y, a, true, true, z, b, true, true));
            }
          }

          template<typename T, typename ScalarType>
          static void generate_avbv_impl(device_specific::execution_handler & handler, std::string const & prefix, device_specific::vector_axpy_template::parameters_type const & parameters, scheduler::operation_node_type ASSIGN_OP,
                                         viennacl::vector_base<T> const * x, viennacl::vector_base<T> const * y, ScalarType const * ha, viennacl::scalar<ScalarType> const * da,
                                         viennacl::vector_base<T> const * z, ScalarType const * hb, viennacl::scalar<ScalarType> const * db)
          {
            //x ASSIGN_OP a*y
            generate_avbv_impl2(handler, prefix + "hv_", parameters, ASSIGN_OP, x, y, ha, (viennacl::vector<T>*)NULL, (T*)NULL);
            generate_avbv_impl2(handler, prefix + "dv_", parameters, ASSIGN_OP, x, y, da, (viennacl::vector<T>*)NULL, (T*)NULL);

            //x ASSIGN_OP a*y + b*z
            generate_avbv_impl2(handler, prefix + "hvhv_", parameters, ASSIGN_OP, x, y, ha, z, hb);
            generate_avbv_impl2(handler, prefix + "dvhv_", parameters, ASSIGN_OP, x, y, da, z, hb);
            generate_avbv_impl2(handler, prefix + "hvdv_", parameters, ASSIGN_OP, x, y, ha, z, db);
            generate_avbv_impl2(handler, prefix + "dvdv_", parameters, ASSIGN_OP, x, y, da, z, db);
          }

        public:
          static device_specific::execution_handler & execution_handler(viennacl::ocl::context & ctx)
          {
            static std::map<cl_context, device_specific::execution_handler> handlers_map;
            cl_context h = ctx.handle().get();
            if(handlers_map.find(h) == handlers_map.end())
            {
              namespace ds = viennacl::device_specific;
              viennacl::ocl::device const & device = ctx.current_device();
              handlers_map.insert(std::make_pair(h, ds::execution_handler(viennacl::ocl::type_to_string<TYPE>::apply() + "_vector", ctx, device)));
              ds::execution_handler & handler = handlers_map.at(h);

              viennacl::vector<TYPE> x;
              viennacl::vector<TYPE> y;
              viennacl::scalar_vector<TYPE> scalary(0,0,viennacl::context(ctx));
              viennacl::vector<TYPE> z;
              viennacl::scalar<TYPE> da;
              viennacl::scalar<TYPE> db;
              TYPE ha;
              TYPE hb;

              ds::vector_axpy_template::parameters_type vector_axpy_params = ds::builtin_database::vector_axpy_params<TYPE>(device);
              ds::reduction_template::parameters_type reduction_params = ds::builtin_database::reduction_params<TYPE>(device);

              generate_avbv_impl(handler, "assign_", vector_axpy_params, scheduler::OPERATION_BINARY_ASSIGN_TYPE, &x, &y, &ha, &da, &z, &hb, &db);
              generate_avbv_impl(handler, "ip_add_", vector_axpy_params, scheduler::OPERATION_BINARY_INPLACE_ADD_TYPE, &x, &y, &ha, &da, &z, &hb, &db);

              handler.add("plane_rotation", new ds::vector_axpy_template(vector_axpy_params, "plane_rotation"), scheduler::preset::plane_rotation(&x, &y, &ha, &hb));
              handler.add("swap", new ds::vector_axpy_template(vector_axpy_params, "swap"), scheduler::preset::swap(&x, &y));
              handler.add("assign_cpu", new ds::vector_axpy_template(vector_axpy_params, "assign_cpu"), scheduler::preset::assign_cpu(&x, &scalary));

              generate_inner_prod_impl(handler, "inner_prod", reduction_params, 1, &x, &y, &da);

              handler.add("norm_1", new ds::reduction_template(reduction_params, "norm_1"), scheduler::preset::norm_1(&da, &x));
              if(is_floating_point<TYPE>::value)
                //BIND_TO_HANDLE for optimization (will load x once in the internal inner product)
                handler.add("norm_2", new ds::reduction_template(reduction_params, "norm_2", ds::BIND_TO_HANDLE), scheduler::preset::norm_2(&da, &x));
              handler.add("norm_inf", new ds::reduction_template(reduction_params, "norm_inf"), scheduler::preset::norm_inf(&da, &x));
              handler.add("index_norm_inf", new ds::reduction_template(reduction_params, "index_norm_inf"), scheduler::preset::index_norm_inf(&da, &x));
              handler.add("sum", new ds::reduction_template(reduction_params, "sum"), scheduler::preset::sum(&da, &x));
            }
            return handlers_map.at(h);
          }
        };

      }  // namespace kernels
    }  // namespace opencl
  }  // namespace linalg
}  // namespace viennacl
#endif

