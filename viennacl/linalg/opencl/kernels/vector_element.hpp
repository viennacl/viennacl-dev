#ifndef VIENNACL_LINALG_OPENCL_KERNELS_VECTOR_ELEMENT_HPP
#define VIENNACL_LINALG_OPENCL_KERNELS_VECTOR_ELEMENT_HPP

#include "viennacl/tools/tools.hpp"
#include "viennacl/ocl/kernel.hpp"
#include "viennacl/ocl/platform.hpp"
#include "viennacl/ocl/utils.hpp"

#include "viennacl/scheduler/preset.hpp"

#include "viennacl/device_specific/builtin_database/vector_axpy.hpp"

/** @file viennacl/linalg/opencl/kernels/vector_element.hpp
 *  @brief OpenCL kernel file for element-wise vector operations */
namespace viennacl
{
  namespace linalg
  {
    namespace opencl
    {
      namespace kernels
      {

        // main kernel class
        /** @brief Main kernel class for generating OpenCL kernels for elementwise operations other than addition and subtraction on/with viennacl::vector<>. */
        template <class TYPE>
        struct vector_element
        {
          static std::string program_name()
          {
            return viennacl::ocl::type_to_string<TYPE>::apply() + "_vector_element";
          }

          static void init(viennacl::ocl::context & ctx)
          {
            using namespace device_specific;

            viennacl::ocl::DOUBLE_PRECISION_CHECKER<TYPE>::apply(ctx);
            std::string numeric_string = viennacl::ocl::type_to_string<TYPE>::apply();

            static std::map<cl_context, bool> init_done;
            if (!init_done[ctx.handle().get()])
            {
              using namespace scheduler;
              using device_specific::tree_parsing::operator_string;

              std::string source;
              source.reserve(8192);

              viennacl::ocl::device const & device = ctx.current_device();
              vector_axpy_template::parameters vector_axpy_params = device_specific::builtin_database::vector_axpy_params<TYPE>(device);
              viennacl::ocl::append_double_precision_pragma<TYPE>(ctx, source);

              viennacl::vector<TYPE> x;
              viennacl::vector<TYPE> y;
              viennacl::vector<TYPE> z;

              // unary operations
#define ADD_UNARY(TYPE) source.append(vector_axpy_template(vector_axpy_params,operator_string(TYPE)).generate(scheduler::preset::unary_element_op(&x, &y, TYPE), device))
              if (numeric_string == "float" || numeric_string == "double")
              {
                ADD_UNARY(OPERATION_UNARY_ACOS_TYPE);
                ADD_UNARY(OPERATION_UNARY_ASIN_TYPE);
                ADD_UNARY(OPERATION_UNARY_ATAN_TYPE);
                ADD_UNARY(OPERATION_UNARY_CEIL_TYPE);
                ADD_UNARY(OPERATION_UNARY_COS_TYPE);
                ADD_UNARY(OPERATION_UNARY_COSH_TYPE);
                ADD_UNARY(OPERATION_UNARY_EXP_TYPE);
                ADD_UNARY(OPERATION_UNARY_FABS_TYPE);
                ADD_UNARY(OPERATION_UNARY_FLOOR_TYPE);
                ADD_UNARY(OPERATION_UNARY_LOG_TYPE);
                ADD_UNARY(OPERATION_UNARY_LOG10_TYPE);
                ADD_UNARY(OPERATION_UNARY_SIN_TYPE);
                ADD_UNARY(OPERATION_UNARY_SINH_TYPE);
                ADD_UNARY(OPERATION_UNARY_SQRT_TYPE);
                ADD_UNARY(OPERATION_UNARY_TAN_TYPE);
                ADD_UNARY(OPERATION_UNARY_TANH_TYPE);
              }
              else
              {
                ADD_UNARY(OPERATION_UNARY_ABS_TYPE);
              }
#undef ADD_UNARY

              // binary operations
#define ADD_BINARY(TYPE) source.append(vector_axpy_template(vector_axpy_params,operator_string(TYPE)).generate(scheduler::preset::binary_element_op(&x, &y, &z, TYPE), device))
              ADD_BINARY(OPERATION_BINARY_ELEMENT_DIV_TYPE);
              ADD_BINARY(OPERATION_BINARY_ELEMENT_PROD_TYPE);
              if (numeric_string == "float" || numeric_string == "double")
              {
                ADD_BINARY(OPERATION_BINARY_ELEMENT_POW_TYPE);
              }
#undef ADD_BINARY

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

