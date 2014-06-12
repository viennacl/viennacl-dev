#ifndef VIENNACL_LINALG_OPENCL_KERNELS_VECTOR_ELEMENT_HPP
#define VIENNACL_LINALG_OPENCL_KERNELS_VECTOR_ELEMENT_HPP

#include "viennacl/tools/tools.hpp"
#include "viennacl/ocl/kernel.hpp"
#include "viennacl/ocl/platform.hpp"
#include "viennacl/ocl/utils.hpp"

#include "viennacl/device_specific/database.hpp"

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
            viennacl::ocl::DOUBLE_PRECISION_CHECKER<TYPE>::apply(ctx);
            std::string numeric_string = viennacl::ocl::type_to_string<TYPE>::apply();

            static std::map<cl_context, bool> init_done;
            if (!init_done[ctx.handle().get()])
            {
              using namespace scheduler;

              std::string source;
              source.reserve(8192);

              viennacl::ocl::append_double_precision_pragma<TYPE>(ctx, source);

              vector_axpy_template vtemplate = vector_axpy_template(database::get<TYPE>(database::vector_axpy));

              viennacl::vector<TYPE> x;
              viennacl::vector<TYPE> y;
              viennacl::vector<TYPE> z;

              // unary operations
              if (numeric_string == "float" || numeric_string == "double")
              {
                source.append(vtemplate.generate(scheduler::preset::unary_element_op(&x, &y, OPERATION_UNARY_ACOS_TYPE)));
                source.append(vtemplate.generate(scheduler::preset::unary_element_op(&x, &y, OPERATION_UNARY_ASIN_TYPE)));
                source.append(vtemplate.generate(scheduler::preset::unary_element_op(&x, &y, OPERATION_UNARY_ATAN_TYPE)));
                source.append(vtemplate.generate(scheduler::preset::unary_element_op(&x, &y, OPERATION_UNARY_CEIL_TYPE)));
                source.append(vtemplate.generate(scheduler::preset::unary_element_op(&x, &y, OPERATION_UNARY_COS_TYPE)));
                source.append(vtemplate.generate(scheduler::preset::unary_element_op(&x, &y, OPERATION_UNARY_COSH_TYPE)));
                source.append(vtemplate.generate(scheduler::preset::unary_element_op(&x, &y, OPERATION_UNARY_EXP_TYPE)));
                source.append(vtemplate.generate(scheduler::preset::unary_element_op(&x, &y, OPERATION_UNARY_FABS_TYPE)));
                source.append(vtemplate.generate(scheduler::preset::unary_element_op(&x, &y, OPERATION_UNARY_FLOOR_TYPE)));
                source.append(vtemplate.generate(scheduler::preset::unary_element_op(&x, &y, OPERATION_UNARY_LOG_TYPE)));
                source.append(vtemplate.generate(scheduler::preset::unary_element_op(&x, &y, OPERATION_UNARY_LOG10_TYPE)));
                source.append(vtemplate.generate(scheduler::preset::unary_element_op(&x, &y, OPERATION_UNARY_SIN_TYPE)));
                source.append(vtemplate.generate(scheduler::preset::unary_element_op(&x, &y, OPERATION_UNARY_SINH_TYPE)));
                source.append(vtemplate.generate(scheduler::preset::unary_element_op(&x, &y, OPERATION_UNARY_SQRT_TYPE)));
                source.append(vtemplate.generate(scheduler::preset::unary_element_op(&x, &y, OPERATION_UNARY_TAN_TYPE)));
                source.append(vtemplate.generate(scheduler::preset::unary_element_op(&x, &y, OPERATION_UNARY_TANH_TYPE)));
              }
              else
              {
                source.append(vtemplate.generate(scheduler::preset::unary_element_op(&x, &y, OPERATION_UNARY_ABS_TYPE)));
              }

              // binary operations
              source.append(vtemplate.generate(scheduler::preset::binary_element_op(&x, &y, &z, OPERATION_BINARY_ELEMENT_DIV_TYPE)));
              source.append(vtemplate.generate(scheduler::preset::binary_element_op(&x, &y, &z, OPERATION_BINARY_ELEMENT_POW_TYPE)));
              source.append(vtemplate.generate(scheduler::preset::binary_element_op(&x, &y, &z, OPERATION_BINARY_ELEMENT_PROD_TYPE)));

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

