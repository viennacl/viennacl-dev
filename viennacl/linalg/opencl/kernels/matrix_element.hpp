#ifndef VIENNACL_LINALG_OPENCL_KERNELS_MATRIX_ELEMENT_HPP
#define VIENNACL_LINALG_OPENCL_KERNELS_MATRIX_ELEMENT_HPP

#include "viennacl/tools/tools.hpp"
#include "viennacl/ocl/kernel.hpp"
#include "viennacl/ocl/platform.hpp"
#include "viennacl/ocl/utils.hpp"
#include "viennacl/linalg/opencl/kernels/matrix.hpp"

#include "viennacl/scheduler/preset.hpp"

#include "viennacl/device_specific/builtin_database/matrix_axpy.hpp"

/** @file viennacl/linalg/opencl/kernels/matrix_element.hpp
 *  @brief OpenCL kernel file for element-wise matrix operations */
namespace viennacl
{
  namespace linalg
  {
    namespace opencl
    {
      namespace kernels
      {


        //////////////////////////// Part 2: Main kernel class ////////////////////////////////////

        // main kernel class
        /** @brief Main kernel class for generating OpenCL kernels for elementwise-operations such as element_sin() on/with dense matrix objects of type viennacl::matrix<>. */
        template <typename NumericT, typename F>
        struct matrix_element
        {
          static std::string program_name()
          {
            return viennacl::ocl::type_to_string<NumericT>::apply() + "_matrix_element_" + detail::type_to_string(F());
          }

          static void init(viennacl::ocl::context & ctx)
          {
            using namespace scheduler;
            using device_specific::tree_parsing::operator_string;


            viennacl::ocl::DOUBLE_PRECISION_CHECKER<NumericT>::apply(ctx);
            std::string numeric_string = viennacl::ocl::type_to_string<NumericT>::apply();

            static std::map<cl_context, bool> init_done;
            if (!init_done[ctx.handle().get()])
            {
              using namespace device_specific;

              std::string source;
              source.reserve(8192);

              viennacl::ocl::append_double_precision_pragma<NumericT>(ctx, source);
              viennacl::ocl::device const & device = ctx.current_device();

              matrix_axpy_template::parameters_type matrix_axpy_params = builtin_database::matrix_axpy_params<NumericT>(device);

              viennacl::matrix<NumericT, F> A;
              viennacl::matrix<NumericT, F> B;
              viennacl::matrix<NumericT, F> C;

#define ADD_UNARY(TYPE) source.append(matrix_axpy_template(matrix_axpy_params,operator_string(TYPE)).generate(scheduler::preset::unary_element_op(&A, &B, TYPE), device))
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
#define ADD_BINARY(TYPE) source.append(matrix_axpy_template(matrix_axpy_params,operator_string(TYPE)).generate(scheduler::preset::binary_element_op(&A, &B, &C, TYPE), device))
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

