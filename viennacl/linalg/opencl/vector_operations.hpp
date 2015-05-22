#ifndef VIENNACL_LINALG_OPENCL_VECTOR_OPERATIONS_HPP_
#define VIENNACL_LINALG_OPENCL_VECTOR_OPERATIONS_HPP_

/* =========================================================================
   Copyright (c) 2010-2014, Institute for Microelectronics,
                            Institute for Analysis and Scientific Computing,
                            TU Wien.
   Portions of this software are copyright by UChicago Argonne, LLC.

                            -----------------
                  ViennaCL - The Vienna Computing Library
                            -----------------

   Project Head:    Karl Rupp                   rupp@iue.tuwien.ac.at

   (A list of authors and contributors can be found in the PDF manual)

   License:         MIT (X11), see file LICENSE in the base directory
============================================================================= */

/** @file viennacl/linalg/opencl/vector_operations.hpp
    @brief Implementations of vector operations using OpenCL
*/

#include <cmath>

#include "viennacl/forwards.h"
#include "viennacl/detail/vector_def.hpp"
#include "viennacl/ocl/device.hpp"
#include "viennacl/ocl/handle.hpp"
#include "viennacl/ocl/kernel.hpp"
#include "viennacl/scalar.hpp"
#include "viennacl/tools/tools.hpp"
#include "viennacl/linalg/opencl/common.hpp"
#include "viennacl/linalg/opencl/kernels/vector.hpp"
#include "viennacl/linalg/opencl/kernels/scan.hpp"
#include "viennacl/meta/predicate.hpp"
#include "viennacl/meta/enable_if.hpp"
#include "viennacl/scheduler/preset.hpp"
#include "viennacl/traits/size.hpp"
#include "viennacl/traits/start.hpp"
#include "viennacl/traits/handle.hpp"
#include "viennacl/traits/stride.hpp"

namespace viennacl
{
namespace linalg
{
namespace opencl
{
//
// Introductory note: By convention, all dimensions are already checked in the dispatcher frontend. No need to double-check again in here!
//

template<typename NumericT, typename ScalarT1>
void av(vector_base<NumericT> & x,
        vector_base<NumericT> const & y, ScalarT1 const & alpha, vcl_size_t /* len_alpha */, bool reciprocal_alpha, bool flip_sign_alpha)
{
  assert(viennacl::traits::opencl_handle(x).context() == viennacl::traits::opencl_handle(y).context() && bool("Vectors do not reside in the same OpenCL context. Automatic migration not yet supported!"));
  std::string kernel_name("assign_*v_**00");
  bool is_scalar_cpu = is_cpu_scalar<ScalarT1>::value;
  kernel_name[7]  =    is_scalar_cpu ? 'h' : 'd';
  kernel_name[10] =  flip_sign_alpha ? '1' : '0';
  kernel_name[11] = reciprocal_alpha ? '1' : '0';

  scheduler::statement statement = scheduler::preset::av(scheduler::OPERATION_BINARY_ASSIGN_TYPE, &x, &y, &alpha, flip_sign_alpha, reciprocal_alpha);
  kernels::vector<NumericT>::execution_handler(viennacl::traits::opencl_context(x)).execute(kernel_name, statement);
}


template<typename NumericT, typename ScalarT1, typename ScalarT2>
void avbv(vector_base<NumericT> & x,
          vector_base<NumericT> const & y, ScalarT1 const & alpha, vcl_size_t /* len_alpha */, bool reciprocal_alpha, bool flip_sign_alpha,
          vector_base<NumericT> const & z, ScalarT2 const & beta,  vcl_size_t /* len_beta */,  bool reciprocal_beta,  bool flip_sign_beta)
{
  assert(viennacl::traits::opencl_handle(x).context() == viennacl::traits::opencl_handle(y).context() && bool("Vectors do not reside in the same OpenCL context. Automatic migration not yet supported!"));
  assert(viennacl::traits::opencl_handle(y).context() == viennacl::traits::opencl_handle(z).context() && bool("Vectors do not reside in the same OpenCL context. Automatic migration not yet supported!"));

  std::string kernel_name("assign_*v*v_****");
  bool is_scalar_cpu1 = is_cpu_scalar<ScalarT1>::value;
  bool is_scalar_cpu2 = is_cpu_scalar<ScalarT2>::value;
  kernel_name[7]  = is_scalar_cpu1   ? 'h' : 'd';
  kernel_name[9]  = is_scalar_cpu2   ? 'h' : 'd';
  kernel_name[12] = flip_sign_alpha  ? '1' : '0';
  kernel_name[13] = reciprocal_alpha ? '1' : '0';
  kernel_name[14] = flip_sign_beta   ? '1' : '0';
  kernel_name[15] = reciprocal_beta  ? '1' : '0';

  scheduler::statement statement = scheduler::preset::avbv(scheduler::OPERATION_BINARY_ASSIGN_TYPE, &x, &y, &alpha, flip_sign_alpha, reciprocal_alpha, &z, &beta, flip_sign_beta, reciprocal_beta);
  kernels::vector<NumericT>::execution_handler(viennacl::traits::opencl_context(x)).execute(kernel_name, statement);
}


template<typename NumericT, typename ScalarT1, typename ScalarT2>
void avbv_v(vector_base<NumericT> & x,
            vector_base<NumericT> const & y, ScalarT1 const & alpha, vcl_size_t /* len_alpha */, bool reciprocal_alpha, bool flip_sign_alpha,
            vector_base<NumericT> const & z, ScalarT2 const & beta,  vcl_size_t /* len_beta */,  bool reciprocal_beta,  bool flip_sign_beta)
{
  assert(viennacl::traits::opencl_handle(x).context() == viennacl::traits::opencl_handle(y).context() && bool("Vectors do not reside in the same OpenCL context. Automatic migration not yet supported!"));
  assert(viennacl::traits::opencl_handle(y).context() == viennacl::traits::opencl_handle(z).context() && bool("Vectors do not reside in the same OpenCL context. Automatic migration not yet supported!"));

  std::string kernel_name("ip_add_*v*v_****");
  bool is_scalar_cpu1 = is_cpu_scalar<ScalarT1>::value;
  bool is_scalar_cpu2 = is_cpu_scalar<ScalarT2>::value;
  kernel_name[7]  = is_scalar_cpu1    ? 'h' : 'd';
  kernel_name[9]  = is_scalar_cpu2    ? 'h' : 'd';
  kernel_name[12] = flip_sign_alpha  ? '1' : '0';
  kernel_name[13] = reciprocal_alpha ? '1' : '0';
  kernel_name[14] = flip_sign_beta   ? '1' : '0';
  kernel_name[15] = reciprocal_beta  ? '1' : '0';

  scheduler::statement statement = scheduler::preset::avbv(scheduler::OPERATION_BINARY_INPLACE_ADD_TYPE, &x, &y, &alpha, flip_sign_alpha, reciprocal_alpha, &z, &beta, flip_sign_beta, reciprocal_beta);
  kernels::vector<NumericT>::execution_handler(viennacl::traits::opencl_context(x)).execute(kernel_name, statement);
}


/** @brief Assign a constant value to a vector (-range/-slice)
*
* @param x   The vector to which the value should be assigned
* @param alpha  The value to be assigned
* @param up_to_internal_size  Specifies whether alpha should also be written to padded memory (mostly used for clearing the whole buffer).
*/
template<typename NumericT>
void vector_assign(vector_base<NumericT> & x, const NumericT & alpha, bool up_to_internal_size = false)
{
  scalar_vector<NumericT> y(viennacl::traits::size(x),alpha,viennacl::traits::context(x));
  scheduler::statement statement = scheduler::preset::assign_cpu(&x, &y);

  dynamic_cast<device_specific::vector_axpy_template*>(kernels::vector<NumericT>::execution_handler(viennacl::traits::opencl_context(x)).template_of("assign_cpu"))->up_to_internal_size(up_to_internal_size);
  kernels::vector<NumericT>::execution_handler(viennacl::traits::opencl_context(x)).execute("assign_cpu", statement);
}


/** @brief Swaps the contents of two vectors, data is copied
*
* @param x   The first vector (or -range, or -slice)
* @param y   The second vector (or -range, or -slice)
*/
template<typename NumericT>
void vector_swap(vector_base<NumericT> & x, vector_base<NumericT> & y)
{
  assert(viennacl::traits::opencl_handle(x).context() == viennacl::traits::opencl_handle(y).context() && bool("Vectors do not reside in the same OpenCL context. Automatic migration not yet supported!"));
  device_specific::statements_container statement = scheduler::preset::swap(&x, &y);
  kernels::vector<NumericT>::execution_handler(viennacl::traits::opencl_context(x)).execute("swap", statement);
}

///////////////////////// Binary Elementwise operations /////////////

/** @brief Implementation of the element-wise operation v1 = v2 .* v3 and v1 = v2 ./ v3    (using MATLAB syntax)
*
* @param x   The result vector (or -range, or -slice)
* @param proxy  The proxy object holding v2, v3 and the operation
*/
template<typename NumericT, typename OP>
void element_op(vector_base<NumericT> & x,
                vector_expression<const vector_base<NumericT>, const vector_base<NumericT>, op_element_binary<OP> > const & proxy)
{
  assert(viennacl::traits::opencl_handle(x).context() == viennacl::traits::opencl_handle(proxy.lhs()).context() && bool("Vectors do not reside in the same OpenCL context. Automatic migration not yet supported!"));
  assert(viennacl::traits::opencl_handle(x).context() == viennacl::traits::opencl_handle(proxy.rhs()).context() && bool("Vectors do not reside in the same OpenCL context. Automatic migration not yet supported!"));

  scheduler::operation_node_type TYPE = scheduler::operation_node_type(scheduler::result_of::op_type_info<op_element_binary<OP> >::id);
  scheduler::statement statement = scheduler::preset::binary_element_op(&x, &proxy.lhs(), &proxy.rhs(),TYPE);
  kernels::vector_element<NumericT>::execution_handler(viennacl::traits::opencl_context(x)).execute(device_specific::tree_parsing::operator_string(TYPE), statement);
}

///////////////////////// Unary Elementwise operations /////////////

/** @brief Implementation of unary element-wise operations v1 = OP(v2)
*
* @param x   The result vector (or -range, or -slice)
* @param proxy  The proxy object holding v2 and the operation
*/
template<typename NumericT, typename OP>
void element_op(vector_base<NumericT> & x,
                vector_expression<const vector_base<NumericT>, const vector_base<NumericT>, op_element_unary<OP> > const & proxy)
{
  assert(viennacl::traits::opencl_handle(x).context() == viennacl::traits::opencl_handle(proxy.lhs()).context() && bool("Vectors do not reside in the same OpenCL context. Automatic migration not yet supported!"));
  assert(viennacl::traits::opencl_handle(x).context() == viennacl::traits::opencl_handle(proxy.rhs()).context() && bool("Vectors do not reside in the same OpenCL context. Automatic migration not yet supported!"));

  scheduler::operation_node_type TYPE = scheduler::operation_node_type(scheduler::result_of::op_type_info<op_element_unary<OP> >::id);
  scheduler::statement statement = scheduler::preset::unary_element_op(&x, &proxy.lhs(),TYPE);
  kernels::vector_element<NumericT>::execution_handler(viennacl::traits::opencl_context(x)).execute(device_specific::tree_parsing::operator_string(TYPE), statement);

}

///////////////////////// Norms and inner product ///////////////////

/** @brief Computes the inner product of two vectors - implementation. Library users should call inner_prod(x, y).
*
* @param x The first vector
* @param y The second vector
* @param result The result scalar (on the gpu)
*/
template<typename NumericT>
void inner_prod_impl(vector_base<NumericT> const & x,
                     vector_base<NumericT> const & y,
                     scalar<NumericT> & result)
{
  assert(viennacl::traits::opencl_handle(x).context() == viennacl::traits::opencl_handle(y).context() && bool("Vectors do not reside in the same OpenCL context. Automatic migration not yet supported!"));
  assert(viennacl::traits::opencl_handle(x).context() == viennacl::traits::opencl_handle(result).context() && bool("Operands do not reside in the same OpenCL context. Automatic migration not yet supported!"));

  scheduler::statement statement = scheduler::preset::inner_prod(&result, &x, &y);
  kernels::vector<NumericT>::execution_handler(viennacl::traits::opencl_context(x)).execute("inner_prod", statement);
}

namespace detail
{
  template<typename NumericT>
  viennacl::ocl::packed_cl_uint make_layout(vector_base<NumericT> const & vec)
  {
    viennacl::ocl::packed_cl_uint ret;
    ret.start           = cl_uint(viennacl::traits::start(vec));
    ret.stride          = cl_uint(viennacl::traits::stride(vec));
    ret.size            = cl_uint(viennacl::traits::size(vec));
    ret.internal_size   = cl_uint(viennacl::traits::internal_size(vec));
    return ret;
  }
}

/** @brief Computes multiple inner products where one argument is common to all inner products. <x, y1>, <x, y2>, ..., <x, yN>
*
* @param x          The common vector
* @param vec_tuple  The tuple of vectors y1, y2, ..., yN
* @param result     The result vector
*/
template<typename NumericT>
void inner_prod_impl(vector_base<NumericT> const & x,
                     vector_tuple<NumericT> const & vec_tuple,
                     vector_base<NumericT> & result)
{
  assert(viennacl::traits::opencl_handle(x).context() == viennacl::traits::opencl_handle(result).context() && bool("Operands do not reside in the same OpenCL context. Automatic migration not yet supported!"));
  typedef viennacl::vector_range< viennacl::vector_base<NumericT> > range_t;

  vcl_size_t current_index = 0;
  while (current_index < vec_tuple.const_size())
  {
    device_specific::statements_container::data_type statements;

    vcl_size_t diff = vec_tuple.const_size() - current_index;
    vcl_size_t upper_bound;
    std::string kernel_prefix;
    if (diff>=8) upper_bound = 8, kernel_prefix = "inner_prod_8";
    else if (diff>=4) upper_bound = 4, kernel_prefix = "inner_prod_4";
    else if (diff>=3) upper_bound = 3, kernel_prefix = "inner_prod_3";
    else if (diff>=2) upper_bound = 2, kernel_prefix = "inner_prod_2";
    else upper_bound = 1, kernel_prefix = "inner_prod_1";

    std::vector<range_t> ranges;
    ranges.reserve(upper_bound);
    for (unsigned int i = 0; i < upper_bound; ++i)
      ranges.push_back(range_t(result, viennacl::range(current_index+i, current_index+i+1)));

    for (unsigned int i = 0; i < upper_bound; ++i)
      statements.push_back(scheduler::preset::inner_prod(&ranges[i], &x, &vec_tuple.const_at(current_index+i)));

    kernels::vector_multi_inner_prod<NumericT>::execution_handler(viennacl::traits::opencl_context(x)).execute(kernel_prefix, device_specific::statements_container(statements, device_specific::statements_container::INDEPENDENT));
    current_index += upper_bound;
  }
}


template<typename NumericT>
void inner_prod_cpu(vector_base<NumericT> const & x,
                    vector_base<NumericT> const & y,
                    NumericT & result)
{
  viennacl::scalar<NumericT> tmp(0, viennacl::traits::context(x));
  inner_prod_impl(x, y, tmp);
  result = tmp;
}


//////////// Norm 1

/** @brief Computes the l^1-norm of a vector
*
* @param x      The vector
* @param result The result scalar
*/
template<typename NumericT>
void norm_1_impl(vector_base<NumericT> const & x,
                 scalar<NumericT> & result)
{
  assert(viennacl::traits::opencl_handle(x).context() == viennacl::traits::opencl_handle(result).context() && bool("Operands do not reside in the same OpenCL context. Automatic migration not yet supported!"));

  scheduler::statement statement = scheduler::preset::norm_1(&result, &x);
  kernels::vector<NumericT>::execution_handler(viennacl::traits::opencl_context(x)).execute("norm_1", statement);
}

/** @brief Computes the l^1-norm of a vector with final reduction on CPU
*
* @param x      The vector
* @param result The result scalar
*/
template<typename NumericT>
void norm_1_cpu(vector_base<NumericT> const & x,
                NumericT & result)
{
  viennacl::scalar<NumericT> tmp(0, viennacl::traits::context(x));
  norm_1_impl(x, tmp);
  result = tmp;
}



//////// Norm 2


/** @brief Computes the l^2-norm of a vector - implementation using OpenCL summation at second step
*
* @param x      The vector
* @param result The result scalar
*/
template<typename NumericT>
void norm_2_impl(vector_base<NumericT> const & x,
                 scalar<NumericT> & result)
{
  assert(viennacl::traits::opencl_handle(x).context() == viennacl::traits::opencl_handle(result).context() && bool("Operands do not reside in the same OpenCL context. Automatic migration not yet supported!"));

  scheduler::statement statement = scheduler::preset::norm_2(&result, &x);
  kernels::vector<NumericT>::execution_handler(viennacl::traits::opencl_context(x)).execute("norm_2", statement);
}

/** @brief Computes the l^1-norm of a vector with final reduction on CPU
*
* @param x      The vector
* @param result The result scalar
*/
template<typename NumericT>
void norm_2_cpu(vector_base<NumericT> const & x,
                NumericT & result)
{
  scalar<NumericT> tmp(0, viennacl::traits::context(x));
  norm_2_impl(x, tmp);
  result = tmp;
}



////////// Norm inf

/** @brief Computes the supremum-norm of a vector
*
* @param x      The vector
* @param result The result scalar
*/
template<typename NumericT>
void norm_inf_impl(vector_base<NumericT> const & x,
                   scalar<NumericT> & result)
{
  assert(viennacl::traits::opencl_handle(x).context() == viennacl::traits::opencl_handle(result).context() && bool("Operands do not reside in the same OpenCL context. Automatic migration not yet supported!"));

  scheduler::statement statement = scheduler::preset::norm_inf(&result, &x);
  kernels::vector<NumericT>::execution_handler(viennacl::traits::opencl_context(x)).execute("norm_inf", statement);
}

/** @brief Computes the supremum-norm of a vector
*
* @param x      The vector
* @param result The result scalar
*/
template<typename NumericT>
void norm_inf_cpu(vector_base<NumericT> const & x,
                  NumericT & result)
{
  scalar<NumericT> tmp(0, viennacl::traits::context(x));
  norm_inf_impl(x, tmp);
  result = tmp;
}


/////////// index norm_inf

//This function should return a CPU scalar, otherwise statements like
// vcl_rhs[index_norm_inf(vcl_rhs)]
// are ambiguous
/** @brief Computes the index of the first entry that is equal to the supremum-norm in modulus.
*
* @param x The vector
* @return The result. Note that the result must be a CPU scalar (unsigned int), since gpu scalars are floating point types.
*/
template<typename NumericT>
cl_uint index_norm_inf(vector_base<NumericT> const & x)
{
  viennacl::scalar<NumericT> result(0, viennacl::traits::context(x));
  scheduler::statement statement = scheduler::preset::index_norm_inf(&result, &x);
  kernels::vector<NumericT>::execution_handler(viennacl::traits::opencl_context(x)).execute("index_norm_inf", statement);
  NumericT host_result = result;
  return static_cast<cl_uint>(host_result);
}

////////// max

/** @brief Computes the maximum of a vector
*
* @param x      The vector
* @param result The result scalar
*/
template<typename NumericT>
void max_impl(vector_base<NumericT> const & x,
                   scalar<NumericT> & result)
{
  assert(viennacl::traits::opencl_handle(x).context() == viennacl::traits::opencl_handle(result).context() && bool("Operands do not reside in the same OpenCL context. Automatic migration not yet supported!"));

  scheduler::statement statement = scheduler::preset::max(&result, &x);
  kernels::vector<NumericT>::execution_handler(viennacl::traits::opencl_context(x)).execute("max", statement);
}

/** @brief Computes the supremum-norm of a vector
*
* @param x      The vector
* @param result The result scalar
*/
template<typename NumericT>
void max_cpu(vector_base<NumericT> const & x,
                  NumericT & result)
{
  scalar<NumericT> tmp(0, viennacl::traits::context(x));
  max_impl(x, tmp);
  result = tmp;
}


////////// min

/** @brief Computes the minimum of a vector
*
* @param x      The vector
* @param result The result scalar
*/
template<typename NumericT>
void min_impl(vector_base<NumericT> const & x,
                   scalar<NumericT> & result)
{
  assert(viennacl::traits::opencl_handle(x).context() == viennacl::traits::opencl_handle(result).context() && bool("Operands do not reside in the same OpenCL context. Automatic migration not yet supported!"));

  scheduler::statement statement = scheduler::preset::min(&result, &x);
  kernels::vector<NumericT>::execution_handler(viennacl::traits::opencl_context(x)).execute("min", statement);
}

/** @brief Computes the supremum-norm of a vector
*
* @param x      The vector
* @param result The result scalar
*/
template<typename NumericT>
void min_cpu(vector_base<NumericT> const & x,
                  NumericT & result)
{
  scalar<NumericT> tmp(0, viennacl::traits::context(x));
  min_impl(x, tmp);
  result = tmp;
}



//TODO: Special case x == y allows improvement!!
/** @brief Computes a plane rotation of two vectors.
*
* Computes (x,y) <- (alpha * x + beta * y, -beta * x + alpha * y)
*
* @param x   The first vector
* @param y   The second vector
* @param alpha  The first transformation coefficient
* @param beta   The second transformation coefficient
*/
template<typename NumericT>
void plane_rotation(vector_base<NumericT> & x,
                    vector_base<NumericT> & y,
                    NumericT alpha, NumericT beta)
{
  assert(viennacl::traits::opencl_handle(x).context() == viennacl::traits::opencl_handle(y).context() && bool("Operands do not reside in the same OpenCL context. Automatic migration not yet supported!"));
  assert(viennacl::traits::size(x) == viennacl::traits::size(y));

  device_specific::statements_container statement = scheduler::preset::plane_rotation(&x, &y, &alpha, &beta);
  kernels::vector<NumericT>::execution_handler(viennacl::traits::opencl_context(x)).execute("plane_rotation", statement);
}

//////////////////////////


namespace detail
{
  /** @brief Worker routine for scan routines using OpenCL
   *
   * Note on performance: For non-in-place scans one could optimize away the temporary 'opencl_carries'-array.
   * This, however, only provides small savings in the latency-dominated regime, yet would effectively double the amount of code to maintain.
   */
  template<typename NumericT>
  void scan_impl(vector_base<NumericT> const & input,
                 vector_base<NumericT>       & output,
                 bool is_inclusive)
  {
    vcl_size_t local_worksize = 128;
    vcl_size_t workgroups = 128;

    viennacl::backend::mem_handle opencl_carries;
    viennacl::backend::memory_create(opencl_carries, sizeof(NumericT)*workgroups, viennacl::traits::context(input));

    viennacl::ocl::context & ctx = const_cast<viennacl::ocl::context &>(viennacl::traits::opencl_handle(input).context());
    viennacl::linalg::opencl::kernels::scan<NumericT>::init(ctx);
    viennacl::ocl::kernel& k1 = ctx.get_kernel(viennacl::linalg::opencl::kernels::scan<NumericT>::program_name(), "scan_1");
    viennacl::ocl::kernel& k2 = ctx.get_kernel(viennacl::linalg::opencl::kernels::scan<NumericT>::program_name(), "scan_2");
    viennacl::ocl::kernel& k3 = ctx.get_kernel(viennacl::linalg::opencl::kernels::scan<NumericT>::program_name(), "scan_3");

    // First step: Scan within each thread group and write carries
    k1.local_work_size(0, local_worksize);
    k1.global_work_size(0, workgroups * local_worksize);
    viennacl::ocl::enqueue(k1( input, cl_uint( input.start()), cl_uint( input.stride()), cl_uint(input.size()),
                              output, cl_uint(output.start()), cl_uint(output.stride()),
                              cl_uint(is_inclusive ? 0 : 1), opencl_carries.opencl_handle())
                          );

    // Second step: Compute offset for each thread group (exclusive scan for each thread group)
    k2.local_work_size(0, workgroups);
    k2.global_work_size(0, workgroups);
    viennacl::ocl::enqueue(k2(opencl_carries.opencl_handle()));

    // Third step: Offset each thread group accordingly
    k3.local_work_size(0, local_worksize);
    k3.global_work_size(0, workgroups * local_worksize);
    viennacl::ocl::enqueue(k3(output, cl_uint(output.start()), cl_uint(output.stride()), cl_uint(output.size()),
                              opencl_carries.opencl_handle())
                          );
  }
}


/** @brief This function implements an inclusive scan using CUDA.
*
* @param input       Input vector.
* @param output      The output vector. Must be non-overlapping with input.
*/
template<typename NumericT>
void inclusive_scan(vector_base<NumericT> const & input,
                    vector_base<NumericT>       & output)
{
  detail::scan_impl(input, output, true);
}


/** @brief This function implements an in-place inclusive scan using CUDA.
*
* @param x       The vector to inclusive-scan
*/
template<typename NumericT>
void inclusive_scan(vector_base<NumericT> & x)
{
  detail::scan_impl(x, x, true);
}

/** @brief This function implements an exclusive scan using CUDA.
*
* @param input       Input vector
* @param output      The output vector. Must be non-overlapping with input
*/
template<typename NumericT>
void exclusive_scan(vector_base<NumericT> const & input,
                    vector_base<NumericT>       & output)
{
  detail::scan_impl(input, output, false);
}

/** @brief This function implements an in-place exclusive scan using CUDA.
*
* @param input       Input vector
* @param output      The output vector.
*/
template<typename NumericT>
void exclusive_scan(vector_base<NumericT> & x)
{
  detail::scan_impl(x, x, false);
}


} //namespace opencl
} //namespace linalg
} //namespace viennacl


#endif
