#ifndef VIENNACL_LINALG_OPENCL_VECTOR_OPERATIONS_HPP_
#define VIENNACL_LINALG_OPENCL_VECTOR_OPERATIONS_HPP_

/* =========================================================================
   Copyright (c) 2010-2013, Institute for Microelectronics,
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
#include "viennacl/ocl/device.hpp"
#include "viennacl/ocl/handle.hpp"
#include "viennacl/ocl/kernel.hpp"
#include "viennacl/scalar.hpp"
#include "viennacl/tools/tools.hpp"
#include "viennacl/linalg/kernels/vector_kernels.h"
#include "viennacl/linalg/kernels/vector_element_kernels.h"
#include "viennacl/meta/predicate.hpp"
#include "viennacl/meta/enable_if.hpp"
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


      template <typename T, typename ScalarType1>
      void av(vector_base<T> & vec1,
              vector_base<T> const & vec2, ScalarType1 const & alpha, std::size_t len_alpha, bool reciprocal_alpha, bool flip_sign_alpha)
      {
        assert(viennacl::traits::opencl_handle(vec1).context() == viennacl::traits::opencl_handle(vec2).context() && bool("Vectors do not reside in the same OpenCL context. Automatic migration not yet supported!"));

        viennacl::ocl::context & ctx = const_cast<viennacl::ocl::context &>(viennacl::traits::opencl_handle(vec1).context());
        viennacl::linalg::kernels::vector<T, 1>::init(ctx);

        cl_uint options_alpha =   ((len_alpha > 1) ? (len_alpha << 2) : 0)
                                + (reciprocal_alpha ? 2 : 0)
                                + (flip_sign_alpha ? 1 : 0);

        viennacl::ocl::kernel & k = ctx.get_kernel(viennacl::linalg::kernels::vector<T, 1>::program_name(),
                                                   (viennacl::is_cpu_scalar<ScalarType1>::value ? "av_cpu" : "av_gpu"));
        k.global_work_size(0, std::min<std::size_t>(128 * k.local_work_size(),
                                                    viennacl::tools::roundUpToNextMultiple<std::size_t>(viennacl::traits::size(vec1), k.local_work_size()) ) );

        viennacl::ocl::packed_cl_uint size_vec1;
        size_vec1.start  = cl_uint(viennacl::traits::start(vec1));
        size_vec1.stride = cl_uint(viennacl::traits::stride(vec1));
        size_vec1.size   = cl_uint(viennacl::traits::size(vec1));
        size_vec1.internal_size   = cl_uint(viennacl::traits::internal_size(vec1));

        viennacl::ocl::packed_cl_uint size_vec2;
        size_vec2.start  = cl_uint(viennacl::traits::start(vec2));
        size_vec2.stride = cl_uint(viennacl::traits::stride(vec2));
        size_vec2.size   = cl_uint(viennacl::traits::size(vec2));
        size_vec2.internal_size   = cl_uint(viennacl::traits::internal_size(vec2));


        viennacl::ocl::enqueue(k(viennacl::traits::opencl_handle(vec1),
                                 size_vec1,

                                 viennacl::traits::opencl_handle(viennacl::tools::promote_if_host_scalar<T>(alpha)),
                                 options_alpha,
                                 viennacl::traits::opencl_handle(vec2),
                                 size_vec2 )
                              );
      }


      template <typename T, typename ScalarType1, typename ScalarType2>
      void avbv(vector_base<T> & vec1,
                vector_base<T> const & vec2, ScalarType1 const & alpha, std::size_t len_alpha, bool reciprocal_alpha, bool flip_sign_alpha,
                vector_base<T> const & vec3, ScalarType2 const & beta,  std::size_t len_beta,  bool reciprocal_beta,  bool flip_sign_beta)
      {
        assert(viennacl::traits::opencl_handle(vec1).context() == viennacl::traits::opencl_handle(vec2).context() && bool("Vectors do not reside in the same OpenCL context. Automatic migration not yet supported!"));
        assert(viennacl::traits::opencl_handle(vec2).context() == viennacl::traits::opencl_handle(vec3).context() && bool("Vectors do not reside in the same OpenCL context. Automatic migration not yet supported!"));

        viennacl::ocl::context & ctx = const_cast<viennacl::ocl::context &>(viennacl::traits::opencl_handle(vec1).context());
        viennacl::linalg::kernels::vector<T, 1>::init(ctx);

        std::string kernel_name;
        if (viennacl::is_cpu_scalar<ScalarType1>::value && viennacl::is_cpu_scalar<ScalarType2>::value)
          kernel_name = "avbv_cpu_cpu";
        else if (viennacl::is_cpu_scalar<ScalarType1>::value && !viennacl::is_cpu_scalar<ScalarType2>::value)
          kernel_name = "avbv_cpu_gpu";
        else if (!viennacl::is_cpu_scalar<ScalarType1>::value && viennacl::is_cpu_scalar<ScalarType2>::value)
          kernel_name = "avbv_gpu_cpu";
        else
          kernel_name = "avbv_gpu_gpu";

        cl_uint options_alpha =   ((len_alpha > 1) ? (len_alpha << 2) : 0)
                                + (reciprocal_alpha ? 2 : 0)
                                + (flip_sign_alpha ? 1 : 0);
        cl_uint options_beta =    ((len_beta > 1) ? (len_beta << 2) : 0)
                                + (reciprocal_beta ? 2 : 0)
                                + (flip_sign_beta ? 1 : 0);

        viennacl::ocl::kernel & k = ctx.get_kernel(viennacl::linalg::kernels::vector<T, 1>::program_name(), kernel_name);
        k.global_work_size(0, std::min<std::size_t>(128 * k.local_work_size(),
                                                    viennacl::tools::roundUpToNextMultiple<std::size_t>(viennacl::traits::size(vec1), k.local_work_size()) ) );

        viennacl::ocl::packed_cl_uint size_vec1;
        size_vec1.start  = cl_uint(viennacl::traits::start(vec1));
        size_vec1.stride = cl_uint(viennacl::traits::stride(vec1));
        size_vec1.size   = cl_uint(viennacl::traits::size(vec1));
        size_vec1.internal_size   = cl_uint(viennacl::traits::internal_size(vec1));

        viennacl::ocl::packed_cl_uint size_vec2;
        size_vec2.start  = cl_uint(viennacl::traits::start(vec2));
        size_vec2.stride = cl_uint(viennacl::traits::stride(vec2));
        size_vec2.size   = cl_uint(viennacl::traits::size(vec2));
        size_vec2.internal_size   = cl_uint(viennacl::traits::internal_size(vec2));

        viennacl::ocl::packed_cl_uint size_vec3;
        size_vec3.start  = cl_uint(viennacl::traits::start(vec3));
        size_vec3.stride = cl_uint(viennacl::traits::stride(vec3));
        size_vec3.size   = cl_uint(viennacl::traits::size(vec3));
        size_vec3.internal_size   = cl_uint(viennacl::traits::internal_size(vec3));

        viennacl::ocl::enqueue(k(viennacl::traits::opencl_handle(vec1),
                                 size_vec1,

                                 viennacl::traits::opencl_handle(viennacl::tools::promote_if_host_scalar<T>(alpha)),
                                 options_alpha,
                                 viennacl::traits::opencl_handle(vec2),
                                 size_vec2,

                                 viennacl::traits::opencl_handle(viennacl::tools::promote_if_host_scalar<T>(beta)),
                                 options_beta,
                                 viennacl::traits::opencl_handle(vec3),
                                 size_vec3 )
                              );
      }


      template <typename T, typename ScalarType1, typename ScalarType2>
      void avbv_v(vector_base<T> & vec1,
                  vector_base<T> const & vec2, ScalarType1 const & alpha, std::size_t len_alpha, bool reciprocal_alpha, bool flip_sign_alpha,
                  vector_base<T> const & vec3, ScalarType2 const & beta,  std::size_t len_beta,  bool reciprocal_beta,  bool flip_sign_beta)
      {
        assert(viennacl::traits::opencl_handle(vec1).context() == viennacl::traits::opencl_handle(vec2).context() && bool("Vectors do not reside in the same OpenCL context. Automatic migration not yet supported!"));
        assert(viennacl::traits::opencl_handle(vec2).context() == viennacl::traits::opencl_handle(vec3).context() && bool("Vectors do not reside in the same OpenCL context. Automatic migration not yet supported!"));

        viennacl::ocl::context & ctx = const_cast<viennacl::ocl::context &>(viennacl::traits::opencl_handle(vec1).context());
        viennacl::linalg::kernels::vector<T, 1>::init(ctx);

        std::string kernel_name;
        if (viennacl::is_cpu_scalar<ScalarType1>::value && viennacl::is_cpu_scalar<ScalarType2>::value)
          kernel_name = "avbv_v_cpu_cpu";
        else if (viennacl::is_cpu_scalar<ScalarType1>::value && !viennacl::is_cpu_scalar<ScalarType2>::value)
          kernel_name = "avbv_v_cpu_gpu";
        else if (!viennacl::is_cpu_scalar<ScalarType1>::value && viennacl::is_cpu_scalar<ScalarType2>::value)
          kernel_name = "avbv_v_gpu_cpu";
        else
          kernel_name = "avbv_v_gpu_gpu";

        cl_uint options_alpha =   ((len_alpha > 1) ? (len_alpha << 2) : 0)
                                + (reciprocal_alpha ? 2 : 0)
                                + (flip_sign_alpha ? 1 : 0);
        cl_uint options_beta =    ((len_beta > 1) ? (len_beta << 2) : 0)
                                + (reciprocal_beta ? 2 : 0)
                                + (flip_sign_beta ? 1 : 0);

        viennacl::ocl::kernel & k = ctx.get_kernel(viennacl::linalg::kernels::vector<T, 1>::program_name(), kernel_name);
        k.global_work_size(0, std::min<std::size_t>(128 * k.local_work_size(),
                                                    viennacl::tools::roundUpToNextMultiple<std::size_t>(viennacl::traits::size(vec1), k.local_work_size()) ) );

        viennacl::ocl::packed_cl_uint size_vec1;
        size_vec1.start  = cl_uint(viennacl::traits::start(vec1));
        size_vec1.stride = cl_uint(viennacl::traits::stride(vec1));
        size_vec1.size   = cl_uint(viennacl::traits::size(vec1));
        size_vec1.internal_size   = cl_uint(viennacl::traits::internal_size(vec1));

        viennacl::ocl::packed_cl_uint size_vec2;
        size_vec2.start  = cl_uint(viennacl::traits::start(vec2));
        size_vec2.stride = cl_uint(viennacl::traits::stride(vec2));
        size_vec2.size   = cl_uint(viennacl::traits::size(vec2));
        size_vec2.internal_size   = cl_uint(viennacl::traits::internal_size(vec2));

        viennacl::ocl::packed_cl_uint size_vec3;
        size_vec3.start  = cl_uint(viennacl::traits::start(vec3));
        size_vec3.stride = cl_uint(viennacl::traits::stride(vec3));
        size_vec3.size   = cl_uint(viennacl::traits::size(vec3));
        size_vec3.internal_size   = cl_uint(viennacl::traits::internal_size(vec3));

        viennacl::ocl::enqueue(k(viennacl::traits::opencl_handle(vec1),
                                 size_vec1,

                                 viennacl::traits::opencl_handle(viennacl::tools::promote_if_host_scalar<T>(alpha)),
                                 options_alpha,
                                 viennacl::traits::opencl_handle(vec2),
                                 size_vec2,

                                 viennacl::traits::opencl_handle(viennacl::tools::promote_if_host_scalar<T>(beta)),
                                 options_beta,
                                 viennacl::traits::opencl_handle(vec3),
                                 size_vec3 )
                              );
      }


      /** @brief Assign a constant value to a vector (-range/-slice)
      *
      * @param vec1   The vector to which the value should be assigned
      * @param alpha  The value to be assigned
      */
      template <typename T>
      void vector_assign(vector_base<T> & vec1, const T & alpha)
      {
        viennacl::ocl::context & ctx = const_cast<viennacl::ocl::context &>(viennacl::traits::opencl_handle(vec1).context());
        viennacl::linalg::kernels::vector<T, 1>::init(ctx);

        viennacl::ocl::kernel & k = ctx.get_kernel(viennacl::linalg::kernels::vector<T, 1>::program_name(), "assign_cpu");
        k.global_work_size(0, std::min<std::size_t>(128 * k.local_work_size(),
                                                    viennacl::tools::roundUpToNextMultiple<std::size_t>(viennacl::traits::size(vec1), k.local_work_size()) ) );
        viennacl::ocl::enqueue(k(viennacl::traits::opencl_handle(vec1),
                                 cl_uint(viennacl::traits::start(vec1)),
                                 cl_uint(viennacl::traits::stride(vec1)),
                                 cl_uint(viennacl::traits::size(vec1)),
                                 cl_uint(vec1.internal_size()),     //Note: Do NOT use traits::internal_size() here, because vector proxies don't require padding.
                                 viennacl::traits::opencl_handle(T(alpha)) )
                              );
      }


      /** @brief Swaps the contents of two vectors, data is copied
      *
      * @param vec1   The first vector (or -range, or -slice)
      * @param vec2   The second vector (or -range, or -slice)
      */
      template <typename T>
      void vector_swap(vector_base<T> & vec1, vector_base<T> & vec2)
      {
        assert(viennacl::traits::opencl_handle(vec1).context() == viennacl::traits::opencl_handle(vec2).context() && bool("Vectors do not reside in the same OpenCL context. Automatic migration not yet supported!"));

        viennacl::ocl::context & ctx = const_cast<viennacl::ocl::context &>(viennacl::traits::opencl_handle(vec1).context());
        viennacl::linalg::kernels::vector<T, 1>::init(ctx);

        viennacl::ocl::kernel & k = ctx.get_kernel(viennacl::linalg::kernels::vector<T, 1>::program_name(), "swap");

        viennacl::ocl::enqueue(k(viennacl::traits::opencl_handle(vec1),
                                 cl_uint(viennacl::traits::start(vec1)),
                                 cl_uint(viennacl::traits::stride(vec1)),
                                 cl_uint(viennacl::traits::size(vec1)),
                                 viennacl::traits::opencl_handle(vec2),
                                 cl_uint(viennacl::traits::start(vec2)),
                                 cl_uint(viennacl::traits::stride(vec2)),
                                 cl_uint(viennacl::traits::size(vec2)))
                              );
      }

      ///////////////////////// Binary Elementwise operations /////////////

      /** @brief Implementation of the element-wise operation v1 = v2 .* v3 and v1 = v2 ./ v3    (using MATLAB syntax)
      *
      * @param vec1   The result vector (or -range, or -slice)
      * @param proxy  The proxy object holding v2, v3 and the operation
      */
      template <typename T, typename OP>
      void element_op(vector_base<T> & vec1,
                      vector_expression<const vector_base<T>, const vector_base<T>, op_element_binary<OP> > const & proxy)
      {
        assert(viennacl::traits::opencl_handle(vec1).context() == viennacl::traits::opencl_handle(proxy.lhs()).context() && bool("Vectors do not reside in the same OpenCL context. Automatic migration not yet supported!"));
        assert(viennacl::traits::opencl_handle(vec1).context() == viennacl::traits::opencl_handle(proxy.rhs()).context() && bool("Vectors do not reside in the same OpenCL context. Automatic migration not yet supported!"));

        viennacl::ocl::context & ctx = const_cast<viennacl::ocl::context &>(viennacl::traits::opencl_handle(vec1).context());
        viennacl::linalg::kernels::vector<T, 1>::init(ctx);

        viennacl::ocl::kernel & k = ctx.get_kernel(viennacl::linalg::kernels::vector<T, 1>::program_name(), "element_op");

        viennacl::ocl::enqueue(k(viennacl::traits::opencl_handle(vec1),
                                 cl_uint(viennacl::traits::start(vec1)),
                                 cl_uint(viennacl::traits::stride(vec1)),
                                 cl_uint(viennacl::traits::size(vec1)),

                                 viennacl::traits::opencl_handle(proxy.lhs()),
                                 cl_uint(viennacl::traits::start(proxy.lhs())),
                                 cl_uint(viennacl::traits::stride(proxy.lhs())),

                                 viennacl::traits::opencl_handle(proxy.rhs()),
                                 cl_uint(viennacl::traits::start(proxy.rhs())),
                                 cl_uint(viennacl::traits::stride(proxy.rhs())),

                                 cl_uint(viennacl::is_division<OP>::value))
                              );
      }

      ///////////////////////// Unary Elementwise operations /////////////

      namespace detail
      {
        inline std::string op_to_string(op_abs)   { return "abs";   }
        inline std::string op_to_string(op_acos)  { return "acos";  }
        inline std::string op_to_string(op_asin)  { return "asin";  }
        inline std::string op_to_string(op_ceil)  { return "ceil";  }
        inline std::string op_to_string(op_cos)   { return "cos";   }
        inline std::string op_to_string(op_cosh)  { return "cosh";  }
        inline std::string op_to_string(op_exp)   { return "exp";   }
        inline std::string op_to_string(op_fabs)  { return "fabs";  }
        inline std::string op_to_string(op_floor) { return "floor"; }
        inline std::string op_to_string(op_log)   { return "log";   }
        inline std::string op_to_string(op_log10) { return "log10"; }
        inline std::string op_to_string(op_sin)   { return "sin";   }
        inline std::string op_to_string(op_sinh)  { return "sinh";  }
        inline std::string op_to_string(op_sqrt)  { return "sqrt";  }
        inline std::string op_to_string(op_tan)   { return "tan";   }
        inline std::string op_to_string(op_tanh)  { return "tanh";  }
      }

      /** @brief Implementation of the element-wise operation v1 = v2 .* v3 and v1 = v2 ./ v3    (using MATLAB syntax)
      *
      * @param vec1   The result vector (or -range, or -slice)
      * @param proxy  The proxy object holding v2, v3 and the operation
      */
      template <typename T, typename OP>
      void element_op(vector_base<T> & vec1,
                      vector_expression<const vector_base<T>, const vector_base<T>, op_element_unary<OP> > const & proxy)
      {
        assert(viennacl::traits::opencl_handle(vec1).context() == viennacl::traits::opencl_handle(proxy.lhs()).context() && bool("Vectors do not reside in the same OpenCL context. Automatic migration not yet supported!"));
        assert(viennacl::traits::opencl_handle(vec1).context() == viennacl::traits::opencl_handle(proxy.rhs()).context() && bool("Vectors do not reside in the same OpenCL context. Automatic migration not yet supported!"));

        viennacl::ocl::context & ctx = const_cast<viennacl::ocl::context &>(viennacl::traits::opencl_handle(vec1).context());
        viennacl::linalg::kernels::vector_element<T, 1>::init(ctx);

        viennacl::ocl::kernel & k = ctx.get_kernel(viennacl::linalg::kernels::vector_element<T, 1>::program_name(), detail::op_to_string(OP()) + "_assign");

        viennacl::ocl::packed_cl_uint size_vec1;
        size_vec1.start  = cl_uint(viennacl::traits::start(vec1));
        size_vec1.stride = cl_uint(viennacl::traits::stride(vec1));
        size_vec1.size   = cl_uint(viennacl::traits::size(vec1));
        size_vec1.internal_size   = cl_uint(viennacl::traits::internal_size(vec1));

        viennacl::ocl::packed_cl_uint size_vec2;
        size_vec2.start  = cl_uint(viennacl::traits::start(proxy.lhs()));
        size_vec2.stride = cl_uint(viennacl::traits::stride(proxy.lhs()));
        size_vec2.size   = cl_uint(viennacl::traits::size(proxy.lhs()));
        size_vec2.internal_size   = cl_uint(viennacl::traits::internal_size(proxy.lhs()));

        viennacl::ocl::enqueue(k(viennacl::traits::opencl_handle(vec1),
                                 size_vec1,
                                 viennacl::traits::opencl_handle(proxy.lhs()),
                                 size_vec2)
                              );
      }

      ///////////////////////// Norms and inner product ///////////////////

      /** @brief Computes the partial inner product of two vectors - implementation. Library users should call inner_prod(vec1, vec2).
      *
      * @param vec1 The first vector
      * @param vec2 The second vector
      * @param partial_result The results of each group
      */
      template <typename T>
      void inner_prod_impl(vector_base<T> const & vec1,
                           vector_base<T> const & vec2,
                           vector_base<T> & partial_result)
      {
        assert(viennacl::traits::opencl_handle(vec1).context() == viennacl::traits::opencl_handle(vec2).context() && bool("Vectors do not reside in the same OpenCL context. Automatic migration not yet supported!"));
        assert(viennacl::traits::opencl_handle(vec2).context() == viennacl::traits::opencl_handle(partial_result).context() && bool("Vectors do not reside in the same OpenCL context. Automatic migration not yet supported!"));

        viennacl::ocl::context & ctx = const_cast<viennacl::ocl::context &>(viennacl::traits::opencl_handle(vec1).context());
        viennacl::linalg::kernels::vector<T, 1>::init(ctx);

        assert( (viennacl::traits::size(vec1) == viennacl::traits::size(vec2))
              && bool("Incompatible vector sizes in inner_prod_impl()!"));

        viennacl::ocl::kernel & k = ctx.get_kernel(viennacl::linalg::kernels::vector<T, 1>::program_name(), "inner_prod");

        assert( (k.global_work_size() / k.local_work_size() <= partial_result.size()) && bool("Size mismatch for partial reduction in inner_prod_impl()") );

        viennacl::ocl::enqueue(k(viennacl::traits::opencl_handle(vec1),
                                 cl_uint(viennacl::traits::start(vec1)),
                                 cl_uint(viennacl::traits::stride(vec1)),
                                 cl_uint(viennacl::traits::size(vec1)),
                                 viennacl::traits::opencl_handle(vec2),
                                 cl_uint(viennacl::traits::start(vec2)),
                                 cl_uint(viennacl::traits::stride(vec2)),
                                 cl_uint(viennacl::traits::size(vec2)),
                                 viennacl::ocl::local_mem(sizeof(T) * k.local_work_size()),
                                 viennacl::traits::opencl_handle(partial_result)
                                )
                              );
      }


      //implementation of inner product:
      //namespace {
      /** @brief Computes the inner product of two vectors - implementation. Library users should call inner_prod(vec1, vec2).
      *
      * @param vec1 The first vector
      * @param vec2 The second vector
      * @param result The result scalar (on the gpu)
      */
      template <typename T>
      void inner_prod_impl(vector_base<T> const & vec1,
                           vector_base<T> const & vec2,
                           scalar<T> & result)
      {
        assert(viennacl::traits::opencl_handle(vec1).context() == viennacl::traits::opencl_handle(vec2).context() && bool("Vectors do not reside in the same OpenCL context. Automatic migration not yet supported!"));
        assert(viennacl::traits::opencl_handle(vec1).context() == viennacl::traits::opencl_handle(result).context() && bool("Operands do not reside in the same OpenCL context. Automatic migration not yet supported!"));

        viennacl::ocl::context & ctx = const_cast<viennacl::ocl::context &>(viennacl::traits::opencl_handle(vec1).context());

        static std::map<viennacl::ocl::context, viennacl::vector<T> > temp_vectors_per_context;

        std::size_t work_groups = 128;
        viennacl::vector<T> & temp = temp_vectors_per_context[ctx];
        temp.resize(work_groups, ctx); // bring default-constructed vectors to the correct size:

        // Step 1: Compute partial inner products for each work group:
        inner_prod_impl(vec1, vec2, temp);

        // Step 2: Sum partial results:
        viennacl::ocl::kernel & ksum = ctx.get_kernel(viennacl::linalg::kernels::vector<T, 1>::program_name(), "sum");

        ksum.local_work_size(0, work_groups);
        ksum.global_work_size(0, work_groups);
        viennacl::ocl::enqueue(ksum(viennacl::traits::opencl_handle(temp),
                                    cl_uint(viennacl::traits::start(temp)),
                                    cl_uint(viennacl::traits::stride(temp)),
                                    cl_uint(viennacl::traits::size(temp)),
                                    cl_uint(1),
                                    viennacl::ocl::local_mem(sizeof(T) * ksum.local_work_size()),
                                    viennacl::traits::opencl_handle(result) )
                              );
      }


      //implementation of inner product:
      //namespace {
      /** @brief Computes the inner product of two vectors - implementation. Library users should call inner_prod(vec1, vec2).
      *
      * @param vec1 The first vector
      * @param vec2 The second vector
      * @param result The result scalar (on the gpu)
      */
      template <typename T>
      void inner_prod_cpu(vector_base<T> const & vec1,
                          vector_base<T> const & vec2,
                          T & result)
      {
        assert(viennacl::traits::opencl_handle(vec1).context() == viennacl::traits::opencl_handle(vec2).context() && bool("Vectors do not reside in the same OpenCL context. Automatic migration not yet supported!"));

        viennacl::ocl::context & ctx = const_cast<viennacl::ocl::context &>(viennacl::traits::opencl_handle(vec1).context());

        static std::map<viennacl::ocl::context, viennacl::vector<T> > temp_vectors_per_context;

        std::size_t work_groups = 128;
        viennacl::vector<T> & temp = temp_vectors_per_context[ctx];
        temp.resize(work_groups, ctx); // bring default-constructed vectors to the correct size:

        // Step 1: Compute partial inner products for each work group:
        inner_prod_impl(vec1, vec2, temp);

        // Step 2: Sum partial results:

        // Now copy partial results from GPU back to CPU and run reduction there:
        static std::vector<T> temp_cpu(work_groups);
        viennacl::fast_copy(temp.begin(), temp.end(), temp_cpu.begin());

        result = 0;
        for (typename std::vector<T>::const_iterator it = temp_cpu.begin(); it != temp_cpu.end(); ++it)
          result += *it;
      }


      //////////// Helper for norms

      /** @brief Computes the partial work group results for vector norms
      *
      * @param vec The vector
      * @param partial_result The result scalar
      * @param norm_id        Norm selector. 0: norm_inf, 1: norm_1, 2: norm_2
      */
      template <typename T>
      void norm_reduction_impl(vector_base<T> const & vec,
                               vector_base<T> & partial_result,
                                cl_uint norm_id)
      {
        assert(viennacl::traits::opencl_handle(vec).context() == viennacl::traits::opencl_handle(partial_result).context() && bool("Operands do not reside in the same OpenCL context. Automatic migration not yet supported!"));

        viennacl::ocl::context & ctx = const_cast<viennacl::ocl::context &>(viennacl::traits::opencl_handle(vec).context());
        viennacl::linalg::kernels::vector<T, 1>::init(ctx);

        viennacl::ocl::kernel & k = ctx.get_kernel(viennacl::linalg::kernels::vector<T, 1>::program_name(), "norm");

        assert( (k.global_work_size() / k.local_work_size() <= partial_result.size()) && bool("Size mismatch for partial reduction in norm_reduction_impl()") );

        viennacl::ocl::enqueue(k(viennacl::traits::opencl_handle(vec),
                                 cl_uint(viennacl::traits::start(vec)),
                                 cl_uint(viennacl::traits::stride(vec)),
                                 cl_uint(viennacl::traits::size(vec)),
                                 cl_uint(norm_id),
                                 viennacl::ocl::local_mem(sizeof(T) * k.local_work_size()),
                                 viennacl::traits::opencl_handle(partial_result) )
                              );
      }


      //////////// Norm 1

      /** @brief Computes the l^1-norm of a vector
      *
      * @param vec The vector
      * @param result The result scalar
      */
      template <typename T>
      void norm_1_impl(vector_base<T> const & vec,
                       scalar<T> & result)
      {
        assert(viennacl::traits::opencl_handle(vec).context() == viennacl::traits::opencl_handle(result).context() && bool("Operands do not reside in the same OpenCL context. Automatic migration not yet supported!"));

        viennacl::ocl::context & ctx = const_cast<viennacl::ocl::context &>(viennacl::traits::opencl_handle(vec).context());

        static std::map<viennacl::ocl::context, viennacl::vector<T> > temp_vectors_per_context;

        std::size_t work_groups = 128;
        viennacl::vector<T> & temp = temp_vectors_per_context[ctx];
        temp.resize(work_groups, ctx); // bring default-constructed vectors to the correct size:

        // Step 1: Compute the partial work group results
        norm_reduction_impl(vec, temp, 1);

        // Step 2: Compute the partial reduction using OpenCL
        viennacl::ocl::kernel & ksum = ctx.get_kernel(viennacl::linalg::kernels::vector<T, 1>::program_name(), "sum");

        ksum.local_work_size(0, work_groups);
        ksum.global_work_size(0, work_groups);
        viennacl::ocl::enqueue(ksum(viennacl::traits::opencl_handle(temp),
                                    cl_uint(viennacl::traits::start(temp)),
                                    cl_uint(viennacl::traits::stride(temp)),
                                    cl_uint(viennacl::traits::size(temp)),
                                    cl_uint(1),
                                    viennacl::ocl::local_mem(sizeof(T) * ksum.local_work_size()),
                                    result)
                              );
      }

      /** @brief Computes the l^1-norm of a vector with final reduction on CPU
      *
      * @param vec The vector
      * @param result The result scalar
      */
      template <typename T>
      void norm_1_cpu(vector_base<T> const & vec,
                      T & result)
      {
        viennacl::ocl::context & ctx = const_cast<viennacl::ocl::context &>(viennacl::traits::opencl_handle(vec).context());

        static std::map<viennacl::ocl::context, viennacl::vector<T> > temp_vectors_per_context;

        std::size_t work_groups = 128;
        viennacl::vector<T> & temp = temp_vectors_per_context[ctx];
        temp.resize(work_groups, ctx); // bring default-constructed vectors to the correct size:

        // Step 1: Compute the partial work group results
        norm_reduction_impl(vec, temp, 1);

        // Step 2: Now copy partial results from GPU back to CPU and run reduction there:
        static std::vector<T> temp_cpu(work_groups);
        viennacl::fast_copy(temp.begin(), temp.end(), temp_cpu.begin());

        result = 0;
        for (typename std::vector<T>::const_iterator it = temp_cpu.begin(); it != temp_cpu.end(); ++it)
          result += *it;
      }



      //////// Norm 2


      /** @brief Computes the l^2-norm of a vector - implementation using OpenCL summation at second step
      *
      * @param vec The vector
      * @param result The result scalar
      */
      template <typename T>
      void norm_2_impl(vector_base<T> const & vec,
                       scalar<T> & result)
      {
        assert(viennacl::traits::opencl_handle(vec).context() == viennacl::traits::opencl_handle(result).context() && bool("Operands do not reside in the same OpenCL context. Automatic migration not yet supported!"));

        viennacl::ocl::context & ctx = const_cast<viennacl::ocl::context &>(viennacl::traits::opencl_handle(vec).context());

        static std::map<viennacl::ocl::context, viennacl::vector<T> > temp_vectors_per_context;

        std::size_t work_groups = 128;
        viennacl::vector<T> & temp = temp_vectors_per_context[ctx];
        temp.resize(work_groups, ctx); // bring default-constructed vectors to the correct size:

        // Step 1: Compute the partial work group results
        norm_reduction_impl(vec, temp, 2);

        // Step 2: Reduction via OpenCL
        viennacl::ocl::kernel & ksum = ctx.get_kernel(viennacl::linalg::kernels::vector<T, 1>::program_name(), "sum");

        ksum.local_work_size(0, work_groups);
        ksum.global_work_size(0, work_groups);
        viennacl::ocl::enqueue( ksum(viennacl::traits::opencl_handle(temp),
                                      cl_uint(viennacl::traits::start(temp)),
                                      cl_uint(viennacl::traits::stride(temp)),
                                      cl_uint(viennacl::traits::size(temp)),
                                      cl_uint(2),
                                      viennacl::ocl::local_mem(sizeof(T) * ksum.local_work_size()),
                                      result)
                              );
      }

      /** @brief Computes the l^1-norm of a vector with final reduction on CPU
      *
      * @param vec The vector
      * @param result The result scalar
      */
      template <typename T>
      void norm_2_cpu(vector_base<T> const & vec,
                      T & result)
      {
        viennacl::ocl::context & ctx = const_cast<viennacl::ocl::context &>(viennacl::traits::opencl_handle(vec).context());

        static std::map<viennacl::ocl::context, viennacl::vector<T> > temp_vectors_per_context;

        std::size_t work_groups = 128;
        viennacl::vector<T> & temp = temp_vectors_per_context[ctx];
        temp.resize(work_groups, ctx); // bring default-constructed vectors to the correct size:

        // Step 1: Compute the partial work group results
        norm_reduction_impl(vec, temp, 2);

        // Step 2: Now copy partial results from GPU back to CPU and run reduction there:
        static std::vector<T> temp_cpu(work_groups);
        viennacl::fast_copy(temp.begin(), temp.end(), temp_cpu.begin());

        result = 0;
        for (typename std::vector<T>::const_iterator it = temp_cpu.begin(); it != temp_cpu.end(); ++it)
          result += *it;
        result = std::sqrt(result);
      }



      ////////// Norm inf

      /** @brief Computes the supremum-norm of a vector
      *
      * @param vec The vector
      * @param result The result scalar
      */
      template <typename T>
      void norm_inf_impl(vector_base<T> const & vec,
                         scalar<T> & result)
      {
        assert(viennacl::traits::opencl_handle(vec).context() == viennacl::traits::opencl_handle(result).context() && bool("Operands do not reside in the same OpenCL context. Automatic migration not yet supported!"));

        viennacl::ocl::context & ctx = const_cast<viennacl::ocl::context &>(viennacl::traits::opencl_handle(vec).context());

        static std::map<viennacl::ocl::context, viennacl::vector<T> > temp_vectors_per_context;

        std::size_t work_groups = 128;
        viennacl::vector<T> & temp = temp_vectors_per_context[ctx];
        temp.resize(work_groups, ctx); // bring default-constructed vectors to the correct size:

        // Step 1: Compute the partial work group results
        norm_reduction_impl(vec, temp, 0);

        //part 2: parallel reduction of reduced kernel:
        viennacl::ocl::kernel & ksum = ctx.get_kernel(viennacl::linalg::kernels::vector<T, 1>::program_name(), "sum");
        ksum.local_work_size(0, work_groups);
        ksum.global_work_size(0, work_groups);

        viennacl::ocl::enqueue( ksum(viennacl::traits::opencl_handle(temp),
                                     cl_uint(viennacl::traits::start(temp)),
                                     cl_uint(viennacl::traits::stride(temp)),
                                     cl_uint(viennacl::traits::size(temp)),
                                     cl_uint(0),
                                     viennacl::ocl::local_mem(sizeof(T) * ksum.local_work_size()),
                                     result)
                              );
      }

      /** @brief Computes the supremum-norm of a vector
      *
      * @param vec The vector
      * @param result The result scalar
      */
      template <typename T>
      void norm_inf_cpu(vector_base<T> const & vec,
                        T & result)
      {
        viennacl::ocl::context & ctx = const_cast<viennacl::ocl::context &>(viennacl::traits::opencl_handle(vec).context());

        static std::map<viennacl::ocl::context, viennacl::vector<T> > temp_vectors_per_context;

        std::size_t work_groups = 128;
        viennacl::vector<T> & temp = temp_vectors_per_context[ctx];
        temp.resize(work_groups, ctx); // bring default-constructed vectors to the correct size:

        // Step 1: Compute the partial work group results
        norm_reduction_impl(vec, temp, 0);

        // Step 2: Now copy partial results from GPU back to CPU and run reduction there:
        static std::vector<T> temp_cpu(work_groups);
        viennacl::fast_copy(temp.begin(), temp.end(), temp_cpu.begin());

        result = 0;
        for (typename std::vector<T>::const_iterator it = temp_cpu.begin(); it != temp_cpu.end(); ++it)
          result = std::max(result, *it);
      }


      /////////// index norm_inf

      //This function should return a CPU scalar, otherwise statements like
      // vcl_rhs[index_norm_inf(vcl_rhs)]
      // are ambiguous
      /** @brief Computes the index of the first entry that is equal to the supremum-norm in modulus.
      *
      * @param vec The vector
      * @return The result. Note that the result must be a CPU scalar (unsigned int), since gpu scalars are floating point types.
      */
      template <typename T>
      cl_uint index_norm_inf(vector_base<T> const & vec)
      {
        viennacl::ocl::context & ctx = const_cast<viennacl::ocl::context &>(viennacl::traits::opencl_handle(vec).context());
        viennacl::linalg::kernels::vector<T, 1>::init(ctx);

        viennacl::ocl::handle<cl_mem> h = ctx.create_memory(CL_MEM_READ_WRITE, sizeof(cl_uint));

        viennacl::ocl::kernel & k = ctx.get_kernel(viennacl::linalg::kernels::vector<T, 1>::program_name(), "index_norm_inf");
        //cl_uint size = static_cast<cl_uint>(vcl_vec.internal_size());

        //TODO: Use multi-group kernel for large vector sizes

        k.global_work_size(0, k.local_work_size());
        viennacl::ocl::enqueue(k(viennacl::traits::opencl_handle(vec),
                                 cl_uint(viennacl::traits::start(vec)),
                                 cl_uint(viennacl::traits::stride(vec)),
                                 cl_uint(viennacl::traits::size(vec)),
                                 viennacl::ocl::local_mem(sizeof(T) * k.local_work_size()),
                                 viennacl::ocl::local_mem(sizeof(cl_uint) * k.local_work_size()), h));

        //read value:
        cl_uint result;
        cl_int err = clEnqueueReadBuffer(ctx.get_queue().handle().get(), h.get(), CL_TRUE, 0, sizeof(cl_uint), &result, 0, NULL, NULL);
        VIENNACL_ERR_CHECK(err);
        return result;
      }

      //TODO: Special case vec1 == vec2 allows improvement!!
      /** @brief Computes a plane rotation of two vectors.
      *
      * Computes (x,y) <- (alpha * x + beta * y, -beta * x + alpha * y)
      *
      * @param vec1   The first vector
      * @param vec2   The second vector
      * @param alpha  The first transformation coefficient
      * @param beta   The second transformation coefficient
      */
      template <typename T>
      void plane_rotation(vector_base<T> & vec1,
                          vector_base<T> & vec2,
                          T alpha, T beta)
      {
        assert(viennacl::traits::opencl_handle(vec1).context() == viennacl::traits::opencl_handle(vec2).context() && bool("Operands do not reside in the same OpenCL context. Automatic migration not yet supported!"));

        viennacl::ocl::context & ctx = const_cast<viennacl::ocl::context &>(viennacl::traits::opencl_handle(vec1).context());
        viennacl::linalg::kernels::vector<T, 1>::init(ctx);

        assert(viennacl::traits::size(vec1) == viennacl::traits::size(vec2));
        viennacl::ocl::kernel & k = ctx.get_kernel(viennacl::linalg::kernels::vector<T, 1>::program_name(), "plane_rotation");

        viennacl::ocl::enqueue(k(viennacl::traits::opencl_handle(vec1),
                                 cl_uint(viennacl::traits::start(vec1)),
                                 cl_uint(viennacl::traits::stride(vec1)),
                                 cl_uint(viennacl::traits::size(vec1)),
                                 viennacl::traits::opencl_handle(vec2),
                                 cl_uint(viennacl::traits::start(vec2)),
                                 cl_uint(viennacl::traits::stride(vec2)),
                                 cl_uint(viennacl::traits::size(vec2)),
                                 alpha,
                                 beta)
                              );
      }

    } //namespace opencl
  } //namespace linalg
} //namespace viennacl


#endif
