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
#include "viennacl/device_specific/database.hpp"
#include "viennacl/device_specific/execute.hpp"
#include "viennacl/ocl/device.hpp"
#include "viennacl/ocl/handle.hpp"
#include "viennacl/ocl/kernel.hpp"
#include "viennacl/scalar.hpp"
#include "viennacl/tools/tools.hpp"
#include "viennacl/linalg/opencl/common.hpp"
#include "viennacl/linalg/opencl/kernels/vector.hpp"
#include "viennacl/linalg/opencl/kernels/vector_element.hpp"
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


      template <typename T, typename ScalarType1>
      void av(vector_base<T> & vec1,
              vector_base<T> const & vec2, ScalarType1 const & alpha, vcl_size_t len_alpha, bool reciprocal_alpha, bool flip_sign_alpha)
      {
        assert(viennacl::traits::opencl_handle(vec1).context() == viennacl::traits::opencl_handle(vec2).context() && bool("Vectors do not reside in the same OpenCL context. Automatic migration not yet supported!"));

        viennacl::ocl::context & ctx = const_cast<viennacl::ocl::context &>(viennacl::traits::opencl_handle(vec1).context());
        viennacl::linalg::opencl::kernels::vector<T>::init(ctx);
        viennacl::device_specific::enqueue(device_specific::database::get<T>(device_specific::database::axpy),
                                           ctx.get_program(linalg::opencl::kernels::vector<T>::program_name()),
                                           scheduler::preset::avbv(scheduler::OPERATION_BINARY_ASSIGN_TYPE, &vec1, &vec2, &alpha, flip_sign_alpha, reciprocal_alpha, (vector_base<T>*)NULL, (T*)NULL, false, false),
                                           device_specific::BIND_ALL_UNIQUE);
      }


      template <typename T, typename ScalarType1, typename ScalarType2>
      void avbv(vector_base<T> & vec1,
                vector_base<T> const & vec2, ScalarType1 const & alpha, vcl_size_t len_alpha, bool reciprocal_alpha, bool flip_sign_alpha,
                vector_base<T> const & vec3, ScalarType2 const & beta,  vcl_size_t len_beta,  bool reciprocal_beta,  bool flip_sign_beta)
      {
        assert(viennacl::traits::opencl_handle(vec1).context() == viennacl::traits::opencl_handle(vec2).context() && bool("Vectors do not reside in the same OpenCL context. Automatic migration not yet supported!"));
        assert(viennacl::traits::opencl_handle(vec2).context() == viennacl::traits::opencl_handle(vec3).context() && bool("Vectors do not reside in the same OpenCL context. Automatic migration not yet supported!"));

        viennacl::ocl::context & ctx = const_cast<viennacl::ocl::context &>(viennacl::traits::opencl_handle(vec1).context());
        viennacl::linalg::opencl::kernels::vector<T>::init(ctx);

        viennacl::device_specific::enqueue(device_specific::database::get<T>(device_specific::database::axpy),
                                           ctx.get_program(linalg::opencl::kernels::vector<T>::program_name()),
                                           scheduler::preset::avbv(scheduler::OPERATION_BINARY_ASSIGN_TYPE, &vec1, &vec2, &alpha, flip_sign_alpha, reciprocal_alpha, &vec3, &beta, flip_sign_beta, reciprocal_beta),
                                           device_specific::BIND_ALL_UNIQUE);
      }


      template <typename T, typename ScalarType1, typename ScalarType2>
      void avbv_v(vector_base<T> & vec1,
                  vector_base<T> const & vec2, ScalarType1 const & alpha, vcl_size_t len_alpha, bool reciprocal_alpha, bool flip_sign_alpha,
                  vector_base<T> const & vec3, ScalarType2 const & beta,  vcl_size_t len_beta,  bool reciprocal_beta,  bool flip_sign_beta)
      {
        assert(viennacl::traits::opencl_handle(vec1).context() == viennacl::traits::opencl_handle(vec2).context() && bool("Vectors do not reside in the same OpenCL context. Automatic migration not yet supported!"));
        assert(viennacl::traits::opencl_handle(vec2).context() == viennacl::traits::opencl_handle(vec3).context() && bool("Vectors do not reside in the same OpenCL context. Automatic migration not yet supported!"));

        viennacl::ocl::context & ctx = const_cast<viennacl::ocl::context &>(viennacl::traits::opencl_handle(vec1).context());
        viennacl::linalg::opencl::kernels::vector<T>::init(ctx);
        viennacl::device_specific::enqueue(device_specific::database::get<T>(device_specific::database::axpy),
                                           ctx.get_program(linalg::opencl::kernels::vector<T>::program_name()),
                                           scheduler::preset::avbv(scheduler::OPERATION_BINARY_INPLACE_ADD_TYPE, &vec1, &vec2, &alpha, flip_sign_alpha, reciprocal_alpha, &vec3, &beta, flip_sign_beta, reciprocal_beta),
                                           device_specific::BIND_ALL_UNIQUE);
      }


      /** @brief Assign a constant value to a vector (-range/-slice)
      *
      * @param vec1   The vector to which the value should be assigned
      * @param alpha  The value to be assigned
      * @param up_to_internal_size  Specifies whether alpha should also be written to padded memory (mostly used for clearing the whole buffer).
      */
      template <typename T>
      void vector_assign(vector_base<T> & vec1, const T & alpha, bool up_to_internal_size = false)
      {
        viennacl::ocl::context & ctx = const_cast<viennacl::ocl::context &>(viennacl::traits::opencl_handle(vec1).context());
        viennacl::linalg::opencl::kernels::vector<T>::init(ctx);
        scalar_vector<T> vec2(viennacl::traits::size(vec1),alpha);
        device_specific::enqueue(device_specific::database::get<T>(device_specific::database::axpy),
                                 ctx.get_program(viennacl::linalg::opencl::kernels::vector<T>::program_name()),
                                 scheduler::preset::assign_cpu(&vec1, &vec2));
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
        viennacl::linalg::opencl::kernels::vector<T>::init(ctx);
        device_specific::enqueue(device_specific::database::get<T>(device_specific::database::axpy),
                                 ctx.get_program(viennacl::linalg::opencl::kernels::vector<T>::program_name()),
                                 scheduler::preset::swap(&vec1, &vec2));
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
        viennacl::linalg::opencl::kernels::vector_element<T>::init(ctx);

        viennacl::ocl::kernel & k = ctx.get_kernel(viennacl::linalg::opencl::kernels::vector_element<T>::program_name(), "element_op");

        cl_uint op_type = 2; //0: product, 1: division, 2: power
        if (viennacl::is_division<OP>::value)
          op_type = 1;
        else if (viennacl::is_product<OP>::value)
          op_type = 0;

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

                                 op_type)
                              );
      }

      ///////////////////////// Unary Elementwise operations /////////////

      /** @brief Implementation of unary element-wise operations v1 = OP(v2)
      *
      * @param vec1   The result vector (or -range, or -slice)
      * @param proxy  The proxy object holding v2 and the operation
      */
      template <typename T, typename OP>
      void element_op(vector_base<T> & vec1,
                      vector_expression<const vector_base<T>, const vector_base<T>, op_element_unary<OP> > const & proxy)
      {
        assert(viennacl::traits::opencl_handle(vec1).context() == viennacl::traits::opencl_handle(proxy.lhs()).context() && bool("Vectors do not reside in the same OpenCL context. Automatic migration not yet supported!"));
        assert(viennacl::traits::opencl_handle(vec1).context() == viennacl::traits::opencl_handle(proxy.rhs()).context() && bool("Vectors do not reside in the same OpenCL context. Automatic migration not yet supported!"));

        viennacl::ocl::context & ctx = const_cast<viennacl::ocl::context &>(viennacl::traits::opencl_handle(vec1).context());
        viennacl::linalg::opencl::kernels::vector_element<T>::init(ctx);

        viennacl::ocl::kernel & k = ctx.get_kernel(viennacl::linalg::opencl::kernels::vector_element<T>::program_name(), detail::op_to_string(OP()) + "_assign");

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
        linalg::opencl::kernels::vector<T>::init(ctx);
        viennacl::device_specific::enqueue(device_specific::database::get<T>(device_specific::database::reduction),
                                           ctx.get_program(linalg::opencl::kernels::vector<T>::program_name()),
                                           scheduler::preset::inner_prod(&result, &vec1, &vec2),
                                           device_specific::BIND_ALL_UNIQUE);
      }

      namespace detail
      {
        template <typename ScalarT>
        viennacl::ocl::packed_cl_uint make_layout(vector_base<ScalarT> const & vec)
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
      template <typename T>
      void inner_prod_impl(vector_base<T> const & x,
                           vector_tuple<T> const & vec_tuple,
                           vector_base<T> & result)
      {
        assert(viennacl::traits::opencl_handle(x).context() == viennacl::traits::opencl_handle(result).context() && bool("Operands do not reside in the same OpenCL context. Automatic migration not yet supported!"));

        viennacl::ocl::context & ctx = const_cast<viennacl::ocl::context &>(viennacl::traits::opencl_handle(x).context());
        viennacl::linalg::opencl::kernels::vector<T>::init(ctx);
        viennacl::linalg::opencl::kernels::vector_multi_inner_prod<T>::init(ctx);

        vcl_size_t work_groups = 128;

        viennacl::vector<T> temp(work_groups, viennacl::traits::context(x));
        temp.resize(8 * work_groups, ctx); // bring default-constructed vectors to the correct size:

        viennacl::ocl::packed_cl_uint layout_x = detail::make_layout(x);

        viennacl::ocl::kernel & ksum = ctx.get_kernel(viennacl::linalg::opencl::kernels::vector_multi_inner_prod<T>::program_name(), "sum_inner_prod");
        viennacl::ocl::kernel & inner_prod_kernel_1 = ctx.get_kernel(viennacl::linalg::opencl::kernels::vector<T>::program_name(), "inner_prod1");
        viennacl::ocl::kernel & inner_prod_kernel_2 = ctx.get_kernel(viennacl::linalg::opencl::kernels::vector_multi_inner_prod<T>::program_name(), "inner_prod2");
        viennacl::ocl::kernel & inner_prod_kernel_3 = ctx.get_kernel(viennacl::linalg::opencl::kernels::vector_multi_inner_prod<T>::program_name(), "inner_prod3");
        viennacl::ocl::kernel & inner_prod_kernel_4 = ctx.get_kernel(viennacl::linalg::opencl::kernels::vector_multi_inner_prod<T>::program_name(), "inner_prod4");
        viennacl::ocl::kernel & inner_prod_kernel_8 = ctx.get_kernel(viennacl::linalg::opencl::kernels::vector_multi_inner_prod<T>::program_name(), "inner_prod8");

        vcl_size_t current_index = 0;
        while (current_index < vec_tuple.const_size())
        {
          switch (vec_tuple.const_size() - current_index)
          {
            case 7:
            case 6:
            case 5:
            case 4:
            {
              vector_base<T> const & y0 = vec_tuple.const_at(current_index    );
              vector_base<T> const & y1 = vec_tuple.const_at(current_index + 1);
              vector_base<T> const & y2 = vec_tuple.const_at(current_index + 2);
              vector_base<T> const & y3 = vec_tuple.const_at(current_index + 3);
              viennacl::ocl::enqueue(inner_prod_kernel_4( viennacl::traits::opencl_handle(x), layout_x,
                                                         viennacl::traits::opencl_handle(y0), detail::make_layout(y0),
                                                         viennacl::traits::opencl_handle(y1), detail::make_layout(y1),
                                                         viennacl::traits::opencl_handle(y2), detail::make_layout(y2),
                                                         viennacl::traits::opencl_handle(y3), detail::make_layout(y3),
                                                         viennacl::ocl::local_mem(sizeof(typename viennacl::result_of::cl_type<T>::type) * 4 * inner_prod_kernel_4.local_work_size()),
                                                         viennacl::traits::opencl_handle(temp)
                                                        ) );

              ksum.local_work_size(0, work_groups);
              ksum.global_work_size(0, 4 * work_groups);
              viennacl::ocl::enqueue(ksum(viennacl::traits::opencl_handle(temp),
                                          viennacl::ocl::local_mem(sizeof(typename viennacl::result_of::cl_type<T>::type) * 4 * ksum.local_work_size()),
                                          viennacl::traits::opencl_handle(result),
                                          cl_uint(viennacl::traits::start(result) + current_index * viennacl::traits::stride(result)),
                                          cl_uint(viennacl::traits::stride(result))
                                          )
                                    );
            }
              current_index += 4;
              break;

            case 3:
            {
              vector_base<T> const & y0 = vec_tuple.const_at(current_index    );
              vector_base<T> const & y1 = vec_tuple.const_at(current_index + 1);
              vector_base<T> const & y2 = vec_tuple.const_at(current_index + 2);
              viennacl::ocl::enqueue(inner_prod_kernel_3( viennacl::traits::opencl_handle(x), layout_x,
                                                          viennacl::traits::opencl_handle(y0), detail::make_layout(y0),
                                                          viennacl::traits::opencl_handle(y1), detail::make_layout(y1),
                                                          viennacl::traits::opencl_handle(y2), detail::make_layout(y2),
                                                          viennacl::ocl::local_mem(sizeof(typename viennacl::result_of::cl_type<T>::type) * 3 * inner_prod_kernel_3.local_work_size()),
                                                          viennacl::traits::opencl_handle(temp)
                                                         ) );

              ksum.local_work_size(0, work_groups);
              ksum.global_work_size(0, 3 * work_groups);
              viennacl::ocl::enqueue(ksum(viennacl::traits::opencl_handle(temp),
                                          viennacl::ocl::local_mem(sizeof(typename viennacl::result_of::cl_type<T>::type) * 3 * ksum.local_work_size()),
                                          viennacl::traits::opencl_handle(result),
                                          cl_uint(viennacl::traits::start(result) + current_index * viennacl::traits::stride(result)),
                                          cl_uint(viennacl::traits::stride(result))
                                          )
                                    );
            }
              current_index += 3;
              break;

            case 2:
            {
              vector_base<T> const & y0 = vec_tuple.const_at(current_index    );
              vector_base<T> const & y1 = vec_tuple.const_at(current_index + 1);
              viennacl::ocl::enqueue(inner_prod_kernel_2( viennacl::traits::opencl_handle(x), layout_x,
                                                          viennacl::traits::opencl_handle(y0), detail::make_layout(y0),
                                                          viennacl::traits::opencl_handle(y1), detail::make_layout(y1),
                                                          viennacl::ocl::local_mem(sizeof(typename viennacl::result_of::cl_type<T>::type) * 2 * inner_prod_kernel_2.local_work_size()),
                                                          viennacl::traits::opencl_handle(temp)
                                                        ) );

              ksum.local_work_size(0, work_groups);
              ksum.global_work_size(0, 2 * work_groups);
              viennacl::ocl::enqueue(ksum(viennacl::traits::opencl_handle(temp),
                                          viennacl::ocl::local_mem(sizeof(typename viennacl::result_of::cl_type<T>::type) * 2 * ksum.local_work_size()),
                                          viennacl::traits::opencl_handle(result),
                                          cl_uint(viennacl::traits::start(result) + current_index * viennacl::traits::stride(result)),
                                          cl_uint(viennacl::traits::stride(result))
                                          )
                                    );
            }
              current_index += 2;
              break;

            case 1:
            {
              vector_base<T> const & y0 = vec_tuple.const_at(current_index    );
              viennacl::ocl::enqueue(inner_prod_kernel_1( viennacl::traits::opencl_handle(x), layout_x,
                                                          viennacl::traits::opencl_handle(y0), detail::make_layout(y0),
                                                          viennacl::ocl::local_mem(sizeof(typename viennacl::result_of::cl_type<T>::type) * 1 * inner_prod_kernel_1.local_work_size()),
                                                          viennacl::traits::opencl_handle(temp)
                                                        ) );

              ksum.local_work_size(0, work_groups);
              ksum.global_work_size(0, 1 * work_groups);
              viennacl::ocl::enqueue(ksum(viennacl::traits::opencl_handle(temp),
                                          viennacl::ocl::local_mem(sizeof(typename viennacl::result_of::cl_type<T>::type) * 1 * ksum.local_work_size()),
                                          viennacl::traits::opencl_handle(result),
                                          cl_uint(viennacl::traits::start(result) + current_index * viennacl::traits::stride(result)),
                                          cl_uint(viennacl::traits::stride(result))
                                          )
                                    );
            }
              current_index += 1;
              break;

            default: //8 or more vectors
            {
              vector_base<T> const & y0 = vec_tuple.const_at(current_index    );
              vector_base<T> const & y1 = vec_tuple.const_at(current_index + 1);
              vector_base<T> const & y2 = vec_tuple.const_at(current_index + 2);
              vector_base<T> const & y3 = vec_tuple.const_at(current_index + 3);
              vector_base<T> const & y4 = vec_tuple.const_at(current_index + 4);
              vector_base<T> const & y5 = vec_tuple.const_at(current_index + 5);
              vector_base<T> const & y6 = vec_tuple.const_at(current_index + 6);
              vector_base<T> const & y7 = vec_tuple.const_at(current_index + 7);
              viennacl::ocl::enqueue(inner_prod_kernel_8( viennacl::traits::opencl_handle(x), layout_x,
                                                          viennacl::traits::opencl_handle(y0), detail::make_layout(y0),
                                                          viennacl::traits::opencl_handle(y1), detail::make_layout(y1),
                                                          viennacl::traits::opencl_handle(y2), detail::make_layout(y2),
                                                          viennacl::traits::opencl_handle(y3), detail::make_layout(y3),
                                                          viennacl::traits::opencl_handle(y4), detail::make_layout(y4),
                                                          viennacl::traits::opencl_handle(y5), detail::make_layout(y5),
                                                          viennacl::traits::opencl_handle(y6), detail::make_layout(y6),
                                                          viennacl::traits::opencl_handle(y7), detail::make_layout(y7),
                                                          viennacl::ocl::local_mem(sizeof(typename viennacl::result_of::cl_type<T>::type) * 8 * inner_prod_kernel_8.local_work_size()),
                                                          viennacl::traits::opencl_handle(temp)
                                                        ) );

              ksum.local_work_size(0, work_groups);
              ksum.global_work_size(0, 8 * work_groups);
              viennacl::ocl::enqueue(ksum(viennacl::traits::opencl_handle(temp),
                                          viennacl::ocl::local_mem(sizeof(typename viennacl::result_of::cl_type<T>::type) * 8 * ksum.local_work_size()),
                                          viennacl::traits::opencl_handle(result),
                                          cl_uint(viennacl::traits::start(result) + current_index * viennacl::traits::stride(result)),
                                          cl_uint(viennacl::traits::stride(result))
                                          )
                                    );
            }
              current_index += 8;
              break;
          }
        }

      }


      template <typename T>
      void inner_prod_cpu(vector_base<T> const & vec1,
                          vector_base<T> const & vec2,
                          T & result)
      {
        scalar<T> tmp(0);
        inner_prod_impl(vec1, vec2, tmp);
        result = tmp;
      }

//      //implementation of inner product:
//      //namespace {
//      /** @brief Computes the inner product of two vectors - implementation. Library users should call inner_prod(vec1, vec2).
//      *
//      * @param vec1 The first vector
//      * @param vec2 The second vector
//      * @param result The result scalar (on the gpu)
//      */
//      template <typename T>
//      void inner_prod_cpu(vector_base<T> const & vec1,
//                          vector_base<T> const & vec2,
//                          T & result)
//      {
//        assert(viennacl::traits::opencl_handle(vec1).context() == viennacl::traits::opencl_handle(vec2).context() && bool("Vectors do not reside in the same OpenCL context. Automatic migration not yet supported!"));

//        viennacl::ocl::context & ctx = const_cast<viennacl::ocl::context &>(viennacl::traits::opencl_handle(vec1).context());

//        vcl_size_t work_groups = 128;
//        viennacl::vector<T> temp(work_groups, viennacl::traits::context(vec1));
//        temp.resize(work_groups, ctx); // bring default-constructed vectors to the correct size:

//        // Step 1: Compute partial inner products for each work group:
//        inner_prod_impl(vec1, vec2, temp);

//        // Step 2: Sum partial results:

//        // Now copy partial results from GPU back to CPU and run reduction there:
//        std::vector<T> temp_cpu(work_groups);
//        viennacl::fast_copy(temp.begin(), temp.end(), temp_cpu.begin());

//        result = 0;
//        for (typename std::vector<T>::const_iterator it = temp_cpu.begin(); it != temp_cpu.end(); ++it)
//          result += *it;
//      }


//      //////////// Helper for norms

//      /** @brief Computes the partial work group results for vector norms
//      *
//      * @param vec The vector
//      * @param partial_result The result scalar
//      * @param norm_id        Norm selector. 0: norm_inf, 1: norm_1, 2: norm_2
//      */
//      template <typename T>
//      void norm_reduction_impl(vector_base<T> const & vec,
//                               vector_base<T> & partial_result,
//                                cl_uint norm_id)
//      {
//        assert(viennacl::traits::opencl_handle(vec).context() == viennacl::traits::opencl_handle(partial_result).context() && bool("Operands do not reside in the same OpenCL context. Automatic migration not yet supported!"));

//        viennacl::ocl::context & ctx = const_cast<viennacl::ocl::context &>(viennacl::traits::opencl_handle(vec).context());
//        viennacl::linalg::opencl::kernels::vector<T>::init(ctx);

//        viennacl::ocl::kernel & k = ctx.get_kernel(viennacl::linalg::opencl::kernels::vector<T>::program_name(), "norm");

//        assert( (k.global_work_size() / k.local_work_size() <= partial_result.size()) && bool("Size mismatch for partial reduction in norm_reduction_impl()") );

//        viennacl::ocl::enqueue(k(viennacl::traits::opencl_handle(vec),
//                                 cl_uint(viennacl::traits::start(vec)),
//                                 cl_uint(viennacl::traits::stride(vec)),
//                                 cl_uint(viennacl::traits::size(vec)),
//                                 cl_uint(norm_id),
//                                 viennacl::ocl::local_mem(sizeof(typename viennacl::result_of::cl_type<T>::type) * k.local_work_size()),
//                                 viennacl::traits::opencl_handle(partial_result) )
//                              );
//      }


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
        linalg::opencl::kernels::vector<T>::init(ctx);
        viennacl::device_specific::enqueue(device_specific::database::get<T>(device_specific::database::reduction),
                                           ctx.get_program(linalg::opencl::kernels::vector<T>::program_name()),
                                           scheduler::preset::norm_1(&result, &vec));
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
        scalar<T> tmp(0);
        norm_1_impl(vec, tmp);
        result = tmp;
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
        linalg::opencl::kernels::vector<T>::init(ctx);
        viennacl::device_specific::enqueue(device_specific::database::get<T>(device_specific::database::reduction),
                                           ctx.get_program(linalg::opencl::kernels::vector<T>::program_name()),
                                           scheduler::preset::norm_2(&result, &vec));
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
        scalar<T> tmp(0);
        norm_2_impl(vec, tmp);
        result = tmp;
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
        linalg::opencl::kernels::vector<T>::init(ctx);
        viennacl::device_specific::enqueue(device_specific::database::get<T>(device_specific::database::reduction),
                                           ctx.get_program(linalg::opencl::kernels::vector<T>::program_name()),
                                           scheduler::preset::norm_inf(&result, &vec));
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
        scalar<T> tmp(0);
        norm_inf_impl(vec, tmp);
        result = tmp;
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
        viennacl::scalar<T> result(0);
        viennacl::ocl::context & ctx = const_cast<viennacl::ocl::context &>(viennacl::traits::opencl_handle(vec).context());
        linalg::opencl::kernels::vector<T>::init(ctx);
        viennacl::device_specific::enqueue(device_specific::database::get<T>(device_specific::database::reduction),
                                           ctx.get_program(linalg::opencl::kernels::vector<T>::program_name()),
                                           scheduler::preset::index_norm_inf(&result, &vec));
        T host_result = result;
        return host_result;
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
        assert(viennacl::traits::size(vec1) == viennacl::traits::size(vec2));

        viennacl::ocl::context & ctx = const_cast<viennacl::ocl::context &>(viennacl::traits::opencl_handle(vec1).context());
        viennacl::linalg::opencl::kernels::vector<T>::init(ctx);
        device_specific::enqueue(device_specific::database::get<T>(device_specific::database::axpy),
                                 ctx.get_program(viennacl::linalg::opencl::kernels::vector<T>::program_name()),
                                 scheduler::preset::plane_rotation(&vec1, &vec2, &alpha, &beta));
      }

    } //namespace opencl
  } //namespace linalg
} //namespace viennacl


#endif
