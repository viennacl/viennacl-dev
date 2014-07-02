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
#include "viennacl/device_specific/templates/vector_axpy_template.hpp"
#include "viennacl/device_specific/templates/reduction_template.hpp"
#include "viennacl/device_specific/database.hpp"
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
              vector_base<T> const & vec2, ScalarType1 const & alpha, vcl_size_t /* len_alpha */, bool reciprocal_alpha, bool flip_sign_alpha)
      {
        assert(viennacl::traits::opencl_handle(vec1).context() == viennacl::traits::opencl_handle(vec2).context() && bool("Vectors do not reside in the same OpenCL context. Automatic migration not yet supported!"));

        viennacl::ocl::context & ctx = const_cast<viennacl::ocl::context &>(viennacl::traits::opencl_handle(vec1).context());
        viennacl::linalg::opencl::kernels::vector<T>::init(ctx);
        device_specific::vector_axpy_template(device_specific::database::get<T>(device_specific::database::vector_axpy))
                                        .enqueue(linalg::opencl::kernels::vector<T>::program_name(),
                                                 scheduler::preset::avbv(scheduler::OPERATION_BINARY_ASSIGN_TYPE, &vec1, &vec2, &alpha, flip_sign_alpha, reciprocal_alpha, (vector_base<T>*)NULL, (T*)NULL, false, false));
      }


      template <typename T, typename ScalarType1, typename ScalarType2>
      void avbv(vector_base<T> & vec1,
                vector_base<T> const & vec2, ScalarType1 const & alpha, vcl_size_t /* len_alpha */, bool reciprocal_alpha, bool flip_sign_alpha,
                vector_base<T> const & vec3, ScalarType2 const & beta,  vcl_size_t /* len_beta */,  bool reciprocal_beta,  bool flip_sign_beta)
      {
        assert(viennacl::traits::opencl_handle(vec1).context() == viennacl::traits::opencl_handle(vec2).context() && bool("Vectors do not reside in the same OpenCL context. Automatic migration not yet supported!"));
        assert(viennacl::traits::opencl_handle(vec2).context() == viennacl::traits::opencl_handle(vec3).context() && bool("Vectors do not reside in the same OpenCL context. Automatic migration not yet supported!"));

        viennacl::ocl::context & ctx = const_cast<viennacl::ocl::context &>(viennacl::traits::opencl_handle(vec1).context());
        viennacl::linalg::opencl::kernels::vector<T>::init(ctx);

        device_specific::vector_axpy_template(device_specific::database::get<T>(device_specific::database::vector_axpy))
                                        .enqueue(linalg::opencl::kernels::vector<T>::program_name(),
                                        scheduler::preset::avbv(scheduler::OPERATION_BINARY_ASSIGN_TYPE, &vec1, &vec2, &alpha, flip_sign_alpha, reciprocal_alpha, &vec3, &beta, flip_sign_beta, reciprocal_beta));
      }


      template <typename T, typename ScalarType1, typename ScalarType2>
      void avbv_v(vector_base<T> & vec1,
                  vector_base<T> const & vec2, ScalarType1 const & alpha, vcl_size_t /* len_alpha */, bool reciprocal_alpha, bool flip_sign_alpha,
                  vector_base<T> const & vec3, ScalarType2 const & beta,  vcl_size_t /* len_beta */,  bool reciprocal_beta,  bool flip_sign_beta)
      {
        assert(viennacl::traits::opencl_handle(vec1).context() == viennacl::traits::opencl_handle(vec2).context() && bool("Vectors do not reside in the same OpenCL context. Automatic migration not yet supported!"));
        assert(viennacl::traits::opencl_handle(vec2).context() == viennacl::traits::opencl_handle(vec3).context() && bool("Vectors do not reside in the same OpenCL context. Automatic migration not yet supported!"));

        viennacl::ocl::context & ctx = const_cast<viennacl::ocl::context &>(viennacl::traits::opencl_handle(vec1).context());
        viennacl::linalg::opencl::kernels::vector<T>::init(ctx);
        device_specific::vector_axpy_template(device_specific::database::get<T>(device_specific::database::vector_axpy))
                                        .enqueue(linalg::opencl::kernels::vector<T>::program_name(),
                                          scheduler::preset::avbv(scheduler::OPERATION_BINARY_INPLACE_ADD_TYPE, &vec1, &vec2, &alpha, flip_sign_alpha, reciprocal_alpha, &vec3, &beta, flip_sign_beta, reciprocal_beta));
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
        device_specific::vector_axpy_template(device_specific::database::get<T>(device_specific::database::vector_axpy))
                                        .enqueue(linalg::opencl::kernels::vector<T>::program_name(), scheduler::preset::assign_cpu(&vec1, &vec2), up_to_internal_size);
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
        device_specific::vector_axpy_template(device_specific::database::get<T>(device_specific::database::vector_axpy))
                                        .enqueue(linalg::opencl::kernels::vector<T>::program_name(), scheduler::preset::swap(&vec1, &vec2));
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
        scheduler::operation_node_type TYPE = scheduler::operation_node_type(scheduler::result_of::op_type_info<op_element_binary<OP> >::id);
        device_specific::vector_axpy_template(device_specific::database::get<T>(device_specific::database::vector_axpy))
                                        .enqueue(linalg::opencl::kernels::vector_element<T>::program_name(),scheduler::preset::binary_element_op(&vec1, &proxy.lhs(), &proxy.rhs(),TYPE));
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
        scheduler::operation_node_type TYPE = scheduler::operation_node_type(scheduler::result_of::op_type_info<op_element_unary<OP> >::id);
        device_specific::vector_axpy_template(device_specific::database::get<T>(device_specific::database::vector_axpy))
                                        .enqueue(linalg::opencl::kernels::vector_element<T>::program_name(),scheduler::preset::unary_element_op(&vec1, &proxy.lhs(),TYPE));

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
        device_specific::reduction_template(device_specific::database::get<T>(device_specific::database::reduction))
                                        .enqueue(linalg::opencl::kernels::vector<T>::program_name(), scheduler::preset::inner_prod(&result, &vec1, &vec2));
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

        typedef viennacl::vector_range< viennacl::vector_base<T> > range_t;



        vcl_size_t current_index = 0;
        while (current_index < vec_tuple.const_size())
        {
          device_specific::statements_container::data_type statements;

          vcl_size_t diff = vec_tuple.const_size() - current_index;
          vcl_size_t upper_bound;
          if(diff>=8) upper_bound = 8;
          else if(diff>=4) upper_bound = 4;
          else if(diff>=3) upper_bound = 3;
          else if(diff>=2) upper_bound = 2;
          else upper_bound = 1;

          std::vector<range_t> ranges;
          ranges.reserve(upper_bound);
          for(unsigned int i = 0 ; i < upper_bound ; ++i)
            ranges.push_back(range_t(result, viennacl::range(current_index+i, current_index+i+1)));

          for(unsigned int i = 0 ; i < upper_bound ; ++i)
            statements.push_back(scheduler::preset::inner_prod(&ranges[i], &x, &vec_tuple.const_at(current_index+i)));

          device_specific::reduction_template(device_specific::database::get<T>(device_specific::database::reduction))
                                          .enqueue(linalg::opencl::kernels::vector_multi_inner_prod<T>::program_name(), device_specific::statements_container(statements,device_specific::statements_container::INDEPENDENT));
          current_index += upper_bound;
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
        device_specific::reduction_template(device_specific::database::get<T>(device_specific::database::reduction))
                                        .enqueue(linalg::opencl::kernels::vector<T>::program_name(), scheduler::preset::norm_1(&result, &vec));
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
        device_specific::reduction_template(device_specific::database::get<T>(device_specific::database::reduction), device_specific::BIND_TO_HANDLE)
                                        .enqueue(linalg::opencl::kernels::vector<T>::program_name(), scheduler::preset::norm_2(&result, &vec));
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
        device_specific::reduction_template(device_specific::database::get<T>(device_specific::database::reduction))
                                        .enqueue(linalg::opencl::kernels::vector<T>::program_name(), scheduler::preset::norm_inf(&result, &vec));
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
        device_specific::reduction_template(device_specific::database::get<T>(device_specific::database::reduction))
                                        .enqueue(linalg::opencl::kernels::vector<T>::program_name(), scheduler::preset::index_norm_inf(&result, &vec));
        T host_result = result;
        return static_cast<cl_uint>(host_result);
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
        device_specific::vector_axpy_template(device_specific::database::get<T>(device_specific::database::vector_axpy))
                                        .enqueue(linalg::opencl::kernels::vector<T>::program_name(), scheduler::preset::plane_rotation(&vec1, &vec2, &alpha, &beta));
      }

    } //namespace opencl
  } //namespace linalg
} //namespace viennacl


#endif
