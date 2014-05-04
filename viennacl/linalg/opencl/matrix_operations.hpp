#ifndef VIENNACL_LINALG_OPENCL_MATRIX_OPERATIONS_HPP_
#define VIENNACL_LINALG_OPENCL_MATRIX_OPERATIONS_HPP_

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

/** @file  viennacl/linalg/opencl/matrix_operations.hpp
    @brief Implementations of dense matrix related operations, including matrix-vector products, using OpenCL.
*/

#include "viennacl/forwards.h"
#include "viennacl/ocl/device.hpp"
#include "viennacl/ocl/handle.hpp"
#include "viennacl/ocl/kernel.hpp"
#include "viennacl/scalar.hpp"
#include "viennacl/vector.hpp"
#include "viennacl/vector_proxy.hpp"
#include "viennacl/tools/tools.hpp"
#include "viennacl/meta/enable_if.hpp"
#include "viennacl/meta/predicate.hpp"
#include "viennacl/meta/result_of.hpp"

#include "viennacl/scheduler/forwards.h"

#include "viennacl/device_specific/code_generator.hpp"

#include "viennacl/traits/size.hpp"
#include "viennacl/traits/start.hpp"
#include "viennacl/traits/handle.hpp"
#include "viennacl/traits/stride.hpp"

#include "viennacl/linalg/opencl/common.hpp"

#include "viennacl/linalg/opencl/kernels/matrix.hpp"
#include "viennacl/linalg/opencl/kernels/matrix_element.hpp"

#include "viennacl/linalg/opencl/kernels/matrix_prod.hpp"


namespace viennacl
{
  namespace linalg
  {
    namespace opencl
    {

      //
      // Introductory note: By convention, all dimensions are already checked in the dispatcher frontend. No need to double-check again in here!
      //

      template <typename NumericT,
                typename ScalarType1>
      void am(matrix_base<NumericT> & mat1,
              matrix_base<NumericT> const & mat2, ScalarType1 const & alpha, vcl_size_t len_alpha, bool reciprocal_alpha, bool flip_sign_alpha)
      {
        assert(mat1.row_major() == mat2.row_major() && bool("Addition/subtraction on mixed matrix layouts not supported yet!"));

        typedef NumericT        value_type;

        cl_uint options_alpha = detail::make_options(len_alpha, reciprocal_alpha, flip_sign_alpha);

        viennacl::ocl::kernel & k = detail::kernel_for_matrix(mat1, viennacl::is_cpu_scalar<ScalarType1>::value ? "am_cpu" : "am_gpu");
        viennacl::ocl::enqueue(k(viennacl::traits::opencl_handle(mat1),
                                cl_uint(viennacl::traits::start1(mat1)),           cl_uint(viennacl::traits::start2(mat1)),
                                cl_uint(viennacl::traits::stride1(mat1)),          cl_uint(viennacl::traits::stride2(mat1)),
                                cl_uint(viennacl::traits::size1(mat1)),            cl_uint(viennacl::traits::size2(mat1)),
                                cl_uint(viennacl::traits::internal_size1(mat1)),   cl_uint(viennacl::traits::internal_size2(mat1)),

                                viennacl::traits::opencl_handle(viennacl::tools::promote_if_host_scalar<value_type>(alpha)),
                                options_alpha,
                                viennacl::traits::opencl_handle(mat2),
                                cl_uint(viennacl::traits::start1(mat2)),           cl_uint(viennacl::traits::start2(mat2)),
                                cl_uint(viennacl::traits::stride1(mat2)),          cl_uint(viennacl::traits::stride2(mat2)),
                                cl_uint(viennacl::traits::internal_size1(mat2)),   cl_uint(viennacl::traits::internal_size2(mat2))
                                )
                              );
      }


      template <typename NumericT,
                typename ScalarType1, typename ScalarType2>
      void ambm(matrix_base<NumericT> & mat1,
                matrix_base<NumericT> const & mat2, ScalarType1 const & alpha, vcl_size_t len_alpha, bool reciprocal_alpha, bool flip_sign_alpha,
                matrix_base<NumericT> const & mat3, ScalarType2 const & beta,  vcl_size_t len_beta,  bool reciprocal_beta,  bool flip_sign_beta)
      {
        assert(mat1.row_major() == mat2.row_major() && mat1.row_major() == mat3.row_major() && bool("Addition/subtraction on mixed matrix layouts not supported yet!"));

        typedef NumericT        value_type;

        std::string kernel_name;
        if      ( viennacl::is_cpu_scalar<ScalarType1>::value &&  viennacl::is_cpu_scalar<ScalarType2>::value)
          kernel_name = "ambm_cpu_cpu";
        else if ( viennacl::is_cpu_scalar<ScalarType1>::value && !viennacl::is_cpu_scalar<ScalarType2>::value)
          kernel_name = "ambm_cpu_gpu";
        else if (!viennacl::is_cpu_scalar<ScalarType1>::value &&  viennacl::is_cpu_scalar<ScalarType2>::value)
          kernel_name = "ambm_gpu_cpu";
        else
          kernel_name = "ambm_gpu_gpu";

        cl_uint options_alpha = detail::make_options(len_alpha, reciprocal_alpha, flip_sign_alpha);
        cl_uint options_beta  = detail::make_options(len_beta,  reciprocal_beta,  flip_sign_beta);

        viennacl::ocl::kernel & k = detail::kernel_for_matrix(mat1, kernel_name);
        viennacl::ocl::enqueue(k(viennacl::traits::opencl_handle(mat1),
                                cl_uint(viennacl::traits::start1(mat1)),           cl_uint(viennacl::traits::start2(mat1)),
                                cl_uint(viennacl::traits::stride1(mat1)),          cl_uint(viennacl::traits::stride2(mat1)),
                                cl_uint(viennacl::traits::size1(mat1)),            cl_uint(viennacl::traits::size2(mat1)),
                                cl_uint(viennacl::traits::internal_size1(mat1)),   cl_uint(viennacl::traits::internal_size2(mat1)),

                                viennacl::traits::opencl_handle(viennacl::tools::promote_if_host_scalar<value_type>(alpha)),
                                options_alpha,
                                viennacl::traits::opencl_handle(mat2),
                                cl_uint(viennacl::traits::start1(mat2)),           cl_uint(viennacl::traits::start2(mat2)),
                                cl_uint(viennacl::traits::stride1(mat2)),          cl_uint(viennacl::traits::stride2(mat2)),
                                cl_uint(viennacl::traits::internal_size1(mat2)),   cl_uint(viennacl::traits::internal_size2(mat2)),

                                viennacl::traits::opencl_handle(viennacl::tools::promote_if_host_scalar<value_type>(beta)),
                                options_beta,
                                viennacl::traits::opencl_handle(mat3),
                                cl_uint(viennacl::traits::start1(mat3)),           cl_uint(viennacl::traits::start2(mat3)),
                                cl_uint(viennacl::traits::stride1(mat3)),          cl_uint(viennacl::traits::stride2(mat3)),
                                cl_uint(viennacl::traits::internal_size1(mat3)),   cl_uint(viennacl::traits::internal_size2(mat3))
                                )
                              );
      }


      template <typename NumericT,
                typename ScalarType1, typename ScalarType2>
      void ambm_m(matrix_base<NumericT> & mat1,
                  matrix_base<NumericT> const & mat2, ScalarType1 const & alpha, vcl_size_t len_alpha, bool reciprocal_alpha, bool flip_sign_alpha,
                  matrix_base<NumericT> const & mat3, ScalarType2 const & beta,  vcl_size_t len_beta,  bool reciprocal_beta,  bool flip_sign_beta)
      {
        assert(mat1.row_major() == mat2.row_major() && mat1.row_major() == mat3.row_major() && bool("Addition/subtraction on mixed matrix layouts not supported yet!"));

        typedef NumericT        value_type;

        std::string kernel_name;
        if      ( viennacl::is_cpu_scalar<ScalarType1>::value &&  viennacl::is_cpu_scalar<ScalarType2>::value)
          kernel_name = "ambm_m_cpu_cpu";
        else if ( viennacl::is_cpu_scalar<ScalarType1>::value && !viennacl::is_cpu_scalar<ScalarType2>::value)
          kernel_name = "ambm_m_cpu_gpu";
        else if (!viennacl::is_cpu_scalar<ScalarType1>::value &&  viennacl::is_cpu_scalar<ScalarType2>::value)
          kernel_name = "ambm_m_gpu_cpu";
        else
          kernel_name = "ambm_m_gpu_gpu";

        cl_uint options_alpha = detail::make_options(len_alpha, reciprocal_alpha, flip_sign_alpha);
        cl_uint options_beta  = detail::make_options(len_beta,  reciprocal_beta,  flip_sign_beta);

        viennacl::ocl::kernel & k = detail::kernel_for_matrix(mat1, kernel_name);
        viennacl::ocl::enqueue(k(viennacl::traits::opencl_handle(mat1),
                                cl_uint(viennacl::traits::start1(mat1)),           cl_uint(viennacl::traits::start2(mat1)),
                                cl_uint(viennacl::traits::stride1(mat1)),          cl_uint(viennacl::traits::stride2(mat1)),
                                cl_uint(viennacl::traits::size1(mat1)),            cl_uint(viennacl::traits::size2(mat1)),
                                cl_uint(viennacl::traits::internal_size1(mat1)),   cl_uint(viennacl::traits::internal_size2(mat1)),

                                viennacl::traits::opencl_handle(viennacl::tools::promote_if_host_scalar<value_type>(alpha)),
                                options_alpha,
                                viennacl::traits::opencl_handle(mat2),
                                cl_uint(viennacl::traits::start1(mat2)),           cl_uint(viennacl::traits::start2(mat2)),
                                cl_uint(viennacl::traits::stride1(mat2)),          cl_uint(viennacl::traits::stride2(mat2)),
                                cl_uint(viennacl::traits::internal_size1(mat2)),   cl_uint(viennacl::traits::internal_size2(mat2)),

                                viennacl::traits::opencl_handle(viennacl::tools::promote_if_host_scalar<value_type>(beta)),
                                options_beta,
                                viennacl::traits::opencl_handle(mat3),
                                cl_uint(viennacl::traits::start1(mat3)),           cl_uint(viennacl::traits::start2(mat3)),
                                cl_uint(viennacl::traits::stride1(mat3)),          cl_uint(viennacl::traits::stride2(mat3)),
                                cl_uint(viennacl::traits::internal_size1(mat3)),   cl_uint(viennacl::traits::internal_size2(mat3))
                                )
                              );
      }



      template <typename NumericT>
      void matrix_assign(matrix_base<NumericT> & mat, NumericT s, bool clear = false)
      {
        typedef NumericT        value_type;

        value_type alpha = static_cast<value_type>(s);

        cl_uint s1 = clear ? cl_uint(viennacl::traits::internal_size1(mat)) : cl_uint(viennacl::traits::size1(mat));
        cl_uint s2 = clear ? cl_uint(viennacl::traits::internal_size2(mat)) : cl_uint(viennacl::traits::size2(mat));

        viennacl::ocl::kernel & k = detail::kernel_for_matrix(mat, "assign_cpu");
        viennacl::ocl::enqueue(k(viennacl::traits::opencl_handle(mat),
                                 cl_uint(viennacl::traits::start1(mat)),           cl_uint(viennacl::traits::start2(mat)),
                                 cl_uint(viennacl::traits::stride1(mat)),          cl_uint(viennacl::traits::stride2(mat)),
                                 s1,                                               s2,
                                 cl_uint(viennacl::traits::internal_size1(mat)),   cl_uint(viennacl::traits::internal_size2(mat)),
                                 viennacl::traits::opencl_handle(viennacl::tools::promote_if_host_scalar<value_type>(alpha))
                                )
                              );
      }

      template <typename NumericT>
      void matrix_diagonal_assign(matrix_base<NumericT> & mat, NumericT s)
      {
        typedef NumericT        value_type;

        value_type alpha = static_cast<value_type>(s);

        viennacl::ocl::kernel & k = detail::kernel_for_matrix(mat, "diagonal_assign_cpu");
        viennacl::ocl::enqueue(k(viennacl::traits::opencl_handle(mat),
                                 cl_uint(viennacl::traits::start1(mat)),           cl_uint(viennacl::traits::start2(mat)),
                                 cl_uint(viennacl::traits::stride1(mat)),          cl_uint(viennacl::traits::stride2(mat)),
                                 cl_uint(viennacl::traits::size1(mat)),            cl_uint(viennacl::traits::size2(mat)),
                                 cl_uint(viennacl::traits::internal_size1(mat)),   cl_uint(viennacl::traits::internal_size2(mat)),
                                 viennacl::traits::opencl_handle(viennacl::tools::promote_if_host_scalar<value_type>(alpha))
                                )
                              );
      }

      template <typename NumericT>
      void matrix_diag_from_vector(const vector_base<NumericT> & vec, int k, matrix_base<NumericT> & mat)
      {
        // Step 1: set everything to zero
        matrix_assign(mat, NumericT(0));

        // Step 2: set the diagonal:

        // reuse vector ambm kernel for assigning the elements:
        viennacl::ocl::context & ctx = const_cast<viennacl::ocl::context &>(viennacl::traits::opencl_handle(mat).context());
        typedef viennacl::linalg::opencl::kernels::vector<NumericT>  KernelClass;
        KernelClass::init(ctx);

        cl_uint options_alpha = 0;
        viennacl::ocl::packed_cl_uint size_mat;
        if (mat.row_major())
        {
          vcl_size_t first_row_index = 0;
          vcl_size_t first_col_index = 0;
          if (k < 0)
            first_row_index = vcl_size_t(-k);
          else
            first_col_index = vcl_size_t(k);
          size_mat.start  = cl_uint( (viennacl::traits::start1(mat) + first_row_index * viennacl::traits::stride1(mat)) * viennacl::traits::internal_size2(mat)
                                    + viennacl::traits::start2(mat) + first_col_index * viennacl::traits::stride2(mat));
          size_mat.stride = cl_uint(viennacl::traits::stride1(mat) * viennacl::traits::internal_size2(mat) + viennacl::traits::stride2(mat));
          size_mat.size   = cl_uint(viennacl::traits::size(vec));
          size_mat.internal_size   = cl_uint(viennacl::traits::internal_size(vec));
        }
        else
        {
          vcl_size_t first_row_index = 0;
          vcl_size_t first_col_index = 0;
          if (k < 0)
            first_row_index = vcl_size_t(-k);
          else
            first_col_index = vcl_size_t(k);
          size_mat.start  = cl_uint(   viennacl::traits::start1(mat) + first_row_index * viennacl::traits::stride1(mat)
                                    + (viennacl::traits::start2(mat) + first_col_index * viennacl::traits::stride2(mat)) * viennacl::traits::internal_size1(mat));
          size_mat.stride = cl_uint(viennacl::traits::stride2(mat) * viennacl::traits::internal_size1(mat) + viennacl::traits::stride1(mat));
          size_mat.size   = cl_uint(viennacl::traits::size(vec));
          size_mat.internal_size   = cl_uint(viennacl::traits::internal_size(vec));
        }

        viennacl::ocl::packed_cl_uint size_vec;
        size_vec.start  = cl_uint(viennacl::traits::start(vec));
        size_vec.stride = cl_uint(viennacl::traits::stride(vec));
        size_vec.size   = cl_uint(viennacl::traits::size(vec));
        size_vec.internal_size   = cl_uint(viennacl::traits::internal_size(vec));

        viennacl::ocl::kernel & kern = ctx.get_kernel(KernelClass::program_name(), "av_cpu");
        viennacl::ocl::enqueue(kern(viennacl::traits::opencl_handle(mat),
                                    size_mat,

                                    viennacl::traits::opencl_handle(NumericT(1)),
                                    options_alpha,
                                    viennacl::traits::opencl_handle(vec),
                                    size_vec)
                              );
      }

      template <typename NumericT>
      void matrix_diag_to_vector(const matrix_base<NumericT> & mat, int k, vector_base<NumericT> & vec)
      {
        // reuse vector ambm kernel for assigning the elements:
        viennacl::ocl::context & ctx = const_cast<viennacl::ocl::context &>(viennacl::traits::opencl_handle(mat).context());
        typedef viennacl::linalg::opencl::kernels::vector<NumericT>  KernelClass;
        KernelClass::init(ctx);

        cl_uint options_alpha = 0;
        viennacl::ocl::packed_cl_uint size_mat;
        if (mat.row_major())
        {
          vcl_size_t first_row_index = 0;
          vcl_size_t first_col_index = 0;
          if (k < 0)
            first_row_index = vcl_size_t(-k);
          else
            first_col_index = vcl_size_t(k);
          size_mat.start  = cl_uint( (viennacl::traits::start1(mat) + first_row_index * viennacl::traits::stride1(mat)) * viennacl::traits::internal_size2(mat)
                                    + viennacl::traits::start2(mat) + first_col_index * viennacl::traits::stride2(mat));
          size_mat.stride = cl_uint(viennacl::traits::stride1(mat) * viennacl::traits::internal_size2(mat) + viennacl::traits::stride2(mat));
          size_mat.size   = cl_uint(viennacl::traits::size(vec));
          size_mat.internal_size   = cl_uint(viennacl::traits::internal_size(vec));
        }
        else
        {
          vcl_size_t first_row_index = 0;
          vcl_size_t first_col_index = 0;
          if (k < 0)
            first_row_index = vcl_size_t(-k);
          else
            first_col_index = vcl_size_t(k);
          size_mat.start  = cl_uint(   viennacl::traits::start1(mat) + first_row_index * viennacl::traits::stride1(mat)
                                    + (viennacl::traits::start2(mat) + first_col_index * viennacl::traits::stride2(mat)) * viennacl::traits::internal_size1(mat));
          size_mat.stride = cl_uint(viennacl::traits::stride2(mat) * viennacl::traits::internal_size1(mat) + viennacl::traits::stride1(mat));
          size_mat.size   = cl_uint(viennacl::traits::size(vec));
          size_mat.internal_size   = cl_uint(viennacl::traits::internal_size(vec));
        }

        viennacl::ocl::packed_cl_uint size_vec;
        size_vec.start  = cl_uint(viennacl::traits::start(vec));
        size_vec.stride = cl_uint(viennacl::traits::stride(vec));
        size_vec.size   = cl_uint(viennacl::traits::size(vec));
        size_vec.internal_size   = cl_uint(viennacl::traits::internal_size(vec));


        viennacl::ocl::kernel & kern = ctx.get_kernel(KernelClass::program_name(), "av_cpu");
        viennacl::ocl::enqueue(kern(viennacl::traits::opencl_handle(vec),
                                    size_vec,

                                    viennacl::traits::opencl_handle(NumericT(1)),
                                    options_alpha,
                                    viennacl::traits::opencl_handle(mat),
                                    size_mat)
                              );
      }

      template <typename NumericT>
      void matrix_row(const matrix_base<NumericT> & mat, unsigned int i, vector_base<NumericT> & vec)
      {
        // reuse vector ambm kernel for assigning the elements:
        viennacl::ocl::context & ctx = const_cast<viennacl::ocl::context &>(viennacl::traits::opencl_handle(mat).context());
        typedef viennacl::linalg::opencl::kernels::vector<NumericT>  KernelClass;
        KernelClass::init(ctx);

        cl_uint options_alpha = 0;
        viennacl::ocl::packed_cl_uint size_mat;
        if (mat.row_major())
        {
          size_mat.start  = cl_uint((viennacl::traits::start1(mat) + i * viennacl::traits::stride1(mat)) * viennacl::traits::internal_size2(mat) + viennacl::traits::start2(mat));
          size_mat.stride = cl_uint(viennacl::traits::stride2(mat));
          size_mat.size   = cl_uint(viennacl::traits::size(vec));
          size_mat.internal_size   = cl_uint(viennacl::traits::internal_size(vec));
        }
        else
        {
          size_mat.start  = cl_uint((viennacl::traits::start1(mat) + i * viennacl::traits::stride1(mat)) + viennacl::traits::start2(mat) * viennacl::traits::internal_size1(mat));
          size_mat.stride = cl_uint(viennacl::traits::stride2(mat) * viennacl::traits::internal_size1(mat));
          size_mat.size   = cl_uint(viennacl::traits::size(vec));
          size_mat.internal_size   = cl_uint(viennacl::traits::internal_size(vec));
        }

        viennacl::ocl::packed_cl_uint size_vec;
        size_vec.start  = cl_uint(viennacl::traits::start(vec));
        size_vec.stride = cl_uint(viennacl::traits::stride(vec));
        size_vec.size   = cl_uint(viennacl::traits::size(vec));
        size_vec.internal_size   = cl_uint(viennacl::traits::internal_size(vec));


        viennacl::ocl::kernel & kern = ctx.get_kernel(KernelClass::program_name(), "av_cpu");
        viennacl::ocl::enqueue(kern(viennacl::traits::opencl_handle(vec),
                                    size_vec,

                                    viennacl::traits::opencl_handle(NumericT(1)),
                                    options_alpha,
                                    viennacl::traits::opencl_handle(mat),
                                    size_mat)
                              );
      }

      template <typename NumericT>
      void matrix_column(const matrix_base<NumericT> & mat, unsigned int j, vector_base<NumericT> & vec)
      {
        // reuse vector ambm kernel for assigning the elements:
        viennacl::ocl::context & ctx = const_cast<viennacl::ocl::context &>(viennacl::traits::opencl_handle(mat).context());
        typedef viennacl::linalg::opencl::kernels::vector<NumericT>  KernelClass;
        KernelClass::init(ctx);

        cl_uint options_alpha = 0;
        viennacl::ocl::packed_cl_uint size_mat;
        if (mat.row_major())
        {
          size_mat.start  = cl_uint(viennacl::traits::start1(mat) * viennacl::traits::internal_size2(mat) + viennacl::traits::start2(mat) + j * viennacl::traits::stride2(mat));
          size_mat.stride = cl_uint(viennacl::traits::stride2(mat) * viennacl::traits::internal_size2(mat));
          size_mat.size   = cl_uint(viennacl::traits::size(vec));
          size_mat.internal_size   = cl_uint(viennacl::traits::internal_size(vec));
        }
        else
        {
          size_mat.start  = cl_uint(viennacl::traits::start1(mat) + (viennacl::traits::start2(mat) + j * viennacl::traits::stride2(mat)) * viennacl::traits::internal_size1(mat));
          size_mat.stride = cl_uint(viennacl::traits::stride2(mat));
          size_mat.size   = cl_uint(viennacl::traits::size(vec));
          size_mat.internal_size   = cl_uint(viennacl::traits::internal_size(vec));
        }

        viennacl::ocl::packed_cl_uint size_vec;
        size_vec.start  = cl_uint(viennacl::traits::start(vec));
        size_vec.stride = cl_uint(viennacl::traits::stride(vec));
        size_vec.size   = cl_uint(viennacl::traits::size(vec));
        size_vec.internal_size   = cl_uint(viennacl::traits::internal_size(vec));


        viennacl::ocl::kernel & kern = ctx.get_kernel(KernelClass::program_name(), "av_cpu");
        viennacl::ocl::enqueue(kern(viennacl::traits::opencl_handle(vec),
                                    size_vec,

                                    viennacl::traits::opencl_handle(NumericT(1)),
                                    options_alpha,
                                    viennacl::traits::opencl_handle(mat),
                                    size_mat)
                              );
      }


      //
      ///////////////////////// Element-wise operation //////////////////////////////////
      //

      // Binary operations A = B .* C and A = B ./ C
      /** @brief Implementation of binary element-wise operations A = OP(B,C)
      *
      * @param A      The result matrix (or -range, or -slice)
      * @param proxy  The proxy object holding B, C, and the operation
      */
      template <typename T, typename OP>
      void element_op(matrix_base<T> & A,
                      matrix_expression<const matrix_base<T>, const matrix_base<T>, op_element_binary<OP> > const & proxy)
      {
        assert(A.row_major() == proxy.lhs().row_major() && bool("Elementwise operations on mixed matrix layouts not supported yet!"));
        assert(A.row_major() == proxy.rhs().row_major() && bool("Elementwise operations on mixed matrix layouts not supported yet!"));
        assert(viennacl::traits::opencl_handle(A).context() == viennacl::traits::opencl_handle(proxy.lhs()).context() && bool("Matrices do not reside in the same OpenCL context. Automatic migration not yet supported!"));
        assert(viennacl::traits::opencl_handle(A).context() == viennacl::traits::opencl_handle(proxy.rhs()).context() && bool("Matrices do not reside in the same OpenCL context. Automatic migration not yet supported!"));

        cl_uint op_type = 2; //0: product, 1: division, 2: power
        if (viennacl::is_division<OP>::value)
          op_type = 1;
        else if (viennacl::is_product<OP>::value)
          op_type = 0;

        viennacl::ocl::kernel & k = detail::kernel_for_matrix(A, "element_op");
        viennacl::ocl::enqueue(k(viennacl::traits::opencl_handle(A),
                                cl_uint(viennacl::traits::start1(A)),           cl_uint(viennacl::traits::start2(A)),
                                cl_uint(viennacl::traits::stride1(A)),          cl_uint(viennacl::traits::stride2(A)),
                                cl_uint(viennacl::traits::size1(A)),            cl_uint(viennacl::traits::size2(A)),
                                cl_uint(viennacl::traits::internal_size1(A)),   cl_uint(viennacl::traits::internal_size2(A)),

                                viennacl::traits::opencl_handle(proxy.lhs()),
                                cl_uint(viennacl::traits::start1(proxy.lhs())),           cl_uint(viennacl::traits::start2(proxy.lhs())),
                                cl_uint(viennacl::traits::stride1(proxy.lhs())),          cl_uint(viennacl::traits::stride2(proxy.lhs())),
                                cl_uint(viennacl::traits::internal_size1(proxy.lhs())),   cl_uint(viennacl::traits::internal_size2(proxy.lhs())),

                                viennacl::traits::opencl_handle(proxy.rhs()),
                                cl_uint(viennacl::traits::start1(proxy.rhs())),           cl_uint(viennacl::traits::start2(proxy.rhs())),
                                cl_uint(viennacl::traits::stride1(proxy.rhs())),          cl_uint(viennacl::traits::stride2(proxy.rhs())),
                                cl_uint(viennacl::traits::internal_size1(proxy.rhs())),   cl_uint(viennacl::traits::internal_size2(proxy.rhs())),

                                op_type)
                              );
      }


      // Unary operations

      /** @brief Implementation of unary element-wise operations A = OP(B)
      *
      * @param A      The result matrix (or -range, or -slice)
      * @param proxy  The proxy object holding B and the operation
      */
      template <typename T, typename OP>
      void element_op(matrix_base<T> & A,
                      matrix_expression<const matrix_base<T>, const matrix_base<T>, op_element_unary<OP> > const & proxy)
      {
        assert(A.row_major() == proxy.lhs().row_major() && bool("Elementwise operations on mixed matrix layouts not supported yet!"));
        assert(A.row_major() == proxy.rhs().row_major() && bool("Elementwise operations on mixed matrix layouts not supported yet!"));

        assert(viennacl::traits::opencl_handle(A).context() == viennacl::traits::opencl_handle(proxy.lhs()).context() && bool("Vectors do not reside in the same OpenCL context. Automatic migration not yet supported!"));
        assert(viennacl::traits::opencl_handle(A).context() == viennacl::traits::opencl_handle(proxy.rhs()).context() && bool("Vectors do not reside in the same OpenCL context. Automatic migration not yet supported!"));

        viennacl::ocl::context & ctx = const_cast<viennacl::ocl::context &>(viennacl::traits::opencl_handle(A).context());

        viennacl::ocl::kernel *k = NULL;
        if (A.row_major())
        {
          viennacl::linalg::opencl::kernels::matrix_element<T, row_major>::init(ctx);
          k = &(ctx.get_kernel(viennacl::linalg::opencl::kernels::matrix_element<T, row_major>::program_name(), detail::op_to_string(OP()) + "_assign"));
        }
        else
        {
          viennacl::linalg::opencl::kernels::matrix_element<T, column_major>::init(ctx);
          k = &(ctx.get_kernel(viennacl::linalg::opencl::kernels::matrix_element<T, column_major>::program_name(), detail::op_to_string(OP()) + "_assign"));
        }

        viennacl::ocl::enqueue((*k)(viennacl::traits::opencl_handle(A),
                                 cl_uint(viennacl::traits::start1(A)),           cl_uint(viennacl::traits::start2(A)),
                                 cl_uint(viennacl::traits::stride1(A)),          cl_uint(viennacl::traits::stride2(A)),
                                 cl_uint(viennacl::traits::size1(A)),            cl_uint(viennacl::traits::size2(A)),
                                 cl_uint(viennacl::traits::internal_size1(A)),   cl_uint(viennacl::traits::internal_size2(A)),

                                 viennacl::traits::opencl_handle(proxy.lhs()),
                                 cl_uint(viennacl::traits::start1(proxy.lhs())),           cl_uint(viennacl::traits::start2(proxy.lhs())),
                                 cl_uint(viennacl::traits::stride1(proxy.lhs())),          cl_uint(viennacl::traits::stride2(proxy.lhs())),
                                 cl_uint(viennacl::traits::internal_size1(proxy.lhs())),   cl_uint(viennacl::traits::internal_size2(proxy.lhs())))
                              );
      }


      //
      /////////////////////////   matrix-vector products /////////////////////////////////
      //

      // A * x

      /** @brief Carries out matrix-vector multiplication
      *
      * Implementation of the convenience expression result = prod(mat, vec);
      *
      * @param mat    The matrix
      * @param vec    The vector
      * @param result The result vector
      */
      template <typename NumericT>
      void prod_impl(const matrix_base<NumericT> & mat, bool trans_mat,
                     const vector_base<NumericT> & vec,
                           vector_base<NumericT> & result)
      {
        typedef NumericT        value_type;

        // Inplace matrix-vector products like x = prod(A, x) are currently illegal: Introduce a temporary like y = prod(A, x); x = y; instead
        assert(viennacl::traits::handle(vec) != viennacl::traits::handle(result) && bool("No direct inplace matrix-vector product possible. Introduce a temporary!"));
        //result.resize(mat.size1());

        viennacl::ocl::kernel & k = detail::kernel_for_matrix(mat, trans_mat ? "trans_vec_mul" : "vec_mul");
        viennacl::ocl::enqueue(k(viennacl::traits::opencl_handle(mat),
                                cl_uint(viennacl::traits::start1(mat)),         cl_uint(viennacl::traits::start2(mat)),
                                cl_uint(viennacl::traits::stride1(mat)),        cl_uint(viennacl::traits::stride2(mat)),
                                cl_uint(viennacl::traits::size1(mat)),          cl_uint(viennacl::traits::size2(mat)),
                                cl_uint(viennacl::traits::internal_size1(mat)), cl_uint(viennacl::traits::internal_size2(mat)),

                                viennacl::traits::opencl_handle(vec),
                                cl_uint(viennacl::traits::start(vec)),
                                cl_uint(viennacl::traits::stride(vec)),
                                cl_uint(viennacl::traits::size(vec)),

                                viennacl::traits::opencl_handle(result),
                                cl_uint(viennacl::traits::start(result)),
                                cl_uint(viennacl::traits::stride(result)),
                                cl_uint(viennacl::traits::size(result)),

                                viennacl::ocl::local_mem(sizeof(value_type) * k.local_work_size())
                              ) );
      }

      //
      /////////////////////////   matrix-matrix products /////////////////////////////////
      //

      namespace detail
      {
        template <typename NumericT>
        viennacl::ocl::kernel & kernel_for_matrix_prod(matrix_base<NumericT> const & A, matrix_base<NumericT> const & B, matrix_base<NumericT> const & C, std::string kernel_name)
        {
          viennacl::ocl::context & ctx = const_cast<viennacl::ocl::context &>(viennacl::traits::opencl_handle(A).context());

          if (A.row_major() && B.row_major() && C.row_major())
          {
            typedef viennacl::linalg::opencl::kernels::matrix_prod<NumericT, row_major, row_major, row_major>  KernelClass;
            KernelClass::init(ctx);
            return ctx.get_kernel(KernelClass::program_name(), kernel_name);
          }
          else if (A.row_major() && B.row_major() && !C.row_major())
          {
            typedef viennacl::linalg::opencl::kernels::matrix_prod<NumericT, row_major, row_major, column_major>  KernelClass;
            KernelClass::init(ctx);
            return ctx.get_kernel(KernelClass::program_name(), kernel_name);
          }
          else if (A.row_major() && !B.row_major() && C.row_major())
          {
            typedef viennacl::linalg::opencl::kernels::matrix_prod<NumericT, row_major, column_major, row_major>  KernelClass;
            KernelClass::init(ctx);
            return ctx.get_kernel(KernelClass::program_name(), kernel_name);
          }
          else if (A.row_major() && !B.row_major() && !C.row_major())
          {
            typedef viennacl::linalg::opencl::kernels::matrix_prod<NumericT, row_major, column_major, column_major>  KernelClass;
            KernelClass::init(ctx);
            return ctx.get_kernel(KernelClass::program_name(), kernel_name);
          }
          else if (!A.row_major() && B.row_major() && C.row_major())
          {
            typedef viennacl::linalg::opencl::kernels::matrix_prod<NumericT, column_major, row_major, row_major>  KernelClass;
            KernelClass::init(ctx);
            return ctx.get_kernel(KernelClass::program_name(), kernel_name);
          }
          else if (!A.row_major() && B.row_major() && !C.row_major())
          {
            typedef viennacl::linalg::opencl::kernels::matrix_prod<NumericT, column_major, row_major, column_major>  KernelClass;
            KernelClass::init(ctx);
            return ctx.get_kernel(KernelClass::program_name(), kernel_name);
          }
          else if (!A.row_major() && !B.row_major() && C.row_major())
          {
            typedef viennacl::linalg::opencl::kernels::matrix_prod<NumericT, column_major, column_major, row_major>  KernelClass;
            KernelClass::init(ctx);
            return ctx.get_kernel(KernelClass::program_name(), kernel_name);
          }

          typedef viennacl::linalg::opencl::kernels::matrix_prod<NumericT, column_major, column_major, column_major>  KernelClass;
          KernelClass::init(ctx);
          return ctx.get_kernel(KernelClass::program_name(), kernel_name);
        }


        // C = A * B and possibly transposed variants
        template <typename T1, typename T2, typename T3, typename ScalarType >
        void prod_slow_kernel(const T1 & A,
                              const T2 & B,
                              T3 & C,
                              ScalarType alpha,
                              ScalarType beta,
                              std::string kernel_name)
        {
          typedef typename viennacl::result_of::cpu_value_type< typename T1::value_type >::type   cpu_value_type;

          viennacl::ocl::kernel & k = kernel_for_matrix_prod(A, B, C, kernel_name);

          k.global_work_size(0, viennacl::tools::align_to_multiple<unsigned int>(static_cast<unsigned int>(viennacl::traits::size1(C)), 16));
          k.global_work_size(1, viennacl::tools::align_to_multiple<unsigned int>(static_cast<unsigned int>(viennacl::traits::size2(C)), 16));
          k.local_work_size(0, 16);
          k.local_work_size(1, 16);

          cpu_value_type cl_alpha = static_cast<cpu_value_type>(alpha);
          cpu_value_type cl_beta  = static_cast<cpu_value_type>(beta);

          viennacl::ocl::enqueue(k(cl_alpha,
                                  viennacl::traits::opencl_handle(A),
                                  cl_uint(viennacl::traits::start1(A)),           cl_uint(viennacl::traits::start2(A)),
                                  cl_uint(viennacl::traits::stride1(A)),          cl_uint(viennacl::traits::stride2(A)),
                                  cl_uint(viennacl::traits::size1(A)),            cl_uint(viennacl::traits::size2(A)),
                                  cl_uint(viennacl::traits::internal_size1(A)),   cl_uint(viennacl::traits::internal_size2(A)),

                                  viennacl::traits::opencl_handle(B),
                                  cl_uint(viennacl::traits::start1(B)),           cl_uint(viennacl::traits::start2(B)),
                                  cl_uint(viennacl::traits::stride1(B)),          cl_uint(viennacl::traits::stride2(B)),
                                  cl_uint(viennacl::traits::size1(B)),            cl_uint(viennacl::traits::size2(B)),
                                  cl_uint(viennacl::traits::internal_size1(B)),   cl_uint(viennacl::traits::internal_size2(B)),

                                  cl_beta,
                                  viennacl::traits::opencl_handle(C),
                                  cl_uint(viennacl::traits::start1(C)),           cl_uint(viennacl::traits::start2(C)),
                                  cl_uint(viennacl::traits::stride1(C)),          cl_uint(viennacl::traits::stride2(C)),
                                  cl_uint(viennacl::traits::size1(C)),            cl_uint(viennacl::traits::size2(C)),
                                  cl_uint(viennacl::traits::internal_size1(C)),   cl_uint(viennacl::traits::internal_size2(C))
                                  )
                                );
        }

        // C = A * B, using fast kernel for NVIDIA
        template <typename T1, typename T2, typename T3, typename ScalarType >
        void prod_fast_kernel(const T1 & A,
                              const T2 & B,
                              T3 & C,
                              ScalarType alpha,
                              ScalarType beta,
                              std::string kernel_name)
        {
          typedef typename viennacl::result_of::cpu_value_type< typename T1::value_type >::type   cpu_value_type;

          viennacl::ocl::kernel & k = kernel_for_matrix_prod(A, B, C, kernel_name);

          k.global_work_size(0, viennacl::traits::size2(C) / 4); //column blocks
          k.global_work_size(1, viennacl::traits::size1(C) / 4); //row blocks
          k.local_work_size(0, 16);  //columns
          k.local_work_size(1, 4);   //rows

          cpu_value_type cl_alpha = static_cast<cpu_value_type>(alpha);
          cpu_value_type cl_beta  = static_cast<cpu_value_type>(beta);

          viennacl::ocl::enqueue(k(cl_alpha,
                                  viennacl::traits::opencl_handle(A),
                                  cl_uint(viennacl::traits::start1(A)),           cl_uint(viennacl::traits::start2(A)),
                                  cl_uint(viennacl::traits::stride1(A)),          cl_uint(viennacl::traits::stride2(A)),
                                  cl_uint(viennacl::traits::size1(A)),            cl_uint(viennacl::traits::size2(A)),
                                  cl_uint(viennacl::traits::internal_size1(A)),   cl_uint(viennacl::traits::internal_size2(A)),

                                  viennacl::traits::opencl_handle(B),
                                  cl_uint(viennacl::traits::start1(B)),           cl_uint(viennacl::traits::start2(B)),
                                  cl_uint(viennacl::traits::stride1(B)),          cl_uint(viennacl::traits::stride2(B)),
                                  cl_uint(viennacl::traits::size1(B)),            cl_uint(viennacl::traits::size2(B)),
                                  cl_uint(viennacl::traits::internal_size1(B)),   cl_uint(viennacl::traits::internal_size2(B)),

                                  cl_beta,
                                  viennacl::traits::opencl_handle(C),
                                  cl_uint(viennacl::traits::start1(C)),           cl_uint(viennacl::traits::start2(C)),
                                  cl_uint(viennacl::traits::stride1(C)),          cl_uint(viennacl::traits::stride2(C)),
                                  cl_uint(viennacl::traits::size1(C)),            cl_uint(viennacl::traits::size2(C)),
                                  cl_uint(viennacl::traits::internal_size1(C)),   cl_uint(viennacl::traits::internal_size2(C))
                                  )
                                );
        }

        template <typename T1, typename T2, typename T3, typename ScalarType >
        void prod(const T1 & A,
                  const T2 & B,
                  T3 & C,
                  ScalarType alpha,
                  ScalarType beta,
                  std::string fast_kernel_name,
                  std::string slow_kernel_name)
        {
          if (   (viennacl::traits::size1(A) < 64)
              || (viennacl::traits::size2(A) < 64)
              || (viennacl::traits::size1(B) < 64)
              || (viennacl::traits::size2(B) < 64) )   //there is most likely not enough to compute, rendering kernel launch overhead considerable
          {
            prod_slow_kernel(A, B, C, alpha, beta, slow_kernel_name);
          }
          else if (   (viennacl::traits::size1(A) % 64 == 0)
                   && (viennacl::traits::size2(A) % 64 == 0)
                   && (viennacl::traits::size1(B) % 64 == 0)
                   && (viennacl::traits::size2(B) % 64 == 0) )   // allows the use of the fast NVIDIA kernel
          {
            prod_fast_kernel(A, B, C, alpha, beta, fast_kernel_name);
            //prod_slow_kernel(A, B, C, slow_kernel_name);
          }
          else //TODO: use four kernels
          {
            prod_slow_kernel(A, B, C, alpha, beta, slow_kernel_name);
          }

        }
      } // namespace detail


      /** @brief Carries out matrix-matrix multiplication
      *
      * Implementation of C = prod(A, B);
      *
      */
      template <typename NumericT, typename ScalarType >
      void prod_impl(const matrix_base<NumericT> & A, bool trans_A,
                     const matrix_base<NumericT> & B, bool trans_B,
                           matrix_base<NumericT> & C,
                     ScalarType alpha,
                     ScalarType beta)
      {
        bool A_not_aligned = (A.internal_size1() % matrix_base<NumericT>::alignment > 0) || (A.internal_size2() % matrix_base<NumericT>::alignment > 0);
        bool B_not_aligned = (B.internal_size1() % matrix_base<NumericT>::alignment > 0) || (B.internal_size2() % matrix_base<NumericT>::alignment > 0);
        bool C_not_aligned = (C.internal_size1() % matrix_base<NumericT>::alignment > 0) || (C.internal_size2() % matrix_base<NumericT>::alignment > 0);

        // Inplace matrix-vector products like B = prod(A, B) are currently illegal: Introduce a temporary like C = prod(A, B); B = C; instead
        /*assert(  (viennacl::traits::handle(C) != viennacl::traits::handle(A))
              && (viennacl::traits::handle(C) != viennacl::traits::handle(B))
              && bool("No direct inplace matrix-matrix product possible. Introduce a temporary!"));*/

        std::string string_prod16("prod16_");
        std::string string_prod("prod_");

        string_prod16.append(trans_A ? "T" : "A");
        string_prod.append(trans_A ? "T" : "A");

        string_prod16.append(trans_B ? "T" : "A");
        string_prod.append(trans_B ? "T" : "A");

        if (A_not_aligned || A.start1() > 0 || A.start2() > 0 || A.stride1() > 1 || A.stride2() > 1
          ||B_not_aligned || B.start1() > 0 || B.start2() > 0 || B.stride1() > 1 || B.stride2() > 1
          ||C_not_aligned || C.start1() > 0 || C.start2() > 0 || C.stride1() > 1 || C.stride2() > 1)
          detail::prod(A, B, C, alpha, beta, string_prod16, string_prod);
        else
        {
          if (!trans_A && !trans_B)
          {
            typedef matrix_expression<const matrix_base<NumericT>, const matrix_base<NumericT>, op_mat_mat_prod> ProdType;
            viennacl::device_specific::execute(device_specific::MATRIX_PRODUCT_NN_TYPE, device_specific::result_of::numeric_type_id<NumericT>::value
                                                                  ,viennacl::scheduler::statement(C, viennacl::op_assign(),alpha*ProdType(A,B)+beta*C));
          }
          else if (!trans_A && trans_B)
          {
            typedef const viennacl::matrix_expression< const matrix_base<NumericT>, const matrix_base<NumericT>, op_trans> RhsType;
            typedef matrix_expression<const matrix_base<NumericT>, RhsType, op_mat_mat_prod> ProdType;
            viennacl::device_specific::execute(device_specific::MATRIX_PRODUCT_NT_TYPE, device_specific::result_of::numeric_type_id<NumericT>::value
                                                                  ,viennacl::scheduler::statement(C, viennacl::op_assign(),alpha*ProdType(A,RhsType(B,B))+beta*C));
          }
          else if (trans_A && !trans_B)
          {
            typedef const viennacl::matrix_expression< const matrix_base<NumericT>, const matrix_base<NumericT>, op_trans> LhsType;
            typedef matrix_expression<LhsType, const matrix_base<NumericT>, op_mat_mat_prod> ProdType;
            viennacl::device_specific::execute(device_specific::MATRIX_PRODUCT_TN_TYPE, device_specific::result_of::numeric_type_id<NumericT>::value
                                                                  ,viennacl::scheduler::statement(C, viennacl::op_assign(),alpha*ProdType(LhsType(A,A),B)+beta*C));
          }
          else if (trans_A && trans_B)
          {
            typedef const viennacl::matrix_expression< const matrix_base<NumericT>, const matrix_base<NumericT>, op_trans> LhsType;
            typedef const viennacl::matrix_expression< const matrix_base<NumericT>, const matrix_base<NumericT>, op_trans> RhsType;
            typedef matrix_expression<LhsType, RhsType, op_mat_mat_prod> ProdType;
            viennacl::device_specific::execute(device_specific::MATRIX_PRODUCT_TT_TYPE, device_specific::result_of::numeric_type_id<NumericT>::value
                                                                  ,viennacl::scheduler::statement(C, viennacl::op_assign(),alpha*ProdType(LhsType(A,A),RhsType(B,B))+beta*C));
          }
        }
      }

      //
      /////////////////////////   miscellaneous operations /////////////////////////////////
      //


      /** @brief The implementation of the operation mat += alpha * vec1 * vec2^T, i.e. a scaled rank 1 update
      *
      * Implementation of the convenience expression result += alpha * outer_prod(vec1, vec2);
      *
      * @param mat1    The matrix to be updated
      * @param alpha            The scaling factor (either a viennacl::scalar<>, float, or double)
      * @param len_alpha        Length of the buffer for an eventual final reduction step (currently always '1')
      * @param reciprocal_alpha Use 1/alpha instead of alpha
      * @param flip_sign_alpha  Use -alpha instead of alpha
      * @param vec1    The first vector
      * @param vec2    The second vector
      */
      template <typename NumericT, typename S1>
      void scaled_rank_1_update(matrix_base<NumericT> & mat1,
                                S1 const & alpha, vcl_size_t len_alpha, bool reciprocal_alpha, bool flip_sign_alpha,
                                const vector_base<NumericT> & vec1,
                                const vector_base<NumericT> & vec2)
      {
        assert( (viennacl::traits::size1(mat1) == viennacl::traits::size(vec1)) && bool("Size mismatch in scaled_rank_1_update: size1(A) != size(v1)"));
        assert( (viennacl::traits::size2(mat1) == viennacl::traits::size(vec2)) && bool("Size mismatch in scaled_rank_1_update: size2(A) != size(v2)"));

        cl_uint options_alpha = detail::make_options(len_alpha, reciprocal_alpha, flip_sign_alpha);

        viennacl::ocl::kernel & k = detail::kernel_for_matrix(mat1, viennacl::is_cpu_scalar<S1>::value ? "scaled_rank1_update_cpu" : "scaled_rank1_update_gpu");

        viennacl::ocl::enqueue(k(viennacl::traits::opencl_handle(mat1),
                                 cl_uint(viennacl::traits::start1(mat1)),           cl_uint(viennacl::traits::start2(mat1)),
                                 cl_uint(viennacl::traits::stride1(mat1)),          cl_uint(viennacl::traits::stride2(mat1)),
                                 cl_uint(viennacl::traits::size1(mat1)),            cl_uint(viennacl::traits::size2(mat1)),
                                 cl_uint(viennacl::traits::internal_size1(mat1)),   cl_uint(viennacl::traits::internal_size2(mat1)),

                                 viennacl::traits::opencl_handle(viennacl::tools::promote_if_host_scalar<NumericT>(alpha)),
                                 options_alpha,

                                 viennacl::traits::opencl_handle(vec1),
                                 cl_uint(viennacl::traits::start(vec1)),
                                 cl_uint(viennacl::traits::stride(vec1)),
                                 cl_uint(viennacl::traits::size(vec1)),

                                 viennacl::traits::opencl_handle(vec2),
                                 cl_uint(viennacl::traits::start(vec2)),
                                 cl_uint(viennacl::traits::stride(vec2)),
                                 cl_uint(viennacl::traits::size(vec2))
                                )
                              );
      }

    } // namespace opencl
  } //namespace linalg
} //namespace viennacl


#endif
