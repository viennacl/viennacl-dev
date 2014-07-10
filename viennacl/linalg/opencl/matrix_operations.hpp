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

#include "viennacl/device_specific/templates/matrix_axpy_template.hpp"
#include "viennacl/device_specific/database.hpp"

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

      namespace detail
      {
        template <typename NumericT>
        viennacl::ocl::program & program_for_matrix(matrix_base<NumericT> const & A, unsigned int id)
        {
          viennacl::ocl::context & ctx = const_cast<viennacl::ocl::context &>(viennacl::traits::opencl_handle(A).context());

          if (A.row_major())
          {
            switch(id)
            {
              case 0:{
                typedef viennacl::linalg::opencl::kernels::matrix<NumericT, row_major>  KernelClass;
                KernelClass::init(ctx);
                return ctx.get_program(KernelClass::program_name());
              }
               case 1:{
                typedef viennacl::linalg::opencl::kernels::matrix_element<NumericT, row_major>  KernelClass;
                KernelClass::init(ctx);
                return ctx.get_program(KernelClass::program_name());
              }
              default:{
                typedef viennacl::linalg::opencl::kernels::matrix_prod<NumericT, column_major, column_major, row_major>  KernelClass;
                KernelClass::init(ctx);
                return ctx.get_program(KernelClass::program_name());
              }
            }
          }
          else
          {
            switch(id)
            {
              case 0:{
                typedef viennacl::linalg::opencl::kernels::matrix<NumericT, column_major>  KernelClass;
                KernelClass::init(ctx);
                return ctx.get_program(KernelClass::program_name());
              }
               case 1:{
                typedef viennacl::linalg::opencl::kernels::matrix_element<NumericT, column_major>  KernelClass;
                KernelClass::init(ctx);
                return ctx.get_program(KernelClass::program_name());
              }
              default:{
                typedef viennacl::linalg::opencl::kernels::matrix_prod<NumericT, column_major, column_major, column_major>  KernelClass;
                KernelClass::init(ctx);
                return ctx.get_program(KernelClass::program_name());
              }
            }
          }
        }

        template <typename NumericT>
        viennacl::ocl::kernel & kernel_for_matrix(matrix_base<NumericT> const & A, std::string kernel_name)
        {
          return program_for_matrix(A,0).get_kernel(kernel_name);
        }

      }


      //
      // Introductory note: By convention, all dimensions are already checked in the dispatcher frontend. No need to double-check again in here!
      //

      template <typename NumericT,
                typename ScalarType1>
      void am(matrix_base<NumericT> & mat1,
              matrix_base<NumericT> const & mat2, ScalarType1 const & alpha, vcl_size_t /* len_alpha */, bool reciprocal_alpha, bool flip_sign_alpha)
      {
        assert(mat1.row_major() == mat2.row_major() && bool("Addition/subtraction on mixed matrix layouts not supported yet!"));

        std::string kernel_name("assign_*m_**00");
        kernel_name[7] = is_cpu_scalar<ScalarType1>::value?'h':'d';
        kernel_name[10] = flip_sign_alpha?'1':'0';
        kernel_name[11] = reciprocal_alpha?'1':'0';
        device_specific::matrix_axpy_template(detail::matrix_axpy_params<NumericT>(detail::current_device(mat1)),kernel_name)
                                        .enqueue(detail::program_for_matrix(mat1,0),
                                                scheduler::preset::av(scheduler::OPERATION_BINARY_ASSIGN_TYPE, &mat1, &mat2, &alpha, flip_sign_alpha, reciprocal_alpha));
      }


      template <typename NumericT,
                typename ScalarType1, typename ScalarType2>
      void ambm(matrix_base<NumericT> & mat1,
                matrix_base<NumericT> const & mat2, ScalarType1 const & alpha, vcl_size_t /* len_alpha */, bool reciprocal_alpha, bool flip_sign_alpha,
                matrix_base<NumericT> const & mat3, ScalarType2 const & beta,  vcl_size_t /* len_beta */,  bool reciprocal_beta,  bool flip_sign_beta)
      {
        assert(mat1.row_major() == mat2.row_major() && mat1.row_major() == mat3.row_major() && bool("Addition/subtraction on mixed matrix layouts not supported yet!"));

        std::string kernel_name("assign_*m*m_****");
        kernel_name[7] = is_cpu_scalar<ScalarType1>::value?'h':'d';
        kernel_name[9] = is_cpu_scalar<ScalarType2>::value?'h':'d';
        kernel_name[12] = flip_sign_alpha?'1':'0';
        kernel_name[13] = reciprocal_alpha?'1':'0';
        kernel_name[14] = flip_sign_beta?'1':'0';
        kernel_name[15] = reciprocal_beta?'1':'0';
        device_specific::matrix_axpy_template(detail::matrix_axpy_params<NumericT>(detail::current_device(mat1)),kernel_name)
                                        .enqueue(detail::program_for_matrix(mat1,0),
                                                scheduler::preset::avbv(scheduler::OPERATION_BINARY_ASSIGN_TYPE, &mat1, &mat2, &alpha, flip_sign_alpha, reciprocal_alpha, &mat3, &beta, flip_sign_beta, reciprocal_beta));
      }


      template <typename NumericT,
                typename ScalarType1, typename ScalarType2>
      void ambm_m(matrix_base<NumericT> & mat1,
                  matrix_base<NumericT> const & mat2, ScalarType1 const & alpha, vcl_size_t /* len_alpha */, bool reciprocal_alpha, bool flip_sign_alpha,
                  matrix_base<NumericT> const & mat3, ScalarType2 const & beta,  vcl_size_t /* len_beta */,  bool reciprocal_beta,  bool flip_sign_beta)
      {
        assert(mat1.row_major() == mat2.row_major() && mat1.row_major() == mat3.row_major() && bool("Addition/subtraction on mixed matrix layouts not supported yet!"));

        std::string kernel_name("ip_add_*v*v_****");
        kernel_name[7] = is_cpu_scalar<ScalarType1>::value?'h':'d';
        kernel_name[9] = is_cpu_scalar<ScalarType2>::value?'h':'d';
        kernel_name[12] = flip_sign_alpha?'1':'0';
        kernel_name[13] = reciprocal_alpha?'1':'0';
        kernel_name[14] = flip_sign_beta?'1':'0';
        kernel_name[15] = reciprocal_beta?'1':'0';
        device_specific::matrix_axpy_template(detail::matrix_axpy_params<NumericT>(detail::current_device(mat1)),kernel_name)
                                        .enqueue(detail::program_for_matrix(mat1,0),
                                                scheduler::preset::avbv(scheduler::OPERATION_BINARY_INPLACE_ADD_TYPE, &mat1, &mat2, &alpha, flip_sign_alpha, reciprocal_alpha, &mat3, &beta, flip_sign_beta, reciprocal_beta));
      }



      template <typename NumericT>
      void matrix_assign(matrix_base<NumericT> & mat1, NumericT s, bool up_to_internal_size = false)
      {

        scalar_matrix<NumericT> mat2(viennacl::traits::size1(mat1),viennacl::traits::size2(mat1),s);
        device_specific::matrix_axpy_template tplt(detail::matrix_axpy_params<NumericT>(detail::current_device(mat1)),"assign_cpu");
        tplt.up_to_internal_size(up_to_internal_size);
        tplt.enqueue(detail::program_for_matrix(mat1,0), scheduler::preset::assign_cpu(&mat1, &mat2));
      }

      template <typename NumericT>
      void matrix_diagonal_assign(matrix_base<NumericT> & mat1, NumericT s)
      {
        viennacl::scalar_vector<NumericT> sx(std::min(viennacl::traits::size1(mat1), viennacl::traits::size2(mat1)), s);
        device_specific::vector_axpy_template(detail::vector_axpy_params<NumericT>(detail::current_device(mat1)), "diagonal_assign_cpu")
                                        .enqueue(detail::program_for_matrix(mat1,0), scheduler::preset::diagonal_assign_cpu(&mat1, &sx));
      }

      template <typename NumericT>
      void matrix_diag_from_vector(const vector_base<NumericT> & vec, int k, matrix_base<NumericT> & mat1)
      {
        device_specific::matrix_axpy_template(detail::matrix_axpy_params<NumericT>(detail::current_device(mat1)),"matrix_diag_from_vector")
                                        .enqueue(detail::program_for_matrix(mat1,0), scheduler::preset::matrix_diag_from_vector(&vec, &mat1, k));
      }

      template <typename NumericT>
      void matrix_diag_to_vector(const matrix_base<NumericT> & mat1, int k, vector_base<NumericT> & vec)
      {
        device_specific::vector_axpy_template(detail::vector_axpy_params<NumericT>(detail::current_device(mat1)),"matrix_diag_to_vector")
                                        .enqueue(detail::program_for_matrix(mat1,0), scheduler::preset::matrix_diag_to_vector(&vec, &mat1, k));
      }

      template <typename NumericT>
      void matrix_row(const matrix_base<NumericT> & mat1, unsigned int i, vector_base<NumericT> & vec)
      {
        device_specific::vector_axpy_template(detail::vector_axpy_params<NumericT>(detail::current_device(mat1)),"matrix_row")
                                        .enqueue(detail::program_for_matrix(mat1,0), scheduler::preset::matrix_row(&vec, &mat1, i));
      }

      template <typename NumericT>
      void matrix_column(const matrix_base<NumericT> & mat1, unsigned int j, vector_base<NumericT> & vec)
      {
        device_specific::vector_axpy_template(detail::vector_axpy_params<NumericT>(detail::current_device(mat1)),"matrix_column")
                                        .enqueue(detail::program_for_matrix(mat1,0), scheduler::preset::matrix_column(&vec, &mat1, j));
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

        scheduler::operation_node_type TYPE = scheduler::operation_node_type(scheduler::result_of::op_type_info<op_element_binary<OP> >::id);
        device_specific::matrix_axpy_template(detail::matrix_axpy_params<T>(detail::current_device(A)),device_specific::tree_parsing::operator_string(TYPE))
                                            .enqueue(detail::program_for_matrix(A,1),scheduler::preset::binary_element_op(&A, &proxy.lhs(), &proxy.rhs(),TYPE));
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

        assert(viennacl::traits::opencl_handle(A).context() == viennacl::traits::opencl_handle(proxy.lhs()).context() && bool("Matrices do not reside in the same OpenCL context. Automatic migration not yet supported!"));
        assert(viennacl::traits::opencl_handle(A).context() == viennacl::traits::opencl_handle(proxy.rhs()).context() && bool("Matrices do not reside in the same OpenCL context. Automatic migration not yet supported!"));

        scheduler::operation_node_type TYPE = scheduler::operation_node_type(scheduler::result_of::op_type_info<op_element_unary<OP> >::id);
        device_specific::matrix_axpy_template(detail::matrix_axpy_params<T>(detail::current_device(A)),device_specific::tree_parsing::operator_string(TYPE))
                                        .enqueue(detail::program_for_matrix(A,1),scheduler::preset::unary_element_op(&A, &proxy.lhs(),TYPE));
      }


      //
      /////////////////////////   matrix-vector products /////////////////////////////////
      //

      // A * x

      /** @brief Carries out matrix-vector multiplication
      *
      * Implementation of the convenience expression result = prod(mat1, vec);
      *
      * @param mat    The matrix
      * @param vec    The vector
      * @param result The result vector
      */
      template <typename NumericT>
      void prod_impl(const matrix_base<NumericT> & mat1, bool trans_mat1,
                     const vector_base<NumericT> & vec,
                           vector_base<NumericT> & result)
      {
        typedef NumericT        value_type;

        // Inplace matrix-vector products like x = prod(A, x) are currently illegal: Introduce a temporary like y = prod(A, x); x = y; instead
        assert(viennacl::traits::handle(vec) != viennacl::traits::handle(result) && bool("No direct inplace matrix-vector product possible. Introduce a temporary!"));
        device_specific::row_wise_reduction_template(detail::row_wise_reduction_params<NumericT>(detail::current_device(mat1), trans_mat1^mat1.row_major()), trans_mat1?"mat_vec_T":"mat_vec_N")
            .enqueue(detail::program_for_matrix(mat1,0),scheduler::preset::mat_vec_prod(&mat1, trans_mat1, &vec, &result));
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

//        // C = A * B, using fast kernel for NVIDIA
//        template <typename T1, typename T2, typename T3, typename ScalarType >
//        void prod_fast_kernel(const T1 & A,
//                              const T2 & B,
//                              T3 & C,
//                              ScalarType alpha,
//                              ScalarType beta,
//                              std::string kernel_name)
//        {
//          typedef typename viennacl::result_of::cpu_value_type< typename T1::value_type >::type   cpu_value_type;

//          viennacl::ocl::kernel & k = kernel_for_matrix_prod(A, B, C, kernel_name);

//          k.global_work_size(0, viennacl::traits::size2(C) / 4); //column blocks
//          k.global_work_size(1, viennacl::traits::size1(C) / 4); //row blocks
//          k.local_work_size(0, 16);  //columns
//          k.local_work_size(1, 4);   //rows

//          cpu_value_type cl_alpha = static_cast<cpu_value_type>(alpha);
//          cpu_value_type cl_beta  = static_cast<cpu_value_type>(beta);

//          viennacl::ocl::enqueue(k(cl_alpha,
//                                  viennacl::traits::opencl_handle(A),
//                                  cl_uint(viennacl::traits::start1(A)),           cl_uint(viennacl::traits::start2(A)),
//                                  cl_uint(viennacl::traits::stride1(A)),          cl_uint(viennacl::traits::stride2(A)),
//                                  cl_uint(viennacl::traits::size1(A)),            cl_uint(viennacl::traits::size2(A)),
//                                  cl_uint(viennacl::traits::internal_size1(A)),   cl_uint(viennacl::traits::internal_size2(A)),

//                                  viennacl::traits::opencl_handle(B),
//                                  cl_uint(viennacl::traits::start1(B)),           cl_uint(viennacl::traits::start2(B)),
//                                  cl_uint(viennacl::traits::stride1(B)),          cl_uint(viennacl::traits::stride2(B)),
//                                  cl_uint(viennacl::traits::size1(B)),            cl_uint(viennacl::traits::size2(B)),
//                                  cl_uint(viennacl::traits::internal_size1(B)),   cl_uint(viennacl::traits::internal_size2(B)),

//                                  cl_beta,
//                                  viennacl::traits::opencl_handle(C),
//                                  cl_uint(viennacl::traits::start1(C)),           cl_uint(viennacl::traits::start2(C)),
//                                  cl_uint(viennacl::traits::stride1(C)),          cl_uint(viennacl::traits::stride2(C)),
//                                  cl_uint(viennacl::traits::size1(C)),            cl_uint(viennacl::traits::size2(C)),
//                                  cl_uint(viennacl::traits::internal_size1(C)),   cl_uint(viennacl::traits::internal_size2(C))
//                                  )
//                                );
//        }

//        template <typename T1, typename T2, typename T3, typename ScalarType >
//        void prod(const T1 & A,
//                  const T2 & B,
//                  T3 & C,
//                  ScalarType alpha,
//                  ScalarType beta,
//                  std::string fast_kernel_name,
//                  std::string slow_kernel_name)
//        {
//          if (   (viennacl::traits::size1(A) < 64)
//              || (viennacl::traits::size2(A) < 64)
//              || (viennacl::traits::size1(B) < 64)
//              || (viennacl::traits::size2(B) < 64) )   //there is most likely not enough to compute, rendering kernel launch overhead considerable
//          {
//            prod_slow_kernel(A, B, C, alpha, beta, slow_kernel_name);
//          }
//          else if (   (viennacl::traits::size1(A) % 64 == 0)
//                   && (viennacl::traits::size2(A) % 64 == 0)
//                   && (viennacl::traits::size1(B) % 64 == 0)
//                   && (viennacl::traits::size2(B) % 64 == 0) )   // allows the use of the fast NVIDIA kernel
//          {
//            prod_fast_kernel(A, B, C, alpha, beta, fast_kernel_name);
//            //prod_slow_kernel(A, B, C, slow_kernel_name);
//          }
//          else //TODO: use four kernels
//          {
//            prod_slow_kernel(A, B, C, alpha, beta, slow_kernel_name);
//          }

//        }
      } // namespace detail


      /** @brief Carries out matrix-matrix multiplication
      *
      * Implementation of C = prod(A, B);
      *
      */
      template <typename NumericT, typename ScalarType >
      void prod_impl(const matrix_base<NumericT> & A, bool A_trans,
                     const matrix_base<NumericT> & B, bool B_trans,
                           matrix_base<NumericT> & C,
                     ScalarType alpha,
                     ScalarType beta)
      {
        // Inplace matrix-vector products like B = prod(A, B) are currently illegal: Introduce a temporary like C = prod(A, B); B = C; instead
        /*assert(  (viennacl::traits::handle(C) != viennacl::traits::handle(A))
              && (viennacl::traits::handle(C) != viennacl::traits::handle(B))
              && bool("No direct inplace matrix-matrix product possible. Introduce a temporary!"));*/

        using namespace viennacl::traits;
        if (start1(A) > 0 || start2(A) > 0 || stride1(A) > 1 || stride2(A) > 1 ||
            start1(B) > 0 || start2(B) > 0 || stride1(B) > 1 || stride2(B) > 1 ||
            start1(C) > 0 || start2(C) > 0 || stride1(C) > 1 || stride2(C) > 1
              || viennacl::traits::handle(A)==viennacl::traits::handle(B))
        {
          std::string string_prod("prod_");
          string_prod.append(A_trans ? "T" : "A");
          string_prod.append(B_trans ? "T" : "A");
          detail::prod_slow_kernel(A, B, C, alpha, beta, string_prod);
        }
        else
        {
          using namespace viennacl::device_specific::database;
          device_specific::matrix_product_template::parameters const * parameters;
          viennacl::ocl::device const & device = traits::opencl_handle(C).context().current_device();

          bool effective_A_trans = A_trans ^ A.row_major();
          bool effective_B_trans = B_trans ^ B.row_major();

          if(!effective_A_trans && !effective_B_trans) parameters = &get<NumericT>(matrix_product_NN, device);
          else if(effective_A_trans && !effective_B_trans) parameters = &get<NumericT>(matrix_product_TN, device);
          else if(!effective_A_trans && effective_B_trans) parameters = &get<NumericT>(matrix_product_NT, device);
          else  parameters = &get<NumericT>(matrix_product_TT, device);

          std::string kernel_prefix("prodopt_");
          kernel_prefix.append(effective_A_trans ? "T" : "N");
          kernel_prefix.append(effective_B_trans ? "T" : "N");
          device_specific::matrix_product_template(*parameters, kernel_prefix).enqueue(detail::program_for_matrix(C,2),scheduler::preset::mat_mat_prod(alpha, &A, effective_A_trans, &B, effective_B_trans, beta, &C));

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
