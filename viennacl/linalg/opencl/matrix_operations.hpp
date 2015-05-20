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

#include "viennacl/traits/size.hpp"
#include "viennacl/traits/start.hpp"
#include "viennacl/traits/handle.hpp"
#include "viennacl/traits/stride.hpp"

#include "viennacl/linalg/opencl/common.hpp"
#include "viennacl/linalg/opencl/kernels/svd.hpp"

#include "viennacl/linalg/opencl/kernels/matrix.hpp"

namespace viennacl
{
namespace linalg
{
namespace opencl
{

//
// Introductory note: By convention, all dimensions are already checked in the dispatcher frontend. No need to double-check again in here!
//

const std::string SVD_BIDIAG_PACK_KERNEL = "bidiag_pack";
const std::string SVD_HOUSEHOLDER_UPDATE_A_LEFT_KERNEL = "house_update_A_left";
const std::string SVD_HOUSEHOLDER_UPDATE_A_RIGHT_KERNEL = "house_update_A_right";
const std::string SVD_HOUSEHOLDER_UPDATE_QL_KERNEL = "house_update_QL";
const std::string SVD_GIVENS_NEXT_KERNEL = "givens_next";
const std::string SVD_COPY_COL_KERNEL = "copy_col";
const std::string SVD_COPY_ROW_KERNEL = "copy_row";

template<typename NumericT,
         typename ScalarT1>
void am(matrix_base<NumericT> & A,
        matrix_base<NumericT> const & B, ScalarT1 const & alpha, vcl_size_t /* len_alpha */, bool reciprocal_alpha, bool flip_sign_alpha)
{
  assert(A.row_major() == B.row_major() && bool("Addition/subtraction on mixed matrix layouts not supported yet!"));

  std::string kernel_name("assign_*m_**00");
  bool is_scalar_cpu = is_cpu_scalar<ScalarT1>::value;
  kernel_name[7]  = is_scalar_cpu    ? 'h' : 'd';
  kernel_name[10] = flip_sign_alpha  ? '1' : '0';
  kernel_name[11] = reciprocal_alpha ? '1' : '0';

  scheduler::statement statement = scheduler::preset::av(scheduler::OPERATION_BINARY_ASSIGN_TYPE, &A, &B, &alpha, flip_sign_alpha, reciprocal_alpha);
  kernels::matrix<NumericT>::execution_handler(A.row_major(), viennacl::traits::opencl_context(A)).execute(kernel_name, statement);
}


template<typename NumericT,
          typename ScalarT1, typename ScalarT2>
void ambm(matrix_base<NumericT> & A,
          matrix_base<NumericT> const & B, ScalarT1 const & alpha, vcl_size_t /* len_alpha */, bool reciprocal_alpha, bool flip_sign_alpha,
          matrix_base<NumericT> const & C, ScalarT2 const & beta,  vcl_size_t /* len_beta */,  bool reciprocal_beta,  bool flip_sign_beta)
{
  assert(A.row_major() == B.row_major() && A.row_major() == C.row_major() && bool("Addition/subtraction on mixed matrix layouts not supported yet!"));

  std::string kernel_name("assign_*m*m_****");
  bool is_scalar_cpu1 = is_cpu_scalar<ScalarT1>::value;
  bool is_scalar_cpu2 = is_cpu_scalar<ScalarT2>::value;
  kernel_name[7]  = is_scalar_cpu1   ? 'h' : 'd';
  kernel_name[9]  = is_scalar_cpu2   ? 'h' : 'd';
  kernel_name[12] = flip_sign_alpha  ? '1' : '0';
  kernel_name[13] = reciprocal_alpha ? '1' : '0';
  kernel_name[14] = flip_sign_beta   ? '1' : '0';
  kernel_name[15] = reciprocal_beta  ? '1' : '0';

  scheduler::statement statement = scheduler::preset::avbv(scheduler::OPERATION_BINARY_ASSIGN_TYPE, &A, &B, &alpha, flip_sign_alpha, reciprocal_alpha, &C, &beta, flip_sign_beta, reciprocal_beta);
  kernels::matrix<NumericT>::execution_handler(A.row_major(), viennacl::traits::opencl_context(A)).execute(kernel_name, statement);
}


template<typename NumericT,
          typename ScalarT1, typename ScalarT2>
void ambm_m(matrix_base<NumericT> & A,
            matrix_base<NumericT> const & B, ScalarT1 const & alpha, vcl_size_t /* len_alpha */, bool reciprocal_alpha, bool flip_sign_alpha,
            matrix_base<NumericT> const & C, ScalarT2 const & beta,  vcl_size_t /* len_beta */,  bool reciprocal_beta,  bool flip_sign_beta)
{
  assert(A.row_major() == B.row_major() && A.row_major() == C.row_major() && bool("Addition/subtraction on mixed matrix layouts not supported yet!"));

  std::string kernel_name("ip_add_*v*v_****");
  bool is_scalar_cpu1 = is_cpu_scalar<ScalarT1>::value;
  bool is_scalar_cpu2 = is_cpu_scalar<ScalarT2>::value;
  kernel_name[7]  = is_scalar_cpu1   ? 'h' : 'd';
  kernel_name[9]  = is_scalar_cpu2   ? 'h' : 'd';
  kernel_name[12] = flip_sign_alpha  ? '1' : '0';
  kernel_name[13] = reciprocal_alpha ? '1' : '0';
  kernel_name[14] = flip_sign_beta   ? '1' : '0';
  kernel_name[15] = reciprocal_beta  ? '1' : '0';


  scheduler::statement statement = scheduler::preset::avbv(scheduler::OPERATION_BINARY_INPLACE_ADD_TYPE, &A, &B, &alpha, flip_sign_alpha, reciprocal_alpha, &C, &beta, flip_sign_beta, reciprocal_beta);
  kernels::matrix<NumericT>::execution_handler(A.row_major(), viennacl::traits::opencl_context(A)).execute(kernel_name, statement);
}

template<typename NumericT,
          typename SizeT, typename DistanceT>
void trans(const matrix_expression<const matrix_base<NumericT, SizeT, DistanceT>,const matrix_base<NumericT, SizeT, DistanceT>, op_trans> & proxy,
           matrix_base<NumericT> & temp_trans)
{
  std::string kernel_name("trans_kernel");
  viennacl::ocl::kernel& kernel = detail::legacy_kernel_for_matrix(proxy.lhs(),kernel_name);
  viennacl::ocl::enqueue(kernel(proxy.lhs(),
                                static_cast<cl_uint>(proxy.lhs().start1()),         static_cast<cl_uint>(proxy.lhs().start2()),
                                static_cast<cl_uint>(proxy.lhs().internal_size1()), static_cast<cl_uint>(proxy.lhs().internal_size2()),
                                static_cast<cl_uint>(proxy.lhs().size1()),          static_cast<cl_uint>(proxy.lhs().size2()),
                                static_cast<cl_uint>(proxy.lhs().stride1()),        static_cast<cl_uint>(proxy.lhs().stride2()),

                                temp_trans,
                                static_cast<cl_uint>(temp_trans.start1()),         static_cast<cl_uint>(temp_trans.start2()),
                                static_cast<cl_uint>(temp_trans.internal_size1()), static_cast<cl_uint>(temp_trans.internal_size2()),
                                static_cast<cl_uint>(temp_trans.stride1()),        static_cast<cl_uint>(temp_trans.stride2())));
}

template<typename NumericT>
void matrix_assign(matrix_base<NumericT> & A, NumericT s, bool up_to_internal_size = false)
{
  scalar_matrix<NumericT> B(viennacl::traits::size1(A),viennacl::traits::size2(A),s,viennacl::traits::context(A));
  scheduler::statement statement = scheduler::preset::assign_cpu(&A, &B);

  dynamic_cast<device_specific::matrix_axpy_template*>(kernels::matrix<NumericT>::execution_handler(A.row_major(), viennacl::traits::opencl_context(A)).template_of("assign_cpu"))->up_to_internal_size(up_to_internal_size);
  kernels::matrix<NumericT>::execution_handler(A.row_major(), viennacl::traits::opencl_context(A)).execute("assign_cpu", statement);
}

template<typename NumericT>
void matrix_diagonal_assign(matrix_base<NumericT> & A, NumericT s)
{
  viennacl::scalar_vector<NumericT> sx(std::min(viennacl::traits::size1(A), viennacl::traits::size2(A)), s);
  scheduler::statement statement = scheduler::preset::diagonal_assign_cpu(&A, &sx);
  kernels::matrix<NumericT>::execution_handler(A.row_major(), viennacl::traits::opencl_context(A)).execute("diagonal_assign_cpu", statement);
}

template<typename NumericT>
void matrix_diag_from_vector(const vector_base<NumericT> & vec, int k, matrix_base<NumericT> & A)
{
  scheduler::statement statement = scheduler::preset::matrix_diag_from_vector(&vec, &A, k);
  kernels::matrix<NumericT>::execution_handler(A.row_major(), viennacl::traits::opencl_context(A)).execute("matrix_diag_from_vector", statement);
}

template<typename NumericT>
void matrix_diag_to_vector(const matrix_base<NumericT> & A, int k, vector_base<NumericT> & vec)
{
  scheduler::statement statement = scheduler::preset::matrix_diag_to_vector(&vec, &A, k);
  kernels::matrix<NumericT>::execution_handler(A.row_major(), viennacl::traits::opencl_context(A)).execute("matrix_diag_to_vector", statement);
}

template<typename NumericT>
void matrix_row(const matrix_base<NumericT> & A, unsigned int i, vector_base<NumericT> & vec)
{
  scheduler::statement statement = scheduler::preset::matrix_row(&vec, &A, i);
  kernels::matrix<NumericT>::execution_handler(A.row_major(), viennacl::traits::opencl_context(A)).execute("matrix_row", statement);
}

template<typename NumericT>
void matrix_column(const matrix_base<NumericT> & A, unsigned int j, vector_base<NumericT> & vec)
{
  scheduler::statement statement = scheduler::preset::matrix_column(&vec, &A, j);
  kernels::matrix<NumericT>::execution_handler(A.row_major(), viennacl::traits::opencl_context(A)).execute("matrix_column", statement);
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
template<typename NumericT, typename OpT>
void element_op(matrix_base<NumericT> & A,
                matrix_expression<const matrix_base<NumericT>, const matrix_base<NumericT>, op_element_binary<OpT> > const & proxy)
{
  assert(A.row_major() == proxy.lhs().row_major() && bool("Elementwise operations on mixed matrix layouts not supported yet!"));
  assert(A.row_major() == proxy.rhs().row_major() && bool("Elementwise operations on mixed matrix layouts not supported yet!"));
  assert(viennacl::traits::opencl_handle(A).context() == viennacl::traits::opencl_handle(proxy.lhs()).context() && bool("Matrices do not reside in the same OpenCL context. Automatic migration not yet supported!"));
  assert(viennacl::traits::opencl_handle(A).context() == viennacl::traits::opencl_handle(proxy.rhs()).context() && bool("Matrices do not reside in the same OpenCL context. Automatic migration not yet supported!"));

  scheduler::operation_node_type TYPE = scheduler::operation_node_type(scheduler::result_of::op_type_info<op_element_binary<OpT> >::id);
  scheduler::statement statement =  scheduler::preset::binary_element_op(&A, &proxy.lhs(), &proxy.rhs(),TYPE);
  kernels::matrix_element<NumericT>::execution_handler(A.row_major(), viennacl::traits::opencl_context(A)).execute(device_specific::tree_parsing::operator_string(TYPE), statement);
}


// Unary operations

/** @brief Implementation of unary element-wise operations A = OP(B)
*
* @param A      The result matrix (or -range, or -slice)
* @param proxy  The proxy object holding B and the operation
*/
template<typename NumericT, typename OpT>
void element_op(matrix_base<NumericT> & A,
                matrix_expression<const matrix_base<NumericT>, const matrix_base<NumericT>, op_element_unary<OpT> > const & proxy)
{
  assert(A.row_major() == proxy.lhs().row_major() && bool("Elementwise operations on mixed matrix layouts not supported yet!"));
  assert(A.row_major() == proxy.rhs().row_major() && bool("Elementwise operations on mixed matrix layouts not supported yet!"));

  assert(viennacl::traits::opencl_handle(A).context() == viennacl::traits::opencl_handle(proxy.lhs()).context() && bool("Matrices do not reside in the same OpenCL context. Automatic migration not yet supported!"));
  assert(viennacl::traits::opencl_handle(A).context() == viennacl::traits::opencl_handle(proxy.rhs()).context() && bool("Matrices do not reside in the same OpenCL context. Automatic migration not yet supported!"));

  scheduler::operation_node_type TYPE = scheduler::operation_node_type(scheduler::result_of::op_type_info<op_element_unary<OpT> >::id);
  scheduler::statement statement = scheduler::preset::unary_element_op(&A, &proxy.lhs(),TYPE);
  kernels::matrix_element<NumericT>::execution_handler(A.row_major(), viennacl::traits::opencl_context(A)).execute(device_specific::tree_parsing::operator_string(TYPE), statement);
}



/** @brief Carries out matrix-vector multiplication
*
* Implementation of the convenience expression result = prod(A, vec);
*
* @param A        The matrix
* @param trans_A  Whether the matrix A should be transposed
* @param vec      The vector
* @param result   The result vector
*/
template<typename NumericT>
void prod_impl(const matrix_base<NumericT> & A, bool trans_A,
               const vector_base<NumericT> & vec,
                     vector_base<NumericT> & result)
{
  // Inplace matrix-vector products like x = prod(A, x) are currently illegal: Introduce a temporary like y = prod(A, x); x = y; instead
  assert(viennacl::traits::handle(vec) != viennacl::traits::handle(result) && bool("No direct inplace matrix-vector product possible. Introduce a temporary!"));

  std::string kernel_name = std::string("mat_vec_") + (trans_A ^ A.row_major()?"T":"N");
  scheduler::statement statement = scheduler::preset::mat_vec_prod(&A, trans_A, &vec, &result);
  kernels::row_wise_reduction<NumericT>::execution_handler(viennacl::traits::opencl_context(A)).execute(kernel_name, statement);
}

//


/** @brief Carries out matrix-matrix multiplication
*
* Implementation of C = prod(A, B);
*
*/
template<typename NumericT, typename ScalarType >
void prod_impl(matrix_base<NumericT> const & A, bool A_trans,
               matrix_base<NumericT> const & B, bool B_trans,
               matrix_base<NumericT>       & C,
               ScalarType alpha,
               ScalarType beta)
{
    bool effective_A_trans = A_trans ^ A.row_major();
    bool effective_B_trans = B_trans ^ B.row_major();

    char cAt = effective_A_trans ? 'T' : 'N';
    char cBt = effective_B_trans ? 'T' : 'N';

    std::string kernel_prefix("prod_");
    kernel_prefix+=cAt;
    kernel_prefix+=cBt;

    scheduler::statement statement = scheduler::preset::mat_mat_prod(alpha, &A, effective_A_trans, &B, effective_B_trans, beta, &C);
    kernels::matrix_prod<NumericT>::execution_handler(C.row_major(), viennacl::traits::opencl_context(C)).execute(kernel_prefix, statement);
}

//
/////////////////////////   miscellaneous operations /////////////////////////////////
//


/** @brief The implementation of the operation mat += alpha * vec1 * vec2^T, i.e. a scaled rank 1 update
*
* Implementation of the convenience expression result += alpha * outer_prod(vec1, vec2);
*
* @param A    The matrix to be updated
* @param alpha            The scaling factor (either a viennacl::scalar<>, float, or double)
* @param len_alpha        Length of the buffer for an eventual final reduction step (currently always '1')
* @param reciprocal_alpha Use 1/alpha instead of alpha
* @param flip_sign_alpha  Use -alpha instead of alpha
* @param vec1    The first vector
* @param vec2    The second vector
*/
template<typename NumericT, typename ScalarT1>
void scaled_rank_1_update(matrix_base<NumericT> & A,
                          ScalarT1 const & alpha, vcl_size_t len_alpha, bool reciprocal_alpha, bool flip_sign_alpha,
                          const vector_base<NumericT> & vec1,
                          const vector_base<NumericT> & vec2)
{
  assert( (viennacl::traits::size1(A) == viennacl::traits::size(vec1)) && bool("Size mismatch in scaled_rank_1_update: size1(A) != size(v1)"));
  assert( (viennacl::traits::size2(A) == viennacl::traits::size(vec2)) && bool("Size mismatch in scaled_rank_1_update: size2(A) != size(v2)"));

  cl_uint options_alpha = detail::make_options(len_alpha, reciprocal_alpha, flip_sign_alpha);
  bool is_cpu = viennacl::is_cpu_scalar<ScalarT1>::value;
  viennacl::ocl::kernel& kernel= detail::legacy_kernel_for_matrix(A, is_cpu ? "scaled_rank1_update_cpu" : "scaled_rank1_update_gpu");

  viennacl::ocl::enqueue(kernel(viennacl::traits::opencl_handle(A),
                           cl_uint(viennacl::traits::start1(A)),           cl_uint(viennacl::traits::start2(A)),
                           cl_uint(viennacl::traits::stride1(A)),          cl_uint(viennacl::traits::stride2(A)),
                           cl_uint(viennacl::traits::size1(A)),            cl_uint(viennacl::traits::size2(A)),
                           cl_uint(viennacl::traits::internal_size1(A)),   cl_uint(viennacl::traits::internal_size2(A)),

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

//
template <typename SCALARTYPE, typename VectorType>
void bidiag_pack_svd(viennacl::matrix<SCALARTYPE>& A,
                 VectorType & dh,
                 VectorType & sh
                )
{
  viennacl::vector<SCALARTYPE> D(dh.size());
  viennacl::vector<SCALARTYPE> S(sh.size());

  viennacl::ocl::context & ctx = const_cast<viennacl::ocl::context &>(viennacl::traits::opencl_handle(A).context());
  viennacl::ocl::kernel& kernel = ctx.get_kernel(viennacl::linalg::opencl::kernels::svd<SCALARTYPE>::program_name(), SVD_BIDIAG_PACK_KERNEL);

  viennacl::ocl::enqueue(kernel(
                                A,
                                D,
                                S,
                                static_cast<cl_uint>(A.size1()),
                                static_cast<cl_uint>(A.size2()),
                                static_cast<cl_uint>(A.internal_size2())
                              ));

  fast_copy(D, dh);
  fast_copy(S, sh);
}


template <typename NumericT>
void bidiag_pack(matrix_base<NumericT> & A,
                 viennacl::vector<NumericT> & dh,
                 viennacl::vector<NumericT> & sh
                )
{
  viennacl::ocl::context & ctx = const_cast<viennacl::ocl::context &>(viennacl::traits::opencl_handle(A).context());

  if(A.row_major())
  {
      viennacl::linalg::opencl::kernels::svd<NumericT, row_major>::init(ctx);
      viennacl::ocl::kernel& kernel = ctx.get_kernel(viennacl::linalg::opencl::kernels::svd<NumericT, row_major>::program_name(), SVD_BIDIAG_PACK_KERNEL);

      viennacl::ocl::enqueue(kernel(
                                    A,
                                    dh,
                                    sh,
                                    cl_uint(viennacl::traits::size1(A)),
                                    cl_uint(viennacl::traits::size2(A)),
                                    cl_uint(viennacl::traits::internal_size2(A))
                                  ));
  }
  else
  {
      viennacl::linalg::opencl::kernels::svd<NumericT, column_major>::init(ctx);
      viennacl::ocl::kernel& kernel = ctx.get_kernel(viennacl::linalg::opencl::kernels::svd<NumericT, column_major>::program_name(), SVD_BIDIAG_PACK_KERNEL);

      viennacl::ocl::enqueue(kernel(
                                    A,
                                    dh,
                                    sh,
                                    cl_uint(viennacl::traits::size1(A)),
                                    cl_uint(viennacl::traits::size2(A)),
                                    cl_uint(viennacl::traits::internal_size2(A))
                                  ));
  }
}


template <typename NumericT>
void house_update_A_left(matrix_base<NumericT> & A,
                         vector_base<NumericT> & D,
                         vcl_size_t start)
{

    viennacl::ocl::context & ctx = const_cast<viennacl::ocl::context &>(viennacl::traits::opencl_handle(A).context());
    if(A.row_major())
    {
        viennacl::linalg::opencl::kernels::svd<NumericT, row_major>::init(ctx);
        viennacl::ocl::kernel& kernel = ctx.get_kernel(viennacl::linalg::opencl::kernels::svd<NumericT, row_major>::program_name(), SVD_HOUSEHOLDER_UPDATE_A_LEFT_KERNEL);
        viennacl::ocl::enqueue(kernel(
                                      A,
                                      D,
                                      static_cast<cl_uint>(start + 1),
                                      static_cast<cl_uint>(start),
                                      cl_uint(viennacl::traits::size1(A)),
                                      cl_uint(viennacl::traits::size2(A)),
                                      cl_uint(viennacl::traits::internal_size2(A)),
                                      viennacl::ocl::local_mem(static_cast<cl_uint>(128 * 4))
                              ));
    }
    else
    {
        viennacl::linalg::opencl::kernels::svd<NumericT, column_major>::init(ctx);
        viennacl::ocl::kernel& kernel = ctx.get_kernel(viennacl::linalg::opencl::kernels::svd<NumericT, column_major>::program_name(), SVD_HOUSEHOLDER_UPDATE_A_LEFT_KERNEL);
        viennacl::ocl::enqueue(kernel(
                                      A,
                                      D,
                                      static_cast<cl_uint>(start + 1),
                                      static_cast<cl_uint>(start),
                                      cl_uint(viennacl::traits::size1(A)),
                                      cl_uint(viennacl::traits::size2(A)),
                                      cl_uint(viennacl::traits::internal_size2(A)),
                                      viennacl::ocl::local_mem(static_cast<cl_uint>(128 * 4))
                              ));
    }




}

template <typename NumericT>
void house_update_A_right(matrix_base<NumericT> & A,
                          vector_base<NumericT> & D)
{
    viennacl::ocl::context & ctx = const_cast<viennacl::ocl::context &>(viennacl::traits::opencl_handle(A).context());

    if(A.row_major())
    {
        viennacl::linalg::opencl::kernels::svd<NumericT, row_major>::init(ctx);
        viennacl::ocl::kernel& kernel = ctx.get_kernel(viennacl::linalg::opencl::kernels::svd<NumericT, row_major>::program_name(), SVD_HOUSEHOLDER_UPDATE_A_RIGHT_KERNEL);

        viennacl::ocl::enqueue(kernel(
                                      A,
                                      D,
                                      static_cast<cl_uint>(0),
                                      static_cast<cl_uint>(0),
                                      cl_uint(viennacl::traits::size1(A)),
                                      cl_uint(viennacl::traits::size2(A)),
                                      cl_uint(viennacl::traits::internal_size2(A)),
                                      viennacl::ocl::local_mem(static_cast<cl_uint>(128 * sizeof(NumericT)))
                              ));
    }
    else
    {
        viennacl::linalg::opencl::kernels::svd<NumericT, column_major>::init(ctx);
        viennacl::ocl::kernel& kernel = ctx.get_kernel(viennacl::linalg::opencl::kernels::svd<NumericT, column_major>::program_name(), SVD_HOUSEHOLDER_UPDATE_A_RIGHT_KERNEL);

        viennacl::ocl::enqueue(kernel(
                                      A,
                                      D,
                                      static_cast<cl_uint>(0),
                                      static_cast<cl_uint>(0),
                                      cl_uint(viennacl::traits::size1(A)),
                                      cl_uint(viennacl::traits::size2(A)),
                                      cl_uint(viennacl::traits::internal_size2(A)),
                                      viennacl::ocl::local_mem(static_cast<cl_uint>(128 * sizeof(NumericT)))
                              ));
    }


}



template <typename NumericT>
void house_update_QL(matrix_base<NumericT> & Q,
                     vector_base<NumericT> & D,
                     vcl_size_t A_size1)

{
    viennacl::ocl::context & ctx = const_cast<viennacl::ocl::context &>(viennacl::traits::opencl_handle(Q).context());

    if(Q.row_major())
    {
        viennacl::linalg::opencl::kernels::svd<NumericT, row_major>::init(ctx);
        viennacl::ocl::kernel& kernel = ctx.get_kernel(viennacl::linalg::opencl::kernels::svd<NumericT, row_major>::program_name(), SVD_HOUSEHOLDER_UPDATE_QL_KERNEL);

        viennacl::ocl::enqueue(kernel(
                                        Q,
                                        D,
                                        cl_uint(A_size1),
                                        cl_uint(viennacl::traits::internal_size2(Q)),
                                        viennacl::ocl::local_mem(static_cast<cl_uint>(128 * sizeof(NumericT)))
                                    ));
    }
    else
    {
        viennacl::linalg::opencl::kernels::svd<NumericT, column_major>::init(ctx);
        viennacl::ocl::kernel& kernel = ctx.get_kernel(viennacl::linalg::opencl::kernels::svd<NumericT, column_major>::program_name(), SVD_HOUSEHOLDER_UPDATE_QL_KERNEL);

        viennacl::ocl::enqueue(kernel(
                                        Q,
                                        D,
                                        cl_uint(A_size1),
                                        cl_uint(viennacl::traits::internal_size2(Q)),
                                        viennacl::ocl::local_mem(static_cast<cl_uint>(128 * sizeof(NumericT)))
                                    ));
    }

}


template<typename NumericT>
  void givens_next(matrix_base<NumericT> & matrix,
                  vector_base<NumericT>& tmp1,
                  vector_base<NumericT>& tmp2,
                  int l,
                  int m
                )
  {
    viennacl::ocl::context & ctx = const_cast<viennacl::ocl::context &>(viennacl::traits::opencl_handle(matrix).context());

    if(matrix.row_major())
    {
        viennacl::linalg::opencl::kernels::svd<NumericT, row_major>::init(ctx);
        viennacl::ocl::kernel& kernel = ctx.get_kernel(viennacl::linalg::opencl::kernels::svd<NumericT, row_major>::program_name(), SVD_GIVENS_NEXT_KERNEL);
        kernel.global_work_size(0, viennacl::tools::align_to_multiple<cl_uint>(cl_uint(viennacl::traits::size1(matrix)), 256));
        kernel.local_work_size(0, 256);

        viennacl::ocl::enqueue(kernel(
                                      matrix,
                                      tmp1,
                                      tmp2,
                                      cl_uint(viennacl::traits::size1(matrix)),
                                      cl_uint(viennacl::traits::internal_size2(matrix)),
                                      static_cast<cl_uint>(l),
                                      static_cast<cl_uint>(m - 1)
                              ));
    }
    else
    {
        viennacl::linalg::opencl::kernels::svd<NumericT, column_major>::init(ctx);
        viennacl::ocl::kernel& kernel = ctx.get_kernel(viennacl::linalg::opencl::kernels::svd<NumericT, column_major>::program_name(), SVD_GIVENS_NEXT_KERNEL);
        kernel.global_work_size(0, viennacl::tools::align_to_multiple<cl_uint>(cl_uint(viennacl::traits::size1(matrix)), 256));
        kernel.local_work_size(0, 256);

        viennacl::ocl::enqueue(kernel(
                                      matrix,
                                      tmp1,
                                      tmp2,
                                      cl_uint(viennacl::traits::size1(matrix)),
                                      cl_uint(viennacl::traits::internal_size2(matrix)),
                                      static_cast<cl_uint>(l),
                                      static_cast<cl_uint>(m - 1)
                              ));
    }


  }

  template <typename NumericT>
  void copy_vec(matrix_base<NumericT>& A,
                vector_base<NumericT> & V,
                vcl_size_t row_start,
                vcl_size_t col_start,
                bool copy_col
  )
  {
    std::string kernel_name = copy_col ? SVD_COPY_COL_KERNEL : SVD_COPY_ROW_KERNEL;
    viennacl::ocl::context & ctx = const_cast<viennacl::ocl::context &>(viennacl::traits::opencl_handle(A).context());

    if(A.row_major())
    {
        viennacl::linalg::opencl::kernels::svd<NumericT, row_major>::init(ctx);
        viennacl::ocl::kernel& kernel = ctx.get_kernel(viennacl::linalg::opencl::kernels::svd<NumericT, row_major>::program_name(), kernel_name);

        viennacl::ocl::enqueue(kernel(
                                      A,
                                      V,
                                      static_cast<cl_uint>(row_start),
                                      static_cast<cl_uint>(col_start),
                                      copy_col ? cl_uint(viennacl::traits::size1(A))
                                               : cl_uint(viennacl::traits::size2(A)),
                                      static_cast<cl_uint>(A.internal_size2())
                              ));
    }
    else
    {
        viennacl::linalg::opencl::kernels::svd<NumericT, column_major>::init(ctx);
        viennacl::ocl::kernel& kernel = ctx.get_kernel(viennacl::linalg::opencl::kernels::svd<NumericT, column_major>::program_name(), kernel_name);

        viennacl::ocl::enqueue(kernel(
                                      A,
                                      V,
                                      static_cast<cl_uint>(row_start),
                                      static_cast<cl_uint>(col_start),
                                      copy_col ? cl_uint(viennacl::traits::size1(A))
                                               : cl_uint(viennacl::traits::size2(A)),
                                      static_cast<cl_uint>(A.internal_size2())
                              ));
    }


  }

} // namespace opencl
} //namespace linalg
} //namespace viennacl


#endif
