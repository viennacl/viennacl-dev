#ifndef VIENNACL_LINALG_OPENCL_MATRIX_OPERATIONS_HPP_
#define VIENNACL_LINALG_OPENCL_MATRIX_OPERATIONS_HPP_

/* =========================================================================
   Copyright (c) 2010-2012, Institute for Microelectronics,
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
#include "viennacl/traits/size.hpp"
#include "viennacl/traits/start.hpp"
#include "viennacl/traits/handle.hpp"
#include "viennacl/traits/stride.hpp"
#include "viennacl/tools/matrix_kernel_class_deducer.hpp"
#include "viennacl/tools/matrix_prod_kernel_class_deducer.hpp"
#include "viennacl/linalg/kernels/vector_kernels.h"
#include "viennacl/linalg/kernels/matrix_row_kernels.h"
#include "viennacl/linalg/kernels/matrix_col_kernels.h"

#include "viennacl/linalg/kernels/matrix_prod_col_col_col_kernels.h"
#include "viennacl/linalg/kernels/matrix_prod_col_col_row_kernels.h"
#include "viennacl/linalg/kernels/matrix_prod_col_row_col_kernels.h"
#include "viennacl/linalg/kernels/matrix_prod_col_row_row_kernels.h"

#include "viennacl/linalg/kernels/matrix_prod_row_col_col_kernels.h"
#include "viennacl/linalg/kernels/matrix_prod_row_col_row_kernels.h"
#include "viennacl/linalg/kernels/matrix_prod_row_row_col_kernels.h"
#include "viennacl/linalg/kernels/matrix_prod_row_row_row_kernels.h"

namespace viennacl
{
  namespace linalg
  {
    namespace opencl
    {
      //
      // Introductory note: By convention, all dimensions are already checked in the dispatcher frontend. No need to double-check again in here!
      //
      
      template <typename M1,
                typename M2, typename ScalarType1>
      typename viennacl::enable_if< viennacl::is_any_dense_nonstructured_matrix<M1>::value
                                    && viennacl::is_any_dense_nonstructured_matrix<M2>::value
                                    && viennacl::is_any_scalar<ScalarType1>::value
                                  >::type
      am(M1 & mat1, 
        M2 const & mat2, ScalarType1 const & alpha, std::size_t len_alpha, bool reciprocal_alpha, bool flip_sign_alpha) 
      {
        typedef typename viennacl::result_of::cpu_value_type<M1>::type        value_type;

        typedef typename viennacl::tools::MATRIX_KERNEL_CLASS_DEDUCER<M1>::ResultType    KernelClass;
        KernelClass::init();
        
        cl_uint options_alpha =   ((len_alpha > 1) ? (len_alpha << 2) : 0)
                                + (reciprocal_alpha ? 2 : 0)
                                + (flip_sign_alpha ? 1 : 0);
                                
        viennacl::ocl::kernel & k = viennacl::ocl::get_kernel(KernelClass::program_name(),
                                                              (viennacl::is_cpu_scalar<ScalarType1>::value ? "am_cpu" : "am_gpu"));
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
      
      
      template <typename M1,
                typename M2, typename ScalarType1,
                typename M3, typename ScalarType2>
      typename viennacl::enable_if< viennacl::is_any_dense_nonstructured_matrix<M1>::value
                                    && viennacl::is_any_dense_nonstructured_matrix<M2>::value
                                    && viennacl::is_any_dense_nonstructured_matrix<M3>::value
                                    && viennacl::is_any_scalar<ScalarType1>::value
                                    && viennacl::is_any_scalar<ScalarType2>::value
                                  >::type
      ambm(M1 & mat1, 
          M2 const & mat2, ScalarType1 const & alpha, std::size_t len_alpha, bool reciprocal_alpha, bool flip_sign_alpha,
          M3 const & mat3, ScalarType2 const & beta, std::size_t len_beta, bool reciprocal_beta, bool flip_sign_beta) 
      {
        typedef typename viennacl::result_of::cpu_value_type<M1>::type        value_type;
        typedef typename viennacl::tools::MATRIX_KERNEL_CLASS_DEDUCER<M1>::ResultType    KernelClass;
        KernelClass::init();
        
        
        std::string kernel_name;
        if      ( viennacl::is_cpu_scalar<ScalarType1>::value &&  viennacl::is_cpu_scalar<ScalarType2>::value)
          kernel_name = "ambm_cpu_cpu";
        else if ( viennacl::is_cpu_scalar<ScalarType1>::value && !viennacl::is_cpu_scalar<ScalarType2>::value)
          kernel_name = "ambm_cpu_gpu";
        else if (!viennacl::is_cpu_scalar<ScalarType1>::value &&  viennacl::is_cpu_scalar<ScalarType2>::value)
          kernel_name = "ambm_gpu_cpu";
        else 
          kernel_name = "ambm_gpu_gpu";
          
        cl_uint options_alpha =   ((len_alpha > 1) ? (len_alpha << 2) : 0)
                                + (reciprocal_alpha ? 2 : 0)
                                + (flip_sign_alpha ? 1 : 0);
        cl_uint options_beta =    ((len_beta > 1) ? (len_beta << 2) : 0)
                                + (reciprocal_beta ? 2 : 0)
                                + (flip_sign_beta ? 1 : 0);

        viennacl::ocl::kernel & k = viennacl::ocl::get_kernel(KernelClass::program_name(), kernel_name);
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
      
      
      template <typename M1,
                typename M2, typename ScalarType1,
                typename M3, typename ScalarType2>
      typename viennacl::enable_if< viennacl::is_any_dense_nonstructured_matrix<M1>::value
                                    && viennacl::is_any_dense_nonstructured_matrix<M2>::value
                                    && viennacl::is_any_dense_nonstructured_matrix<M3>::value
                                    && viennacl::is_any_scalar<ScalarType1>::value
                                    && viennacl::is_any_scalar<ScalarType2>::value
                                  >::type
      ambm_m(M1 & mat1,
             M2 const & mat2, ScalarType1 const & alpha, std::size_t len_alpha, bool reciprocal_alpha, bool flip_sign_alpha,
             M3 const & mat3, ScalarType2 const & beta,  std::size_t len_beta,  bool reciprocal_beta,  bool flip_sign_beta) 
      {
        typedef typename viennacl::result_of::cpu_value_type<M1>::type        value_type;
        typedef typename viennacl::tools::MATRIX_KERNEL_CLASS_DEDUCER<M1>::ResultType    KernelClass;
        KernelClass::init();
        
        
        std::string kernel_name;
        if      ( viennacl::is_cpu_scalar<ScalarType1>::value &&  viennacl::is_cpu_scalar<ScalarType2>::value)
          kernel_name = "ambm_m_cpu_cpu";
        else if ( viennacl::is_cpu_scalar<ScalarType1>::value && !viennacl::is_cpu_scalar<ScalarType2>::value)
          kernel_name = "ambm_m_cpu_gpu";
        else if (!viennacl::is_cpu_scalar<ScalarType1>::value &&  viennacl::is_cpu_scalar<ScalarType2>::value)
          kernel_name = "ambm_m_gpu_cpu";
        else 
          kernel_name = "ambm_m_gpu_gpu";
          
        cl_uint options_alpha =   ((len_alpha > 1) ? (len_alpha << 2) : 0)
                                + (reciprocal_alpha ? 2 : 0)
                                + (flip_sign_alpha ? 1 : 0);
        cl_uint options_beta =    ((len_beta > 1) ? (len_beta << 2) : 0)
                                + (reciprocal_beta ? 2 : 0)
                                + (flip_sign_beta ? 1 : 0);

        viennacl::ocl::kernel & k = viennacl::ocl::get_kernel(KernelClass::program_name(), kernel_name);
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


      
      template <typename M1, typename ScalarType>
      typename viennacl::enable_if<    viennacl::is_any_dense_nonstructured_matrix<M1>::value
                                    && viennacl::is_cpu_scalar<ScalarType>::value
                                  >::type    
      matrix_assign(M1 & mat, ScalarType s)
      {
        typedef typename viennacl::result_of::cpu_value_type<M1>::type        value_type;
        typedef typename viennacl::tools::MATRIX_KERNEL_CLASS_DEDUCER<M1>::ResultType    KernelClass;
        KernelClass::init();
        value_type alpha = static_cast<value_type>(s);
        
        viennacl::ocl::kernel & k = viennacl::ocl::get_kernel(KernelClass::program_name(), "assign_cpu");
        viennacl::ocl::enqueue(k(viennacl::traits::opencl_handle(mat),
                                 cl_uint(viennacl::traits::start1(mat)),           cl_uint(viennacl::traits::start2(mat)),
                                 cl_uint(viennacl::traits::stride1(mat)),          cl_uint(viennacl::traits::stride2(mat)),
                                 cl_uint(viennacl::traits::size1(mat)),            cl_uint(viennacl::traits::size2(mat)),
                                 cl_uint(viennacl::traits::internal_size1(mat)),   cl_uint(viennacl::traits::internal_size2(mat)),
                                 alpha
                                )
                              );
      }
      
      template <typename M1, typename ScalarType>
      typename viennacl::enable_if<    viennacl::is_any_dense_nonstructured_matrix<M1>::value
                                    && viennacl::is_cpu_scalar<ScalarType>::value
                                  >::type    
      matrix_diagonal_assign(M1 & mat, ScalarType s)
      {
        typedef typename viennacl::result_of::cpu_value_type<M1>::type        value_type;
        typedef typename viennacl::tools::MATRIX_KERNEL_CLASS_DEDUCER<M1>::ResultType    KernelClass;
        KernelClass::init();
        
        value_type alpha = static_cast<value_type>(s);
        
        viennacl::ocl::kernel & k = viennacl::ocl::get_kernel(KernelClass::program_name(), "diagonal_assign_cpu");
        viennacl::ocl::enqueue(k(viennacl::traits::opencl_handle(mat),
                                 cl_uint(viennacl::traits::start1(mat)),           cl_uint(viennacl::traits::start2(mat)),
                                 cl_uint(viennacl::traits::stride1(mat)),          cl_uint(viennacl::traits::stride2(mat)),
                                 cl_uint(viennacl::traits::size1(mat)),            cl_uint(viennacl::traits::size2(mat)),
                                 cl_uint(viennacl::traits::internal_size1(mat)),   cl_uint(viennacl::traits::internal_size2(mat)),
                                 alpha
                                )
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
      template <typename MatrixType, typename VectorType1, typename VectorType2>
      typename viennacl::enable_if<   viennacl::is_any_dense_nonstructured_matrix<MatrixType>::value 
                                    && viennacl::is_any_dense_nonstructured_vector<VectorType1>::value 
                                    && viennacl::is_any_dense_nonstructured_vector<VectorType2>::value >::type
      prod_impl(const MatrixType & mat, 
                const VectorType1 & vec, 
                      VectorType2 & result)
      {
        typedef typename viennacl::result_of::cpu_value_type<VectorType1>::type        value_type;
        typedef typename viennacl::tools::MATRIX_KERNEL_CLASS_DEDUCER<MatrixType>::ResultType    KernelClass;
        KernelClass::init();
        
        
        assert(mat.size2() == vec.size());
        // Inplace matrix-vector products like x = prod(A, x) are currently illegal: Introduce a temporary like y = prod(A, x); x = y; instead
        assert(viennacl::traits::handle(vec) != viennacl::traits::handle(result) && bool("No direct inplace matrix-vector product possible. Introduce a temporary!"));
        //result.resize(mat.size1());

        viennacl::ocl::kernel & k = viennacl::ocl::get_kernel(KernelClass::program_name(), "vec_mul");
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


      // trans(A) * x
      
      /** @brief Carries out matrix-vector multiplication with a transposed matrix
      *
      * Implementation of the convenience expression result = trans(mat) * vec;
      *
      * @param mat_trans  The transposed matrix proxy
      * @param vec        The vector
      * @param result     The result vector
      */
      template <typename M1, typename V1, typename V2>
      typename viennacl::enable_if<    viennacl::is_any_dense_nonstructured_matrix<M1>::value 
                                    && viennacl::is_any_dense_nonstructured_vector<V1>::value 
                                    && viennacl::is_any_dense_nonstructured_vector<V2>::value 
                                  >::type
      prod_impl(const viennacl::matrix_expression< const M1,
                                                   const M1,
                                                   op_trans> & mat_trans,
                const V1 & vec, 
                      V2 & result)
      {
        assert( (viennacl::traits::size1(mat_trans) == viennacl::traits::size(result)) && bool("Size check failed for transposed matrix-vector product: size1(A^T) == size(result)"));
        assert( (viennacl::traits::size2(mat_trans) == viennacl::traits::size(vec)) && bool("Size check failed for transposed matrix-vector product: size2(A^T) == size(x)"));  //remember: mat is transposed!
        
        typedef typename viennacl::result_of::cpu_value_type<V1>::type    value_type;
        typedef typename viennacl::tools::MATRIX_KERNEL_CLASS_DEDUCER<M1>::ResultType    KernelClass;
        KernelClass::init();
        
        
        // Inplace matrix-vector products like x = prod(A, x) are currently illegal: Introduce a temporary like y = prod(A, x); x = y; instead
        assert(viennacl::traits::handle(vec) != viennacl::traits::handle(result) && bool("No direct inplace transposed matrix-vector product possible. Introduce a temporary!"));
        result.resize(viennacl::traits::size1(mat_trans));

        viennacl::ocl::kernel & k = viennacl::ocl::get_kernel(KernelClass::program_name(), "trans_vec_mul");
        
        viennacl::ocl::enqueue(k(viennacl::traits::opencl_handle(mat_trans.lhs()),
                                cl_uint(viennacl::traits::start1(mat_trans.lhs())),         cl_uint(viennacl::traits::start2(mat_trans.lhs())), 
                                cl_uint(viennacl::traits::stride1(mat_trans.lhs())),        cl_uint(viennacl::traits::stride2(mat_trans.lhs())),
                                cl_uint(viennacl::traits::size1(mat_trans.lhs())),          cl_uint(viennacl::traits::size2(mat_trans.lhs())),
                                cl_uint(viennacl::traits::internal_size1(mat_trans.lhs())), cl_uint(viennacl::traits::internal_size2(mat_trans.lhs())),
                                 
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
          
          typedef typename viennacl::tools::MATRIX_PROD_KERNEL_CLASS_DEDUCER< T1, T2, T3 >::ResultType    KernelClass;
          KernelClass::init();
          
          //std::cout << "KernelClass::program_name() : " << KernelClass::program_name() << std::endl;
          viennacl::ocl::kernel & k = viennacl::ocl::get_kernel(KernelClass::program_name(), kernel_name);
          
          k.global_work_size(0, viennacl::tools::roundUpToNextMultiple<unsigned int>(viennacl::traits::size1(C), 16));
          k.global_work_size(1, viennacl::tools::roundUpToNextMultiple<unsigned int>(viennacl::traits::size2(C), 16));
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
        
        // C = A * B, using fast kernel
        template <typename T1, typename T2, typename T3, typename ScalarType >
        void prod_fast_kernel(const T1 & A, 
                              const T2 & B, 
                              T3 & C,
                              ScalarType alpha,
                              ScalarType beta,
                              std::string kernel_name)
        {
          typedef typename viennacl::result_of::cpu_value_type< typename T1::value_type >::type   cpu_value_type;
          
          typedef typename viennacl::tools::MATRIX_PROD_KERNEL_CLASS_DEDUCER< T1, T2, T3 >::ResultType    KernelClass;
          KernelClass::init();
          
          //std::cout << "KernelClass::program_name() : " << KernelClass::program_name() << std::endl;
          viennacl::ocl::kernel & k = viennacl::ocl::get_kernel(KernelClass::program_name(), kernel_name);
          
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
              || (viennacl::traits::size1(B) < 64) )   //there is most likely not enough to compute, rendering kernel launch overhead considerable
          {
            prod_slow_kernel(A, B, C, alpha, beta, slow_kernel_name);
          }
          else if (   (viennacl::traits::size1(A) % 64 == 0)
                  && (viennacl::traits::size2(A) % 64 == 0)
                  && (viennacl::traits::size1(B) % 64 == 0) )   // allows the use of the fast kernel only
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
      template <typename T1, typename T2, typename T3, typename ScalarType >
      typename viennacl::enable_if<    viennacl::is_any_dense_nonstructured_matrix<T1>::value
                                    && viennacl::is_any_dense_nonstructured_matrix<T2>::value
                                    && viennacl::is_any_dense_nonstructured_matrix<T3>::value
                                  >::type
      prod_impl(const T1 & A, 
                const T2 & B, 
                      T3 & C,
                ScalarType alpha,
                ScalarType beta)
      {
        assert( (viennacl::traits::size1(A) == viennacl::traits::size1(C)) && bool("Size mismatch in C = prod(A, B): size1(A) != size1(C)"));
        assert( (viennacl::traits::size2(A) == viennacl::traits::size1(B)) && bool("Size mismatch in C = prod(A, B): size2(A) != size1(B)"));
        assert( (viennacl::traits::size2(B) == viennacl::traits::size2(C)) && bool("Size mismatch in C = prod(A, B): size2(B) != size2(C)"));
        
        // Inplace matrix-vector products like B = prod(A, B) are currently illegal: Introduce a temporary like C = prod(A, B); B = C; instead
        /*assert(  (viennacl::traits::handle(C) != viennacl::traits::handle(A))
              && (viennacl::traits::handle(C) != viennacl::traits::handle(B))
              && bool("No direct inplace matrix-matrix product possible. Introduce a temporary!"));*/

        
        detail::prod(A, B, C, alpha, beta, "prod16_AA", "prod_AA");
      }



      /** @brief Carries out matrix-matrix multiplication
      *
      * Implementation of C = prod(trans(A), B);
      *
      */
      template <typename T1, typename T2, typename T3, typename ScalarType >
      typename viennacl::enable_if<    viennacl::is_any_dense_nonstructured_matrix<T1>::value
                                    && viennacl::is_any_dense_nonstructured_matrix<T2>::value
                                    && viennacl::is_any_dense_nonstructured_matrix<T3>::value
                                  >::type
      prod_impl(const viennacl::matrix_expression< const T1,
                                                  const T1,
                                                  op_trans> & A, 
                const T2 & B, 
                      T3 & C,
                ScalarType alpha,
                ScalarType beta)
      {
        //std::cout << "size2(A): " << viennacl::traits::size2(A.lhs()) << std::endl;
        //std::cout << "size1(C): " << viennacl::traits::size1(C) << std::endl;
        assert( (viennacl::traits::size2(A.lhs()) == viennacl::traits::size1(C)) && bool("Size mismatch in C = prod(trans(A), B): size2(A) != size1(C)"));
        assert( (viennacl::traits::size1(A.lhs()) == viennacl::traits::size1(B)) && bool("Size mismatch in C = prod(trans(A), B): size1(A) != size1(B)"));
        assert( (viennacl::traits::size2(B)       == viennacl::traits::size2(C)) && bool("Size mismatch in C = prod(trans(A), B): size2(B) != size2(C)"));
        
        // Inplace matrix-vector products like B = prod(A, B) are currently illegal: Introduce a temporary like C = prod(A, B); B = C; instead
        /*assert(  (viennacl::traits::handle(C) != viennacl::traits::handle(A.lhs()))
              && (viennacl::traits::handle(C) != viennacl::traits::handle(B))
              && bool("No direct inplace matrix-matrix product possible. Introduce a temporary!"));*/
        
        detail::prod(A.lhs(), B, C, alpha, beta, "prod16_TA", "prod_TA");
      }




      /** @brief Carries out matrix-matrix multiplication
      *
      * Implementation of C = prod(A, trans(B));
      *
      */
      template <typename T1, typename T2, typename T3, typename ScalarType >
      typename viennacl::enable_if<    viennacl::is_any_dense_nonstructured_matrix<T1>::value
                                    && viennacl::is_any_dense_nonstructured_matrix<T2>::value
                                    && viennacl::is_any_dense_nonstructured_matrix<T3>::value
                                  >::type
      prod_impl(const T1 & A, 
                const viennacl::matrix_expression< const T2,
                                                  const T2,
                                                  op_trans> & B,
                      T3 & C,
                ScalarType alpha,
                ScalarType beta)
      {
        assert( (viennacl::traits::size1(A)       == viennacl::traits::size1(C))       && bool("Size mismatch in C = prod(A, trans(B)): size1(A) != size1(C)"));
        assert( (viennacl::traits::size2(A)       == viennacl::traits::size2(B.lhs())) && bool("Size mismatch in C = prod(A, trans(B)): size2(A) != size2(B)"));
        assert( (viennacl::traits::size1(B.lhs()) == viennacl::traits::size2(C))       && bool("Size mismatch in C = prod(A, trans(B)): size1(B) != size2(C)"));
        
        // Inplace matrix-vector products like B = prod(A, B) are currently illegal: Introduce a temporary like C = prod(A, B); B = C; instead
        /*assert(  (viennacl::traits::handle(C) != viennacl::traits::handle(A))
              && (viennacl::traits::handle(C) != viennacl::traits::handle(B.lhs()))
              && bool("No direct inplace matrix-matrix product possible. Introduce a temporary!"));*/
        
        detail::prod(A, B.lhs(), C, alpha, beta, "prod16_AT", "prod_AT");
      }



      /** @brief Carries out matrix-matrix multiplication
      *
      * Implementation of C = prod(trans(A), trans(B));
      *
      */
      template <typename T1, typename T2, typename T3, typename ScalarType >
      typename viennacl::enable_if<    viennacl::is_any_dense_nonstructured_matrix<T1>::value
                                    && viennacl::is_any_dense_nonstructured_matrix<T2>::value
                                    && viennacl::is_any_dense_nonstructured_matrix<T3>::value
                                  >::type
      prod_impl(const viennacl::matrix_expression< const T1, const T1, op_trans> & A,
                const viennacl::matrix_expression< const T2, const T2, op_trans> & B,
                T3 & C,
                ScalarType alpha,
                ScalarType beta)
      {
        assert(viennacl::traits::size2(A.lhs()) == viennacl::traits::size1(C)       && bool("Size mismatch in C = prod(trans(A), trans(B)): size2(A) != size1(C)"));
        assert(viennacl::traits::size1(A.lhs()) == viennacl::traits::size2(B.lhs()) && bool("Size mismatch in C = prod(trans(A), trans(B)): size1(A) != size2(B)"));
        assert(viennacl::traits::size1(B.lhs()) == viennacl::traits::size2(C)       && bool("Size mismatch in C = prod(trans(A), trans(B)): size1(B) != size2(C)"));
        
        // Inplace matrix-vector products like B = prod(A, B) are currently illegal: Introduce a temporary like C = prod(A, B); B = C; instead
        /*assert(  (viennacl::traits::handle(C) != viennacl::traits::handle(A.lhs()))
              && (viennacl::traits::handle(C) != viennacl::traits::handle(B.lhs()))
              && bool("No direct inplace matrix-matrix product possible. Introduce a temporary!"));*/
        
        detail::prod(A.lhs(), B.lhs(), C, alpha, beta, "prod16_TT", "prod_TT");
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
      template <typename M1, typename S1, typename V1, typename V2>
      typename viennacl::enable_if<    viennacl::is_any_dense_nonstructured_matrix<M1>::value
                                    && viennacl::is_any_scalar<S1>::value
                                    && viennacl::is_any_dense_nonstructured_vector<V1>::value
                                    && viennacl::is_any_dense_nonstructured_vector<V2>::value
                                  >::type
      scaled_rank_1_update(M1 & mat1,
                    S1 const & alpha, std::size_t len_alpha, bool reciprocal_alpha, bool flip_sign_alpha,
                    const V1 & vec1, 
                    const V2 & vec2)
      {
        assert( (viennacl::traits::size1(mat1) == viennacl::traits::size(vec1)) && bool("Size mismatch in scaled_rank_1_update: size1(A) != size(v1)"));
        assert( (viennacl::traits::size2(mat1) == viennacl::traits::size(vec2)) && bool("Size mismatch in scaled_rank_1_update: size2(A) != size(v2)"));

        typedef typename viennacl::result_of::cpu_value_type<V1>::type        value_type;
        typedef typename viennacl::tools::MATRIX_KERNEL_CLASS_DEDUCER<M1>::ResultType    KernelClass;
        KernelClass::init();
        
        
        cl_uint options_alpha =   ((len_alpha > 1) ? (len_alpha << 2) : 0)
                                + (reciprocal_alpha ? 2 : 0)
                                + (flip_sign_alpha ? 1 : 0);
        
        viennacl::ocl::kernel & k = viennacl::ocl::get_kernel(KernelClass::program_name(), viennacl::is_cpu_scalar<S1>::value ? "scaled_rank1_update_cpu" : "scaled_rank1_update_gpu");

        viennacl::ocl::enqueue(k(viennacl::traits::opencl_handle(mat1), 
                                 cl_uint(viennacl::traits::start1(mat1)),           cl_uint(viennacl::traits::start2(mat1)), 
                                 cl_uint(viennacl::traits::stride1(mat1)),          cl_uint(viennacl::traits::stride2(mat1)),
                                 cl_uint(viennacl::traits::size1(mat1)),            cl_uint(viennacl::traits::size2(mat1)),
                                 cl_uint(viennacl::traits::internal_size1(mat1)),   cl_uint(viennacl::traits::internal_size2(mat1)),
                                 
                                 viennacl::traits::opencl_handle(viennacl::tools::promote_if_host_scalar<value_type>(alpha)),
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
