#ifndef VIENNACL_LINALG_OPENCL_DIRECT_SOLVE_HPP
#define VIENNACL_LINALG_OPENCL_DIRECT_SOLVE_HPP

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

/** @file viennacl/linalg/opencl/direct_solve.hpp
    @brief Implementations of dense direct solvers are found here.
*/

#include "viennacl/vector.hpp"
#include "viennacl/matrix.hpp"
#include "viennacl/tools/matrix_kernel_class_deducer.hpp"
#include "viennacl/tools/matrix_solve_kernel_class_deducer.hpp"
#include "viennacl/ocl/kernel.hpp"
#include "viennacl/ocl/device.hpp"
#include "viennacl/ocl/handle.hpp"


namespace viennacl
{
  namespace linalg
  {
    namespace opencl
    {
      namespace detail
      {
        inline cl_uint get_option_for_solver_tag(viennacl::linalg::upper_tag)      { return 0; }
        inline cl_uint get_option_for_solver_tag(viennacl::linalg::unit_upper_tag) { return (1 << 0); }
        inline cl_uint get_option_for_solver_tag(viennacl::linalg::lower_tag)      { return (1 << 2); }
        inline cl_uint get_option_for_solver_tag(viennacl::linalg::unit_lower_tag) { return (1 << 2) | (1 << 0); }
        
        template <typename M1, typename M2, typename KernelType>
        void inplace_solve_impl(M1 const & A, M2 & B, KernelType & k)
        {
          viennacl::ocl::enqueue(k(viennacl::traits::opencl_handle(A),
                                   cl_uint(viennacl::traits::start1(A)),         cl_uint(viennacl::traits::start2(A)),
                                   cl_uint(viennacl::traits::stride1(A)),        cl_uint(viennacl::traits::stride2(A)),
                                   cl_uint(viennacl::traits::size1(A)),          cl_uint(viennacl::traits::size2(A)),
                                   cl_uint(viennacl::traits::internal_size1(A)), cl_uint(viennacl::traits::internal_size2(A)),
                                   viennacl::traits::opencl_handle(B),
                                   cl_uint(viennacl::traits::start1(B)),         cl_uint(viennacl::traits::start2(B)),
                                   cl_uint(viennacl::traits::stride1(B)),        cl_uint(viennacl::traits::stride2(B)),
                                   cl_uint(viennacl::traits::size1(B)),          cl_uint(viennacl::traits::size2(B)),
                                   cl_uint(viennacl::traits::internal_size1(B)), cl_uint(viennacl::traits::internal_size2(B))
                                  )
                                );        
        }
      }
      
      
      //
      // Note: By convention, all size checks are performed in the calling frontend. No need to double-check here.
      //
      
      ////////////////// upper triangular solver (upper_tag) //////////////////////////////////////
    /** @brief Direct inplace solver for dense triangular systems. Matlab notation: A \ B
    *
    * @param A    The system matrix
    * @param B    The matrix of row vectors, where the solution is directly written to
    */
      template <typename M1,
                typename M2, typename SOLVERTAG>
      typename viennacl::enable_if<    viennacl::is_any_dense_nonstructured_matrix<M1>::value
                                    && viennacl::is_any_dense_nonstructured_matrix<M2>::value
                                  >::type
      inplace_solve(const M1 & A, M2 & B, SOLVERTAG)
      {
        typedef typename viennacl::tools::MATRIX_SOLVE_KERNEL_CLASS_DEDUCER< M1, M2 >::ResultType    KernelClass;
        KernelClass::init();
        
        std::stringstream ss;
        ss << SOLVERTAG::name() << "_solve";
        viennacl::ocl::kernel & k = viennacl::ocl::get_kernel(KernelClass::program_name(), ss.str());

        k.global_work_size(0, B.size2() * k.local_work_size());
        detail::inplace_solve_impl(A, B, k);
      }
      
      /** @brief Direct inplace solver for dense triangular systems with transposed right hand side
      *
      * @param A       The system matrix
      * @param proxy_B The transposed matrix of row vectors, where the solution is directly written to
      */
      template <typename M1,
                typename M2, typename SOLVERTAG>
      typename viennacl::enable_if<    viennacl::is_any_dense_nonstructured_matrix<M1>::value
                                    && viennacl::is_any_dense_nonstructured_matrix<M2>::value
                                  >::type
      inplace_solve(const M1 & A,
                    matrix_expression< const M2, const M2, op_trans> proxy_B,
                    SOLVERTAG)
      {
        typedef typename viennacl::tools::MATRIX_SOLVE_KERNEL_CLASS_DEDUCER< M1, M2 >::ResultType    KernelClass;
        KernelClass::init();

        std::stringstream ss;
        ss << SOLVERTAG::name() << "_trans_solve";
        viennacl::ocl::kernel & k = viennacl::ocl::get_kernel(KernelClass::program_name(), ss.str());

        k.global_work_size(0, proxy_B.lhs().size1() * k.local_work_size());
        detail::inplace_solve_impl(A, proxy_B.lhs(), k);
      }
      
      //upper triangular solver for transposed lower triangular matrices
      /** @brief Direct inplace solver for dense triangular systems that stem from transposed triangular systems
      *
      * @param proxy_A  The system matrix proxy
      * @param B        The matrix holding the load vectors, where the solution is directly written to
      */
      template <typename M1,
                typename M2, typename SOLVERTAG>
      typename viennacl::enable_if<    viennacl::is_any_dense_nonstructured_matrix<M1>::value
                                    && viennacl::is_any_dense_nonstructured_matrix<M2>::value
                                  >::type
      inplace_solve(const matrix_expression< const M1, const M1, op_trans> & proxy_A,
                    M2 & B,
                    SOLVERTAG)
      {
        typedef typename viennacl::tools::MATRIX_SOLVE_KERNEL_CLASS_DEDUCER< M1, M2 >::ResultType    KernelClass;
        KernelClass::init();

        std::stringstream ss;
        ss << "trans_" << SOLVERTAG::name() << "_solve";
        viennacl::ocl::kernel & k = viennacl::ocl::get_kernel(KernelClass::program_name(), ss.str());

        k.global_work_size(0, B.size2() * k.local_work_size());
        detail::inplace_solve_impl(proxy_A.lhs(), B, k);
      }

      /** @brief Direct inplace solver for dense transposed triangular systems with transposed right hand side. Matlab notation: A' \ B'
      *
      * @param proxy_A  The system matrix proxy
      * @param proxy_B  The matrix holding the load vectors, where the solution is directly written to
      */
      template <typename M1,
                typename M2, typename SOLVERTAG>
      typename viennacl::enable_if<    viennacl::is_any_dense_nonstructured_matrix<M1>::value
                                    && viennacl::is_any_dense_nonstructured_matrix<M2>::value
                                  >::type
      inplace_solve(const matrix_expression< const M1, const M1, op_trans> & proxy_A,
                          matrix_expression< const M2, const M2, op_trans>   proxy_B,
                          SOLVERTAG)
      {
        typedef typename viennacl::tools::MATRIX_SOLVE_KERNEL_CLASS_DEDUCER< M1, M2 >::ResultType    KernelClass;
        KernelClass::init();

        std::stringstream ss;
        ss << "trans_" << SOLVERTAG::name() << "_trans_solve";
        viennacl::ocl::kernel & k = viennacl::ocl::get_kernel(KernelClass::program_name(), ss.str());

        k.global_work_size(0, proxy_B.lhs().size1() * k.local_work_size());
        detail::inplace_solve_impl(proxy_A.lhs(), proxy_B.lhs(), k);
      }
      
      
      
      //
      //  Solve on vector
      //

      template <typename M1,
                typename V1, typename SOLVERTAG>
      typename viennacl::enable_if<    viennacl::is_any_dense_nonstructured_matrix<M1>::value
                                    && viennacl::is_any_dense_nonstructured_vector<V1>::value
                                  >::type
      inplace_solve(const M1 & mat,
                          V1 & vec,
                    SOLVERTAG)
      {
        typedef typename viennacl::tools::MATRIX_KERNEL_CLASS_DEDUCER< M1 >::ResultType    KernelClass;
        KernelClass::init();

        cl_uint options = detail::get_option_for_solver_tag(SOLVERTAG());
        viennacl::ocl::kernel & k = viennacl::ocl::get_kernel(KernelClass::program_name(), "triangular_substitute_inplace");
        
        k.global_work_size(0, k.local_work_size());
        viennacl::ocl::enqueue(k(viennacl::traits::opencl_handle(mat),
                                 cl_uint(viennacl::traits::start1(mat)),         cl_uint(viennacl::traits::start2(mat)),
                                 cl_uint(viennacl::traits::stride1(mat)),        cl_uint(viennacl::traits::stride2(mat)),
                                 cl_uint(viennacl::traits::size1(mat)),          cl_uint(viennacl::traits::size2(mat)),
                                 cl_uint(viennacl::traits::internal_size1(mat)), cl_uint(viennacl::traits::internal_size2(mat)),
                                 viennacl::traits::opencl_handle(vec),
                                 cl_uint(viennacl::traits::start(vec)),
                                 cl_uint(viennacl::traits::stride(vec)),
                                 cl_uint(viennacl::traits::size(vec)),
                                 options
                                )
                              );        
      }

      /** @brief Direct inplace solver for dense upper triangular systems that stem from transposed lower triangular systems
      *
      * @param proxy    The system matrix proxy
      * @param vec    The load vector, where the solution is directly written to
      */
      template <typename M1,
                typename V1, typename SOLVERTAG>
      typename viennacl::enable_if<    viennacl::is_any_dense_nonstructured_matrix<M1>::value
                                    && viennacl::is_any_dense_nonstructured_vector<V1>::value
                                  >::type
      inplace_solve(const matrix_expression< const M1, const M1, op_trans> & proxy,
                    V1 & vec,
                    SOLVERTAG)
      {
        typedef typename viennacl::tools::MATRIX_KERNEL_CLASS_DEDUCER< M1 >::ResultType    KernelClass;
        KernelClass::init();
        
        cl_uint options = detail::get_option_for_solver_tag(SOLVERTAG()) | 0x02;  //add transpose-flag
        viennacl::ocl::kernel & k = viennacl::ocl::get_kernel(KernelClass::program_name(), "triangular_substitute_inplace");
        
        k.global_work_size(0, k.local_work_size());
        viennacl::ocl::enqueue(k(viennacl::traits::opencl_handle(proxy.lhs()),
                                 cl_uint(viennacl::traits::start1(proxy.lhs())),         cl_uint(viennacl::traits::start2(proxy.lhs())),
                                 cl_uint(viennacl::traits::stride1(proxy.lhs())),        cl_uint(viennacl::traits::stride2(proxy.lhs())),
                                 cl_uint(viennacl::traits::size1(proxy.lhs())),          cl_uint(viennacl::traits::size2(proxy.lhs())),
                                 cl_uint(viennacl::traits::internal_size1(proxy.lhs())), cl_uint(viennacl::traits::internal_size2(proxy.lhs())),
                                 viennacl::traits::opencl_handle(vec),
                                 cl_uint(viennacl::traits::start(vec)),
                                 cl_uint(viennacl::traits::stride(vec)),
                                 cl_uint(viennacl::traits::size(vec)),
                                 options
                                )
                              );        
      }
      
      
    }
  }
}

#endif
