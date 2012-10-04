#ifndef VIENNACL_LINALG_OPENCL_DIRECT_SOLVE_HPP
#define VIENNACL_LINALG_OPENCL_DIRECT_SOLVE_HPP

/* =========================================================================
   Copyright (c) 2010-2012, Institute for Microelectronics,
                            Institute for Analysis and Scientific Computing,
                            TU Wien.

                            -----------------
                  ViennaCL - The Vienna Computing Library
                            -----------------

   Project Head:    Karl Rupp                   rupp@iue.tuwien.ac.at
               
   (A list of authors and contributors can be found in the PDF manual)

   License:         MIT (X11), see file LICENSE in the base directory
============================================================================= */

/** @file viennacl/linalg/direct_solve.hpp
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
      }
      
      
      //
      // Note: By convention, all size checks are performed in the calling frontend. No need to double-check here.
      //
      
      ////////////////// upper triangular solver (upper_tag) //////////////////////////////////////
      /** @brief Direct inplace solver for dense upper triangular systems
      *
      * @param mat    The system matrix
      * @param B      The matrix of row vectors, where the solution is directly written to
      */
      template<typename SCALARTYPE, typename F1, typename F2, unsigned int A1, unsigned int A2, typename SOLVERTAG>
      void inplace_solve(const matrix<SCALARTYPE, F1, A1> & mat,
                        matrix<SCALARTYPE, F2, A2> & B,
                        SOLVERTAG)
      {
        typedef typename viennacl::tools::MATRIX_SOLVE_KERNEL_CLASS_DEDUCER< matrix<SCALARTYPE, F1, A1>,
                                                                             matrix<SCALARTYPE, F2, A2> >::ResultType    KernelClass;
        KernelClass::init();
        
        std::stringstream ss;
        ss << SOLVERTAG::name() << "_solve";
        viennacl::ocl::kernel & k = viennacl::ocl::get_kernel(KernelClass::program_name(), ss.str());

        k.global_work_size(0, B.size2() * k.local_work_size());
        viennacl::ocl::enqueue(k(viennacl::traits::opencl_handle(mat),
                                 cl_uint(viennacl::traits::start1(mat)),         cl_uint(viennacl::traits::start2(mat)),
                                 cl_uint(viennacl::traits::stride1(mat)),        cl_uint(viennacl::traits::stride2(mat)),
                                 cl_uint(viennacl::traits::size1(mat)),          cl_uint(viennacl::traits::size2(mat)),
                                 cl_uint(viennacl::traits::internal_size1(mat)), cl_uint(viennacl::traits::internal_size2(mat)),
                                 viennacl::traits::opencl_handle(B),
                                 cl_uint(viennacl::traits::start1(B)),         cl_uint(viennacl::traits::start2(B)),
                                 cl_uint(viennacl::traits::stride1(B)),        cl_uint(viennacl::traits::stride2(B)),
                                 cl_uint(viennacl::traits::size1(B)),          cl_uint(viennacl::traits::size2(B)),
                                 cl_uint(viennacl::traits::internal_size1(B)), cl_uint(viennacl::traits::internal_size2(B))
                                )
                              );        
      }
      
      /** @brief Direct inplace solver for dense upper triangular systems
      *
      * @param mat    The system matrix
      * @param B      The (transposed) matrix of row vectors, where the solution is directly written to
      */
      template<typename SCALARTYPE, typename F1, typename F2, unsigned int A1, unsigned int A2, typename SOLVERTAG>
      void inplace_solve(const matrix<SCALARTYPE, F1, A1> & mat,
                        const matrix_expression< const matrix<SCALARTYPE, F2, A2>,
                                                  const matrix<SCALARTYPE, F2, A2>,
                                                  op_trans> & B,
                        SOLVERTAG)
      {
        assert(mat.size1() == mat.size2());
        assert(mat.size2() == B.lhs().size2());
        
        typedef typename viennacl::tools::MATRIX_SOLVE_KERNEL_CLASS_DEDUCER< matrix<SCALARTYPE, F1, A1>,
                                                                            matrix<SCALARTYPE, F2, A2> >::ResultType    KernelClass;
        KernelClass::init();

        std::stringstream ss;
        ss << SOLVERTAG::name() << "_trans_solve";
        viennacl::ocl::kernel & k = viennacl::ocl::get_kernel(KernelClass::program_name(), ss.str());

        k.global_work_size(0, B.lhs().size1() * k.local_work_size());
        viennacl::ocl::enqueue(k(viennacl::traits::opencl_handle(mat),
                                 cl_uint(viennacl::traits::start1(mat)),         cl_uint(viennacl::traits::start2(mat)),
                                 cl_uint(viennacl::traits::stride1(mat)),        cl_uint(viennacl::traits::stride2(mat)),
                                 cl_uint(viennacl::traits::size1(mat)),          cl_uint(viennacl::traits::size2(mat)),
                                 cl_uint(viennacl::traits::internal_size1(mat)), cl_uint(viennacl::traits::internal_size2(mat)),
                                 viennacl::traits::opencl_handle(B.lhs()),
                                 cl_uint(viennacl::traits::start1(B.lhs())),         cl_uint(viennacl::traits::start2(B.lhs())),
                                 cl_uint(viennacl::traits::stride1(B.lhs())),        cl_uint(viennacl::traits::stride2(B.lhs())),
                                 cl_uint(viennacl::traits::size1(B.lhs())),          cl_uint(viennacl::traits::size2(B.lhs())),
                                 cl_uint(viennacl::traits::internal_size1(B.lhs())), cl_uint(viennacl::traits::internal_size2(B.lhs()))
                                )
                              );        
      }
      
      //upper triangular solver for transposed lower triangular matrices
      /** @brief Direct inplace solver for dense upper triangular systems that stem from transposed lower triangular systems
      *
      * @param proxy    The system matrix proxy
      * @param B        The matrix holding the load vectors, where the solution is directly written to
      */
      template<typename SCALARTYPE, typename F1, typename F2, unsigned int A1, unsigned int A2, typename SOLVERTAG>
      void inplace_solve(const matrix_expression< const matrix<SCALARTYPE, F1, A1>,
                                                  const matrix<SCALARTYPE, F1, A1>,
                                                  op_trans> & proxy,
                        matrix<SCALARTYPE, F2, A2> & B,
                        SOLVERTAG)
      {
        assert(proxy.lhs().size1() == proxy.lhs().size2());
        assert(proxy.lhs().size2() == B.size1());
        
        typedef typename viennacl::tools::MATRIX_SOLVE_KERNEL_CLASS_DEDUCER< matrix<SCALARTYPE, F1, A1>,
                                                                            matrix<SCALARTYPE, F2, A2> >::ResultType    KernelClass;
        KernelClass::init();

        std::stringstream ss;
        ss << "trans_" << SOLVERTAG::name() << "_solve";
        viennacl::ocl::kernel & k = viennacl::ocl::get_kernel(KernelClass::program_name(), ss.str());

        k.global_work_size(0, B.size2() * k.local_work_size());
        viennacl::ocl::enqueue(k(viennacl::traits::opencl_handle(proxy.lhs()),
                                 cl_uint(viennacl::traits::start1(proxy.lhs())),         cl_uint(viennacl::traits::start2(proxy.lhs())),
                                 cl_uint(viennacl::traits::stride1(proxy.lhs())),        cl_uint(viennacl::traits::stride2(proxy.lhs())),
                                 cl_uint(viennacl::traits::size1(proxy.lhs())),          cl_uint(viennacl::traits::size2(proxy.lhs())),
                                 cl_uint(viennacl::traits::internal_size1(proxy.lhs())), cl_uint(viennacl::traits::internal_size2(proxy.lhs())),
                                 viennacl::traits::opencl_handle(B),
                                 cl_uint(viennacl::traits::start1(B)),         cl_uint(viennacl::traits::start2(B)),
                                 cl_uint(viennacl::traits::stride1(B)),        cl_uint(viennacl::traits::stride2(B)),
                                 cl_uint(viennacl::traits::size1(B)),          cl_uint(viennacl::traits::size2(B)),
                                 cl_uint(viennacl::traits::internal_size1(B)), cl_uint(viennacl::traits::internal_size2(B))
                                )
                              );        
      }

      /** @brief Direct inplace solver for dense upper triangular systems that stem from transposed lower triangular systems
      *
      * @param proxy    The system matrix proxy
      * @param B        The matrix holding the load vectors, where the solution is directly written to
      */
      template<typename SCALARTYPE, typename F1, typename F2, unsigned int A1, unsigned int A2, typename SOLVERTAG>
      void inplace_solve(const matrix_expression< const matrix<SCALARTYPE, F1, A1>,
                                                  const matrix<SCALARTYPE, F1, A1>,
                                                  op_trans> & proxy,
                        const matrix_expression< const matrix<SCALARTYPE, F2, A2>,
                                                  const matrix<SCALARTYPE, F2, A2>,
                                                  op_trans> & B,
                        SOLVERTAG)
      {
        assert(proxy.lhs().size1() == proxy.lhs().size2());
        assert(proxy.lhs().size2() == B.lhs().size2());
        
        typedef typename viennacl::tools::MATRIX_SOLVE_KERNEL_CLASS_DEDUCER< matrix<SCALARTYPE, F1, A1>,
                                                                            matrix<SCALARTYPE, F2, A2> >::ResultType    KernelClass;
        KernelClass::init();

        std::stringstream ss;
        ss << "trans_" << SOLVERTAG::name() << "_trans_solve";
        viennacl::ocl::kernel & k = viennacl::ocl::get_kernel(KernelClass::program_name(), ss.str());

        k.global_work_size(0, B.lhs().size1() * k.local_work_size());
        viennacl::ocl::enqueue(k(viennacl::traits::opencl_handle(proxy.lhs()),
                                 cl_uint(viennacl::traits::start1(proxy.lhs())),         cl_uint(viennacl::traits::start2(proxy.lhs())),
                                 cl_uint(viennacl::traits::stride1(proxy.lhs())),        cl_uint(viennacl::traits::stride2(proxy.lhs())),
                                 cl_uint(viennacl::traits::size1(proxy.lhs())),          cl_uint(viennacl::traits::size2(proxy.lhs())),
                                 cl_uint(viennacl::traits::internal_size1(proxy.lhs())), cl_uint(viennacl::traits::internal_size2(proxy.lhs())),
                                 viennacl::traits::opencl_handle(B.lhs()),
                                 cl_uint(viennacl::traits::start1(B.lhs())),         cl_uint(viennacl::traits::start2(B.lhs())),
                                 cl_uint(viennacl::traits::stride1(B.lhs())),        cl_uint(viennacl::traits::stride2(B.lhs())),
                                 cl_uint(viennacl::traits::size1(B.lhs())),          cl_uint(viennacl::traits::size2(B.lhs())),
                                 cl_uint(viennacl::traits::internal_size1(B.lhs())), cl_uint(viennacl::traits::internal_size2(B.lhs()))
                                )
                              );        
      }
      
      //
      //  Solve on vector
      //

      template<typename SCALARTYPE, typename F, unsigned int ALIGNMENT, unsigned int VEC_ALIGNMENT, typename SOLVERTAG>
      void inplace_solve(const matrix<SCALARTYPE, F, ALIGNMENT> & mat,
                        vector<SCALARTYPE, VEC_ALIGNMENT> & vec,
                        SOLVERTAG)
      {
        typedef typename viennacl::tools::MATRIX_KERNEL_CLASS_DEDUCER< matrix<SCALARTYPE, F, ALIGNMENT> >::ResultType    KernelClass;

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
      template<typename SCALARTYPE, typename F, unsigned int ALIGNMENT, unsigned int VEC_ALIGNMENT, typename SOLVERTAG>
      void inplace_solve(const matrix_expression< const matrix<SCALARTYPE, F, ALIGNMENT>,
                                                  const matrix<SCALARTYPE, F, ALIGNMENT>,
                                                  op_trans> & proxy,
                        vector<SCALARTYPE, VEC_ALIGNMENT> & vec,
                        SOLVERTAG)
      {
        typedef typename viennacl::tools::MATRIX_KERNEL_CLASS_DEDUCER< matrix<SCALARTYPE, F, ALIGNMENT> >::ResultType    KernelClass;
        
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
      
      
      ///////////////////////////// lu factorization ///////////////////////
      /** @brief LU factorization of a dense matrix.
      *
      * @param mat    The system matrix, where the LU matrices are directly written to. The implicit unit diagonal of L is not written.
      */
      template<typename SCALARTYPE, typename F, unsigned int ALIGNMENT>
      void lu_factorize(matrix<SCALARTYPE, F, ALIGNMENT> & mat)
      {
        assert(mat.size1() == mat.size2());

        typedef typename viennacl::tools::MATRIX_KERNEL_CLASS_DEDUCER< matrix<SCALARTYPE, F, ALIGNMENT> >::ResultType    KernelClass;
        
        viennacl::ocl::kernel & k = viennacl::ocl::get_kernel(KernelClass::program_name(), "lu_factorize");
        
        k.global_work_size(0, k.local_work_size());
        viennacl::ocl::enqueue(k(viennacl::traits::opencl_handle(mat),
                                 cl_uint(mat.size1()), cl_uint(mat.size2()),
                                 cl_uint(mat.internal_size1()), cl_uint(mat.internal_size2())) );        
      }

    }
  }
}

#endif
