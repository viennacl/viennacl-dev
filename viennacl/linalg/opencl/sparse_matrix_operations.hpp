#ifndef VIENNACL_LINALG_OPENCL_SPARSE_MATRIX_OPERATIONS_HPP_
#define VIENNACL_LINALG_OPENCL_SPARSE_MATRIX_OPERATIONS_HPP_

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

/** @file viennacl/linalg/opencl/sparse_matrix_operations.hpp
    @brief Implementations of operations using sparse matrices and OpenCL
*/

#include "viennacl/forwards.h"
#include "viennacl/ocl/device.hpp"
#include "viennacl/ocl/handle.hpp"
#include "viennacl/ocl/kernel.hpp"
#include "viennacl/scalar.hpp"
#include "viennacl/vector.hpp"
#include "viennacl/tools/tools.hpp"
#include "viennacl/linalg/kernels/compressed_matrix_kernels.h"
#include "viennacl/linalg/kernels/coordinate_matrix_kernels.h"
#include "viennacl/linalg/kernels/ell_matrix_kernels.h"
#include "viennacl/linalg/kernels/hyb_matrix_kernels.h"


namespace viennacl
{
  namespace linalg
  {
    namespace opencl
    {
      //
      // Compressed matrix
      //
      
      namespace detail
      {
        template<typename SCALARTYPE, unsigned int MAT_ALIGNMENT, unsigned int VEC_ALIGNMENT>
        void row_info(compressed_matrix<SCALARTYPE, MAT_ALIGNMENT> const & mat,
                      vector<SCALARTYPE, VEC_ALIGNMENT> & vec,
                      viennacl::linalg::detail::row_info_types info_selector)
        {
          viennacl::linalg::kernels::compressed_matrix<SCALARTYPE, MAT_ALIGNMENT>::init();
          viennacl::ocl::kernel & row_info_kernel = viennacl::ocl::get_kernel(viennacl::linalg::kernels::compressed_matrix<SCALARTYPE, MAT_ALIGNMENT>::program_name(), "row_info_extractor");
          
          viennacl::ocl::enqueue(row_info_kernel(mat.handle1().opencl_handle(), mat.handle2().opencl_handle(), mat.handle().opencl_handle(),
                                                 viennacl::traits::opencl_handle(vec),
                                                 cl_uint(mat.size1()),
                                                 cl_uint(info_selector)
                                                )
                                );
        }
      }
      
      /** @brief Carries out matrix-vector multiplication with a compressed_matrix
      *
      * Implementation of the convenience expression result = prod(mat, vec);
      *
      * @param mat    The matrix
      * @param vec    The vector
      * @param result The result vector
      */
      template<class TYPE, unsigned int ALIGNMENT, unsigned int VECTOR_ALIGNMENT>
      void prod_impl(const viennacl::compressed_matrix<TYPE, ALIGNMENT> & mat, 
                     const viennacl::vector<TYPE, VECTOR_ALIGNMENT> & vec,
                           viennacl::vector<TYPE, VECTOR_ALIGNMENT> & result)
      {
        viennacl::linalg::kernels::compressed_matrix<TYPE, ALIGNMENT>::init();
        viennacl::ocl::kernel & k = viennacl::ocl::get_kernel(viennacl::linalg::kernels::compressed_matrix<TYPE, ALIGNMENT>::program_name(), "vec_mul");
        
        viennacl::ocl::enqueue(k(mat.handle1().opencl_handle(), mat.handle2().opencl_handle(), mat.handle().opencl_handle(),
                                vec, result, static_cast<cl_uint>(mat.size1())));
      }
      
      
      
      
      // triangular solvers

      /** @brief Inplace solution of a lower triangular compressed_matrix with unit diagonal. Typically used for LU substitutions
      *
      * @param L    The matrix
      * @param vec  The vector holding the right hand side. Is overwritten by the solution.
      */
      template<typename SCALARTYPE, unsigned int MAT_ALIGNMENT, unsigned int VEC_ALIGNMENT>
      void inplace_solve(compressed_matrix<SCALARTYPE, MAT_ALIGNMENT> const & L,
                         vector<SCALARTYPE, VEC_ALIGNMENT> & vec,
                         viennacl::linalg::unit_lower_tag)
      {
        viennacl::linalg::kernels::compressed_matrix<SCALARTYPE, MAT_ALIGNMENT>::init();
        viennacl::ocl::kernel & k = viennacl::ocl::get_kernel(viennacl::linalg::kernels::compressed_matrix<SCALARTYPE, MAT_ALIGNMENT>::program_name(), "unit_lu_forward");
        
        k.local_work_size(0, 128);
        k.global_work_size(0, k.local_work_size());
        viennacl::ocl::enqueue(k(L.handle1().opencl_handle(), L.handle2().opencl_handle(), L.handle().opencl_handle(),
                                 viennacl::traits::opencl_handle(vec),
                                 cl_uint(L.size1())
                                )
                              );
      }
      
      /** @brief Inplace solution of a lower triangular compressed_matrix. Typically used for LU substitutions
      *
      * @param L    The matrix
      * @param vec  The vector holding the right hand side. Is overwritten by the solution.
      */
      template<typename SCALARTYPE, unsigned int MAT_ALIGNMENT, unsigned int VEC_ALIGNMENT>
      void inplace_solve(compressed_matrix<SCALARTYPE, MAT_ALIGNMENT> const & L,
                         vector<SCALARTYPE, VEC_ALIGNMENT> & vec,
                         viennacl::linalg::lower_tag)
      {
        viennacl::linalg::kernels::compressed_matrix<SCALARTYPE, MAT_ALIGNMENT>::init();
        
        viennacl::ocl::kernel & k = viennacl::ocl::get_kernel(viennacl::linalg::kernels::compressed_matrix<SCALARTYPE, MAT_ALIGNMENT>::program_name(), "lu_forward");
        
        k.local_work_size(0, 128);
        k.global_work_size(0, k.local_work_size());
        viennacl::ocl::enqueue(k(L.handle1().opencl_handle(), L.handle2().opencl_handle(), L.handle().opencl_handle(),
                                 viennacl::traits::opencl_handle(vec),
                                 cl_uint(L.size1())
                                )
                              );
      }
      
      
      /** @brief Inplace solution of an upper triangular compressed_matrix with unit diagonal. Typically used for LU substitutions
      *
      * @param U    The matrix
      * @param vec  The vector holding the right hand side. Is overwritten by the solution.
      */
      template<typename SCALARTYPE, unsigned int MAT_ALIGNMENT, unsigned int VEC_ALIGNMENT>
      void inplace_solve(compressed_matrix<SCALARTYPE, MAT_ALIGNMENT> const & U,
                         vector<SCALARTYPE, VEC_ALIGNMENT> & vec,
                         viennacl::linalg::unit_upper_tag)
      {
        viennacl::linalg::kernels::compressed_matrix<SCALARTYPE, MAT_ALIGNMENT>::init();
        viennacl::ocl::kernel & k = viennacl::ocl::get_kernel(viennacl::linalg::kernels::compressed_matrix<SCALARTYPE, MAT_ALIGNMENT>::program_name(), "unit_lu_backward");
        
        k.local_work_size(0, 128);
        k.global_work_size(0, k.local_work_size());
        viennacl::ocl::enqueue(k(U.handle1().opencl_handle(), U.handle2().opencl_handle(), U.handle().opencl_handle(),
                                 viennacl::traits::opencl_handle(vec),
                                 cl_uint(U.size1())
                                )
                              );
      }
      
      /** @brief Inplace solution of an upper triangular compressed_matrix. Typically used for LU substitutions
      *
      * @param U    The matrix
      * @param vec  The vector holding the right hand side. Is overwritten by the solution.
      */
      template<typename SCALARTYPE, unsigned int MAT_ALIGNMENT, unsigned int VEC_ALIGNMENT>
      void inplace_solve(compressed_matrix<SCALARTYPE, MAT_ALIGNMENT> const & U,
                         vector<SCALARTYPE, VEC_ALIGNMENT> & vec,
                         viennacl::linalg::upper_tag)
      {
        viennacl::linalg::kernels::compressed_matrix<SCALARTYPE, MAT_ALIGNMENT>::init();
        
        viennacl::ocl::kernel & k = viennacl::ocl::get_kernel(viennacl::linalg::kernels::compressed_matrix<SCALARTYPE, MAT_ALIGNMENT>::program_name(), "lu_backward");
        
        k.local_work_size(0, 128);
        k.global_work_size(0, k.local_work_size());
        viennacl::ocl::enqueue(k(U.handle1().opencl_handle(), U.handle2().opencl_handle(), U.handle().opencl_handle(),
                                 viennacl::traits::opencl_handle(vec),
                                 cl_uint(U.size1())
                                )
                              );
      }
      
      
      
      
      
      // transposed triangular solvers
      
      namespace detail
      {
        //
        // block solves
        //
        template<typename ScalarType, unsigned int MAT_ALIGNMENT, unsigned int VEC_ALIGNMENT>
        void block_inplace_solve(const matrix_expression<const compressed_matrix<ScalarType, MAT_ALIGNMENT>,
                                                         const compressed_matrix<ScalarType, MAT_ALIGNMENT>,
                                                         op_trans> & L, 
                                 viennacl::backend::mem_handle const & block_indices, std::size_t num_blocks,
                                 vector<ScalarType> const & /* L_diagonal */,  //ignored
                                 vector<ScalarType, VEC_ALIGNMENT> & vec,
                                 viennacl::linalg::unit_lower_tag)
        {
          viennacl::linalg::kernels::compressed_matrix<ScalarType, 1>::init();
          viennacl::ocl::kernel & block_solve_kernel = viennacl::ocl::get_kernel(viennacl::linalg::kernels::compressed_matrix<ScalarType, 1>::program_name(), "block_trans_unit_lu_forward");
          block_solve_kernel.global_work_size(0, num_blocks * block_solve_kernel.local_work_size(0));
          
          viennacl::ocl::enqueue(block_solve_kernel(L.lhs().handle1().opencl_handle(),
                                                    L.lhs().handle2().opencl_handle(),
                                                    L.lhs().handle().opencl_handle(),
                                                    block_indices.opencl_handle(),
                                                    vec,
                                                    static_cast<cl_uint>(vec.size())));
        }
        
        
        template<typename ScalarType, unsigned int MAT_ALIGNMENT, unsigned int VEC_ALIGNMENT>
        void block_inplace_solve(const matrix_expression<const compressed_matrix<ScalarType, MAT_ALIGNMENT>,
                                                         const compressed_matrix<ScalarType, MAT_ALIGNMENT>,
                                                         op_trans> & U, 
                                 viennacl::backend::mem_handle const & block_indices, std::size_t num_blocks,
                                 vector<ScalarType> const & U_diagonal,
                                 vector<ScalarType, VEC_ALIGNMENT> & vec,
                                 viennacl::linalg::upper_tag)
        {
          viennacl::linalg::kernels::compressed_matrix<ScalarType, 1>::init();
          viennacl::ocl::kernel & block_solve_kernel = viennacl::ocl::get_kernel(viennacl::linalg::kernels::compressed_matrix<ScalarType, 1>::program_name(), "block_trans_lu_backward");
          block_solve_kernel.global_work_size(0, num_blocks * block_solve_kernel.local_work_size(0));

          viennacl::ocl::enqueue(block_solve_kernel(U.lhs().handle1().opencl_handle(),
                                                    U.lhs().handle2().opencl_handle(),
                                                    U.lhs().handle().opencl_handle(),
                                                    U_diagonal,
                                                    block_indices.opencl_handle(),
                                                    vec,
                                                    static_cast<cl_uint>(vec.size())));
        }
        
        
      }
      
      
      /** @brief Inplace solution of a lower triangular compressed_matrix with unit diagonal. Typically used for LU substitutions
      *
      * @param proxy_L  The transposed matrix proxy
      * @param vec      The vector
      */
      template<typename SCALARTYPE, unsigned int MAT_ALIGNMENT, unsigned int VEC_ALIGNMENT>
      void inplace_solve(matrix_expression< const compressed_matrix<SCALARTYPE, MAT_ALIGNMENT>,
                                            const compressed_matrix<SCALARTYPE, MAT_ALIGNMENT>,
                                            op_trans> const & proxy_L,
                         vector<SCALARTYPE, VEC_ALIGNMENT> & vec,
                         viennacl::linalg::unit_lower_tag)
      {
        viennacl::linalg::kernels::compressed_matrix<SCALARTYPE, MAT_ALIGNMENT>::init();
        viennacl::ocl::kernel & k = viennacl::ocl::get_kernel(viennacl::linalg::kernels::compressed_matrix<SCALARTYPE, MAT_ALIGNMENT>::program_name(), "trans_unit_lu_forward");

        k.local_work_size(0, 128);
        k.global_work_size(0, k.local_work_size());
        viennacl::ocl::enqueue(k(proxy_L.lhs().handle1().opencl_handle(), proxy_L.lhs().handle2().opencl_handle(), proxy_L.lhs().handle().opencl_handle(),
                                 viennacl::traits::opencl_handle(vec),
                                 cl_uint(proxy_L.lhs().size1())
                                )
                              );
      }
      
      
      /** @brief Inplace solution of a lower triangular compressed_matrix. Typically used for LU substitutions
      *
      * @param proxy_L  The transposed matrix proxy
      * @param vec      The vector
      */
      template<typename SCALARTYPE, unsigned int MAT_ALIGNMENT, unsigned int VEC_ALIGNMENT>
      void inplace_solve(matrix_expression< const compressed_matrix<SCALARTYPE, MAT_ALIGNMENT>,
                                            const compressed_matrix<SCALARTYPE, MAT_ALIGNMENT>,
                                            op_trans> const & proxy_L,
                         vector<SCALARTYPE, VEC_ALIGNMENT> & vec,
                         viennacl::linalg::lower_tag)
      {
        viennacl::linalg::kernels::compressed_matrix<SCALARTYPE, MAT_ALIGNMENT>::init();

        viennacl::vector<SCALARTYPE> diagonal(vec.size());
        detail::row_info(proxy_L.lhs(), diagonal, viennacl::linalg::detail::SPARSE_ROW_DIAGONAL);
        
        viennacl::ocl::kernel & k = viennacl::ocl::get_kernel(viennacl::linalg::kernels::compressed_matrix<SCALARTYPE, MAT_ALIGNMENT>::program_name(), "trans_lu_forward");
        
        k.local_work_size(0, 128);
        k.global_work_size(0, k.local_work_size());
        viennacl::ocl::enqueue(k(proxy_L.lhs().handle1().opencl_handle(), proxy_L.lhs().handle2().opencl_handle(), proxy_L.lhs().handle().opencl_handle(),
                                 viennacl::traits::opencl_handle(diagonal),
                                 viennacl::traits::opencl_handle(vec),
                                 cl_uint(proxy_L.lhs().size1())
                                )
                              );
      }
      
      /** @brief Inplace solution of a lower triangular compressed_matrix with unit diagonal. Typically used for LU substitutions
      *
      * @param proxy_U  The transposed matrix proxy
      * @param vec      The vector
      */
      template<typename SCALARTYPE, unsigned int MAT_ALIGNMENT, unsigned int VEC_ALIGNMENT>
      void inplace_solve(matrix_expression< const compressed_matrix<SCALARTYPE, MAT_ALIGNMENT>,
                                            const compressed_matrix<SCALARTYPE, MAT_ALIGNMENT>,
                                            op_trans> const & proxy_U,
                         vector<SCALARTYPE, VEC_ALIGNMENT> & vec,
                         viennacl::linalg::unit_upper_tag)
      {
        viennacl::linalg::kernels::compressed_matrix<SCALARTYPE, MAT_ALIGNMENT>::init();
        viennacl::ocl::kernel & k = viennacl::ocl::get_kernel(viennacl::linalg::kernels::compressed_matrix<SCALARTYPE, MAT_ALIGNMENT>::program_name(), "trans_unit_lu_backward");

        k.local_work_size(0, 128);
        k.global_work_size(0, k.local_work_size());
        viennacl::ocl::enqueue(k(proxy_U.lhs().handle1().opencl_handle(), proxy_U.lhs().handle2().opencl_handle(), proxy_U.lhs().handle().opencl_handle(),
                                 viennacl::traits::opencl_handle(vec),
                                 cl_uint(proxy_U.lhs().size1())
                                )
                              );
      }
      
      
      /** @brief Inplace solution of a lower triangular compressed_matrix. Typically used for LU substitutions
      *
      * @param proxy_U  The transposed matrix proxy
      * @param vec      The vector
      */
      template<typename SCALARTYPE, unsigned int MAT_ALIGNMENT, unsigned int VEC_ALIGNMENT>
      void inplace_solve(matrix_expression< const compressed_matrix<SCALARTYPE, MAT_ALIGNMENT>,
                                            const compressed_matrix<SCALARTYPE, MAT_ALIGNMENT>,
                                            op_trans> const & proxy_U,
                         vector<SCALARTYPE, VEC_ALIGNMENT> & vec,
                         viennacl::linalg::upper_tag)
      {
        viennacl::linalg::kernels::compressed_matrix<SCALARTYPE, MAT_ALIGNMENT>::init();

        viennacl::vector<SCALARTYPE> diagonal(vec.size());
        detail::row_info(proxy_U.lhs(), diagonal, viennacl::linalg::detail::SPARSE_ROW_DIAGONAL);
        
        viennacl::ocl::kernel & k = viennacl::ocl::get_kernel(viennacl::linalg::kernels::compressed_matrix<SCALARTYPE, MAT_ALIGNMENT>::program_name(), "trans_lu_backward");
        
        k.local_work_size(0, 128);
        k.global_work_size(0, k.local_work_size());
        viennacl::ocl::enqueue(k(proxy_U.lhs().handle1().opencl_handle(), proxy_U.lhs().handle2().opencl_handle(), proxy_U.lhs().handle().opencl_handle(),
                                 viennacl::traits::opencl_handle(diagonal),
                                 viennacl::traits::opencl_handle(vec),
                                 cl_uint(proxy_U.lhs().size1())
                                )
                              );
      }
      

      
      //
      // Coordinate matrix
      //
      
      namespace detail
      {
        template<typename SCALARTYPE, unsigned int MAT_ALIGNMENT, unsigned int VEC_ALIGNMENT>
        void row_info(coordinate_matrix<SCALARTYPE, MAT_ALIGNMENT> const & mat,
                      vector<SCALARTYPE, VEC_ALIGNMENT> & vec,
                      viennacl::linalg::detail::row_info_types info_selector)
        {
          viennacl::linalg::kernels::coordinate_matrix<SCALARTYPE, MAT_ALIGNMENT>::init();
          viennacl::ocl::kernel & row_info_kernel = viennacl::ocl::get_kernel(viennacl::linalg::kernels::coordinate_matrix<SCALARTYPE, MAT_ALIGNMENT>::program_name(), "row_info_extractor");
          unsigned int thread_num = 256; //k.local_work_size(0);
          
          row_info_kernel.local_work_size(0, thread_num);
          
          row_info_kernel.global_work_size(0, 64 * thread_num);  //64 work groups are hard-coded for now. Gives reasonable performance in most cases
          viennacl::ocl::enqueue(row_info_kernel(mat.handle12().opencl_handle(), mat.handle().opencl_handle(), mat.handle3().opencl_handle(),
                                                 viennacl::traits::opencl_handle(vec),
                                                 cl_uint(info_selector),
                                                 viennacl::ocl::local_mem(sizeof(cl_uint)*thread_num),
                                                 viennacl::ocl::local_mem(sizeof(SCALARTYPE)*thread_num)) );
        }
      }
      
      /** @brief Carries out matrix-vector multiplication with a coordinate_matrix
      *
      * Implementation of the convenience expression result = prod(mat, vec);
      *
      * @param mat    The matrix
      * @param vec    The vector
      * @param result The result vector
      */
      template<class SCALARTYPE, unsigned int ALIGNMENT, unsigned int VECTOR_ALIGNMENT>
      void prod_impl(const viennacl::coordinate_matrix<SCALARTYPE, ALIGNMENT> & mat, 
                     const viennacl::vector<SCALARTYPE, VECTOR_ALIGNMENT> & vec,
                           viennacl::vector<SCALARTYPE, VECTOR_ALIGNMENT> & result)
      {
        viennacl::linalg::kernels::coordinate_matrix<SCALARTYPE, ALIGNMENT>::init();
        
        result.clear();
        
        //std::cout << "prod(coordinate_matrix" << ALIGNMENT << ", vector) called with internal_nnz=" << mat.internal_nnz() << std::endl;
        
        viennacl::ocl::kernel & k = viennacl::ocl::get_kernel(viennacl::linalg::kernels::coordinate_matrix<SCALARTYPE, ALIGNMENT>::program_name(), "vec_mul");
        unsigned int thread_num = 256; //k.local_work_size(0);
        
        k.local_work_size(0, thread_num);
        
        k.global_work_size(0, 64 * thread_num);  //64 work groups are hard-coded for now. Gives reasonable performance in most cases
        //k.global_work_size(0, thread_num);  //Only one work group
        viennacl::ocl::enqueue(k(mat.handle12().opencl_handle(), mat.handle().opencl_handle(), mat.handle3().opencl_handle(),
                                 viennacl::traits::opencl_handle(vec),
                                 viennacl::traits::opencl_handle(result),
                                 viennacl::ocl::local_mem(sizeof(cl_uint)*thread_num),
                                 viennacl::ocl::local_mem(sizeof(SCALARTYPE)*thread_num)) );

      }
      
      
      //
      // ELL Matrix
      //
      
      template<class TYPE, unsigned int ALIGNMENT, unsigned int VECTOR_ALIGNMENT>
      void prod_impl( const viennacl::ell_matrix<TYPE, ALIGNMENT> & mat, 
                      const viennacl::vector<TYPE, VECTOR_ALIGNMENT> & vec,
                      viennacl::vector<TYPE, VECTOR_ALIGNMENT> & result)
      {
        assert(mat.size1() == result.size());
        assert(mat.size2() == vec.size());

        viennacl::linalg::kernels::ell_matrix<TYPE, ALIGNMENT>::init();
        result.clear();

        std::stringstream ss;
        ss << "vec_mul_" << 1;//(ALIGNMENT != 1?4:1);
        viennacl::ocl::kernel& k = viennacl::ocl::get_kernel(viennacl::linalg::kernels::ell_matrix<TYPE, ALIGNMENT>::program_name(), "vec_mul");

        unsigned int thread_num = 128;
        unsigned int group_num = 256;

        k.local_work_size(0, thread_num);
        k.global_work_size(0, thread_num * group_num);

        viennacl::ocl::enqueue(k(mat.handle2().opencl_handle(), 
                                 mat.handle().opencl_handle(),
                                 viennacl::traits::opencl_handle(vec),
                                 viennacl::traits::opencl_handle(result),
                                 cl_uint(mat.size1()),
                                 cl_uint(mat.size2()),
                                 cl_uint(mat.internal_size1()),
                                 cl_uint(mat.maxnnz()),
                                 cl_uint(mat.internal_maxnnz())
                                ) 
        );


      }

      //
      // Hybrid Matrix
      //
      
      template<class TYPE, unsigned int ALIGNMENT, unsigned int VECTOR_ALIGNMENT>
      void prod_impl( const viennacl::hyb_matrix<TYPE, ALIGNMENT>& mat, 
                      const viennacl::vector<TYPE, VECTOR_ALIGNMENT>& vec,
                      viennacl::vector<TYPE, VECTOR_ALIGNMENT>& result)
      {
        assert(mat.size1() == result.size());
        assert(mat.size2() == vec.size());

        viennacl::linalg::kernels::hyb_matrix<TYPE, ALIGNMENT>::init();
        
        result.clear();

        viennacl::ocl::kernel& k = viennacl::ocl::get_kernel(viennacl::linalg::kernels::hyb_matrix<TYPE, ALIGNMENT>::program_name(), "vec_mul");

        unsigned int thread_num = 256;
        unsigned int group_num = 32;

        k.local_work_size(0, thread_num);
        k.global_work_size(0, thread_num * group_num);

        viennacl::ocl::enqueue(k(mat.handle2().opencl_handle(), 
                                 mat.handle().opencl_handle(),
                                 mat.handle3().opencl_handle(),
                                 mat.handle4().opencl_handle(),
                                 mat.handle5().opencl_handle(),
                                 viennacl::traits::opencl_handle(vec),
                                 viennacl::traits::opencl_handle(result),
                                 cl_uint(mat.size1()),
                                 cl_uint(mat.internal_size1()),
                                 cl_uint(mat.ell_nnz()),
                                 cl_uint(mat.internal_ellnnz())
                                ) 
        );
      }
      
    } // namespace opencl
  } //namespace linalg
} //namespace viennacl


#endif
