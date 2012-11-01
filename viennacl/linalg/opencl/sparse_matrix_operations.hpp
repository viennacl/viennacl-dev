#ifndef VIENNACL_LINALG_OPENCL_SPARSE_MATRIX_OPERATIONS_HPP_
#define VIENNACL_LINALG_OPENCL_SPARSE_MATRIX_OPERATIONS_HPP_

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

/** @file viennacl/linalg/opencl/compressed_matrix_operations.hpp
    @brief Implementations of operations using compressed_matrix and OpenCL
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
      
      /** @brief Carries out matrix-vector multiplication with a compressed_matrix
      *
      * Implementation of the convenience expression result = prod(mat, vec);
      *
      * @param mat    The matrix
      * @param vec    The vector
      * @param result The result vector
      * @param NUM_THREADS Number of threads per work group. Can be used for fine-tuning.
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

      /** @brief Inplace solution of a lower triangular compressed_matrix with unit diagonal. Typically used for LU substitutions
      *
      * @param L    The matrix
      * @param vec    The vector
      */
      template<typename SCALARTYPE, unsigned int MAT_ALIGNMENT, unsigned int VEC_ALIGNMENT>
      void inplace_solve(compressed_matrix<SCALARTYPE, MAT_ALIGNMENT> const & L, vector<SCALARTYPE, VEC_ALIGNMENT> & vec, viennacl::linalg::unit_lower_tag)
      {
        viennacl::linalg::kernels::compressed_matrix<SCALARTYPE, MAT_ALIGNMENT>::init();
        viennacl::ocl::kernel & k = viennacl::ocl::get_kernel(viennacl::linalg::kernels::compressed_matrix<SCALARTYPE, MAT_ALIGNMENT>::program_name(), "lu_forward");
        unsigned int threads = k.local_work_size();

        k.global_work_size(k.local_work_size());
        viennacl::ocl::enqueue(k(L.handle1().opencl_handle(), L.handle2().opencl_handle(), L.handle().opencl_handle(),
                                viennacl::ocl::local_mem(sizeof(int) * (threads+1)),
                                viennacl::ocl::local_mem(sizeof(SCALARTYPE) * threads),
                                vec, L.size1()));        
      }
      
      
      /** @brief Inplace solution of a upper triangular compressed_matrix. Typically used for LU substitutions
      *
      * @param U      The upper triangular matrix
      * @param vec    The vector
      */
      template<typename SCALARTYPE, unsigned int MAT_ALIGNMENT, unsigned int VEC_ALIGNMENT>
      void inplace_solve(compressed_matrix<SCALARTYPE, MAT_ALIGNMENT> const & U, vector<SCALARTYPE, VEC_ALIGNMENT> & vec, viennacl::linalg::upper_tag)
      {
        viennacl::linalg::kernels::compressed_matrix<SCALARTYPE, MAT_ALIGNMENT>::init();
        viennacl::ocl::kernel & k = viennacl::ocl::get_kernel(viennacl::linalg::kernels::compressed_matrix<SCALARTYPE, MAT_ALIGNMENT>::program_name(), "lu_backward");
        unsigned int threads = k.local_work_size();
        
        k.global_work_size(k.local_work_size());
        viennacl::ocl::enqueue(k(U.handle1().opencl_handle().get(), U.handle2().opencl_handle().get(), U.handle().opencl_handle().get(),
                                viennacl::ocl::local_mem(sizeof(int) * (threads+2)),
                                viennacl::ocl::local_mem(sizeof(SCALARTYPE) * (threads+2)),
                                vec, U.size1()));        
      }

      
      //
      // Coordinate matrix
      //
      
      /** @brief Carries out matrix-vector multiplication with a coordinate_matrix
      *
      * Implementation of the convenience expression result = prod(mat, vec);
      *
      * @param mat    The matrix
      * @param vec    The vector
      * @param result The result vector
      */
      template<class TYPE, unsigned int ALIGNMENT, unsigned int VECTOR_ALIGNMENT>
      void prod_impl(const viennacl::coordinate_matrix<TYPE, ALIGNMENT> & mat, 
                     const viennacl::vector<TYPE, VECTOR_ALIGNMENT> & vec,
                           viennacl::vector<TYPE, VECTOR_ALIGNMENT> & result)
      {
        viennacl::linalg::kernels::coordinate_matrix<TYPE, ALIGNMENT>::init();
        
        result.clear();
        
        //std::cout << "prod(coordinate_matrix" << ALIGNMENT << ", vector) called with internal_nnz=" << mat.internal_nnz() << std::endl;
        
        viennacl::ocl::kernel & k = viennacl::ocl::get_kernel(viennacl::linalg::kernels::coordinate_matrix<TYPE, ALIGNMENT>::program_name(), "vec_mul");
        unsigned int thread_num = 256; //k.local_work_size(0);
        
        k.local_work_size(0, thread_num);
        
        k.global_work_size(0, 64 * thread_num);  //64 work groups are hard-coded for now. Gives reasonable performance in most cases
        //k.global_work_size(0, thread_num);  //Only one work group
        viennacl::ocl::enqueue(k(mat.handle12().opencl_handle(), mat.handle().opencl_handle(), mat.handle3().opencl_handle(),
                                 viennacl::traits::opencl_handle(vec),
                                 viennacl::traits::opencl_handle(result),
                                 viennacl::ocl::local_mem(sizeof(cl_uint)*thread_num),
                                 viennacl::ocl::local_mem(sizeof(TYPE)*thread_num)) );

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
