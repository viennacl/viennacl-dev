#ifndef VIENNACL_LINALG_OPENCL_VANDERMONDE_MATRIX_OPERATIONS_HPP_
#define VIENNACL_LINALG_OPENCL_VANDERMONDE_MATRIX_OPERATIONS_HPP_

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

/** @file viennacl/linalg/opencl/vandermonde_matrix_operations.hpp
    @brief Implementations of operations using vandermonde_matrix
*/

#include "viennacl/forwards.h"
#include "viennacl/ocl/device.hpp"
#include "viennacl/ocl/handle.hpp"
#include "viennacl/ocl/kernel.hpp"
#include "viennacl/scalar.hpp"
#include "viennacl/vector.hpp"
#include "viennacl/tools/tools.hpp"
#include "viennacl/fft.hpp"
//#include "viennacl/linalg/kernels/coordinate_matrix_kernels.h"

namespace viennacl
{
  namespace linalg
  {
    namespace opencl
    {
    
      /** @brief Carries out matrix-vector multiplication with a vandermonde_matrix
      *
      * Implementation of the convenience expression result = prod(mat, vec);
      *
      * @param mat    The matrix
      * @param vec    The vector
      * @param result The result vector
      */
        template<class SCALARTYPE, unsigned int ALIGNMENT, unsigned int VECTOR_ALIGNMENT>
        void prod_impl(const viennacl::vandermonde_matrix<SCALARTYPE, ALIGNMENT> & mat, 
                      const viennacl::vector<SCALARTYPE, VECTOR_ALIGNMENT> & vec,
                            viennacl::vector<SCALARTYPE, VECTOR_ALIGNMENT> & result)
        {
          viennacl::linalg::kernels::fft<SCALARTYPE, 1>::init();
          
          viennacl::ocl::kernel& kernel = viennacl::ocl::current_context()
                                            .get_program(viennacl::linalg::kernels::fft<SCALARTYPE, 1>::program_name())
                                            .get_kernel("vandermonde_prod");
          viennacl::ocl::enqueue(kernel(viennacl::traits::opencl_handle(mat),
                                        viennacl::traits::opencl_handle(vec),
                                        viennacl::traits::opencl_handle(result),
                                        static_cast<cl_uint>(mat.size1())));
        }
        
    } //namespace opencl
  } //namespace linalg
} //namespace viennacl


#endif
