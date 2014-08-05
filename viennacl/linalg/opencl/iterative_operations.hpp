#ifndef VIENNACL_LINALG_OPENCL_ITERATIVE_OPERATIONS_HPP_
#define VIENNACL_LINALG_OPENCL_ITERATIVE_OPERATIONS_HPP_

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

/** @file viennacl/linalg/opencl/iterative_operations.hpp
    @brief  Implementations of specialized kernels for fast iterative solvers using OpenCL
*/

#include <cmath>

#include "viennacl/forwards.h"
#include "viennacl/vector_def.hpp"
#include "viennacl/ocl/device.hpp"
#include "viennacl/ocl/handle.hpp"
#include "viennacl/ocl/kernel.hpp"
#include "viennacl/scalar.hpp"
#include "viennacl/tools/tools.hpp"
#include "viennacl/linalg/opencl/common.hpp"
#include "viennacl/linalg/opencl/kernels/iterative.hpp"
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

      template <typename T>
      void pipelined_cg_vector_update(vector_base<T> & result,
                                      T alpha,
                                      vector_base<T> & p,
                                      vector_base<T> & r,
                                      vector_base<T> const & Ap,
                                      T beta,
                                      vector_base<T> & inner_prod_buffer)
      {
        viennacl::ocl::context & ctx = const_cast<viennacl::ocl::context &>(viennacl::traits::opencl_handle(result).context());
        viennacl::linalg::opencl::kernels::iterative<T>::init(ctx);

        viennacl::ocl::kernel & k = ctx.get_kernel(viennacl::linalg::opencl::kernels::iterative<T>::program_name(), "cg_vector_update");
        cl_uint    vec_size = cl_uint(viennacl::traits::size(result));

        viennacl::ocl::enqueue(k(result, alpha, p, r, Ap, beta, inner_prod_buffer, vec_size, viennacl::ocl::local_mem(k.local_work_size() * sizeof(T))));
      }

      template <typename T>
      void pipelined_cg_prod(compressed_matrix<T> const & A,
                             vector_base<T> const & p,
                             vector_base<T> & Ap,
                             vector_base<T> & inner_prod_buffer)
      {
        viennacl::ocl::context & ctx = const_cast<viennacl::ocl::context &>(viennacl::traits::opencl_handle(A).context());
        viennacl::linalg::opencl::kernels::iterative<T>::init(ctx);

        viennacl::ocl::kernel & k = ctx.get_kernel(viennacl::linalg::opencl::kernels::iterative<T>::program_name(), "cg_csr_prod");

        cl_uint vec_size               = cl_uint(viennacl::traits::size(p));
        cl_uint buffer_size_per_vector = cl_uint(inner_prod_buffer.size()) / cl_uint(3);

        k.local_work_size(0, 128);
        k.global_work_size(0, 128*128);
        viennacl::ocl::enqueue(k(A.handle1().opencl_handle(), A.handle2().opencl_handle(), A.handle().opencl_handle(),
                                 p,
                                 Ap,
                                 vec_size,
                                 inner_prod_buffer,
                                 buffer_size_per_vector,
                                 viennacl::ocl::local_mem(k.local_work_size() * sizeof(T)),
                                 viennacl::ocl::local_mem(k.local_work_size() * sizeof(T))
                                ));

      }
    } //namespace opencl
  } //namespace linalg
} //namespace viennacl


#endif
