#ifndef VIENNACL_LINALG_ITERATIVE_OPERATIONS_HPP_
#define VIENNACL_LINALG_ITERATIVE_OPERATIONS_HPP_

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

/** @file viennacl/linalg/iterative_operations.hpp
    @brief Implementations of specialized routines for the iterative solvers.
*/

#include "viennacl/forwards.h"
#include "viennacl/range.hpp"
#include "viennacl/scalar.hpp"
#include "viennacl/tools/tools.hpp"
#include "viennacl/meta/predicate.hpp"
#include "viennacl/meta/enable_if.hpp"
#include "viennacl/traits/size.hpp"
#include "viennacl/traits/start.hpp"
#include "viennacl/traits/handle.hpp"
#include "viennacl/traits/stride.hpp"
#include "viennacl/linalg/host_based/iterative_operations.hpp"

#ifdef VIENNACL_WITH_OPENCL
  #include "viennacl/linalg/opencl/iterative_operations.hpp"
#endif

#ifdef VIENNACL_WITH_CUDA
  #include "viennacl/linalg/cuda/iterative_operations.hpp"
#endif

namespace viennacl
{
  namespace linalg
  {

    /** @brief Performs a joint vector update operation needed for an efficient pipelined CG algorithm.
      *
      * This routines computes for vectors 'result', 'p', 'r', 'Ap':
      *   result += alpha * p;
      *   r      -= alpha * Ap;
      *   p       = r + beta * p;
      * and runs the parallel reduction stage for computing inner_prod(r,r)
      */
    template <typename T>
    void pipelined_cg_vector_update(vector_base<T> & result,
                                    T alpha,
                                    vector_base<T> & p,
                                    vector_base<T> & r,
                                    vector_base<T> const & Ap,
                                    T beta,
                                    vector_base<T> & inner_prod_buffer)
    {
      switch (viennacl::traits::handle(result).get_active_handle_id())
      {
        case viennacl::MAIN_MEMORY:
          viennacl::linalg::host_based::pipelined_cg_vector_update(result, alpha, p, r, Ap, beta, inner_prod_buffer);
          break;
#ifdef VIENNACL_WITH_OPENCL
        case viennacl::OPENCL_MEMORY:
          viennacl::linalg::opencl::pipelined_cg_vector_update(result, alpha, p, r, Ap, beta, inner_prod_buffer);
          break;
#endif
#ifdef VIENNACL_WITH_CUDA
        case viennacl::CUDA_MEMORY:
          viennacl::linalg::cuda::pipelined_cg_vector_update(result, alpha, p, r, Ap, beta, inner_prod_buffer);
          break;
#endif
        case viennacl::MEMORY_NOT_INITIALIZED:
          throw memory_exception("not initialised!");
        default:
          throw memory_exception("not implemented");
      }
    }


    /** @brief Performs a joint vector update operation needed for an efficient pipelined CG algorithm.
      *
      * This routines computes for a matrix A and vectors 'p' and 'Ap':
      *   Ap = prod(A, p);
      * and computes the two reduction stages for computing inner_prod(p,Ap), inner_prod(Ap,Ap)
      */
    template <typename MatrixType, typename T>
    void pipelined_cg_prod(MatrixType const & A,
                           vector_base<T> const & p,
                           vector_base<T> & Ap,
                           vector_base<T> & inner_prod_buffer)
    {
      switch (viennacl::traits::handle(p).get_active_handle_id())
      {
        case viennacl::MAIN_MEMORY:
          viennacl::linalg::host_based::pipelined_cg_prod(A, p, Ap, inner_prod_buffer);
          break;
#ifdef VIENNACL_WITH_OPENCL
        case viennacl::OPENCL_MEMORY:
          viennacl::linalg::opencl::pipelined_cg_prod(A, p, Ap, inner_prod_buffer);
          break;
#endif
#ifdef VIENNACL_WITH_CUDA
        case viennacl::CUDA_MEMORY:
          viennacl::linalg::cuda::pipelined_cg_prod(A, p, Ap, inner_prod_buffer);
          break;
#endif
        case viennacl::MEMORY_NOT_INITIALIZED:
          throw memory_exception("not initialised!");
        default:
          throw memory_exception("not implemented");
      }
    }

  } //namespace linalg

} //namespace viennacl


#endif
