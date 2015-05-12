#ifndef VIENNACL_LINALG_AMG_OPERATIONS_HPP_
#define VIENNACL_LINALG_AMG_OPERATIONS_HPP_

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

/** @file viennacl/linalg/amg_operations.hpp
    @brief Implementations of operations for algebraic multigrid
*/

#include "viennacl/forwards.h"
#include "viennacl/scalar.hpp"
#include "viennacl/vector.hpp"
#include "viennacl/matrix.hpp"
#include "viennacl/tools/tools.hpp"
#include "viennacl/linalg/detail/amg/amg_base.hpp"
#include "viennacl/linalg/host_based/amg_operations.hpp"

#ifdef VIENNACL_WITH_OPENCL
  #include "viennacl/linalg/opencl/amg_operations.hpp"
#endif

#ifdef VIENNACL_WITH_CUDA
  #include "viennacl/linalg/cuda/amg_operations.hpp"
#endif

namespace viennacl
{
namespace linalg
{
namespace detail
{
namespace amg
{

template<typename NumericT, typename AMGContextT>
void amg_influence(compressed_matrix<NumericT> const & A, AMGContextT & amg_context, amg_tag & tag)
{
  switch (viennacl::traits::handle(A).get_active_handle_id())
  {
    case viennacl::MAIN_MEMORY:
      viennacl::linalg::host_based::amg::amg_influence(A, amg_context, tag);
      break;
#ifdef VIENNACL_WITH_OPENCL
    case viennacl::OPENCL_MEMORY:
      viennacl::linalg::opencl::amg::amg_influence(A, amg_context, tag);
      break;
#endif
#ifdef VIENNACL_WITH_CUDA
    case viennacl::CUDA_MEMORY:
      viennacl::linalg::cuda::amg::amg_influence(A, amg_context, tag);
      break;
#endif
    case viennacl::MEMORY_NOT_INITIALIZED:
      throw memory_exception("not initialised!");
    default:
      throw memory_exception("not implemented");
  }
}


template<typename NumericT, typename AMGContextT>
void amg_coarse(compressed_matrix<NumericT> const & A, AMGContextT & amg_context, amg_tag & tag)
{
  switch (viennacl::traits::handle(A).get_active_handle_id())
  {
    case viennacl::MAIN_MEMORY:
      viennacl::linalg::host_based::amg::amg_coarse(A, amg_context, tag);
      break;
#ifdef VIENNACL_WITH_OPENCL
    case viennacl::OPENCL_MEMORY:
      viennacl::linalg::opencl::amg::amg_coarse(A, amg_context, tag);
      break;
#endif
#ifdef VIENNACL_WITH_CUDA
    case viennacl::CUDA_MEMORY:
      viennacl::linalg::cuda::amg::amg_coarse(A, amg_context, tag);
      break;
#endif
    case viennacl::MEMORY_NOT_INITIALIZED:
      throw memory_exception("not initialised!");
    default:
      throw memory_exception("not implemented");
  }
}


template<typename NumericT, typename AMGContextT>
void amg_interpol(compressed_matrix<NumericT> const & A,
                  compressed_matrix<NumericT>       & P,
                  AMGContextT & amg_context,
                  amg_tag & tag)
{
  switch (viennacl::traits::handle(A).get_active_handle_id())
  {
    case viennacl::MAIN_MEMORY:
      viennacl::linalg::host_based::amg::amg_interpol(A, P, amg_context, tag);
      break;
#ifdef VIENNACL_WITH_OPENCL
    case viennacl::OPENCL_MEMORY:
      viennacl::linalg::opencl::amg::amg_interpol(A, P, amg_context, tag);
      break;
#endif
#ifdef VIENNACL_WITH_CUDA
    case viennacl::CUDA_MEMORY:
      viennacl::linalg::cuda::amg::amg_interpol(A, P, amg_context, tag);
      break;
#endif
    case viennacl::MEMORY_NOT_INITIALIZED:
      throw memory_exception("not initialised!");
    default:
      throw memory_exception("not implemented");
  }
}

} //namespace amg
} //namespace detail
} //namespace linalg
} //namespace viennacl


#endif
