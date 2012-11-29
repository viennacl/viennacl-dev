#ifndef VIENNACL_TRAITS_HANDLE_HPP_
#define VIENNACL_TRAITS_HANDLE_HPP_

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

/** @file traits/handle.hpp
    @brief Extracts the underlying OpenCL handle from a vector, a matrix, an expression etc.
*/

#include <string>
#include <fstream>
#include <sstream>
#include "viennacl/forwards.h"
#include "viennacl/backend/memory.hpp"

namespace viennacl
{
  namespace traits
  {
    // 
    // Generic memory handle
    //
    template <typename T>
    viennacl::backend::mem_handle & handle(T & obj)
    {
      return obj.handle();
    }

    template <typename T>
    viennacl::backend::mem_handle const & handle(T const & obj)
    {
      return obj.handle();
    }
    
    inline float  handle(float val)  { return val; }  //for unification purposes when passing CPU-scalars to kernels
    inline double handle(double val) { return val; }  //for unification purposes when passing CPU-scalars to kernels

    template <typename T>
    viennacl::backend::mem_handle       & handle(viennacl::scalar_expression< const scalar<T>, const scalar<T>, op_flip_sign> & obj)
    {
      return obj.lhs().handle();
    }

    template <typename T>
    viennacl::backend::mem_handle const & handle(viennacl::scalar_expression< const scalar<T>, const scalar<T>, op_flip_sign> const & obj)
    {
      return obj.lhs().handle();
    }
    
    // proxy objects require extra care (at the moment)
    template <typename T>
    viennacl::backend::mem_handle       & handle(viennacl::vector_range<T>       & obj)
    {
      return obj.get().handle();
    }
    
    template <typename T>
    viennacl::backend::mem_handle const & handle(viennacl::vector_range<T> const & obj)
    {
      return obj.get().handle();
    }


    template <typename T>
    viennacl::backend::mem_handle       & handle(viennacl::vector_slice<T>       & obj)
    {
      return obj.get().handle();
    }

    template <typename T>
    viennacl::backend::mem_handle const & handle(viennacl::vector_slice<T> const & obj)
    {
      return obj.get().handle();
    }



    template <typename T>
    viennacl::backend::mem_handle       & handle(viennacl::matrix_range<T>       & obj)
    {
      return obj.get().handle();
    }

    template <typename T>
    viennacl::backend::mem_handle const & handle(viennacl::matrix_range<T> const & obj)
    {
      return obj.get().handle();
    }


    template <typename T>
    viennacl::backend::mem_handle       & handle(viennacl::matrix_slice<T>      & obj)
    {
      return obj.get().handle();
    }

    template <typename T>
    viennacl::backend::mem_handle const & handle(viennacl::matrix_slice<T> const & obj)
    {
      return obj.get().handle();
    }

    
    //
    // RAM handle extraction
    //
    
    template <typename T>
    typename viennacl::backend::mem_handle::ram_handle_type & ram_handle(T & obj)
    {
      return viennacl::traits::handle(obj).ram_handle();
    }

    template <typename T>
    typename viennacl::backend::mem_handle::ram_handle_type const & ram_handle(T const & obj)
    {
      return viennacl::traits::handle(obj).ram_handle();
    }

    inline viennacl::backend::mem_handle::ram_handle_type & ram_handle(viennacl::backend::mem_handle & h)
    {
      return h.ram_handle();
    }
    
    inline viennacl::backend::mem_handle::ram_handle_type const & ram_handle(viennacl::backend::mem_handle const & h)
    {
      return h.ram_handle();
    }
    
    //
    // OpenCL handle extraction
    //
#ifdef VIENNACL_WITH_OPENCL    
    template <typename T>
    viennacl::ocl::handle<cl_mem> & opencl_handle(T & obj)
    {
      return viennacl::traits::handle(obj).opencl_handle();
    }

    template <typename T>
    viennacl::ocl::handle<cl_mem> const & opencl_handle(T const & obj)
    {
      return viennacl::traits::handle(obj).opencl_handle();
    }
    
    inline float  opencl_handle(float val)  { return val; }  //for unification purposes when passing CPU-scalars to kernels
    inline double opencl_handle(double val) { return val; }  //for unification purposes when passing CPU-scalars to kernels
#endif



    //
    // Active handle ID
    //
    template <typename T>
    viennacl::memory_types active_handle_id(T const & obj)
    {
      return handle(obj).get_active_handle_id();
    }

    template <typename LHS, typename RHS, typename OP>
    viennacl::memory_types active_handle_id(viennacl::vector_expression<LHS, RHS, OP> const & obj)
    {
      return active_handle_id(obj.lhs());
    }

    template <typename LHS, typename RHS, typename OP>
    viennacl::memory_types active_handle_id(viennacl::matrix_expression<LHS, RHS, OP> const & obj)
    {
      return active_handle_id(obj.lhs());
    }
    
  } //namespace traits
} //namespace viennacl
    

#endif
