#ifndef VIENNACL_TRAITS_HANDLE_HPP_
#define VIENNACL_TRAITS_HANDLE_HPP_

/* =========================================================================
   Copyright (c) 2010-2013, Institute for Microelectronics,
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

/** @file viennacl/traits/handle.hpp
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
    /** @brief Returns the generic memory handle of an object. Non-const version. */
    template <typename T>
    viennacl::backend::mem_handle & handle(T & obj)
    {
      return obj.handle();
    }

    /** @brief Returns the generic memory handle of an object. Const-version. */
    template <typename T>
    viennacl::backend::mem_handle const & handle(T const & obj)
    {
      return obj.handle();
    }
    
    /** \cond */
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
    viennacl::backend::mem_handle       & handle(viennacl::vector_base<T>       & obj)
    {
      return obj.handle();
    }
    
    template <typename T>
    viennacl::backend::mem_handle const & handle(viennacl::vector_base<T> const & obj)
    {
      return obj.handle();
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

    template <typename LHS, typename RHS, typename OP>
    viennacl::backend::mem_handle const & handle(viennacl::vector_expression<LHS, RHS, OP> const & obj)
    {
      return handle(obj.lhs());
    }

    template <typename LHS, typename RHS, typename OP>
    viennacl::backend::mem_handle const & handle(viennacl::matrix_expression<LHS, RHS, OP> const & obj)
    {
      return handle(obj.lhs());
    }
    
    /** \endcond */    
    
    //
    // RAM handle extraction
    //
    /** @brief Generic helper routine for extracting the RAM handle of a ViennaCL object. Non-const version. */
    template <typename T>
    typename viennacl::backend::mem_handle::ram_handle_type & ram_handle(T & obj)
    {
      return viennacl::traits::handle(obj).ram_handle();
    }
    
    /** @brief Generic helper routine for extracting the RAM handle of a ViennaCL object. Const version. */
    template <typename T>
    typename viennacl::backend::mem_handle::ram_handle_type const & ram_handle(T const & obj)
    {
      return viennacl::traits::handle(obj).ram_handle();
    }

    /** \cond */
    inline viennacl::backend::mem_handle::ram_handle_type & ram_handle(viennacl::backend::mem_handle & h)
    {
      return h.ram_handle();
    }
    
    inline viennacl::backend::mem_handle::ram_handle_type const & ram_handle(viennacl::backend::mem_handle const & h)
    {
      return h.ram_handle();
    }
    /** \endcond */
    
    //
    // OpenCL handle extraction
    //
#ifdef VIENNACL_WITH_OPENCL    
    /** @brief Generic helper routine for extracting the OpenCL handle of a ViennaCL object. Non-const version. */
    template <typename T>
    viennacl::ocl::handle<cl_mem> & opencl_handle(T & obj)
    {
      return viennacl::traits::handle(obj).opencl_handle();
    }

    /** @brief Generic helper routine for extracting the OpenCL handle of a ViennaCL object. Const version. */
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
    /** @brief Returns an ID for the currently active memory domain of an object */
    template <typename T>
    viennacl::memory_types active_handle_id(T const & obj)
    {
      return handle(obj).get_active_handle_id();
    }

    /** \cond */
    template <typename LHS, typename RHS, typename OP>
    viennacl::memory_types active_handle_id(viennacl::vector_expression<LHS, RHS, OP> const & obj);

    template <typename LHS, typename RHS, typename OP>
    viennacl::memory_types active_handle_id(viennacl::scalar_expression<LHS, RHS, OP> const & obj)
    {
      return active_handle_id(obj.lhs());
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
    /** \endcond */
    
  } //namespace traits
} //namespace viennacl
    

#endif
