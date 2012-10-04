#ifndef VIENNACL_TRAITS_HANDLE_HPP_
#define VIENNACL_TRAITS_HANDLE_HPP_

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
    viennacl::backend::mem_handle       & handle(viennacl::scalar_expression< const scalar<T>, const scalar<T>, op_flip_sign> const & obj)
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
    // OpenCL handle extraction
    //
    
    template <typename T>
    viennacl::ocl::handle<cl_mem> opencl_handle(T & obj)
    {
      return viennacl::traits::handle(obj).opencl_handle();
    }

    inline float  opencl_handle(float val)  { return val; }  //for unification purposes when passing CPU-scalars to kernels
    inline double opencl_handle(double val) { return val; }  //for unification purposes when passing CPU-scalars to kernels

    
  } //namespace traits
} //namespace viennacl
    

#endif
