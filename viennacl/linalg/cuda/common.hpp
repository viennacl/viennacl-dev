#ifndef VIENNACL_LINALG_CUDA_COMMON_HPP_
#define VIENNACL_LINALG_CUDA_COMMON_HPP_

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

/** @file viennacl/linalg/cuda/common.hpp
    @brief Common routines for CUDA execution
*/

#include "viennacl/traits/handle.hpp"

#define VIENNACL_CUDA_LAST_ERROR_CHECK(message)  detail::cuda_last_error_check (message, __FILE__, __LINE__)

namespace viennacl
{
  namespace linalg
  {
    namespace cuda
    {
      namespace detail
      {
        inline void cuda_last_error_check(const char * message, const char * file, const int line )
        {
          cudaError_t error_code = cudaGetLastError();
          
          if(cudaSuccess != error_code)
          {
            std::cerr << file << "(" << line << "): " << ": getLastCudaError() CUDA error " << error_code << ": " << cudaGetErrorString( error_code ) << " @ " << message << std::endl;
            throw "CUDA error";
          }
        }
        
        template <typename ScalarType, typename T>
        typename viennacl::enable_if<    viennacl::is_scalar<T>::value 
                                      || viennacl::is_any_dense_nonstructured_vector<T>::value 
                                      || viennacl::is_any_dense_nonstructured_matrix<T>::value,
                                      ScalarType *>::type
        cuda_arg(T & obj)
        {
          return reinterpret_cast<ScalarType *>(viennacl::traits::handle(obj).cuda_handle().get());
        }

        template <typename ScalarType, typename T>
        typename viennacl::enable_if<    viennacl::is_scalar<T>::value 
                                      || viennacl::is_any_dense_nonstructured_vector<T>::value 
                                      || viennacl::is_any_dense_nonstructured_matrix<T>::value,
                                      const ScalarType *>::type
        cuda_arg(T const & obj)
        {
          return reinterpret_cast<const ScalarType *>(viennacl::traits::handle(obj).cuda_handle().get());
        }

        template <typename ScalarType>
        ScalarType *  cuda_arg(viennacl::backend::mem_handle::cuda_handle_type & h)
        { 
          return reinterpret_cast<ScalarType *>(h.get());
        }
        
        template <typename ScalarType>
        ScalarType const *  cuda_arg(viennacl::backend::mem_handle::cuda_handle_type const & h)
        { 
          return reinterpret_cast<const ScalarType *>(h.get());
        }
        
        template <typename ScalarType>
        ScalarType cuda_arg(ScalarType const & val)  { return val; }
        
        
        
        template <typename T, typename U>
        typename viennacl::backend::mem_handle::cuda_handle_type & arg_reference(viennacl::scalar<T> & s, U) { return s.handle().cuda_handle(); }
        
        template <typename T, typename U>
        typename viennacl::backend::mem_handle::cuda_handle_type const & arg_reference(viennacl::scalar<T> const & s, U) { return s.handle().cuda_handle(); }
        
        // all other cases where T is not a ViennaCL scalar 
        template <typename T>
        typename viennacl::enable_if< viennacl::is_cpu_scalar<T>::value,
                                      float const &>::type
        arg_reference(T, float const & val)  { return val; }
        
        template <typename T>
        typename viennacl::enable_if< viennacl::is_cpu_scalar<T>::value,
                                      double const &>::type
        arg_reference(T, double const & val)  { return val; }
      } //namespace detail

    } //namespace cuda
  } //namespace linalg
} //namespace viennacl


#endif
