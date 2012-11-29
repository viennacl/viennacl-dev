#ifndef VIENNACL_BACKEND_CUDA_HPP_
#define VIENNACL_BACKEND_CUDA_HPP_

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

/** @file viennacl/backend/cuda.hpp
    @brief Implementations for the CUDA backend functionality
*/


#include <iostream>
#include <vector>
#include "viennacl/tools/shared_ptr.hpp"

// includes CUDA
#include <cuda_runtime.h>

#define VIENNACL_CUDA_ERROR_CHECK(err)  detail::cuda_error_check (err, __FILE__, __LINE__)

namespace viennacl
{
  namespace backend
  {
    namespace cuda
    {
      typedef viennacl::tools::shared_ptr<char>  handle_type;
      // Requirements for backend:
      
      // * memory_create(size, host_ptr)
      // * memory_copy(src, dest, offset_src, offset_dest, size)
      // * memory_write_from_main_memory(src, offset, size,
      //                                 dest, offset, size)
      // * memory_read_to_main_memory(src, offset, size
      //                              dest, offset, size)
      // * 
      //
      
      namespace detail
      {


        inline void cuda_error_check(cudaError error_code, const char *file, const int line )
        {
          if(cudaSuccess != error_code)
          {
            std::cerr << file << "(" << line << "): " << ": CUDA Runtime API error " << error_code << ": " << cudaGetErrorString( error_code ) << std::endl;
            throw "CUDA error";
          }
        }

        
        template <typename U>
        struct cuda_deleter
        {
          void operator()(U * p) const 
          {
            //std::cout << "Freeing handle " << reinterpret_cast<void *>(p) << std::endl;
            cudaFree(p);
          }
        };
        
      }
      
      inline handle_type  memory_create(std::size_t size_in_bytes, const void * host_ptr = NULL)
      {
        void * dev_ptr = NULL;
        VIENNACL_CUDA_ERROR_CHECK( cudaMalloc(&dev_ptr, size_in_bytes) );
        //std::cout << "Allocated new dev_ptr " << dev_ptr << " of size " <<  size_in_bytes << std::endl;
        
        if (!host_ptr)
          return handle_type(reinterpret_cast<char *>(dev_ptr), detail::cuda_deleter<char>());
        
        handle_type new_handle(reinterpret_cast<char*>(dev_ptr), detail::cuda_deleter<char>());
        
        // copy data:
        //std::cout << "Filling new handle from host_ptr " << host_ptr << std::endl;
        cudaMemcpy(new_handle.get(), host_ptr, size_in_bytes, cudaMemcpyHostToDevice);
        
        return new_handle;
      }
    
      inline void memory_copy(handle_type const & src_buffer,
                              handle_type & dst_buffer,
                              std::size_t src_offset,
                              std::size_t dst_offset,
                              std::size_t bytes_to_copy)
      {
        assert( (dst_buffer.get() != NULL) && bool("Memory not initialized!"));
        assert( (src_buffer.get() != NULL) && bool("Memory not initialized!"));
        
        cudaMemcpy(reinterpret_cast<void *>(dst_buffer.get() + dst_offset),
                   reinterpret_cast<void *>(src_buffer.get() + src_offset),
                   bytes_to_copy,
                   cudaMemcpyDeviceToDevice);
      }
      
      inline void memory_write(handle_type & dst_buffer,
                               std::size_t dst_offset,
                               std::size_t bytes_to_copy,
                               const void * ptr)
      {
        assert( (dst_buffer.get() != NULL) && bool("Memory not initialized!"));
        
        cudaMemcpy(reinterpret_cast<char *>(dst_buffer.get()) + dst_offset,
                   reinterpret_cast<const char *>(ptr),
                   bytes_to_copy,
                   cudaMemcpyHostToDevice);
      }
      
      
      inline void memory_read(handle_type const & src_buffer,
                              std::size_t src_offset,
                              std::size_t bytes_to_copy,
                              void * ptr)
      {
        assert( (src_buffer.get() != NULL) && bool("Memory not initialized!"));
        
        cudaMemcpy(reinterpret_cast<char *>(ptr),
                   reinterpret_cast<char *>(src_buffer.get()) + src_offset,
                   bytes_to_copy,
                   cudaMemcpyDeviceToHost);
      }
    
    } //cuda
  } //backend
} //viennacl
#endif
