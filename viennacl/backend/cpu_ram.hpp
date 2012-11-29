#ifndef VIENNACL_BACKEND_CPU_RAM_HPP_
#define VIENNACL_BACKEND_CPU_RAM_HPP_

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

/** @file viennacl/backend/cpu_ram.hpp
    @brief Implementations for the OpenCL backend functionality
*/


#include <vector>
#include "viennacl/tools/shared_ptr.hpp"

namespace viennacl
{
  namespace backend
  {
    namespace cpu_ram
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
        template<class U>
        struct array_deleter
        {
          void operator()(U* p) const { delete[] p; }
        };
        
      }
      
      inline handle_type  memory_create(std::size_t size_in_bytes, const void * host_ptr = NULL)
      {
        if (!host_ptr)
          return handle_type(new char[size_in_bytes], detail::array_deleter<char>());
        
        handle_type new_handle(new char[size_in_bytes], detail::array_deleter<char>());
        
        // copy data:
        char * raw_ptr = new_handle.get();
        const char * data_ptr = static_cast<const char *>(host_ptr);
        for (std::size_t i=0; i<size_in_bytes; ++i)
          raw_ptr[i] = data_ptr[i];
        
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
        
        for (std::size_t i=0; i<bytes_to_copy; ++i)
          dst_buffer.get()[i+dst_offset] = src_buffer.get()[i + src_offset];
      }
      
      inline void memory_write(handle_type & dst_buffer,
                               std::size_t dst_offset,
                               std::size_t bytes_to_copy,
                               const void * ptr)
      {
        assert( (dst_buffer.get() != NULL) && bool("Memory not initialized!"));
        
        for (std::size_t i=0; i<bytes_to_copy; ++i)
          dst_buffer.get()[i+dst_offset] = static_cast<const char *>(ptr)[i];
      }
      
      
      inline void memory_read(handle_type const & src_buffer,
                              std::size_t src_offset,
                              std::size_t bytes_to_copy,
                              void * ptr)
      {
        assert( (src_buffer.get() != NULL) && bool("Memory not initialized!"));
        
        for (std::size_t i=0; i<bytes_to_copy; ++i)
          static_cast<char *>(ptr)[i] = src_buffer.get()[i+src_offset];
      }
      
    
    }
  } //backend
} //viennacl
#endif
