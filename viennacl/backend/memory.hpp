#ifndef VIENNACL_BACKEND_MEMORY_HPP
#define VIENNACL_BACKEND_MEMORY_HPP

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

/** @file viennacl/backend/memory.hpp
    @brief Main interface routines for memory management
*/

#include <vector>
#include "viennacl/backend/opencl.hpp"

namespace viennacl
{
  namespace backend
  {
    
    enum memory_types
    {
      MEMORY_NOT_INITIALIZED
      , MAIN_MEMORY
//#ifdef VIENNACL_WITH_OPENCL
      , OPENCL_MEMORY
//#endif
#ifdef VIENNACL_WITH_CUDA
      , CUDA_MEMORY
#endif
    };
    
    inline memory_types default_memory_type() { return OPENCL_MEMORY; }
    
    class mem_handle
    {
      public:
        mem_handle() : active_handle_(MEMORY_NOT_INITIALIZED) {}
        
        void *       & ram_handle() { return ram_handle_; }
        void * const & ram_handle() const { return ram_handle_; }
        
        viennacl::ocl::handle<cl_mem>       & opencl_handle() { return opencl_handle_; }
        viennacl::ocl::handle<cl_mem> const & opencl_handle() const { return opencl_handle_; }
        
        memory_types get_active_handle_id() const { return active_handle_; }
        void switch_active_handle_id(memory_types new_id)
        {
          if (new_id != active_handle_)
          {
            if (active_handle_ == MEMORY_NOT_INITIALIZED)
              active_handle_ = new_id;
            else
              throw "Not implemented!";
          }
        }
        
        bool operator==(mem_handle const & other) const
        {
          if (active_handle_ != other.active_handle_)
            return false;
          
          switch (active_handle_)
          {
            case OPENCL_MEMORY:
              return opencl_handle_.get() == other.opencl_handle_.get();
            default:
              return false;
          }
          
          return false;
        }
        
        bool operator!=(mem_handle const & other) const { return !(*this == other); }

        void swap(mem_handle & other)
        {
          // swap handle type:
          memory_types active_handle_tmp = other.active_handle_;
          other.active_handle_ = active_handle_;
          active_handle_ = active_handle_tmp;
          
          // swap ram handle:
          void * ram_handle_tmp = other.ram_handle_;
          other.ram_handle_ = ram_handle_;
          ram_handle_ = ram_handle_tmp;
          
          // swap OpenCL handle:
          opencl_handle_.swap(other.opencl_handle_);
        }
        
      private:
        memory_types active_handle_;
        void * ram_handle_;
//#ifdef VIENNACL_WITH_OPENCL
        viennacl::ocl::handle<cl_mem> opencl_handle_;
//#endif
#ifdef VIENNACL_WITH_CUDA
        viennacl::cuda::handle        cuda_handle_;
#endif
    };

    
    
    // Requirements for backend:
    
    // ---- Memory ----
    //
    // * memory_create(size, host_ptr)
    // * memory_copy(src, dest, offset_src, offset_dest, size)
    // * memory_write_from_main_memory(src, offset, size,
    //                                 dest, offset, size)
    // * memory_read_to_main_memory(src, offset, size
    //                              dest, offset, size)
    // * memory_free()
    //

    inline void memory_create(mem_handle & handle, std::size_t size_in_bytes, void * host_ptr = NULL)
    {
      if (size_in_bytes > 0)
      {
        switch(handle.get_active_handle_id())
        {
          case OPENCL_MEMORY:
            handle.opencl_handle() = opencl::memory_create(size_in_bytes, host_ptr);
            break;
          default:
            throw "unknown memory handle!";
        }
      }
    }
    
    inline void memory_copy(mem_handle const & src_buffer,
                            mem_handle & dst_buffer,
                            std::size_t src_offset,
                            std::size_t dst_offset,
                            std::size_t bytes_to_copy)
    {
      assert( (src_buffer.get_active_handle_id() == dst_buffer.get_active_handle_id()) && "Different memory locations for source and destination! Not supported!");
      
      if (bytes_to_copy > 0)
      {
        switch(src_buffer.get_active_handle_id())
        {
          case OPENCL_MEMORY:
            opencl::memory_copy(src_buffer.opencl_handle(), dst_buffer.opencl_handle(), src_offset, dst_offset, bytes_to_copy);
            break;
          default:
            throw "unknown memory handle!";
        }
      }
    }
    
    inline void memory_write(mem_handle & dst_buffer,
                             std::size_t dst_offset,
                             std::size_t bytes_to_write,
                             const void * ptr)
    {
      if (bytes_to_write > 0)
      {
        switch(dst_buffer.get_active_handle_id())
        {
          case OPENCL_MEMORY:
            opencl::memory_write(dst_buffer.opencl_handle(), dst_offset, bytes_to_write, ptr);
            break;
          default:
            throw "unknown memory handle!";
        }
      }
    }
    
    inline void memory_read(mem_handle const & src_buffer,
                            std::size_t src_offset,
                            std::size_t bytes_to_read,
                            void * ptr)
    {
      if (bytes_to_read > 0)
      {
        switch(src_buffer.get_active_handle_id())
        {
          case OPENCL_MEMORY:
            opencl::memory_read(src_buffer.opencl_handle(), src_offset, bytes_to_read, ptr);
            break;
          default:
            throw "unknown memory handle!";
        }
      }
    }
    
    
  } //backend
} //viennacl
#endif
