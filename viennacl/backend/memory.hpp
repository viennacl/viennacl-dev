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
#include <cassert>
#include "viennacl/tools/shared_ptr.hpp"
#include "viennacl/backend/cpu_ram.hpp"

#ifdef VIENNACL_WITH_OPENCL
  #include "viennacl/backend/opencl.hpp"
#endif

#ifdef VIENNACL_WITH_CUDA
  #include "viennacl/backend/cuda.hpp"
#endif


namespace viennacl
{
  namespace backend
  {
    
    enum memory_types
    {
      MEMORY_NOT_INITIALIZED
      , MAIN_MEMORY
      , OPENCL_MEMORY
      , CUDA_MEMORY
    };


// if a user compiles with CUDA, it is reasonable to expect that CUDA should be the default
#ifdef VIENNACL_WITH_CUDA
    inline memory_types default_memory_type() { return CUDA_MEMORY; }
    inline void finish() { cudaDeviceSynchronize(); }
#elif defined(VIENNACL_WITH_OPENCL)
    inline memory_types default_memory_type() { return OPENCL_MEMORY; }
    inline void finish() { viennacl::ocl::get_queue().finish(); }
#else
    inline memory_types default_memory_type() { return MAIN_MEMORY; }
    inline void finish() {}
#endif    


    class mem_handle
    {
      public:
        typedef viennacl::tools::shared_ptr<char>      ram_handle_type;
        typedef viennacl::tools::shared_ptr<char>      cuda_handle_type;
        
        mem_handle() : active_handle_(MEMORY_NOT_INITIALIZED) {}
        
        ram_handle_type       & ram_handle()       { return ram_handle_; }
        ram_handle_type const & ram_handle() const { return ram_handle_; }
        
#ifdef VIENNACL_WITH_OPENCL
        viennacl::ocl::handle<cl_mem>       & opencl_handle()       { return opencl_handle_; }
        viennacl::ocl::handle<cl_mem> const & opencl_handle() const { return opencl_handle_; }
#endif        

#ifdef VIENNACL_WITH_CUDA
        cuda_handle_type       & cuda_handle()       { return cuda_handle_; }
        cuda_handle_type const & cuda_handle() const { return cuda_handle_; }
#endif        

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
            case MAIN_MEMORY:
              return ram_handle_.get() == other.ram_handle_.get();
#ifdef VIENNACL_WITH_OPENCL
            case OPENCL_MEMORY:
              return opencl_handle_.get() == other.opencl_handle_.get();
#endif
#ifdef VIENNACL_WITH_CUDA
            case CUDA_MEMORY:
              return cuda_handle_.get() == other.cuda_handle_.get();
#endif
            default: break;
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
          ram_handle_type ram_handle_tmp = other.ram_handle_;
          other.ram_handle_ = ram_handle_;
          ram_handle_ = ram_handle_tmp;
          
          // swap OpenCL handle:
#ifdef VIENNACL_WITH_OPENCL
          opencl_handle_.swap(other.opencl_handle_);
#endif          
#ifdef VIENNACL_WITH_CUDA
          cuda_handle_type cuda_handle_tmp = other.cuda_handle_;
          other.cuda_handle_ = cuda_handle_;
          cuda_handle_ = cuda_handle_tmp;
#endif          
        }
        
      private:
        memory_types active_handle_;
        ram_handle_type ram_handle_;
#ifdef VIENNACL_WITH_OPENCL
        viennacl::ocl::handle<cl_mem> opencl_handle_;
#endif
#ifdef VIENNACL_WITH_CUDA
        cuda_handle_type        cuda_handle_;
#endif
    };


    template <typename T>
    struct integral_type_host_array;
    
    template <>
    struct integral_type_host_array<unsigned int>
    {
      public:
        explicit integral_type_host_array() : convert_to_opencl_( (default_memory_type() == OPENCL_MEMORY) ? true : false), bytes_buffer_(NULL), buffer_size_(0) {}
        
        explicit integral_type_host_array(mem_handle const & handle, std::size_t num = 0) : convert_to_opencl_(false), bytes_buffer_(NULL), buffer_size_(sizeof(unsigned int) * num)
        {
          
#ifdef VIENNACL_WITH_OPENCL
          memory_types mem_type = handle.get_active_handle_id();
          if (mem_type == MEMORY_NOT_INITIALIZED)
            mem_type = default_memory_type();
          
          if (mem_type == OPENCL_MEMORY)
          {
            convert_to_opencl_ = true;
            buffer_size_ = sizeof(cl_uint) * num;
          }
#endif

          if (num > 0)
          {
            bytes_buffer_ = new char[buffer_size_];
            
            for (std::size_t i=0; i<buffer_size_; ++i)
              bytes_buffer_[i] = 0;
          }
        }
        
        ~integral_type_host_array() { delete[] bytes_buffer_; }
        
        template <typename U>
        void set(std::size_t index, U value)
        {
#ifdef VIENNACL_WITH_OPENCL
          if (convert_to_opencl_)
            reinterpret_cast<cl_uint *>(bytes_buffer_)[index] = static_cast<cl_uint>(value);
          else
#endif
            reinterpret_cast<unsigned int *>(bytes_buffer_)[index] = static_cast<unsigned int>(value);
        }

        void * get() { return reinterpret_cast<void *>(bytes_buffer_); }
        unsigned int operator[](std::size_t index) const 
        {
          assert(index < size() && bool("index out of bounds"));
#ifdef VIENNACL_WITH_OPENCL
          if (convert_to_opencl_)
            return static_cast<unsigned int>(reinterpret_cast<cl_uint *>(bytes_buffer_)[index]);
#endif
          return reinterpret_cast<unsigned int *>(bytes_buffer_)[index];
        }
        
        std::size_t raw_size() const { return buffer_size_; }
        std::size_t element_size() const
        {
#ifdef VIENNACL_WITH_OPENCL
          if (convert_to_opencl_)
            return sizeof(cl_uint);
#endif
          return sizeof(unsigned int); 
        }
        std::size_t size() const { return buffer_size_ / element_size(); }
        
      private:
        bool convert_to_opencl_;
        char * bytes_buffer_;
        std::size_t buffer_size_;
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
        if (handle.get_active_handle_id() == MEMORY_NOT_INITIALIZED)
          handle.switch_active_handle_id(default_memory_type());
        
        switch(handle.get_active_handle_id())
        {
          case MAIN_MEMORY:
            handle.ram_handle() = cpu_ram::memory_create(size_in_bytes, host_ptr);
            break;
#ifdef VIENNACL_WITH_OPENCL
          case OPENCL_MEMORY:
            handle.opencl_handle() = opencl::memory_create(size_in_bytes, host_ptr);
            break;
#endif
#ifdef VIENNACL_WITH_CUDA
          case CUDA_MEMORY:
            handle.cuda_handle() = cuda::memory_create(size_in_bytes, host_ptr);
            break;
#endif
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
      assert( (src_buffer.get_active_handle_id() == dst_buffer.get_active_handle_id()) && bool("Different memory locations for source and destination! Not supported!"));
      
      if (bytes_to_copy > 0)
      {
        switch(src_buffer.get_active_handle_id())
        {
          case MAIN_MEMORY:
            cpu_ram::memory_copy(src_buffer.ram_handle(), dst_buffer.ram_handle(), src_offset, dst_offset, bytes_to_copy);
            break;
#ifdef VIENNACL_WITH_OPENCL
          case OPENCL_MEMORY:
            opencl::memory_copy(src_buffer.opencl_handle(), dst_buffer.opencl_handle(), src_offset, dst_offset, bytes_to_copy);
            break;
#endif
#ifdef VIENNACL_WITH_CUDA
          case CUDA_MEMORY:
            cuda::memory_copy(src_buffer.cuda_handle(), dst_buffer.cuda_handle(), src_offset, dst_offset, bytes_to_copy);
            break;
#endif
          default:
            throw "unknown memory handle!";
        }
      }
    }

    // TODO: Refine this concept. Maybe move to constructor?
    inline void memory_shallow_copy(mem_handle const & src_buffer,
                                    mem_handle & dst_buffer)
    {
      assert( (dst_buffer.get_active_handle_id() == MEMORY_NOT_INITIALIZED) && bool("Shallow copy on already initialized memory not supported!"));

      switch(src_buffer.get_active_handle_id())
      {
        case MAIN_MEMORY:
          dst_buffer.switch_active_handle_id(src_buffer.get_active_handle_id());
          dst_buffer.ram_handle() = src_buffer.ram_handle();
          break;
#ifdef VIENNACL_WITH_OPENCL
        case OPENCL_MEMORY:
          dst_buffer.switch_active_handle_id(src_buffer.get_active_handle_id());
          dst_buffer.opencl_handle() = src_buffer.opencl_handle();
          break;
#endif
#ifdef VIENNACL_WITH_CUDA
          case CUDA_MEMORY:
            dst_buffer.switch_active_handle_id(src_buffer.get_active_handle_id());
            dst_buffer.cuda_handle() = src_buffer.cuda_handle();
            break;
#endif
        default:
          throw "unknown memory handle!";
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
          case MAIN_MEMORY:
            cpu_ram::memory_write(dst_buffer.ram_handle(), dst_offset, bytes_to_write, ptr);
            break;
#ifdef VIENNACL_WITH_OPENCL
          case OPENCL_MEMORY:
            opencl::memory_write(dst_buffer.opencl_handle(), dst_offset, bytes_to_write, ptr);
            break;
#endif
#ifdef VIENNACL_WITH_CUDA
          case CUDA_MEMORY:
            cuda::memory_write(dst_buffer.cuda_handle(), dst_offset, bytes_to_write, ptr);
            break;
#endif
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
          case MAIN_MEMORY:
            cpu_ram::memory_read(src_buffer.ram_handle(), src_offset, bytes_to_read, ptr);
            break;
#ifdef VIENNACL_WITH_OPENCL
          case OPENCL_MEMORY:
            opencl::memory_read(src_buffer.opencl_handle(), src_offset, bytes_to_read, ptr);
            break;
#endif            
#ifdef VIENNACL_WITH_CUDA
          case CUDA_MEMORY:
            cuda::memory_read(src_buffer.cuda_handle(), src_offset, bytes_to_read, ptr);
            break;
#endif
          default:
            throw "unknown memory handle!";
        }
      }
    }
    
    
  } //backend
} //viennacl
#endif
