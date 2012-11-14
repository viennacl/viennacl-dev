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
        
        mem_handle() : active_handle_(MEMORY_NOT_INITIALIZED), size_in_bytes_(0) {}
        
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
            else if (active_handle_ == MAIN_MEMORY)
            {
              active_handle_ = new_id;
            }
            else if (active_handle_ == OPENCL_MEMORY)
            {
#ifdef VIENNACL_WITH_OPENCL
              active_handle_ = new_id;
#else
              throw "compiled without OpenCL suppport!";
#endif              
            }
            else if (active_handle_ == CUDA_MEMORY)
            {
#ifdef VIENNACL_WITH_CUDA
              active_handle_ = new_id;
#else
              throw "compiled without CUDA suppport!";
#endif              
            }
            else
              throw "invalid new memory region!";
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
        
        std::size_t raw_size() const               { return size_in_bytes_; }
        void        raw_size(std::size_t new_size) { size_in_bytes_ = new_size; }
        
      private:
        memory_types active_handle_;
        ram_handle_type ram_handle_;
#ifdef VIENNACL_WITH_OPENCL
        viennacl::ocl::handle<cl_mem> opencl_handle_;
#endif
#ifdef VIENNACL_WITH_CUDA
        cuda_handle_type        cuda_handle_;
#endif
        std::size_t size_in_bytes_;
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
    // * memory_write(src, offset, size, ptr)
    // * memory_read(src, offset, size, ptr)
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
            handle.raw_size(size_in_bytes);
            break;
#ifdef VIENNACL_WITH_OPENCL
          case OPENCL_MEMORY:
            handle.opencl_handle() = opencl::memory_create(size_in_bytes, host_ptr);
            handle.raw_size(size_in_bytes);
            break;
#endif
#ifdef VIENNACL_WITH_CUDA
          case CUDA_MEMORY:
            handle.cuda_handle() = cuda::memory_create(size_in_bytes, host_ptr);
            handle.raw_size(size_in_bytes);
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
      assert( src_buffer.get_active_handle_id() == dst_buffer.get_active_handle_id() && bool("memory_copy() must be called on buffers from the same domain") );
      
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
    
    
    
    namespace detail
    {
      template <typename T>
      std::size_t element_size(memory_types mem_type)
      {
        return sizeof(T);
      }

      
      template <>
      inline std::size_t element_size<unsigned long>(memory_types mem_type)
      {
#ifdef VIENNACL_WITH_OPENCL
        if (mem_type == OPENCL_MEMORY)
          return sizeof(cl_ulong);
#endif
        return sizeof(unsigned long);
      }

      template <>
      inline std::size_t element_size<long>(memory_types mem_type)
      {
#ifdef VIENNACL_WITH_OPENCL
        if (mem_type == OPENCL_MEMORY)
          return sizeof(cl_long);
#endif
        return sizeof(long);
      }
      
      
      template <>
      inline std::size_t element_size<unsigned int>(memory_types mem_type)
      {
#ifdef VIENNACL_WITH_OPENCL
        if (mem_type == OPENCL_MEMORY)
          return sizeof(cl_uint);
#endif
        return sizeof(unsigned int);
      }

      template <>
      inline std::size_t element_size<int>(memory_types mem_type)
      {
#ifdef VIENNACL_WITH_OPENCL
        if (mem_type == OPENCL_MEMORY)
          return sizeof(cl_int);
#endif
        return sizeof(int);
      }
      
      
    }

    
    /** @brief Switches the active memory domain within a memory handle. Data is copied if the new active domain differs from the old one. Memory in the source handle is not free'd. */
    template <typename DataType>
    void switch_memory_domain(mem_handle & handle, viennacl::backend::memory_types new_mem_domain)
    {
      if (handle.get_active_handle_id() == new_mem_domain)
        return;
      
      std::size_t size_dst = detail::element_size<DataType>(handle.get_active_handle_id());
      std::size_t size_src = detail::element_size<DataType>(new_mem_domain);
      
      if (size_dst != size_src)  // OpenCL data element size not the same as host data element size
      {
        throw "Heterogeneous data element sizes not yet supported!";
      }
      else //no data conversion required
      {
        if (handle.get_active_handle_id() == MAIN_MEMORY) //we can access the existing data directly
        {
          switch (new_mem_domain)
          {
#ifdef VIENNACL_WITH_OPENCL
            case OPENCL_MEMORY:
              handle.opencl_handle() = opencl::memory_create(handle.raw_size(), handle.ram_handle().get());
              break;
#endif              
#ifdef VIENNACL_WITH_CUDA
            case CUDA_MEMORY:
              handle.cuda_handle() = cuda::memory_create(handle.raw_size(), handle.ram_handle().get());
              break;
#endif
            default:
              throw "Invalid destination domain";
          }
        }
        else if (handle.get_active_handle_id() == MAIN_MEMORY) // data can be dumped into destination directly
        {
          switch (new_mem_domain)
          {
#ifdef VIENNACL_WITH_OPENCL
            case OPENCL_MEMORY:
              handle.ram_handle() = cpu_ram::memory_create(handle.raw_size());
              opencl::memory_read(handle.opencl_handle(), 0, handle.raw_size(), handle.ram_handle().get());
              break;
#endif
#ifdef VIENNACL_WITH_CUDA
            case CUDA_MEMORY:
              handle.ram_handle() = cpu_ram::memory_create(handle.raw_size());
              cuda::memory_read(handle.cuda_handle(), 0, handle.raw_size(), handle.ram_handle().get());
              break;
#endif
            default:
              throw "Invalid source domain";
          }
        }
        else //copy between CUDA and OpenCL
        {
          std::vector<DataType> buffer( handle.raw_size() / sizeof(DataType));
          
          // read into buffer
          switch (handle.get_active_handle_id())
          {
#ifdef VIENNACL_WITH_OPENCL
            case OPENCL_MEMORY:
              opencl::memory_read(handle.opencl_handle(), 0, handle.raw_size(), &(buffer[0]));
              break;
#endif
#ifdef VIENNACL_WITH_CUDA
            case CUDA_MEMORY:
              cuda::memory_read(handle.cuda_handle(), 0, handle.raw_size(), &(buffer[0]));
              break;
#endif
            default:
              throw "Unsupported source memory domain";
          }

            
          // write
          switch (new_mem_domain)
          {
#ifdef VIENNACL_WITH_OPENCL
            case OPENCL_MEMORY:
              handle.opencl_handle() = opencl::memory_create(handle.raw_size(), &(buffer[0]));
              break;
#endif
#ifdef VIENNACL_WITH_CUDA
            case   CUDA_MEMORY:
              handle.cuda_handle() = cuda::memory_create(handle.raw_size(), &(buffer[0]));
              break;
#endif
            default:
              throw "Unsupported source memory domain";
          }
        }

        // everything succeeded so far, now switch to new domain:
        handle.switch_active_handle_id(new_mem_domain);
        
      } // no data conversion
    }
    
    
    
    
    template <typename DataType>
    void typesafe_memory_copy(mem_handle const & handle_src, mem_handle & handle_dst)
    {
      std::size_t element_size_src = detail::element_size<DataType>(handle_src.get_active_handle_id());
      std::size_t element_size_dst = detail::element_size<DataType>(handle_dst.get_active_handle_id());
      
      if (element_size_src != element_size_dst)
      {
        // Data needs to be converted.
        
        integral_type_host_array<DataType> buffer_dst(handle_dst, handle_src.raw_size() / element_size_src);
        
        //
        // Step 1: Fill buffer_dst depending on where the data resides:
        //
        switch (handle_src.get_active_handle_id())
        {
          case MAIN_MEMORY:
            DataType const * src_data = reinterpret_cast<DataType const *>(handle_src.ram_handle().get());
            for (std::size_t i=0; i<buffer_dst.size(); ++i)
              buffer_dst.set(i, src_data[i]);
            break;
            
#ifdef VIENNACL_WITH_OPENCL
          case OPENCL_MEMORY:
            integral_type_host_array<DataType> buffer_src_opencl(handle_src, handle_src.raw_size() / element_size_src);
            opencl::memory_read(handle_src, 0, buffer_src_opencl.raw_size(), buffer_src_opencl.get());
            for (std::size_t i=0; i<buffer_dst.size(); ++i)
              buffer_dst.set(i, buffer_src_opencl[i]);
            break;
#endif
#ifdef VIENNACL_WITH_CUDA
          case CUDA_MEMORY:
            integral_type_host_array<DataType> buffer_src_cuda(handle_src, handle_src.raw_size() / element_size_src);
            cuda::memory_read(handle_src, 0, buffer_src_cuda.raw_size(), buffer_src_cuda.get());
            for (std::size_t i=0; i<buffer_dst.size(); ++i)
              buffer_dst.set(i, buffer_src_cuda[i]);
            break;
#endif
            
          default:
            throw "unsupported memory domain";
        }

        //
        // Step 2: Write to destination
        //
        if (handle_dst.raw_size() == buffer_dst.raw_size())
          viennacl::backend::memory_write(handle_dst, 0, buffer_dst.raw_size(), buffer_dst.get());
        else
          viennacl::backend::memory_create(handle_dst, buffer_dst.raw_size(), buffer_dst.get());
          
      }
      else
      {
        // No data conversion required.
        switch (handle_src.get_active_handle_id())
        {
          case MAIN_MEMORY:
            switch (handle_dst.get_active_handle_id())
            {
              case MAIN_MEMORY:
              case OPENCL_MEMORY:
              case CUDA_MEMORY:
                if (handle_dst.raw_size() == handle_src.raw_size())
                  viennacl::backend::memory_write(handle_dst, 0, handle_src.raw_size(), handle_src.ram_handle().get());
                else
                  viennacl::backend::memory_create(handle_dst, handle_src.raw_size(), handle_src.ram_handle().get());
                break;
                
              default:
                throw "unsupported destination memory domain";
            }
            break;
            
          case OPENCL_MEMORY:
            switch (handle_dst.get_active_handle_id())
            {
              case MAIN_MEMORY:
                viennacl::backend::memory_read(handle_src, 0, handle_src.raw_size(), handle_dst.ram_handle().get());
                break;
                
              case OPENCL_MEMORY:
                viennacl::backend::memory_copy(handle_src, handle_dst, 0, 0, handle_src.raw_size());
                break;
                
              case CUDA_MEMORY:
                integral_type_host_array<DataType> buffer(handle_src, handle_src.raw_size() / element_size_src);
                viennacl::backend::memory_read(handle_src, 0, handle_src.raw_size(), buffer.get());
                viennacl::backend::memory_write(handle_dst, 0, handle_src.raw_size(), buffer.get());
                break;
                
              default:
                throw "unsupported destination memory domain";
            }
            break;
            
          case CUDA_MEMORY:
            switch (handle_dst.get_active_handle_id())
            {
              case MAIN_MEMORY:
                viennacl::backend::memory_read(handle_src, 0, handle_src.raw_size(), handle_dst.ram_handle().get());
                break;
                
              case OPENCL_MEMORY:
                integral_type_host_array<DataType> buffer(handle_src, handle_src.raw_size() / element_size_src);
                viennacl::backend::memory_read(handle_src, 0, handle_src.raw_size(), buffer.get());
                viennacl::backend::memory_write(handle_dst, 0, handle_src.raw_size(), buffer.get());
                break;
                
              case CUDA_MEMORY:
                viennacl::backend::memory_copy(handle_src, handle_dst, 0, 0, handle_src.raw_size());
                break;
                
              default:
                throw "unsupported destination memory domain";
            }
            break;
            
          default:
            throw "unsupported source memory domain";
        }
        
      }
    }
    
    
  } //backend

  
  //
  // Convenience layer:
  //
  
  template <typename T>
  void switch_memory_domain(T & obj, viennacl::backend::memory_types new_mem_domain)
  {
    obj.switch_memory_domain(new_mem_domain);
  }
  
  
} //viennacl
#endif
