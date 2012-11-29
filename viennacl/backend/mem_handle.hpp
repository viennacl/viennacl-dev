#ifndef VIENNACL_BACKEND_MEM_HANDLE_HPP
#define VIENNACL_BACKEND_MEM_HANDLE_HPP

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

/** @file viennacl/backend/mem_handle.hpp
    @brief Implements the multi-memory-domain handle
*/

#include <vector>
#include <cassert>
#include "viennacl/forwards.h"
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
    

// if a user compiles with CUDA, it is reasonable to expect that CUDA should be the default
#ifdef VIENNACL_WITH_CUDA
    inline memory_types default_memory_type() { return CUDA_MEMORY; }
#elif defined(VIENNACL_WITH_OPENCL)
    inline memory_types default_memory_type() { return OPENCL_MEMORY; }
#else
    inline memory_types default_memory_type() { return MAIN_MEMORY; }
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

    
  } //backend

  
} //viennacl
#endif
