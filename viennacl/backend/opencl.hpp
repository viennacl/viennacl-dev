#ifndef VIENNACL_BACKEND_OPENCL_HPP_
#define VIENNACL_BACKEND_OPENCL_HPP_

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

/** @file viennacl/backend/opencl.hpp
    @brief Implementations for the OpenCL backend functionality
*/


#include <vector>
#include "viennacl/ocl/handle.hpp"
#include "viennacl/ocl/backend.hpp"

namespace viennacl
{
  namespace backend
  {
    namespace opencl
    {
    
      // Requirements for backend:
      
      // * memory_create(size, host_ptr)
      // * memory_copy(src, dest, offset_src, offset_dest, size)
      // * memory_write_from_main_memory(src, offset, size,
      //                                 dest, offset, size)
      // * memory_read_to_main_memory(src, offset, size
      //                              dest, offset, size)
      // * 
      //
      
      inline cl_mem memory_create(std::size_t size_in_bytes, const void * host_ptr = NULL)
      {
        //std::cout << "Creating buffer (" << size_in_bytes << " bytes) host buffer " << host_ptr << std::endl;
        return viennacl::ocl::current_context().create_memory_without_smart_handle(CL_MEM_READ_WRITE, size_in_bytes, const_cast<void *>(host_ptr));
      }
    
      inline void memory_copy(viennacl::ocl::handle<cl_mem> const & src_buffer,
                       viennacl::ocl::handle<cl_mem> & dst_buffer,
                       std::size_t src_offset,
                       std::size_t dst_offset,
                       std::size_t bytes_to_copy)
      {
        cl_int err = clEnqueueCopyBuffer(viennacl::ocl::get_queue().handle().get(),
                                         src_buffer.get(),
                                         dst_buffer.get(),
                                         src_offset,
                                         dst_offset,
                                         bytes_to_copy,
                                         0, NULL, NULL);  //events
        VIENNACL_ERR_CHECK(err);
      }
      
      inline void memory_write(viennacl::ocl::handle<cl_mem> & dst_buffer,
                        std::size_t dst_offset,
                        std::size_t bytes_to_copy,
                        const void * ptr)
      {
        //std::cout << "Writing data (" << bytes_to_copy << " bytes, offset " << dst_offset << ") to OpenCL buffer" << std::endl;
        cl_int err = clEnqueueWriteBuffer(viennacl::ocl::get_queue().handle().get(),
                                          dst_buffer.get(),
                                          CL_TRUE,             //blocking
                                          dst_offset,
                                          bytes_to_copy,
                                          ptr,
                                          0, NULL, NULL);      //events
        VIENNACL_ERR_CHECK(err);
      }
      
      
      inline void memory_read(viennacl::ocl::handle<cl_mem> const & src_buffer,
                       std::size_t src_offset,
                       std::size_t bytes_to_copy,
                       void * ptr)
      {
        //std::cout << "Reading data (" << bytes_to_copy << " bytes, offset " << src_offset << ") from OpenCL buffer " << src_buffer.get() << " to " << ptr << std::endl;
        cl_int err =  clEnqueueReadBuffer(viennacl::ocl::get_queue().handle().get(),
                                          src_buffer.get(),
                                          CL_TRUE,             //blocking
                                          src_offset,
                                          bytes_to_copy,
                                          ptr,
                                          0, NULL, NULL);      //events
        VIENNACL_ERR_CHECK(err);
      }
      
    
    }
  } //backend
} //viennacl
#endif
