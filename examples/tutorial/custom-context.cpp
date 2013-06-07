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

/*
*
*   Tutorial:  Use ViennaCL within user-defined (i.e. your own) OpenCL contexts
*   
*/


//
// include necessary system headers
//
#include <iostream>
#include <string>

#ifndef VIENNACL_WITH_OPENCL
  #define VIENNACL_WITH_OPENCL
#endif

//
// ViennaCL includes
//
#include "viennacl/ocl/backend.hpp"
#include "viennacl/vector.hpp"
#include "viennacl/matrix.hpp"
#include "viennacl/linalg/matrix_operations.hpp"
#include "viennacl/linalg/norm_2.hpp"
#include "viennacl/linalg/prod.hpp"


// Some helper functions for this tutorial:
#include "Random.hpp"




//
// A custom compute kernel which computes an elementwise product of two vectors
// Input: v1 ... vector
//        v2 ... vector
// Output: result ... vector
//
// Algorithm: set result[i] <- v1[i] * v2[i]
//            (in MATLAB notation this is something like 'result = v1 .* v2');
//

const char * my_compute_program = 
"__kernel void elementwise_prod(\n"
"          __global const float * vec1,\n"
"          __global const float * vec2, \n"
"          __global float * result,\n"
"          unsigned int size) \n"
"{ \n"
"  for (unsigned int i = get_global_id(0); i < size; i += get_global_size(0))\n"
"    result[i] = vec1[i] * vec2[i];\n"
"};\n";


int main()
{
  typedef float       ScalarType;
  
  
  /////////////////////////////////////////////////////////////////////////////////////////////////////////
  //////////////////////// Part 1: Set up a custom context and perform a sample operation. ////////////////
  ////////////////////////         This is rather lengthy due to the OpenCL framework.     ////////////////
  ////////////////////////         The following does essentially the same as the          ////////////////
  ////////////////////////         'custom_kernels'-tutorial!                               ////////////////
  /////////////////////////////////////////////////////////////////////////////////////////////////////////
  
  //manually set up a custom OpenCL context:
  std::vector<cl_device_id> device_id_array;
  
  //get all available devices
  viennacl::ocl::platform pf;
  std::cout << "Platform info: " << pf.info() << std::endl;
  std::vector<viennacl::ocl::device> devices = pf.devices(CL_DEVICE_TYPE_DEFAULT);
  std::cout << devices[0].name() << std::endl;
  std::cout << "Number of devices for custom context: " << devices.size() << std::endl;
  
  //set up context using all found devices:
  for (std::size_t i=0; i<devices.size(); ++i)
  {
      device_id_array.push_back(devices[i].id());
  }
     
  std::cout << "Creating context..." << std::endl;
  cl_int err;
  cl_context my_context = clCreateContext(0, device_id_array.size(), &(device_id_array[0]), NULL, NULL, &err);
  VIENNACL_ERR_CHECK(err);
   
  
  //create two Vectors:
  unsigned int vector_size = 10;
  std::vector<ScalarType> vec1(vector_size);
  std::vector<ScalarType> vec2(vector_size);
  std::vector<ScalarType> result(vector_size);
  
  //
  // fill the operands vec1 and vec2:
  //
  for (unsigned int i=0; i<vector_size; ++i)
  {
    vec1[i] = static_cast<ScalarType>(i);
    vec2[i] = static_cast<ScalarType>(vector_size-i);
  }
  
  //
  // create memory in OpenCL context:
  //
  cl_mem mem_vec1 = clCreateBuffer(my_context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, vector_size * sizeof(ScalarType), &(vec1[0]), &err);
  VIENNACL_ERR_CHECK(err);
  cl_mem mem_vec2 = clCreateBuffer(my_context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, vector_size * sizeof(ScalarType), &(vec2[0]), &err);
  VIENNACL_ERR_CHECK(err);
  cl_mem mem_result = clCreateBuffer(my_context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, vector_size * sizeof(ScalarType), &(result[0]), &err);
  VIENNACL_ERR_CHECK(err);

  // 
  // create a command queue for each device:
  // 
  
  std::vector<cl_command_queue> queues(devices.size());
  for (std::size_t i=0; i<devices.size(); ++i)
  {
    queues[i] = clCreateCommandQueue(my_context, devices[i].id(), 0, &err);
    VIENNACL_ERR_CHECK(err);
  }
  
  // 
  // create and build a program in the context:
  // 
  std::size_t source_len = std::string(my_compute_program).length();
  cl_program my_prog = clCreateProgramWithSource(my_context, 1, &my_compute_program, &source_len, &err);
  err = clBuildProgram(my_prog, 0, NULL, NULL, NULL, NULL);
  
/*            char buffer[1024];
            cl_build_status status;
            clGetProgramBuildInfo(my_prog, devices[1].id(), CL_PROGRAM_BUILD_STATUS, sizeof(cl_build_status), &status, NULL);
            clGetProgramBuildInfo(my_prog, devices[1].id(), CL_PROGRAM_BUILD_LOG, sizeof(char)*1024, &buffer, NULL);
            std::cout << "Build Scalar: Err = " << err << " Status = " << status << std::endl;
            std::cout << "Log: " << buffer << std::endl;*/
  
  VIENNACL_ERR_CHECK(err);
  
  // 
  // create a kernel from the program:
  // 
  const char * kernel_name = "elementwise_prod";
  cl_kernel my_kernel = clCreateKernel(my_prog, kernel_name, &err);
  VIENNACL_ERR_CHECK(err);

  
  //
  // Execute elementwise_prod kernel on first queue: result = vec1 .* vec2;
  //
  err = clSetKernelArg(my_kernel, 0, sizeof(cl_mem), (void*)&mem_vec1);
  VIENNACL_ERR_CHECK(err);
  err = clSetKernelArg(my_kernel, 1, sizeof(cl_mem), (void*)&mem_vec2);
  VIENNACL_ERR_CHECK(err);
  err = clSetKernelArg(my_kernel, 2, sizeof(cl_mem), (void*)&mem_result);
  VIENNACL_ERR_CHECK(err);
  err = clSetKernelArg(my_kernel, 3, sizeof(unsigned int), (void*)&vector_size);
  VIENNACL_ERR_CHECK(err);
  std::size_t global_size = vector_size;
  std::size_t local_size = vector_size;
  err = clEnqueueNDRangeKernel(queues[0], my_kernel, 1, NULL, &global_size, &local_size, 0, NULL, NULL);
  VIENNACL_ERR_CHECK(err);
  
  
  //
  // Read and output result:
  //
  err = clEnqueueReadBuffer(queues[0], mem_vec1, CL_TRUE, 0, sizeof(ScalarType)*vector_size, &(vec1[0]), 0, NULL, NULL);
  VIENNACL_ERR_CHECK(err);
  err = clEnqueueReadBuffer(queues[0], mem_result, CL_TRUE, 0, sizeof(ScalarType)*vector_size, &(result[0]), 0, NULL, NULL);
  VIENNACL_ERR_CHECK(err);

  std::cout << "vec1  : ";
  for (std::size_t i=0; i<vec1.size(); ++i)
    std::cout << vec1[i] << " ";
  std::cout << std::endl;

  std::cout << "vec2  : ";
  for (std::size_t i=0; i<vec2.size(); ++i)
    std::cout << vec2[i] << " ";
  std::cout << std::endl;

  std::cout << "result: ";
  for (std::size_t i=0; i<result.size(); ++i)
    std::cout << result[i] << " ";
  std::cout << std::endl;
  
  ////////////////////////////////////////////////////////////////////////////////////////////////////////
  /////////////////////// Part 2: Let ViennaCL use the already created context: //////////////////////////
  ////////////////////////////////////////////////////////////////////////////////////////////////////////

  //Tell ViennaCL to use the previously created context.
  //This context is assigned an id '0' when using viennacl::ocl::switch_context().
  viennacl::ocl::setup_context(0, my_context, device_id_array, queues);
  viennacl::ocl::switch_context(0); //activate the new context (only mandatory with context-id not equal to zero)
  
  //
  // Proof that ViennaCL really uses the new context:
  //
  std::cout << "Existing context: " << my_context << std::endl;
  std::cout << "ViennaCL uses context: " << viennacl::ocl::current_context().handle().get() << std::endl;

  //
  // Wrap existing OpenCL objects into ViennaCL:
  //
  viennacl::vector<ScalarType> vcl_vec1(mem_vec1, vector_size);
  viennacl::vector<ScalarType> vcl_vec2(mem_vec2, vector_size);
  viennacl::vector<ScalarType> vcl_result(mem_result, vector_size);
  viennacl::scalar<ScalarType> vcl_s = 2.0;

  std::cout << "Standard vector operations within ViennaCL:" << std::endl;
  vcl_result = vcl_s * vcl_vec1 + vcl_vec2;
  
  std::cout << "vec1  : ";
  std::cout << vcl_vec1 << std::endl;

  std::cout << "vec2  : ";
  std::cout << vcl_vec2 << std::endl;

  std::cout << "result: ";
  std::cout << vcl_result << std::endl;
  
  //
  // We can also reuse the existing elementwise_prod kernel. 
  // Therefore, we first have to make the existing program known to ViennaCL
  // For more details on the three lines, see tutorial 'custom-kernels'
  //
  std::cout << "Using existing kernel within the OpenCL backend of ViennaCL:" << std::endl;
  viennacl::ocl::program & my_vcl_prog = viennacl::ocl::current_context().add_program(my_prog, "my_compute_program");
  viennacl::ocl::kernel & my_vcl_kernel = my_vcl_prog.add_kernel("elementwise_prod");
  viennacl::ocl::enqueue(my_vcl_kernel(vcl_vec1, vcl_vec2, vcl_result, static_cast<cl_uint>(vcl_vec1.size())));  //Note that std::size_t might differ between host and device. Thus, a cast to cl_uint is necessary here.
  
  std::cout << "vec1  : ";
  std::cout << vcl_vec1 << std::endl;

  std::cout << "vec2  : ";
  std::cout << vcl_vec2 << std::endl;

  std::cout << "result: ";
  std::cout << vcl_result << std::endl;
  
  
  //
  // Since a linear piece of memory can be interpreted in several ways, 
  // we will now create a 3x3 row-major matrix out of the linear memory in mem_vec1/
  // The first three entries in vcl_vec2 and vcl_result are used to carry out matrix-vector products:
  //
  viennacl::matrix<ScalarType> vcl_matrix(mem_vec1, 3, 3);
  
  vcl_vec2.resize(3);   //note that the resize operation leads to new memory, thus vcl_vec2 is now at a different memory location (values are copied)
  vcl_result.resize(3); //note that the resize operation leads to new memory, thus vcl_vec2 is now at a different memory location (values are copied)
  vcl_result = viennacl::linalg::prod(vcl_matrix, vcl_vec2);

  std::cout << "result of matrix-vector product: ";
  std::cout << vcl_result << std::endl;

  //
  //  That's it.
  //
  std::cout << "!!!! TUTORIAL COMPLETED SUCCESSFULLY !!!!" << std::endl;
  
  return 0;
}

