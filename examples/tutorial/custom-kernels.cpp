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
*   Tutorial:  Use user-provided OpenCL compute kernels with ViennaCL objects
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
#include "viennacl/linalg/norm_2.hpp"


// Some helper functions for this tutorial:
#include "Random.hpp"



//
// Custom compute kernels which compute an elementwise product/division of two vectors
// Input: v1 ... vector
//        v2 ... vector
// Output: result ... vector
//
// Algorithm: set result[i] <- v1[i] * v2[i]
//            or  result[i] <- v1[i] / v2[i]
//            (in MATLAB notation this is something like 'result = v1 .* v2' and 'result = v1 ./ v2');
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
"};\n\n"
"__kernel void elementwise_div(\n"
"          __global const float * vec1,\n"
"          __global const float * vec2, \n"
"          __global float * result,\n"
"          unsigned int size) \n"
"{ \n"
"  for (unsigned int i = get_global_id(0); i < size; i += get_global_size(0))\n"
"    result[i] = vec1[i] / vec2[i];\n"
"};\n";

int main()
{
  typedef float       ScalarType;

  //
  // Initialize OpenCL vectors:
  //
  unsigned int vector_size = 10;
  viennacl::scalar<ScalarType>  s = 1.0; //dummy
  viennacl::vector<ScalarType>  vec1(vector_size);
  viennacl::vector<ScalarType>  vec2(vector_size);
  viennacl::vector<ScalarType>  result_mul(vector_size);
  viennacl::vector<ScalarType>  result_div(vector_size);

  //
  // fill the operands vec1 and vec2:
  //
  for (unsigned int i=0; i<vector_size; ++i)
  {
    vec1[i] = static_cast<ScalarType>(i);
    vec2[i] = static_cast<ScalarType>(vector_size-i);
  }

  //
  // Set up the OpenCL program given in my_compute_kernel:
  // A program is one compilation unit and can hold many different compute kernels.
  //
  viennacl::ocl::program & my_prog = viennacl::ocl::current_context().add_program(my_compute_program, "my_compute_program");
  my_prog.add_kernel("elementwise_prod");  //register elementwise product kernel
  my_prog.add_kernel("elementwise_div");   //register elementwise division kernel
  
  //
  // Now we can get the kernels from the program 'my_program'.
  // (Note that first all kernels need to be registered via add_kernel() before get_kernel() can be called,
  // otherwise existing references might be invalidated)
  //
  viennacl::ocl::kernel & my_kernel_mul = my_prog.get_kernel("elementwise_prod");
  viennacl::ocl::kernel & my_kernel_div = my_prog.get_kernel("elementwise_div");
  
  //
  // Launch the kernel with 'vector_size' threads in one work group
  // Note that std::size_t might differ between host and device. Thus, a cast to cl_uint is necessary for the forth argument.
  //
  viennacl::ocl::enqueue(my_kernel_mul(vec1, vec2, result_mul, static_cast<cl_uint>(vec1.size())));  
  viennacl::ocl::enqueue(my_kernel_div(vec1, vec2, result_div, static_cast<cl_uint>(vec1.size())));
  
  //
  // Print the result:
  //
  std::cout << "        vec1: " << vec1 << std::endl;
  std::cout << "        vec2: " << vec2 << std::endl;
  std::cout << "vec1 .* vec2: " << result_mul << std::endl;
  std::cout << "vec1 /* vec2: " << result_div << std::endl;
  std::cout << "norm_2(vec1 .* vec2): " << viennacl::linalg::norm_2(result_mul) << std::endl;
  std::cout << "norm_2(vec1 /* vec2): " << viennacl::linalg::norm_2(result_div) << std::endl;
  
  //
  //  That's it.
  //
  std::cout << "!!!! TUTORIAL COMPLETED SUCCESSFULLY !!!!" << std::endl;
  
  return 0;
}

