/* =========================================================================
   Copyright (c) 2010-2015, Institute for Microelectronics,
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

/** \example custom-kernels.cpp
*
*   This tutorial explains how users can inject their own OpenCL compute kernels for use with ViennaCL objects
*
*   We start with including the necessary headers:
**/


// System headers
#include <iostream>
#include <string>

#ifndef VIENNACL_WITH_OPENCL
  #define VIENNACL_WITH_OPENCL
#endif

// ViennaCL headers
#include "viennacl/ocl/backend.hpp"
#include "viennacl/vector.hpp"
#include "viennacl/linalg/norm_2.hpp"


/**
* The next step is to define the custom compute kernels in a string.
* It is assumed that you are familiar with writing basic OpenCL kernels.
* If this is not the case, please have a look at one of the many OpenCL tutorials in the web.
*
* We define two custom compute kernels which compute an elementwise product and the element-wise division of two vectors. <br />
* Input: v1 ... vector <br />
*        v2 ... vector <br />
* Output: result ... vector <br />
*
* Algorithm: set result[i] <- v1[i] * v2[i] <br />
*            or  result[i] <- v1[i] / v2[i] <br />
*            (in MATLAB notation this is 'result = v1 .* v2' and 'result = v1 ./ v2');
**/
static const char * my_compute_program =
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

/**
*   Since no auxiliary routines are needed, we can directly start with main().
**/
int main()
{
  typedef float       ScalarType;

  /**
  * Initialize OpenCL vectors:
  **/
  unsigned int vector_size = 10;
  viennacl::vector<ScalarType>  vec1(vector_size);
  viennacl::vector<ScalarType>  vec2(vector_size);
  viennacl::vector<ScalarType>  result_mul(vector_size);
  viennacl::vector<ScalarType>  result_div(vector_size);

  /**
  * Fill the operands vec1 and vec2 with some numbers.
  **/
  for (unsigned int i=0; i<vector_size; ++i)
  {
    vec1[i] = static_cast<ScalarType>(i);
    vec2[i] = static_cast<ScalarType>(vector_size-i);
  }

  /**
  * Set up the OpenCL program given in my_compute_kernel:
  * A program is one compilation unit and can hold many different compute kernels.
  **/
  viennacl::ocl::program & my_prog = viennacl::ocl::current_context().add_program(my_compute_program, "my_compute_program");
  // Note: Releases older than ViennaCL 1.5.0 required calls to add_kernel(). This is no longer needed, the respective interface has been removed.

  /**
  * Now we can get the kernels from the program 'my_program'.
  * (Note that first all kernels need to be registered via add_kernel() before get_kernel() can be called,
  * otherwise existing references might be invalidated)
  **/
  viennacl::ocl::kernel & my_kernel_mul = my_prog.get_kernel("elementwise_prod");
  viennacl::ocl::kernel & my_kernel_div = my_prog.get_kernel("elementwise_div");

  /**
  * Launch the kernel with 'vector_size' threads in one work group
  * Note that std::size_t might differ between host and device. Thus, a cast to cl_uint is necessary for the forth argument.
  **/
  viennacl::ocl::enqueue(my_kernel_mul(vec1, vec2, result_mul, static_cast<cl_uint>(vec1.size())));
  viennacl::ocl::enqueue(my_kernel_div(vec1, vec2, result_div, static_cast<cl_uint>(vec1.size())));

  /**
  * Print the result:
  **/
  std::cout << "        vec1: " << vec1 << std::endl;
  std::cout << "        vec2: " << vec2 << std::endl;
  std::cout << "vec1 .* vec2: " << result_mul << std::endl;
  std::cout << "vec1 /* vec2: " << result_div << std::endl;
  std::cout << "norm_2(vec1 .* vec2): " << viennacl::linalg::norm_2(result_mul) << std::endl;
  std::cout << "norm_2(vec1 /* vec2): " << viennacl::linalg::norm_2(result_div) << std::endl;

  /**
  *  We are already done. We only needed a few lines of code by letting ViennaCL deal with the details :-)
  **/
  std::cout << "!!!! TUTORIAL COMPLETED SUCCESSFULLY !!!!" << std::endl;

  return EXIT_SUCCESS;
}

