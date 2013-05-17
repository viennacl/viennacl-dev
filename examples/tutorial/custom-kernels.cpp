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
"#if defined(cl_khr_fp64)\n"
"#  pragma OPENCL EXTENSION cl_khr_fp64: enable\n"
"#elif defined(cl_amd_fp64)\n"
"#  pragma OPENCL EXTENSION cl_amd_fp64: enable\n"
"#endif\n"
"__attribute__((reqd_work_group_size(64,64,1)))\n"
"__kernel void _k0(__global float* arg0\n"
", unsigned int arg0internal_size1_\n"
", unsigned int arg0internal_size2_\n"
",__global float* arg1\n"
", unsigned int arg1internal_size1_\n"
", unsigned int arg1internal_size2_\n"
",__global float* arg2\n"
", unsigned int arg2internal_size1_\n"
", unsigned int arg2internal_size2_\n"
")\n"
"{\n"
"        float prod_val_0_0_0 = (float)(0) ;\n"
"        float prod_val_0_0_1 = (float)(0) ;\n"
"        float prod_val_0_1_0 = (float)(0) ;\n"
"        float prod_val_0_1_1 = (float)(0) ;\n"
"        unsigned int block_num = arg1internal_size2_/16;\n"
"        __global float* res_ptr = arg0 + (get_global_id(0)*2)*arg0internal_size2_+ (get_global_id(1)*2);\n"
"        __global float * arg2_ptr_0 = arg2 + (0)*arg2internal_size2_+ (get_local_id(1)*2 +  get_group_id(1)*256);\n"
"        __global float * arg2_ptr_1 = arg2 + (1)*arg2internal_size2_+ (get_local_id(1)*2 +  get_group_id(1)*256);\n"
"        __global float * arg1_ptr_0 = arg1 + (get_group_id(0)*256+get_local_id(0)*2+0)*arg1internal_size2_+ (0);\n"
"        __global float * arg1_ptr_1 = arg1 + (get_group_id(0)*256+get_local_id(0)*2+1)*arg1internal_size2_+ (0);\n"
"        for(unsigned int bl=0 ; bl<block_num ; ++bl){\n"
"                 for(unsigned int bs=0 ; bs < 8 ; ++bs){\n"
"                        float val_rhs_0_0 = *arg2_ptr_0;++arg2_ptr_0;\n"
"                        float val_rhs_0_1 = *arg2_ptr_0;++arg2_ptr_0;\n"
"                        float val_rhs_1_0 = *arg2_ptr_1;++arg2_ptr_1;\n"
"                        float val_rhs_1_1 = *arg2_ptr_1;++arg2_ptr_1;\n"
"                        float val_lhs_0_0 = *arg1_ptr_0;++arg1_ptr_0;\n"
"                        float val_lhs_1_0 = *arg1_ptr_1;++arg1_ptr_1;\n"
"                        float val_lhs_0_1 = *arg1_ptr_0;++arg1_ptr_0;\n"
"                        float val_lhs_1_1 = *arg1_ptr_1;++arg1_ptr_1;\n"
"                        prod_val_0_0_0 = prod_val_0_0_0+val_lhs_0_0*val_rhs_0_0;\n"
"                        prod_val_0_1_0 = prod_val_0_1_0+val_lhs_1_0*val_rhs_0_0;\n"
"                        prod_val_0_0_1 = prod_val_0_0_1+val_lhs_0_0*val_rhs_0_1;\n"
"                        prod_val_0_1_1 = prod_val_0_1_1+val_lhs_1_0*val_rhs_0_1;\n"
"                        prod_val_0_0_0 = prod_val_0_0_0+val_lhs_0_1*val_rhs_1_0;\n"
"                        prod_val_0_1_0 = prod_val_0_1_0+val_lhs_1_1*val_rhs_1_0;\n"
"                        prod_val_0_0_1 = prod_val_0_0_1+val_lhs_0_1*val_rhs_1_1;\n"
"                        prod_val_0_1_1 = prod_val_0_1_1+val_lhs_1_1*val_rhs_1_1;\n"
"                        arg2_ptr_0 += 2*arg2internal_size2_ - 2;\n"
"                        arg2_ptr_1 += 2*arg2internal_size2_ - 2;\n"
"                }\n"
"        }\n"
"        *res_ptr++=prod_val_0_0_0;\n"
"        *res_ptr++=prod_val_0_0_1;\n"
"        res_ptr+=arg0internal_size2_ - 2;\n"
"        *res_ptr++=prod_val_0_1_0;\n"
"        *res_ptr++=prod_val_0_1_1;\n"
"}\n";

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
  //my_prog.add_kernel("elementwise_prod");  //register elementwise product kernel
  //my_prog.add_kernel("elementwise_div");   //register elementwise division kernel
  
  ////
  //// Now we can get the kernels from the program 'my_program'.
  //// (Note that first all kernels need to be registered via add_kernel() before get_kernel() can be called,
  //// otherwise existing references might be invalidated)
  ////
  //viennacl::ocl::kernel & my_kernel_mul = my_prog.get_kernel("elementwise_prod");
  //viennacl::ocl::kernel & my_kernel_div = my_prog.get_kernel("elementwise_div");
  
  ////
  //// Launch the kernel with 'vector_size' threads in one work group
  //// Note that std::size_t might differ between host and device. Thus, a cast to cl_uint is necessary for the forth argument.
  ////
  //viennacl::ocl::enqueue(my_kernel_mul(vec1, vec2, result_mul, static_cast<cl_uint>(vec1.size())));  
  //viennacl::ocl::enqueue(my_kernel_div(vec1, vec2, result_div, static_cast<cl_uint>(vec1.size())));
  
  ////
  //// Print the result:
  ////
  //std::cout << "        vec1: " << vec1 << std::endl;
  //std::cout << "        vec2: " << vec2 << std::endl;
  //std::cout << "vec1 .* vec2: " << result_mul << std::endl;
  //std::cout << "vec1 /* vec2: " << result_div << std::endl;
  //std::cout << "norm_2(vec1 .* vec2): " << viennacl::linalg::norm_2(result_mul) << std::endl;
  //std::cout << "norm_2(vec1 /* vec2): " << viennacl::linalg::norm_2(result_div) << std::endl;
  
  //
  //  That's it.
  //
  std::cout << "!!!! TUTORIAL COMPLETED SUCCESSFULLY !!!!" << std::endl;
  
  return 0;
}

