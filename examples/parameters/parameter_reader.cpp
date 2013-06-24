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

//#define VIENNACL_DEBUG_ALL
//#define NDEBUG

#ifndef VIENNACL_WITH_OPENCL
  #define VIENNACL_WITH_OPENCL
#endif

#include "viennacl/scalar.hpp"
#include "viennacl/vector.hpp"
#include "viennacl/matrix.hpp"
#include "viennacl/compressed_matrix.hpp"
#include "viennacl/io/kernel_parameters.hpp"


#include <iostream>
#include <vector>





int main(int, char **)
{
  // -----------------------------------------
  std::cout << std::endl;
  std::cout << "----------------------------------------------" << std::endl;
  std::cout << "               Device Info" << std::endl;
  std::cout << "----------------------------------------------" << std::endl;

  std::cout << viennacl::ocl::current_device().info() << std::endl;

  viennacl::io::read_kernel_parameters< viennacl::vector<float> >("vector_parameters.xml");
  viennacl::io::read_kernel_parameters< viennacl::matrix<float> >("matrix_parameters.xml");
  viennacl::io::read_kernel_parameters< viennacl::compressed_matrix<float> >("sparse_parameters.xml");
  // -----------------------------------------

  //check:
  std::cout << "vector add:" << std::endl;
  std::cout << viennacl::ocl::get_kernel("f_vector_1", "add").local_work_size() << std::endl;
  std::cout << viennacl::ocl::get_kernel("f_vector_1", "add").global_work_size() << std::endl;

  std::cout << "matrix vec_mul:" << std::endl;
  std::cout << viennacl::ocl::get_kernel("f_matrix_row_1", "vec_mul").local_work_size() << std::endl;
  std::cout << viennacl::ocl::get_kernel("f_matrix_row_1", "vec_mul").global_work_size() << std::endl;

  std::cout << "compressed_matrix vec_mul:" << std::endl;
  std::cout << viennacl::ocl::get_kernel("f_compressed_matrix_1", "vec_mul").local_work_size() << std::endl;
  std::cout << viennacl::ocl::get_kernel("f_compressed_matrix_1", "vec_mul").global_work_size() << std::endl;


  return 0;
}

