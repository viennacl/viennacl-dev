/* =========================================================================
   Copyright (c) 2010-2016, Institute for Microelectronics,
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

/** \file tests/src/matrix_col_double.cpp  Tests routines for dense matrices, column-major, double precision.
*   \test Tests routines for dense matrices, column-major, double precision.
**/


#include "matrix_float_double.hpp"


int main (int, const char **)
{
  std::cout << std::endl;
  std::cout << "----------------------------------------------" << std::endl;
  std::cout << "----------------------------------------------" << std::endl;
  std::cout << "## Test :: Matrix operations, column-major, double precision " << std::endl;
  std::cout << "----------------------------------------------" << std::endl;
  std::cout << "----------------------------------------------" << std::endl;
  std::cout << std::endl;

#ifdef VIENNACL_WITH_OPENCL
   if ( viennacl::ocl::current_device().double_support() )
#endif
  {
    double epsilon = 1e-12;
    std::cout << "# Testing setup:" << std::endl;
    std::cout << "  eps:     " << epsilon << std::endl;
    std::cout << "  numeric: double" << std::endl;

    if (run_test<viennacl::column_major, double>(epsilon) != EXIT_SUCCESS)
      return EXIT_FAILURE;
  }

   std::cout << std::endl;
   std::cout << "------- Test completed --------" << std::endl;
   std::cout << std::endl;


  return EXIT_SUCCESS;
}

