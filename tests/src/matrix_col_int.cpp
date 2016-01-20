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

/** \file tests/src/matrix_col_int.cpp  Tests routines for dense matrices, column-major, signed integers.
*   \test Tests routines for dense matrices, column-major, signed integers.
**/

#include "matrix_int.hpp"

int main (int, const char **)
{
  std::cout << std::endl;
  std::cout << "----------------------------------------------" << std::endl;
  std::cout << "----------------------------------------------" << std::endl;
  std::cout << "## Test :: Matrix operations, column-major, integers " << std::endl;
  std::cout << "----------------------------------------------" << std::endl;
  std::cout << "----------------------------------------------" << std::endl;
  std::cout << std::endl;

  std::cout << "# Testing setup:" << std::endl;
  std::cout << "  numeric: int" << std::endl;
  std::cout << " --- column-major ---" << std::endl;
  if (run_test<viennacl::column_major, int>() != EXIT_SUCCESS)
    return EXIT_FAILURE;

  std::cout << "# Testing setup:" << std::endl;
  std::cout << "  numeric: long" << std::endl;
  std::cout << " --- column-major ---" << std::endl;
  if (run_test<viennacl::column_major, long>() != EXIT_SUCCESS)
    return EXIT_FAILURE;

  std::cout << std::endl;
  std::cout << "------- Test completed --------" << std::endl;
  std::cout << std::endl;

  return EXIT_SUCCESS;
}

