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

/** \example wrap-host-buffer.cpp
*
*   This tutorial shows how ViennaCL can be used to wrap user-provided memory buffers allocated on the host.
*   The benefit of such a wrapper is that the algorithms in ViennaCL can directly be run without pre- or postprocessing the data.
*
*   We start with including the required headers:
**/

// System headers
#include <iostream>
#include <cstdlib>
#include <string>

// ViennaCL headers
#include "viennacl/vector.hpp"

/**
*   We setup two STL-vectors and add them up as a reference.
*   Then the buffer get wrapped by ViennaCL and added up.
**/
int main()
{
  typedef float       ScalarType;

  /**
  *  <h2>Part 1: Allocate some buffers on the host</h2>
  **/
  std::size_t size = 10;

  std::vector<ScalarType> host_x(size, 1.0);
  std::vector<ScalarType> host_y(size, 2.0);

  std::cout << "Result on host: ";
  for (std::size_t i=0; i<size; ++i)
    std::cout << host_x[i] + host_y[i] << " ";
  std::cout << std::endl;

  /**
  *   <h2>Part 2: Now do the same computations within ViennaCL</h2>
  **/

  // wrap host buffer within ViennaCL vectors:
  viennacl::vector<ScalarType> vcl_vec1(&(host_x[0]), viennacl::MAIN_MEMORY, size); // Second parameter specifies that this is host memory rather than CUDA memory
  viennacl::vector<ScalarType> vcl_vec2(&(host_y[0]), viennacl::MAIN_MEMORY, size); // Second parameter specifies that this is host memory rather than CUDA memory

  vcl_vec1 += vcl_vec2;

  std::cout << "Result with ViennaCL: " << vcl_vec1 << std::endl;

  std::cout << "Data in STL-vector: ";
  for (std::size_t i=0; i<size; ++i)
    std::cout << host_x[i] << " ";
  std::cout << std::endl;

  /**
  *  That's it. Print success message and exit.
  **/
  std::cout << "!!!! TUTORIAL COMPLETED SUCCESSFULLY !!!!" << std::endl;

  return EXIT_SUCCESS;
}

