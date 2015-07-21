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

/** \file tests/src/global_variables.cpp  Ensures that ViennaCL works properly when objects are used as global variables.
*   \test Ensures that ViennaCL works properly when objects are used as global variables.
**/

//
// *** System
//
#include <iostream>
#include <algorithm>
#include <cmath>

//
// *** ViennaCL
//
#include "viennacl/scalar.hpp"
#include "viennacl/vector.hpp"
#include "viennacl/matrix.hpp"
#include "viennacl/compressed_matrix.hpp"
#include "viennacl/coordinate_matrix.hpp"
#include "viennacl/ell_matrix.hpp"
#include "viennacl/hyb_matrix.hpp"
#ifdef VIENNACL_WITH_OPENCL
  #include "viennacl/circulant_matrix.hpp"
  #include "viennacl/hankel_matrix.hpp"
  #include "viennacl/toeplitz_matrix.hpp"
  #include "viennacl/vandermonde_matrix.hpp"
#endif


// forward declarations of global variables:
extern viennacl::scalar<float> s1;
extern viennacl::scalar<int>   s2;

extern viennacl::vector<float> v1;
extern viennacl::vector<int>   v2;

extern viennacl::matrix<float> m1;

// instantiation of global variables:
viennacl::scalar<float>  s1;
viennacl::scalar<int> s2;

viennacl::vector<float>  v1;
viennacl::vector<int> v2;

viennacl::matrix<float>  m1;
//viennacl::matrix<int> m2;

// TODO: Add checks for other types

//
// -------------------------------------------------------------
//
int main()
{
  std::cout << std::endl;
  std::cout << "----------------------------------------------" << std::endl;
  std::cout << "----------------------------------------------" << std::endl;
  std::cout << "## Test :: Instantiation of global variables" << std::endl;
  std::cout << "----------------------------------------------" << std::endl;
  std::cout << "----------------------------------------------" << std::endl;
  std::cout << std::endl;

  s1 = viennacl::scalar<float>(1.0f);
  s2 = viennacl::scalar<int>(1);

  v1 = viennacl::vector<float>(5);
  v2 = viennacl::vector<int>(5);

  m1 = viennacl::matrix<float>(5, 4);
  //m2 = viennacl::matrix<int>(5, 4);

  std::cout << std::endl;
  std::cout << "------- Test completed --------" << std::endl;
  std::cout << std::endl;


  return EXIT_SUCCESS;
}
//
// -------------------------------------------------------------
//

