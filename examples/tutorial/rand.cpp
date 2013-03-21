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
*   Tutorial: Dumps random values into the supplied vector/matrix
*
*
*/

#define VIENNACL_DEBUG_BUILD

// include necessary system headers
#include <iostream>
#include "viennacl/rand/gaussian.hpp"
#include "viennacl/rand/uniform.hpp"
#include "viennacl/matrix.hpp"


int main(){

  typedef float NumericT;

  static const unsigned int size1 = 8;
  static const unsigned int size2 = 9;

  static const float sigma = 0.8;
  static const float mu = 1.4;

  static const float a = 0;
  static const float b = 4;

  //Dumps size1xsize2 observations of a N(mu,sigma) random variable.
  viennacl::matrix<NumericT> mat = viennacl::random_matrix<NumericT>(size1, size2, viennacl::rand::gaussian_tag(mu,sigma));
  std::cout << "------------------" << std::endl;
  std::cout << "Dump Gaussian(" << mu << "," << sigma << ") Observations into matrix : " << std::endl;
  std::cout << "------------------" << std::endl;
  std::cout << mat << std::endl;

  std::cout << std::endl;

  //Dumps size1 observations of a U(a,b) random variable.
  viennacl::vector<NumericT> vec = viennacl::random_vector<NumericT>(size1,viennacl::rand::uniform_tag(a,b));
  std::cout << "------------------" << std::endl;
  std::cout << "Uniform(" << a << "," << b << ") Observations into vector : " << std::endl;
  std::cout << "------------------" << std::endl;
  std::cout << vec << std::endl;

  //
  //  That's it.
  //
  std::cout << std::endl;
  std::cout << "!!!! TUTORIAL COMPLETED SUCCESSFULLY !!!!" << std::endl;


}
