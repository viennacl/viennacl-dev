/* =========================================================================
 Copyright (c) 2010-2014, Institute for Microelectronics,
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
 *   Tutorial: NMF functionality
 *
 */

//include header file for NMF computation
#include "viennacl/linalg/nmf_operations.hpp"

typedef float ScalarType;
template<typename MAJOR>
void fill_random(viennacl::matrix<ScalarType, MAJOR>& v)
{
  for (std::size_t i = 0; i < v.size1(); i++)
  {
    for (std::size_t j = 0; j < v.size2(); ++j)
      v(i, j) = static_cast<ScalarType>(rand()) / RAND_MAX;
  }
}

int main()
{
  std::cout << std::endl;
  std::cout << "------- Tutorial NMF --------" << std::endl;
  std::cout << std::endl;

  unsigned int m = 3; //size1 of W and size1 of V
  unsigned int n = 3; //size2 of V and size2 of H
  unsigned int k = 3; //size2 of W and sze 1 of H

  //create V,W,H matrix where the result will be computed V.size1() == W.size1() && V.size2() == H.size2() ==> m == m && n = n
  viennacl::matrix<ScalarType, viennacl::column_major> V(m, n);
  viennacl::matrix<ScalarType, viennacl::column_major> W(m, k);
  viennacl::matrix<ScalarType, viennacl::column_major> H(k, n);

  //fill the matrix randomly the can not be zero!NON-NEGATIV
  fill_random(V);
  fill_random(W);
  fill_random(H);

  std::cout << "INPUT MATRIX:" << std::endl;
  std::cout << "V" << V << std::endl;
  std::cout << "W" << W << std::endl;
  std::cout << "H" << H << "\n" << std::endl;

  //create configuration class this hold all neceserry informations about computing
  viennacl::linalg::nmf_config conf;
  conf.print_relative_error(false);
  conf.max_iterations(5000); //5000 iterations are enough for the test

  //call the NMF rutine
  std::cout << "Computing NMF" << std::endl;
  viennacl::linalg::nmf(V, W, H, conf);

  std::cout << "RESULT:" << std::endl;
  std::cout << "V" << V << std::endl;
  std::cout << "W" << W << std::endl;
  std::cout << "H" << H << "\n" << std::endl;

  std::cout << "RESULT AFTER MULTIPLIKATION W*H:" << std::endl;
  viennacl::matrix<ScalarType> resultCorrect = viennacl::linalg::prod(W, H);
  std::cout << resultCorrect << std::endl;

  std::cout << std::endl;
  std::cout << "------- Tutorial completed --------" << std::endl;
  std::cout << std::endl;

}
