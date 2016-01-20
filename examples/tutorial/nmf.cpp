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

/** \example nmf.cpp
*
*   This tutorial explains how to use the nonnegative matrix factorization (NMF) functionality in ViennaCL.
*
*   The first step is to include the necessary headers and define the floating point type used for the computations:
**/


#include "viennacl/matrix.hpp"
#include "viennacl/linalg/nmf.hpp"

// feel free to change this to 'double' if supported by your hardware.
typedef float ScalarType;

/**
*   The following routine fills a matrix with uniformly distributed random values from the interval [0,1].
*   This is used to initialize the matrices to be factored
**/
template<typename MajorT>
void fill_random(viennacl::matrix<ScalarType, MajorT> & A)
{
  for (std::size_t i = 0; i < A.size1(); i++)
    for (std::size_t j = 0; j < A.size2(); ++j)
      A(i, j) = static_cast<ScalarType>(rand()) / ScalarType(RAND_MAX);
}

/**
*  In the main routine we set up a matrix V and compute approximate factors W and H such that \f$V \approx W H \f$, where all three matrices carry nonnegative entries only.
*  Since W and H are usually chosen to represent a rank-k-approximation of V, we use a similar low-rank approximation here.
**/
int main()
{
  std::cout << std::endl;
  std::cout << "------- Tutorial NMF --------" << std::endl;
  std::cout << std::endl;

  /**
  *   Approximate the 7-by-6-matrix V by a 7-by-3-matrix W and a 3-by-6-matrix H
  **/
  unsigned int m = 7; //size1 of W and size1 of V
  unsigned int n = 6; //size2 of V and size2 of H
  unsigned int k = 3; //size2 of W and size1 of H

  viennacl::matrix<ScalarType, viennacl::column_major> V(m, n);
  viennacl::matrix<ScalarType, viennacl::column_major> W(m, k);
  viennacl::matrix<ScalarType, viennacl::column_major> H(k, n);

  /**
  *  Fill the matrices randomly. Initial guesses for W and H consisting of only zeros won't work.
  **/
  fill_random(V);
  fill_random(W);
  fill_random(H);

  std::cout << "Input matrices:" << std::endl;
  std::cout << "V" << V << std::endl;
  std::cout << "W" << W << std::endl;
  std::cout << "H" << H << "\n" << std::endl;

  /**
  *  Create configuration object to hold (and adjust) the respective parameters.
  **/
  viennacl::linalg::nmf_config conf;
  conf.print_relative_error(false);
  conf.max_iterations(50); // 50 iterations are enough here

  /**
  *  Call the NMF routine and print the results
  **/
  std::cout << "Computing NMF" << std::endl;
  viennacl::linalg::nmf(V, W, H, conf);

  std::cout << "RESULT:" << std::endl;
  std::cout << "V" << V << std::endl;
  std::cout << "W" << W << std::endl;
  std::cout << "H" << H << "\n" << std::endl;

  /**
  *   Print the product W*H approximating V for comparison and exit:
  **/
  std::cout << "W*H:" << std::endl;
  viennacl::matrix<ScalarType> resultCorrect = viennacl::linalg::prod(W, H);
  std::cout << resultCorrect << std::endl;

  std::cout << std::endl;
  std::cout << "------- Tutorial completed --------" << std::endl;
  std::cout << std::endl;

}
