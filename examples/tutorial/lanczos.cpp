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
*   Tutorial: Calculation of eigenvalues using Lanczos' method (lanczos.cpp and lanczos.cu are identical, the latter being required for compilation using CUDA nvcc)
*
*/

// include necessary system headers
#include <iostream>

#ifndef NDEBUG
  #define NDEBUG
#endif

#define VIENNACL_WITH_UBLAS

//include basic scalar and vector types of ViennaCL
#include "viennacl/scalar.hpp"
#include "viennacl/vector.hpp"
#include "viennacl/compressed_matrix.hpp"


#include "viennacl/linalg/lanczos.hpp"
#include "viennacl/io/matrix_market.hpp"
// Some helper functions for this tutorial:
#include <iostream>
#include <fstream>
#include <limits>
#include <string>
#include <iomanip>
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/matrix_proxy.hpp>
#include <boost/numeric/ublas/matrix_expression.hpp>
#include <boost/numeric/ublas/matrix_sparse.hpp>
#include <boost/numeric/ublas/vector.hpp>
#include <boost/numeric/ublas/operation.hpp>
#include <boost/numeric/ublas/vector_expression.hpp>



template <typename MatrixType>
std::vector<double> initEig(MatrixType const & A)
{
  viennacl::linalg::lanczos_tag ltag(0.75, 10, viennacl::linalg::lanczos_tag::partial_reorthogonalization, 1700);
  std::vector<double> lanczos_eigenvalues = viennacl::linalg::eig(A, ltag);
  for(std::size_t i = 0; i< lanczos_eigenvalues.size(); i++){
          std::cout << "Eigenvalue " << i+1 << ": " << std::setprecision(10) << lanczos_eigenvalues[i] << std::endl;
  }

  return lanczos_eigenvalues;
}


int main()
{
  typedef double     ScalarType;

  boost::numeric::ublas::compressed_matrix<ScalarType> ublas_A;

  if (!viennacl::io::read_matrix_market_file(ublas_A, "../examples/testdata/mat65k.mtx"))
  {
    std::cout << "Error reading Matrix file" << std::endl;
    return 0;
  }

  std::cout << "Running Lanczos algorithm (this might take a while)..." << std::endl;
  std::vector<double> eigenvalues = initEig(ublas_A);
}

