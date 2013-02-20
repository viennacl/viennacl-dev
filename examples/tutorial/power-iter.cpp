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
*   Tutorial: Calculation of the eigenvalue with largest modulus using the power iteration method
*             (power-iter.cpp and power-iter.cu are identical, the latter being required for compilation using CUDA nvcc)
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


#include "viennacl/linalg/power_iter.hpp"
#include "viennacl/io/matrix_market.hpp"
// Some helper functions for this tutorial:
#include <iostream>
#include <fstream>
#include <limits>
#include <string>
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/matrix_proxy.hpp>
#include <boost/numeric/ublas/matrix_expression.hpp>
#include <boost/numeric/ublas/matrix_sparse.hpp>
#include <boost/numeric/ublas/vector.hpp>
#include <boost/numeric/ublas/operation.hpp> 
#include <boost/numeric/ublas/vector_expression.hpp>



int main()
{
  typedef double     ScalarType;
  
  boost::numeric::ublas::compressed_matrix<ScalarType> ublas_A;

  if (!viennacl::io::read_matrix_market_file(ublas_A, "../examples/testdata/mat65k.mtx"))
  {
    std::cout << "Error reading Matrix file" << std::endl;
    return 0;
  }
  
  viennacl::compressed_matrix<double>  vcl_A(ublas_A.size1(), ublas_A.size2());  
  viennacl::copy(ublas_A, vcl_A);
  
  viennacl::linalg::power_iter_tag ptag(1e-8);

  std::cout << "Starting computation of eigenvalue with largest modulus (might take about a minute)..." << std::endl;
  std::cout << "Result of power iteration with ublas matrix (single-threaded): " << viennacl::linalg::eig(ublas_A, ptag) << std::endl;
  std::cout << "Result of power iteration with ViennaCL (OpenCL accelerated): " << viennacl::linalg::eig(vcl_A, ptag) << std::endl;
  
}

