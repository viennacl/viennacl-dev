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

/** \example iterative-armadillo.cpp
*
*   The following tutorial shows how to use the iterative solvers in ViennaCL with objects from the <a href="http://eigen.tuxfamily.org/">Eigen Library</a> directly.
*
*   \note Eigen provides its own iterative solvers in the meanwhile. Check these first.
*
*   We begin with including the necessary headers:
**/

// System headers
#include <iostream>

#ifndef NDEBUG
 #define NDEBUG
#endif

// Armadillo headers (disable BLAS and LAPACK to avoid linking issues)
#define ARMA_DONT_USE_BLAS
#define ARMA_DONT_USE_LAPACK
#include <armadillo>

// IMPORTANT: Must be set prior to any ViennaCL includes if you want to use ViennaCL algorithms on Armadillo objects
#define VIENNACL_WITH_ARMADILLO 1

// ViennaCL headers
#include "viennacl/linalg/cg.hpp"
#include "viennacl/linalg/bicgstab.hpp"
#include "viennacl/linalg/gmres.hpp"
#include "viennacl/io/matrix_market.hpp"


// Some helper functions for this tutorial:
#include "vector-io.hpp"

/**
*  In the following we run the CG method, the BiCGStab method, and the GMRES method with Armadillo types directly.
*  First, the matrices are set up, then the respective solvers are called.
**/
int main(int, char *[])
{
  typedef float ScalarType;

  /**
  * Read system from file. This is a little tricky, since Armadillo does not provide a fast enough element-insertion.
  * Therefore, we read the matrix market file to an STL-matrix and then pass the data on when creating the Armadillo sparse matrix object.
  **/
  std::vector<std::map<unsigned int, ScalarType> > stl_matrix;
  std::cout << "Reading matrix (this might take some time)..." << std::endl;
  if (!viennacl::io::read_matrix_market_file(stl_matrix, "../examples/testdata/mat65k.mtx"))
  {
    std::cout << "Error reading Matrix file. Make sure you run from the build/-folder." << std::endl;
    return EXIT_FAILURE;
  }

  // Copy over to Armadillo sparse matrix by putting the indices into a matrix and the values into a vector:
  std::size_t num_nnz = 0;
  for (std::size_t i=0; i<stl_matrix.size(); ++i)
    num_nnz += stl_matrix[i].size();

  arma::Mat<arma::uword>  arma_indices(2, num_nnz);
  arma::Col<ScalarType>   arma_values(num_nnz);

  std::size_t index = 0;
  for (std::size_t i=0; i<stl_matrix.size(); ++i)
  {
    for (std::map<unsigned int, ScalarType>::const_iterator it = stl_matrix[i].begin(); it != stl_matrix[i].end(); ++it)
    {
      arma_indices(0, index) = i;
      arma_indices(1, index) = it->first;
      arma_values(index) = it->second;
      ++index;
    }
  }
  std::cout << "Done: reading matrix" << std::endl;



  /**
  * Initialize Armadillo types for iterative solvers
  **/
  arma::SpMat<ScalarType> arma_matrix(arma_indices, arma_values, 65025, 65025);
  arma::Col<ScalarType>   arma_rhs;
  arma::Col<ScalarType>   arma_result;
  arma::Col<ScalarType>   residual;

  /**
   * Read the right hand side as well as the result vector from files:
   **/
  if (!readVectorFromFile("../examples/testdata/rhs65025.txt", arma_rhs))
  {
    std::cout << "Error reading RHS file" << std::endl;
    return EXIT_FAILURE;
  }

  if (!readVectorFromFile("../examples/testdata/result65025.txt", arma_result))
  {
    std::cout << "Error reading Result file" << std::endl;
    return EXIT_FAILURE;
  }

  /**
  *  Conjugate Gradient (CG) solver:
  **/
  std::cout << "----- Running CG -----" << std::endl;
  arma_result = viennacl::linalg::solve(arma_matrix, arma_rhs, viennacl::linalg::cg_tag());

  residual = arma_matrix * arma_result - arma_rhs;
  std::cout << "Relative residual: " << viennacl::linalg::norm_2(residual) / viennacl::linalg::norm_2(arma_rhs) << std::endl;

  /**
  *  Stabilized Bi-Conjugate Gradient (BiCGStab) solver:
  **/
  std::cout << "----- Running BiCGStab -----" << std::endl;
  arma_result = viennacl::linalg::solve(arma_matrix, arma_rhs, viennacl::linalg::bicgstab_tag());

  residual = arma_matrix * arma_result - arma_rhs;
  std::cout << "Relative residual: " << viennacl::linalg::norm_2(residual) / viennacl::linalg::norm_2(arma_rhs) << std::endl;

  /**
  *  Generalized Minimum Residual (GMRES) solver:
  **/
  std::cout << "----- Running GMRES -----" << std::endl;
  arma_result = viennacl::linalg::solve(arma_matrix, arma_rhs, viennacl::linalg::gmres_tag());

  residual = arma_matrix * arma_result - arma_rhs;
  std::cout << "Relative residual: " << viennacl::linalg::norm_2(residual) / viennacl::linalg::norm_2(arma_rhs) << std::endl;

  /**
  *   That's it. Print a success message and exit.
  **/
  std::cout << std::endl;
  std::cout << "!!!! TUTORIAL COMPLETED SUCCESSFULLY !!!!" << std::endl;
  std::cout << std::endl;
}

