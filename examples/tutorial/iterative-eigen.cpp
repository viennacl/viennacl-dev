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

/** \example iterative-eigen.cpp
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


// Eigen headers
#include <Eigen/Core>
#include <Eigen/Sparse>

// Must be set prior to any ViennaCL includes if you want to use ViennaCL algorithms on Eigen objects
#define VIENNACL_WITH_EIGEN 1

// ViennaCL headers
#include "viennacl/linalg/ilu.hpp"
#include "viennacl/linalg/cg.hpp"
#include "viennacl/linalg/bicgstab.hpp"
#include "viennacl/linalg/gmres.hpp"
#include "viennacl/io/matrix_market.hpp"


// Some helper functions for this tutorial:
#include "vector-io.hpp"

/**
*  In the following we run the CG method, the BiCGStab method, and the GMRES method with Eigen types directly.
*  First, the matrices are set up, then the respective solvers are called.
**/
int main(int, char *[])
{
  typedef float ScalarType;

  Eigen::SparseMatrix<ScalarType, Eigen::RowMajor> eigen_matrix(65025, 65025);
  Eigen::VectorXf eigen_rhs;
  Eigen::VectorXf eigen_result;
  Eigen::VectorXf ref_result;
  Eigen::VectorXf residual;

  /**
  * Read system from file
  **/
  std::cout << "Reading matrix (this might take some time)..." << std::endl;
  eigen_matrix.reserve(65025 * 7);
  if (!viennacl::io::read_matrix_market_file(eigen_matrix, "../examples/testdata/mat65k.mtx"))
  {
    std::cout << "Error reading Matrix file. Make sure you run from the build/-folder." << std::endl;
    return EXIT_FAILURE;
  }
  //eigen_matrix.endFill();
  std::cout << "Done: reading matrix" << std::endl;

  if (!readVectorFromFile("../examples/testdata/rhs65025.txt", eigen_rhs))
  {
    std::cout << "Error reading RHS file" << std::endl;
    return EXIT_FAILURE;
  }

  if (!readVectorFromFile("../examples/testdata/result65025.txt", ref_result))
  {
    std::cout << "Error reading Result file" << std::endl;
    return EXIT_FAILURE;
  }

  /**
  *  Conjugate Gradient (CG) solver:
  **/
  std::cout << "----- Running CG -----" << std::endl;
  eigen_result = viennacl::linalg::solve(eigen_matrix, eigen_rhs, viennacl::linalg::cg_tag());

  residual = eigen_matrix * eigen_result - eigen_rhs;
  std::cout << "Relative residual: " << viennacl::linalg::norm_2(residual) / viennacl::linalg::norm_2(eigen_rhs) << std::endl;

  /**
  *  Stabilized Bi-Conjugate Gradient (BiCGStab) solver:
  **/
  std::cout << "----- Running BiCGStab -----" << std::endl;
  eigen_result = viennacl::linalg::solve(eigen_matrix, eigen_rhs, viennacl::linalg::bicgstab_tag());

  residual = eigen_matrix * eigen_result - eigen_rhs;
  std::cout << "Relative residual: " << viennacl::linalg::norm_2(residual) / viennacl::linalg::norm_2(eigen_rhs) << std::endl;

  /**
  *  Generalized Minimum Residual (GMRES) solver:
  **/
  std::cout << "----- Running GMRES -----" << std::endl;
  eigen_result = viennacl::linalg::solve(eigen_matrix, eigen_rhs, viennacl::linalg::gmres_tag());

  residual = eigen_matrix * eigen_result - eigen_rhs;
  std::cout << "Relative residual: " << viennacl::linalg::norm_2(residual) / viennacl::linalg::norm_2(eigen_rhs) << std::endl;

  /**
  *   That's it. Print a success message and exit.
  **/
  std::cout << std::endl;
  std::cout << "!!!! TUTORIAL COMPLETED SUCCESSFULLY !!!!" << std::endl;
  std::cout << std::endl;
}

