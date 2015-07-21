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

/** \example iterative-mtl4.cpp
*
*   The following tutorial shows how to use the iterative solvers in ViennaCL with objects from the <a href="http://www.mtl4.org/">MTL4 Library</a> directly.
*
*   \note MTL4 provides iterative solvers through the ITK library. You might also want to check these.
*
*   We begin with including the necessary headers:
**/

// include necessary system headers
#include <iostream>

// MTL4 headers
#include <boost/numeric/mtl/mtl.hpp>
#include <boost/numeric/itl/itl.hpp>

// Must be set prior to any ViennaCL includes if you want to use ViennaCL algorithms on Eigen objects
#define VIENNACL_WITH_MTL4 1

// ViennaCL includes
#include "viennacl/linalg/ilu.hpp"
#include "viennacl/linalg/cg.hpp"
#include "viennacl/linalg/bicgstab.hpp"
#include "viennacl/linalg/gmres.hpp"
#include "viennacl/io/matrix_market.hpp"


// Some helper functions for this tutorial:
#include "vector-io.hpp"

/**
*  In the following we run the CG method, the BiCGStab method, and the GMRES method with MTL4 types directly.
*  First, the matrices are set up, then the respective solvers are called.
**/
int main(int, char *[])
{
  typedef double    ScalarType;

  mtl::compressed2D<ScalarType> mtl4_matrix;
  mtl4_matrix.change_dim(65025, 65025);
  set_to_zero(mtl4_matrix);

  mtl::dense_vector<ScalarType> mtl4_rhs(65025, 1.0);
  mtl::dense_vector<ScalarType> mtl4_result(65025, 0.0);
  mtl::dense_vector<ScalarType> mtl4_residual(65025, 0.0);

  /**
  * Read system from file
  **/

  mtl::io::matrix_market_istream("../examples/testdata/mat65k.mtx") >> mtl4_matrix;

  /**
  *  Conjugate Gradient (CG) solver:
  **/
  std::cout << "----- Running CG -----" << std::endl;
  mtl4_result = viennacl::linalg::solve(mtl4_matrix, mtl4_rhs, viennacl::linalg::cg_tag());

  mtl4_residual = mtl4_matrix * mtl4_result - mtl4_rhs;
  std::cout << "Relative residual: " << viennacl::linalg::norm_2(mtl4_residual) / viennacl::linalg::norm_2(mtl4_rhs) << std::endl;

  /**
  *  Stabilized Bi-Conjugate Gradient (BiCGStab) solver:
  **/
  std::cout << "----- Running BiCGStab -----" << std::endl;
  mtl4_result = viennacl::linalg::solve(mtl4_matrix, mtl4_rhs, viennacl::linalg::bicgstab_tag());

  mtl4_residual = mtl4_matrix * mtl4_result - mtl4_rhs;
  std::cout << "Relative residual: " << viennacl::linalg::norm_2(mtl4_residual) / viennacl::linalg::norm_2(mtl4_rhs) << std::endl;

  /**
  *  Generalized Minimum Residual (GMRES) solver:
  **/
  std::cout << "----- Running GMRES -----" << std::endl;
  mtl4_result = viennacl::linalg::solve(mtl4_matrix, mtl4_rhs, viennacl::linalg::gmres_tag());

  mtl4_residual = mtl4_matrix * mtl4_result - mtl4_rhs;
  std::cout << "Relative residual: " << viennacl::linalg::norm_2(mtl4_residual) / viennacl::linalg::norm_2(mtl4_rhs) << std::endl;

  /**
  *   That's it. Print a success message and exit.
  **/
  std::cout << std::endl;
  std::cout << "!!!! TUTORIAL COMPLETED SUCCESSFULLY !!!!" << std::endl;
  std::cout << std::endl;
}

