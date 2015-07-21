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

/** \example iterative-ublas.cpp
*
*   In this example we show how one can use the iterative solvers in ViennaCL with objects from Boost.uBLAS.
*
*   The first step is to include the necessary headers:
**/

//
// include necessary system headers
//
#include <iostream>

//
// Necessary to obtain a suitable performance in ublas
#ifndef NDEBUG
 #define BOOST_UBLAS_NDEBUG
#endif


//
// ublas includes
//
#include <boost/numeric/ublas/io.hpp>
#include <boost/numeric/ublas/triangular.hpp>
#include <boost/numeric/ublas/matrix_sparse.hpp>
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/matrix_proxy.hpp>
#include <boost/numeric/ublas/operation.hpp>
#include <boost/numeric/ublas/operation_sparse.hpp>
#include <boost/numeric/ublas/io.hpp>
#include <boost/numeric/ublas/lu.hpp>

// Must be set if you want to use ViennaCL algorithms on ublas objects
#define VIENNACL_WITH_UBLAS 1

//
// ViennaCL includes
//
#include "viennacl/linalg/ilu.hpp"
#include "viennacl/linalg/cg.hpp"
#include "viennacl/linalg/bicgstab.hpp"
#include "viennacl/linalg/gmres.hpp"
#include "viennacl/io/matrix_market.hpp"

// Some helper functions for this tutorial:
#include "vector-io.hpp"

/**
*  A shortcut for convience (use `ublas::` instead of `boost::numeric::ublas::`)
**/
using namespace boost::numeric;

/**
*  In the main() routine we create matrices and vectors using Boost.uBLAS types, fill them with data and launch the iterative solvers.
**/
int main()
{
  typedef float       ScalarType;

  //
  // Set up some ublas objects
  //
  ublas::vector<ScalarType> rhs;
  ublas::vector<ScalarType> ref_result;
  ublas::vector<ScalarType> result;
  ublas::compressed_matrix<ScalarType> ublas_matrix;

  /**
  * Read system from matrix-market file
  **/
  if (!viennacl::io::read_matrix_market_file(ublas_matrix, "../examples/testdata/mat65k.mtx"))
  {
    std::cout << "Error reading Matrix file" << std::endl;
    return EXIT_FAILURE;
  }

  /**
  * Read associated vectors from files
  **/
  if (!readVectorFromFile("../examples/testdata/rhs65025.txt", rhs))
  {
    std::cout << "Error reading RHS file" << std::endl;
    return EXIT_FAILURE;
  }
  //std::cout << "done reading rhs" << std::endl;

  if (!readVectorFromFile("../examples/testdata/result65025.txt", ref_result))
  {
    std::cout << "Error reading Result file" << std::endl;
    return EXIT_FAILURE;
  }
  //std::cout << "done reading result" << std::endl;


  /**
  * Set up ILUT preconditioners for ViennaCL and ublas objects. Other preconditioners can also be used, see \ref manual-algorithms-preconditioners "Preconditioners"
  **/
  viennacl::linalg::ilut_precond< ublas::compressed_matrix<ScalarType> >    ublas_ilut(ublas_matrix, viennacl::linalg::ilut_tag());
  viennacl::linalg::ilu0_precond< ublas::compressed_matrix<ScalarType> >    ublas_ilu0(ublas_matrix, viennacl::linalg::ilu0_tag());
  viennacl::linalg::block_ilu_precond< ublas::compressed_matrix<ScalarType>,
                                       viennacl::linalg::ilu0_tag>          ublas_block_ilu0(ublas_matrix, viennacl::linalg::ilu0_tag());

  /**
  * First we run an conjugate gradient solver with different preconditioners:
  **/
  std::cout << "----- CG Test -----" << std::endl;

  result = viennacl::linalg::solve(ublas_matrix, rhs, viennacl::linalg::cg_tag());
  std::cout << "Residual norm: " << norm_2(prod(ublas_matrix, result) - rhs) << std::endl;
  result = viennacl::linalg::solve(ublas_matrix, rhs, viennacl::linalg::cg_tag(1e-6, 20), ublas_ilut);
  std::cout << "Residual norm: " << norm_2(prod(ublas_matrix, result) - rhs) << std::endl;
  result = viennacl::linalg::solve(ublas_matrix, rhs, viennacl::linalg::cg_tag(1e-6, 20), ublas_ilu0);
  std::cout << "Residual norm: " << norm_2(prod(ublas_matrix, result) - rhs) << std::endl;
  result = viennacl::linalg::solve(ublas_matrix, rhs, viennacl::linalg::cg_tag(1e-6, 20), ublas_block_ilu0);
  std::cout << "Residual norm: " << norm_2(prod(ublas_matrix, result) - rhs) << std::endl;

  /**
  * Run the stabilized BiConjugate gradient solver without and with preconditioners (ILUT, ILU0)
  **/
  std::cout << "----- BiCGStab Test -----" << std::endl;

  result = viennacl::linalg::solve(ublas_matrix, rhs, viennacl::linalg::bicgstab_tag());          //without preconditioner
  result = viennacl::linalg::solve(ublas_matrix, rhs, viennacl::linalg::bicgstab_tag(1e-6, 20), ublas_ilut); //with preconditioner
  result = viennacl::linalg::solve(ublas_matrix, rhs, viennacl::linalg::bicgstab_tag(1e-6, 20), ublas_ilu0); //with preconditioner

  /**
  * Run the generalized minimum residual method witout and with preconditioners (ILUT, ILU0):
  **/
  std::cout << "----- GMRES Test -----" << std::endl;

  result = viennacl::linalg::solve(ublas_matrix, rhs, viennacl::linalg::gmres_tag());   //without preconditioner
  result = viennacl::linalg::solve(ublas_matrix, rhs, viennacl::linalg::gmres_tag(1e-6, 20), ublas_ilut);//with preconditioner
  result = viennacl::linalg::solve(ublas_matrix, rhs, viennacl::linalg::gmres_tag(1e-6, 20), ublas_ilu0);//with preconditioner

  /**
  *  That's it.
  **/
  std::cout << "!!!! TUTORIAL COMPLETED SUCCESSFULLY !!!!" << std::endl;

  return EXIT_SUCCESS;
}

