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

/** \example armadillo-with-viennacl.cpp
*
*   This tutorial shows how data can be directly transferred from the <a href="http://arma.sourceforge.net/">Armadillo Library</a> to ViennaCL objects using the built-in convenience wrappers.
*
*   The first step is to include the necessary headers and activate the Armadillo convenience functions in ViennaCL:
**/

// System headers
#include <iostream>

// Armadillo headers (disable BLAS and LAPACK to avoid linking issues)
#define ARMA_DONT_USE_BLAS
#define ARMA_DONT_USE_LAPACK
#include <armadillo>

// IMPORTANT: Must be set prior to any ViennaCL includes if you want to use ViennaCL algorithms on Armadillo objects
#define VIENNACL_WITH_ARMADILLO 1


// ViennaCL includes
#include "viennacl/vector.hpp"
#include "viennacl/matrix.hpp"
#include "viennacl/compressed_matrix.hpp"
#include "viennacl/linalg/prod.hpp"


// Helper functions for this tutorial:
#include "vector-io.hpp"


/**
*    The following function contains the main code for this tutorial.
*    It consists of the following steps:
*      - Creates Armadillo matrices and vectors
*      - Initializes them with data
*      - Create ViennaCL objects
*      - Copy them over to the respective ViennaCL objects
*      - Compute matrix-vector products in both Armadillo and ViennaCL and compare results.
*
**/
template<typename NumericT>
void run_tutorial()
{
  typedef arma::SpMat<NumericT>  ArmaSparseMatrix;
  typedef arma::Mat<NumericT>    ArmaMatrix;
  typedef arma::Col<NumericT>    ArmaVector;

  /**
  * Create and fill dense matrices from the Armadillo library:
  **/
  ArmaMatrix arma_densemat(6, 5);
  ArmaMatrix arma_densemat2(6, 5);
  arma_densemat(0,0) = 2.0;   arma_densemat(0,1) = -1.0;
  arma_densemat(1,0) = -1.0;  arma_densemat(1,1) =  2.0;  arma_densemat(1,2) = -1.0;
  arma_densemat(2,1) = -1.0;  arma_densemat(2,2) = -1.0;  arma_densemat(2,3) = -1.0;
  arma_densemat(3,2) = -1.0;  arma_densemat(3,3) =  2.0;  arma_densemat(3,4) = -1.0;
                              arma_densemat(5,4) = -1.0;  arma_densemat(4,4) = -1.0;

  /**
  * Create and fill sparse matrices from the Armadillo library:
  **/
  ArmaSparseMatrix arma_sparsemat(6, 5);
  ArmaSparseMatrix arma_sparsemat2(6, 5);
  arma_sparsemat(0,0) = 2.0;   arma_sparsemat(0,1) = -1.0;
  arma_sparsemat(1,1) = 2.0;   arma_sparsemat(1,2) = -1.0;
  arma_sparsemat(2,2) = -1.0;  arma_sparsemat(2,3) = -1.0;
  arma_sparsemat(3,3) = 2.0;   arma_sparsemat(3,4) = -1.0;
  arma_sparsemat(5,4) = -1.0;

  /**
  * Create and fill a few vectors from the Armadillo library:
  **/
  ArmaVector arma_rhs(5);
  ArmaVector arma_result(6);
  ArmaVector arma_temp(6);

  arma_rhs(0) = 10.0;
  arma_rhs(1) = 11.0;
  arma_rhs(2) = 12.0;
  arma_rhs(3) = 13.0;
  arma_rhs(4) = 14.0;


  /**
  * Create the corresponding ViennaCL objects:
  **/
  viennacl::vector<NumericT> vcl_rhs(5);
  viennacl::vector<NumericT> vcl_result(6);
  viennacl::matrix<NumericT> vcl_densemat(6, 5);
  viennacl::compressed_matrix<NumericT> vcl_sparsemat(6, 5);


  /**
  * Directly copy the Armadillo objects to ViennaCL objects
  **/
  viennacl::copy(arma_rhs.memptr(), arma_rhs.memptr() + arma_rhs.n_elem, vcl_rhs.begin());  //method 1: via iterator interface (cf. std::copy())
  viennacl::copy(arma_rhs, vcl_rhs);  //method 2: via built-in wrappers (convenience layer)

  viennacl::copy(arma_densemat, vcl_densemat);
  viennacl::copy(arma_sparsemat, vcl_sparsemat);
  std::cout << "VCL sparsematrix dimensions: " << vcl_sparsemat.size1() << ", " << vcl_sparsemat.size2() << std::endl;

  // For completeness: Copy matrices from ViennaCL back to Eigen:
  viennacl::copy(vcl_densemat, arma_densemat2);
  viennacl::copy(vcl_sparsemat, arma_sparsemat2);


  /**
  * Run dense matrix-vector products and compare results:
  **/
  arma_result = arma_densemat * arma_rhs;
  vcl_result = viennacl::linalg::prod(vcl_densemat, vcl_rhs);
  viennacl::copy(vcl_result, arma_temp);
  std::cout << "Difference for dense matrix-vector product: " << norm(arma_result - arma_temp) << std::endl;
  std::cout << "Difference for dense matrix-vector product (Armadillo -> ViennaCL -> Armadillo): "
            << norm(arma_densemat2 * arma_rhs - arma_temp) << std::endl;

  /**
  * Run sparse matrix-vector products and compare results:
  **/
  arma_result = arma_sparsemat * arma_rhs;
  vcl_result = viennacl::linalg::prod(vcl_sparsemat, vcl_rhs);
  viennacl::copy(vcl_result, arma_temp);
  std::cout << "Difference for sparse matrix-vector product: " << norm(arma_result - arma_temp) << std::endl;
  std::cout << "Difference for sparse matrix-vector product (Armadillo -> ViennaCL -> Armadillo): "
            << norm(arma_sparsemat2 * arma_rhs - arma_temp) << std::endl;
}


/**
*   In the main() routine we only call the worker function defined above with both single and double precision arithmetic.
**/
int main(int, char *[])
{
  std::cout << "----------------------------------------------" << std::endl;
  std::cout << "## Single precision" << std::endl;
  std::cout << "----------------------------------------------" << std::endl;
  run_tutorial<float>();

#ifdef VIENNACL_HAVE_OPENCL
  if ( viennacl::ocl::current_device().double_support() )
#endif
  {
    std::cout << "----------------------------------------------" << std::endl;
    std::cout << "## Double precision" << std::endl;
    std::cout << "----------------------------------------------" << std::endl;
    run_tutorial<double>();
  }

  /**
  *   That's it. Print a success message and exit.
  **/
  std::cout << std::endl;
  std::cout << "!!!! TUTORIAL COMPLETED SUCCESSFULLY !!!!" << std::endl;
  std::cout << std::endl;

}
