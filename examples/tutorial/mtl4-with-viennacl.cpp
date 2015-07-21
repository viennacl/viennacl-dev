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


/** \example mtl4-with-viennacl.cpp
*
*   This tutorial shows how data can be directly transferred from the <a href="http://www.mtl4.org/">MTL4 Library</a> to ViennaCL objects using the built-in convenience wrappers.
*
*   The first step is to include the necessary headers and activate the MTL4 convenience functions in ViennaCL:
**/

// System headers
#include <iostream>


// MTL4 headers
#include <boost/numeric/mtl/mtl.hpp>
#include <boost/numeric/itl/itl.hpp>


// Must be set prior to any ViennaCL includes if you want to use ViennaCL algorithms on MTL4 objects
#define VIENNACL_WITH_MTL4 1


// ViennaCL headers
#include "viennacl/vector.hpp"
#include "viennacl/matrix.hpp"
#include "viennacl/compressed_matrix.hpp"

#include "viennacl/linalg/ilu.hpp"
#include "viennacl/linalg/cg.hpp"
#include "viennacl/linalg/bicgstab.hpp"
#include "viennacl/linalg/gmres.hpp"


// Some helper functions for this tutorial:
#include "vector-io.hpp"

/**
*    The following function contains the main code for this tutorial.
*    It consists of the following steps:
*      - Creates MTL4 matrices and vectors
*      - Initializes them with data
*      - Create ViennaCL objects
*      - Copy them over to the respective ViennaCL objects
*      - Compute matrix-vector products in both MTL4 and ViennaCL and compare results.
*
**/
template<typename ScalarType>
void run_tutorial()
{
  typedef mtl::dense2D<ScalarType>        MTL4DenseMatrix;
  typedef mtl::compressed2D<ScalarType>   MTL4SparseMatrix;

  /**
  * Create and fill dense matrices from the MTL4 library:
  **/
  mtl::dense2D<ScalarType>   mtl4_densemat(5, 5);
  mtl::dense2D<ScalarType>   mtl4_densemat2(5, 5);
  mtl4_densemat(0,0) = 2.0;   mtl4_densemat(0,1) = -1.0;
  mtl4_densemat(1,0) = -1.0;  mtl4_densemat(1,1) =  2.0;  mtl4_densemat(1,2) = -1.0;
  mtl4_densemat(2,1) = -1.0;  mtl4_densemat(2,2) = -1.0;  mtl4_densemat(2,3) = -1.0;
  mtl4_densemat(3,2) = -1.0;  mtl4_densemat(3,3) =  2.0;  mtl4_densemat(3,4) = -1.0;
                              mtl4_densemat(4,4) = -1.0;  mtl4_densemat(4,4) = -1.0;


  /**
  * Create and fill sparse matrices from the MTL4 library:
  **/
  MTL4SparseMatrix mtl4_sparsemat;
  set_to_zero(mtl4_sparsemat);
  mtl4_sparsemat.change_dim(5, 5);

  MTL4SparseMatrix mtl4_sparsemat2;
  set_to_zero(mtl4_sparsemat2);
  mtl4_sparsemat2.change_dim(5, 5);

  {
    mtl::matrix::inserter< MTL4SparseMatrix >  ins(mtl4_sparsemat);
    typedef typename mtl::Collection<MTL4SparseMatrix>::value_type  ValueType;
    ins(0,0) <<  ValueType(2.0);   ins(0,1) << ValueType(-1.0);
    ins(1,1) <<  ValueType(2.0);   ins(1,2) << ValueType(-1.0);
    ins(2,2) << ValueType(-1.0);   ins(2,3) << ValueType(-1.0);
    ins(3,3) <<  ValueType(2.0);   ins(3,4) << ValueType(-1.0);
    ins(4,4) << ValueType(-1.0);
  }

  /**
  * Create and fill a few vectors from the MTL4 library:
  **/
  mtl::dense_vector<ScalarType> mtl4_rhs(5, 0.0);
  mtl::dense_vector<ScalarType> mtl4_result(5, 0.0);
  mtl::dense_vector<ScalarType> mtl4_temp(5, 0.0);


  mtl4_rhs(0) = 10.0;
  mtl4_rhs(1) = 11.0;
  mtl4_rhs(2) = 12.0;
  mtl4_rhs(3) = 13.0;
  mtl4_rhs(4) = 14.0;

  /**
  * Create the corresponding ViennaCL objects:
  **/
  viennacl::vector<ScalarType> vcl_rhs(5);
  viennacl::vector<ScalarType> vcl_result(5);
  viennacl::matrix<ScalarType> vcl_densemat(5, 5);
  viennacl::compressed_matrix<ScalarType> vcl_sparsemat(5, 5);

  /**
  * Directly copy the MTL4 objects to ViennaCL objects
  **/
  viennacl::copy(&(mtl4_rhs[0]), &(mtl4_rhs[0]) + 5, vcl_rhs.begin());  //method 1: via iterator interface (cf. std::copy())
  viennacl::copy(mtl4_rhs, vcl_rhs);  //method 2: via built-in wrappers (convenience layer)

  viennacl::copy(mtl4_densemat, vcl_densemat);
  viennacl::copy(mtl4_sparsemat, vcl_sparsemat);

  // For completeness: Copy matrices from ViennaCL back to Eigen:
  viennacl::copy(vcl_densemat, mtl4_densemat2);
  viennacl::copy(vcl_sparsemat, mtl4_sparsemat2);

  /**
  * Run dense matrix-vector products and compare results:
  **/
  mtl4_result = mtl4_densemat * mtl4_rhs;
  vcl_result = viennacl::linalg::prod(vcl_densemat, vcl_rhs);
  viennacl::copy(vcl_result, mtl4_temp);
  mtl4_result -= mtl4_temp;
  std::cout << "Difference for dense matrix-vector product: " << mtl::two_norm(mtl4_result) << std::endl;
  mtl4_result = mtl4_densemat2 * mtl4_rhs - mtl4_temp;
  std::cout << "Difference for dense matrix-vector product (MTL4->ViennaCL->MTL4): "
            << mtl::two_norm(mtl4_result) << std::endl;

  /**
  * Run sparse matrix-vector products and compare results:
  **/
  mtl4_result = mtl4_sparsemat * mtl4_rhs;
  vcl_result = viennacl::linalg::prod(vcl_sparsemat, vcl_rhs);
  viennacl::copy(vcl_result, mtl4_temp);
  mtl4_result -= mtl4_temp;
  std::cout << "Difference for sparse matrix-vector product: " << mtl::two_norm(mtl4_result) << std::endl;
  mtl4_result = mtl4_sparsemat2 * mtl4_rhs - mtl4_temp;
  std::cout << "Difference for sparse matrix-vector product (MTL4->ViennaCL->MTL4): "
            << mtl::two_norm(mtl4_result) << std::endl;

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
