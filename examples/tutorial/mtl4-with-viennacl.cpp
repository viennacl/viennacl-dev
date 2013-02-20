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
*   Tutorial:  Shows how to exchange data between ViennaCL and MTL4 (http://www.mtl4.org/) objects.
*   
*/

//
// include necessary system headers
//
#include <iostream>

//
// Include MTL4 headers
//
#include <boost/numeric/mtl/mtl.hpp>
#include <boost/numeric/itl/itl.hpp>

//
// Must be set prior to any ViennaCL includes if you want to use ViennaCL algorithms on MTL4 objects
//
#define VIENNACL_WITH_MTL4 1

//#define VIENNACL_BUILD_INFO
//#define VIENNACL_DEBUG_ALL

//
// ViennaCL includes
//
#include "viennacl/vector.hpp"
#include "viennacl/matrix.hpp"
#include "viennacl/linalg/vector_operations.hpp"
#include "viennacl/compressed_matrix.hpp"

#include "viennacl/linalg/ilu.hpp"
#include "viennacl/linalg/cg.hpp"
#include "viennacl/linalg/bicgstab.hpp"
#include "viennacl/linalg/gmres.hpp"


// Some helper functions for this tutorial:
#include "Random.hpp"
#include "vector-io.hpp"
#include "../benchmarks/benchmark-utils.hpp"

template <typename ScalarType>
void run_test()
{
  typedef mtl::dense2D<ScalarType>        MTL4DenseMatrix;
  typedef mtl::compressed2D<ScalarType>   MTL4SparseMatrix;
  
  //
  // Create and fill dense matrices from the MTL4 library:
  //
  mtl::dense2D<ScalarType>   mtl4_densemat(5, 5);
  mtl::dense2D<ScalarType>   mtl4_densemat2(5, 5);
  mtl4_densemat(0,0) = 2.0;   mtl4_densemat(0,1) = -1.0;
  mtl4_densemat(1,0) = -1.0;  mtl4_densemat(1,1) =  2.0;  mtl4_densemat(1,2) = -1.0;
  mtl4_densemat(2,1) = -1.0;  mtl4_densemat(2,2) = -1.0;  mtl4_densemat(2,3) = -1.0;
  mtl4_densemat(3,2) = -1.0;  mtl4_densemat(3,3) =  2.0;  mtl4_densemat(3,4) = -1.0;
                              mtl4_densemat(4,4) = -1.0;  mtl4_densemat(4,4) = -1.0;
  
  
  //
  // Create and fill sparse matrices from the MTL4 library:
  //
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
  
  //
  // Create and fill a few vectors from the MTL4 library:
  //
  mtl::dense_vector<ScalarType> mtl4_rhs(5, 0.0);
  mtl::dense_vector<ScalarType> mtl4_result(5, 0.0);
  mtl::dense_vector<ScalarType> mtl4_temp(5, 0.0);
  

  mtl4_rhs(0) = 10.0;
  mtl4_rhs(1) = 11.0;
  mtl4_rhs(2) = 12.0;
  mtl4_rhs(3) = 13.0;
  mtl4_rhs(4) = 14.0;
  
  //
  // Let us create the ViennaCL analogues:
  //
  viennacl::vector<ScalarType> vcl_rhs(5);
  viennacl::vector<ScalarType> vcl_result(5);
  viennacl::matrix<ScalarType> vcl_densemat(5, 5);
  viennacl::compressed_matrix<ScalarType> vcl_sparsemat(5, 5);

  //
  // Directly copy the MTL4 objects to ViennaCL objects
  //
  viennacl::copy(&(mtl4_rhs[0]), &(mtl4_rhs[0]) + 5, vcl_rhs.begin());  //method 1: via iterator interface (cf. std::copy())
  viennacl::copy(mtl4_rhs, vcl_rhs);  //method 2: via built-in wrappers (convenience layer)
  
  viennacl::copy(mtl4_densemat, vcl_densemat);
  viennacl::copy(mtl4_sparsemat, vcl_sparsemat);
  
  // For completeness: Copy matrices from ViennaCL back to Eigen:
  viennacl::copy(vcl_densemat, mtl4_densemat2);
  viennacl::copy(vcl_sparsemat, mtl4_sparsemat2);
  
  //
  // Run matrix-vector products and compare results:
  //
  mtl4_result = mtl4_densemat * mtl4_rhs;
  vcl_result = viennacl::linalg::prod(vcl_densemat, vcl_rhs);
  viennacl::copy(vcl_result, mtl4_temp);
  mtl4_result -= mtl4_temp;
  std::cout << "Difference for dense matrix-vector product: " << mtl::two_norm(mtl4_result) << std::endl;
  mtl4_result = mtl4_densemat2 * mtl4_rhs - mtl4_temp;
  std::cout << "Difference for dense matrix-vector product (MTL4->ViennaCL->MTL4): "
            << mtl::two_norm(mtl4_result) << std::endl;
  
  //
  // Same for sparse matrix:
  //          
  mtl4_result = mtl4_sparsemat * mtl4_rhs;
  vcl_result = viennacl::linalg::prod(vcl_sparsemat, vcl_rhs);
  viennacl::copy(vcl_result, mtl4_temp);
  mtl4_result -= mtl4_temp;
  std::cout << "Difference for sparse matrix-vector product: " << mtl::two_norm(mtl4_result) << std::endl;
  mtl4_result = mtl4_sparsemat2 * mtl4_rhs - mtl4_temp;
  std::cout << "Difference for sparse matrix-vector product (MTL4->ViennaCL->MTL4): "
            << mtl::two_norm(mtl4_result) << std::endl;
            
  //
  // Please have a look at the other tutorials on how to use the ViennaCL types
  //
}



int main(int, char *[])
{
  std::cout << "----------------------------------------------" << std::endl;
  std::cout << "## Single precision" << std::endl;
  std::cout << "----------------------------------------------" << std::endl;
  run_test<float>();
  
#ifdef VIENNACL_HAVE_OPENCL   
  if( viennacl::ocl::current_device().double_support() )
#endif
  {
    std::cout << "----------------------------------------------" << std::endl;
    std::cout << "## Double precision" << std::endl;
    std::cout << "----------------------------------------------" << std::endl;
    run_test<double>();
  }
  
  std::cout << std::endl;
  std::cout << "!!!! TUTORIAL COMPLETED SUCCESSFULLY !!!!" << std::endl;
  std::cout << std::endl;
}
