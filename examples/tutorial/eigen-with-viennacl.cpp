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
*   Tutorial:  Shows how to exchange data between ViennaCL and Eigen (http://eigen.tuxfamily.org/) objects.
*   
*/

//
// include necessary system headers
//
#include <iostream>

//
// Include Eigen headers
//
#include <Eigen/Core>
#include <Eigen/Sparse>

//
// IMPORTANT: Must be set prior to any ViennaCL includes if you want to use ViennaCL algorithms on Eigen objects
//
#define VIENNACL_WITH_EIGEN 1

//
// ViennaCL includes
//
#include "viennacl/vector.hpp"
#include "viennacl/matrix.hpp"
#include "viennacl/compressed_matrix.hpp"
#include "viennacl/linalg/prod.hpp"



// Some helper functions for this tutorial:
#include "Random.hpp"
#include "vector-io.hpp"
#include "../benchmarks/benchmark-utils.hpp"

//
// A little bit of template magic follows (to get rid of the hard-wired scalar types in Eigen)
//

//dense matrix:
template <typename T>
struct Eigen_dense_matrix
{
  typedef typename T::ERROR_NO_EIGEN_TYPE_AVAILABLE   error_type;
};

template <>
struct Eigen_dense_matrix<float>
{
  typedef Eigen::MatrixXf  type; 
};

template <>
struct Eigen_dense_matrix<double>
{
  typedef Eigen::MatrixXd  type; 
};


//sparse matrix
template <typename T>
struct Eigen_vector
{
  typedef typename T::ERROR_NO_EIGEN_TYPE_AVAILABLE   error_type;
};

template <>
struct Eigen_vector<float>
{
  typedef Eigen::VectorXf  type; 
};

template <>
struct Eigen_vector<double>
{
  typedef Eigen::VectorXd  type; 
};





template <typename ScalarType>
void run_test()
{
  //
  // get Eigen matrix and vector types for the provided ScalarType:
  //
  typedef typename Eigen_dense_matrix<ScalarType>::type  EigenMatrix;
  typedef typename Eigen_vector<ScalarType>::type        EigenVector;
  
  //
  // Create and fill dense matrices from the Eigen library:
  //
  EigenMatrix eigen_densemat(6, 5);
  EigenMatrix eigen_densemat2(6, 5);
  eigen_densemat(0,0) = 2.0;   eigen_densemat(0,1) = -1.0;
  eigen_densemat(1,0) = -1.0;  eigen_densemat(1,1) =  2.0;  eigen_densemat(1,2) = -1.0;
  eigen_densemat(2,1) = -1.0;  eigen_densemat(2,2) = -1.0;  eigen_densemat(2,3) = -1.0;
  eigen_densemat(3,2) = -1.0;  eigen_densemat(3,3) =  2.0;  eigen_densemat(3,4) = -1.0;
                               eigen_densemat(5,4) = -1.0;  eigen_densemat(4,4) = -1.0;

  //
  // Create and fill sparse matrices from the Eigen library:
  //
  Eigen::SparseMatrix<ScalarType, Eigen::RowMajor> eigen_sparsemat(6, 5);
  Eigen::SparseMatrix<ScalarType, Eigen::RowMajor> eigen_sparsemat2(6, 5);
  eigen_sparsemat.reserve(5*2);
  eigen_sparsemat.insert(0,0) = 2.0;   eigen_sparsemat.insert(0,1) = -1.0;
  eigen_sparsemat.insert(1,1) = 2.0;   eigen_sparsemat.insert(1,2) = -1.0;
  eigen_sparsemat.insert(2,2) = -1.0;  eigen_sparsemat.insert(2,3) = -1.0;
  eigen_sparsemat.insert(3,3) = 2.0;   eigen_sparsemat.insert(3,4) = -1.0;
  eigen_sparsemat.insert(5,4) = -1.0;
  //eigen_sparsemat.endFill();
  
  //
  // Create and fill a few vectors from the Eigen library:
  //
  EigenVector eigen_rhs(5);
  EigenVector eigen_result(6);
  EigenVector eigen_temp(6);

  eigen_rhs(0) = 10.0;
  eigen_rhs(1) = 11.0;
  eigen_rhs(2) = 12.0;
  eigen_rhs(3) = 13.0;
  eigen_rhs(4) = 14.0;
  
  
  //
  // Let us create the ViennaCL analogues:
  //
  viennacl::vector<ScalarType> vcl_rhs(5);
  viennacl::vector<ScalarType> vcl_result(6);
  viennacl::matrix<ScalarType> vcl_densemat(6, 5);
  viennacl::compressed_matrix<ScalarType> vcl_sparsemat(6, 5);
  
  
  //
  // Directly copy the Eigen objects to ViennaCL objects
  //
  viennacl::copy(&(eigen_rhs[0]), &(eigen_rhs[0]) + 5, vcl_rhs.begin());  //method 1: via iterator interface (cf. std::copy())
  viennacl::copy(eigen_rhs, vcl_rhs);  //method 2: via built-in wrappers (convenience layer)
  
  viennacl::copy(eigen_densemat, vcl_densemat);
  viennacl::copy(eigen_sparsemat, vcl_sparsemat);
  std::cout << "VCL sparsematrix dimensions: " << vcl_sparsemat.size1() << ", " << vcl_sparsemat.size2() << std::endl;
  
  // For completeness: Copy matrices from ViennaCL back to Eigen:
  viennacl::copy(vcl_densemat, eigen_densemat2);
  viennacl::copy(vcl_sparsemat, eigen_sparsemat2);
  
  
  //
  // Run matrix-vector products and compare results:
  //
  eigen_result = eigen_densemat * eigen_rhs;
  vcl_result = viennacl::linalg::prod(vcl_densemat, vcl_rhs);
  viennacl::copy(vcl_result, eigen_temp);
  std::cout << "Difference for dense matrix-vector product: " << (eigen_result - eigen_temp).norm() << std::endl;
  std::cout << "Difference for dense matrix-vector product (Eigen->ViennaCL->Eigen): "
            << (eigen_densemat2 * eigen_rhs - eigen_temp).norm() << std::endl;
  
  //
  // Same for sparse matrix:
  //          
  eigen_result = eigen_sparsemat * eigen_rhs;
  vcl_result = viennacl::linalg::prod(vcl_sparsemat, vcl_rhs);
  viennacl::copy(vcl_result, eigen_temp);
  std::cout << "Difference for sparse matrix-vector product: " << (eigen_result - eigen_temp).norm() << std::endl;
  std::cout << "Difference for sparse matrix-vector product (Eigen->ViennaCL->Eigen): "
            << (eigen_sparsemat2 * eigen_rhs - eigen_temp).norm() << std::endl;
            
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
