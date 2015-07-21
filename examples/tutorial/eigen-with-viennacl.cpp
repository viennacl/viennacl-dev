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

/** \example eigen-with-viennacl.cpp
*
*   This tutorial shows how data can be directly transferred from the <a href="http://eigen.tuxfamily.org/">Eigen Library</a> to ViennaCL objects using the built-in convenience wrappers.
*
*   The first step is to include the necessary headers and activate the Eigen convenience functions in ViennaCL:
**/

// System headers
#include <iostream>

// Eigen headers
#include <Eigen/Core>
#include <Eigen/Sparse>

// IMPORTANT: Must be set prior to any ViennaCL includes if you want to use ViennaCL algorithms on Eigen objects
#define VIENNACL_WITH_EIGEN 1


// ViennaCL includes
#include "viennacl/vector.hpp"
#include "viennacl/matrix.hpp"
#include "viennacl/compressed_matrix.hpp"
#include "viennacl/linalg/prod.hpp"


// Helper functions for this tutorial:
#include "vector-io.hpp"

/**
*   The following is a set of auxiliary dispatchers for obtaining the right Eigen types for a given floating point type.
*   This is merely an implementation detail, so feel free to skip over it.
**/

//dense matrix:
template<typename T>
struct Eigen_dense_matrix
{
  typedef typename T::ERROR_NO_EIGEN_TYPE_AVAILABLE   error_type;
};

template<>
struct Eigen_dense_matrix<float>
{
  typedef Eigen::MatrixXf  type;
};

template<>
struct Eigen_dense_matrix<double>
{
  typedef Eigen::MatrixXd  type;
};


//sparse matrix
template<typename T>
struct Eigen_vector
{
  typedef typename T::ERROR_NO_EIGEN_TYPE_AVAILABLE   error_type;
};

template<>
struct Eigen_vector<float>
{
  typedef Eigen::VectorXf  type;
};

template<>
struct Eigen_vector<double>
{
  typedef Eigen::VectorXd  type;
};



/**
*    The following function contains the main code for this tutorial.
*    It consists of the following steps:
*      - Creates Eigen matrices and vectors
*      - Initializes them with data
*      - Create ViennaCL objects
*      - Copy them over to the respective ViennaCL objects
*      - Compute matrix-vector products in both Eigen and ViennaCL and compare results.
*
**/
template<typename ScalarType>
void run_tutorial()
{
  /**
  * Get Eigen matrix and vector types for the provided ScalarType.
  * Involves a little bit of template-metaprogramming.
  **/
  typedef typename Eigen_dense_matrix<ScalarType>::type  EigenMatrix;
  typedef typename Eigen_vector<ScalarType>::type        EigenVector;

  /**
  * Create and fill dense matrices from the Eigen library:
  **/
  EigenMatrix eigen_densemat(6, 5);
  EigenMatrix eigen_densemat2(6, 5);
  eigen_densemat(0,0) = 2.0;   eigen_densemat(0,1) = -1.0;
  eigen_densemat(1,0) = -1.0;  eigen_densemat(1,1) =  2.0;  eigen_densemat(1,2) = -1.0;
  eigen_densemat(2,1) = -1.0;  eigen_densemat(2,2) = -1.0;  eigen_densemat(2,3) = -1.0;
  eigen_densemat(3,2) = -1.0;  eigen_densemat(3,3) =  2.0;  eigen_densemat(3,4) = -1.0;
                               eigen_densemat(5,4) = -1.0;  eigen_densemat(4,4) = -1.0;
  Eigen::Map<EigenMatrix> eigen_densemat_map(eigen_densemat.data(), 6, 5); // same as eigen_densemat, but emulating user-provided buffer

  /**
  * Create and fill sparse matrices from the Eigen library:
  **/
  Eigen::SparseMatrix<ScalarType, Eigen::RowMajor> eigen_sparsemat(6, 5);
  Eigen::SparseMatrix<ScalarType, Eigen::RowMajor> eigen_sparsemat2(6, 5);
  eigen_sparsemat.reserve(5*2);
  eigen_sparsemat.insert(0,0) = 2.0;   eigen_sparsemat.insert(0,1) = -1.0;
  eigen_sparsemat.insert(1,1) = 2.0;   eigen_sparsemat.insert(1,2) = -1.0;
  eigen_sparsemat.insert(2,2) = -1.0;  eigen_sparsemat.insert(2,3) = -1.0;
  eigen_sparsemat.insert(3,3) = 2.0;   eigen_sparsemat.insert(3,4) = -1.0;
  eigen_sparsemat.insert(5,4) = -1.0;
  //eigen_sparsemat.endFill();

  /**
  * Create and fill a few vectors from the Eigen library:
  **/
  EigenVector eigen_rhs(5);
  Eigen::Map<EigenVector> eigen_rhs_map(eigen_rhs.data(), 5);
  EigenVector eigen_result(6);
  EigenVector eigen_temp(6);

  eigen_rhs(0) = 10.0;
  eigen_rhs(1) = 11.0;
  eigen_rhs(2) = 12.0;
  eigen_rhs(3) = 13.0;
  eigen_rhs(4) = 14.0;


  /**
  * Create the corresponding ViennaCL objects:
  **/
  viennacl::vector<ScalarType> vcl_rhs(5);
  viennacl::vector<ScalarType> vcl_result(6);
  viennacl::matrix<ScalarType> vcl_densemat(6, 5);
  viennacl::compressed_matrix<ScalarType> vcl_sparsemat(6, 5);


  /**
  * Directly copy the Eigen objects to ViennaCL objects
  **/
  viennacl::copy(&(eigen_rhs[0]), &(eigen_rhs[0]) + 5, vcl_rhs.begin());  // Method 1: via iterator interface (cf. std::copy())
  viennacl::copy(eigen_rhs, vcl_rhs);                                     // Method 2: via built-in wrappers (convenience layer)
  viennacl::copy(eigen_rhs_map, vcl_rhs);                                 // Same as method 2, but for a mapped vector

  viennacl::copy(eigen_densemat, vcl_densemat);
  viennacl::copy(eigen_densemat_map, vcl_densemat); //same as above, using mapped matrix
  viennacl::copy(eigen_sparsemat, vcl_sparsemat);
  std::cout << "VCL sparsematrix dimensions: " << vcl_sparsemat.size1() << ", " << vcl_sparsemat.size2() << std::endl;

  // For completeness: Copy matrices from ViennaCL back to Eigen:
  viennacl::copy(vcl_densemat, eigen_densemat2);
  viennacl::copy(vcl_sparsemat, eigen_sparsemat2);


  /**
  * Run dense matrix-vector products and compare results:
  **/
  eigen_result = eigen_densemat * eigen_rhs;
  vcl_result = viennacl::linalg::prod(vcl_densemat, vcl_rhs);
  viennacl::copy(vcl_result, eigen_temp);
  std::cout << "Difference for dense matrix-vector product: " << (eigen_result - eigen_temp).norm() << std::endl;
  std::cout << "Difference for dense matrix-vector product (Eigen->ViennaCL->Eigen): "
            << (eigen_densemat2 * eigen_rhs - eigen_temp).norm() << std::endl;

  /**
  * Run sparse matrix-vector products and compare results:
  **/
  eigen_result = eigen_sparsemat * eigen_rhs;
  vcl_result = viennacl::linalg::prod(vcl_sparsemat, vcl_rhs);
  viennacl::copy(vcl_result, eigen_temp);
  std::cout << "Difference for sparse matrix-vector product: " << (eigen_result - eigen_temp).norm() << std::endl;
  std::cout << "Difference for sparse matrix-vector product (Eigen->ViennaCL->Eigen): "
            << (eigen_sparsemat2 * eigen_rhs - eigen_temp).norm() << std::endl;
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
