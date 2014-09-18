/* =========================================================================
   Copyright (c) 2010-2014, Institute for Microelectronics,
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
*   Test file for tql-algorithm
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

#include "viennacl/io/matrix_market.hpp"

#include <iostream>
#include <fstream>
#include <limits>
#include <string>
#include <iomanip>

#include "viennacl/linalg/qr-method.hpp"
#include "viennacl/linalg/qr-method-common.hpp"
#include "viennacl/linalg/host_based/matrix_operations.hpp"
#include "Random.hpp"

#define EPS 10.0e-5



namespace ublas = boost::numeric::ublas;
typedef float     ScalarType;


template <typename MatrixLayout>
void matrix_print(viennacl::matrix<ScalarType, MatrixLayout>& A_orig)
{
    ublas::matrix<ScalarType> A(A_orig.size1(), A_orig.size2());
    viennacl::copy(A_orig, A);
    for (unsigned int i = 0; i < A.size1(); i++) {
        for (unsigned int j = 0; j < A.size2(); j++)
           std::cout << std::setprecision(6) << std::fixed << A(i, j) << "\t";
        std::cout << std::endl;
    }
    std::cout << std::endl;
}

void vector_print(std::vector<ScalarType>& v )
{
    for (unsigned int i = 0; i < v.size(); i++)
      std::cout << std::setprecision(6) << std::fixed << v[i] << "\t";
    std::cout << "\n";
}

template <typename MatrixLayout>
void eig_tutorial()
{
  std::size_t sz = 10;
  std::cout << "Compute eigenvalues and eigenvectors of matrix of size " << sz << "-by-" << sz << std::endl << std::endl;

  std::vector<ScalarType> d(sz), e(sz);
  // Initialize diagonal and superdiagonal elements of the tridiagonal matrix
  d[0] = 1; e[0] = 0;
  d[1] = 2; e[1] = 4;
  d[2] =-4; e[2] = 5;
  d[3] = 6; e[3] = 1;
  d[4] = 3; e[4] = 2;
  d[5] = 4; e[5] =-3;
  d[6] = 7; e[6] = 5;
  d[7] = 9; e[7] = 1;
  d[8] = 3; e[8] = 5;
  d[9] = 8; e[9] = 2;

  // Initialize Q as the identity matrix
  viennacl::matrix<ScalarType, MatrixLayout> Q = viennacl::identity_matrix<ScalarType>(sz);

  //--------------------------------------------------------
  // Compute the eigenvalues and eigenvectors
  viennacl::linalg::tql2(Q, d, e);

  // Eigenvalues are stored in d:
  std::cout << "Eigenvalues: " << std::endl;
  vector_print(d);

  std::cout << std::endl << "Eigenvectors corresponding to the eigenvalues above are the columns: " << std::endl << std::endl;
  matrix_print(Q);
}

int main()
{

  std::cout << std::endl << "Starting tutorial for symmetric tridiagonal matrices..." << std::endl;
  eig_tutorial<viennacl::row_major>();

  std::cout << std::endl <<"--------TUTORIAL COMPLETED----------" << std::endl;
}
