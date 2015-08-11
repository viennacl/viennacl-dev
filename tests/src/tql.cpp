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



/** \file tests/src/tql.cpp  Tests the tql algorithm for eigenvalue computations for symmetric tridiagonal matrices.
*   \test  Tests the tql algorithm for eigenvalue computations for symmetric tridiagonal matrices.
**/

/*
*
*   Test file for tql-algorithm
*
*/

// include necessary system headers
#include <iostream>


//include basic scalar and vector types of ViennaCL
#include "viennacl/scalar.hpp"
#include "viennacl/vector.hpp"
#include "viennacl/linalg/tql2.hpp"

#define EPS 10.0e-5


typedef float     ScalarType;


// Test the eigenvectors
// Perform the multiplication (T - lambda * I) * Q, with the original tridiagonal matrx T, the
// eigenvalues lambda and the eigenvectors in Q. Result has to be 0.

template <typename MatrixLayout>
bool test_eigen_val_vec(viennacl::matrix<ScalarType, MatrixLayout> & Q,
                      std::vector<ScalarType> & eigenvalues,
                      std::vector<ScalarType> & d,
                      std::vector<ScalarType> & e)
{
  std::size_t Q_size = Q.size2();
  ScalarType value = 0;

  for(std::size_t j = 0; j < Q_size; j++)
  {
    // calculate first row
    value = (d[0]- eigenvalues[j]) * Q(0, j) + e[1] * Q(1, j);
    if (value > EPS)
      return false;

    // calcuate inner rows
    for(std::size_t i = 1; i < Q_size - 1; i++)
    {
      value = e[i] * Q(i - 1, j) + (d[i]- eigenvalues[j]) * Q(i, j) + e[i + 1] * Q(i + 1, j);
      if (value > EPS)
        return false;
    }

    // calculate last row
    value = e[Q_size - 1] * Q(Q_size - 2, j) + (d[Q_size - 1] - eigenvalues[j]) * Q(Q_size - 1, j);
    if (value > EPS)
      return false;
  }
  return true;
}


/**
 * Test the tql2 algorithm for symmetric tridiagonal matrices.
 */

template <typename MatrixLayout>
void test_qr_method_sym()
{
  std::size_t sz = 220;

  viennacl::matrix<ScalarType, MatrixLayout> Q = viennacl::identity_matrix<ScalarType>(sz);
  std::vector<ScalarType> d(sz), e(sz), d_ref(sz), e_ref(sz);

  std::cout << "Testing matrix of size " << sz << "-by-" << sz << std::endl << std::endl;

  // Initialize diagonal and superdiagonal elements
  for(unsigned int i = 0; i < sz; ++i)
  {
    d[i] = ((float)(i % 9)) - 4.5f;
    e[i] = ((float)(i % 5)) - 4.5f;
  }
  e[0] = 0.0f;
  d_ref = d;
  e_ref = e;

//---Run the tql2 algorithm-----------------------------------
  viennacl::linalg::tql2(Q, d, e);


// ---Test the computed eigenvalues and eigenvectors
  if(!test_eigen_val_vec<MatrixLayout>(Q, d, d_ref, e_ref))
     exit(EXIT_FAILURE);
/*
  for( unsigned int i = 0; i < sz; ++i)
    std::cout << "Eigenvalue " << i << "= " << d[i] << std::endl;
    */
}

int main()
{

  std::cout << std::endl << "COMPUTATION OF EIGENVALUES AND EIGENVECTORS" << std::endl;
  std::cout << std::endl << "Testing QL algorithm for symmetric tridiagonal row-major matrices..." << std::endl;
  test_qr_method_sym<viennacl::row_major>();

  std::cout << std::endl << "Testing QL algorithm for symmetric tridiagonal column-major matrices..." << std::endl;
  test_qr_method_sym<viennacl::column_major>();

  std::cout << std::endl <<"--------TEST SUCCESSFULLY COMPLETED----------" << std::endl;
}
