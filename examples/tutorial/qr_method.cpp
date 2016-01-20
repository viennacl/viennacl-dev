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

/** \example qr_method.cpp
*
* In this tutorial the eigenvalues and eigenvectors of a symmetric 9-by-9 matrix are calculated using the QR-method.
*
* The first step is to include the necessary headers and to define the floating point type used.
**/

// System headers:
#include <iostream>
#include <fstream>
#include <stdexcept>
#include <vector>

// ViennaCL headers:
#include "viennacl/matrix.hpp"
#include "viennacl/linalg/prod.hpp"
#include "viennacl/linalg/qr-method.hpp"


/**
* Helper routine for initializing a ViennaCL matrix and returning its associated eigenvalues
**/
template <typename ScalarType>
void initialize(viennacl::matrix<ScalarType> & A, std::vector<ScalarType> & v)
{
  // System matrix:
  ScalarType M[9][9] = {{ 4,  1, -2, 2, -7,  3,  9, -6, -2},
                        { 1, -2,  0, 1, -1,  5,  4,  7,  3},
                        {-2,  0,  3, 2,  0,  3,  6,  1, -1},
                        { 2,  1,  2, 1,  4,  5,  6,  7,  8},
                        {-7, -1,  0, 4,  5,  4,  9,  1, -8},
                        { 3,  5,  3, 5,  4,  9, -3,  3,  3},
                        { 9,  4,  6, 6,  9, -3,  3,  6, -7},
                        {-6,  7,  1, 7,  1,  3,  6,  2,  6},
                        {-2,  3, -1, 8, -8,  3, -7,  6,  1}};

  for(std::size_t i = 0; i < 9; i++)
    for(std::size_t j = 0; j < 9; j++)
      A(i, j) = M[i][j];

  // Known eigenvalues:
  ScalarType V[9] = {ScalarType(12.6005), ScalarType(19.5905), ScalarType(8.06067), ScalarType(2.95074), ScalarType(0.223506),
                     ScalarType(24.3642), ScalarType(-9.62084), ScalarType(-13.8374), ScalarType(-18.3319)};

  for(std::size_t i = 0; i < 9; i++)
    v[i] = V[i];
}

/**
*  Helper routine: Print an STL vector
**/
template <typename ScalarType>
void vector_print(std::vector<ScalarType>& v )
{
  for (unsigned int i = 0; i < v.size(); i++)
    std::cout << std::setprecision(6) << std::fixed << v[i] << "\t";
  std::cout << std::endl;
}

/**
*   Create a system of size 9-by-9 with known eigenvalues and compare it with the eigenvalues computed from the QR method implemented in ViennaCL.
**/
int main()
{
  // Change to 'double' if supported by your hardware.
  typedef float ScalarType;

  std::cout << "Testing matrix of size " << 9 << "-by-" << 9 << std::endl;

  viennacl::matrix<ScalarType> A_input(9,9);
  viennacl::matrix<ScalarType> Q(9, 9);
  std::vector<ScalarType> eigenvalues_ref(9);
  std::vector<ScalarType> eigenvalues(9);

  viennacl::vector<ScalarType> vcl_eigenvalues(9);

  initialize(A_input, eigenvalues_ref);

  std::cout << std::endl <<"Input matrix: " << std::endl;
  std::cout << A_input << std::endl;

  viennacl::matrix<ScalarType> A_input2(A_input); // duplicate to demonstrate the use with both std::vector and viennacl::vector


  /**
  * Call the function qr_method_sym() to calculate eigenvalues and eigenvectors
  * Parameters:
  *  -     A_input      - input matrix to find eigenvalues and eigenvectors from
  *  -     Q            - matrix, where the calculated eigenvectors will be stored in
  *  -     eigenvalues  - vector, where the calculated eigenvalues will be stored in
  **/

  std::cout << "Calculation..." << std::endl;
  viennacl::linalg::qr_method_sym(A_input, Q, eigenvalues);

  /**
   *  Same as before, but writing the eigenvalues to a ViennaCL-vector:
   **/
  viennacl::linalg::qr_method_sym(A_input2, Q, vcl_eigenvalues);

  /**
  *   Print the computed eigenvalues and eigenvectors:
  **/
  std::cout << std::endl << "Eigenvalues with std::vector<T>:" << std::endl;
  vector_print(eigenvalues);
  std::cout << "Eigenvalues with viennacl::vector<T>: " << std::endl << vcl_eigenvalues << std::endl;
  std::cout << std::endl << "Reference eigenvalues:" << std::endl;
  vector_print(eigenvalues_ref);
  std::cout << std::endl;
  std::cout << "Eigenvectors - each column is an eigenvector" << std::endl;
  std::cout << Q << std::endl;

  /**
  *  That's it. Print a success message and exit.
  **/
  std::cout << std::endl;
  std::cout << "------- Tutorial completed --------" << std::endl;
  std::cout << std::endl;

  return EXIT_SUCCESS;
}

