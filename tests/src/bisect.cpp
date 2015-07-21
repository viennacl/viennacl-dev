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

/** \file  tests/src/bisect.cpp  Computation of eigenvalues of a symmetric, tridiagonal matrix using bisection.
*   \test  Tests the bisection implementation for symmetric tridiagonal matrices.
**/

// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>


// includes, project

#include "viennacl/scalar.hpp"
#include "viennacl/vector.hpp"


#include "viennacl/linalg/bisect.hpp"
#include "viennacl/linalg/bisect_gpu.hpp"
#include "viennacl/linalg/tql2.hpp"

#define EPS 10.0e-4

typedef float NumericT;
////////////////////////////////////////////////////////////////////////////////
// declaration, forward
bool runTest(unsigned int mat_size);

////////////////////////////////////////////////////////////////////////////////
/// \brief initInputData   Initialize the diagonal and superdiagonal elements of
///                        the matrix
/// \param diagonal        diagonal elements of the matrix
/// \param superdiagonal   superdiagonal elements of the matrix
/// \param mat_size        Dimension of the matrix
///
template<typename NumericT>
void initInputData(std::vector<NumericT> &diagonal, std::vector<NumericT> &superdiagonal, unsigned int mat_size)
{
 
  srand(278217421);

#define RANDOM_VALUES false

  if (RANDOM_VALUES == true)
  {
    // Initialize diagonal and superdiagonal elements with random values
    for (unsigned int i = 0; i < mat_size; ++i)
    {
        diagonal[i] =      static_cast<NumericT>(2.0 * (((double)rand()
                                     / (double) RAND_MAX) - 0.5));
        superdiagonal[i] = static_cast<NumericT>(2.0 * (((double)rand()
                                     / (double) RAND_MAX) - 0.5));
    }
  }
  else
  { 
    // Initialize diagonal and superdiagonal elements with modulo values
    // This will cause in many multiple eigenvalues.
    for (unsigned int i = 0; i < mat_size; ++i)
    {
       diagonal[i] = ((NumericT)(i % 3)) - 4.5f;
       superdiagonal[i] = ((NumericT)(i % 3)) - 5.5f;
    }
  }
  // the first element of s is used as padding on the device (thus the
  // whole vector is copied to the device but the kernels are launched
  // with (s+1) as start address
  superdiagonal[0] = 0.0f; 
}


////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int main()
{
    bool test_result = false;

    // run test for large matrix
    test_result = runTest(550);
    if(test_result == true)
    {
      std::cout << "First Test Succeeded!" << std::endl << std::endl;
    }
    else
    {
      std::cout << "---TEST FAILED---" << std::endl;
      return EXIT_FAILURE;
    }

    // run test for small matrix
    test_result = runTest(96);

    if(test_result == true)
   {
      std::cout << std::endl << "---TEST SUCCESSFULLY COMPLETED---" << std::endl;
      return EXIT_SUCCESS;
    }
    else
    {
      std::cout << "---TEST FAILED---" << std::endl;
      return EXIT_FAILURE;
    }
}

////////////////////////////////////////////////////////////////////////////////
//! Run a simple test
////////////////////////////////////////////////////////////////////////////////
bool runTest(unsigned int mat_size)
{
    bool bResult = false;

    std::vector<NumericT> diagonal(mat_size);
    std::vector<NumericT> superdiagonal(mat_size);
    std::vector<NumericT> eigenvalues_bisect(mat_size);

    // -------------Initialize data-------------------
    // Fill the diagonal and superdiagonal elements of the vector
    initInputData(diagonal, superdiagonal, mat_size);

    // -------Start the bisection algorithm------------
    std::cout << "Start the bisection algorithm" << std::endl;
    std::cout << "Matrix size: " << mat_size << std::endl;
    bResult = viennacl::linalg::bisect(diagonal, superdiagonal, eigenvalues_bisect);
    // Exit if an error occured during the execution of the algorithm
    if (bResult == false)
     return false;

    // ---------------Check the results---------------
    // The results of the bisection algorithm will be checked with the tql algorithm
    // Initialize Data for tql1 algorithm
    std::vector<NumericT> diagonal_tql(mat_size);
    std::vector<NumericT> superdiagonal_tql(mat_size);
    diagonal_tql = diagonal;
    superdiagonal_tql = superdiagonal;

    // Start the tql algorithm
    std::cout << "Start the tql algorithm..." << std::endl;
    viennacl::linalg::tql1<NumericT>(mat_size, diagonal_tql, superdiagonal_tql);

    // Ensure that eigenvalues from tql1 algorithm are sorted in ascending order
    std::sort(diagonal_tql.begin(), diagonal_tql.end());

    // Compare the results from the bisection algorithm with the results
    // from the tql algorithm.
    std::cout << "Start comparison..." << std::endl;
    for (unsigned int i = 0; i < mat_size; i++)
    {
       if (std::abs(diagonal_tql[i] - eigenvalues_bisect[i]) > EPS)
       {
         std::cout << std::setprecision(12) << diagonal_tql[i] << "  != " << eigenvalues_bisect[i] << "\n";
         return false;
       }
    }
/*
    // ------------Print the results---------------
    std::cout << "mat_size = " << mat_size << std::endl;
    for (unsigned int i = 0; i < mat_size; ++i)
    {
      std::cout << "Eigenvalue " << i << ":  \tbisect: " << std::setprecision(14) << eigenvalues_bisect[i] << "\ttql: " << diagonal_tql[i] << std::endl;
    }
*/
  return bResult;
    
}
