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


/** \example bisect.cpp
*
* This tutorial shows how the eigenvalues of a symmetric, tridiagonal matrix can be computed using bisection.
* We begin with the usual header inclusions:
**/

// System headers
#include <stdlib.h>
#include <stdio.h>
#include <string.h>


// ViennaCL headers

#include "viennacl/scalar.hpp"
#include "viennacl/vector.hpp"
#include "viennacl/matrix.hpp"

#include "viennacl/linalg/bisect_gpu.hpp"



/**
* The first step is to generate a suitable input tridiagonal input matrix in the function initInputData():
**/

////////////////////////////////////////////////////////////////////////////////
/// \brief initInputData   Initialize the diagonal and superdiagonal elements of
///                        the matrix. Can be filled with random values or with
///                        repeating values.
/// \param diagonal        diagonal elements of the matrix
/// \param superdiagonal   superdiagonal elements of the matrix
/// \param mat_size        Dimension of the matrix
///
template <typename NumericT>
void initInputData(std::vector<NumericT> &diagonal, std::vector<NumericT> &superdiagonal, const unsigned int mat_size)
{

  srand(278217421);
  bool randomValues = false;


  if(randomValues == true)
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
    for(unsigned int i = 0; i < mat_size; ++i)
    {
       diagonal[i] = ((NumericT)(i % 8)) - 4.5f;
       superdiagonal[i] = ((NumericT)(i % 5)) - 4.5f;
    }
  }
  // the first element of s is used as padding on the device (thus the
  // whole vector is copied to the device but the kernels are launched
  // with (s+1) as start address
  superdiagonal[0] = 0.0f;
}


/**
* The main program is now as follows:
**/
int main()
{
    typedef float NumericT;

    bool bResult = false;
    unsigned int mat_size = 30;

/**
*   Create STL-vectors holding the diagonal, the superdiagonal, and the computed eigenvalues:
**/
    std::vector<NumericT> diagonal(mat_size);
    std::vector<NumericT> superdiagonal(mat_size);
    std::vector<NumericT> eigenvalues_bisect(mat_size);

/**
* Initialize the data with the helper routine defined earlier:
**/
    initInputData(diagonal, superdiagonal, mat_size);


/**
* Run the bisection algorithm for the provided input
**/
    std::cout << "Start the bisection algorithm" << std::endl;
    bResult = viennacl::linalg::bisect(diagonal, superdiagonal, eigenvalues_bisect);
    std::cout << std::endl << "---TUTORIAL COMPLETED---" << std::endl;

/**
* Uncomment the following code to also have the results printed:
**/
/*
    // ------------Print the results---------------
    std::cout << "mat_size = " << mat_size << std::endl;
    for (unsigned int i = 0; i < mat_size; ++i)
    {
      std::cout << "Eigenvalue " << i << ": " << std::setprecision(8) << eigenvalues_bisect[i] << std::endl;
    }
*/
    exit(bResult == true ? EXIT_SUCCESS : EXIT_FAILURE);
}
