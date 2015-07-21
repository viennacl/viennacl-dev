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

/** \example fft.cpp
*
*   This tutorial showcasts FFT functionality.
*
*   \note The FFT module is experimental in ViennaCL. The API might change in future versions.
*
*   We start with including the respective headers:
**/

// include necessary system headers
#include <iostream>
#include <vector>
#include <cmath>
#include <complex>
#include <fstream>

// include basic scalar and vector types of ViennaCL
#include "viennacl/vector.hpp"
#include "viennacl/matrix.hpp"

// include FFT routines
#include "viennacl/fft.hpp"
#include "viennacl/linalg/fft_operations.hpp"

/**
*   In the main()-routine we create a few vectors and matrices and then run FFT on them.
**/
int main()
{
  // Feel free to change this type definition to double if your gpu supports that
  typedef float ScalarType;

  // Create vectors of eight complex values (represented as pairs of floating point values: [real_0, imag_0, real_1, imag_1, etc.])
  viennacl::vector<ScalarType> input_vec(16);
  viennacl::vector<ScalarType> output_vec(16);
  viennacl::vector<ScalarType> input2_vec(16);

  viennacl::matrix<ScalarType> m(4, 8);
  viennacl::matrix<ScalarType> o(4, 8);

  for (std::size_t i = 0; i < m.size1(); i++)
    for (std::size_t s = 0; s < m.size2(); s++)
      m(i, s) = ScalarType((i + s) / 2);

  /**
  *  Fill the vectors and matrices with values by using operator(). Use viennacl::copy() for larger data!
  **/
  for (std::size_t i = 0; i < input_vec.size(); ++i)
  {
    if (i % 2 == 0)
    {
      input_vec(i) = ScalarType(i / 2); // even indices represent real part
      input2_vec(i) = ScalarType(i / 2);
    } else
      input_vec(i) = 0;            // odd indices represent imaginary part
  }

  /**
  * Compute the FFT and store result in 'output_vec'
  **/
  std::cout << "Computing FFT Matrix" << std::endl;
  std::cout << "m: " << m << std::endl;
  std::cout << "o: " << o << std::endl;
  viennacl::fft(m, o);
  std::cout << "Done" << std::endl;
  std::cout << "m: " << m << std::endl;
  std::cout << "o: " << o << std::endl;
  std::cout << "Transpose" << std::endl;

  viennacl::linalg::transpose(m, o);
  //viennacl::linalg::transpose(m);
  std::cout << "m: " << m << std::endl;
  std::cout << "o: " << o << std::endl;

  std::cout << "---------------------" << std::endl;

  /**
  *  Compute the FFT using the Bluestein algorithm (usually faster, but higher memory footprint)
  **/
  std::cout << "Computing FFT bluestein" << std::endl;
  // Print the vector
  std::cout << "input_vec: " << input_vec << std::endl;
  std::cout << "Done" << std::endl;
  viennacl::linalg::bluestein(input_vec, output_vec, 0);
  std::cout << "input_vec: " << input_vec << std::endl;
  std::cout << "output_vec: " << output_vec << std::endl;
  std::cout << "---------------------" << std::endl;

  /**
  *  Computing the standard radix-FFT for a vector
  **/
  std::cout << "Computing FFT " << std::endl;
  // Print the vector
  std::cout << "input_vec: " << input_vec << std::endl;
  std::cout << "Done" << std::endl;
  viennacl::fft(input_vec, output_vec);
  std::cout << "input_vec: " << input_vec << std::endl;
  std::cout << "output_vec: " << output_vec << std::endl;
  std::cout << "---------------------" << std::endl;

  /**
  *  Computing the standard inverse radix-FFT for a vector
  **/
  std::cout << "Computing inverse FFT..." << std::endl;
  //viennacl::ifft(output_vec, input_vec); // either store result into output_vec
  viennacl::inplace_ifft(output_vec);     // or compute in-place
  std::cout << "input_vec: " << input_vec << std::endl;
  std::cout << "output_vec: " << output_vec << std::endl;
  std::cout << "---------------------" << std::endl;

  /**
  *  Convert a real vector to an interleaved complex vector and back.
  *  Entries with even indices represent real parts, odd indices imaginary parts.
  **/
  std::cout << "Computing real to complex..." << std::endl;
  std::cout << "input_vec: " << input_vec << std::endl;
  viennacl::linalg::real_to_complex(input_vec, output_vec, input_vec.size() / 2); // or compute in-place
  std::cout << "output_vec: " << output_vec << std::endl;
  std::cout << "---------------------" << std::endl;

  std::cout << "Computing complex to real..." << std::endl;
  std::cout << "input_vec: " << input_vec << std::endl;
  //viennacl::ifft(output_vec, input_vec); // either store result into output_vec
  viennacl::linalg::complex_to_real(input_vec, output_vec, input_vec.size() / 2); // or compute in-place
  std::cout << "output_vec: " << output_vec << std::endl;
  std::cout << "---------------------" << std::endl;

  /**
  *  Point-wise multiplication of two complex vectors.
  **/
  std::cout << "Computing multiply complex" << std::endl;
  // Print the vector
  std::cout << "input_vec: " << input_vec << std::endl;
  std::cout << "input2_vec: " << input2_vec << std::endl;
  viennacl::linalg::multiply_complex(input_vec, input2_vec, output_vec);
  std::cout << "Done" << std::endl;
  std::cout << "output_vec: " << output_vec << std::endl;
  std::cout << "---------------------" << std::endl;

  /**
  *  That's it.
  **/
  std::cout << "!!!! TUTORIAL COMPLETED SUCCESSFULLY !!!!" << std::endl;
  return EXIT_SUCCESS;
}
