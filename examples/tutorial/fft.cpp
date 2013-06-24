
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
*   Tutorial: FFT functionality (experimental)
*
*/


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

int main()
{
  // Change this type definition to double if your gpu supports that
  typedef float       ScalarType;

  // Create vectors of eight complex values (represented as pairs of floating point values: [real_0, imag_0, real_1, imag_1, etc.])
  viennacl::vector<ScalarType> input_vec(16);
  viennacl::vector<ScalarType> output_vec(16);

  // Fill with values (use viennacl::copy() for larger data!)
  for (std::size_t i=0; i<input_vec.size(); ++i)
  {
    if (i%2 == 0)
      input_vec(i) = ScalarType(i/2);  // even indices represent real part
    else
      input_vec(i) = 0;                // odd indices represent imaginary part
  }

  // Print the vector
  std::cout << "input_vec: " << input_vec << std::endl;

  // Compute FFT and store result in 'output_vec'
  std::cout << "Computing FFT..." << std::endl;
  viennacl::fft(input_vec, output_vec);

  // Compute FFT and store result directly in 'input_vec'
  viennacl::inplace_fft(input_vec);

  // Print result
  std::cout << "input_vec: " << input_vec << std::endl;
  std::cout << "output_vec: " << output_vec << std::endl;

  std::cout << "Computing inverse FFT..." << std::endl;
  viennacl::ifft(input_vec, output_vec); // either store result into output_vec
  viennacl::inplace_ifft(input_vec);     // or compute in-place

  std::cout << "input_vec: " << input_vec << std::endl;
  std::cout << "output_vec: " << output_vec << std::endl;

  //
  //  That's it.
  //
  std::cout << "!!!! TUTORIAL COMPLETED SUCCESSFULLY !!!!" << std::endl;

  return EXIT_SUCCESS;
}
