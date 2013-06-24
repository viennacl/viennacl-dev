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
*   Tutorial: BLAS level 1 functionality (blas1.cpp and blas1.cu are identical, the latter being required for compilation using CUDA nvcc)
*
*/


// include necessary system headers
#include <iostream>

//include basic scalar and vector types of ViennaCL
#include "viennacl/scalar.hpp"
#include "viennacl/vector.hpp"

//include the generic inner product functions of ViennaCL
#include "viennacl/linalg/inner_prod.hpp"

//include the generic norm functions of ViennaCL
#include "viennacl/linalg/norm_1.hpp"
#include "viennacl/linalg/norm_2.hpp"
#include "viennacl/linalg/norm_inf.hpp"

// Some helper functions for this tutorial:
#include "Random.hpp"

int main()
{
  //Change this type definition to double if your gpu supports that
  typedef float       ScalarType;

  /////////////////////////////////////////////////
  ///////////// Scalar operations /////////////////
  /////////////////////////////////////////////////

  //
  // Define a few CPU scalars:
  //
  ScalarType s1 = ScalarType(3.1415926);   //note: writing ScalarType s1 = 3.1415926; leads to warnings with some compilers if ScalarType is 'float'.
  ScalarType s2 = ScalarType(2.71763);
  ScalarType s3 = ScalarType(42.0);

  //
  // ViennaCL scalars are defined in the same way:
  //
  //std::cout << "Creating a few scalars..." << std::endl;
  viennacl::scalar<ScalarType> vcl_s1;
  viennacl::scalar<ScalarType> vcl_s2 = ScalarType(1.0);
  viennacl::scalar<ScalarType> vcl_s3 = ScalarType(1.0);

  //
  // CPU scalars can be transparently assigned to GPU scalars and vice versa:
  //
  std::cout << "Copying a few scalars..." << std::endl;
  vcl_s1 = s1;
  s2 = vcl_s2;
  vcl_s3 = s3;

  //
  // Operations between GPU scalars work just as for CPU scalars:
  // (Note that such single compute kernels on the GPU are considerably slower than on the CPU)
  //

  std::cout << "Manipulating a few scalars..." << std::endl;
  std::cout << "operator +=" << std::endl;
  s1 += s2;
  vcl_s1 += vcl_s2;

  /*std::cout << "operator *=" << std::endl;
  s1 *= s2;
  vcl_s1 *= vcl_s2;

  std::cout << "operator -=" << std::endl;
  s1 -= s2;
  vcl_s1 -= vcl_s2;

  std::cout << "operator /=" << std::endl;
  s1 /= s2;
  vcl_s1 /= vcl_s2;*/

  std::cout << "operator +" << std::endl;
  s1 = s2 + s3;
  vcl_s1 = vcl_s2 + vcl_s3;

  std::cout << "multiple operators" << std::endl;
  s1 = s2 + s3 * s2 - s3 / s1;
  vcl_s1 = vcl_s2 + vcl_s3 * vcl_s2 - vcl_s3 / vcl_s1;


  //
  // Operations can also be mixed:
  //

  std::cout << "mixed operations" << std::endl;
  vcl_s1 = s1 * vcl_s2 + s3 - vcl_s3;


  //
  // Output stream is overloaded as well:
  //

  std::cout << "CPU scalar s3: " << s3 << std::endl;
  std::cout << "GPU scalar vcl_s3: " << vcl_s3 << std::endl;


  /////////////////////////////////////////////////
  ///////////// Vector operations /////////////////
  /////////////////////////////////////////////////

  //
  // Define a few vectors (from STL and plain C) and viennacl::vectors
  //
  std::vector<ScalarType>      std_vec1(10);
  std::vector<ScalarType>      std_vec2(10);
  ScalarType                   plain_vec3[10];  //plain C array

  viennacl::vector<ScalarType> vcl_vec1(10);
  viennacl::vector<ScalarType> vcl_vec2(10);
  viennacl::vector<ScalarType> vcl_vec3(10);

  //
  // Let us fill the CPU vectors with random values:
  // (random<> is a helper function from Random.hpp)
  //

  for (unsigned int i = 0; i < 10; ++i)
  {
    std_vec1[i] = random<ScalarType>();
    vcl_vec2(i) = random<ScalarType>();  //also works for GPU vectors, but is MUCH slower (approx. factor 10.000) than the CPU analogue
    plain_vec3[i] = random<ScalarType>();
  }

  //
  // Copy the CPU vectors to the GPU vectors and vice versa
  //
  viennacl::copy(std_vec1.begin(), std_vec1.end(), vcl_vec1.begin()); //either the STL way
  viennacl::copy(vcl_vec2.begin(), vcl_vec2.end(), std_vec2.begin()); //either the STL way
  viennacl::copy(vcl_vec2, std_vec2);                                 //using the short hand notation for objects that provide .begin() and .end() members
  viennacl::copy(vcl_vec2.begin(), vcl_vec2.end(), plain_vec3);       //copy to plain C vector

  //
  // Also partial copies by providing the corresponding iterators are possible:
  //
  viennacl::copy(std_vec1.begin() + 4, std_vec1.begin() + 8, vcl_vec1.begin() + 4);   //cpu to gpu
  viennacl::copy(vcl_vec1.begin() + 4, vcl_vec1.begin() + 8, vcl_vec2.begin() + 1);   //gpu to gpu
  viennacl::copy(vcl_vec1.begin() + 4, vcl_vec1.begin() + 8, std_vec1.begin() + 1);   //gpu to cpu

  //
  // Compute the inner product of two GPU vectors and write the result to either CPU or GPU
  //

  vcl_s1 = viennacl::linalg::inner_prod(vcl_vec1, vcl_vec2);
  s1 = viennacl::linalg::inner_prod(vcl_vec1, vcl_vec2);
  s2 = viennacl::linalg::inner_prod(std_vec1, std_vec2); //inner prod can also be used with std::vector (computations are carried out on CPU then)

  //
  // Compute norms:
  //

  s1 = viennacl::linalg::norm_1(vcl_vec1);
  vcl_s2 = viennacl::linalg::norm_2(vcl_vec2);
  s3 = viennacl::linalg::norm_inf(vcl_vec3);

  //
  // Plane rotation of two vectors:
  //

  viennacl::linalg::plane_rotation(vcl_vec1, vcl_vec2, 1.1f, 2.3f);

  //
  // Use viennacl::vector via the overloaded operators just as you would write it on paper:
  //

  //simple expression:
  vcl_vec1 = vcl_s1 * vcl_vec2 / vcl_s3;

  //more complicated expression:
  vcl_vec1 = vcl_vec2 / vcl_s3 + vcl_s2 * (vcl_vec1 - vcl_s2 * vcl_vec2);

  //
  // Swap the content of two vectors without a temporary vector:
  //

  viennacl::swap(vcl_vec1, vcl_vec2);  //swaps all entries in memory
  viennacl::fast_swap(vcl_vec1, vcl_vec2); //swaps OpenCL memory handles only

  //
  // The vectors can also be cleared directly:
  //
  vcl_vec1.clear();
  vcl_vec2.clear();

  //
  //  That's it.
  //
  std::cout << "!!!! TUTORIAL COMPLETED SUCCESSFULLY !!!!" << std::endl;

  return EXIT_SUCCESS;
}

