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
*   Test file for inclusive and exclusive scan
*
*/


#ifndef NDEBUG
  #define NDEBUG
#endif

//include basic scalar and vector types of ViennaCL
#include "viennacl/scalar.hpp"
#include "viennacl/vector.hpp"
#include "viennacl/compressed_matrix.hpp"
#include "viennacl/linalg/matrix_operations.hpp"

// Some helper functions for this tutorial:
#include <iostream>
#include <limits>
#include <string>
#include <iomanip>


typedef float     ScalarType;

#define EPS 0.0001


void vector_print(viennacl::vector<ScalarType>& v )
{
  for (unsigned int i = 0; i < v.size(); i++)
      std::cout << std::setprecision(0) << std::fixed << v(i) << "\t";
    std::cout << "\n";
}


void init_vector(viennacl::vector<ScalarType>& vcl_v)
{
    std::vector<ScalarType> v(vcl_v.size());
    for (unsigned int i = 0; i < v.size(); ++i)
      v[i] =  i;
    viennacl::copy(v, vcl_v);
}

void test_inclusive_scan_values(viennacl::vector<ScalarType> & vcl_vec)
{
  std::vector<ScalarType> vec(vcl_vec.size());
  viennacl::copy(vcl_vec, vec);
  for(int i = 1; i < vec.size(); i++)
  {
    ScalarType abs_error = std::fabs((float)i* ((float)i + 1.) / 2. - vec[i]);
    if (abs_error > EPS * vec[i])
    {
      std::cout << "Fail at vector index " << i << " Absolute error:  " << abs_error;
      std::cout << "\t Relative error:  " << std::setprecision(7) << abs_error / vec[i] * 100 << "%"<< std::endl;
      std::cout << "vec[" << i << "] = " << vec[i] << std::endl;
      exit(EXIT_FAILURE);
    }
  }

}


void test_exclusive_scan_values(viennacl::vector<ScalarType> & vcl_vec)
{
  std::vector<ScalarType> vec(vcl_vec.size());
  viennacl::copy(vcl_vec, vec);
  for(int i = 1; i < vec.size() - 1; i++)
  {
    ScalarType abs_error = std::fabs((float)i* ((float)i + 1.) / 2. - vec[i + 1]);
    if (abs_error > EPS * vec[i])
     {
      std::cout << "Fail at vector index " << i << " Absolute error:  " << abs_error;
      std::cout << "\t Relative error:  " << std::setprecision(7) << abs_error / vec[i] * 100 << "%"<< std::endl;
      std::cout << "vec[" << i << "] = " << vec[i] << std::endl;
      exit(EXIT_FAILURE);
    }
  }

}



void test_scans()
{
  unsigned int sz = 100000;
  viennacl::vector<ScalarType> vec1(sz), vec2(sz);


  std::cout << "Initialize vector..." << std::endl;
  init_vector(vec1);


  // INCLUSIVE SCAN
  std::cout << "Inclusive scan started!" << std::endl;
  viennacl::linalg::inclusive_scan(vec1, vec2);
  std::cout << "Inclusive scan finished!" << std::endl;
  //vector_print(vec2);
  std::cout << "Testing inclusive scan results..." << std::endl;
  test_inclusive_scan_values(vec2);
  std::cout << "Inclusive scan tested successfully!" << std::endl << std::endl;



  std::cout << "Initialize vector..." << std::endl;
  init_vector(vec1);

  // EXCLUSIVE SCAN
  std::cout << "Exlusive scan started!" << std::endl;
  viennacl::linalg::exclusive_scan(vec1, vec2);
  std::cout << "Exclusive scan finished!" << std::endl;
  //vector_print(vec2);
  std::cout << "Testing exclusive scan results..."  << std::endl;
  test_exclusive_scan_values(vec2);
  std::cout << "Exclusive scan tested successfully!" << std::endl << std::endl;

}

int main()
{

  std::cout << std::endl << "----TEST INCLUSIVE and EXCLUSIVE SCAN----" << std::endl << std::endl;
  test_scans();

  std::cout << std::endl <<"--------TEST SUCCESSFULLY COMPLETED----------" << std::endl;
}
