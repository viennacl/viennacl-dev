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



/** \file tests/src/scan.cpp  Tests inclusive and exclusive scan operations.
*   \test Tests inclusive and exclusive scan operations.
**/

/*
*
*   Test file for inclusive and exclusive scan
*
*/

//include basic scalar and vector types of ViennaCL
#include "viennacl/scalar.hpp"
#include "viennacl/vector.hpp"

// Some helper functions for this tutorial:
#include <iostream>
#include <limits>
#include <string>
#include <iomanip>


typedef int     ScalarType;

static void init_vector(viennacl::vector<ScalarType>& vcl_v)
{
  std::vector<ScalarType> v(vcl_v.size());
  for (std::size_t i = 0; i < v.size(); ++i)
    v[i] = ScalarType(i % 7 + 1);
  viennacl::copy(v, vcl_v);
}

static void test_scan_values(viennacl::vector<ScalarType> const & input,
                             viennacl::vector<ScalarType> & result,
                             bool is_inclusive_scan)
{
  std::vector<ScalarType> host_input(input.size());
  std::vector<ScalarType> host_result(result.size());

  viennacl::copy(input, host_input);
  viennacl::copy(result, host_result);

  ScalarType sum = 0;
  if (is_inclusive_scan)
  {
    for(viennacl::vcl_size_t i = 0; i < input.size(); i++)
    {
      sum += host_input[i];
      host_input[i] = sum;
    }
  }
  else
  {
    for(viennacl::vcl_size_t i = 0; i < input.size(); i++)
    {
      ScalarType tmp = host_input[i];
      host_input[i] = sum;
      sum += tmp;
    }
  }


  for(viennacl::vcl_size_t i = 0; i < input.size(); i++)
  {
    if (host_input[i] != host_result[i])
    {
      std::cout << "Fail at vector index " << i << std::endl;
      std::cout << " result[" << i << "] = " << host_result[i] << std::endl;
      std::cout << " Reference = " << host_input[i] << std::endl;
      if (i > 0)
      {
        std::cout << " previous result[" << i-1 << "] = " << host_result[i-1] << std::endl;
        std::cout << " previous Reference = " << host_input[i-1] << std::endl;
      }
      exit(EXIT_FAILURE);
    }
  }
  std::cout << "PASSED!" << std::endl;

}


static void test_scans(unsigned int sz)
{
  viennacl::vector<ScalarType> vec1(sz), vec2(sz);

  std::cout << "Initialize vector..." << std::endl;
  init_vector(vec1);


  // INCLUSIVE SCAN
  std::cout << " --- Inclusive scan ---" << std::endl;
  std::cout << "Separate vectors: ";
  viennacl::linalg::inclusive_scan(vec1, vec2);
  test_scan_values(vec1, vec2, true);

  std::cout << "In-place: ";
  vec2 = vec1;
  viennacl::linalg::inclusive_scan(vec2);
  test_scan_values(vec1, vec2, true);
  std::cout << "Inclusive scan tested successfully!" << std::endl << std::endl;



  std::cout << "Initialize vector..." << std::endl;
  init_vector(vec1);

  // EXCLUSIVE SCAN
  std::cout << " --- Exclusive scan ---" << std::endl;
  std::cout << "Separate vectors: ";
  viennacl::linalg::exclusive_scan(vec1, vec2);
  test_scan_values(vec1, vec2, false);

  std::cout << "In-place: ";
  vec2 = vec1;
  viennacl::linalg::exclusive_scan(vec2);
  test_scan_values(vec1, vec2, false);
  std::cout << "Exclusive scan tested successfully!" << std::endl << std::endl;

}

int main()
{

  std::cout << std::endl << "----TEST INCLUSIVE and EXCLUSIVE SCAN----" << std::endl << std::endl;
  std::cout << " //// Tiny vectors ////" << std::endl;
  test_scans(27);
  std::cout << " //// Small vectors ////" << std::endl;
  test_scans(298);
  std::cout << " //// Medium vectors ////" << std::endl;
  test_scans(12345);
  std::cout << " //// Large vectors ////" << std::endl;
  test_scans(123456);

  std::cout << std::endl <<"--------TEST SUCCESSFULLY COMPLETED----------" << std::endl;
}
