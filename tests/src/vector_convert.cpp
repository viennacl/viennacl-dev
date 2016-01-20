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


/** \file tests/src/vector_convert.cpp  Tests conversion between vectors with different numeric type
*   \test  Tests conversion between vectors with different numeric type
**/

//
// *** System
//
#include <iostream>
#include <iomanip>
#include <vector>


//
// *** ViennaCL
//
//#define VIENNACL_DEBUG_ALL
#include "viennacl/vector.hpp"
#include "viennacl/vector_proxy.hpp"


template<typename NumericT, typename VectorT>
int check(std::vector<NumericT> const & std_dest, std::size_t start_dest, std::size_t inc_dest, std::size_t size_dest,
          VectorT const & vcl_dest)
{
  std::vector<NumericT> tempvec(vcl_dest.size());
  viennacl::copy(vcl_dest, tempvec);

  for (std::size_t i=0; i < size_dest; ++i)
  {
    if (   std_dest[start_dest + i * inc_dest] < tempvec[i]
        || std_dest[start_dest + i * inc_dest] > tempvec[i])
    {
      std::cerr << "Failure at index " << i << ": STL value " << std_dest[start_dest + i * inc_dest] << ", ViennaCL value " << tempvec[i] << std::endl;
      return EXIT_FAILURE;
    }
  }

  return EXIT_SUCCESS;
}


//
// -------------------------------------------------------------
//
template<typename STLVectorT1, typename STLVectorT2, typename ViennaCLVectorT1, typename ViennaCLVectorT2 >
int test(STLVectorT1 & std_src,  std::size_t start_src,  std::size_t inc_src,  std::size_t size_src,
         STLVectorT2 & std_dest, std::size_t start_dest, std::size_t inc_dest, std::size_t size_dest,
         ViennaCLVectorT1 const & vcl_src, ViennaCLVectorT2 & vcl_dest)
{
  assert(size_src       == size_dest       && bool("Size mismatch for STL vectors"));
  assert(vcl_src.size() == vcl_dest.size() && bool("Size mismatch for ViennaCL vectors"));
  assert(size_src       == vcl_src.size()  && bool("Size mismatch for STL and ViennaCL vectors"));

  typedef typename STLVectorT2::value_type  DestNumericT;

  for (std::size_t i=0; i<size_src; ++i)
    std_dest[start_dest + i * inc_dest] = static_cast<DestNumericT>(std_src[start_src + i * inc_src]);

  vcl_dest = vcl_src; // here is the conversion taking place

  if (check(std_dest, start_dest, inc_dest, size_dest, vcl_dest) != EXIT_SUCCESS)
    return EXIT_FAILURE;

  viennacl::vector<DestNumericT> x(vcl_src);
  if (check(std_dest, start_dest, inc_dest, size_dest, x) != EXIT_SUCCESS)
    return EXIT_FAILURE;

  // --------------------------------------------------------------------------
  return EXIT_SUCCESS;
}

inline std::string type_string(unsigned int)    { return "unsigned int"; }
inline std::string type_string(int)             { return "int"; }
inline std::string type_string(unsigned long)   { return "unsigned long"; }
inline std::string type_string(long)            { return "long"; }
inline std::string type_string(float)           { return "float"; }
inline std::string type_string(double)          { return "double"; }

template<typename FromNumericT, typename ToNumericT>
int test()
{
  int retval = EXIT_SUCCESS;

  std::cout << std::endl;
  std::cout << "-----------------------------------------------" << std::endl;
  std::cout << std::endl;
  std::cout << "Conversion test from " << type_string(FromNumericT()) << " to " << type_string(ToNumericT()) << std::endl;
  std::cout << std::endl;

  std::size_t full_size  = 12345;
  std::size_t small_size = full_size / 4;
  //
  // Set up STL objects
  //
  std::vector<FromNumericT> std_src(full_size);
  std::vector<ToNumericT>   std_dest(std_src.size());

  for (std::size_t i=0; i<std_src.size(); ++i)
    std_src[i]  = FromNumericT(1.0) + FromNumericT(i);

  //
  // Set up ViennaCL objects
  //
  viennacl::vector<FromNumericT> vcl_src(std_src.size());
  viennacl::vector<ToNumericT>   vcl_dest(std_src.size());

  viennacl::copy(std_src, vcl_src);

  viennacl::vector<FromNumericT> vcl_src_small(small_size);
  viennacl::copy(std_src.begin(), std_src.begin() + typename std::vector<FromNumericT>::difference_type(small_size), vcl_src_small.begin());
  viennacl::vector<ToNumericT> vcl_dest_small(small_size);

  std::size_t r1_start = 1 +     vcl_src.size() / 4;
  std::size_t r1_stop  = 1 + 2 * vcl_src.size() / 4;
  viennacl::range vcl_r1(r1_start, r1_stop);

  std::size_t r2_start = 2 * vcl_src.size() / 4;
  std::size_t r2_stop  = 3 * vcl_src.size() / 4;
  viennacl::range vcl_r2(r2_start, r2_stop);

  viennacl::vector_range< viennacl::vector<FromNumericT> > vcl_range_src(vcl_src, vcl_r1);
  viennacl::vector_range< viennacl::vector<ToNumericT> >   vcl_range_dest(vcl_dest, vcl_r2);



  std::size_t s1_start = 1 + vcl_src.size() / 5;
  std::size_t s1_inc   = 3;
  std::size_t s1_size  = vcl_src.size() / 4;
  viennacl::slice vcl_s1(s1_start, s1_inc, s1_size);

  std::size_t s2_start = 2 * vcl_dest.size() / 5;
  std::size_t s2_inc   = 2;
  std::size_t s2_size  = vcl_dest.size() / 4;
  viennacl::slice vcl_s2(s2_start, s2_inc, s2_size);

  viennacl::vector_slice< viennacl::vector<FromNumericT> > vcl_slice_src(vcl_src, vcl_s1);
  viennacl::vector_slice< viennacl::vector<ToNumericT> >   vcl_slice_dest(vcl_dest, vcl_s2);

  //
  // Now start running tests for vectors, ranges and slices:
  //

  std::cout << " ** vcl_src = vector, vcl_dest = vector **" << std::endl;
  retval = test(std_src,  0, 1, std_src.size(),
                std_dest, 0, 1, std_dest.size(),
                vcl_src, vcl_dest);
  if (retval != EXIT_SUCCESS)
    return EXIT_FAILURE;

  std::cout << " ** vcl_src = vector, vcl_dest = range **" << std::endl;
  retval = test(std_src,  0, 1, small_size,
                std_dest, r2_start, 1, r2_stop - r2_start,
                vcl_src_small, vcl_range_dest);
  if (retval != EXIT_SUCCESS)
    return EXIT_FAILURE;

  std::cout << " ** vcl_src = vector, vcl_dest = slice **" << std::endl;
  retval = test(std_src,  0, 1, small_size,
                std_dest, s2_start, s2_inc, s2_size,
                vcl_src_small, vcl_slice_dest);
  if (retval != EXIT_SUCCESS)
    return EXIT_FAILURE;

  ///////

  std::cout << " ** vcl_src = range, vcl_dest = vector **" << std::endl;
  retval = test(std_src,  r1_start, 1, r1_stop - r1_start,
                std_dest, 0, 1, small_size,
                vcl_range_src, vcl_dest_small);
  if (retval != EXIT_SUCCESS)
    return EXIT_FAILURE;

  std::cout << " ** vcl_src = range, vcl_dest = range **" << std::endl;
  retval = test(std_src,  r1_start, 1, r1_stop - r1_start,
                std_dest, r2_start, 1, r2_stop - r2_start,
                vcl_range_src, vcl_range_dest);
  if (retval != EXIT_SUCCESS)
    return EXIT_FAILURE;

  std::cout << " ** vcl_src = range, vcl_dest = slice **" << std::endl;
  retval = test(std_src,  r1_start, 1, r1_stop - r1_start,
                std_dest, s2_start, s2_inc, s2_size,
                vcl_range_src, vcl_slice_dest);
  if (retval != EXIT_SUCCESS)
    return EXIT_FAILURE;

  ///////

  std::cout << " ** vcl_src = slice, vcl_dest = vector **" << std::endl;
  retval = test(std_src,  s1_start, s1_inc, s1_size,
                std_dest, 0, 1, small_size,
                vcl_slice_src, vcl_dest_small);
  if (retval != EXIT_SUCCESS)
    return EXIT_FAILURE;

  std::cout << " ** vcl_src = slice, vcl_dest = range **" << std::endl;
  retval = test(std_src,  s1_start, s1_inc, s1_size,
                std_dest, r2_start, 1, r2_stop - r2_start,
                vcl_slice_src, vcl_range_dest);
  if (retval != EXIT_SUCCESS)
    return EXIT_FAILURE;

  std::cout << " ** vcl_src = slice, vcl_dest = slice **" << std::endl;
  retval = test(std_src,  s1_start, s1_inc, s1_size,
                std_dest, s2_start, s2_inc, s2_size,
                vcl_slice_src, vcl_slice_dest);
  if (retval != EXIT_SUCCESS)
    return EXIT_FAILURE;

  return EXIT_SUCCESS;
}



//
// -------------------------------------------------------------
//
int main()
{
  std::cout << std::endl;
  std::cout << "----------------------------------------------" << std::endl;
  std::cout << "----------------------------------------------" << std::endl;
  std::cout << "## Test :: Type conversion test for vectors   " << std::endl;
  std::cout << "----------------------------------------------" << std::endl;
  std::cout << "----------------------------------------------" << std::endl;
  std::cout << std::endl;

  int retval = EXIT_SUCCESS;

  //
  // from int
  //
  retval = test<int, int>();
  if ( retval == EXIT_SUCCESS ) std::cout << "# Test passed" << std::endl;
  else return retval;

  retval = test<int, unsigned int>();
  if ( retval == EXIT_SUCCESS ) std::cout << "# Test passed" << std::endl;
  else return retval;

  retval = test<int, long>();
  if ( retval == EXIT_SUCCESS ) std::cout << "# Test passed" << std::endl;
  else return retval;

  retval = test<int, unsigned long>();
  if ( retval == EXIT_SUCCESS ) std::cout << "# Test passed" << std::endl;
  else return retval;

  retval = test<int, float>();
  if ( retval == EXIT_SUCCESS ) std::cout << "# Test passed" << std::endl;
  else return retval;

  retval = test<int, unsigned long>();
  if ( retval == EXIT_SUCCESS ) std::cout << "# Test passed" << std::endl;
  else return retval;

#ifdef VIENNACL_WITH_OPENCL
  if ( viennacl::ocl::current_device().double_support() )
#endif
  {
    retval = test<int, double>();
    if ( retval == EXIT_SUCCESS ) std::cout << "# Test passed" << std::endl;
    else return retval;
  }


  //
  // from unsigned int
  //
  retval = test<unsigned int, int>();
  if ( retval == EXIT_SUCCESS ) std::cout << "# Test passed" << std::endl;
  else return retval;

  retval = test<unsigned int, unsigned int>();
  if ( retval == EXIT_SUCCESS ) std::cout << "# Test passed" << std::endl;
  else return retval;

  retval = test<unsigned int, long>();
  if ( retval == EXIT_SUCCESS ) std::cout << "# Test passed" << std::endl;
  else return retval;

  retval = test<unsigned int, unsigned long>();
  if ( retval == EXIT_SUCCESS ) std::cout << "# Test passed" << std::endl;
  else return retval;

  retval = test<unsigned int, float>();
  if ( retval == EXIT_SUCCESS ) std::cout << "# Test passed" << std::endl;
  else return retval;

  retval = test<unsigned int, unsigned long>();
  if ( retval == EXIT_SUCCESS ) std::cout << "# Test passed" << std::endl;
  else return retval;

#ifdef VIENNACL_WITH_OPENCL
  if ( viennacl::ocl::current_device().double_support() )
#endif
  {
    retval = test<unsigned int, double>();
    if ( retval == EXIT_SUCCESS ) std::cout << "# Test passed" << std::endl;
    else return retval;
  }


  //
  // from long
  //
  retval = test<long, int>();
  if ( retval == EXIT_SUCCESS ) std::cout << "# Test passed" << std::endl;
  else return retval;

  retval = test<long, unsigned int>();
  if ( retval == EXIT_SUCCESS ) std::cout << "# Test passed" << std::endl;
  else return retval;

  retval = test<long, long>();
  if ( retval == EXIT_SUCCESS ) std::cout << "# Test passed" << std::endl;
  else return retval;

  retval = test<long, unsigned long>();
  if ( retval == EXIT_SUCCESS ) std::cout << "# Test passed" << std::endl;
  else return retval;

  retval = test<long, float>();
  if ( retval == EXIT_SUCCESS ) std::cout << "# Test passed" << std::endl;
  else return retval;

  retval = test<long, unsigned long>();
  if ( retval == EXIT_SUCCESS ) std::cout << "# Test passed" << std::endl;
  else return retval;

#ifdef VIENNACL_WITH_OPENCL
  if ( viennacl::ocl::current_device().double_support() )
#endif
  {
    retval = test<long, double>();
    if ( retval == EXIT_SUCCESS ) std::cout << "# Test passed" << std::endl;
    else return retval;
  }


  //
  // from unsigned long
  //
  retval = test<unsigned long, int>();
  if ( retval == EXIT_SUCCESS ) std::cout << "# Test passed" << std::endl;
  else return retval;

  retval = test<unsigned long, unsigned int>();
  if ( retval == EXIT_SUCCESS ) std::cout << "# Test passed" << std::endl;
  else return retval;

  retval = test<unsigned long, long>();
  if ( retval == EXIT_SUCCESS ) std::cout << "# Test passed" << std::endl;
  else return retval;

  retval = test<unsigned long, unsigned long>();
  if ( retval == EXIT_SUCCESS ) std::cout << "# Test passed" << std::endl;
  else return retval;

  retval = test<unsigned long, float>();
  if ( retval == EXIT_SUCCESS ) std::cout << "# Test passed" << std::endl;
  else return retval;

  retval = test<unsigned long, unsigned long>();
  if ( retval == EXIT_SUCCESS ) std::cout << "# Test passed" << std::endl;
  else return retval;

#ifdef VIENNACL_WITH_OPENCL
  if ( viennacl::ocl::current_device().double_support() )
#endif
  {
    retval = test<unsigned long, double>();
    if ( retval == EXIT_SUCCESS ) std::cout << "# Test passed" << std::endl;
    else return retval;
  }

  //
  // from float
  //
  retval = test<float, int>();
  if ( retval == EXIT_SUCCESS ) std::cout << "# Test passed" << std::endl;
  else return retval;

  retval = test<float, unsigned int>();
  if ( retval == EXIT_SUCCESS ) std::cout << "# Test passed" << std::endl;
  else return retval;

  retval = test<float, long>();
  if ( retval == EXIT_SUCCESS ) std::cout << "# Test passed" << std::endl;
  else return retval;

  retval = test<float, unsigned long>();
  if ( retval == EXIT_SUCCESS ) std::cout << "# Test passed" << std::endl;
  else return retval;

  retval = test<float, float>();
  if ( retval == EXIT_SUCCESS ) std::cout << "# Test passed" << std::endl;
  else return retval;

  retval = test<float, unsigned long>();
  if ( retval == EXIT_SUCCESS ) std::cout << "# Test passed" << std::endl;
  else return retval;

#ifdef VIENNACL_WITH_OPENCL
  if ( viennacl::ocl::current_device().double_support() )
#endif
  {
    retval = test<float, double>();
    if ( retval == EXIT_SUCCESS ) std::cout << "# Test passed" << std::endl;
    else return retval;
  }

  //
  // from double
  //
#ifdef VIENNACL_WITH_OPENCL
  if ( viennacl::ocl::current_device().double_support() )
#endif
  {
    retval = test<double, int>();
    if ( retval == EXIT_SUCCESS ) std::cout << "# Test passed" << std::endl;
    else return retval;

    retval = test<double, unsigned int>();
    if ( retval == EXIT_SUCCESS ) std::cout << "# Test passed" << std::endl;
    else return retval;

    retval = test<double, long>();
    if ( retval == EXIT_SUCCESS ) std::cout << "# Test passed" << std::endl;
    else return retval;

    retval = test<double, unsigned long>();
    if ( retval == EXIT_SUCCESS ) std::cout << "# Test passed" << std::endl;
    else return retval;

    retval = test<double, float>();
    if ( retval == EXIT_SUCCESS ) std::cout << "# Test passed" << std::endl;
    else return retval;

    retval = test<double, unsigned long>();
    if ( retval == EXIT_SUCCESS ) std::cout << "# Test passed" << std::endl;
    else return retval;

    retval = test<double, double>();
    if ( retval == EXIT_SUCCESS ) std::cout << "# Test passed" << std::endl;
    else return retval;
  }


  std::cout << std::endl;
  std::cout << "------- Test completed --------" << std::endl;
  std::cout << std::endl;

  return retval;
}
