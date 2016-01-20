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


/** \file tests/src/matrix_convert.cpp  Tests conversion between matrices with different numeric type
*   \test  Tests conversion between matrices with different numeric type
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
#include "viennacl/backend/memory.hpp"
#include "viennacl/matrix.hpp"
#include "viennacl/matrix_proxy.hpp"


template<typename NumericT, typename MatrixT>
int check(std::vector<NumericT> const & std_dest,
          std::size_t start1, std::size_t inc1, std::size_t size1,
          std::size_t start2, std::size_t inc2, std::size_t size2, std::size_t internal_size2,
          MatrixT const & vcl_dest)
{
  viennacl::backend::typesafe_host_array<NumericT> tempmat(vcl_dest.handle(), vcl_dest.internal_size());
  viennacl::backend::memory_read(vcl_dest.handle(), 0, tempmat.raw_size(), reinterpret_cast<NumericT*>(tempmat.get()));

  for (std::size_t i=0; i < size1; ++i)
  {
    for (std::size_t j=0; j < size2; ++j)
    {
      NumericT value_std  = std_dest[(i*inc1 + start1) * internal_size2 + (j*inc2 + start2)];
      NumericT value_dest = vcl_dest.row_major() ? tempmat[(i * vcl_dest.stride1() + vcl_dest.start1()) * vcl_dest.internal_size2() + (j * vcl_dest.stride2() + vcl_dest.start2())]
                                                 : tempmat[(i * vcl_dest.stride1() + vcl_dest.start1())                             + (j * vcl_dest.stride2() + vcl_dest.start2()) * vcl_dest.internal_size1()];

      if (value_std < value_dest || value_std > value_dest)
      {
        std::cerr << "Failure at row " << i << ", col " << j << ": STL value " << value_std << ", ViennaCL value " << value_dest << std::endl;
        return EXIT_FAILURE;
      }
    }
  }

  return EXIT_SUCCESS;
}


//
// -------------------------------------------------------------
//
template<typename STLVectorT1, typename STLVectorT2, typename ViennaCLVectorT1, typename ViennaCLVectorT2 >
int test(STLVectorT1 & std_src,  std::size_t start1_src,  std::size_t inc1_src,  std::size_t size1_src,  std::size_t start2_src,  std::size_t inc2_src,  std::size_t size2_src,  std::size_t internal_size2_src,
         STLVectorT2 & std_dest, std::size_t start1_dest, std::size_t inc1_dest, std::size_t size1_dest, std::size_t start2_dest, std::size_t inc2_dest, std::size_t size2_dest, std::size_t internal_size2_dest,
         ViennaCLVectorT1 const & vcl_src, ViennaCLVectorT2 & vcl_dest)
{
  assert(size1_src       == size1_dest       && bool("Size1 mismatch for STL matrices"));
  assert(size2_src       == size2_dest       && bool("Size2 mismatch for STL matrices"));
  assert(vcl_src.size1() == vcl_dest.size1() && bool("Size1 mismatch for ViennaCL matrices"));
  assert(vcl_src.size2() == vcl_dest.size2() && bool("Size2 mismatch for ViennaCL matrices"));
  assert(size1_src       == vcl_src.size1()  && bool("Size1 mismatch for STL and ViennaCL matrices"));
  assert(size2_src       == vcl_src.size2()  && bool("Size2 mismatch for STL and ViennaCL matrices"));

  typedef typename STLVectorT2::value_type  DestNumericT;

  for (std::size_t i=0; i<size1_src; ++i)
    for (std::size_t j=0; j<size2_src; ++j)
      std_dest[(start1_dest + i * inc1_dest) * internal_size2_dest + (start2_dest + j * inc2_dest)] = static_cast<DestNumericT>(std_src[(start1_src + i * inc1_src) * internal_size2_src + (start2_src + j * inc2_src)]);

  vcl_dest = vcl_src; // here is the conversion taking place

  if (check(std_dest, start1_dest, inc1_dest, size1_dest, start2_dest, inc2_dest, size2_dest, internal_size2_dest, vcl_dest) != EXIT_SUCCESS)
    return EXIT_FAILURE;

  if (vcl_src.row_major())
  {
    viennacl::matrix<DestNumericT> A(vcl_src);
    if (check(std_dest, start1_dest, inc1_dest, size1_dest, start2_dest, inc2_dest, size2_dest, internal_size2_dest, A) != EXIT_SUCCESS)
      return EXIT_FAILURE;
  }
  else
  {
    viennacl::matrix<DestNumericT, viennacl::column_major> A(vcl_src);
    if (check(std_dest, start1_dest, inc1_dest, size1_dest, start2_dest, inc2_dest, size2_dest, internal_size2_dest, A) != EXIT_SUCCESS)
      return EXIT_FAILURE;
  }

  // --------------------------------------------------------------------------
  return EXIT_SUCCESS;
}

inline std::string type_string(unsigned int)    { return "unsigned int"; }
inline std::string type_string(int)             { return "int"; }
inline std::string type_string(unsigned long)   { return "unsigned long"; }
inline std::string type_string(long)            { return "long"; }
inline std::string type_string(float)           { return "float"; }
inline std::string type_string(double)          { return "double"; }

template<typename LayoutT, typename FromNumericT, typename ToNumericT>
int test()
{
  int retval = EXIT_SUCCESS;

  std::cout << std::endl;
  std::cout << "-----------------------------------------------" << std::endl;
  std::cout << std::endl;
  std::cout << "Conversion test from " << type_string(FromNumericT()) << " to " << type_string(ToNumericT()) << std::endl;
  std::cout << std::endl;

  std::size_t full_size1  = 578;
  std::size_t small_size1 = full_size1 / 4;

  std::size_t full_size2  = 687;
  std::size_t small_size2 = full_size2 / 4;

  //
  // Set up STL objects
  //
  std::vector<FromNumericT>               std_src(full_size1 * full_size2);
  std::vector<std::vector<FromNumericT> > std_src2(full_size1, std::vector<FromNumericT>(full_size2));
  std::vector<std::vector<FromNumericT> > std_src_small(small_size1, std::vector<FromNumericT>(small_size2));
  std::vector<ToNumericT> std_dest(std_src.size());

  for (std::size_t i=0; i<full_size1; ++i)
    for (std::size_t j=0; j<full_size2; ++j)
    {
      std_src[i * full_size2 + j]  = FromNumericT(1.0) + FromNumericT(i) + FromNumericT(j);
      std_src2[i][j]  = FromNumericT(1.0) + FromNumericT(i) + FromNumericT(j);
      if (i < small_size1 && j < small_size2)
        std_src_small[i][j]  = FromNumericT(1.0) + FromNumericT(i) + FromNumericT(j);
    }

  //
  // Set up ViennaCL objects
  //
  viennacl::matrix<FromNumericT, LayoutT> vcl_src(full_size1, full_size2);
  viennacl::matrix<ToNumericT,   LayoutT> vcl_dest(full_size1, full_size2);

  viennacl::copy(std_src2, vcl_src);

  viennacl::matrix<FromNumericT, LayoutT> vcl_src_small(small_size1, small_size2);
  viennacl::copy(std_src_small, vcl_src_small);
  viennacl::matrix<ToNumericT, LayoutT> vcl_dest_small(small_size1, small_size2);

  std::size_t r11_start = 1 + full_size1 / 4;
  std::size_t r11_stop  = r11_start + small_size1;
  viennacl::range vcl_r11(r11_start, r11_stop);

  std::size_t r12_start = 2 * full_size1 / 4;
  std::size_t r12_stop  = r12_start + small_size1;
  viennacl::range vcl_r12(r12_start, r12_stop);

  std::size_t r21_start = 2 * full_size2 / 4;
  std::size_t r21_stop  = r21_start + small_size2;
  viennacl::range vcl_r21(r21_start, r21_stop);

  std::size_t r22_start = 1 + full_size2 / 4;
  std::size_t r22_stop  = r22_start + small_size2;
  viennacl::range vcl_r22(r22_start, r22_stop);

  viennacl::matrix_range< viennacl::matrix<FromNumericT, LayoutT> > vcl_range_src(vcl_src, vcl_r11, vcl_r21);
  viennacl::matrix_range< viennacl::matrix<ToNumericT, LayoutT> >   vcl_range_dest(vcl_dest, vcl_r12, vcl_r22);



  std::size_t s11_start = 1 + full_size1 / 5;
  std::size_t s11_inc   = 3;
  std::size_t s11_size  = small_size1;
  viennacl::slice vcl_s11(s11_start, s11_inc, s11_size);

  std::size_t s12_start = 2 * full_size1 / 5;
  std::size_t s12_inc   = 2;
  std::size_t s12_size  = small_size1;
  viennacl::slice vcl_s12(s12_start, s12_inc, s12_size);

  std::size_t s21_start = 1 + full_size2 / 5;
  std::size_t s21_inc   = 3;
  std::size_t s21_size  = small_size2;
  viennacl::slice vcl_s21(s21_start, s21_inc, s21_size);

  std::size_t s22_start = 2 * full_size2 / 5;
  std::size_t s22_inc   = 2;
  std::size_t s22_size  = small_size2;
  viennacl::slice vcl_s22(s22_start, s22_inc, s22_size);

  viennacl::matrix_slice< viennacl::matrix<FromNumericT, LayoutT> > vcl_slice_src(vcl_src, vcl_s11, vcl_s21);
  viennacl::matrix_slice< viennacl::matrix<ToNumericT, LayoutT> >   vcl_slice_dest(vcl_dest, vcl_s12, vcl_s22);

  //
  // Now start running tests for vectors, ranges and slices:
  //

  std::cout << " ** vcl_src = matrix, vcl_dest = matrix **" << std::endl;
  retval = test(std_src,  0, 1, full_size1, 0, 1, full_size2, full_size2,
                std_dest, 0, 1, full_size1, 0, 1, full_size2, full_size2,
                vcl_src, vcl_dest);
  if (retval != EXIT_SUCCESS)
    return EXIT_FAILURE;

  std::cout << " ** vcl_src = matrix, vcl_dest = range **" << std::endl;
  retval = test(std_src,          0, 1, small_size1,                  0, 1,          small_size2, full_size2,
                std_dest, r12_start, 1, r12_stop - r12_start, r22_start, 1, r22_stop - r22_start, full_size2,
                vcl_src_small, vcl_range_dest);
  if (retval != EXIT_SUCCESS)
    return EXIT_FAILURE;

  std::cout << " ** vcl_src = matrix, vcl_dest = slice **" << std::endl;
  retval = test(std_src,          0,       1, small_size1,         0,       1, small_size2, full_size2,
                std_dest, s12_start, s12_inc,    s12_size, s22_start, s22_inc,    s22_size, full_size2,
                vcl_src_small, vcl_slice_dest);
  if (retval != EXIT_SUCCESS)
    return EXIT_FAILURE;

  ///////

  std::cout << " ** vcl_src = range, vcl_dest = matrix **" << std::endl;
  retval = test(std_src,  r11_start, 1, r11_stop - r11_start, r21_start, 1, r21_stop - r21_start, full_size2,
                std_dest,         0, 1,          small_size1,         0, 1,          small_size2, full_size2,
                vcl_range_src, vcl_dest_small);
  if (retval != EXIT_SUCCESS)
    return EXIT_FAILURE;

  std::cout << " ** vcl_src = range, vcl_dest = range **" << std::endl;
  retval = test(std_src,  r11_start, 1, r11_stop - r11_start, r21_start, 1, r21_stop - r21_start, full_size2,
                std_dest, r12_start, 1, r12_stop - r12_start, r22_start, 1, r22_stop - r22_start, full_size2,
                vcl_range_src, vcl_range_dest);
  if (retval != EXIT_SUCCESS)
    return EXIT_FAILURE;

  std::cout << " ** vcl_src = range, vcl_dest = slice **" << std::endl;
  retval = test(std_src,  r11_start,       1, r11_stop - r11_start, r21_start,       1, r21_stop - r21_start, full_size2,
                std_dest, s12_start, s12_inc,             s12_size, s22_start, s22_inc,             s22_size, full_size2,
                vcl_range_src, vcl_slice_dest);
  if (retval != EXIT_SUCCESS)
    return EXIT_FAILURE;

  ///////

  std::cout << " ** vcl_src = slice, vcl_dest = matrix **" << std::endl;
  retval = test(std_src,  s11_start, s11_inc,    s11_size, s21_start, s21_inc,    s21_size, full_size2,
                std_dest,         0,       1, small_size1,         0,       1, small_size2, full_size2,
                vcl_slice_src, vcl_dest_small);
  if (retval != EXIT_SUCCESS)
    return EXIT_FAILURE;

  std::cout << " ** vcl_src = slice, vcl_dest = range **" << std::endl;
  retval = test(std_src,  s11_start, s11_inc,             s11_size, s21_start, s21_inc,             s21_size, full_size2,
                std_dest, r12_start,       1, r12_stop - r12_start, r22_start,       1, r22_stop - r22_start, full_size2,
                vcl_slice_src, vcl_range_dest);
  if (retval != EXIT_SUCCESS)
    return EXIT_FAILURE;

  std::cout << " ** vcl_src = slice, vcl_dest = slice **" << std::endl;
  retval = test(std_src,  s11_start, s11_inc, s11_size, s21_start, s21_inc, s21_size, full_size2,
                std_dest, s12_start, s12_inc, s12_size, s22_start, s22_inc, s22_size, full_size2,
                vcl_slice_src, vcl_slice_dest);
  if (retval != EXIT_SUCCESS)
    return EXIT_FAILURE;

  return EXIT_SUCCESS;
}


template<typename FromNumericT, typename ToNumericT>
int test()
{
  int retval = test<viennacl::row_major, FromNumericT, ToNumericT>();
  if (retval == EXIT_SUCCESS)
  {
    retval = test<viennacl::column_major, FromNumericT, ToNumericT>();
    if (retval != EXIT_SUCCESS)
      std::cerr << "Test failed for column-major!" << std::endl;
  }
  else
    std::cerr << "Test failed for row-major!" << std::endl;

  return retval;
}

//
// -------------------------------------------------------------
//
int main()
{
  std::cout << std::endl;
  std::cout << "----------------------------------------------" << std::endl;
  std::cout << "----------------------------------------------" << std::endl;
  std::cout << "## Test :: Type conversion test for matrices  " << std::endl;
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
