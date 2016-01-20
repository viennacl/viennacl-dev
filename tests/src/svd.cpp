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



/** \file tests/src/svd.cpp  Tests the singular value decomposition.
*   \test  Tests the singular value decomposition.
**/

#include <stdexcept>
#include <iostream>
#include <string>
#include <vector>
#include <cmath>

#include "viennacl/matrix.hpp"
#include "viennacl/linalg/prod.hpp"

#include "viennacl/linalg/svd.hpp"

#include "viennacl/tools/timer.hpp"


inline void read_matrix_size(std::fstream& f, std::size_t & sz1, std::size_t & sz2)
{
  if (!f.is_open())
    throw std::invalid_argument("File is not opened");

  f >> sz1 >> sz2;
}


template<typename ScalarType>
void read_matrix_body(std::fstream& f, viennacl::matrix<ScalarType>& A)
{
  if (!f.is_open())
    throw std::invalid_argument("File is not opened");

  boost::numeric::ublas::matrix<ScalarType> h_A(A.size1(), A.size2());

  for (std::size_t i = 0; i < h_A.size1(); i++)
  {
    for (std::size_t j = 0; j < h_A.size2(); j++)
    {
      ScalarType val = 0.0;
      f >> val;
      h_A(i, j) = val;
    }
  }

  viennacl::copy(h_A, A);
}


template<typename ScalarType>
void read_vector_body(std::fstream& f, std::vector<ScalarType>& v)
{
  if (!f.is_open())
    throw std::invalid_argument("File is not opened");

  for (std::size_t i = 0; i < v.size(); i++)
  {
    ScalarType val = 0.0;
    f >> val;
    v[i] = val;
  }
}


template<typename ScalarType>
void random_fill(std::vector<ScalarType>& in)
{
  for (std::size_t i = 0; i < in.size(); i++)
    in[i] = static_cast<ScalarType>(rand()) / ScalarType(RAND_MAX);
}


template<typename ScalarType>
bool check_bidiag(viennacl::matrix<ScalarType>& A)
{
  const ScalarType EPS = 0.0001f;

  std::vector<ScalarType> aA(A.size1() * A.size2());
  viennacl::fast_copy(A, &aA[0]);

  for (std::size_t i = 0; i < A.size1(); i++)
  {
    for (std::size_t j = 0; j < A.size2(); j++)
    {
      ScalarType val = aA[i * A.size2() + j];
      if ((fabs(val) > EPS) && (i != j) && ((i + 1) != j))
      {
        std::cout << "Failed at " << i << " " << j << " " << val << std::endl;
        return false;
      }
    }
  }

  return true;
}

template<typename ScalarType>
ScalarType matrix_compare(viennacl::matrix<ScalarType>& res,
                     viennacl::matrix<ScalarType>& ref)
{
  std::vector<ScalarType> res_std(res.internal_size());
  std::vector<ScalarType> ref_std(ref.internal_size());

  viennacl::fast_copy(res, &res_std[0]);
  viennacl::fast_copy(ref, &ref_std[0]);

  ScalarType diff = 0.0;
  ScalarType mx = 0.0;

  for (std::size_t i = 0; i < res_std.size(); i++)
  {
    diff = std::max(diff, std::abs(res_std[i] - ref_std[i]));
    mx = std::max(mx, res_std[i]);
  }

  return diff / mx;
}


template<typename ScalarType>
ScalarType sigmas_compare(viennacl::matrix<ScalarType>& res,
                               std::vector<ScalarType>& ref)
{
    std::vector<ScalarType> res_std(ref.size());

    for (std::size_t i = 0; i < ref.size(); i++)
        res_std[i] = res(i, i);

    std::sort(ref.begin(), ref.end());
    std::sort(res_std.begin(), res_std.end());

    ScalarType diff = 0.0;
    ScalarType mx = 0.0;
    for (std::size_t i = 0; i < ref.size(); i++)
    {
        diff = std::max(diff, std::abs(res_std[i] - ref[i]));
        mx = std::max(mx, res_std[i]);
    }

    return diff / mx;
}


template<typename ScalarType>
void test_svd(const std::string & fn, ScalarType EPS)
{
  std::size_t sz1, sz2;

  //read matrix

  // sz1 = 2048, sz2 = 2048;
  // std::vector<ScalarType> in(sz1 * sz2);
  // random_fill(in);

  // read file
  std::fstream f(fn.c_str(), std::fstream::in);
  //read size of input matrix
  read_matrix_size(f, sz1, sz2);

  std::size_t to = std::min(sz1, sz2);

  viennacl::matrix<ScalarType> Ai(sz1, sz2), Aref(sz1, sz2), QL(sz1, sz1), QR(sz2, sz2);
  read_matrix_body(f, Ai);

  std::vector<ScalarType> sigma_ref(to);
  read_vector_body(f, sigma_ref);

  f.close();

  // viennacl::fast_copy(&in[0], &in[0] + in.size(), Ai);

  Aref = Ai;

  viennacl::tools::timer timer;
  timer.start();

  viennacl::linalg::svd(Ai, QL, QR);

  viennacl::backend::finish();

  double time_spend = timer.get();

  viennacl::matrix<ScalarType> result1(sz1, sz2), result2(sz1, sz2);
  result1 = viennacl::linalg::prod(QL, Ai);
  result2 = viennacl::linalg::prod(result1, trans(QR));

  ScalarType sigma_diff = sigmas_compare(Ai, sigma_ref);
  ScalarType prods_diff  = matrix_compare(result2, Aref);

  bool sigma_ok = (fabs(sigma_diff) < EPS)
                   && (fabs(prods_diff) < std::sqrt(EPS));  //note: computing the product is not accurate down to 10^{-16}, so we allow for a higher tolerance here

  printf("%6s [%dx%d] %40s sigma_diff = %.6f; prod_diff = %.6f; time = %.6f\n", sigma_ok?"[[OK]]":"[FAIL]", (int)Aref.size1(), (int)Aref.size2(), fn.c_str(), sigma_diff, prods_diff, time_spend);
  if (!sigma_ok)
    exit(EXIT_FAILURE);
}


template<typename ScalarType>
void time_svd(std::size_t sz1, std::size_t sz2)
{
  viennacl::matrix<ScalarType> Ai(sz1, sz2), QL(sz1, sz1), QR(sz2, sz2);

  std::vector<ScalarType> in(Ai.internal_size1() * Ai.internal_size2());
  random_fill(in);

  viennacl::fast_copy(&in[0], &in[0] + in.size(), Ai);


  viennacl::tools::timer timer;
  timer.start();

  viennacl::linalg::svd(Ai, QL, QR);
  viennacl::backend::finish();
  double time_spend = timer.get();

  printf("[%dx%d] time = %.6f\n", static_cast<int>(sz1), static_cast<int>(sz2), time_spend);
}


template<typename ScalarType>
int test(ScalarType epsilon)
{

    test_svd<ScalarType>(std::string("../examples/testdata/svd/qr.example"), epsilon);
    test_svd<ScalarType>(std::string("../examples/testdata/svd/wiki.example"), epsilon);
    test_svd<ScalarType>(std::string("../examples/testdata/svd/wiki.qr.example"), epsilon);
    test_svd<ScalarType>(std::string("../examples/testdata/svd/pysvd.example"), epsilon);
    test_svd<ScalarType>(std::string("../examples/testdata/svd/random.example"), epsilon);

    time_svd<ScalarType>(500, 500);
    time_svd<ScalarType>(1024, 1024);
    time_svd<ScalarType>(2048, 512);
    //time_svd<ScalarType>(2048, 2048);
    //time_svd(4096, 4096);  //takes too long for a standard sanity test. Feel free to uncomment

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
   std::cout << "## Test :: Singular Value Decomposition" << std::endl;
   std::cout << "----------------------------------------------" << std::endl;
   std::cout << "----------------------------------------------" << std::endl;
   std::cout << std::endl;

   int retval = EXIT_SUCCESS;

   std::cout << std::endl;
   std::cout << "----------------------------------------------" << std::endl;
   std::cout << std::endl;
   {
      typedef float NumericT;
      NumericT epsilon = NumericT(1.0E-4);
      std::cout << "# Testing setup:" << std::endl;
      std::cout << "  eps:     " << epsilon << std::endl;
      std::cout << "  numeric: float" << std::endl;
      retval = test<NumericT>(epsilon);
      if ( retval == EXIT_SUCCESS )
        std::cout << "# Test passed" << std::endl;
      else
        return retval;
   }
   std::cout << std::endl;
   std::cout << "----------------------------------------------" << std::endl;
   std::cout << std::endl;
   if ( viennacl::ocl::current_device().double_support() )
   {
      {
        typedef double NumericT;
        NumericT epsilon = 1.0E-6;  //Note: higher accuracy not possible, because data only available with floating point precision
        std::cout << "# Testing setup:" << std::endl;
        std::cout << "  eps:     " << epsilon << std::endl;
        std::cout << "  numeric: double" << std::endl;
        retval = test<NumericT>(epsilon);
        if ( retval == EXIT_SUCCESS )
          std::cout << "# Test passed" << std::endl;
        else
          return retval;
      }
      std::cout << std::endl;
      std::cout << "----------------------------------------------" << std::endl;
      std::cout << std::endl;
   }

  std::cout << std::endl;
  std::cout << "------- Test completed --------" << std::endl;
  std::cout << std::endl;


   return retval;
}


