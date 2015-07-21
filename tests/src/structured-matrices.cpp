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



/** \file tests/src/structured-matrices.cpp  Tests structured matrices.
*   \test   Tests structured matrices.
**/

#include <iostream>
#include <vector>
#include <cmath>
#include <complex>
#include <fstream>

//#define VIENNACL_BUILD_INFO

//#define VIENNACL_DEBUG_ALL

#include "viennacl/toeplitz_matrix.hpp"
#include "viennacl/circulant_matrix.hpp"
#include "viennacl/vandermonde_matrix.hpp"
#include "viennacl/hankel_matrix.hpp"
#include "viennacl/linalg/prod.hpp"

#include "viennacl/fft.hpp"

//
// A simple dense matrix class (in order to avoid an unnecessary boost dependency)
//
template<typename T>
class dense_matrix
{
  public:
    typedef std::size_t   size_type;

    dense_matrix(std::size_t rows, std::size_t cols) : elements_(rows * cols), rows_(rows), cols_(cols) {}

    T & operator()(std::size_t i, std::size_t j) { return elements_[i*cols_ + j]; }
    T const & operator()(std::size_t i, std::size_t j) const { return elements_[i*cols_ + j]; }

    std::size_t size1() const { return rows_; }
    std::size_t size2() const { return cols_; }

    dense_matrix & operator+=(dense_matrix const & other)
    {
      for (std::size_t i = 0; i < other.size1(); i++)
        for (std::size_t j = 0; j < other.size2(); j++)
          elements_[i*cols_ + j] = other.elements_[i*cols_+j];
      return *this;
    }

  private:
    std::vector<T> elements_;
    std::size_t rows_;
    std::size_t cols_;
};

template<typename T>
std::ostream & operator<<(std::ostream & os, dense_matrix<T> const & mat)
{
  std::cout << "[" << mat.size1() << "," << mat.size2() << "](";
  for (std::size_t i=0; i<mat.size1(); ++i)
  {
    std::cout << "(";
    for (std::size_t j=0; j<mat.size2(); ++j)
      std::cout << mat(i,j) << ",";
    std::cout << ")";
  }

  return os;
}


template<typename ScalarType>
ScalarType diff(dense_matrix<ScalarType> const & m1, dense_matrix<ScalarType> const & m2)
{
    ScalarType df = 0.0;
    ScalarType d1 = 0;
    ScalarType d2 = 0;

    for (std::size_t i = 0; i < m1.size1(); i++)
      for (std::size_t j = 0; j < m1.size2(); j++)
      {
        df += (m1(i,j) - m2(i,j)) * (m1(i,j) - m2(i,j));
        d1 += m1(i,j) * m1(i,j);
        d2 += m2(i,j) * m2(i,j);
      }

    if ( d1 + d2 <= 0 )
      return 0;

    return std::sqrt(df / std::max<ScalarType>(d1, d2));
}


template<typename ScalarType>
ScalarType diff(std::vector<ScalarType>& vec, std::vector<ScalarType>& ref)
{
    ScalarType df = 0.0;
    ScalarType norm_ref = 0;

    for (std::size_t i = 0; i < vec.size(); i++)
    {
        df = df + pow(vec[i] - ref[i], 2);
        norm_ref += ref[i] * ref[i];
    }

    return std::sqrt(df / norm_ref);
}

template<typename ScalarType>
ScalarType diff_max(std::vector<ScalarType>& vec, std::vector<ScalarType>& ref)
{
  ScalarType df = 0.0;
  ScalarType mx = 0.0;
  ScalarType norm_max = 0;

  for (std::size_t i = 0; i < vec.size(); i++)
  {
    df = std::max<ScalarType>(std::fabs(vec[i] - ref[i]), df);
    mx = std::max<ScalarType>(std::fabs(vec[i]), mx);

    if (mx > 0)
    {
      if (norm_max < df / mx)
        norm_max = df / mx;
    }
  }

  return norm_max;
}


template<typename ScalarType>
void transpose_test()
{
    int w = 5, h = 7;
    std::vector<ScalarType> s_normal(2 * w * h);
    viennacl::matrix<ScalarType> normal(w, 2 * h);
    viennacl::matrix<ScalarType> transp(h, 2 * w);

    for (unsigned int i = 0; i < s_normal.size(); i+=2) {
        s_normal[i] = i;
        s_normal[i+1] = i;
    }
    viennacl::fast_copy(&s_normal[0], &s_normal[0] + s_normal.size(), normal);
    std::cout << normal << std::endl;
    viennacl::linalg::transpose(normal);
    std::cout << normal << std::endl;
}



template<typename ScalarType>
int toeplitz_test(ScalarType epsilon)
{
    std::size_t TOEPLITZ_SIZE = 47;
    viennacl::toeplitz_matrix<ScalarType> vcl_toeplitz1(TOEPLITZ_SIZE, TOEPLITZ_SIZE);
    viennacl::toeplitz_matrix<ScalarType> vcl_toeplitz2(TOEPLITZ_SIZE, TOEPLITZ_SIZE);

    viennacl::vector<ScalarType> vcl_input(TOEPLITZ_SIZE);
    viennacl::vector<ScalarType> vcl_result(TOEPLITZ_SIZE);

    std::vector<ScalarType> input_ref(TOEPLITZ_SIZE);
    std::vector<ScalarType> result_ref(TOEPLITZ_SIZE);

    dense_matrix<ScalarType> m1(TOEPLITZ_SIZE, TOEPLITZ_SIZE);
    dense_matrix<ScalarType> m2(TOEPLITZ_SIZE, TOEPLITZ_SIZE);

    for (std::size_t i = 0; i < TOEPLITZ_SIZE; i++)
      for (std::size_t j = 0; j < TOEPLITZ_SIZE; j++)
      {
        m1(i,j) = static_cast<ScalarType>(i) - static_cast<ScalarType>(j);
        m2(i,j) = m1(i,j) * m1(i,j) + ScalarType(1);
      }

    for (std::size_t i = 0; i < TOEPLITZ_SIZE; i++)
      input_ref[i] = ScalarType(i);

    // Copy to ViennaCL
    viennacl::copy(m1, vcl_toeplitz1);
    viennacl::copy(m2, vcl_toeplitz2);
    viennacl::copy(input_ref, vcl_input);

    //
    // Matrix-Vector product:
    //
    vcl_result = viennacl::linalg::prod(vcl_toeplitz1, vcl_input);

    for (std::size_t i = 0; i < m1.size1(); i++)     //reference calculation
    {
      ScalarType entry = 0;
      for (std::size_t j = 0; j < m1.size2(); j++)
        entry += m1(i,j) * input_ref[j];

      result_ref[i] = entry;
    }

    viennacl::copy(vcl_result, input_ref);
    std::cout << "Matrix-Vector Product: " << diff_max(input_ref, result_ref);
    if (diff_max(input_ref, result_ref) < epsilon)
      std::cout << " [OK]" << std::endl;
    else
    {
      for (std::size_t i=0; i<input_ref.size(); ++i)
        std::cout << "Should: " << result_ref[i] << ", is: " << input_ref[i] << std::endl;
      std::cout << " [FAILED]" << std::endl;
      return EXIT_FAILURE;
    }


    //
    // Matrix addition:
    //
    vcl_toeplitz1 += vcl_toeplitz2;

    for (std::size_t i = 0; i < m1.size1(); i++)    //reference calculation
      for (std::size_t j = 0; j < m1.size2(); j++)
        m1(i,j) += m2(i,j);

    viennacl::copy(vcl_toeplitz1, m2);
    std::cout << "Matrix Addition: " << diff(m1, m2);
    if (diff(m1, m2) < epsilon)
      std::cout << " [OK]" << std::endl;
    else
    {
      std::cout << " [FAILED]" << std::endl;
      return EXIT_FAILURE;
    }

    //
    // Per-Element access:
    //
    vcl_toeplitz1(2,4) = 42;

    for (std::size_t i=0; i<m1.size1(); ++i)    //reference calculation
    {
      if (i + 2 < m1.size2())
        m1(i, i+2) = 42;
    }

    viennacl::copy(vcl_toeplitz1, m2);
    std::cout << "Element manipulation: " << diff(m1, m2);
    if (diff(m1, m2) < epsilon)
      std::cout << " [OK]" << std::endl;
    else
    {
      std::cout << " [FAILED]" << std::endl;
      return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}

template<typename ScalarType>
int circulant_test(ScalarType epsilon)
{
    std::size_t CIRCULANT_SIZE = 53;
    viennacl::circulant_matrix<ScalarType> vcl_circulant1(CIRCULANT_SIZE, CIRCULANT_SIZE);
    viennacl::circulant_matrix<ScalarType> vcl_circulant2(CIRCULANT_SIZE, CIRCULANT_SIZE);

    viennacl::vector<ScalarType> vcl_input(CIRCULANT_SIZE);
    viennacl::vector<ScalarType> vcl_result(CIRCULANT_SIZE);

    std::vector<ScalarType> input_ref(CIRCULANT_SIZE);
    std::vector<ScalarType> result_ref(CIRCULANT_SIZE);

    dense_matrix<ScalarType> m1(vcl_circulant1.size1(), vcl_circulant1.size2());
    dense_matrix<ScalarType> m2(vcl_circulant1.size1(), vcl_circulant1.size2());

    for (std::size_t i = 0; i < m1.size1(); i++)
      for (std::size_t j = 0; j < m1.size2(); j++)
      {
        m1(i,j) = static_cast<ScalarType>((i - j + m1.size1()) % m1.size1());
        m2(i,j) = m1(i,j) * m1(i,j) + ScalarType(1);
      }

    for (std::size_t i = 0; i < input_ref.size(); i++)
      input_ref[i] = ScalarType(i);

    // Copy to ViennaCL
    viennacl::copy(m1, vcl_circulant1);
    viennacl::copy(m2, vcl_circulant2);
    viennacl::copy(input_ref, vcl_input);

    //
    // Matrix-Vector product:
    //
    vcl_result = viennacl::linalg::prod(vcl_circulant1, vcl_input);

    for (std::size_t i = 0; i < m1.size1(); i++)     //reference calculation
    {
      ScalarType entry = 0;
      for (std::size_t j = 0; j < m1.size2(); j++)
        entry += m1(i,j) * input_ref[j];

      result_ref[i] = entry;
    }

    viennacl::copy(vcl_result, input_ref);
    std::cout << "Matrix-Vector Product: " << diff_max(input_ref, result_ref);
    if (diff_max(input_ref, result_ref) < epsilon)
      std::cout << " [OK]" << std::endl;
    else
    {
      for (std::size_t i=0; i<input_ref.size(); ++i)
        std::cout << "Should: " << result_ref[i] << ", is: " << input_ref[i] << std::endl;
      std::cout << " [FAILED]" << std::endl;
      return EXIT_FAILURE;
    }


    //
    // Matrix addition:
    //
    vcl_circulant1 += vcl_circulant2;

    for (std::size_t i = 0; i < m1.size1(); i++)    //reference calculation
      for (std::size_t j = 0; j < m1.size2(); j++)
        m1(i,j) += m2(i,j);

    viennacl::copy(vcl_circulant1, m2);
    std::cout << "Matrix Addition: " << diff(m1, m2);
    if (diff(m1, m2) < epsilon)
      std::cout << " [OK]" << std::endl;
    else
    {
      std::cout << " [FAILED]" << std::endl;
      return EXIT_FAILURE;
    }

    //
    // Per-Element access:
    //
    vcl_circulant1(4,2) = 42;

    for (std::size_t i = 0; i < m1.size1(); i++)    //reference calculation
      for (std::size_t j = 0; j < m1.size2(); j++)
      {
        if ((i - j + m1.size1()) % m1.size1() == 2)
          m1(i, j) = 42;
      }

    viennacl::copy(vcl_circulant1, m2);
    std::cout << "Element manipulation: " << diff(m1, m2);
    if (diff(m1, m2) < epsilon)
      std::cout << " [OK]" << std::endl;
    else
    {
      std::cout << " [FAILED]" << std::endl;
      return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}

template<typename ScalarType>
int vandermonde_test(ScalarType epsilon)
{
    std::size_t VANDERMONDE_SIZE = 61;

    viennacl::vandermonde_matrix<ScalarType> vcl_vandermonde1(VANDERMONDE_SIZE, VANDERMONDE_SIZE);
    viennacl::vandermonde_matrix<ScalarType> vcl_vandermonde2(VANDERMONDE_SIZE, VANDERMONDE_SIZE);

    viennacl::vector<ScalarType> vcl_input(VANDERMONDE_SIZE);
    viennacl::vector<ScalarType> vcl_result(VANDERMONDE_SIZE);

    std::vector<ScalarType> input_ref(VANDERMONDE_SIZE);
    std::vector<ScalarType> result_ref(VANDERMONDE_SIZE);

    dense_matrix<ScalarType> m1(vcl_vandermonde1.size1(), vcl_vandermonde1.size2());
    dense_matrix<ScalarType> m2(m1.size1(), m1.size2());

    for (std::size_t i = 0; i < m1.size1(); i++)
      for (std::size_t j = 0; j < m1.size2(); j++)
      {
        m1(i,j) = std::pow(ScalarType(1.0) + ScalarType(i)/ScalarType(1000.0), ScalarType(j));
        m2(i,j) = std::pow(ScalarType(1.0) - ScalarType(i)/ScalarType(2000.0), ScalarType(j));
      }

    for (std::size_t i = 0; i < input_ref.size(); i++)
      input_ref[i] = ScalarType(i);

    // Copy to ViennaCL
    viennacl::copy(m1, vcl_vandermonde1);
    viennacl::copy(m2, vcl_vandermonde2);
    viennacl::copy(input_ref, vcl_input);

    //
    // Matrix-Vector product:
    //
    vcl_result = viennacl::linalg::prod(vcl_vandermonde1, vcl_input);

    for (std::size_t i = 0; i < m1.size1(); i++)     //reference calculation
    {
      ScalarType entry = 0;
      for (std::size_t j = 0; j < m1.size2(); j++)
        entry += m1(i,j) * input_ref[j];

      result_ref[i] = entry;
    }

    viennacl::copy(vcl_result, input_ref);
    std::cout << "Matrix-Vector Product: " << diff_max(input_ref, result_ref);
    if (diff_max(input_ref, result_ref) < epsilon)
      std::cout << " [OK]" << std::endl;
    else
    {
      for (std::size_t i=0; i<input_ref.size(); ++i)
        std::cout << "Should: " << result_ref[i] << ", is: " << input_ref[i] << std::endl;
      std::cout << " [FAILED]" << std::endl;
      return EXIT_FAILURE;
    }


    //
    // Note: Matrix addition does not make sense for a Vandermonde matrix
    //


    //
    // Per-Element access:
    //
    vcl_vandermonde1(4) = static_cast<ScalarType>(1.0001);

    for (std::size_t j = 0; j < m1.size2(); j++)
    {
      m1(4, j) = std::pow(ScalarType(1.0001), ScalarType(j));
    }

    viennacl::copy(vcl_vandermonde1, m2);
    std::cout << "Element manipulation: " << diff(m1, m2);
    if (diff(m1, m2) < epsilon)
      std::cout << " [OK]" << std::endl;
    else
    {
      std::cout << " [FAILED]" << std::endl;
      return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}

template<typename ScalarType>
int hankel_test(ScalarType epsilon)
{
    std::size_t HANKEL_SIZE = 7;
    viennacl::hankel_matrix<ScalarType> vcl_hankel1(HANKEL_SIZE, HANKEL_SIZE);
    viennacl::hankel_matrix<ScalarType> vcl_hankel2(HANKEL_SIZE, HANKEL_SIZE);

    viennacl::vector<ScalarType> vcl_input(HANKEL_SIZE);
    viennacl::vector<ScalarType> vcl_result(HANKEL_SIZE);

    std::vector<ScalarType> input_ref(HANKEL_SIZE);
    std::vector<ScalarType> result_ref(HANKEL_SIZE);

    dense_matrix<ScalarType> m1(vcl_hankel1.size1(), vcl_hankel1.size2());
    dense_matrix<ScalarType> m2(m1.size1(), m1.size2());

    for (std::size_t i = 0; i < m1.size1(); i++)
      for (std::size_t j = 0; j < m1.size2(); j++)
      {
        m1(i,j) = static_cast<ScalarType>((i + j) % (2 * m1.size1()));
        m2(i,j) = m1(i,j) * m1(i,j) + ScalarType(1);
      }

    for (std::size_t i = 0; i < input_ref.size(); i++)
      input_ref[i] = ScalarType(i);

    // Copy to ViennaCL
    viennacl::copy(m1, vcl_hankel1);
    viennacl::copy(m2, vcl_hankel2);
    viennacl::copy(input_ref, vcl_input);

    //
    // Matrix-Vector product:
    //
    vcl_result = viennacl::linalg::prod(vcl_hankel1, vcl_input);

    for (std::size_t i = 0; i < m1.size1(); i++)     //reference calculation
    {
      ScalarType entry = 0;
      for (std::size_t j = 0; j < m1.size2(); j++)
        entry += m1(i,j) * input_ref[j];

      result_ref[i] = entry;
    }

    viennacl::copy(vcl_result, input_ref);
    std::cout << "Matrix-Vector Product: " << diff_max(input_ref, result_ref);
    if (diff_max(input_ref, result_ref) < epsilon)
      std::cout << " [OK]" << std::endl;
    else
    {
      for (std::size_t i=0; i<input_ref.size(); ++i)
        std::cout << "Should: " << result_ref[i] << ", is: " << input_ref[i] << std::endl;
      std::cout << " [FAILED]" << std::endl;
      return EXIT_FAILURE;
    }


    //
    // Matrix addition:
    //
    vcl_hankel1 += vcl_hankel2;

    for (std::size_t i = 0; i < m1.size1(); i++)    //reference calculation
      for (std::size_t j = 0; j < m1.size2(); j++)
        m1(i,j) += m2(i,j);

    viennacl::copy(vcl_hankel1, m2);
    std::cout << "Matrix Addition: " << diff(m1, m2);
    if (diff(m1, m2) < epsilon)
      std::cout << " [OK]" << std::endl;
    else
    {
      std::cout << " [FAILED]" << std::endl;
      return EXIT_FAILURE;
    }

    //
    // Per-Element access:
    //
    vcl_hankel1(4,2) = 42;

    for (std::size_t i = 0; i < m1.size1(); i++)    //reference calculation
      for (std::size_t j = 0; j < m1.size2(); j++)
      {
        if ((i + j) % (2*m1.size1()) == 6)
          m1(i, j) = 42;
      }

    viennacl::copy(vcl_hankel1, m2);
    std::cout << "Element manipulation: " << diff(m1, m2);
    if (diff(m1, m2) < epsilon)
      std::cout << " [OK]" << std::endl;
    else
    {
      std::cout << " [FAILED]" << std::endl;
      return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}

int main()
{
  std::cout << std::endl;
  std::cout << "----------------------------------------------" << std::endl;
  std::cout << "----------------------------------------------" << std::endl;
  std::cout << "## Test :: Structured Matrices" << std::endl;
  std::cout << "----------------------------------------------" << std::endl;
  std::cout << "----------------------------------------------" << std::endl;
  std::cout << std::endl;

  double eps = 1e-3;

  std::cout << "# Testing setup:" << std::endl;
  std::cout << "  eps:     " << eps << std::endl;
  std::cout << "  numeric: float" << std::endl;
  std::cout << std::endl;
  std::cout << " -- Vandermonde matrix -- " << std::endl;
  if (vandermonde_test<float>(static_cast<float>(eps)) == EXIT_FAILURE)
    return EXIT_FAILURE;

  std::cout << " -- Circulant matrix -- " << std::endl;
  if (circulant_test<float>(static_cast<float>(eps)) == EXIT_FAILURE)
    return EXIT_FAILURE;

  std::cout << " -- Toeplitz matrix -- " << std::endl;
  if (toeplitz_test<float>(static_cast<float>(eps)) == EXIT_FAILURE)
    return EXIT_FAILURE;

  std::cout << " -- Hankel matrix -- " << std::endl;
  if (hankel_test<float>(static_cast<float>(eps)) == EXIT_FAILURE)
    return EXIT_FAILURE;


  std::cout << std::endl;

  if ( viennacl::ocl::current_device().double_support() )
  {
    eps = 1e-10;

    std::cout << std::endl;
    std::cout << "# Testing setup:" << std::endl;
    std::cout << "  eps:     " << eps << std::endl;
    std::cout << "  numeric: double" << std::endl;
    std::cout << std::endl;

    std::cout << " -- Vandermonde matrix -- " << std::endl;
    if (vandermonde_test<double>(eps) == EXIT_FAILURE)
      return EXIT_FAILURE;

    std::cout << " -- Circulant matrix -- " << std::endl;
    if (circulant_test<double>(eps) == EXIT_FAILURE)
      return EXIT_FAILURE;

    std::cout << " -- Toeplitz matrix -- " << std::endl;
    if (toeplitz_test<double>(eps) == EXIT_FAILURE)
      return EXIT_FAILURE;

    std::cout << " -- Hankel matrix -- " << std::endl;
    if (hankel_test<double>(eps) == EXIT_FAILURE)
      return EXIT_FAILURE;
  }

  std::cout << std::endl;
  std::cout << "------- Test completed --------" << std::endl;
  std::cout << std::endl;


  return EXIT_SUCCESS;
}
