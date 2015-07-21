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


/** \file tests/src/nmf.cpp  Tests the nonnegative matrix factorization.
*   \test Tests the nonnegative matrix factorization.
**/

#include <ctime>
#include <cmath>

#include "viennacl/linalg/prod.hpp"
#include "viennacl/linalg/nmf.hpp"

typedef float ScalarType;

const ScalarType EPS = ScalarType(0.03);

template<typename MATRIX>
float matrix_compare(MATRIX & res, viennacl::matrix_base<ScalarType>& ref)
{
  float diff = 0.0;
  float mx = 0.0;

  for (std::size_t i = 0; i < ref.size1(); i++)
  {
    for (std::size_t j = 0; j < ref.size2(); ++j)
    {
      diff = std::max(diff, std::abs(res(i, j) - ref(i, j)));
      ScalarType valRes = (ScalarType) res(i, j);
      mx = std::max(mx, valRes);
    }
  }
  return diff / mx;
}

void fill_random(viennacl::matrix_base<ScalarType>& v);

void fill_random(viennacl::matrix_base<ScalarType>& v)
{
  for (std::size_t i = 0; i < v.size1(); i++)
  {
    for (std::size_t j = 0; j < v.size2(); ++j)
      v(i, j) = static_cast<ScalarType>(rand()) / ScalarType(RAND_MAX);
  }
}

void test_nmf(std::size_t m, std::size_t k, std::size_t n);

void test_nmf(std::size_t m, std::size_t k, std::size_t n)
{
  viennacl::matrix<ScalarType> v_ref(m, n);
  viennacl::matrix<ScalarType> w_ref(m, k);
  viennacl::matrix<ScalarType> h_ref(k, n);

  fill_random(w_ref);
  fill_random(h_ref);

  v_ref = viennacl::linalg::prod(w_ref, h_ref);  //reference result

  viennacl::matrix<ScalarType> w_nmf(m, k);
  viennacl::matrix<ScalarType> h_nmf(k, n);

  fill_random(w_nmf);
  fill_random(h_nmf);

  viennacl::linalg::nmf_config conf;
  conf.print_relative_error(true);
  conf.max_iterations(3000); //3000 iterations are enough for the test

  viennacl::linalg::nmf(v_ref, w_nmf, h_nmf, conf);

  viennacl::matrix<ScalarType> v_nmf = viennacl::linalg::prod(w_nmf, h_nmf);

  float diff = matrix_compare(v_ref, v_nmf);
  bool diff_ok = fabs(diff) < EPS;

  long iterations = static_cast<long>(conf.iters());
  printf("%6s [%lux%lux%lu] diff = %.6f (%ld iterations)\n", diff_ok ? "[[OK]]" : "[FAIL]", m, k, n,
      diff, iterations);

  if (!diff_ok)
    exit(EXIT_FAILURE);
}

int main()
{
  //srand(time(NULL));  //let's use deterministic tests, so keep the default srand() initialization
  std::cout << std::endl;
  std::cout << "------- Test NMF --------" << std::endl;
  std::cout << std::endl;

  test_nmf(3, 3, 3);
  test_nmf(5, 4, 5);
  test_nmf(16, 7, 12);
  test_nmf(140, 86, 113);

  std::cout << std::endl;
  std::cout << "------- Test completed --------" << std::endl;
  std::cout << std::endl;

  return EXIT_SUCCESS;
}
