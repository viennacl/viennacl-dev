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

#include <ctime>
#include <cmath>


#include "viennacl/linalg/prod.hpp"
#include "viennacl/linalg/nmf.hpp"

typedef float ScalarType;

const ScalarType EPS = 0.1;

float matrix_compare(viennacl::matrix<ScalarType>& res,
                     viennacl::matrix<ScalarType>& ref)
{
    std::vector<ScalarType> res_std(res.internal_size());
    std::vector<ScalarType> ref_std(ref.internal_size());

    viennacl::fast_copy(res, &res_std[0]);
    viennacl::fast_copy(ref, &ref_std[0]);

    float diff = 0.0;
    float mx = 0.0;

    for(std::size_t i = 0; i < res_std.size(); i++) {
        diff = std::max(diff, std::abs(res_std[i] - ref_std[i]));
        mx = std::max(mx, res_std[i]);
    }

    return diff / mx;
}


void fill_random(std::vector<ScalarType>& v)
{
    for(std::size_t j = 0; j < v.size(); j++)
        v[j] = static_cast<ScalarType>(rand()) / RAND_MAX;
}


void test_nmf(std::size_t m, std::size_t k, std::size_t n)
{
    std::vector<ScalarType> stl_w(m * k);
    std::vector<ScalarType> stl_h(k * n);

    viennacl::matrix<ScalarType> v_ref(m, n);
    viennacl::matrix<ScalarType> w_ref(m, k);
    viennacl::matrix<ScalarType> h_ref(k, n);

    fill_random(stl_w);
    fill_random(stl_h);

    viennacl::fast_copy(&stl_w[0], &stl_w[0] + stl_w.size(), w_ref);
    viennacl::fast_copy(&stl_h[0], &stl_h[0] + stl_h.size(), h_ref);

    v_ref = viennacl::linalg::prod(w_ref, h_ref);  //reference

    // Fill again with random numbers:
    fill_random(stl_w);
    fill_random(stl_h);

    viennacl::matrix<ScalarType> w_nmf(m, k);
    viennacl::matrix<ScalarType> h_nmf(k, n);

    viennacl::fast_copy(&stl_w[0], &stl_w[0] + stl_w.size(), w_nmf);
    viennacl::fast_copy(&stl_h[0], &stl_h[0] + stl_h.size(), h_nmf);



    viennacl::linalg::nmf_config conf;
    viennacl::linalg::nmf(v_ref, w_nmf, h_nmf, conf);

    viennacl::matrix<ScalarType> v_nmf = viennacl::linalg::prod(w_nmf, h_nmf);

    float diff  = matrix_compare(v_ref, v_nmf);
    bool diff_ok = fabs(diff) < EPS;

    long iterations = static_cast<long>(conf.iters());
    printf("%6s [%lux%lux%lu] diff = %.6f (%ld iterations)\n", diff_ok ? "[[OK]]":"[FAIL]", m, k, n, diff, iterations);

    if (!diff_ok)
      exit(EXIT_FAILURE);
}

int main()
{
  //srand(time(NULL));  //let's use deterministic tests, so keep the default srand() initialization

  test_nmf(3, 3, 3);
  test_nmf(3, 2, 3);
  test_nmf(16, 7, 12);
  test_nmf(160, 73, 200);
  test_nmf(1000, 15, 1000);

  std::cout << std::endl;
  std::cout << "------- Test completed --------" << std::endl;
  std::cout << std::endl;


  return EXIT_SUCCESS;
}
