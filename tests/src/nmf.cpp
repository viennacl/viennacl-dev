#include <ctime>
#include <cmath>

#include "viennacl/linalg/prod.hpp"
#include "viennacl/linalg/nmf.hpp"

#include "examples/benchmarks/benchmark-utils.hpp"

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

    viennacl::matrix<ScalarType> v_nmf(m, n);
    viennacl::matrix<ScalarType> w_nmf(m, k);
    viennacl::matrix<ScalarType> h_nmf(k, n);

    fill_random(stl_w);
    fill_random(stl_h);

    viennacl::fast_copy(&stl_w[0], &stl_w[0] + stl_w.size(), w_ref);
    viennacl::fast_copy(&stl_h[0], &stl_h[0] + stl_h.size(), h_ref);

    v_ref = viennacl::linalg::prod(w_ref, h_ref);

    viennacl::ocl::get_queue().finish();

    //Timer timer;
    //timer.start();

    viennacl::linalg::nmf(v_ref, w_nmf, h_nmf, k);
    viennacl::ocl::get_queue().finish();

    //double time_spent = timer.get();

    v_nmf = viennacl::linalg::prod(w_nmf, h_nmf);

    float diff  = matrix_compare(v_ref, v_nmf);
    bool diff_ok = fabs(diff) < EPS;

    printf("%6s [%lux%lux%lu] diff = %.6f\n", diff_ok?"[[OK]]":"[FAIL]", m, k, n, diff);
}

int main()
{
    srand(time(NULL));

    test_nmf(3, 3, 3);
    test_nmf(3, 2, 3);
    test_nmf(16, 7, 12);
    test_nmf(160, 73, 200);
    test_nmf(1000, 15, 1000);

    return 0;
}
