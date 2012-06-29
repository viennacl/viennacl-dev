#include <stdexcept>
#include <iostream>
#include <string>
#include <vector>
#include <cmath>

#include "viennacl/matrix.hpp"
#include "viennacl/linalg/prod.hpp"

#include "viennacl/linalg/svd.hpp"

#include "examples/benchmarks/benchmark-utils.hpp"

typedef float ScalarType;

const float EPS = 0.001;

void read_matrix_size(std::fstream& f, unsigned int& sz1, unsigned int& sz2) {
    if(!f.is_open())
        throw std::invalid_argument("File is not opened");

    f >> sz1 >> sz2;
}

void read_matrix_body(std::fstream& f, viennacl::matrix<ScalarType>& A) {
    if(!f.is_open())
        throw std::invalid_argument("File is not opened");

	boost::numeric::ublas::matrix<float> h_A(A.size1(), A.size2());

    for(unsigned int i = 0; i < h_A.size1(); i++) {
        for(unsigned int j = 0; j < h_A.size2(); j++) {
            ScalarType val = 0.0;
            f >> val;
            h_A(i, j) = val;
        }
    }

	viennacl::copy(h_A, A);
}

void read_vector_body(std::fstream& f, std::vector<ScalarType>& v) {
    if(!f.is_open())
        throw std::invalid_argument("File is not opened");

    for(unsigned int i = 0; i < v.size(); i++)
    {
            ScalarType val = 0.0;
            f >> val;
            v[i] = val;
    }
}

void random_fill(std::vector<ScalarType>& in) {
    for(unsigned int i = 0; i < in.size(); i++) {
        in[i] = (float)rand() / RAND_MAX;
    }
}

bool check_bidiag(viennacl::matrix<ScalarType>& A) {
    const float EPS = 0.0001f;

    std::vector<ScalarType> aA(A.size1() * A.size2());
    viennacl::fast_copy(A, &aA[0]);

    for(unsigned int i = 0; i < A.size1(); i++) {
        for(unsigned int j = 0; j < A.size2(); j++) {
            ScalarType val = aA[i * A.size2() + j];
            if((fabs(val) > EPS) && (i != j) && ((i + 1) != j)) {
                std::cout << "Failed at " << i << " " << j << " " << val << std::endl;
                return false;
            }
        }
    }

    return true;
}

float matrix_compare(viennacl::matrix<ScalarType>& res,
                     viennacl::matrix<ScalarType>& ref) 
{
    std::vector<ScalarType> res_std(res.internal_size());
    std::vector<ScalarType> ref_std(ref.internal_size());

    viennacl::fast_copy(res, &res_std[0]);
    viennacl::fast_copy(ref, &ref_std[0]);

    float diff = 0.0;
    float mx = 0.0;

    for(unsigned int i = 0; i < res_std.size(); i++) {
        diff = std::max(diff, std::abs(res_std[i] - ref_std[i]));
        mx = std::max(mx, res_std[i]);
    }

    return diff / mx;
}

float sigmas_compare(viennacl::matrix<ScalarType>& res, 
                        std::vector<ScalarType>& ref) 
{
    std::vector<ScalarType> res_std(ref.size());

    for(size_t i = 0; i < ref.size(); i++)
    {
        res_std[i] = res(i, i);
    }

    std::sort(ref.begin(), ref.end());
    std::sort(res_std.begin(), res_std.end());

    float diff = 0.0;
    float mx = 0.0;
    for(size_t i = 0; i < ref.size(); i++) 
    {
        diff = std::max(diff, std::abs(res_std[i] - ref[i]));
        mx = std::max(mx, res_std[i]);
    }

    return diff / mx;
}


void test_svd(const std::string& fn) 
{
    unsigned int sz1, sz2;

    //read matrix

    // sz1 = 2048, sz2 = 2048;
    // std::vector<ScalarType> in(sz1 * sz2);
    // random_fill(in);

    // read file
    std::fstream f(fn.c_str(), std::fstream::in);
    //read size of input matrix
    read_matrix_size(f, sz1, sz2);

    unsigned int to = std::min(sz1, sz2);

    viennacl::matrix<ScalarType> Ai(sz1, sz2), Aref(sz1, sz2), QL(sz1, sz1), QR(sz2, sz2);
    read_matrix_body(f, Ai);

    std::vector<ScalarType> sigma_ref(to);
    read_vector_body(f, sigma_ref);

    f.close();

    // viennacl::fast_copy(&in[0], &in[0] + in.size(), Ai);

    Aref = Ai;

    Timer timer;
    timer.start();

    viennacl::linalg::svd(Ai, QL, QR);

    viennacl::ocl::get_queue().finish();

    double time_spend = timer.get();

    viennacl::matrix<ScalarType> result1(sz1, sz2), result2(sz1, sz2);
    result1 = viennacl::linalg::prod(QL, Ai);
    result2 = viennacl::linalg::prod(result1, trans(QR));

    float sigma_diff = sigmas_compare(Ai, sigma_ref);
    float prods_diff  = matrix_compare(result2, Aref);

    bool sigma_ok = (fabs(sigma_diff) < EPS) && (fabs(prods_diff) < EPS);

	printf("%6s [%dx%d] %40s sigma_diff = %.6f; prod_diff = %.6f; time = %.6f\n", sigma_ok?"[[OK]]":"[FAIL]", (int)Aref.size1(), (int)Aref.size2(), fn.c_str(), sigma_diff, prods_diff, time_spend);
}


void time_svd(size_t sz1, size_t sz2) 
{

    std::vector<ScalarType> in(sz1 * sz2);
    random_fill(in);

    viennacl::matrix<ScalarType> Ai(sz1, sz2), QL(sz1, sz1), QR(sz2, sz2);

    viennacl::fast_copy(&in[0], &in[0] + in.size(), Ai);


    Timer timer;
    timer.start();

    viennacl::linalg::svd(Ai, QL, QR);

    viennacl::ocl::get_queue().finish();

    double time_spend = timer.get();

    printf("[%dx%d] time = %.6f\n", (int)sz1, (int)sz2, time_spend);
}

int main() 
{

    test_svd(std::string("../../examples/testdata/svd/qr.example"));
    test_svd(std::string("../../examples/testdata/svd/wiki.example"));
    test_svd(std::string("../../examples/testdata/svd/wiki.qr.example"));
    test_svd(std::string("../../examples/testdata/svd/pysvd.example"));
    test_svd(std::string("../../examples/testdata/svd/random.example"));

    time_svd(500, 500);
    time_svd(1000, 1000);
    time_svd(4096, 512);
    time_svd(2048, 2048);
    //time_svd(4096, 4096);  //takes too long for a standard sanity test. Feel free to uncomment

    return EXIT_SUCCESS;
}
