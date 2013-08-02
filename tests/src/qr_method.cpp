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

/*
Solutions for testdata were generated with Scilab line:

M=fscanfMat('nsm1.example');e=spec(M);e=gsort(e);rr=real(e);ii=imag(e);e=cat(1, rr, ii); s=strcat(string(e), ' ');write('tmp', s);
*/

#ifndef NDEBUG
  #define NDEBUG
#endif

//#define VIENNACL_DEBUG_ALL
#include <iostream>
#include <fstream>
#include <stdexcept>
#include <vector>

#include "viennacl/linalg/prod.hpp"
#include "viennacl/linalg/qr-method.hpp"

#include <examples/benchmarks/benchmark-utils.hpp>

#include <boost/numeric/ublas/vector.hpp>
#include <boost/numeric/ublas/matrix.hpp>

namespace ublas = boost::numeric::ublas;

typedef float ScalarType;

const ScalarType EPS = 0.00001f;

void read_matrix_size(std::fstream& f, std::size_t& sz)
{
    if(!f.is_open())
    {
        throw std::invalid_argument("File is not opened");
    }

    f >> sz;
}

void read_matrix_body(std::fstream& f, viennacl::matrix<ScalarType>& A)
{
    if(!f.is_open())
    {
        throw std::invalid_argument("File is not opened");
    }

    boost::numeric::ublas::matrix<ScalarType> h_A(A.size1(), A.size2());

    for(std::size_t i = 0; i < h_A.size1(); i++) {
        for(std::size_t j = 0; j < h_A.size2(); j++) {
            ScalarType val = 0.0;
            f >> val;
            h_A(i, j) = val;
        }
    }

    viennacl::copy(h_A, A);
}

void read_vector_body(std::fstream& f, ublas::vector<ScalarType>& v) {
    if(!f.is_open())
        throw std::invalid_argument("File is not opened");

    for(std::size_t i = 0; i < v.size(); i++)
    {
            ScalarType val = 0.0;
            f >> val;
            v[i] = val;
    }
}

bool check_tridiag(viennacl::matrix<ScalarType>& A_orig)
{
    ublas::matrix<ScalarType> A(A_orig.size1(), A_orig.size2());
    viennacl::copy(A_orig, A);

    for (unsigned int i = 0; i < A.size1(); i++) {
        for (unsigned int j = 0; j < A.size2(); j++) {
            if ((std::abs(A(i, j)) > EPS) && ((i - 1) != j) && (i != j) && ((i + 1) != j))
            {
                // std::cout << "Failed at " << i << " " << j << " " << A(i, j) << "\n";
                return false;
            }
        }
    }
    return true;
}

bool check_hessenberg(viennacl::matrix<ScalarType>& A_orig)
{
    ublas::matrix<ScalarType> A(A_orig.size1(), A_orig.size2());
    viennacl::copy(A_orig, A);

    for (std::size_t i = 0; i < A.size1(); i++) {
        for (std::size_t j = 0; j < A.size2(); j++) {
            if ((std::abs(A(i, j)) > EPS) && (i > (j + 1)))
            {
                // std::cout << "Failed at " << i << " " << j << " " << A(i, j) << "\n";
                return false;
            }
        }
    }
    return true;
}

ScalarType matrix_compare(ublas::matrix<ScalarType>& res,
                            ublas::matrix<ScalarType>& ref)
{
    ScalarType diff = 0.0;
    ScalarType mx = 0.0;

    for(std::size_t i = 0; i < res.size1(); i++)
    {
        for(std::size_t j = 0; j < res.size2(); j++)
        {
            diff = std::max(diff, std::abs(res(i, j) - ref(i, j)));
            mx = std::max(mx, res(i, j));
        }
    }

    return diff / mx;
}

ScalarType vector_compare(ublas::vector<ScalarType>& res,
                          ublas::vector<ScalarType>& ref)
{
    std::sort(ref.begin(), ref.end());
    std::sort(res.begin(), res.end());

    ScalarType diff = 0.0;
    ScalarType mx = 0.0;
    for(size_t i = 0; i < ref.size(); i++)
    {
        diff = std::max(diff, std::abs(res[i] - ref[i]));
        mx = std::max(mx, res[i]);
    }

    return diff / mx;
}

void test_eigen(const std::string& fn, bool is_symm)
{
    std::cout << "Reading..." << "\n";
    std::size_t sz;
    // read file
    std::fstream f(fn.c_str(), std::fstream::in);
    //read size of input matrix
    read_matrix_size(f, sz);

    viennacl::matrix<ScalarType> A_input(sz, sz), A_ref(sz, sz), Q(sz, sz);
    ublas::vector<ScalarType> eigen_ref_re(sz, 0), eigen_ref_im(sz, 0), eigen_re(sz, 0), eigen_im(sz, 0);

    read_matrix_body(f, A_input);

    read_vector_body(f, eigen_ref_re);

    if(!is_symm)
        read_vector_body(f, eigen_ref_im);

    f.close();

    A_ref = A_input;

    std::cout << "Calculation..." << "\n";

    Timer timer;
    timer.start();

    if(is_symm)
        viennacl::linalg::qr_method_sym(A_input, Q, eigen_re);
    else
        viennacl::linalg::qr_method_nsm(A_input, Q, eigen_re, eigen_im);

    // std::cout << A_input << "\n";
    viennacl::backend::finish();

    double time_spend = timer.get();

    std::cout << "Verification..." << "\n";

    bool is_hessenberg = check_hessenberg(A_input);
    bool is_tridiag = check_tridiag(A_input);

    ublas::matrix<ScalarType> result1(sz, sz), result2(sz, sz), tmp(sz, sz);
    viennacl::copy(A_ref, tmp);
    viennacl::copy(A_input, result1);
    viennacl::copy(Q, result2);

    result1 = ublas::prod(result2, result1);
    result2 = ublas::prod(tmp, result2);

    ScalarType prods_diff = matrix_compare(result1, result2);
    ScalarType eigen_diff = vector_compare(eigen_ref_re, eigen_re);

    bool is_ok = is_hessenberg;

    if(is_symm)
        is_ok = is_ok && is_tridiag;

    is_ok = is_ok && (eigen_diff < EPS);
    is_ok = is_ok && (prods_diff < EPS);

    // std::cout << A_ref << "\n";
    // std::cout << A_input << "\n";
    // std::cout << Q << "\n";
    // std::cout << eigen_re << "\n";
    // std::cout << eigen_im << "\n";
    // std::cout << eigen_ref_re << "\n";
    // std::cout << eigen_ref_im << "\n";

    // std::cout << result1 << "\n";
    // std::cout << result2 << "\n";
    // std::cout << eigen_ref << "\n";
    // std::cout << eigen << "\n";

    printf("%6s [%dx%d] %40s time = %.4f\n", is_ok?"[[OK]]":"[FAIL]", (int)A_ref.size1(), (int)A_ref.size2(), fn.c_str(), time_spend);
    printf("tridiagonal = %d, hessenberg = %d prod-diff = %f eigen-diff = %f\n", is_tridiag, is_hessenberg, prods_diff, eigen_diff);
}

int main()
{
  // test_eigen("../../examples/testdata/eigen/symm1.example", true);
  // test_eigen("../../examples/testdata/eigen/symm2.example", true);
  // test_eigen("../../examples/testdata/eigen/symm3.example", true);

  test_eigen("../../examples/testdata/eigen/nsm1.example", false);
  test_eigen("../../examples/testdata/eigen/nsm2.example", false);
  test_eigen("../../examples/testdata/eigen/nsm3.example", false);
  test_eigen("../../examples/testdata/eigen/nsm4.example", false);

  std::cout << std::endl;
  std::cout << "------- Test completed --------" << std::endl;
  std::cout << std::endl;

  return EXIT_SUCCESS;
}
