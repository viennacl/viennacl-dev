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


/** \file tests/src/qr_method_func.cpp  Tests the individual building blocks of the eigenvalue routines based on the QR method.
*   \test Tests the individual building blocks of the eigenvalue routines based on the QR method.
**/

/*
*
*   Test file for qr-method
*
*/

// include necessary system headers
#include <iostream>

#ifndef NDEBUG
  #define NDEBUG
#endif

#define VIENNACL_WITH_UBLAS

#include "viennacl/scalar.hpp"
#include "viennacl/vector.hpp"
#include "viennacl/linalg/prod.hpp"

#include <fstream>
#include <iomanip>
#include "viennacl/linalg/qr-method.hpp"
#include "viennacl/linalg/qr-method-common.hpp"
#include "viennacl/linalg/matrix_operations.hpp"
#include "viennacl/tools/random.hpp"

#define EPS 10.0e-3


namespace ublas = boost::numeric::ublas;
typedef float     ScalarType;

void read_matrix_size(std::fstream& f, std::size_t& sz);

void read_matrix_size(std::fstream& f, std::size_t& sz)
{
    if(!f.is_open())
    {
        throw std::invalid_argument("File is not opened");
    }

    f >> sz;
}
template <typename MatrixLayout>
void read_matrix_body(std::fstream& f, viennacl::matrix<ScalarType, MatrixLayout>& A)
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

void matrix_print(viennacl::matrix<ScalarType>& A_orig);

void matrix_print(viennacl::matrix<ScalarType>& A_orig)
{
    ublas::matrix<ScalarType> A(A_orig.size1(), A_orig.size2());
    viennacl::copy(A_orig, A);
    for (unsigned int i = 0; i < A.size1(); i++) {
        for (unsigned int j = 0; j < A.size2(); j++)
           std::cout << std::setprecision(6) << std::fixed << A(i, j) << "\t";
        std::cout << std::endl;
    }
    std::cout << std::endl;
}

void matrix_print(ublas::matrix<ScalarType>& A);

void matrix_print(ublas::matrix<ScalarType>& A)
{
    for (unsigned int i = 0; i < A.size1(); i++) {
        for (unsigned int j = 0; j < A.size2(); j++)
           std::cout << std::setprecision(6) << std::fixed << A(i, j) << "\t";
        std::cout << std::endl;
    }
    std::cout << std::endl;
}

void vector_print(std::vector<ScalarType>& v );

void vector_print(std::vector<ScalarType>& v )
{
    for (unsigned int i = 0; i < v.size(); i++)
      std::cout << std::setprecision(6) << std::fixed << v[i] << ",\t";
    std::cout << "\n";
}


template <typename MatrixType, typename VCLMatrixType>
bool check_for_equality(MatrixType const & ublas_A, VCLMatrixType const & vcl_A)
{
  typedef typename MatrixType::value_type   value_type;

  ublas::matrix<value_type> vcl_A_cpu(vcl_A.size1(), vcl_A.size2());
  viennacl::backend::finish();  //workaround for a bug in APP SDK 2.7 on Trinity APUs (with Catalyst 12.8)
  viennacl::copy(vcl_A, vcl_A_cpu);

  for (std::size_t i=0; i<ublas_A.size1(); ++i)
  {
    for (std::size_t j=0; j<ublas_A.size2(); ++j)
    {
      if (std::abs(ublas_A(i,j) - vcl_A_cpu(i,j)) > EPS * std::max(std::abs(ublas_A(i, j)), std::abs(vcl_A_cpu(i, j))))
      {
        std::cout << "Error at index (" << i << ", " << j << "): " << ublas_A(i,j) << " vs. " << vcl_A_cpu(i,j) << std::endl;
        std::cout << std::endl << "TEST failed!" << std::endl;
        return false;
      }
    }
  }
  std::cout << "PASSED!" << std::endl;
  return true;
}

template <typename VectorType>
bool check_for_equality(VectorType const & vec_A, VectorType const & vec_B)
{

  for (std::size_t i=0; i<vec_A.size(); ++i)
  {
      if (std::abs(vec_A[i] - vec_B[i]) > EPS)
      {
        std::cout << "Error at index (" << i << "): " << vec_A[i] << " vs " <<vec_B[i] << std::endl;
        std::cout << std::endl << "TEST failed!" << std::endl;
        return false;
      }
  }
  std::cout << "PASSED!" << std::endl;
  return true;
}

void fill_vector(std::vector<ScalarType>& v);

void fill_vector(std::vector<ScalarType>& v)
{
  viennacl::tools::uniform_random_numbers<ScalarType> randomNumber;

  for (unsigned int i = 0; i < v.size(); ++i)
    v[i] =  randomNumber();
}

/*
 *
 * ------------Functions to be tested---------------
 *
 */


template <typename NumericT>
void house_update_A_left(ublas::matrix<NumericT> & A,
                         std::vector<NumericT> D,
                         unsigned int start)
{
  NumericT ss = 0;

  std::size_t row_start = start + 1;
  for(std::size_t i = 0; i < A.size2(); i++)
    {
      ss = 0;
      for(std::size_t j = row_start; j < A.size1(); j++)
          ss = ss +(D[j] * A(j, i));

      for(std::size_t j = row_start; j < A.size1(); j++)
          A(j, i) = A(j, i) - (2 * D[j] * ss);
    }
}

template <typename NumericT>
void house_update_A_right(ublas::matrix<NumericT> & A,
                          std::vector<NumericT> D)
{
  NumericT ss = 0;

  for(std::size_t i = 0; i < A.size1(); i++)
    {
      ss = 0;
      for(std::size_t j = 0; j < A.size2(); j++)
          ss = ss + (D[j] * A(i, j));

      NumericT sum_Av = ss;

      for(std::size_t j = 0; j < A.size2(); j++)
          A(i, j) = A(i, j) - (2 * D[j] * sum_Av);
    }
}


template <typename NumericT>
void house_update_QL(ublas::matrix<NumericT> & Q,
                     std::vector<NumericT> D,
                     std::size_t A_size1)

{
  NumericT beta = 2;
  ublas::matrix<NumericT> ubl_P(A_size1, A_size1);
  ublas::matrix<ScalarType> I = ublas::identity_matrix<ScalarType>(Q.size1());
  ublas::matrix<NumericT> Q_temp(Q.size1(), Q.size2());

  for(std::size_t i = 0; i < Q.size1(); i++)
  {
      for(std::size_t j = 0; j < Q.size2(); j++)
      {
          Q_temp(i, j) = Q(i, j);
      }
  }

  ubl_P = ublas::identity_matrix<NumericT>(A_size1);

  //scaled_rank_1 update
  for(std::size_t i = 0; i < A_size1; i++)
  {
      for(std::size_t j = 0; j < A_size1; j++)
      {
          ubl_P(i, j) = I(i, j) - beta * (D[i] * D[j]);
      }
  }
  Q = ublas::prod(Q_temp, ubl_P);
}

template <typename NumericT>
void givens_next(ublas::matrix<NumericT> & Q,
                 std::vector<NumericT> & tmp1,
                 std::vector<NumericT> & tmp2,
                 int l,
                 int m)
{
  for(int i2 = m - 1; i2 >= l; i2--)
  {
    std::size_t i = static_cast<std::size_t>(i2);
    for(std::size_t k = 0; k < Q.size1(); k++)
    {
      NumericT h = Q(k, i+1);
      Q(k, i+1) = tmp2[i] * Q(k, i) + tmp1[i]*h;
      Q(k, i) = tmp1[i] * Q(k, i) - tmp2[i]*h;
    }
  }
}


template <typename NumericT>
void copy_vec(ublas::matrix<NumericT>& A,
              std::vector<NumericT> & V,
              std::size_t row_start,
              std::size_t col_start,
              bool copy_col)
{
  if(copy_col)
  {
      for(std::size_t i = row_start; i < A.size1(); i++)
      {
         V[i - row_start] = A(i, col_start);
      }
  }
  else
  {
      for(std::size_t i = col_start; i < A.size1(); i++)
      {
         V[i - col_start] = A(row_start, i);
      }
  }
}

template <typename NumericT>
void bidiag_pack(ublas::matrix<NumericT> & A,
                 std::vector<NumericT> & D,
                 std::vector<NumericT> & S)

{
  std::size_t size = std::min(D.size(), S.size());
  std::size_t i = 0;
  for(i = 0;  i < size - 1; i++)
  {
      D[i] = A(i, i);
      S[i + 1] = A(i, i + 1);
  }
  D[size - 1] = A(size - 1, size - 1);
}


template <typename MatrixLayout>
void test_qr_method_sym(const std::string& fn)
{
  std::cout << "Reading..." << std::endl;
  std::size_t sz;

  // read file
  std::fstream f(fn.c_str(), std::fstream::in);
  //read size of input matrix
  read_matrix_size(f, sz);

  viennacl::matrix<ScalarType, MatrixLayout> vcl_A(sz, sz), vcl_Q(sz, sz);
  viennacl::vector<ScalarType> vcl_D(sz), vcl_E(sz), vcl_F(sz), vcl_G(sz), vcl_H(sz);
  std::vector<ScalarType> std_D(sz), std_E(sz), std_F(sz), std_G(sz), std_H(sz);
  ublas::matrix<ScalarType> ubl_A(sz, sz), ubl_Q(sz, sz);


  std::cout << "Testing matrix of size " << sz << "-by-" << sz << std::endl << std::endl;

  read_matrix_body(f, vcl_A);
  f.close();
  viennacl::copy(vcl_A, ubl_A);

  fill_vector(std_D);
  copy(std_D, vcl_D);
//--------------------------------------------------------
  std::cout << std::endl << "Testing house_update_left..." << std::endl;
  viennacl::linalg::house_update_A_left(vcl_A, vcl_D, 0);
  house_update_A_left(ubl_A, std_D, 0);

  if(!check_for_equality(ubl_A, vcl_A))
    exit(EXIT_FAILURE);
//--------------------------------------------------------
  std::cout << std::endl << "Testing house_update_right..." << std::endl;
  copy(ubl_A, vcl_A);
  copy(std_D, vcl_D);
  viennacl::linalg::house_update_A_right(vcl_A, vcl_D);
  house_update_A_right(ubl_A, std_D);

  if(!check_for_equality(ubl_A, vcl_A))
     exit(EXIT_FAILURE);
//--------------------------------------------------------

  std::cout << std::endl << "Testing house_update_QL..." << std::endl;
  ubl_Q = ublas::identity_matrix<ScalarType>(ubl_Q.size1());
  copy(ubl_Q, vcl_Q);
  copy(ubl_A, vcl_A);
  copy(std_D, vcl_D);
  viennacl::linalg::house_update_QL(vcl_Q, vcl_D, vcl_A.size1());
  house_update_QL(ubl_Q, std_D, ubl_A.size1());
  if(!check_for_equality(ubl_Q, vcl_Q))
     exit(EXIT_FAILURE);
//--------------------------------------------------------

  std::cout << std::endl << "Testing givens next..." << std::endl;
  fill_vector(std_E);
  fill_vector(std_F);
  copy(std_E, vcl_E);
  copy(std_F, vcl_F);
  copy(ubl_Q, vcl_Q);
  copy(ubl_A, vcl_A);
  viennacl::linalg::givens_next(vcl_Q, vcl_E, vcl_F, 2, 5);
  givens_next(ubl_Q, std_E, std_F, 2, 5);
  if(!check_for_equality(ubl_Q, vcl_Q))
      exit(EXIT_FAILURE);
//--------------------------------------------------------
  std::cout << std::endl << "Testing copy vec..." << std::endl;
  viennacl::linalg::copy_vec(vcl_A, vcl_D, 0, 2, 1);
  copy_vec(ubl_A, std_D, 0, 2, 1);
  copy(vcl_D, std_E); //check for equality only for same vector types
  if(!check_for_equality(std_D, std_E))
      exit(EXIT_FAILURE);

//--------------------------------------------------------
  std::cout << std::endl << "Testing bidiag pack..." << std::endl;
  viennacl::linalg::bidiag_pack(vcl_A, vcl_D, vcl_F);
  vcl_F[0] = 0;  // first element in superdiagonal is irrelevant.
  bidiag_pack(ubl_A, std_G, std_H);
  std_H[0] = 0;
  copy(std_G, vcl_G);
  copy(std_H, vcl_H);

  if(!check_for_equality(vcl_D, vcl_G))
      exit(EXIT_FAILURE);
  if(!check_for_equality(vcl_F, vcl_H))
      exit(EXIT_FAILURE);
//--------------------------------------------------------
}

int main()
{

  std::cout << std::endl << "Test qr_method_sym for row_major matrix" << std::endl;
  test_qr_method_sym<viennacl::row_major>("../examples/testdata/eigen/symm5.example");

  std::cout << std::endl << "Test qr_method_sym for column_major matrix" << std::endl;
  test_qr_method_sym<viennacl::column_major>("../examples/testdata/eigen/symm5.example");


  std::cout << std::endl <<"--------TEST SUCCESSFULLY COMPLETED----------" << std::endl;
}
