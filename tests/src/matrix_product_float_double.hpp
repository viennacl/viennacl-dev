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

#ifndef TEST_MATRIX_PRODUCT_FLOAT_DOUBLE_HPP_
#define TEST_MATRIX_PRODUCT_FLOAT_DOUBLE_HPP_

// We don't need debug mode in UBLAS:
#define BOOST_UBLAS_NDEBUG

#include <cstddef>

#include "viennacl/matrix.hpp"
#include "viennacl/matrix_proxy.hpp"
#include "viennacl/linalg/prod.hpp"

#include "boost/numeric/ublas/matrix.hpp"
#include "boost/numeric/ublas/matrix_proxy.hpp"
#include "boost/numeric/ublas/io.hpp"

#include "viennacl/tools/random.hpp"

template<typename ScalarType, typename VCLMatrixType>
ScalarType diff(boost::numeric::ublas::matrix<ScalarType> const & mat1, VCLMatrixType  const & mat2)
{
   boost::numeric::ublas::matrix<ScalarType> mat2_cpu(mat2.size1(), mat2.size2());
   viennacl::backend::finish();  //workaround for a bug in APP SDK 2.7 on Trinity APUs (with Catalyst 12.8)
   viennacl::copy(mat2, mat2_cpu);
   ScalarType ret = 0;
   ScalarType act = 0;

    for (unsigned int i = 0; i < mat2_cpu.size1(); ++i)
    {
      for (unsigned int j = 0; j < mat2_cpu.size2(); ++j)
      {
         act = std::fabs(mat2_cpu(i,j) - mat1(i,j)) / std::max( std::fabs(mat2_cpu(i, j)), std::fabs(mat1(i,j)) );
         if (act > ret)
           ret = act;
      }
    }

    return ret;
}


template<class UBlasType, class F>
struct matrix_maker;

template<class T, class F>
struct matrix_maker< boost::numeric::ublas::matrix<T>, F>
{
  typedef viennacl::matrix<T, F> result_type;
  static result_type make(viennacl::matrix<T, F> const &, boost::numeric::ublas::matrix<T> & base)
  {
    viennacl::matrix<T, F> result(base.size1(), base.size2());
    viennacl::copy(base, result);
    return result;
  }
};

template<class MatrixT, class F>
struct matrix_maker< boost::numeric::ublas::matrix_range<MatrixT>, F>
{
  typedef typename MatrixT::value_type T;
  typedef viennacl::matrix_range< viennacl::matrix<T, F> > result_type;

  static result_type make(viennacl::matrix<T, F> & M, boost::numeric::ublas::matrix_range<MatrixT> & base)
  {
    viennacl::range r0(base.start1(), base.start1() + base.size1());
    viennacl::range r1(base.start2(), base.start2() + base.size2());
    result_type result(M, r0, r1);
    viennacl::copy(base, result);
    return result;
  }
};

template<class MatrixT, class F>
struct matrix_maker< boost::numeric::ublas::matrix_slice<MatrixT>, F>
{
  typedef typename MatrixT::value_type T;
  typedef viennacl::matrix_slice< viennacl::matrix<T, F> > result_type;

  static result_type make(viennacl::matrix<T, F> & M, boost::numeric::ublas::matrix_slice<MatrixT> & base)
  {
    viennacl::slice s0(base.start1(), std::size_t(base.stride1()), base.size1());
    viennacl::slice s1(base.start2(), std::size_t(base.stride2()), base.size2());
    result_type result(M, s0, s1);
    viennacl::copy(base, result);
    return result;
  }
};

template<typename T, typename CType, typename AType, typename BType>
int test_layout(CType & C, AType const & A, AType const & AT, BType const & B, BType const & BT,
                boost::numeric::ublas::matrix<T> const & ground, T epsilon, bool with_composite)
{
  using viennacl::linalg::prod;
  using viennacl::trans;

  std::cout << "C = A.B" << std::endl;
  C = prod(A, B);
  if (diff(ground, C)>epsilon)
    return EXIT_FAILURE;

  std::cout << "C = A'.B" << std::endl;
  C = prod(trans(AT), B);
  if (diff(ground, C)>epsilon)
    return EXIT_FAILURE;

  std::cout << "C = A.B'" << std::endl;
  C = prod(A, trans(BT));
  if (diff(ground, C)>epsilon)
    return EXIT_FAILURE;

  std::cout << "C = A'.B'" << std::endl;
  C = prod(trans(AT), trans(BT));
  if (diff(ground, C)>epsilon)
    return EXIT_FAILURE;

  // composite operations:
  if (with_composite)
  {
    boost::numeric::ublas::matrix<T> ground2 = T(2) * ground;

    std::cout << "C = (A + A).B" << std::endl;
    C = prod(A + A, B);
    if (diff(ground2, C)>epsilon)
      return EXIT_FAILURE;

    std::cout << "C = trans(AT + AT).B" << std::endl;
    C = prod(viennacl::trans(AT + AT), B);
    if (diff(ground2, C)>epsilon)
      return EXIT_FAILURE;

    std::cout << "C = A.(B + B)" << std::endl;
    C = prod(A, T(2) * B);
    if (diff(ground2, C)>epsilon)
      return EXIT_FAILURE;

    std::cout << "C = A.trans(BT + BT)" << std::endl;
    C = prod(A, trans(BT + BT));
    if (diff(ground2, C)>epsilon)
      return EXIT_FAILURE;

    std::cout << "C = (A + A).(B + B)" << std::endl;
    C = T(0.25) * prod(A + A, B + B);
    if (diff(ground, C)>epsilon)
      return EXIT_FAILURE;

    std::cout << "C = trans(AT + AT).trans(BT + BT)" << std::endl;
    C += prod(trans(AT + AT), trans(BT + BT));
    C -= prod(trans(AT + AT), trans(BT + BT));
    if (diff(ground, C)>epsilon)
      return EXIT_FAILURE;

  }

  return EXIT_SUCCESS;
}

template<typename T, typename RefAType, typename RefBType, typename RefCType>
int test_all_layouts(std::size_t CM, std::size_t CN, RefCType & cC,
                     std::size_t AM, std::size_t AK, RefAType & cA, RefAType & cAT,
                     std::size_t BK, std::size_t BN, RefBType & cB,  RefBType & cBT,
                     T epsilon)
{
  viennacl::matrix<T, viennacl::row_major> ArowTmp(AM, AK);
  viennacl::matrix<T, viennacl::row_major> ATrowTmp(AK, AM);
  viennacl::matrix<T, viennacl::row_major> BrowTmp(BK, BN);
  viennacl::matrix<T, viennacl::row_major> BTrowTmp(BN, BK);
  viennacl::matrix<T, viennacl::row_major> CrowTmp(CM, CN);

  viennacl::matrix<T, viennacl::column_major> AcolTmp(AM, AK);
  viennacl::matrix<T, viennacl::column_major> ATcolTmp(AK, AM);
  viennacl::matrix<T, viennacl::column_major> BcolTmp(BK, BN);
  viennacl::matrix<T, viennacl::column_major> BTcolTmp(BN, BK);
  viennacl::matrix<T, viennacl::column_major> CcolTmp(CM, CN);


  typename matrix_maker<RefCType, viennacl::row_major>::result_type Crow = matrix_maker<RefCType, viennacl::row_major>::make(CrowTmp, cC);
  typename matrix_maker<RefAType, viennacl::row_major>::result_type Arow = matrix_maker<RefAType, viennacl::row_major>::make(ArowTmp, cA);
  typename matrix_maker<RefAType, viennacl::row_major>::result_type ATrow = matrix_maker<RefAType, viennacl::row_major>::make(ATrowTmp, cAT);
  typename matrix_maker<RefBType, viennacl::row_major>::result_type Brow = matrix_maker<RefBType, viennacl::row_major>::make(BrowTmp, cB);
  typename matrix_maker<RefBType, viennacl::row_major>::result_type BTrow = matrix_maker<RefBType, viennacl::row_major>::make(BTrowTmp, cBT);

  typename matrix_maker<RefCType, viennacl::column_major>::result_type Ccol = matrix_maker<RefCType, viennacl::column_major>::make(CcolTmp, cC);
  typename matrix_maker<RefAType, viennacl::column_major>::result_type Acol = matrix_maker<RefAType, viennacl::column_major>::make(AcolTmp, cA);
  typename matrix_maker<RefAType, viennacl::column_major>::result_type ATcol = matrix_maker<RefAType, viennacl::column_major>::make(ATcolTmp, cAT);
  typename matrix_maker<RefBType, viennacl::column_major>::result_type Bcol = matrix_maker<RefBType, viennacl::column_major>::make(BcolTmp, cB);
  typename matrix_maker<RefBType, viennacl::column_major>::result_type BTcol = matrix_maker<RefBType, viennacl::column_major>::make(BTcolTmp, cBT);


  boost::numeric::ublas::matrix<T> ground = boost::numeric::ublas::prod(cA, cB);

#define TEST_LAYOUT(Clayout, Alayout, Blayout, composite) \
  std::cout << "> "  #Clayout " = " #Alayout "." #Blayout << std::endl;  \
  if (test_layout(C ## Clayout, A ## Alayout, AT ## Alayout, B ## Blayout, BT ## Blayout, ground, epsilon, composite) != EXIT_SUCCESS) \
    return EXIT_FAILURE; \

  TEST_LAYOUT(row, row, row, true);
  TEST_LAYOUT(row, row, col, false);
  TEST_LAYOUT(row, col, row, false);
  TEST_LAYOUT(row, col, col, false);
  TEST_LAYOUT(col, row, row, false);
  TEST_LAYOUT(col, row, col, false);
  TEST_LAYOUT(col, col, row, false);
  TEST_LAYOUT(col, col, col, true);

#undef TEST_LAYOUT

  return EXIT_SUCCESS;
}

template<class MatrixType>
void init_rand(MatrixType & A)
{
  typedef typename MatrixType::value_type T;

  viennacl::tools::uniform_random_numbers<T> randomNumber;

  for (unsigned int i = 0; i < A.size1(); ++i)
    for (unsigned int j = 0; j < A.size2(); ++j)
      A(i, j) = static_cast<T>(0.1) * randomNumber();
}

template<typename T>
int run_test(T epsilon)
{
    typedef boost::numeric::ublas::range range_type;
    typedef boost::numeric::ublas::slice slice_type;
    typedef boost::numeric::ublas::matrix<T> matrix_type;
    typedef boost::numeric::ublas::matrix_range<matrix_type> matrix_range_type;
    typedef boost::numeric::ublas::matrix_slice<matrix_type> matrix_slice_type;

    typedef typename matrix_type::difference_type difference_type;

    std::size_t matrix_holder_M = 143;
    std::size_t matrix_holder_N = 124;
    std::size_t matrix_holder_K = 184;

    std::size_t start_M = 14;
    std::size_t start_N = 20;
    std::size_t start_K = 73;

    std::size_t range_holder_M = start_M + matrix_holder_M;
    std::size_t range_holder_N = start_N + matrix_holder_N;
    std::size_t range_holder_K = start_K + matrix_holder_K;

    range_type range_M(start_M, range_holder_M);
    range_type range_N(start_N, range_holder_N);
    range_type range_K(start_K, range_holder_K);

    difference_type stride_M = 9;
    difference_type stride_N = 13;
    difference_type stride_K = 4;

    std::size_t slice_holder_M = start_M + std::size_t(stride_M)*matrix_holder_M;
    std::size_t slice_holder_N = start_N + std::size_t(stride_N)*matrix_holder_N;
    std::size_t slice_holder_K = start_K + std::size_t(stride_K)*matrix_holder_K;

    slice_type slice_M(start_M, stride_M, matrix_holder_M);
    slice_type slice_N(start_N, stride_N, matrix_holder_N);
    slice_type slice_K(start_K, stride_K, matrix_holder_K);

#define DECLARE(NAME, size1, size2) \
    matrix_type NAME ## _matrix(matrix_holder_ ## size1, matrix_holder_ ## size2);\
    init_rand(NAME ## _matrix);\
    matrix_type NAME ## T_matrix = boost::numeric::ublas::trans(NAME ## _matrix);\
    \
    matrix_type NAME ## _range_holder(range_holder_ ## size1, range_holder_ ## size2);\
    init_rand(NAME ## _range_holder);\
    matrix_range_type NAME ## _range(NAME ## _range_holder, range_ ## size1, range_ ## size2);\
    matrix_type NAME ## T_range_holder = boost::numeric::ublas::trans(NAME ## _range_holder);\
    matrix_range_type NAME ## T_range(NAME ## T_range_holder, range_ ## size2, range_ ## size1);\
   \
    matrix_type NAME ## _slice_holder(slice_holder_ ## size1, slice_holder_ ## size2);\
    init_rand(NAME ## _slice_holder);\
    matrix_slice_type NAME ## _slice(NAME ## _slice_holder, slice_ ## size1, slice_ ## size2);\
    matrix_type NAME ## T_slice_holder = boost::numeric::ublas::trans(NAME ## _slice_holder);\
    matrix_slice_type NAME ## T_slice(NAME ## T_slice_holder, slice_ ## size2, slice_ ## size1);\

    DECLARE(A, M, K);
    DECLARE(B, K, N);
    DECLARE(C, M, N);
#undef DECLARE

#define TEST_ALL_LAYOUTS(C_TYPE, A_TYPE, B_TYPE)\
    std::cout << ">> " #C_TYPE " = " #A_TYPE "." #B_TYPE << std::endl;\
    if (test_all_layouts<T>(C_TYPE ## _holder_M, C_TYPE ## _holder_N, C_ ## C_TYPE,\
                            A_TYPE ## _holder_M, A_TYPE ## _holder_K, A_ ## A_TYPE, AT_ ## A_TYPE,\
                            B_TYPE ## _holder_K, B_TYPE ## _holder_N, B_ ## B_TYPE, BT_ ## B_TYPE, epsilon) != EXIT_SUCCESS)\
      return EXIT_FAILURE;\

//    //C=matrix
    TEST_ALL_LAYOUTS(matrix, matrix, matrix)
    TEST_ALL_LAYOUTS(matrix, matrix, range)
    TEST_ALL_LAYOUTS(matrix, matrix, slice)

    TEST_ALL_LAYOUTS(matrix, range, matrix)
    TEST_ALL_LAYOUTS(matrix, range, range)
    TEST_ALL_LAYOUTS(matrix, range, slice)

    TEST_ALL_LAYOUTS(matrix, slice, matrix)
    TEST_ALL_LAYOUTS(matrix, slice, range)
    TEST_ALL_LAYOUTS(matrix, slice, slice)

//    C = range
    TEST_ALL_LAYOUTS(range, matrix, matrix)
    TEST_ALL_LAYOUTS(range, matrix, range)
    TEST_ALL_LAYOUTS(range, matrix, slice)

    TEST_ALL_LAYOUTS(range, range, matrix)
    TEST_ALL_LAYOUTS(range, range, range)
    TEST_ALL_LAYOUTS(range, range, slice)

    TEST_ALL_LAYOUTS(range, slice, matrix)
    TEST_ALL_LAYOUTS(range, slice, range)
    TEST_ALL_LAYOUTS(range, slice, slice)

//    C = slice
    TEST_ALL_LAYOUTS(slice, matrix, matrix)
    TEST_ALL_LAYOUTS(slice, matrix, range)
    TEST_ALL_LAYOUTS(slice, matrix, slice)

    TEST_ALL_LAYOUTS(slice, range, matrix)
    TEST_ALL_LAYOUTS(slice, range, range)
    TEST_ALL_LAYOUTS(slice, range, slice)

    TEST_ALL_LAYOUTS(slice, slice, matrix)
    TEST_ALL_LAYOUTS(slice, slice, range)
    TEST_ALL_LAYOUTS(slice, slice, slice)

#undef TEST_ALL_LAYOUTS

    return EXIT_SUCCESS;
}

#endif
