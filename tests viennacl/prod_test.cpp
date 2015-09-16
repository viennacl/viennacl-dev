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

/** \file tests/src/libviennacl_blas3.cpp  Testing the BLAS level 3 routines in the ViennaCL BLAS-like shared library
 *   \test Testing the BLAS level 3 routines in the ViennaCL BLAS-like shared library
n **/

#include "../viennacl/matrix.hpp"
#include "../viennacl/matrix_proxy.hpp"

#include "../viennacl/linalg/prod.hpp"

// include necessary system headers
#include <iostream>
#include <vector>

#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/matrix_proxy.hpp>
#include <boost/numeric/ublas/matrix_expression.hpp>
#include <boost/numeric/ublas/io.hpp>

//#define TEST_FLOAT  //if defined, floats are tested
#define TEST_DOUBLE //if defined, doubles are tested

/*
  template<typename T, typename U, typename EpsilonT>
  void check(T const & t, U const & u, EpsilonT eps)
  {
  EpsilonT rel_error = std::fabs(static_cast<EpsilonT>(diff(t,u)));
  if (rel_error > eps)
  {
  std::cerr << "Relative error: " << rel_error << std::endl;
  std::cerr << "Aborting!" << std::endl;
  exit(EXIT_FAILURE);
  }
  std::cout << "SUCCESS ";
  }


  template<typename T>
  T get_value(std::vector<T> & array, ViennaCLInt i, ViennaCLInt j,
  n            ViennaCLInt start1, ViennaCLInt start2,
  ViennaCLInt stride1, ViennaCLInt stride2,
  ViennaCLInt rows, ViennaCLInt cols,
  ViennaCLOrder order, ViennaCLTranspose trans)
  {
  // row-major
  if (order == ViennaCLRowMajor && trans == ViennaCLTrans)
  return array[static_cast<std::size_t>((j*stride1 + start1) * cols + (i*stride2 + start2))];
  else if (order == ViennaCLRowMajor && trans != ViennaCLTrans)
  return array[static_cast<std::size_t>((i*stride1 + start1) * cols + (j*stride2 + start2))];

  // column-major
  else if (order != ViennaCLRowMajor && trans == ViennaCLTrans)
  return array[static_cast<std::size_t>((j*stride1 + start1) + (i*stride2 + start2) * rows)];
  return array[static_cast<std::size_t>((i*stride1 + start1) + (j*stride2 + start2) * rows)];
  }*/

/*
template<typename ScalarType, typename ViennaCLVectorType>
ScalarType diff(std::vector<ScalarType> const & v1, ViennaCLVectorType const & vcl_vec)
{
   ScalarType inf_norm = 0;

   for (unsigned int i=0;i<v1.size(); ++i)
   {
      if ( std::max( std::fabs(v2_cpu[i]), std::fabs(v1[i]) ) > 0 )
         v2_cpu[i] = std::fabs(v2_cpu[i] - v1[i]) / std::max( std::fabs(v2_cpu[i]), std::fabs(v1[i]) );
      else
         v2_cpu[i] = 0.0;

      if (v2_cpu[i] > inf_norm)
        inf_norm = v2_cpu[i];
   }

   return inf_norm;
}

template<typename NumericT, typename orde, typename EpsilonT>
void check(T const & t, U const & u, EpsilonT eps)
{
  EpsilonT rel_error = std::fabs(static_cast<EpsilonT>(diff(t,u)));
  if (rel_error > eps)
  {
    std::cerr << "Relative error: " << rel_error << std::endl;
    std::cerr << "Aborting!" << std::endl;
    exit(EXIT_FAILURE);
  }
  std::cout << "SUCCESS ";
}
*/

#define UBLAS boost::numeric::ublas

template<typename NumericT, typename order_viennacl, typename order_ublas, typename EpsilonT>
int check(viennacl::matrix<NumericT,order_viennacl> result, UBLAS::matrix<NumericT,order_ublas> reference_ublas, EpsilonT eps)
{
  //assuming ublas-matrix is copied into a viennacl-matrix with same internal size
  viennacl::matrix<NumericT,order_viennacl> reference;
  viennacl::copy(reference_ublas, reference);

  
  NumericT * p_ref = viennacl::linalg::host_based::detail::extract_raw_pointer<NumericT>(reference);
  NumericT * p_res = viennacl::linalg::host_based::detail::extract_raw_pointer<NumericT>(result);

  EpsilonT inf_norm = 0;

  for (std::size_t i=0; i<reference.internal_size(); ++i)
  {
    if ( std::max( std::fabs(p_ref[i]), std::fabs(p_res[i]) ) > 0 )
      p_ref[i] = std::fabs(p_ref[i] - p_res[i]) / std::max( std::fabs(p_ref[i]), std::fabs(p_res[i]) );
    else
      p_ref[i] = 0.0;

    if (p_ref[i] > inf_norm)
      inf_norm = p_ref[i];
  }

  EpsilonT rel_error = std::fabs(inf_norm);
  if (rel_error > eps)
  {
    std::cerr << "    -> CURRENT TEST FAILED! Relative error: " << rel_error <<  " continuing..." <<std::endl; 
    return 1;
  }
  else
  {
    std::cout << "-----> SUCCESS <----- " << std::endl;
    return 0;
  }

}


template<class NumericT, class orderC, class orderA, class orderB, class orderC_ublas, class orderA_ublas, class orderB_ublas>
int test_prod(viennacl::matrix<NumericT,orderC>  & C, viennacl::matrix<NumericT,orderA> & A, viennacl::matrix<NumericT,orderB> & B, 
               UBLAS::matrix<NumericT,orderC_ublas> & C_ublas, UBLAS::matrix<NumericT,orderA_ublas> & A_ublas, UBLAS::matrix<NumericT,orderB_ublas> & B_ublas,
               NumericT eps)
{
  using viennacl::linalg::prod;
  using viennacl::trans;
  using boost::numeric::ublas::matrix;
  using boost::numeric::ublas::column_major;
  using boost::numeric::ublas::row_major;
  
  viennacl::matrix<NumericT,orderA> At = trans(A);
  viennacl::matrix<NumericT,orderB> Bt = trans(B);

  matrix<NumericT,orderA_ublas> At_ublas = UBLAS::trans(A_ublas);
  matrix<NumericT,orderB_ublas> Bt_ublas = UBLAS::trans(B_ublas);
  
  int error_count = 0;

  //std::cout << "A:       " << A << "At " << At << "sizes " << At.size1() << " " << At.size2() << " is At row-major? " << At.row_major() << std::endl;
  //std::cout << "B:       " << B << std::endl;
  //std::cout << "A_ublas: " << A_ublas << std::endl;
  //std::cout << "B_ublas: " << B_ublas << std::endl;

  //std::cout << "A[1] is " << viennacl::linalg::host_based::detail::extract_raw_pointer<NumericT>(A)[1];//DEBUG
  //std::cout << "At[1] is " << viennacl::linalg::host_based::detail::extract_raw_pointer<NumericT>(At)[1];//DEBUG

  std::cout << "    -> trans-trans: ";
  C = prod(trans(At),trans(Bt));
  C_ublas = UBLAS::prod(UBLAS::trans(At_ublas),UBLAS::trans(Bt_ublas)); 
  error_count += check(C, C_ublas, eps);
  //std::cout << "                     C       is:" << C << std::endl << "                     C_ublas is:" << C_ublas <<std::endl;//DEBUG
  
  std::cout << "    -> trans-no:    ";
  C = prod(trans(At),B);
  C_ublas = UBLAS::prod(UBLAS::trans(At_ublas),B_ublas); 
  error_count += check(C, C_ublas, eps);
  //  std::cout << "                     C       is:" << C << std::endl << "                     C_ublas is:" << C_ublas <<std::endl;//DEBUG

  std::cout << "    -> no-trans:    ";
  C = prod(A,trans(Bt));
  C_ublas = UBLAS::prod(A_ublas,UBLAS::trans(Bt_ublas)); 
  error_count += check(C, C_ublas, eps);
  //std::cout << "                     C       is:" << C << std::endl << "                     C_ublas is:" << C_ublas <<std::endl;//DEBUG

  std::cout << "    -> no-no:       ";
  C = prod(A,B);
  C_ublas = UBLAS::prod(A_ublas,B_ublas);
  error_count += check(C, C_ublas, eps);
  //std::cout << "                     C       is:" << C << std::endl << "                     C_ublas is:" << C_ublas <<std::endl;//DEBUG
  
  return error_count;
}

template<class T, class F>
void init_random(viennacl::matrix<T, F> & M)
{
  std::vector<T> cM(M.internal_size());
  for (std::size_t i = 0; i < M.size1(); ++i)
    for (std::size_t j = 0; j < M.size2(); ++j)
      cM[F::mem_index(i, j, M.internal_size1(), M.internal_size2())] = T(rand())/T(RAND_MAX);
  viennacl::fast_copy(&cM[0],&cM[0] + cM.size(),M);
}

int main(int argc, char **argv)
{

  using boost::numeric::ublas::matrix;
  using boost::numeric::ublas::column_major;
  using boost::numeric::ublas::row_major;

  #ifdef TEST_FLOAT
  float  eps_float  = 1e-5f;
  #endif
  #ifdef TEST_DOUBLE
  double eps_double = 1e-12;
  #endif

  std::size_t n;
  std::size_t k;
  std::size_t m;

  int error_count = 0;

  if(argc == 4)
  {
    m = std::stoi(argv[1]);
    k = std::stoi(argv[2]);
    n = std::stoi(argv[3]);
  }
  else 
  {
    m = 2;
    k = 3;
    n = 2;
  }

  /* VIENNACL */
  /* float matrices */
  viennacl::matrix<float,viennacl::column_major> Af_col(m,k);
  viennacl::matrix<float,viennacl::column_major> Bf_col(k,n);
  viennacl::matrix<float,viennacl::column_major> Cf_col(m,n);
  init_random(Af_col);
  init_random(Bf_col);
  init_random(Cf_col);
  viennacl::matrix<float,viennacl::row_major> Af_row(m,k);
  viennacl::matrix<float,viennacl::row_major> Bf_row(k,n);
  viennacl::matrix<float,viennacl::row_major> Cf_row(m,n);
  init_random(Af_row);
  init_random(Bf_row);
  init_random(Cf_row);

  /* double matrices */
  viennacl::matrix<double,viennacl::column_major> Ad_col(m,k);
  viennacl::matrix<double,viennacl::column_major> Bd_col(k,n);
  viennacl::matrix<double,viennacl::column_major> Cd_col(m,n);
  init_random(Ad_col);
  init_random(Bd_col);
  init_random(Cd_col);
  viennacl::matrix<double,viennacl::row_major> Ad_row(m,k);
  viennacl::matrix<double,viennacl::row_major> Bd_row(k,n);
  viennacl::matrix<double,viennacl::row_major> Cd_row(m,n);
  init_random(Ad_row);
  init_random(Bd_row);
  init_random(Cd_row);

  /* UBLAS */
  /* float matrices */
  matrix<float,column_major> Af_col_ublas(m,k);
  matrix<float,column_major> Bf_col_ublas(k,n);
  matrix<float,column_major> Cf_col_ublas(m,n);
  viennacl::copy(Af_col, Af_col_ublas);
  viennacl::copy(Bf_col, Bf_col_ublas);
  viennacl::copy(Cf_col, Cf_col_ublas);
  matrix<float,row_major> Af_row_ublas(m,k);
  matrix<float,row_major> Bf_row_ublas(k,n);
  matrix<float,row_major> Cf_row_ublas(m,n);
  viennacl::copy(Af_row, Af_row_ublas);
  viennacl::copy(Bf_row, Bf_row_ublas);
  viennacl::copy(Cf_row, Cf_row_ublas);

  /* double matrices */
  matrix<double,column_major> Ad_col_ublas(m,k);
  matrix<double,column_major> Bd_col_ublas(k,n);
  matrix<double,column_major> Cd_col_ublas(m,n);
  viennacl::copy(Ad_col, Ad_col_ublas);
  viennacl::copy(Bd_col, Bd_col_ublas);
  viennacl::copy(Cd_col, Cd_col_ublas);
  matrix<double,row_major> Ad_row_ublas(m,k);
  matrix<double,row_major> Bd_row_ublas(k,n);
  matrix<double,row_major> Cd_row_ublas(m,n);
  viennacl::copy(Ad_row, Ad_row_ublas);
  viennacl::copy(Bd_row, Bd_row_ublas);
  viennacl::copy(Cd_row, Cd_row_ublas);

  /* ****************************************** */
  std::cout << "*** STARTING TEST ***" << std::endl;
  std::cout << "m = " << m << std::endl;
  std::cout << "k = " << k << std::endl;
  std::cout << "n = " << n << std::endl;
  //std::cout << "Af_col is: " << Af_col << std::endl;//DEBUG
  //std::cout << "Af_row is: " << Af_row << std::endl;//DEBUG
  //std::cout << "Bf_col is: " << Bf_col << std::endl;//DEBUG
  //std::cout << "Bf_row is: " << Bf_row << std::endl;//DEBUG
  //std::cout << "Ad_row is: " << Ad_row << std::endl;//DEBUG
  //std::cout << "Bd_row is: " << Bd_row << std::endl;//DEBUG
  
  std::cout << "*********************" << std::endl << std::endl;

#ifdef TEST_FLOAT
  std::cout << "*******FLOAT*********" << std::endl;
  std::cout << "  -> C: row, A: row, B: row" << std::endl;
  error_count += test_prod(Cf_row, Af_row, Bf_row, Cf_row_ublas, Af_row_ublas, Bf_row_ublas, eps_float);

  std::cout << "  -> C: row, A: row, B: col" << std::endl;
  error_count += test_prod(Cf_row, Af_row, Bf_col, Cf_row_ublas, Af_row_ublas, Bf_col_ublas, eps_float);

  std::cout << "  -> C: row, A: col, B: row" << std::endl;
  error_count += test_prod(Cf_row, Af_col, Bf_row, Cf_row_ublas, Af_col_ublas, Bf_row_ublas, eps_float);

  std::cout << "  -> C: row, A: col, B: col" << std::endl;
  error_count += test_prod(Cf_row, Af_col, Bf_col, Cf_row_ublas, Af_col_ublas, Bf_col_ublas, eps_float);

  std::cout << "  -> C: col, A: row, B: row" << std::endl;
  error_count += test_prod(Cf_col, Af_row, Bf_row, Cf_col_ublas, Af_row_ublas, Bf_row_ublas, eps_float);

  std::cout << "  -> C: col, A: row, B: col" << std::endl;
  error_count += test_prod(Cf_col, Af_row, Bf_col, Cf_col_ublas, Af_row_ublas, Bf_col_ublas, eps_float);

  std::cout << "  -> C: col, A: col, B: row" << std::endl;
  error_count += test_prod(Cf_col, Af_col, Bf_row, Cf_col_ublas, Af_col_ublas, Bf_row_ublas, eps_float);

  std::cout << "  -> C: col, A: col, B: col" << std::endl;
  error_count += test_prod(Cf_col, Af_col, Bf_col, Cf_col_ublas, Af_col_ublas, Bf_col_ublas, eps_float);

#endif
#ifdef TEST_DOUBLE
  std::cout << "******DOUBLE*********" << std::endl;
  std::cout << "  -> C: row, A: row, B: row" << std::endl;
  error_count += test_prod(Cd_row, Ad_row, Bd_row, Cd_row_ublas, Ad_row_ublas, Bd_row_ublas, eps_double);

  std::cout << "  -> C: row, A: row, B: col" << std::endl;
  error_count += test_prod(Cd_row, Ad_row, Bd_col, Cd_row_ublas, Ad_row_ublas, Bd_col_ublas, eps_double);

  std::cout << "  -> C: row, A: col, B: row" << std::endl;
  error_count += test_prod(Cd_row, Ad_col, Bd_row, Cd_row_ublas, Ad_col_ublas, Bd_row_ublas, eps_double);

  std::cout << "  -> C: row, A: col, B: col" << std::endl;
  error_count += test_prod(Cd_row, Ad_col, Bd_col, Cd_row_ublas, Ad_col_ublas, Bd_col_ublas, eps_double);

  std::cout << "  -> C: col, A: row, B: row" << std::endl;
  error_count += test_prod(Cd_col, Ad_row, Bd_row, Cd_col_ublas, Ad_row_ublas, Bd_row_ublas, eps_double);

  std::cout << "  -> C: col, A: row, B: col" << std::endl;
  error_count += test_prod(Cd_col, Ad_row, Bd_col, Cd_col_ublas, Ad_row_ublas, Bd_col_ublas, eps_double);

  std::cout << "  -> C: col, A: col, B: row" << std::endl;
  error_count += test_prod(Cd_col, Ad_col, Bd_row, Cd_col_ublas, Ad_col_ublas, Bd_row_ublas, eps_double);

  std::cout << "  -> C: col, A: col, B: col" << std::endl;
  error_count += test_prod(Cd_col, Ad_col, Bd_col, Cd_col_ublas, Ad_col_ublas, Bd_col_ublas, eps_double);

#endif

  std::cout << "*********************" << std::endl << std::endl;
  
  if (error_count > 0)
  {
    std::cout << std::endl << "!!!! TEST FAILED MISERABLY: " << error_count << " ERRORS !!!!" << std::endl;
    return EXIT_FAILURE;
  }
  std::cout << std::endl << "!!!! TEST COMPLETED SUCCESSFULLY !!!!" << std::endl;
  return EXIT_SUCCESS;
}
