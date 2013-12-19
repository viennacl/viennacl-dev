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

// include necessary system headers
#include <iostream>

#include "viennacl.hpp"
#include "viennacl_private.hpp"

#include "blas3.hpp"

//include basic scalar and vector types of ViennaCL
#include "viennacl/scalar.hpp"
#include "viennacl/vector.hpp"
#include "viennacl/matrix.hpp"
#include "viennacl/linalg/direct_solve.hpp"
#include "viennacl/linalg/prod.hpp"


//
// xGEMV
//

namespace detail
{
  template <typename NumericT>
  ViennaCLStatus ViennaCLHostgemm_impl(ViennaCLBackend /*backend*/,
                                       ViennaCLOrder orderA, ViennaCLTranspose transA,
                                       ViennaCLOrder orderB, ViennaCLTranspose transB,
                                       ViennaCLOrder orderC,
                                       ViennaCLInt m, ViennaCLInt n, ViennaCLInt k,
                                       NumericT alpha,
                                       NumericT *A, ViennaCLInt offA_row, ViennaCLInt offA_col, ViennaCLInt incA_row, ViennaCLInt incA_col, ViennaCLInt lda,
                                       NumericT *B, ViennaCLInt offB_row, ViennaCLInt offB_col, ViennaCLInt incB_row, ViennaCLInt incB_col, ViennaCLInt ldb,
                                       NumericT beta,
                                       NumericT *C, ViennaCLInt offC_row, ViennaCLInt offC_col, ViennaCLInt incC_row, ViennaCLInt incC_col, ViennaCLInt ldc)
  {
    ViennaCLInt A_size1 = (transA == ViennaCLTrans) ? k : m;
    ViennaCLInt A_size2 = (transA == ViennaCLTrans) ? m : k;

    ViennaCLInt B_size1 = (transB == ViennaCLTrans) ? n : k;
    ViennaCLInt B_size2 = (transB == ViennaCLTrans) ? k : n;

    /////// A row-major
    if (orderA == ViennaCLRowMajor && orderB == ViennaCLRowMajor && orderC == ViennaCLRowMajor)
    {
      viennacl::matrix_base<NumericT> matA(A, viennacl::MAIN_MEMORY,
                                           A_size1, offA_row, incA_row, m,
                                           A_size2, offA_col, incA_col, lda);

      viennacl::matrix_base<NumericT> matB(B, viennacl::MAIN_MEMORY,
                                           B_size1, offB_row, incB_row, k,
                                           B_size2, offB_col, incB_col, ldb);

      viennacl::matrix_base<NumericT> matC(C, viennacl::MAIN_MEMORY,
                                           m, offC_row, incC_row, m,
                                           n, offC_col, incC_col, ldc);

      detail::gemm_dispatch(alpha, matA, transA, matB, transB, beta, matC);
    }
    else if (orderA == ViennaCLRowMajor && orderB == ViennaCLRowMajor && orderC == ViennaCLColumnMajor)
    {
      viennacl::matrix_base<NumericT> matA(A, viennacl::MAIN_MEMORY,
                                           A_size1, offA_row, incA_row, m,
                                           A_size2, offA_col, incA_col, lda);

      viennacl::matrix_base<NumericT> matB(B, viennacl::MAIN_MEMORY,
                                           B_size1, offB_row, incB_row, k,
                                           B_size2, offB_col, incB_col, ldb);

      viennacl::matrix_base<NumericT, viennacl::column_major> matC(C, viennacl::MAIN_MEMORY,
                                                                   m, offC_row, incC_row, ldc,
                                                                   n, offC_col, incC_col, n);

      detail::gemm_dispatch(alpha, matA, transA, matB, transB, beta, matC);
    }
    else if (orderA == ViennaCLRowMajor && orderB == ViennaCLColumnMajor && orderC == ViennaCLRowMajor)
    {
      viennacl::matrix_base<NumericT> matA(A, viennacl::MAIN_MEMORY,
                                           A_size1, offA_row, incA_row, m,
                                           A_size2, offA_col, incA_col, lda);

      viennacl::matrix_base<NumericT, viennacl::column_major> matB(B, viennacl::MAIN_MEMORY,
                                                                   B_size1, offB_row, incB_row, ldb,
                                                                   B_size2, offB_col, incB_col, n);

      viennacl::matrix_base<NumericT> matC(C, viennacl::MAIN_MEMORY,
                                           m, offC_row, incC_row, m,
                                           n, offC_col, incC_col, ldc);

      detail::gemm_dispatch(alpha, matA, transA, matB, transB, beta, matC);
    }
    else if (orderA == ViennaCLRowMajor && orderB == ViennaCLColumnMajor && orderC == ViennaCLColumnMajor)
    {
      viennacl::matrix_base<NumericT> matA(A, viennacl::MAIN_MEMORY,
                                           A_size1, offA_row, incA_row, m,
                                           A_size2, offA_col, incA_col, lda);

      viennacl::matrix_base<NumericT, viennacl::column_major> matB(B, viennacl::MAIN_MEMORY,
                                                                   B_size1, offB_row, incB_row, ldb,
                                                                   B_size2, offB_col, incB_col, n);

      viennacl::matrix_base<NumericT, viennacl::column_major> matC(C, viennacl::MAIN_MEMORY,
                                                                   m, offC_row, incC_row, ldc,
                                                                   n, offC_col, incC_col, n);

      detail::gemm_dispatch(alpha, matA, transA, matB, transB, beta, matC);
    }

    /////// A column-major

    else if (orderA == ViennaCLColumnMajor && orderB == ViennaCLRowMajor && orderC == ViennaCLRowMajor)
    {
      viennacl::matrix_base<NumericT, viennacl::column_major> matA(A, viennacl::MAIN_MEMORY,
                                                                   A_size1, offA_row, incA_row, lda,
                                                                   A_size2, offA_col, incA_col, k);

      viennacl::matrix_base<NumericT> matB(B, viennacl::MAIN_MEMORY,
                                           B_size1, offB_row, incB_row, k,
                                           B_size2, offB_col, incB_col, ldb);

      viennacl::matrix_base<NumericT> matC(C, viennacl::MAIN_MEMORY,
                                           m, offC_row, incC_row, m,
                                           n, offC_col, incC_col, ldc);

      detail::gemm_dispatch(alpha, matA, transA, matB, transB, beta, matC);
    }
    else if (orderA == ViennaCLColumnMajor && orderB == ViennaCLRowMajor && orderC == ViennaCLColumnMajor)
    {
      viennacl::matrix_base<NumericT, viennacl::column_major> matA(A, viennacl::MAIN_MEMORY,
                                                                   A_size1, offA_row, incA_row, lda,
                                                                   A_size2, offA_col, incA_col, k);

      viennacl::matrix_base<NumericT> matB(B, viennacl::MAIN_MEMORY,
                                           B_size1, offB_row, incB_row, k,
                                           B_size2, offB_col, incB_col, ldb);

      viennacl::matrix_base<NumericT, viennacl::column_major> matC(C, viennacl::MAIN_MEMORY,
                                                                   m, offC_row, incC_row, ldc,
                                                                   n, offC_col, incC_col, n);

      detail::gemm_dispatch(alpha, matA, transA, matB, transB, beta, matC);
    }
    else if (orderA == ViennaCLColumnMajor && orderB == ViennaCLColumnMajor && orderC == ViennaCLRowMajor)
    {
      viennacl::matrix_base<NumericT, viennacl::column_major> matA(A, viennacl::MAIN_MEMORY,
                                                                   A_size1, offA_row, incA_row, lda,
                                                                   A_size2, offA_col, incA_col, k);

      viennacl::matrix_base<NumericT, viennacl::column_major> matB(B, viennacl::MAIN_MEMORY,
                                                                   B_size1, offB_row, incB_row, ldb,
                                                                   B_size2, offB_col, incB_col, n);

      viennacl::matrix_base<NumericT> matC(C, viennacl::MAIN_MEMORY,
                                        m, offC_row, incC_row, m,
                                        n, offC_col, incC_col, ldc);

      detail::gemm_dispatch(alpha, matA, transA, matB, transB, beta, matC);
    }
    else if (orderA == ViennaCLColumnMajor && orderB == ViennaCLColumnMajor && orderC == ViennaCLColumnMajor)
    {
      viennacl::matrix_base<NumericT, viennacl::column_major> matA(A, viennacl::MAIN_MEMORY,
                                                                   A_size1, offA_row, incA_row, lda,
                                                                   A_size2, offA_col, incA_col, k);

      viennacl::matrix_base<NumericT, viennacl::column_major> matB(B, viennacl::MAIN_MEMORY,
                                                                   B_size1, offB_row, incB_row, ldb,
                                                                   B_size2, offB_col, incB_col, n);

      viennacl::matrix_base<NumericT, viennacl::column_major> matC(C, viennacl::MAIN_MEMORY,
                                                                   m, offC_row, incC_row, ldc,
                                                                   n, offC_col, incC_col, n);

      detail::gemm_dispatch(alpha, matA, transA, matB, transB, beta, matC);
    }

    return ViennaCLSuccess;
  }

}


VIENNACL_EXPORTED_FUNCTION ViennaCLStatus ViennaCLHostSgemm(ViennaCLBackend backend,
                                                            ViennaCLOrder orderA, ViennaCLTranspose transA,
                                                            ViennaCLOrder orderB, ViennaCLTranspose transB,
                                                            ViennaCLOrder orderC,
                                                            ViennaCLInt m, ViennaCLInt n, ViennaCLInt k,
                                                            float alpha,
                                                            float *A, ViennaCLInt offA_row, ViennaCLInt offA_col, ViennaCLInt incA_row, ViennaCLInt incA_col, ViennaCLInt lda,
                                                            float *B, ViennaCLInt offB_row, ViennaCLInt offB_col, ViennaCLInt incB_row, ViennaCLInt incB_col, ViennaCLInt ldb,
                                                            float beta,
                                                            float *C, ViennaCLInt offC_row, ViennaCLInt offC_col, ViennaCLInt incC_row, ViennaCLInt incC_col, ViennaCLInt ldc)
{
  return detail::ViennaCLHostgemm_impl<float>(backend,
                                              orderA, transA,
                                              orderB, transB,
                                              orderC,
                                              m, n, k,
                                              alpha,
                                              A, offA_row, offA_col, incA_row, incA_col, lda,
                                              B, offB_row, offB_col, incB_row, incB_col, ldb,
                                              beta,
                                              C, offC_row, offC_col, incC_row, incC_col, ldc);
}

VIENNACL_EXPORTED_FUNCTION ViennaCLStatus ViennaCLHostDgemm(ViennaCLBackend backend,
                                                            ViennaCLOrder orderA, ViennaCLTranspose transA,
                                                            ViennaCLOrder orderB, ViennaCLTranspose transB,
                                                            ViennaCLOrder orderC,
                                                            ViennaCLInt m, ViennaCLInt n, ViennaCLInt k,
                                                            double alpha,
                                                            double *A, ViennaCLInt offA_row, ViennaCLInt offA_col, ViennaCLInt incA_row, ViennaCLInt incA_col, ViennaCLInt lda,
                                                            double *B, ViennaCLInt offB_row, ViennaCLInt offB_col, ViennaCLInt incB_row, ViennaCLInt incB_col, ViennaCLInt ldb,
                                                            double beta,
                                                            double *C, ViennaCLInt offC_row, ViennaCLInt offC_col, ViennaCLInt incC_row, ViennaCLInt incC_col, ViennaCLInt ldc)
{
  return detail::ViennaCLHostgemm_impl<double>(backend,
                                               orderA, transA,
                                               orderB, transB,
                                               orderC,
                                               m, n, k,
                                               alpha,
                                               A, offA_row, offA_col, incA_row, incA_col, lda,
                                               B, offB_row, offB_col, incB_row, incB_col, ldb,
                                               beta,
                                               C, offC_row, offC_col, incC_row, incC_col, ldc);
}


