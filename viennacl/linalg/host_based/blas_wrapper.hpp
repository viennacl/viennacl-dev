#ifndef VIENNACL_LINALG_HOST_BASED_BLAS_HELPER_HPP_
#define VIENNACL_LINALG_HOST_BASED_BLAS_HELPER_HPP_

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

/** @file viennacl/linalg/host_based/blas_helper.hpp
    @brief Implementations of generic wrapper for cblas
*/

#include "cblas.h"
#include "default_blas.hpp"

#include "viennacl/forwards.h"

namespace viennacl
{
  namespace linalg
  {
    namespace host_based
    {

      template<class T>
      struct cblas_wrapper{
          static void copy(const vcl_size_t N, T const * x, const vcl_size_t incx, T * y, const vcl_size_t incy)
          { default_blas::copy(N,x,incx,y,incy); }
          static void swap(const vcl_size_t N, T * x, const vcl_size_t incx, T * y, const vcl_size_t incy)
          { default_blas::swap(N,x,incx,y,incy); }
          static T asum(const vcl_size_t N, T const * x, const vcl_size_t incx)
          { return default_blas::asum(N,x,incx); }
          static T nrm2(const vcl_size_t N, T const * x, const vcl_size_t incx)
          { return default_blas::nrm2(N,x,incx); }
          static T dot(const vcl_size_t N, T const * x, const vcl_size_t incx, T const * y, const vcl_size_t incy)
          { return default_blas::dot(N,x,incx,y,incy); }
          static void gemv(bool is_row_major, bool is_transposed, const vcl_size_t M, const vcl_size_t N
                           , T alpha, T const * A, const vcl_size_t lda
                           , T const * x, const vcl_size_t incx, T beta, T * y, const vcl_size_t incy)
          { default_blas::gemv(is_row_major, is_transposed, M, N, alpha, A, lda, 1, x, incx, beta, y, incy); }
          static void gemm(bool is_all_row_major, bool is_A_trans,
                           bool is_B_trans, const vcl_size_t M, const vcl_size_t N,
                           const vcl_size_t K, const float alpha, T const *A,
                           const vcl_size_t lda, T const *B, const vcl_size_t ldb,
                           const float beta, float *C, const vcl_size_t ldc)
          { default_blas::gemm(is_all_row_major,is_all_row_major,is_all_row_major,is_A_trans, is_B_trans, M, N, K
                               ,alpha,A,lda,1,B,ldb,1,beta,C,ldc,1); }

      };

      template<>
      struct cblas_wrapper<float>{
          static void copy(const vcl_size_t N, float const * x, const vcl_size_t incx, float * y, const vcl_size_t incy)
          { cblas_scopy(N,x,incx,y,incy); }
          static void swap(const vcl_size_t N, float * x, const vcl_size_t incx, float * y, const vcl_size_t incy)
          { cblas_sswap(N,x,incx,y,incy); }
          static float asum(const vcl_size_t N, float const * x, const vcl_size_t incx)
          { return cblas_sasum(N,x,incx);}
          static float nrm2(const vcl_size_t N, float const * x, const vcl_size_t incx)
          { return cblas_snrm2(N,x,incx); }
          static float dot(const vcl_size_t N, float const * x, const vcl_size_t incx, float const * y, const vcl_size_t incy)
          { return cblas_sdot(N,x,incx,y,incy); }
          static void gemv(bool is_row_major, bool is_transposed, const vcl_size_t M, const vcl_size_t N, float alpha, float const * A, const vcl_size_t lda, float const * x, const vcl_size_t incx, float beta, float * y, const vcl_size_t incy)
          { cblas_sgemv(is_row_major?CblasRowMajor:CblasColMajor
                                     ,is_transposed?CblasTrans:CblasNoTrans
                                     ,M,N,alpha,A,lda,x,incx,beta,y,incy);  }
          static void gemm(bool is_all_row_major, bool is_A_trans,
                           bool is_B_trans, const vcl_size_t M, const vcl_size_t N,
                           const vcl_size_t K, const float alpha, float const *A,
                           const vcl_size_t lda, float const *B, const vcl_size_t ldb,
                           const float beta, float *C, const vcl_size_t ldc)
          { cblas_sgemm(is_all_row_major?CblasRowMajor:CblasColMajor,is_A_trans?CblasTrans:CblasNoTrans,is_B_trans?CblasTrans:CblasNoTrans
                       , M, N, K,alpha,A,lda,B,ldb,beta,C,ldc); }

      };

      template<>
      struct cblas_wrapper<double>{
          static void copy(const vcl_size_t N, double const * x, const vcl_size_t incx, double * y, const vcl_size_t incy)
          { cblas_dcopy(N,x,incx,y,incy); }
          static void swap(const vcl_size_t N, double * x, const vcl_size_t incx, double * y, const vcl_size_t incy)
          { cblas_dswap(N,x,incx,y,incy); }
          static void axpy(const vcl_size_t N, double alpha, double const * x, const vcl_size_t incx, double * y, const vcl_size_t incy)
          { cblas_daxpy(N,alpha,x,incx,y,incy); }
          static void scale(const vcl_size_t N, double alpha, double * x, const vcl_size_t incx)
          { cblas_dscal(N,alpha,x,incx); }
          static double asum(const vcl_size_t N, double const * x, const vcl_size_t incx)
          { return cblas_dasum(N,x,incx);}
          static double nrm2(const vcl_size_t N, double const * x, const vcl_size_t incx)
          { return cblas_dnrm2(N,x,incx); }
          static double dot(const vcl_size_t N, double const * x, const vcl_size_t incx, double const * y, const vcl_size_t incy)
          { return cblas_ddot(N,x,incx,y,incy); }
          static void symv(const vcl_size_t N, double alpha, double const * A, const vcl_size_t lda, double const * x, const vcl_size_t incx, double beta, double * y, const vcl_size_t incy)
          { cblas_dsymv(CblasRowMajor,CblasUpper,N,alpha,A,lda,x,1,beta,y,1);  }
          static void gemv(bool is_row_major, bool is_transposed, const vcl_size_t M, const vcl_size_t N, double alpha, double const * A, const vcl_size_t lda, double const * x, const vcl_size_t incx, double beta, double * y, const vcl_size_t incy)
          { cblas_dgemv(is_row_major?CblasRowMajor:CblasColMajor
                                     ,is_transposed?CblasTrans:CblasNoTrans
                                     ,M,N,alpha,A,lda,x,incx,beta,y,incy);  }
          static void gemm(bool is_all_row_major, bool is_A_trans,
                           bool is_B_trans, const vcl_size_t M, const vcl_size_t N,
                           const vcl_size_t K, const double alpha, double const *A,
                           const vcl_size_t lda, double const *B, const vcl_size_t ldb,
                           const double beta, double *C, const vcl_size_t ldc)
          { cblas_dgemm(is_all_row_major?CblasRowMajor:CblasColMajor,is_A_trans?CblasTrans:CblasNoTrans,is_B_trans?CblasTrans:CblasNoTrans
                       , M, N, K,alpha,A,lda,B,ldb,beta,C,ldc); }


      };

    }
  }
}

#endif
