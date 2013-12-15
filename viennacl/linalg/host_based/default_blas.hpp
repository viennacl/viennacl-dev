#ifndef VIENNACL_LINALG_HOST_BASED_DEFAULT_BLAS_HPP_
#define VIENNACL_LINALG_HOST_BASED_DEFAULT_BLAS_HPP_

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

/** @file viennacl/linalg/host_based/default_blas.hpp
    @brief Default implementation of some BLAS routines
*/

#include "viennacl/forwards.h"

namespace viennacl
{
  namespace linalg
  {
    namespace host_based
    {
      namespace default_blas{

        template<class T>
        inline void copy(vcl_size_t N, T const * x, vcl_size_t incx, T * y, vcl_size_t incy)
        {
#ifdef VIENNACL_WITH_OPENMP
#pragma omp parallel for if (N > VIENNACL_OPENMP_VECyR_MIN_SIZE)
#endif
          for (long i = 0; i < static_cast<long>(N); ++i)
          {
            y[i*incy] = x[i*incx];
          }
        }

        template<class T>
        inline void swap(vcl_size_t N, T * x, vcl_size_t incx, T * y, vcl_size_t incy)
        {
#ifdef VIENNACL_WITH_OPENMP
#pragma omp parallel for if (N > VIENNACL_OPENMP_VECTOR_MIN_SIZE)
#endif
          for (long i = 0; i < static_cast<long>(N); ++i)
          {
            T temp = y[i*incy];
            y[i*incy] = x[i*incx];
            x[i*incx] = temp;
          }
        }

        template<class T>
        inline T asum(vcl_size_t N, T const * x, vcl_size_t incx)
        {
          T temp = 0;
#ifdef VIENNACL_WITH_OPENMP
#pragma omp parallel for reduction(+: temp) if (N > VIENNACL_OPENMP_VECTOR_MIN_SIZE)
#endif
          for (long i = 0; i < static_cast<long>(N); ++i)
            temp += static_cast<T>(std::fabs(x[i*incx]));

          return temp;
        }

        template<class T>
        inline T nrm2(vcl_size_t N, T const * x, vcl_size_t incx)
        {
          T temp = 0;
          T data = 0;

#ifdef VIENNACL_WITH_OPENMP
#pragma omp parallel for reduction(+: temp) private(data) if (N > VIENNACL_OPENMP_VECTOR_MIN_SIZE)
#endif
          for (long i = 0; i < static_cast<long>(N); ++i)
          {
            data = x[i*incx];
            temp += data * data;
          }

          return std::sqrt(temp);
        }

        template<class T>
        inline T dot(vcl_size_t N, T const * x, vcl_size_t incx, T const * y, vcl_size_t incy)
        {
          T temp = 0;
#ifdef VIENNACL_WITH_OPENMP
#pragma omp parallel for reduction(+: temp) if (N > VIENNACL_OPENMP_VECTOR_MIN_SIZE)
#endif
          for (long i = 0; i < static_cast<long>(N); ++i)
            temp += x[i*incx] * y[i*incy];
          return temp;
        }

        namespace detail{

          /** @brief Implementation of GEMV. The access flow is strided when is_transposed&&row_major or !is_transposed&&!is_row_major.
           * Avoids code duplication
           */
          template<class T>
          inline void gemv_strided_flow(vcl_size_t M, vcl_size_t N, T alpha, T const * A, vcl_size_t lda, vcl_size_t nlda, T const * x, vcl_size_t incx, T beta, T * y, vcl_size_t incy){
            for (vcl_size_t row = 0; row < M; ++row)
              y[row * incy] *= beta;

            for (vcl_size_t col = 0; col < N; ++col)
            {
              T temp = alpha*x[col * incx];
              for (vcl_size_t row = 0; row < M; ++row)
                y[row * incy] += A[row*nlda + col*lda]*temp;
            }
          }

          template<class T>
          inline void gemv_sequential_flow(vcl_size_t M, vcl_size_t N, T alpha, T const * A, vcl_size_t lda, vcl_size_t nlda, T const * x, vcl_size_t incx, T beta, T * y, vcl_size_t incy){
#ifdef VIENNACL_WITH_OPENMP
#pragma omp parallel for
#endif
            for (long row = 0; row < static_cast<long>(M); ++row)
            {
              T temp = 0;
              for (vcl_size_t col = 0; col < N; ++col){
                temp += A[row*lda + col*nlda]*x[col * incx];
              }
              y[row * incy] = alpha*temp + beta*y[row * incy];
            }
          }

        }

        template<class T>
        inline void gemv(bool is_row_major, bool is_transposed, vcl_size_t M, vcl_size_t N
                         , T alpha, T const * A, vcl_size_t lda, vcl_size_t nlda
                         , T const * x, vcl_size_t incx
                         , T beta, T * y, vcl_size_t incy){
          if(is_transposed)
            if(is_row_major)
              detail::gemv_strided_flow(N, M, alpha, A, lda, nlda, x, incx, beta, y, incy);
            else
              detail::gemv_sequential_flow(N, M, alpha, A, lda, nlda, x, incx, beta, y, incy);
          else
            if(is_row_major)
              detail::gemv_sequential_flow(M, N, alpha, A, lda, nlda, x, incx, beta, y, incy);
            else
              detail::gemv_strided_flow(M, N, alpha, A, lda, nlda, x, incx, beta, y, incy);
        }


        namespace detail
        {

          template<class T, bool is_strided>
          struct array_wrap{
              array_wrap(T* data, vcl_size_t lda, vcl_size_t nlda) : data_(data), lda_(lda), nlda_(nlda){ }
              T & operator()(vcl_size_t i, vcl_size_t j) const {
                return data_[i*lda_ + j*nlda_];
              }
            private:
              T* data_;
              vcl_size_t lda_;
              vcl_size_t nlda_;
          };

          template<class T>
          struct array_wrap<T, true>{
              array_wrap(T* data, vcl_size_t lda, vcl_size_t nlda) : data_(data), lda_(lda), nlda_(nlda){ }
              T & operator()(vcl_size_t i, vcl_size_t j) const {
                return data_[i*nlda_ + j*lda_];
              }
            private:
              T* data_;
              vcl_size_t lda_;
              vcl_size_t nlda_;
          };

          template <typename A, typename B, typename C, typename NumericT>
          void prod(A const & a, B const & b, C const & c,
                    vcl_size_t M, vcl_size_t N, vcl_size_t K,
                    NumericT alpha, NumericT beta)
          {
            for (vcl_size_t i=0; i<M; ++i)
            {
              for (vcl_size_t j=0; j<N; ++j)
              {
                NumericT temp = 0;
                for (vcl_size_t k=0; k<K; ++k)
                  temp += a(i, k) * b(k, j);

                temp *= alpha;
                if (beta != 0)
                  temp += beta * c(i,j);
                c(i,j) = temp;
              }
            }
          }

        }

        template<class T>
        void gemm(bool is_C_row_major, bool is_A_row_major, bool is_B_row_major
                  ,bool is_A_transposed, bool is_B_transposed,
                  const vcl_size_t M, const vcl_size_t N
                         , const vcl_size_t K, const T alpha
                         , const T *A, const vcl_size_t lda, const vcl_size_t nlda
                         , const T *B, const vcl_size_t ldb, const vcl_size_t nldb
                         , const T beta, T *C, const vcl_size_t ldc, const vcl_size_t nldc)
        {
          typedef const T constT;
          bool is_A_strided = is_A_row_major && is_A_transposed || !is_A_row_major && !is_A_transposed;
          bool is_B_strided = is_B_row_major && is_B_transposed || !is_B_row_major && !is_B_transposed;
          bool is_C_strided = !is_C_row_major;
          if(is_C_strided){
            if(is_A_strided && is_B_strided)
               detail::prod(detail::array_wrap<constT,true>(A,lda,nlda), detail::array_wrap<constT,true>(B,ldb,nldb), detail::array_wrap<T,true>(C,ldc,nldc),M, N, K, alpha, beta);
            else if(is_A_strided && !is_B_strided)
              detail::prod(detail::array_wrap<constT,true>(A,lda,nlda), detail::array_wrap<constT,false>(B,ldb,nldb), detail::array_wrap<T,true>(C,ldc,nldc),M, N, K, alpha, beta);
            else if(!is_A_strided && is_B_strided)
              detail::prod(detail::array_wrap<constT,false>(A,lda,nlda), detail::array_wrap<constT,true>(B,ldb,nldb), detail::array_wrap<T,true>(C,ldc,nldc),M, N, K, alpha, beta);
            else if(!is_A_strided && !is_B_strided)
              detail::prod(detail::array_wrap<constT,false>(A,lda,nlda), detail::array_wrap<constT,false>(B,ldb,nldb), detail::array_wrap<T,true>(C,ldc,nldc),M, N, K, alpha, beta);
          }
          else{
            if(is_A_strided && is_B_strided)
               detail::prod(detail::array_wrap<constT,true>(A,lda,nlda), detail::array_wrap<constT,true>(B,ldb,nldb), detail::array_wrap<T,false>(C,ldc,nldc),M, N, K, alpha, beta);
            else if(is_A_strided && !is_B_strided)
              detail::prod(detail::array_wrap<constT,true>(A,lda,nlda), detail::array_wrap<constT,false>(B,ldb,nldb), detail::array_wrap<T,false>(C,ldc,nldc),M, N, K, alpha, beta);
            else if(!is_A_strided && is_B_strided)
              detail::prod(detail::array_wrap<constT,false>(A,lda,nlda), detail::array_wrap<constT,true>(B,ldb,nldb), detail::array_wrap<T,false>(C,ldc,nldc),M, N, K, alpha, beta);
            else if(!is_A_strided && !is_B_strided)
              detail::prod(detail::array_wrap<constT,false>(A,lda,nlda), detail::array_wrap<constT,false>(B,ldb,nldb), detail::array_wrap<T,false>(C,ldc,nldc),M, N, K, alpha, beta);
          }

        }




      }
    }
  }
}

#endif
