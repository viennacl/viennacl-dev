#ifndef VIENNACL_BACKEND_BLAS_HPP
#define VIENNACL_BACKEND_BLAS_HPP

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

/** @file viennacl/backend/blas.hpp
    @brief Main interface routines for overriding some underlying BLAS functions
*/

#include <cassert>

#include "viennacl/backend/mem_handle.hpp"

#ifdef VIENNACL_WITH_CBLAS
#include "cblas.h"
#endif


#ifdef VIENNACL_WITH_CUBLAS
#include "cublas.h"
#endif

namespace viennacl
{
  namespace backend
  {


      template<class TransposeType>
      struct matrix_blas{
          matrix_blas(TransposeType _trans, TransposeType _notrans, bool is_row_major, bool is_transposed
                              ,vcl_size_t internal_size1, vcl_size_t internal_size2
                              ,vcl_size_t start1, vcl_size_t start2
                              ,vcl_size_t stride1, vcl_size_t stride2){
            trans = !(is_transposed^is_row_major)?_trans:_notrans;
            negtrans = (trans==_trans)?_notrans:_trans;
            ld = is_row_major?stride1*internal_size2:stride2*internal_size1;
            off = is_row_major?start1*internal_size2+start2:start2*internal_size1+start1;
          }
          vcl_size_t ld;
          vcl_size_t off;
          TransposeType trans;
          TransposeType negtrans;
      };


#ifdef VIENNACL_WITH_CBLAS
      template<class T>
      struct cblas;

      template<>
      struct cblas<float>{
        private:
          typedef matrix_blas<CBLAS_TRANSPOSE> wrapper;
        public:
          static bool gemm(bool C_row_major, bool A_row_major, bool B_row_major
                           ,bool is_A_trans, bool is_B_trans
                           ,const vcl_size_t M, const vcl_size_t N, const vcl_size_t K, const float alpha
                           ,viennacl::backend::mem_handle const hA, const vcl_size_t A_internal_size1, const vcl_size_t A_internal_size2
                           ,const vcl_size_t A_start1, const vcl_size_t A_start2, const vcl_size_t A_inc1, const vcl_size_t A_inc2
                           ,viennacl::backend::mem_handle const hB, const vcl_size_t B_internal_size1, const vcl_size_t B_internal_size2
                           ,const vcl_size_t B_start1, const vcl_size_t B_start2, const vcl_size_t B_inc1, const vcl_size_t B_inc2
                           ,const float beta, viennacl::backend::mem_handle  hC, const vcl_size_t C_internal_size1, const vcl_size_t C_internal_size2
                           ,const vcl_size_t C_start1, const vcl_size_t C_start2, const vcl_size_t C_inc1, const vcl_size_t C_inc2)
          {
            if(A_inc1!=1 || A_inc2!=1 || B_inc1!=1 || B_inc2!=1 || C_inc1!=1 || C_inc2!=1)
              return false;

            float const * pA = reinterpret_cast<float*>(hA.ram_handle().get());
            float const * pB = reinterpret_cast<float*>(hB.ram_handle().get());
            float * pC = reinterpret_cast<float*>(hC.ram_handle().get());

            wrapper A(CblasTrans, CblasNoTrans, A_row_major,is_A_trans,A_internal_size1, A_internal_size2, A_start1, A_start2, A_inc1, A_inc2);
            wrapper B(CblasTrans, CblasNoTrans, B_row_major,is_B_trans,B_internal_size1, B_internal_size2, B_start1, B_start2, B_inc1, B_inc2);
            wrapper C(CblasTrans, CblasNoTrans, C_row_major,false,C_internal_size1, C_internal_size2, C_start1, C_start2, C_inc1, C_inc2);

            if(C_row_major)
              cblas_sgemm(CblasColMajor,B.trans, A.trans, N, M, K, alpha, pB+B.off, B.ld, pA+A.off, A.ld, beta, pC+C.off, C.ld);
            else
              cblas_sgemm(CblasColMajor,A.negtrans, B.negtrans, M, N, K, alpha, pA+A.off, A.ld, pB+B.off, B.ld, beta, pC+C.off, C.ld);

            return true;
          }
      };


      template<>
      struct cblas<double>{
        private:
          typedef matrix_blas<CBLAS_TRANSPOSE> wrapper;
        public:
          static bool gemm(bool C_row_major, bool A_row_major, bool B_row_major
                           ,bool is_A_trans, bool is_B_trans
                           ,const vcl_size_t M, const vcl_size_t N, const vcl_size_t K, const double alpha
                           ,double const *hA, const vcl_size_t A_internal_size1, const vcl_size_t A_internal_size2
                           ,const vcl_size_t A_start1, const vcl_size_t A_start2, const vcl_size_t A_inc1, const vcl_size_t A_inc2
                           ,double const *hB, const vcl_size_t B_internal_size1, const vcl_size_t B_internal_size2
                           ,const vcl_size_t B_start1, const vcl_size_t B_start2, const vcl_size_t B_inc1, const vcl_size_t B_inc2
                           ,const double beta, double *hC, const vcl_size_t C_internal_size1, const vcl_size_t C_internal_size2
                           ,const vcl_size_t C_start1, const vcl_size_t C_start2, const vcl_size_t C_inc1, const vcl_size_t C_inc2)
          {
            if(A_inc1!=1 || A_inc2!=1 || B_inc1!=1 || B_inc2!=1 || C_inc1!=1 || C_inc2!=1)
              return false;

            wrapper A(CblasTrans, CblasNoTrans, A_row_major,is_A_trans,A_internal_size1, A_internal_size2, A_start1, A_start2, A_inc1, A_inc2);
            wrapper B(CblasTrans, CblasNoTrans, B_row_major,is_B_trans,B_internal_size1, B_internal_size2, B_start1, B_start2, B_inc1, B_inc2);
            wrapper C(CblasTrans, CblasNoTrans, C_row_major,false,C_internal_size1, C_internal_size2, C_start1, C_start2, C_inc1, C_inc2);

            double const * pA = reinterpret_cast<double*>(hA.ram_handle().get());
            double const * pB = reinterpret_cast<double*>(hB.ram_handle().get());
            double * pC = reinterpret_cast<double*>(hC.ram_handle().get());

            if(C_row_major)
              cblas_dgemm(CblasColMajor,B.trans, A.trans, N, M, K, alpha, pB+B.off, B.ld, pA+A.off, A.ld, beta, pC+C.off, C.ld);
            else
              cblas_dgemm(CblasColMajor,A.negtrans, B.negtrans, M, N, K, alpha, pA+A.off, A.ld, pB+B.off, B.ld, beta, pC+C.off, C.ld);

            return true;
          }
      };
#endif


#ifdef VIENNACL_WITH_CUBLAS
      template<class T>
      struct cublas;

      template<>
      struct cublas<float>{
        private:
          typedef matrix_blas<char> wrapper;
        public:
          static bool gemm(bool C_row_major, bool A_row_major, bool B_row_major
                           ,bool is_A_trans, bool is_B_trans
                           ,const vcl_size_t M, const vcl_size_t N, const vcl_size_t K, const float alpha
                           ,viennacl::backend::mem_handle const hA, const vcl_size_t A_internal_size1, const vcl_size_t A_internal_size2
                           ,const vcl_size_t A_start1, const vcl_size_t A_start2, const vcl_size_t A_inc1, const vcl_size_t A_inc2
                           ,viennacl::backend::mem_handle const hB, const vcl_size_t B_internal_size1, const vcl_size_t B_internal_size2
                           ,const vcl_size_t B_start1, const vcl_size_t B_start2, const vcl_size_t B_inc1, const vcl_size_t B_inc2
                           ,const float beta, viennacl::backend::mem_handle  hC, const vcl_size_t C_internal_size1, const vcl_size_t C_internal_size2
                           ,const vcl_size_t C_start1, const vcl_size_t C_start2, const vcl_size_t C_inc1, const vcl_size_t C_inc2)
          {
            if(A_inc1!=1 || A_inc2!=1 || B_inc1!=1 || B_inc2!=1 || C_inc1!=1 || C_inc2!=1)
              return false;

            float const * pA = reinterpret_cast<float*>(hA.cuda_handle().get());
            float const * pB = reinterpret_cast<float*>(hB.cuda_handle().get());
            float * pC = reinterpret_cast<float*>(hC.cuda_handle().get());

            wrapper A('T', 'N', A_row_major, is_A_trans, A_internal_size1, A_internal_size2, A_start1, A_start2, A_inc1, A_inc2);
            wrapper B('T', 'N', B_row_major, is_B_trans, B_internal_size1, B_internal_size2, B_start1, B_start2, B_inc1, B_inc2);
            wrapper C('T', 'N', C_row_major, false, C_internal_size1, C_internal_size2, C_start1, C_start2, C_inc1, C_inc2);

            if(C_row_major)
              cublasSgemm(B.trans, A.trans, N, M, K, alpha, pB+B.off, B.ld, pA+A.off, A.ld, beta, pC+C.off, C.ld);
            else
              cublasSgemm(A.negtrans, B.negtrans, M, N, K, alpha, pA+A.off, A.ld, pB+B.off, B.ld, beta, pC+C.off, C.ld);

            return true;
          }
      };


      template<>
      struct cublas<double>{
        private:
          typedef matrix_blas<char> wrapper;
        public:
          static bool gemm(bool C_row_major, bool A_row_major, bool B_row_major
                           ,bool is_A_trans, bool is_B_trans
                           ,const vcl_size_t M, const vcl_size_t N, const vcl_size_t K, const double alpha
                           ,viennacl::backend::mem_handle const hA, const vcl_size_t A_internal_size1, const vcl_size_t A_internal_size2
                           ,const vcl_size_t A_start1, const vcl_size_t A_start2, const vcl_size_t A_inc1, const vcl_size_t A_inc2
                           ,viennacl::backend::mem_handle const hB, const vcl_size_t B_internal_size1, const vcl_size_t B_internal_size2
                           ,const vcl_size_t B_start1, const vcl_size_t B_start2, const vcl_size_t B_inc1, const vcl_size_t B_inc2
                           ,const double beta, viennacl::backend::mem_handle hC, const vcl_size_t C_internal_size1, const vcl_size_t C_internal_size2
                           ,const vcl_size_t C_start1, const vcl_size_t C_start2, const vcl_size_t C_inc1, const vcl_size_t C_inc2)
          {
            if(A_inc1!=1 || A_inc2!=1 || B_inc1!=1 || B_inc2!=1 || C_inc1!=1 || C_inc2!=1)
              return false;

            double const * pA = reinterpret_cast<double *>(hA.cuda_handle().get());
            double const * pB = reinterpret_cast<double *>(hB.cuda_handle().get());
            double * pC = reinterpret_cast<double *>(hC.cuda_handle().get());

            wrapper A('T', 'N', A_row_major,is_A_trans,A_internal_size1, A_internal_size2, A_start1, A_start2, A_inc1, A_inc2);
            wrapper B('T', 'N', B_row_major,is_B_trans,B_internal_size1, B_internal_size2, B_start1, B_start2, B_inc1, B_inc2);
            wrapper C('T', 'N', C_row_major,false,C_internal_size1, C_internal_size2, C_start1, C_start2, C_inc1, C_inc2);

            if(C_row_major)
              cublasDgemm(B.trans, A.trans, N, M, K, alpha, pB+B.off, B.ld, pA+A.off, A.ld, beta, pC+C.off, C.ld);
            else
              cublasDgemm(A.negtrans, B.negtrans, M, N, K, alpha, pA+A.off, A.ld, pB+B.off, B.ld, beta, pC+C.off, C.ld);

            return true;
          }
      };
#endif


    template<class T>
    struct blas_function_types{
        typedef T value_type;

        typedef bool (*gemm)(bool /*C_row_major*/, bool /*A_row_major*/, bool /*B_row_major*/
                             ,bool /*is_A_trans*/, bool /*is_B_trans*/
                             ,const vcl_size_t /*M*/, const vcl_size_t /*N*/, const vcl_size_t /*K*/, const T /*alpha*/
                             ,viennacl::backend::mem_handle  const /*A*/ , const vcl_size_t /*A_internal_size1*/, const vcl_size_t /*A_internal_size2*/
                             ,const vcl_size_t /*A_start1*/, const vcl_size_t /*A_start2*/, const vcl_size_t /*A_inc1*/, const vcl_size_t /*A_inc2*/
                             ,viennacl::backend::mem_handle  const /*B*/, const vcl_size_t /*B_internal_size1*/, const vcl_size_t /*B_internal_size2*/
                             ,const vcl_size_t /*B_start1*/, const vcl_size_t /*B_start2*/, const vcl_size_t /*B_inc1*/, const vcl_size_t /*B_inc2*/
                             ,const T /*beta*/, viennacl::backend::mem_handle /*C*/, const vcl_size_t /*C_internal_size1*/, const vcl_size_t /*C_internal_size2*/
                             ,const vcl_size_t /*C_start1*/, const vcl_size_t /*C_start2*/, const vcl_size_t /*C_inc1*/, const vcl_size_t /*C_inc2*/);
    };

#define HAS_MEM_FUNC(func, name)                                        \
  template<typename T, typename Sign>                                 \
  struct name {                                                       \
  typedef char yes[1];                                            \
  typedef char no [2];                                            \
  template <typename U, U> struct type_check;                     \
  template <typename _1> static yes &chk(type_check<Sign, &_1::gemm> *); \
  template <typename   > static no  &chk(...);                    \
  static bool const value = sizeof(chk<T>(0)) == sizeof(yes);     \
  }



    namespace detail{
      template<typename U, typename Ret>
      struct init_gemm { static const Ret value() { return NULL; } };
      template<template<class> class U, typename Ret>
      struct init_gemm< U<float>, Ret> {  static const Ret value() { return &U<float>::gemm; } };
      template<template<class> class U, typename Ret>
      struct init_gemm< U<double>, Ret> {  static const Ret value() { return &U<double>::gemm; } };
    }

    template<class T>
    class blas{
      public:
        typedef blas_function_types<T> functions_type;
      private:
        typedef T value_type;
        typedef typename functions_type::gemm gemm_t;


      public:
        blas() : gemm_(NULL) {
#ifdef VIENNACL_WITH_CBLAS
          gemm_ = detail::init_gemm< viennacl::backend::cblas<value_type>, gemm_t>::value();
#endif
#ifdef VIENNACL_WITH_CUBLAS
          gemm_ = detail::init_gemm< viennacl::backend::cublas<value_type>, gemm_t >::value();
#endif
        }


        gemm_t gemm() const { return gemm_; }
        void gemm(gemm_t fptr) { gemm_ = fptr; }

      private:
        gemm_t gemm_;
    };

  }

}

#endif
