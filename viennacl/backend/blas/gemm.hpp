#ifndef VIENNACL_BACKEND_BLAS_GEMM_HPP
#define VIENNACL_BACKEND_BLAS_GEMM_HPP

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

namespace viennacl
{
  namespace backend
  {


      namespace wrap{

        template<class T, class OrderType, OrderType ColMajorVal, class TransposeType,
                 void (*_gemm)(const OrderType, const TransposeType, const TransposeType, const int, const int,
                                     const int, const T, T const *, const int, T const *, const int, const T, T*, const int)>
        class gemm_cblas{
          public:
            static void execute(const TransposeType transA, const TransposeType transB, const int M , const int N, const int K,
                              const T alpha, T const * A, const int offa, const int lda,
                             T const * B, const int offb, const int ldb,
                             const T beta, T* C, const int offc, const int ldc)
            {
              _gemm(ColMajorVal, transA, transB, M, N, K, alpha, A+offa, lda, B+offb, ldb, beta, C+offc, ldc);
            }
        };

        template<class T, class TransposeType,
                 void (*_gemm)(const TransposeType, const TransposeType, const int, const int,
                                     const int, const T, T const *, const int, T const *, const int, const T, T*, const int)>
        class gemm_blas{
          public:
            static void execute(const TransposeType transA, const TransposeType transB, const int M , const int N, const int K,
                              const T alpha, T const * A, const int offa, const int lda,
                             T const * B, const int offb, const int ldb,
                             const T beta, T* C, const int offc, const int ldc)
            {
              _gemm(transA, transB, M, N, K, alpha, A+offa, lda, B+offb, ldb, beta, C+offc, ldc);
            }
        };

      }


    namespace detail{
      template<typename U, typename Ret>
      struct init_gemm { static const Ret value() { return NULL; } };
      template<template<class> class U, typename Ret>
      struct init_gemm< U<float>, Ret> {  static const Ret value() { return &U<float>::gemm; } };
      template<template<class> class U, typename Ret>
      struct init_gemm< U<double>, Ret> {  static const Ret value() { return &U<double>::gemm; } };
    }

  }

}

#endif
