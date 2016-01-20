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

/** \file tests/src/libviennacl_blas3.cpp  Testing the BLAS level 3 routines in the ViennaCL BLAS-like shared library
*   \test Testing the BLAS level 3 routines in the ViennaCL BLAS-like shared library
**/


// include necessary system headers
#include <iostream>
#include <vector>

// Some helper functions for this tutorial:
#include "viennacl.hpp"

#include "viennacl/tools/random.hpp"


#include "viennacl/vector.hpp"

template<typename ScalarType>
ScalarType diff(ScalarType const & s1, ScalarType const & s2)
{
  if (s1 > s2 || s1 < s2)
    return (s1 - s2) / std::max(std::fabs(s1), std::fabs(s2));
  return ScalarType(0);
}

template<typename ScalarType, typename ViennaCLVectorType>
ScalarType diff(std::vector<ScalarType> const & v1, ViennaCLVectorType const & vcl_vec)
{
   std::vector<ScalarType> v2_cpu(vcl_vec.size());
   viennacl::backend::finish();
   viennacl::copy(vcl_vec, v2_cpu);

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
            ViennaCLInt start1, ViennaCLInt start2,
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
}


void test_blas(ViennaCLBackend my_backend,
               float eps_float, double eps_double,
               std::vector<float> & C_float, std::vector<double> & C_double,
               std::vector<float> & A_float, std::vector<double> & A_double,
               std::vector<float> & B_float, std::vector<double> & B_double,
               ViennaCLOrder order_C, ViennaCLOrder order_A, ViennaCLOrder order_B,
               ViennaCLTranspose trans_A, ViennaCLTranspose trans_B,
               viennacl::vector<float> & host_C_float, viennacl::vector<double> & host_C_double,
               viennacl::vector<float> & host_A_float, viennacl::vector<double> & host_A_double,
               viennacl::vector<float> & host_B_float, viennacl::vector<double> & host_B_double
#ifdef VIENNACL_WITH_CUDA
               , viennacl::vector<float> & cuda_C_float, viennacl::vector<double> & cuda_C_double
               , viennacl::vector<float> & cuda_A_float, viennacl::vector<double> & cuda_A_double
               , viennacl::vector<float> & cuda_B_float, viennacl::vector<double> & cuda_B_double
#endif
#ifdef VIENNACL_WITH_OPENCL
               , viennacl::vector<float> & opencl_C_float, viennacl::vector<double> * opencl_C_double
               , viennacl::vector<float> & opencl_A_float, viennacl::vector<double> * opencl_A_double
               , viennacl::vector<float> & opencl_B_float, viennacl::vector<double> * opencl_B_double
#endif
               );

void test_blas(ViennaCLBackend my_backend,
               float eps_float, double eps_double,
               std::vector<float> & C_float, std::vector<double> & C_double,
               std::vector<float> & A_float, std::vector<double> & A_double,
               std::vector<float> & B_float, std::vector<double> & B_double,
               ViennaCLOrder order_C, ViennaCLOrder order_A, ViennaCLOrder order_B,
               ViennaCLTranspose trans_A, ViennaCLTranspose trans_B,
               viennacl::vector<float> & host_C_float, viennacl::vector<double> & host_C_double,
               viennacl::vector<float> & host_A_float, viennacl::vector<double> & host_A_double,
               viennacl::vector<float> & host_B_float, viennacl::vector<double> & host_B_double
#ifdef VIENNACL_WITH_CUDA
               , viennacl::vector<float> & cuda_C_float, viennacl::vector<double> & cuda_C_double
               , viennacl::vector<float> & cuda_A_float, viennacl::vector<double> & cuda_A_double
               , viennacl::vector<float> & cuda_B_float, viennacl::vector<double> & cuda_B_double
#endif
#ifdef VIENNACL_WITH_OPENCL
               , viennacl::vector<float> & opencl_C_float, viennacl::vector<double> * opencl_C_double
               , viennacl::vector<float> & opencl_A_float, viennacl::vector<double> * opencl_A_double
               , viennacl::vector<float> & opencl_B_float, viennacl::vector<double> * opencl_B_double
#endif
               )
{
  ViennaCLInt C_size1   = 42;
  ViennaCLInt C_size2   = 43;
  ViennaCLInt C_start1  = 10;
  ViennaCLInt C_start2  = 11;
  ViennaCLInt C_stride1 = 2;
  ViennaCLInt C_stride2 = 3;
  ViennaCLInt C_rows    = C_size1 * C_stride1 + C_start1 + 5;
  ViennaCLInt C_columns = C_size2 * C_stride2 + C_start2 + 5;

  ViennaCLInt A_size1   = trans_A ? 44 : 42;
  ViennaCLInt A_size2   = trans_A ? 42 : 44;
  ViennaCLInt A_start1  = 12;
  ViennaCLInt A_start2  = 13;
  ViennaCLInt A_stride1 = 4;
  ViennaCLInt A_stride2 = 5;
  ViennaCLInt A_rows    = A_size1 * A_stride1 + A_start1 + 5;
  ViennaCLInt A_columns = A_size2 * A_stride2 + A_start2 + 5;

  ViennaCLInt B_size1   = trans_B ? 43 : 44;
  ViennaCLInt B_size2   = trans_B ? 44 : 43;
  ViennaCLInt B_start1  = 14;
  ViennaCLInt B_start2  = 15;
  ViennaCLInt B_stride1 = 6;
  ViennaCLInt B_stride2 = 7;
  ViennaCLInt B_rows    = B_size1 * B_stride1 + B_start1 + 5;
  ViennaCLInt B_columns = B_size2 * B_stride2 + B_start2 + 5;

  // Compute reference:
  ViennaCLInt size_k = trans_A ? A_size1 : A_size2;
  for (ViennaCLInt i=0; i<C_size1; ++i)
    for (ViennaCLInt j=0; j<C_size2; ++j)
    {
      float val_float = 0;
      double val_double = 0;
      for (ViennaCLInt k=0; k<size_k; ++k)
      {
        float  val_A_float  = get_value(A_float,  i, k, A_start1, A_start2, A_stride1, A_stride2, A_rows, A_columns, order_A, trans_A);
        double val_A_double = get_value(A_double, i, k, A_start1, A_start2, A_stride1, A_stride2, A_rows, A_columns, order_A, trans_A);

        float  val_B_float  = get_value(B_float,  k, j, B_start1, B_start2, B_stride1, B_stride2, B_rows, B_columns, order_B, trans_B);
        double val_B_double = get_value(B_double, k, j, B_start1, B_start2, B_stride1, B_stride2, B_rows, B_columns, order_B, trans_B);

        val_float  += val_A_float  * val_B_float;
        val_double += val_A_double * val_B_double;
      }

      // write result
      if (order_C == ViennaCLRowMajor)
      {
        C_float [static_cast<std::size_t>((i*C_stride1 + C_start1) * C_columns + (j*C_stride2 + C_start2))] = val_float;
        C_double[static_cast<std::size_t>((i*C_stride1 + C_start1) * C_columns + (j*C_stride2 + C_start2))] = val_double;
      }
      else
      {
        C_float [static_cast<std::size_t>((i*C_stride1 + C_start1) + (j*C_stride2 + C_start2) * C_rows)] = val_float;
        C_double[static_cast<std::size_t>((i*C_stride1 + C_start1) + (j*C_stride2 + C_start2) * C_rows)] = val_double;
      }
    }

  // Run GEMM and compare results:
  ViennaCLHostSgemm(my_backend,
                    order_A, trans_A, order_B, trans_B, order_C,
                    C_size1, C_size2, size_k,
                    1.0f,
                    viennacl::linalg::host_based::detail::extract_raw_pointer<float>(host_A_float), A_start1, A_start2, A_stride1, A_stride2, (order_A == ViennaCLRowMajor) ? A_columns : A_rows,
                    viennacl::linalg::host_based::detail::extract_raw_pointer<float>(host_B_float), B_start1, B_start2, B_stride1, B_stride2, (order_B == ViennaCLRowMajor) ? B_columns : B_rows,
                    0.0f,
                    viennacl::linalg::host_based::detail::extract_raw_pointer<float>(host_C_float), C_start1, C_start2, C_stride1, C_stride2, (order_C == ViennaCLRowMajor) ? C_columns : C_rows);
  check(C_float, host_C_float, eps_float);

  ViennaCLHostDgemm(my_backend,
                    order_A, trans_A, order_B, trans_B, order_C,
                    C_size1, C_size2, size_k,
                    1.0,
                    viennacl::linalg::host_based::detail::extract_raw_pointer<double>(host_A_double), A_start1, A_start2, A_stride1, A_stride2, (order_A == ViennaCLRowMajor) ? A_columns : A_rows,
                    viennacl::linalg::host_based::detail::extract_raw_pointer<double>(host_B_double), B_start1, B_start2, B_stride1, B_stride2, (order_B == ViennaCLRowMajor) ? B_columns : B_rows,
                    0.0,
                    viennacl::linalg::host_based::detail::extract_raw_pointer<double>(host_C_double), C_start1, C_start2, C_stride1, C_stride2, (order_C == ViennaCLRowMajor) ? C_columns : C_rows);
  check(C_double, host_C_double, eps_double);

#ifdef VIENNACL_WITH_CUDA
  ViennaCLCUDASgemm(my_backend,
                    order_A, trans_A, order_B, trans_B, order_C,
                    C_size1, C_size2, size_k,
                    1.0f,
                    viennacl::cuda_arg(cuda_A_float), A_start1, A_start2, A_stride1, A_stride2, (order_A == ViennaCLRowMajor) ? A_columns : A_rows,
                    viennacl::cuda_arg(cuda_B_float), B_start1, B_start2, B_stride1, B_stride2, (order_B == ViennaCLRowMajor) ? B_columns : B_rows,
                    0.0f,
                    viennacl::cuda_arg(cuda_C_float), C_start1, C_start2, C_stride1, C_stride2, (order_C == ViennaCLRowMajor) ? C_columns : C_rows);
  check(C_float, cuda_C_float, eps_float);

  ViennaCLCUDADgemm(my_backend,
                    order_A, trans_A, order_B, trans_B, order_C,
                    C_size1, C_size2, size_k,
                    1.0,
                    viennacl::cuda_arg(cuda_A_double), A_start1, A_start2, A_stride1, A_stride2, (order_A == ViennaCLRowMajor) ? A_columns : A_rows,
                    viennacl::cuda_arg(cuda_B_double), B_start1, B_start2, B_stride1, B_stride2, (order_B == ViennaCLRowMajor) ? B_columns : B_rows,
                    0.0,
                    viennacl::cuda_arg(cuda_C_double), C_start1, C_start2, C_stride1, C_stride2, (order_C == ViennaCLRowMajor) ? C_columns : C_rows);
  check(C_double, cuda_C_double, eps_double);
#endif

#ifdef VIENNACL_WITH_OPENCL
  ViennaCLOpenCLSgemm(my_backend,
                    order_A, trans_A, order_B, trans_B, order_C,
                    C_size1, C_size2, size_k,
                    1.0f,
                    viennacl::traits::opencl_handle(opencl_A_float), A_start1, A_start2, A_stride1, A_stride2, (order_A == ViennaCLRowMajor) ? A_columns : A_rows,
                    viennacl::traits::opencl_handle(opencl_B_float), B_start1, B_start2, B_stride1, B_stride2, (order_B == ViennaCLRowMajor) ? B_columns : B_rows,
                    0.0f,
                    viennacl::traits::opencl_handle(opencl_C_float), C_start1, C_start2, C_stride1, C_stride2, (order_C == ViennaCLRowMajor) ? C_columns : C_rows);
  check(C_float, opencl_C_float, eps_float);

  if (opencl_A_double != NULL && opencl_B_double != NULL && opencl_C_double != NULL)
  {
    ViennaCLOpenCLDgemm(my_backend,
                      order_A, trans_A, order_B, trans_B, order_C,
                      C_size1, C_size2, size_k,
                      1.0,
                      viennacl::traits::opencl_handle(*opencl_A_double), A_start1, A_start2, A_stride1, A_stride2, (order_A == ViennaCLRowMajor) ? A_columns : A_rows,
                      viennacl::traits::opencl_handle(*opencl_B_double), B_start1, B_start2, B_stride1, B_stride2, (order_B == ViennaCLRowMajor) ? B_columns : B_rows,
                      0.0,
                      viennacl::traits::opencl_handle(*opencl_C_double), C_start1, C_start2, C_stride1, C_stride2, (order_C == ViennaCLRowMajor) ? C_columns : C_rows);
    check(C_double, *opencl_C_double, eps_double);
  }
#endif

  std::cout << std::endl;
}

void test_blas(ViennaCLBackend my_backend,
               float eps_float, double eps_double,
               std::vector<float> & C_float, std::vector<double> & C_double,
               std::vector<float> & A_float, std::vector<double> & A_double,
               std::vector<float> & B_float, std::vector<double> & B_double,
               ViennaCLOrder order_C, ViennaCLOrder order_A, ViennaCLOrder order_B,
               viennacl::vector<float> & host_C_float, viennacl::vector<double> & host_C_double,
               viennacl::vector<float> & host_A_float, viennacl::vector<double> & host_A_double,
               viennacl::vector<float> & host_B_float, viennacl::vector<double> & host_B_double
#ifdef VIENNACL_WITH_CUDA
               , viennacl::vector<float> & cuda_C_float, viennacl::vector<double> & cuda_C_double
               , viennacl::vector<float> & cuda_A_float, viennacl::vector<double> & cuda_A_double
               , viennacl::vector<float> & cuda_B_float, viennacl::vector<double> & cuda_B_double
#endif
#ifdef VIENNACL_WITH_OPENCL
               , viennacl::vector<float> & opencl_C_float, viennacl::vector<double> * opencl_C_double
               , viennacl::vector<float> & opencl_A_float, viennacl::vector<double> * opencl_A_double
               , viennacl::vector<float> & opencl_B_float, viennacl::vector<double> * opencl_B_double
#endif
               );

void test_blas(ViennaCLBackend my_backend,
               float eps_float, double eps_double,
               std::vector<float> & C_float, std::vector<double> & C_double,
               std::vector<float> & A_float, std::vector<double> & A_double,
               std::vector<float> & B_float, std::vector<double> & B_double,
               ViennaCLOrder order_C, ViennaCLOrder order_A, ViennaCLOrder order_B,
               viennacl::vector<float> & host_C_float, viennacl::vector<double> & host_C_double,
               viennacl::vector<float> & host_A_float, viennacl::vector<double> & host_A_double,
               viennacl::vector<float> & host_B_float, viennacl::vector<double> & host_B_double
#ifdef VIENNACL_WITH_CUDA
               , viennacl::vector<float> & cuda_C_float, viennacl::vector<double> & cuda_C_double
               , viennacl::vector<float> & cuda_A_float, viennacl::vector<double> & cuda_A_double
               , viennacl::vector<float> & cuda_B_float, viennacl::vector<double> & cuda_B_double
#endif
#ifdef VIENNACL_WITH_OPENCL
               , viennacl::vector<float> & opencl_C_float, viennacl::vector<double> * opencl_C_double
               , viennacl::vector<float> & opencl_A_float, viennacl::vector<double> * opencl_A_double
               , viennacl::vector<float> & opencl_B_float, viennacl::vector<double> * opencl_B_double
#endif
               )
{
  std::cout << "    -> trans-trans: ";
  test_blas(my_backend,
            eps_float, eps_double,
            C_float, C_double, A_float, A_double, B_float, B_double,
            order_C, order_A, order_B,
            ViennaCLTrans, ViennaCLTrans,
            host_C_float, host_C_double, host_A_float, host_A_double, host_B_float, host_B_double
#ifdef VIENNACL_WITH_CUDA
            , cuda_C_float, cuda_C_double, cuda_A_float, cuda_A_double, cuda_B_float, cuda_B_double
#endif
#ifdef VIENNACL_WITH_OPENCL
            , opencl_C_float, opencl_C_double, opencl_A_float, opencl_A_double, opencl_B_float, opencl_B_double
#endif
            );

  std::cout << "    -> trans-no:    ";
  test_blas(my_backend,
            eps_float, eps_double,
            C_float, C_double, A_float, A_double, B_float, B_double,
            order_C, order_A, order_B,
            ViennaCLTrans, ViennaCLNoTrans,
            host_C_float, host_C_double, host_A_float, host_A_double, host_B_float, host_B_double
#ifdef VIENNACL_WITH_CUDA
            , cuda_C_float, cuda_C_double, cuda_A_float, cuda_A_double, cuda_B_float, cuda_B_double
#endif
#ifdef VIENNACL_WITH_OPENCL
            , opencl_C_float, opencl_C_double, opencl_A_float, opencl_A_double, opencl_B_float, opencl_B_double
#endif
            );

  std::cout << "    -> no-trans:    ";
  test_blas(my_backend,
            eps_float, eps_double,
            C_float, C_double, A_float, A_double, B_float, B_double,
            order_C, order_A, order_B,
            ViennaCLNoTrans, ViennaCLTrans,
            host_C_float, host_C_double, host_A_float, host_A_double, host_B_float, host_B_double
#ifdef VIENNACL_WITH_CUDA
            , cuda_C_float, cuda_C_double, cuda_A_float, cuda_A_double, cuda_B_float, cuda_B_double
#endif
#ifdef VIENNACL_WITH_OPENCL
            , opencl_C_float, opencl_C_double, opencl_A_float, opencl_A_double, opencl_B_float, opencl_B_double
#endif
            );

  std::cout << "    -> no-no:       ";
  test_blas(my_backend,
            eps_float, eps_double,
            C_float, C_double, A_float, A_double, B_float, B_double,
            order_C, order_A, order_B,
            ViennaCLNoTrans, ViennaCLNoTrans,
            host_C_float, host_C_double, host_A_float, host_A_double, host_B_float, host_B_double
#ifdef VIENNACL_WITH_CUDA
            , cuda_C_float, cuda_C_double, cuda_A_float, cuda_A_double, cuda_B_float, cuda_B_double
#endif
#ifdef VIENNACL_WITH_OPENCL
            , opencl_C_float, opencl_C_double, opencl_A_float, opencl_A_double, opencl_B_float, opencl_B_double
#endif
            );

}


void test_blas(ViennaCLBackend my_backend,
               float eps_float, double eps_double,
               std::vector<float> & C_float, std::vector<double> & C_double,
               std::vector<float> & A_float, std::vector<double> & A_double,
               std::vector<float> & B_float, std::vector<double> & B_double,
               viennacl::vector<float> & host_C_float, viennacl::vector<double> & host_C_double,
               viennacl::vector<float> & host_A_float, viennacl::vector<double> & host_A_double,
               viennacl::vector<float> & host_B_float, viennacl::vector<double> & host_B_double
#ifdef VIENNACL_WITH_CUDA
               , viennacl::vector<float> & cuda_C_float, viennacl::vector<double> & cuda_C_double
               , viennacl::vector<float> & cuda_A_float, viennacl::vector<double> & cuda_A_double
               , viennacl::vector<float> & cuda_B_float, viennacl::vector<double> & cuda_B_double
#endif
#ifdef VIENNACL_WITH_OPENCL
               , viennacl::vector<float> & opencl_C_float, viennacl::vector<double> * opencl_C_double
               , viennacl::vector<float> & opencl_A_float, viennacl::vector<double> * opencl_A_double
               , viennacl::vector<float> & opencl_B_float, viennacl::vector<double> * opencl_B_double
#endif
               );

void test_blas(ViennaCLBackend my_backend,
               float eps_float, double eps_double,
               std::vector<float> & C_float, std::vector<double> & C_double,
               std::vector<float> & A_float, std::vector<double> & A_double,
               std::vector<float> & B_float, std::vector<double> & B_double,
               viennacl::vector<float> & host_C_float, viennacl::vector<double> & host_C_double,
               viennacl::vector<float> & host_A_float, viennacl::vector<double> & host_A_double,
               viennacl::vector<float> & host_B_float, viennacl::vector<double> & host_B_double
#ifdef VIENNACL_WITH_CUDA
               , viennacl::vector<float> & cuda_C_float, viennacl::vector<double> & cuda_C_double
               , viennacl::vector<float> & cuda_A_float, viennacl::vector<double> & cuda_A_double
               , viennacl::vector<float> & cuda_B_float, viennacl::vector<double> & cuda_B_double
#endif
#ifdef VIENNACL_WITH_OPENCL
               , viennacl::vector<float> & opencl_C_float, viennacl::vector<double> * opencl_C_double
               , viennacl::vector<float> & opencl_A_float, viennacl::vector<double> * opencl_A_double
               , viennacl::vector<float> & opencl_B_float, viennacl::vector<double> * opencl_B_double
#endif
               )
{
  std::cout << "  -> C: row, A: row, B: row" << std::endl;
  test_blas(my_backend,
            eps_float, eps_double,
            C_float, C_double, A_float, A_double, B_float, B_double,
            ViennaCLRowMajor, ViennaCLRowMajor, ViennaCLRowMajor,
            host_C_float, host_C_double, host_A_float, host_A_double, host_B_float, host_B_double
#ifdef VIENNACL_WITH_CUDA
            , cuda_C_float, cuda_C_double, cuda_A_float, cuda_A_double, cuda_B_float, cuda_B_double
#endif
#ifdef VIENNACL_WITH_OPENCL
            , opencl_C_float, opencl_C_double, opencl_A_float, opencl_A_double, opencl_B_float, opencl_B_double
#endif
            );

  std::cout << "  -> C: row, A: row, B: col" << std::endl;
  test_blas(my_backend,
            eps_float, eps_double,
            C_float, C_double, A_float, A_double, B_float, B_double,
            ViennaCLRowMajor, ViennaCLRowMajor, ViennaCLColumnMajor,
            host_C_float, host_C_double, host_A_float, host_A_double, host_B_float, host_B_double
#ifdef VIENNACL_WITH_CUDA
            , cuda_C_float, cuda_C_double, cuda_A_float, cuda_A_double, cuda_B_float, cuda_B_double
#endif
#ifdef VIENNACL_WITH_OPENCL
            , opencl_C_float, opencl_C_double, opencl_A_float, opencl_A_double, opencl_B_float, opencl_B_double
#endif
            );

  std::cout << "  -> C: row, A: col, B: row" << std::endl;
  test_blas(my_backend,
            eps_float, eps_double,
            C_float, C_double, A_float, A_double, B_float, B_double,
            ViennaCLRowMajor, ViennaCLColumnMajor, ViennaCLRowMajor,
            host_C_float, host_C_double, host_A_float, host_A_double, host_B_float, host_B_double
#ifdef VIENNACL_WITH_CUDA
            , cuda_C_float, cuda_C_double, cuda_A_float, cuda_A_double, cuda_B_float, cuda_B_double
#endif
#ifdef VIENNACL_WITH_OPENCL
            , opencl_C_float, opencl_C_double, opencl_A_float, opencl_A_double, opencl_B_float, opencl_B_double
#endif
            );

  std::cout << "  -> C: row, A: col, B: col" << std::endl;
  test_blas(my_backend,
            eps_float, eps_double,
            C_float, C_double, A_float, A_double, B_float, B_double,
            ViennaCLRowMajor, ViennaCLColumnMajor, ViennaCLColumnMajor,
            host_C_float, host_C_double, host_A_float, host_A_double, host_B_float, host_B_double
#ifdef VIENNACL_WITH_CUDA
            , cuda_C_float, cuda_C_double, cuda_A_float, cuda_A_double, cuda_B_float, cuda_B_double
#endif
#ifdef VIENNACL_WITH_OPENCL
            , opencl_C_float, opencl_C_double, opencl_A_float, opencl_A_double, opencl_B_float, opencl_B_double
#endif
            );


  std::cout << "  -> C: col, A: row, B: row" << std::endl;
  test_blas(my_backend,
            eps_float, eps_double,
            C_float, C_double, A_float, A_double, B_float, B_double,
            ViennaCLColumnMajor, ViennaCLRowMajor, ViennaCLRowMajor,
            host_C_float, host_C_double, host_A_float, host_A_double, host_B_float, host_B_double
#ifdef VIENNACL_WITH_CUDA
            , cuda_C_float, cuda_C_double, cuda_A_float, cuda_A_double, cuda_B_float, cuda_B_double
#endif
#ifdef VIENNACL_WITH_OPENCL
            , opencl_C_float, opencl_C_double, opencl_A_float, opencl_A_double, opencl_B_float, opencl_B_double
#endif
            );

  std::cout << "  -> C: col, A: row, B: col" << std::endl;
  test_blas(my_backend,
            eps_float, eps_double,
            C_float, C_double, A_float, A_double, B_float, B_double,
            ViennaCLColumnMajor, ViennaCLRowMajor, ViennaCLColumnMajor,
            host_C_float, host_C_double, host_A_float, host_A_double, host_B_float, host_B_double
#ifdef VIENNACL_WITH_CUDA
            , cuda_C_float, cuda_C_double, cuda_A_float, cuda_A_double, cuda_B_float, cuda_B_double
#endif
#ifdef VIENNACL_WITH_OPENCL
            , opencl_C_float, opencl_C_double, opencl_A_float, opencl_A_double, opencl_B_float, opencl_B_double
#endif
            );

  std::cout << "  -> C: col, A: col, B: row" << std::endl;
  test_blas(my_backend,
            eps_float, eps_double,
            C_float, C_double, A_float, A_double, B_float, B_double,
            ViennaCLColumnMajor, ViennaCLColumnMajor, ViennaCLRowMajor,
            host_C_float, host_C_double, host_A_float, host_A_double, host_B_float, host_B_double
#ifdef VIENNACL_WITH_CUDA
            , cuda_C_float, cuda_C_double, cuda_A_float, cuda_A_double, cuda_B_float, cuda_B_double
#endif
#ifdef VIENNACL_WITH_OPENCL
            , opencl_C_float, opencl_C_double, opencl_A_float, opencl_A_double, opencl_B_float, opencl_B_double
#endif
            );

  std::cout << "  -> C: col, A: col, B: col" << std::endl;
  test_blas(my_backend,
            eps_float, eps_double,
            C_float, C_double, A_float, A_double, B_float, B_double,
            ViennaCLColumnMajor, ViennaCLColumnMajor, ViennaCLColumnMajor,
            host_C_float, host_C_double, host_A_float, host_A_double, host_B_float, host_B_double
#ifdef VIENNACL_WITH_CUDA
            , cuda_C_float, cuda_C_double, cuda_A_float, cuda_A_double, cuda_B_float, cuda_B_double
#endif
#ifdef VIENNACL_WITH_OPENCL
            , opencl_C_float, opencl_C_double, opencl_A_float, opencl_A_double, opencl_B_float, opencl_B_double
#endif
            );

}




int main()
{
  viennacl::tools::uniform_random_numbers<float>  randomFloat;
  viennacl::tools::uniform_random_numbers<double> randomDouble;

  std::size_t size  = 500*500;
  float  eps_float  = 1e-5f;
  double eps_double = 1e-12;

  std::vector<float> C_float(size);
  std::vector<float> A_float(size);
  std::vector<float> B_float(size);

  std::vector<double> C_double(size);
  std::vector<double> A_double(size);
  std::vector<double> B_double(size);

  // fill with random data:

  for (std::size_t i = 0; i < size; ++i)
  {
    C_float[i] = 0.5f + 0.1f * randomFloat();
    A_float[i] = 0.5f + 0.1f * randomFloat();
    B_float[i] = 0.5f + 0.1f * randomFloat();

    C_double[i] = 0.5 + 0.2 * randomDouble();
    A_double[i] = 0.5 + 0.2 * randomDouble();
    B_double[i] = 0.5 + 0.2 * randomDouble();
  }


  // Host setup
  ViennaCLBackend my_backend;
  ViennaCLBackendCreate(&my_backend);

  viennacl::vector<float> host_C_float(size, viennacl::context(viennacl::MAIN_MEMORY));  viennacl::copy(C_float, host_C_float);
  viennacl::vector<float> host_A_float(size, viennacl::context(viennacl::MAIN_MEMORY));  viennacl::copy(A_float, host_A_float);
  viennacl::vector<float> host_B_float(size, viennacl::context(viennacl::MAIN_MEMORY));  viennacl::copy(B_float, host_B_float);

  viennacl::vector<double> host_C_double(size, viennacl::context(viennacl::MAIN_MEMORY));  viennacl::copy(C_double, host_C_double);
  viennacl::vector<double> host_A_double(size, viennacl::context(viennacl::MAIN_MEMORY));  viennacl::copy(A_double, host_A_double);
  viennacl::vector<double> host_B_double(size, viennacl::context(viennacl::MAIN_MEMORY));  viennacl::copy(B_double, host_B_double);

  // CUDA setup
#ifdef VIENNACL_WITH_CUDA
  viennacl::vector<float> cuda_C_float(size, viennacl::context(viennacl::CUDA_MEMORY));  viennacl::copy(C_float, cuda_C_float);
  viennacl::vector<float> cuda_A_float(size, viennacl::context(viennacl::CUDA_MEMORY));  viennacl::copy(A_float, cuda_A_float);
  viennacl::vector<float> cuda_B_float(size, viennacl::context(viennacl::CUDA_MEMORY));  viennacl::copy(B_float, cuda_B_float);

  viennacl::vector<double> cuda_C_double(size, viennacl::context(viennacl::CUDA_MEMORY));  viennacl::copy(C_double, cuda_C_double);
  viennacl::vector<double> cuda_A_double(size, viennacl::context(viennacl::CUDA_MEMORY));  viennacl::copy(A_double, cuda_A_double);
  viennacl::vector<double> cuda_B_double(size, viennacl::context(viennacl::CUDA_MEMORY));  viennacl::copy(B_double, cuda_B_double);
#endif

  // OpenCL setup
#ifdef VIENNACL_WITH_OPENCL
  ViennaCLInt context_id = 0;
  viennacl::vector<float> opencl_C_float(size, viennacl::context(viennacl::ocl::get_context(context_id)));  viennacl::copy(C_float, opencl_C_float);
  viennacl::vector<float> opencl_A_float(size, viennacl::context(viennacl::ocl::get_context(context_id)));  viennacl::copy(A_float, opencl_A_float);
  viennacl::vector<float> opencl_B_float(size, viennacl::context(viennacl::ocl::get_context(context_id)));  viennacl::copy(B_float, opencl_B_float);

  viennacl::vector<double> *opencl_C_double = NULL;
  viennacl::vector<double> *opencl_A_double = NULL;
  viennacl::vector<double> *opencl_B_double = NULL;

  if ( viennacl::ocl::current_device().double_support() )
  {
    opencl_C_double = new viennacl::vector<double>(size, viennacl::context(viennacl::ocl::get_context(context_id)));  viennacl::copy(C_double, *opencl_C_double);
    opencl_A_double = new viennacl::vector<double>(size, viennacl::context(viennacl::ocl::get_context(context_id)));  viennacl::copy(A_double, *opencl_A_double);
    opencl_B_double = new viennacl::vector<double>(size, viennacl::context(viennacl::ocl::get_context(context_id)));  viennacl::copy(B_double, *opencl_B_double);
  }

  ViennaCLBackendSetOpenCLContextID(my_backend, context_id);
#endif

  // consistency checks:
  check(C_float, host_C_float, eps_float);
  check(A_float, host_A_float, eps_float);
  check(B_float, host_B_float, eps_float);

  check(C_double, host_C_double, eps_double);
  check(A_double, host_A_double, eps_double);
  check(B_double, host_B_double, eps_double);

#ifdef VIENNACL_WITH_CUDA
  check(C_float, cuda_C_float, eps_float);
  check(A_float, cuda_A_float, eps_float);
  check(B_float, cuda_B_float, eps_float);

  check(C_double, cuda_C_double, eps_double);
  check(A_double, cuda_A_double, eps_double);
  check(B_double, cuda_B_double, eps_double);
#endif
#ifdef VIENNACL_WITH_OPENCL
  check(C_float, opencl_C_float, eps_float);
  check(A_float, opencl_A_float, eps_float);
  check(B_float, opencl_B_float, eps_float);

  if ( viennacl::ocl::current_device().double_support() )
  {
    check(C_double, *opencl_C_double, eps_double);
    check(A_double, *opencl_A_double, eps_double);
    check(B_double, *opencl_B_double, eps_double);
  }
#endif

  std::cout << std::endl;

  test_blas(my_backend,
            eps_float, eps_double,
            C_float, C_double,
            A_float, A_double,
            B_float, B_double,
            host_C_float, host_C_double,
            host_A_float, host_A_double,
            host_B_float, host_B_double
#ifdef VIENNACL_WITH_CUDA
            , cuda_C_float, cuda_C_double
            , cuda_A_float, cuda_A_double
            , cuda_B_float, cuda_B_double
#endif
#ifdef VIENNACL_WITH_OPENCL
            , opencl_C_float, opencl_C_double
            , opencl_A_float, opencl_A_double
            , opencl_B_float, opencl_B_double
#endif
            );


#ifdef VIENNACL_WITH_OPENCL
  //cleanup
  if ( viennacl::ocl::current_device().double_support() )
  {
    delete opencl_C_double;
    delete opencl_A_double;
    delete opencl_B_double;
  }

#endif

  ViennaCLBackendDestroy(&my_backend);

  //
  //  That's it.
  //
  std::cout << std::endl << "!!!! TEST COMPLETED SUCCESSFULLY !!!!" << std::endl;

  return EXIT_SUCCESS;
}

