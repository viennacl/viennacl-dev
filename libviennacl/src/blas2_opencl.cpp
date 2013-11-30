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

//include basic scalar and vector types of ViennaCL
#include "viennacl/scalar.hpp"
#include "viennacl/vector.hpp"

#include "viennacl/vector.hpp"
#include "viennacl/matrix.hpp"
#include "viennacl/linalg/direct_solve.hpp"
#include "viennacl/linalg/prod.hpp"


// xGEMV

ViennaCLStatus ViennaCLOpenCLSgemv(ViennaCLOpenCLBackend backend,
                                 ViennaCLOrder order, ViennaCLTranspose transA,
                                 ViennaCLInt m, ViennaCLInt n, float alpha, cl_mem A, ViennaCLInt offA_row, ViennaCLInt offA_col, ViennaCLInt incA_row, ViennaCLInt incA_col, ViennaCLInt lda,
                                 cl_mem x, ViennaCLInt offx, ViennaCLInt incx,
                                 float beta,
                                 cl_mem y, ViennaCLInt offy, ViennaCLInt incy)
{
  if (order == ViennaCLRowMajor)
  {
    viennacl::vector_base<float> v1(x, n, offx, incx, viennacl::ocl::get_context(backend->context_id));
    viennacl::vector_base<float> v2(y, m, offy, incy, viennacl::ocl::get_context(backend->context_id));
    viennacl::matrix_base<float> mat(A, viennacl::ocl::get_context(backend->context_id),
                                     m, offA_row, incA_row, m,
                                     n, offA_col, incA_col, lda);
    v2 *= beta;
    if (transA == ViennaCLTrans)
      v2 += alpha * viennacl::linalg::prod(viennacl::trans(mat), v1);
    else
      v2 += alpha * viennacl::linalg::prod(mat, v1);
  }
  else
  {
    viennacl::vector_base<float> v1(x, n, offx, incx);
    viennacl::vector_base<float> v2(y, m, offy, incy);
    viennacl::matrix_base<float, viennacl::column_major> mat(A, viennacl::ocl::get_context(backend->context_id),
                                                             m, offA_row, incA_row, lda,
                                                             n, offA_col, incA_col, n);
    v2 *= beta;
    if (transA == ViennaCLTrans)
      v2 += alpha * viennacl::linalg::prod(viennacl::trans(mat), v1);
    else
      v2 += alpha * viennacl::linalg::prod(mat, v1);
  }

  return ViennaCLSuccess;
}

ViennaCLStatus ViennaCLOpenCLDgemv(ViennaCLOpenCLBackend backend,
                                 ViennaCLOrder order, ViennaCLTranspose transA,
                                 ViennaCLInt m, ViennaCLInt n, double alpha, cl_mem A, ViennaCLInt offA_row, ViennaCLInt offA_col, ViennaCLInt incA_row, ViennaCLInt incA_col, ViennaCLInt lda,
                                 cl_mem x, ViennaCLInt offx, ViennaCLInt incx,
                                 double beta,
                                 cl_mem y, ViennaCLInt offy, ViennaCLInt incy)
{
  if (order == ViennaCLRowMajor)
  {
    viennacl::vector_base<double> v1(x, n, offx, incx, viennacl::ocl::get_context(backend->context_id));
    viennacl::vector_base<double> v2(y, m, offy, incy, viennacl::ocl::get_context(backend->context_id));
    viennacl::matrix_base<double> mat(A, viennacl::ocl::get_context(backend->context_id),
                                      m, offA_row, incA_row, m,
                                      n, offA_col, incA_col, lda);
    v2 *= beta;
    if (transA == ViennaCLTrans)
      v2 += alpha * viennacl::linalg::prod(viennacl::trans(mat), v1);
    else
      v2 += alpha * viennacl::linalg::prod(mat, v1);
  }
  else
  {
    viennacl::vector_base<double> v1(x, n, offx, incx, viennacl::ocl::get_context(backend->context_id));
    viennacl::vector_base<double> v2(y, m, offy, incy, viennacl::ocl::get_context(backend->context_id));
    viennacl::matrix_base<double, viennacl::column_major> mat(A, viennacl::ocl::get_context(backend->context_id),
                                                              m, offA_row, incA_row, lda,
                                                              n, offA_col, incA_col, n);
    v2 *= beta;
    if (transA == ViennaCLTrans)
      v2 += alpha * viennacl::linalg::prod(viennacl::trans(mat), v1);
    else
      v2 += alpha * viennacl::linalg::prod(mat, v1);
  }

  return ViennaCLSuccess;
}



// xTRSV

ViennaCLStatus ViennaCLOpenCLStrsv(ViennaCLOpenCLBackend backend,
                                 ViennaCLUplo uplo, ViennaCLOrder order, ViennaCLTranspose transA,
                                 ViennaCLInt n, cl_mem A, ViennaCLInt offA_row, ViennaCLInt offA_col, ViennaCLInt incA_row, ViennaCLInt incA_col, ViennaCLInt lda,
                                 cl_mem x, ViennaCLInt offx, ViennaCLInt incx)
{
  if (order == ViennaCLRowMajor)
  {
    viennacl::vector_base<float> v(x, n, offx, incx, viennacl::ocl::get_context(backend->context_id));
    viennacl::matrix_base<float> mat(A, viennacl::ocl::get_context(backend->context_id),
                                     n, offA_row, incA_row, n,
                                     n, offA_col, incA_col, lda);
    if (transA == ViennaCLTrans)
    {
      if (uplo == ViennaCLUpper)
        viennacl::linalg::inplace_solve(viennacl::trans(mat), v, viennacl::linalg::upper_tag());
      else
        viennacl::linalg::inplace_solve(viennacl::trans(mat), v, viennacl::linalg::lower_tag());
    }
    else
    {
      if (uplo == ViennaCLUpper)
        viennacl::linalg::inplace_solve(mat, v, viennacl::linalg::upper_tag());
      else
        viennacl::linalg::inplace_solve(mat, v, viennacl::linalg::lower_tag());
    }
  }
  else
  {
    viennacl::vector_base<float> v(x, n, offx, incx, viennacl::ocl::get_context(backend->context_id));
    viennacl::matrix_base<float, viennacl::column_major> mat(A, viennacl::ocl::get_context(backend->context_id),
                                                             n, offA_row, incA_row, lda,
                                                             n, offA_col, incA_col, n);
    if (transA == ViennaCLTrans)
    {
      if (uplo == ViennaCLUpper)
        viennacl::linalg::inplace_solve(viennacl::trans(mat), v, viennacl::linalg::upper_tag());
      else
        viennacl::linalg::inplace_solve(viennacl::trans(mat), v, viennacl::linalg::lower_tag());
    }
    else
    {
      if (uplo == ViennaCLUpper)
        viennacl::linalg::inplace_solve(mat, v, viennacl::linalg::upper_tag());
      else
        viennacl::linalg::inplace_solve(mat, v, viennacl::linalg::lower_tag());
    }
  }

  return ViennaCLSuccess;
}

ViennaCLStatus ViennaCLOpenCLDtrsv(ViennaCLOpenCLBackend backend,
                                 ViennaCLUplo uplo, ViennaCLOrder order, ViennaCLTranspose transA,
                                 ViennaCLInt n, cl_mem A, ViennaCLInt offA_row, ViennaCLInt offA_col, ViennaCLInt incA_row, ViennaCLInt incA_col, ViennaCLInt lda,
                                 cl_mem x, ViennaCLInt offx, ViennaCLInt incx)
{
  if (order == ViennaCLRowMajor)
  {
    viennacl::vector_base<double> v(x, n, offx, incx, viennacl::ocl::get_context(backend->context_id));
    viennacl::matrix_base<double> mat(A, viennacl::ocl::get_context(backend->context_id),
                                      n, offA_row, incA_row, n,
                                      n, offA_col, incA_col, lda);
    if (transA == ViennaCLTrans)
    {
      if (uplo == ViennaCLUpper)
        viennacl::linalg::inplace_solve(viennacl::trans(mat), v, viennacl::linalg::upper_tag());
      else
        viennacl::linalg::inplace_solve(viennacl::trans(mat), v, viennacl::linalg::lower_tag());
    }
    else
    {
      if (uplo == ViennaCLUpper)
        viennacl::linalg::inplace_solve(mat, v, viennacl::linalg::upper_tag());
      else
        viennacl::linalg::inplace_solve(mat, v, viennacl::linalg::lower_tag());
    }
  }
  else
  {
    viennacl::vector_base<double> v(x, n, offx, incx, viennacl::ocl::get_context(backend->context_id));
    viennacl::matrix_base<double, viennacl::column_major> mat(A, viennacl::ocl::get_context(backend->context_id),
                                                              n, offA_row, incA_row, lda,
                                                              n, offA_col, incA_col, n);
    if (transA == ViennaCLTrans)
    {
      if (uplo == ViennaCLUpper)
        viennacl::linalg::inplace_solve(viennacl::trans(mat), v, viennacl::linalg::upper_tag());
      else
        viennacl::linalg::inplace_solve(viennacl::trans(mat), v, viennacl::linalg::lower_tag());
    }
    else
    {
      if (uplo == ViennaCLUpper)
        viennacl::linalg::inplace_solve(mat, v, viennacl::linalg::upper_tag());
      else
        viennacl::linalg::inplace_solve(mat, v, viennacl::linalg::lower_tag());
    }
  }

  return ViennaCLSuccess;
}



// xGER

ViennaCLStatus ViennaCLOpenCLSger(ViennaCLOpenCLBackend backend,
                                ViennaCLOrder order,
                                ViennaCLInt m, ViennaCLInt n,
                                float alpha,
                                cl_mem x, ViennaCLInt offx, ViennaCLInt incx,
                                cl_mem y, ViennaCLInt offy, ViennaCLInt incy,
                                cl_mem A, ViennaCLInt offA_row, ViennaCLInt offA_col, ViennaCLInt incA_row, ViennaCLInt incA_col, ViennaCLInt lda)
{
  if (order == ViennaCLRowMajor)
  {
    viennacl::vector_base<float> v1(x, n, offx, incx, viennacl::ocl::get_context(backend->context_id));
    viennacl::vector_base<float> v2(y, m, offy, incy, viennacl::ocl::get_context(backend->context_id));
    viennacl::matrix_base<float> mat(A, viennacl::ocl::get_context(backend->context_id),
                                     m, offA_row, incA_row, m,
                                     n, offA_col, incA_col, lda);

    mat += alpha * viennacl::linalg::outer_prod(v1, v2);
  }
  else
  {
    viennacl::vector_base<float> v1(x, n, offx, incx, viennacl::ocl::get_context(backend->context_id));
    viennacl::vector_base<float> v2(y, m, offy, incy, viennacl::ocl::get_context(backend->context_id));
    viennacl::matrix_base<float, viennacl::column_major> mat(A, viennacl::ocl::get_context(backend->context_id),
                                                             m, offA_row, incA_row, lda,
                                                             n, offA_col, incA_col, n);

    mat += alpha * viennacl::linalg::outer_prod(v1, v2);
  }

  return ViennaCLSuccess;
}

ViennaCLStatus ViennaCLOpenCLDger(ViennaCLOpenCLBackend backend,
                                ViennaCLOrder order,
                                ViennaCLInt m, ViennaCLInt n,
                                double alpha,
                                cl_mem x, ViennaCLInt offx, ViennaCLInt incx,
                                cl_mem y, ViennaCLInt offy, ViennaCLInt incy,
                                cl_mem A, ViennaCLInt offA_row, ViennaCLInt offA_col, ViennaCLInt incA_row, ViennaCLInt incA_col, ViennaCLInt lda)
{
  if (order == ViennaCLRowMajor)
  {
    viennacl::vector_base<double> v1(x, n, offx, incx, viennacl::ocl::get_context(backend->context_id));
    viennacl::vector_base<double> v2(y, m, offy, incy, viennacl::ocl::get_context(backend->context_id));
    viennacl::matrix_base<double> mat(A, viennacl::ocl::get_context(backend->context_id),
                                      m, offA_row, incA_row, m,
                                      n, offA_col, incA_col, lda);

    mat += alpha * viennacl::linalg::outer_prod(v1, v2);
  }
  else
  {
    viennacl::vector_base<double> v1(x, n, offx, incx, viennacl::ocl::get_context(backend->context_id));
    viennacl::vector_base<double> v2(y, m, offy, incy, viennacl::ocl::get_context(backend->context_id));
    viennacl::matrix_base<double, viennacl::column_major> mat(A, viennacl::ocl::get_context(backend->context_id),
                                                              m, offA_row, incA_row, lda,
                                                              n, offA_col, incA_col, n);

    mat += alpha * viennacl::linalg::outer_prod(v1, v2);
  }

  return ViennaCLSuccess;
}

