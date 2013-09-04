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
#include "helper.hpp"

//include basic scalar and vector types of ViennaCL
#include "viennacl/scalar.hpp"
#include "viennacl/vector.hpp"
#include "viennacl/matrix.hpp"
#include "viennacl/linalg/direct_solve.hpp"
#include "viennacl/linalg/prod.hpp"

// GEMV

ViennaCLStatus ViennaCLgemv(ViennaCLHostScalar alpha, ViennaCLMatrix A, ViennaCLVector x, ViennaCLHostScalar beta, ViennaCLVector y)
{
  viennacl::backend::mem_handle v1_handle;
  viennacl::backend::mem_handle v2_handle;
  viennacl::backend::mem_handle A_handle;

  if (init_vector(v1_handle, x) != ViennaCLSuccess)
    return ViennaCLGenericFailure;

  if (init_vector(v2_handle, y) != ViennaCLSuccess)
    return ViennaCLGenericFailure;

  if (init_matrix(A_handle, A) != ViennaCLSuccess)
    return ViennaCLGenericFailure;

  switch (x->precision)
  {
    case ViennaCLFloat:
    {
      viennacl::vector_base<float> v1(v1_handle, x->size, x->offset, x->inc);
      viennacl::vector_base<float> v2(v2_handle, y->size, y->offset, y->inc);

      if (A->order == ViennaCLRowMajor)
      {
        viennacl::matrix_base<float> mat(A_handle,
                                         A->size1, A->start1, A->stride1, A->internal_size1,
                                         A->size2, A->start2, A->stride2, A->internal_size2);
        v2 *= beta->value_float;
        if (A->trans == ViennaCLTrans)
          v2 += alpha->value_float * viennacl::linalg::prod(viennacl::trans(mat), v1);
        else
          v2 += alpha->value_float * viennacl::linalg::prod(mat, v1);
      }
      else
      {
        viennacl::matrix_base<float, viennacl::column_major> mat(A_handle,
                                                                 A->size1, A->start1, A->stride1, A->internal_size1,
                                                                 A->size2, A->start2, A->stride2, A->internal_size2);
        v2 *= beta->value_float;
        if (A->trans == ViennaCLTrans)
          v2 += alpha->value_float * viennacl::linalg::prod(viennacl::trans(mat), v1);
        else
          v2 += alpha->value_float * viennacl::linalg::prod(mat, v1);
      }

      return ViennaCLSuccess;
    }

    case ViennaCLDouble:
    {
      viennacl::vector_base<double> v1(v1_handle, x->size, x->offset, x->inc);
      viennacl::vector_base<double> v2(v2_handle, y->size, y->offset, y->inc);

      if (A->order == ViennaCLRowMajor)
      {
        viennacl::matrix_base<double> mat(A_handle,
                                          A->size1, A->start1, A->stride1, A->internal_size1,
                                          A->size2, A->start2, A->stride2, A->internal_size2);
        v2 *= beta->value_double;
        if (A->trans == ViennaCLTrans)
          v2 += alpha->value_double * viennacl::linalg::prod(viennacl::trans(mat), v1);
        else
          v2 += alpha->value_double * viennacl::linalg::prod(mat, v1);
      }
      else
      {
        viennacl::matrix_base<double, viennacl::column_major> mat(A_handle,
                                                                  A->size1, A->start1, A->stride1, A->internal_size1,
                                                                  A->size2, A->start2, A->stride2, A->internal_size2);
        v2 *= beta->value_double;
        if (A->trans == ViennaCLTrans)
          v2 += alpha->value_double * viennacl::linalg::prod(viennacl::trans(mat), v1);
        else
          v2 += alpha->value_double * viennacl::linalg::prod(mat, v1);
      }

      return ViennaCLSuccess;
    }

    default:
      return ViennaCLGenericFailure;
  }
}


// xTRSV

ViennaCLStatus ViennaCLtrsv(ViennaCLMatrix A, ViennaCLVector x, ViennaCLUplo uplo)
{
  viennacl::backend::mem_handle v1_handle;
  viennacl::backend::mem_handle A_handle;

  if (init_vector(v1_handle, x) != ViennaCLSuccess)
    return ViennaCLGenericFailure;

  if (init_matrix(A_handle, A) != ViennaCLSuccess)
    return ViennaCLGenericFailure;

  switch (x->precision)
  {
    case ViennaCLFloat:
    {
      viennacl::vector_base<float> v1(v1_handle, x->size, x->offset, x->inc);

      if (A->order == ViennaCLRowMajor)
      {
        viennacl::matrix_base<float> mat(A_handle,
                                         A->size1, A->start1, A->stride1, A->internal_size1,
                                         A->size2, A->start2, A->stride2, A->internal_size2);
        if (A->trans == ViennaCLTrans)
        {
          if (uplo == ViennaCLUpper)
            viennacl::linalg::inplace_solve(viennacl::trans(mat), v1, viennacl::linalg::upper_tag());
          else
            viennacl::linalg::inplace_solve(viennacl::trans(mat), v1, viennacl::linalg::lower_tag());
        }
        else
        {
          if (uplo == ViennaCLUpper)
            viennacl::linalg::inplace_solve(mat, v1, viennacl::linalg::upper_tag());
          else
            viennacl::linalg::inplace_solve(mat, v1, viennacl::linalg::lower_tag());
        }
      }
      else
      {
        viennacl::matrix_base<float, viennacl::column_major> mat(A_handle,
                                                                 A->size1, A->start1, A->stride1, A->internal_size1,
                                                                 A->size2, A->start2, A->stride2, A->internal_size2);
        if (A->trans == ViennaCLTrans)
        {
          if (uplo == ViennaCLUpper)
            viennacl::linalg::inplace_solve(viennacl::trans(mat), v1, viennacl::linalg::upper_tag());
          else
            viennacl::linalg::inplace_solve(viennacl::trans(mat), v1, viennacl::linalg::lower_tag());
        }
        else
        {
          if (uplo == ViennaCLUpper)
            viennacl::linalg::inplace_solve(mat, v1, viennacl::linalg::upper_tag());
          else
            viennacl::linalg::inplace_solve(mat, v1, viennacl::linalg::lower_tag());
        }
      }

      return ViennaCLSuccess;
    }
    case ViennaCLDouble:
    {
      viennacl::vector_base<double> v1(v1_handle, x->size, x->offset, x->inc);

      if (A->order == ViennaCLRowMajor)
      {
        viennacl::matrix_base<double> mat(A_handle,
                                          A->size1, A->start1, A->stride1, A->internal_size1,
                                          A->size2, A->start2, A->stride2, A->internal_size2);
        if (A->trans == ViennaCLTrans)
        {
          if (uplo == ViennaCLUpper)
            viennacl::linalg::inplace_solve(viennacl::trans(mat), v1, viennacl::linalg::upper_tag());
          else
            viennacl::linalg::inplace_solve(viennacl::trans(mat), v1, viennacl::linalg::lower_tag());
        }
        else
        {
          if (uplo == ViennaCLUpper)
            viennacl::linalg::inplace_solve(mat, v1, viennacl::linalg::upper_tag());
          else
            viennacl::linalg::inplace_solve(mat, v1, viennacl::linalg::lower_tag());
        }
      }
      else
      {
        viennacl::matrix_base<double, viennacl::column_major> mat(A_handle,
                                                                  A->size1, A->start1, A->stride1, A->internal_size1,
                                                                  A->size2, A->start2, A->stride2, A->internal_size2);
        if (A->trans == ViennaCLTrans)
        {
          if (uplo == ViennaCLUpper)
            viennacl::linalg::inplace_solve(viennacl::trans(mat), v1, viennacl::linalg::upper_tag());
          else
            viennacl::linalg::inplace_solve(viennacl::trans(mat), v1, viennacl::linalg::lower_tag());
        }
        else
        {
          if (uplo == ViennaCLUpper)
            viennacl::linalg::inplace_solve(mat, v1, viennacl::linalg::upper_tag());
          else
            viennacl::linalg::inplace_solve(mat, v1, viennacl::linalg::lower_tag());
        }
      }

      return ViennaCLSuccess;
    }

    default:
      return  ViennaCLGenericFailure;
  }
}


// xGER

ViennaCLStatus ViennaCLger(ViennaCLHostScalar alpha, ViennaCLVector x, ViennaCLVector y, ViennaCLMatrix A)
{
  viennacl::backend::mem_handle v1_handle;
  viennacl::backend::mem_handle v2_handle;
  viennacl::backend::mem_handle A_handle;

  if (init_vector(v1_handle, x) != ViennaCLSuccess)
    return ViennaCLGenericFailure;

  if (init_vector(v2_handle, y) != ViennaCLSuccess)
    return ViennaCLGenericFailure;

  if (init_matrix(A_handle, A) != ViennaCLSuccess)
    return ViennaCLGenericFailure;

  switch (x->precision)
  {
    case ViennaCLFloat:
    {
      viennacl::vector_base<float> v1(v1_handle, x->size, x->offset, x->inc);
      viennacl::vector_base<float> v2(v2_handle, y->size, y->offset, y->inc);

      if (A->order == ViennaCLRowMajor)
      {
        viennacl::matrix_base<float> mat(A_handle,
                                         A->size1, A->start1, A->stride1, A->internal_size1,
                                         A->size2, A->start2, A->stride2, A->internal_size2);

        mat += alpha->value_float * viennacl::linalg::outer_prod(v1, v2);
      }
      else
      {
        viennacl::matrix_base<float, viennacl::column_major> mat(A_handle,
                                                                 A->size1, A->start1, A->stride1, A->internal_size1,
                                                                 A->size2, A->start2, A->stride2, A->internal_size2);

        mat += alpha->value_float * viennacl::linalg::outer_prod(v1, v2);
      }

      return ViennaCLSuccess;
    }
    case ViennaCLDouble:
    {
      viennacl::vector_base<double> v1(v1_handle, x->size, x->offset, x->inc);
      viennacl::vector_base<double> v2(v2_handle, y->size, y->offset, y->inc);

      if (A->order == ViennaCLRowMajor)
      {
        viennacl::matrix_base<double> mat(A_handle,
                                          A->size1, A->start1, A->stride1, A->internal_size1,
                                          A->size2, A->start2, A->stride2, A->internal_size2);

        mat += alpha->value_double * viennacl::linalg::outer_prod(v1, v2);
      }
      else
      {
        viennacl::matrix_base<double, viennacl::column_major> mat(A_handle,
                                                                  A->size1, A->start1, A->stride1, A->internal_size1,
                                                                  A->size2, A->start2, A->stride2, A->internal_size2);

        mat += alpha->value_double * viennacl::linalg::outer_prod(v1, v2);
      }

      return ViennaCLSuccess;
    }
    default:
      return  ViennaCLGenericFailure;
  }
}


