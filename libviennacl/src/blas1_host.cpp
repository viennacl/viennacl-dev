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

//include the generic inner product functions of ViennaCL
#include "viennacl/linalg/inner_prod.hpp"

//include the generic norm functions of ViennaCL
#include "viennacl/linalg/norm_1.hpp"
#include "viennacl/linalg/norm_2.hpp"
#include "viennacl/linalg/norm_inf.hpp"


ViennaCLStatus ViennaCLswap(ViennaCLVector x, ViennaCLVector y)
{
  if (x->precision != y->precision)
    return ViennaCLGenericFailure;

  switch (x->precision)
  {
    case ViennaCLFloat:
    {
      viennacl::backend::mem_handle v1_handle;
      viennacl::backend::mem_handle v2_handle;

      if (init(v1_handle, x) != ViennaCLSuccess)
        return ViennaCLGenericFailure;

      if (init(v2_handle, y) != ViennaCLSuccess)
        return ViennaCLGenericFailure;

      viennacl::vector_base<float> v1(v1_handle, x->size, x->offset, x->inc);
      viennacl::vector_base<float> v2(v2_handle, y->size, y->offset, y->inc);

      viennacl::swap(v1, v2);
      return ViennaCLSuccess;
    }

    case ViennaCLDouble:
    {
      viennacl::backend::mem_handle v1_handle; init(v1_handle, x);
      viennacl::backend::mem_handle v2_handle; init(v2_handle, y);

      viennacl::vector_base<double> v1(v1_handle, x->size, x->offset, x->inc);
      viennacl::vector_base<double> v2(v2_handle, y->size, y->offset, y->inc);

      viennacl::swap(v1, v2);
      return ViennaCLSuccess;
    }

    default:
      return ViennaCLGenericFailure;
  }
}


ViennaCLStatus ViennaCLHostSswap(ViennaCLHostBackend backend, size_t n,
                                 float *x, size_t offx, int incx,
                                 float *y, size_t offy, int incy)
{
  viennacl::vector_base<float> v1(x, n, viennacl::MAIN_MEMORY, offx, incx);
  viennacl::vector_base<float> v2(y, n, viennacl::MAIN_MEMORY, offy, incy);

  viennacl::swap(v1, v2);
  return ViennaCLSuccess;
}

ViennaCLStatus ViennaCLHostDswap(ViennaCLHostBackend backend, size_t n,
                                 double *x, size_t offx, int incx,
                                 double *y, size_t offy, int incy)
{
  viennacl::vector_base<double> v1(x, n, viennacl::MAIN_MEMORY, offx, incx);
  viennacl::vector_base<double> v2(y, n, viennacl::MAIN_MEMORY, offy, incy);

  viennacl::swap(v1, v2);
  return ViennaCLSuccess;
}

