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
#include "init_vector.hpp"

//include basic scalar and vector types of ViennaCL
#include "viennacl/scalar.hpp"
#include "viennacl/vector.hpp"

//include the generic inner product functions of ViennaCL
#include "viennacl/linalg/inner_prod.hpp"

//include the generic norm functions of ViennaCL
#include "viennacl/linalg/norm_1.hpp"
#include "viennacl/linalg/norm_2.hpp"
#include "viennacl/linalg/norm_inf.hpp"

// IxAMAX

ViennaCLStatus ViennaCLiamax(ViennaCLInt *index, ViennaCLVector x)
{
  viennacl::backend::mem_handle v1_handle;

  if (init_vector(v1_handle, x) != ViennaCLSuccess)
    return ViennaCLGenericFailure;

  switch (x->precision)
  {
    case ViennaCLFloat:
    {
      viennacl::vector_base<float> v1(v1_handle, x->size, x->offset, x->inc);

      *index = viennacl::linalg::index_norm_inf(v1);
      return ViennaCLSuccess;
    }

    case ViennaCLDouble:
    {
      viennacl::vector_base<double> v1(v1_handle, x->size, x->offset, x->inc);

      *index = viennacl::linalg::index_norm_inf(v1);
      return ViennaCLSuccess;
    }

    default:
      return ViennaCLGenericFailure;
  }
}




// xASUM

ViennaCLStatus ViennaCLasum(ViennaCLHostScalar *alpha, ViennaCLVector x)
{
  if ((*alpha)->precision != x->precision)
    return ViennaCLGenericFailure;

  viennacl::backend::mem_handle v1_handle;

  if (init_vector(v1_handle, x) != ViennaCLSuccess)
    return ViennaCLGenericFailure;

  switch (x->precision)
  {
    case ViennaCLFloat:
    {
      viennacl::vector_base<float> v1(v1_handle, x->size, x->offset, x->inc);

      (*alpha)->value_float = viennacl::linalg::norm_1(v1);
      return ViennaCLSuccess;
    }

    case ViennaCLDouble:
    {
      viennacl::vector_base<double> v1(v1_handle, x->size, x->offset, x->inc);

      (*alpha)->value_double = viennacl::linalg::norm_1(v1);
      return ViennaCLSuccess;
    }

    default:
      return ViennaCLGenericFailure;
  }
}



// xAXPY

ViennaCLStatus ViennaCLaxpy(ViennaCLHostScalar alpha, ViennaCLVector x, ViennaCLVector y)
{
  if (alpha->precision != x->precision)
    return ViennaCLGenericFailure;

  if (x->precision != y->precision)
    return ViennaCLGenericFailure;

  viennacl::backend::mem_handle v1_handle;
  viennacl::backend::mem_handle v2_handle;

  if (init_vector(v1_handle, x) != ViennaCLSuccess)
    return ViennaCLGenericFailure;

  if (init_vector(v2_handle, y) != ViennaCLSuccess)
    return ViennaCLGenericFailure;

  switch (x->precision)
  {
    case ViennaCLFloat:
    {
      viennacl::vector_base<float> v1(v1_handle, x->size, x->offset, x->inc);
      viennacl::vector_base<float> v2(v2_handle, y->size, y->offset, y->inc);

      v2 += alpha->value_float * v1;
      return ViennaCLSuccess;
    }

    case ViennaCLDouble:
    {
      viennacl::vector_base<double> v1(v1_handle, x->size, x->offset, x->inc);
      viennacl::vector_base<double> v2(v2_handle, y->size, y->offset, y->inc);

      v2 += alpha->value_double * v1;
      return ViennaCLSuccess;
    }

    default:
      return ViennaCLGenericFailure;
  }
}


// xCOPY

ViennaCLStatus ViennaCLcopy(ViennaCLVector x, ViennaCLVector y)
{
  if (x->precision != y->precision)
    return ViennaCLGenericFailure;

  viennacl::backend::mem_handle v1_handle;
  viennacl::backend::mem_handle v2_handle;

  if (init_vector(v1_handle, x) != ViennaCLSuccess)
    return ViennaCLGenericFailure;

  if (init_vector(v2_handle, y) != ViennaCLSuccess)
    return ViennaCLGenericFailure;

  switch (x->precision)
  {
    case ViennaCLFloat:
    {
      viennacl::vector_base<float> v1(v1_handle, x->size, x->offset, x->inc);
      viennacl::vector_base<float> v2(v2_handle, y->size, y->offset, y->inc);

      v2 = v1;
      return ViennaCLSuccess;
    }

    case ViennaCLDouble:
    {
      viennacl::vector_base<double> v1(v1_handle, x->size, x->offset, x->inc);
      viennacl::vector_base<double> v2(v2_handle, y->size, y->offset, y->inc);

      v2 = v1;
      return ViennaCLSuccess;
    }

    default:
      return ViennaCLGenericFailure;
  }
}

// xDOT

ViennaCLStatus ViennaCLdot(ViennaCLHostScalar *alpha, ViennaCLVector x, ViennaCLVector y)
{
  if ((*alpha)->precision != x->precision)
    return ViennaCLGenericFailure;

  if (x->precision != y->precision)
    return ViennaCLGenericFailure;

  viennacl::backend::mem_handle v1_handle;
  viennacl::backend::mem_handle v2_handle;

  if (init_vector(v1_handle, x) != ViennaCLSuccess)
    return ViennaCLGenericFailure;

  if (init_vector(v2_handle, y) != ViennaCLSuccess)
    return ViennaCLGenericFailure;

  switch (x->precision)
  {
    case ViennaCLFloat:
    {
      viennacl::vector_base<float> v1(v1_handle, x->size, x->offset, x->inc);
      viennacl::vector_base<float> v2(v2_handle, y->size, y->offset, y->inc);

      (*alpha)->value_float = viennacl::linalg::inner_prod(v1, v2);
      return ViennaCLSuccess;
    }

    case ViennaCLDouble:
    {
      viennacl::vector_base<double> v1(v1_handle, x->size, x->offset, x->inc);
      viennacl::vector_base<double> v2(v2_handle, y->size, y->offset, y->inc);

      (*alpha)->value_double = viennacl::linalg::inner_prod(v1, v2);
      return ViennaCLSuccess;
    }

    default:
      return ViennaCLGenericFailure;
  }
}

// xNRM2

ViennaCLStatus ViennaCLnrm2(ViennaCLHostScalar *alpha, ViennaCLVector x)
{
  if ((*alpha)->precision != x->precision)
    return ViennaCLGenericFailure;

  viennacl::backend::mem_handle v1_handle;

  if (init_vector(v1_handle, x) != ViennaCLSuccess)
    return ViennaCLGenericFailure;

  switch (x->precision)
  {
    case ViennaCLFloat:
    {
      viennacl::vector_base<float> v1(v1_handle, x->size, x->offset, x->inc);

      (*alpha)->value_float = viennacl::linalg::norm_2(v1);
      return ViennaCLSuccess;
    }

    case ViennaCLDouble:
    {
      viennacl::vector_base<double> v1(v1_handle, x->size, x->offset, x->inc);

      (*alpha)->value_double = viennacl::linalg::norm_2(v1);
      return ViennaCLSuccess;
    }

    default:
      return ViennaCLGenericFailure;
  }
}



// xROT

ViennaCLStatus ViennaCLrot(ViennaCLVector     x, ViennaCLVector     y,
                           ViennaCLHostScalar c, ViennaCLHostScalar s)
{
  if (c->precision != x->precision)
    return ViennaCLGenericFailure;

  if (s->precision != x->precision)
    return ViennaCLGenericFailure;

  if (x->precision != y->precision)
    return ViennaCLGenericFailure;

  viennacl::backend::mem_handle v1_handle;
  viennacl::backend::mem_handle v2_handle;

  if (init_vector(v1_handle, x) != ViennaCLSuccess)
    return ViennaCLGenericFailure;

  if (init_vector(v2_handle, y) != ViennaCLSuccess)
    return ViennaCLGenericFailure;

  switch (x->precision)
  {
    case ViennaCLFloat:
    {
      viennacl::vector_base<float> v1(v1_handle, x->size, x->offset, x->inc);
      viennacl::vector_base<float> v2(v2_handle, y->size, y->offset, y->inc);

      viennacl::linalg::plane_rotation(v1, v2, c->value_float, s->value_float);
      return ViennaCLSuccess;
    }

    case ViennaCLDouble:
    {
      viennacl::vector_base<double> v1(v1_handle, x->size, x->offset, x->inc);
      viennacl::vector_base<double> v2(v2_handle, y->size, y->offset, y->inc);

      viennacl::linalg::plane_rotation(v1, v2, c->value_double, s->value_double);
      return ViennaCLSuccess;
    }

    default:
      return ViennaCLGenericFailure;
  }
}

// xSCAL

ViennaCLStatus ViennaCLscal(ViennaCLHostScalar alpha, ViennaCLVector x)
{
  if (alpha->precision != x->precision)
    return ViennaCLGenericFailure;

  viennacl::backend::mem_handle v1_handle;

  if (init_vector(v1_handle, x) != ViennaCLSuccess)
    return ViennaCLGenericFailure;

  switch (x->precision)
  {
    case ViennaCLFloat:
    {
      viennacl::vector_base<float> v1(v1_handle, x->size, x->offset, x->inc);

      v1 *= alpha->value_float;
      return ViennaCLSuccess;
    }

    case ViennaCLDouble:
    {
      viennacl::vector_base<double> v1(v1_handle, x->size, x->offset, x->inc);

      v1 *= alpha->value_double;
      return ViennaCLSuccess;
    }

    default:
      return ViennaCLGenericFailure;
  }
}


// xSWAP

ViennaCLStatus ViennaCLswap(ViennaCLVector x, ViennaCLVector y)
{
  if (x->precision != y->precision)
    return ViennaCLGenericFailure;

  viennacl::backend::mem_handle v1_handle;
  viennacl::backend::mem_handle v2_handle;

  if (init_vector(v1_handle, x) != ViennaCLSuccess)
    return ViennaCLGenericFailure;

  if (init_vector(v2_handle, y) != ViennaCLSuccess)
    return ViennaCLGenericFailure;

  switch (x->precision)
  {
    case ViennaCLFloat:
    {
      viennacl::vector_base<float> v1(v1_handle, x->size, x->offset, x->inc);
      viennacl::vector_base<float> v2(v2_handle, y->size, y->offset, y->inc);

      viennacl::swap(v1, v2);
      return ViennaCLSuccess;
    }

    case ViennaCLDouble:
    {
      viennacl::vector_base<double> v1(v1_handle, x->size, x->offset, x->inc);
      viennacl::vector_base<double> v2(v2_handle, y->size, y->offset, y->inc);

      viennacl::swap(v1, v2);
      return ViennaCLSuccess;
    }

    default:
      return ViennaCLGenericFailure;
  }
}


