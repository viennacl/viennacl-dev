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
//#include "helper.hpp"

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

ViennaCLStatus ViennaCLHostiSamax(ViennaCLHostBackend /*backend*/, size_t n,
                                 size_t *index,
                                 float *x, size_t offx, int incx)
{
  viennacl::vector_base<float> v1(x, viennacl::MAIN_MEMORY, n, offx, incx);

  *index = viennacl::linalg::index_norm_inf(v1);
  return ViennaCLSuccess;
}

ViennaCLStatus ViennaCLHostiDamax(ViennaCLHostBackend /*backend*/, size_t n,
                                 size_t *index,
                                 double *x, size_t offx, int incx)
{
  viennacl::vector_base<double> v1(x, viennacl::MAIN_MEMORY, n, offx, incx);

  *index = viennacl::linalg::index_norm_inf(v1);
  return ViennaCLSuccess;
}



// xASUM

ViennaCLStatus ViennaCLHostSasum(ViennaCLHostBackend /*backend*/, size_t n,
                                 float *alpha,
                                 float *x, size_t offx, int incx)
{
  viennacl::vector_base<float> v1(x, viennacl::MAIN_MEMORY, n, offx, incx);

  *alpha = viennacl::linalg::norm_1(v1);
  return ViennaCLSuccess;
}

ViennaCLStatus ViennaCLHostDasum(ViennaCLHostBackend /*backend*/, size_t n,
                                 double *alpha,
                                 double *x, size_t offx, int incx)
{
  viennacl::vector_base<double> v1(x, viennacl::MAIN_MEMORY, n, offx, incx);

  *alpha = viennacl::linalg::norm_1(v1);
  return ViennaCLSuccess;
}



// xAXPY

ViennaCLStatus ViennaCLHostSaxpy(ViennaCLHostBackend /*backend*/, size_t n,
                                 float alpha,
                                 float *x, size_t offx, int incx,
                                 float *y, size_t offy, int incy)
{
  viennacl::vector_base<float> v1(x, viennacl::MAIN_MEMORY, n, offx, incx);
  viennacl::vector_base<float> v2(y, viennacl::MAIN_MEMORY, n, offy, incy);

  v2 += alpha * v1;
  return ViennaCLSuccess;
}

ViennaCLStatus ViennaCLHostDaxpy(ViennaCLHostBackend /*backend*/, size_t n,
                                 double alpha,
                                 double *x, size_t offx, int incx,
                                 double *y, size_t offy, int incy)
{
  viennacl::vector_base<double> v1(x, viennacl::MAIN_MEMORY, n, offx, incx);
  viennacl::vector_base<double> v2(y, viennacl::MAIN_MEMORY, n, offy, incy);

  v2 += alpha * v1;
  return ViennaCLSuccess;
}


// xCOPY

ViennaCLStatus ViennaCLHostScopy(ViennaCLHostBackend /*backend*/, size_t n,
                                 float *x, size_t offx, int incx,
                                 float *y, size_t offy, int incy)
{
  viennacl::vector_base<float> v1(x, viennacl::MAIN_MEMORY, n, offx, incx);
  viennacl::vector_base<float> v2(y, viennacl::MAIN_MEMORY, n, offy, incy);

  v2 = v1;
  return ViennaCLSuccess;
}

ViennaCLStatus ViennaCLHostDcopy(ViennaCLHostBackend /*backend*/, size_t n,
                                 double *x, size_t offx, int incx,
                                 double *y, size_t offy, int incy)
{
  viennacl::vector_base<double> v1(x, viennacl::MAIN_MEMORY, n, offx, incx);
  viennacl::vector_base<double> v2(y, viennacl::MAIN_MEMORY, n, offy, incy);

  v2 = v1;
  return ViennaCLSuccess;
}

// xAXPY

ViennaCLStatus ViennaCLHostSdot(ViennaCLHostBackend /*backend*/, size_t n,
                                float *alpha,
                                float *x, size_t offx, int incx,
                                float *y, size_t offy, int incy)
{
  viennacl::vector_base<float> v1(x, viennacl::MAIN_MEMORY, n, offx, incx);
  viennacl::vector_base<float> v2(y, viennacl::MAIN_MEMORY, n, offy, incy);

  *alpha = viennacl::linalg::inner_prod(v1, v2);
  return ViennaCLSuccess;
}

ViennaCLStatus ViennaCLHostDdot(ViennaCLHostBackend /*backend*/, size_t n,
                                double *alpha,
                                double *x, size_t offx, int incx,
                                double *y, size_t offy, int incy)
{
  viennacl::vector_base<double> v1(x, viennacl::MAIN_MEMORY, n, offx, incx);
  viennacl::vector_base<double> v2(y, viennacl::MAIN_MEMORY, n, offy, incy);

  *alpha = viennacl::linalg::inner_prod(v1, v2);
  return ViennaCLSuccess;
}

// xNRM2

ViennaCLStatus ViennaCLHostSnrm2(ViennaCLHostBackend /*backend*/, size_t n,
                                 float *alpha,
                                 float *x, size_t offx, int incx)
{
  viennacl::vector_base<float> v1(x, viennacl::MAIN_MEMORY, n, offx, incx);

  *alpha = viennacl::linalg::norm_2(v1);
  return ViennaCLSuccess;
}

ViennaCLStatus ViennaCLHostDnrm2(ViennaCLHostBackend /*backend*/, size_t n,
                                 double *alpha,
                                 double *x, size_t offx, int incx)
{
  viennacl::vector_base<double> v1(x, viennacl::MAIN_MEMORY, n, offx, incx);

  *alpha = viennacl::linalg::norm_2(v1);
  return ViennaCLSuccess;
}


// xROT

ViennaCLStatus ViennaCLHostSrot(ViennaCLHostBackend /*backend*/, size_t n,
                                float *x, size_t offx, int incx,
                                float *y, size_t offy, int incy,
                                float c, float s)
{
  viennacl::vector_base<float> v1(x, viennacl::MAIN_MEMORY, n, offx, incx);
  viennacl::vector_base<float> v2(y, viennacl::MAIN_MEMORY, n, offy, incy);

  viennacl::linalg::plane_rotation(v1, v2, c, s);
  return ViennaCLSuccess;
}

ViennaCLStatus ViennaCLHostDrot(ViennaCLHostBackend /*backend*/, size_t n,
                                double *x, size_t offx, int incx,
                                double *y, size_t offy, int incy,
                                double c, double s)
{
  viennacl::vector_base<double> v1(x, viennacl::MAIN_MEMORY, n, offx, incx);
  viennacl::vector_base<double> v2(y, viennacl::MAIN_MEMORY, n, offy, incy);

  viennacl::linalg::plane_rotation(v1, v2, c, s);
  return ViennaCLSuccess;
}



// xSCAL

ViennaCLStatus ViennaCLHostSscal(ViennaCLHostBackend /*backend*/, size_t n,
                                 float alpha,
                                 float *x, size_t offx, int incx)
{
  viennacl::vector_base<float> v1(x, viennacl::MAIN_MEMORY, n, offx, incx);

  v1 *= alpha;
  return ViennaCLSuccess;
}

ViennaCLStatus ViennaCLHostDscal(ViennaCLHostBackend /*backend*/, size_t n,
                                 double alpha,
                                 double *x, size_t offx, int incx)
{
  viennacl::vector_base<double> v1(x, viennacl::MAIN_MEMORY, n, offx, incx);

  v1 *= alpha;
  return ViennaCLSuccess;
}

// xSWAP

ViennaCLStatus ViennaCLHostSswap(ViennaCLHostBackend /*backend*/, size_t n,
                                 float *x, size_t offx, int incx,
                                 float *y, size_t offy, int incy)
{
  viennacl::vector_base<float> v1(x, viennacl::MAIN_MEMORY, n, offx, incx);
  viennacl::vector_base<float> v2(y, viennacl::MAIN_MEMORY, n, offy, incy);

  viennacl::swap(v1, v2);
  return ViennaCLSuccess;
}

ViennaCLStatus ViennaCLHostDswap(ViennaCLHostBackend /*backend*/, size_t n,
                                 double *x, size_t offx, int incx,
                                 double *y, size_t offy, int incy)
{
  viennacl::vector_base<double> v1(x, viennacl::MAIN_MEMORY, n, offx, incx);
  viennacl::vector_base<double> v2(y, viennacl::MAIN_MEMORY, n, offy, incy);

  viennacl::swap(v1, v2);
  return ViennaCLSuccess;
}
