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


#ifdef VIENNACL_WITH_CUDA


// IxAMAX

ViennaCLStatus ViennaCLCUDAiSamax(ViennaCLCUDABackend backend, int n,
                                  size_t *index,
                                  float *x, int offx, int incx)
{
  viennacl::vector_base<float> v1(x, n, viennacl::CUDA_MEMORY, offx, incx);

  *index = viennacl::linalg::index_norm_inf(v1);
  return ViennaCLSuccess;
}

ViennaCLStatus ViennaCLCUDAiDamax(ViennaCLCUDABackend backend, int n,
                                  size_t *index,
                                  double *x, int offx, int incx)
{
  viennacl::vector_base<double> v1(x, n, viennacl::CUDA_MEMORY, offx, incx);

  *index = viennacl::linalg::index_norm_inf(v1);
  return ViennaCLSuccess;
}



// xASUM

ViennaCLStatus ViennaCLCUDASasum(ViennaCLCUDABackend backend, int n,
                                 float *alpha,
                                 float *x, int offx, int incx)
{
  viennacl::vector_base<float> v1(x, n, viennacl::CUDA_MEMORY, offx, incx);

  *alpha = viennacl::linalg::norm_1(v1);
  return ViennaCLSuccess;
}

ViennaCLStatus ViennaCLCUDADasum(ViennaCLCUDABackend backend, int n,
                                 double *alpha,
                                 double *x, int offx, int incx)
{
  viennacl::vector_base<double> v1(x, n, viennacl::CUDA_MEMORY, offx, incx);

  *alpha = viennacl::linalg::norm_1(v1);
  return ViennaCLSuccess;
}


// xAXPY

ViennaCLStatus ViennaCLCUDASaxpy(ViennaCLCUDABackend backend, int n,
                                 float alpha,
                                 float *x, int offx, int incx,
                                 float *y, int offy, int incy)
{
  viennacl::vector_base<float> v1(x, n, viennacl::CUDA_MEMORY, offx, incx);
  viennacl::vector_base<float> v2(y, n, viennacl::CUDA_MEMORY, offy, incy);

  v2 += alpha * v1;
  return ViennaCLSuccess;
}

ViennaCLStatus ViennaCLCUDADaxpy(ViennaCLCUDABackend backend, int n,
                                 double alpha,
                                 double *x, int offx, int incx,
                                 double *y, int offy, int incy)
{
  viennacl::vector_base<double> v1(x, n, viennacl::CUDA_MEMORY, offx, incx);
  viennacl::vector_base<double> v2(y, n, viennacl::CUDA_MEMORY, offy, incy);

  v2 += alpha * v1;
  return ViennaCLSuccess;
}


// xCOPY

ViennaCLStatus ViennaCLCUDAScopy(ViennaCLCUDABackend backend, int n,
                                 float *x, int offx, int incx,
                                 float *y, int offy, int incy)
{
  viennacl::vector_base<float> v1(x, n, viennacl::CUDA_MEMORY, offx, incx);
  viennacl::vector_base<float> v2(y, n, viennacl::CUDA_MEMORY, offy, incy);

  v2 = v1;
  return ViennaCLSuccess;
}

ViennaCLStatus ViennaCLCUDADcopy(ViennaCLCUDABackend backend, int n,
                                 double *x, int offx, int incx,
                                 double *y, int offy, int incy)
{
  viennacl::vector_base<double> v1(x, n, viennacl::CUDA_MEMORY, offx, incx);
  viennacl::vector_base<double> v2(y, n, viennacl::CUDA_MEMORY, offy, incy);

  v2 = v1;
  return ViennaCLSuccess;
}

// xDOT

ViennaCLStatus ViennaCLCUDASdot(ViennaCLCUDABackend backend, int n,
                                float *alpha,
                                float *x, int offx, int incx,
                                float *y, int offy, int incy)
{
  viennacl::vector_base<float> v1(x, n, viennacl::CUDA_MEMORY, offx, incx);
  viennacl::vector_base<float> v2(y, n, viennacl::CUDA_MEMORY, offy, incy);

  *alpha = viennacl::linalg::inner_prod(v1, v2);
  return ViennaCLSuccess;
}

ViennaCLStatus ViennaCLCUDADdot(ViennaCLCUDABackend backend, int n,
                                double *alpha,
                                double *x, int offx, int incx,
                                double *y, int offy, int incy)
{
  viennacl::vector_base<double> v1(x, n, viennacl::CUDA_MEMORY, offx, incx);
  viennacl::vector_base<double> v2(y, n, viennacl::CUDA_MEMORY, offy, incy);

  *alpha = viennacl::linalg::inner_prod(v1, v2);
  return ViennaCLSuccess;
}

// xNRM2

ViennaCLStatus ViennaCLCUDASnrm2(ViennaCLCUDABackend backend, int n,
                                 float *alpha,
                                 float *x, int offx, int incx)
{
  viennacl::vector_base<float> v1(x, n, viennacl::CUDA_MEMORY, offx, incx);

  *alpha = viennacl::linalg::norm_2(v1);
  return ViennaCLSuccess;
}

ViennaCLStatus ViennaCLCUDADnrm2(ViennaCLCUDABackend backend, int n,
                                 double *alpha,
                                 double *x, int offx, int incx)
{
  viennacl::vector_base<double> v1(x, n, viennacl::CUDA_MEMORY, offx, incx);

  *alpha = viennacl::linalg::norm_2(v1);
  return ViennaCLSuccess;
}



// xROT

ViennaCLStatus ViennaCLCUDASrot(ViennaCLCUDABackend backend, int n,
                                float *x, int offx, int incx,
                                float *y, int offy, int incy,
                                float c, float s)
{
  viennacl::vector_base<float> v1(x, n, viennacl::CUDA_MEMORY, offx, incx);
  viennacl::vector_base<float> v2(y, n, viennacl::CUDA_MEMORY, offy, incy);

  viennacl::linalg::plane_rotation(v1, v2, c, s);
  return ViennaCLSuccess;
}

ViennaCLStatus ViennaCLCUDADrot(ViennaCLCUDABackend backend, int n,
                                double *x, int offx, int incx,
                                double *y, int offy, int incy,
                                double c, double s)
{
  viennacl::vector_base<double> v1(x, n, viennacl::CUDA_MEMORY, offx, incx);
  viennacl::vector_base<double> v2(y, n, viennacl::CUDA_MEMORY, offy, incy);

  viennacl::linalg::plane_rotation(v1, v2, c, s);
  return ViennaCLSuccess;
}



// xSCAL

ViennaCLStatus ViennaCLCUDASscal(ViennaCLCUDABackend backend, int n,
                                 float alpha,
                                 float *x, int offx, int incx)
{
  viennacl::vector_base<float> v1(x, n, viennacl::CUDA_MEMORY, offx, incx);

  v1 *= alpha;
  return ViennaCLSuccess;
}

ViennaCLStatus ViennaCLCUDADscal(ViennaCLCUDABackend backend, int n,
                                 double alpha,
                                 double *x, int offx, int incx)
{
  viennacl::vector_base<double> v1(x, n, viennacl::CUDA_MEMORY, offx, incx);

  v1 *= alpha;
  return ViennaCLSuccess;
}


// xSWAP

ViennaCLStatus ViennaCLCUDASswap(ViennaCLCUDABackend backend, int n,
                                 float *x, int offx, int incx,
                                 float *y, int offy, int incy)
{
  viennacl::vector_base<float> v1(x, n, viennacl::CUDA_MEMORY, offx, incx);
  viennacl::vector_base<float> v2(y, n, viennacl::CUDA_MEMORY, offy, incy);

  viennacl::swap(v1, v2);
  return ViennaCLSuccess;
}

ViennaCLStatus ViennaCLCUDADswap(ViennaCLCUDABackend backend, int n,
                                 double *x, int offx, int incx,
                                 double *y, int offy, int incy)
{
  viennacl::vector_base<double> v1(x, n, viennacl::CUDA_MEMORY, offx, incx);
  viennacl::vector_base<double> v2(y, n, viennacl::CUDA_MEMORY, offy, incy);

  viennacl::swap(v1, v2);
  return ViennaCLSuccess;
}
#endif


