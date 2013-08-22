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

#ifdef VIENNACL_WITH_OPENCL

// IxAMAX

ViennaCLStatus ViennaCLOpenCLiSamax(ViennaCLOpenCLBackend backend, size_t n,
                                    size_t *index,
                                    cl_mem x, size_t offx, int incx)
{
  viennacl::vector_base<float> v1(x, n, offx, incx, viennacl::ocl::get_context(backend->context_id));

  *index = viennacl::linalg::index_norm_inf(v1);
  return ViennaCLSuccess;
}

ViennaCLStatus ViennaCLOpenCLiDamax(ViennaCLOpenCLBackend backend, size_t n,
                                    size_t *index,
                                    cl_mem x, size_t offx, int incx)
{
  viennacl::vector_base<double> v1(x, n, offx, incx, viennacl::ocl::get_context(backend->context_id));

  *index = viennacl::linalg::index_norm_inf(v1);
  return ViennaCLSuccess;
}




// xASUM

ViennaCLStatus ViennaCLOpenCLSasum(ViennaCLOpenCLBackend backend, size_t n,
                                   float *alpha,
                                   cl_mem x, size_t offx, int incx)
{
  viennacl::vector_base<float> v1(x, n, offx, incx, viennacl::ocl::get_context(backend->context_id));

  *alpha = viennacl::linalg::norm_1(v1);
  return ViennaCLSuccess;
}

ViennaCLStatus ViennaCLOpenCLDasum(ViennaCLOpenCLBackend backend, size_t n,
                                   double *alpha,
                                   cl_mem x, size_t offx, int incx)
{
  viennacl::vector_base<double> v1(x, n, offx, incx, viennacl::ocl::get_context(backend->context_id));

  *alpha = viennacl::linalg::norm_1(v1);
  return ViennaCLSuccess;
}



// xAXPY

ViennaCLStatus ViennaCLOpenCLSaxpy(ViennaCLOpenCLBackend backend, size_t n,
                                   float alpha,
                                   cl_mem x, size_t offx, int incx,
                                   cl_mem y, size_t offy, int incy)
{
  viennacl::vector_base<float> v1(x, n, offx, incx, viennacl::ocl::get_context(backend->context_id));
  viennacl::vector_base<float> v2(y, n, offy, incy, viennacl::ocl::get_context(backend->context_id));

  v2 += alpha * v1;
  return ViennaCLSuccess;
}

ViennaCLStatus ViennaCLOpenCLDaxpy(ViennaCLOpenCLBackend backend, size_t n,
                                   double alpha,
                                   cl_mem x, size_t offx, int incx,
                                   cl_mem y, size_t offy, int incy)
{
  viennacl::vector_base<double> v1(x, n, offx, incx, viennacl::ocl::get_context(backend->context_id));
  viennacl::vector_base<double> v2(y, n, offy, incy, viennacl::ocl::get_context(backend->context_id));

  v2 += alpha * v1;
  return ViennaCLSuccess;
}


// xCOPY

ViennaCLStatus ViennaCLOpenCLScopy(ViennaCLOpenCLBackend backend, size_t n,
                                   cl_mem x, size_t offx, int incx,
                                   cl_mem y, size_t offy, int incy)
{
  viennacl::vector_base<float> v1(x, n, offx, incx, viennacl::ocl::get_context(backend->context_id));
  viennacl::vector_base<float> v2(y, n, offy, incy, viennacl::ocl::get_context(backend->context_id));

  v2 = v1;
  return ViennaCLSuccess;
}

ViennaCLStatus ViennaCLOpenCLDcopy(ViennaCLOpenCLBackend backend, size_t n,
                                   cl_mem x, size_t offx, int incx,
                                   cl_mem y, size_t offy, int incy)
{
  viennacl::vector_base<double> v1(x, n, offx, incx, viennacl::ocl::get_context(backend->context_id));
  viennacl::vector_base<double> v2(y, n, offy, incy, viennacl::ocl::get_context(backend->context_id));

  v2 = v1;
  return ViennaCLSuccess;
}

// xDOT

ViennaCLStatus ViennaCLOpenCLSdot(ViennaCLOpenCLBackend backend, size_t n,
                                  float *alpha,
                                  cl_mem x, size_t offx, int incx,
                                  cl_mem y, size_t offy, int incy)
{
  viennacl::vector_base<float> v1(x, n, offx, incx, viennacl::ocl::get_context(backend->context_id));
  viennacl::vector_base<float> v2(y, n, offy, incy, viennacl::ocl::get_context(backend->context_id));

  *alpha = viennacl::linalg::inner_prod(v1, v2);
  return ViennaCLSuccess;
}

ViennaCLStatus ViennaCLOpenCLDdot(ViennaCLOpenCLBackend backend, size_t n,
                                  double *alpha,
                                  cl_mem x, size_t offx, int incx,
                                  cl_mem y, size_t offy, int incy)
{
  viennacl::vector_base<double> v1(x, n, offx, incx, viennacl::ocl::get_context(backend->context_id));
  viennacl::vector_base<double> v2(y, n, offy, incy, viennacl::ocl::get_context(backend->context_id));

  *alpha = viennacl::linalg::inner_prod(v1, v2);
  return ViennaCLSuccess;
}


// xNRM2

ViennaCLStatus ViennaCLOpenCLSnrm2(ViennaCLOpenCLBackend backend, size_t n,
                                   float *alpha,
                                   cl_mem x, size_t offx, int incx)
{
  viennacl::vector_base<float> v1(x, n, offx, incx, viennacl::ocl::get_context(backend->context_id));

  *alpha = viennacl::linalg::norm_2(v1);
  return ViennaCLSuccess;
}

ViennaCLStatus ViennaCLOpenCLDnrm2(ViennaCLOpenCLBackend backend, size_t n,
                                   double *alpha,
                                   cl_mem x, size_t offx, int incx)
{
  viennacl::vector_base<double> v1(x, n, offx, incx, viennacl::ocl::get_context(backend->context_id));

  *alpha = viennacl::linalg::norm_2(v1);
  return ViennaCLSuccess;
}


// xROT

ViennaCLStatus ViennaCLOpenCLSrot(ViennaCLOpenCLBackend backend, size_t n,
                                  cl_mem x, size_t offx, int incx,
                                  cl_mem y, size_t offy, int incy,
                                  float c, float s)
{
  viennacl::vector_base<float> v1(x, n, offx, incx, viennacl::ocl::get_context(backend->context_id));
  viennacl::vector_base<float> v2(y, n, offy, incy, viennacl::ocl::get_context(backend->context_id));

  viennacl::linalg::plane_rotation(v1, v2, c, s);
  return ViennaCLSuccess;
}

ViennaCLStatus ViennaCLOpenCLDrot(ViennaCLOpenCLBackend backend, size_t n,
                                  cl_mem x, size_t offx, int incx,
                                  cl_mem y, size_t offy, int incy,
                                  double c, double s)
{
  viennacl::vector_base<double> v1(x, n, offx, incx, viennacl::ocl::get_context(backend->context_id));
  viennacl::vector_base<double> v2(y, n, offy, incy, viennacl::ocl::get_context(backend->context_id));

  viennacl::linalg::plane_rotation(v1, v2, c, s);
  return ViennaCLSuccess;
}



// xSCAL

ViennaCLStatus ViennaCLOpenCLSscal(ViennaCLOpenCLBackend backend, size_t n,
                                   float alpha,
                                   cl_mem x, size_t offx, int incx)
{
  viennacl::vector_base<float> v1(x, n, offx, incx, viennacl::ocl::get_context(backend->context_id));

  v1 *= alpha;
  return ViennaCLSuccess;
}

ViennaCLStatus ViennaCLOpenCLDscal(ViennaCLOpenCLBackend backend, size_t n,
                                   double alpha,
                                   cl_mem x, size_t offx, int incx)
{
  viennacl::vector_base<double> v1(x, n, offx, incx, viennacl::ocl::get_context(backend->context_id));

  v1 *= alpha;
  return ViennaCLSuccess;
}

// xSWAP

ViennaCLStatus ViennaCLOpenCLSswap(ViennaCLOpenCLBackend backend, size_t n,
                                   cl_mem x, size_t offx, int incx,
                                   cl_mem y, size_t offy, int incy)
{
  viennacl::vector_base<float> v1(x, n, offx, incx, viennacl::ocl::get_context(backend->context_id));
  viennacl::vector_base<float> v2(y, n, offy, incy, viennacl::ocl::get_context(backend->context_id));

  viennacl::swap(v1, v2);
  return ViennaCLSuccess;
}

ViennaCLStatus ViennaCLOpenCLDswap(ViennaCLOpenCLBackend backend, size_t n,
                                   cl_mem x, size_t offx, int incx,
                                   cl_mem y, size_t offy, int incy)
{
  viennacl::vector_base<double> v1(x, n, offx, incx, viennacl::ocl::get_context(backend->context_id));
  viennacl::vector_base<double> v2(y, n, offy, incy, viennacl::ocl::get_context(backend->context_id));

  viennacl::swap(v1, v2);
  return ViennaCLSuccess;
}
#endif
