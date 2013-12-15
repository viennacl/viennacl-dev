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
#include "viennacl_private.hpp"

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

ViennaCLStatus ViennaCLOpenCLiSamax(ViennaCLBackend backend, ViennaCLInt n,
                                    ViennaCLInt *index,
                                    cl_mem x, ViennaCLInt offx, ViennaCLInt incx)
{
  viennacl::vector_base<float> v1(x, n, offx, incx, viennacl::ocl::get_context(backend->opencl_backend.context_id));

  *index = viennacl::linalg::index_norm_inf(v1);
  return ViennaCLSuccess;
}

ViennaCLStatus ViennaCLOpenCLiDamax(ViennaCLBackend backend, ViennaCLInt n,
                                    ViennaCLInt *index,
                                    cl_mem x, ViennaCLInt offx, ViennaCLInt incx)
{
  viennacl::vector_base<double> v1(x, n, offx, incx, viennacl::ocl::get_context(backend->opencl_backend.context_id));

  *index = viennacl::linalg::index_norm_inf(v1);
  return ViennaCLSuccess;
}




// xASUM

ViennaCLStatus ViennaCLOpenCLSasum(ViennaCLBackend backend, ViennaCLInt n,
                                   float *alpha,
                                   cl_mem x, ViennaCLInt offx, ViennaCLInt incx)
{
  viennacl::vector_base<float> v1(x, n, offx, incx, viennacl::ocl::get_context(backend->opencl_backend.context_id));

  *alpha = viennacl::linalg::norm_1(v1);
  return ViennaCLSuccess;
}

ViennaCLStatus ViennaCLOpenCLDasum(ViennaCLBackend backend, ViennaCLInt n,
                                   double *alpha,
                                   cl_mem x, ViennaCLInt offx, ViennaCLInt incx)
{
  viennacl::vector_base<double> v1(x, n, offx, incx, viennacl::ocl::get_context(backend->opencl_backend.context_id));

  *alpha = viennacl::linalg::norm_1(v1);
  return ViennaCLSuccess;
}



// xAXPY

ViennaCLStatus ViennaCLOpenCLSaxpy(ViennaCLBackend backend, ViennaCLInt n,
                                   float alpha,
                                   cl_mem x, ViennaCLInt offx, ViennaCLInt incx,
                                   cl_mem y, ViennaCLInt offy, ViennaCLInt incy)
{
  viennacl::vector_base<float> v1(x, n, offx, incx, viennacl::ocl::get_context(backend->opencl_backend.context_id));
  viennacl::vector_base<float> v2(y, n, offy, incy, viennacl::ocl::get_context(backend->opencl_backend.context_id));

  v2 += alpha * v1;
  return ViennaCLSuccess;
}

ViennaCLStatus ViennaCLOpenCLDaxpy(ViennaCLBackend backend, ViennaCLInt n,
                                   double alpha,
                                   cl_mem x, ViennaCLInt offx, ViennaCLInt incx,
                                   cl_mem y, ViennaCLInt offy, ViennaCLInt incy)
{
  viennacl::vector_base<double> v1(x, n, offx, incx, viennacl::ocl::get_context(backend->opencl_backend.context_id));
  viennacl::vector_base<double> v2(y, n, offy, incy, viennacl::ocl::get_context(backend->opencl_backend.context_id));

  v2 += alpha * v1;
  return ViennaCLSuccess;
}


// xCOPY

ViennaCLStatus ViennaCLOpenCLScopy(ViennaCLBackend backend, ViennaCLInt n,
                                   cl_mem x, ViennaCLInt offx, ViennaCLInt incx,
                                   cl_mem y, ViennaCLInt offy, ViennaCLInt incy)
{
  viennacl::vector_base<float> v1(x, n, offx, incx, viennacl::ocl::get_context(backend->opencl_backend.context_id));
  viennacl::vector_base<float> v2(y, n, offy, incy, viennacl::ocl::get_context(backend->opencl_backend.context_id));

  v2 = v1;
  return ViennaCLSuccess;
}

ViennaCLStatus ViennaCLOpenCLDcopy(ViennaCLBackend backend, ViennaCLInt n,
                                   cl_mem x, ViennaCLInt offx, ViennaCLInt incx,
                                   cl_mem y, ViennaCLInt offy, ViennaCLInt incy)
{
  viennacl::vector_base<double> v1(x, n, offx, incx, viennacl::ocl::get_context(backend->opencl_backend.context_id));
  viennacl::vector_base<double> v2(y, n, offy, incy, viennacl::ocl::get_context(backend->opencl_backend.context_id));

  v2 = v1;
  return ViennaCLSuccess;
}

// xDOT

ViennaCLStatus ViennaCLOpenCLSdot(ViennaCLBackend backend, ViennaCLInt n,
                                  float *alpha,
                                  cl_mem x, ViennaCLInt offx, ViennaCLInt incx,
                                  cl_mem y, ViennaCLInt offy, ViennaCLInt incy)
{
  viennacl::vector_base<float> v1(x, n, offx, incx, viennacl::ocl::get_context(backend->opencl_backend.context_id));
  viennacl::vector_base<float> v2(y, n, offy, incy, viennacl::ocl::get_context(backend->opencl_backend.context_id));

  *alpha = viennacl::linalg::inner_prod(v1, v2);
  return ViennaCLSuccess;
}

ViennaCLStatus ViennaCLOpenCLDdot(ViennaCLBackend backend, ViennaCLInt n,
                                  double *alpha,
                                  cl_mem x, ViennaCLInt offx, ViennaCLInt incx,
                                  cl_mem y, ViennaCLInt offy, ViennaCLInt incy)
{
  viennacl::vector_base<double> v1(x, n, offx, incx, viennacl::ocl::get_context(backend->opencl_backend.context_id));
  viennacl::vector_base<double> v2(y, n, offy, incy, viennacl::ocl::get_context(backend->opencl_backend.context_id));

  *alpha = viennacl::linalg::inner_prod(v1, v2);
  return ViennaCLSuccess;
}


// xNRM2

ViennaCLStatus ViennaCLOpenCLSnrm2(ViennaCLBackend backend, ViennaCLInt n,
                                   float *alpha,
                                   cl_mem x, ViennaCLInt offx, ViennaCLInt incx)
{
  viennacl::vector_base<float> v1(x, n, offx, incx, viennacl::ocl::get_context(backend->opencl_backend.context_id));

  *alpha = viennacl::linalg::norm_2(v1);
  return ViennaCLSuccess;
}

ViennaCLStatus ViennaCLOpenCLDnrm2(ViennaCLBackend backend, ViennaCLInt n,
                                   double *alpha,
                                   cl_mem x, ViennaCLInt offx, ViennaCLInt incx)
{
  viennacl::vector_base<double> v1(x, n, offx, incx, viennacl::ocl::get_context(backend->opencl_backend.context_id));

  *alpha = viennacl::linalg::norm_2(v1);
  return ViennaCLSuccess;
}


// xROT

ViennaCLStatus ViennaCLOpenCLSrot(ViennaCLBackend backend, ViennaCLInt n,
                                  cl_mem x, ViennaCLInt offx, ViennaCLInt incx,
                                  cl_mem y, ViennaCLInt offy, ViennaCLInt incy,
                                  float c, float s)
{
  viennacl::vector_base<float> v1(x, n, offx, incx, viennacl::ocl::get_context(backend->opencl_backend.context_id));
  viennacl::vector_base<float> v2(y, n, offy, incy, viennacl::ocl::get_context(backend->opencl_backend.context_id));

  viennacl::linalg::plane_rotation(v1, v2, c, s);
  return ViennaCLSuccess;
}

ViennaCLStatus ViennaCLOpenCLDrot(ViennaCLBackend backend, ViennaCLInt n,
                                  cl_mem x, ViennaCLInt offx, ViennaCLInt incx,
                                  cl_mem y, ViennaCLInt offy, ViennaCLInt incy,
                                  double c, double s)
{
  viennacl::vector_base<double> v1(x, n, offx, incx, viennacl::ocl::get_context(backend->opencl_backend.context_id));
  viennacl::vector_base<double> v2(y, n, offy, incy, viennacl::ocl::get_context(backend->opencl_backend.context_id));

  viennacl::linalg::plane_rotation(v1, v2, c, s);
  return ViennaCLSuccess;
}



// xSCAL

ViennaCLStatus ViennaCLOpenCLSscal(ViennaCLBackend backend, ViennaCLInt n,
                                   float alpha,
                                   cl_mem x, ViennaCLInt offx, ViennaCLInt incx)
{
  viennacl::vector_base<float> v1(x, n, offx, incx, viennacl::ocl::get_context(backend->opencl_backend.context_id));

  v1 *= alpha;
  return ViennaCLSuccess;
}

ViennaCLStatus ViennaCLOpenCLDscal(ViennaCLBackend backend, ViennaCLInt n,
                                   double alpha,
                                   cl_mem x, ViennaCLInt offx, ViennaCLInt incx)
{
  viennacl::vector_base<double> v1(x, n, offx, incx, viennacl::ocl::get_context(backend->opencl_backend.context_id));

  v1 *= alpha;
  return ViennaCLSuccess;
}

// xSWAP

ViennaCLStatus ViennaCLOpenCLSswap(ViennaCLBackend backend, ViennaCLInt n,
                                   cl_mem x, ViennaCLInt offx, ViennaCLInt incx,
                                   cl_mem y, ViennaCLInt offy, ViennaCLInt incy)
{
  viennacl::vector_base<float> v1(x, n, offx, incx, viennacl::ocl::get_context(backend->opencl_backend.context_id));
  viennacl::vector_base<float> v2(y, n, offy, incy, viennacl::ocl::get_context(backend->opencl_backend.context_id));

  viennacl::swap(v1, v2);
  return ViennaCLSuccess;
}

ViennaCLStatus ViennaCLOpenCLDswap(ViennaCLBackend backend, ViennaCLInt n,
                                   cl_mem x, ViennaCLInt offx, ViennaCLInt incx,
                                   cl_mem y, ViennaCLInt offy, ViennaCLInt incy)
{
  viennacl::vector_base<double> v1(x, n, offx, incx, viennacl::ocl::get_context(backend->opencl_backend.context_id));
  viennacl::vector_base<double> v2(y, n, offy, incy, viennacl::ocl::get_context(backend->opencl_backend.context_id));

  viennacl::swap(v1, v2);
  return ViennaCLSuccess;
}
#endif
