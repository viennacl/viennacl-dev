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



// xSCAL

ViennaCLStatus ViennaCLHostSscal(ViennaCLHostBackend backend, size_t n,
                                 float alpha,
                                 float *x, size_t offx, int incx)
{
  viennacl::vector_base<float> v1(x, n, viennacl::MAIN_MEMORY, offx, incx);

  v1 *= alpha;
  return ViennaCLSuccess;
}

ViennaCLStatus ViennaCLHostDscal(ViennaCLHostBackend backend, size_t n,
                                 double alpha,
                                 double *x, size_t offx, int incx)
{
  viennacl::vector_base<double> v1(x, n, viennacl::MAIN_MEMORY, offx, incx);

  v1 *= alpha;
  return ViennaCLSuccess;
}

// xSWAP

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
