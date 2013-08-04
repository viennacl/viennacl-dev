#ifndef VIENNACL_VIENNACL_HPP
#define VIENNACL_VIENNACL_HPP


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

#include <stdlib.h>

#ifdef VIENNACL_WITH_OPENCL
#ifdef __APPLE__
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif
#endif


#ifdef __cplusplus
extern "C" {
#endif


/************** Enums ***************/

typedef enum
{
  ViennaCLCUDA,
  ViennaCLOpenCL,
  ViennaCLHost
} ViennaCLBackendTypes;


typedef enum
{
  ViennaCLRowMajor,
  ViennaCLColumnMajor
} ViennaCLOrder;

typedef enum
{
  ViennaCLNoTrans,
  ViennaCLTrans
} ViennaCLTranspose;

typedef enum
{
  ViennaCLUpper,
  ViennaCLLower
} ViennaCLUplo;

typedef enum
{
  ViennaCLUnit,
  ViennaCLNonUnit
} ViennaCLDiag;


typedef enum
{
  ViennaCLSuccess = 0,
  ViennaCLGenericFailure
} ViennaCLStatus;

typedef enum
{
  ViennaCLFloat,
  ViennaCLDouble
} ViennaCLPrecision;



/************* Backend Management ******************/

struct ViennaCLCUDABackend_impl
{
    //TODO: Add stream and/or device descriptors here
};
typedef ViennaCLCUDABackend_impl*   ViennaCLCUDABackend;

struct ViennaCLOpenCLBackend_impl
{
  std::size_t context_id;
};
typedef ViennaCLOpenCLBackend_impl*   ViennaCLOpenCLBackend;

struct ViennaCLHostBackend_impl
{
  // Nothing to specify *at the moment*
};
typedef ViennaCLHostBackend_impl*   ViennaCLHostBackend;


/** @brief Generic backend for CUDA, OpenCL, host-based stuff */
struct ViennaCLBackend_impl
{
  ViennaCLBackendTypes backend_type;

#ifdef VIENNACL_WITH_CUDA
  ViennaCLCUDABackend     cuda_backend;
#endif
#ifdef VIENNACL_WITH_OPENCL
  ViennaCLOpenCLBackend   opencl_backend;
#endif
  ViennaCLHostBackend     host_backend;
};
typedef ViennaCLBackend_impl*   ViennaCLBackend;



/******** User Types **********/

struct ViennaCLHostScalar_impl
{
  ViennaCLPrecision  precision;

  union {
    float  value_float;
    double value_double;
  };
};

typedef ViennaCLHostScalar_impl*    ViennaCLHostScalar;

struct ViennaCLScalar_impl
{
  ViennaCLBackend    backend;
  ViennaCLPrecision  precision;

  // buffer:
#ifdef VIENNACL_WITH_CUDA
  char * cuda_mem;
#endif
#ifdef VIENNACL_WITH_OPENCL
  cl_mem opencl_mem;
#endif
  char * host_mem;

  size_t   offset;
};

typedef ViennaCLScalar_impl*    ViennaCLScalar;




struct ViennaCLVector_impl
{
  ViennaCLBackend    backend;
  ViennaCLPrecision  precision;

  // buffer:
#ifdef VIENNACL_WITH_CUDA
  char * cuda_mem;
#endif
#ifdef VIENNACL_WITH_OPENCL
  cl_mem opencl_mem;
#endif
  char * host_mem;

  size_t   offset;
  int      inc;
  size_t   size;
};

typedef ViennaCLVector_impl*    ViennaCLVector;


/******************** BLAS Level 1 ***********************/


// xSWAP

ViennaCLStatus ViennaCLswap(ViennaCLVector x, ViennaCLVector y);

#ifdef VIENNACL_WITH_CUDA
ViennaCLStatus ViennaCLCUDASswap(ViennaCLCUDABackend backend, int n,
                                 float *x, int offx, int incx,
                                 float *y, int offy, int incy);
ViennaCLStatus ViennaCLCUDADswap(ViennaCLCUDABackend backend, int n,
                                 double *x, int offx, int incx,
                                 double *y, int offy, int incy);
#endif

#ifdef VIENNACL_WITH_OPENCL
ViennaCLStatus ViennaCLOpenCLSswap(ViennaCLOpenCLBackend backend, size_t n,
                                   cl_mem x, size_t offx, int incx,
                                   cl_mem y, size_t offy, int incy);
ViennaCLStatus ViennaCLOpenCLDswap(ViennaCLOpenCLBackend backend, size_t n,
                                   cl_mem x, size_t offx, int incx,
                                   cl_mem y, size_t offy, int incy);
#endif

ViennaCLStatus ViennaCLHostSswap(ViennaCLHostBackend backend, size_t n,
                                 float *x, size_t offx, int incx,
                                 float *y, size_t offy, int incy);
ViennaCLStatus ViennaCLHostDswap(ViennaCLHostBackend backend, size_t n,
                                 double *x, size_t offx, int incx,
                                 double *y, size_t offy, int incy);

// xSCAL

ViennaCLStatus ViennaCLscal(ViennaCLHostScalar alpha, ViennaCLVector x);

#ifdef VIENNACL_WITH_CUDA
ViennaCLStatus ViennaCLCUDASscal(ViennaCLCUDABackend backend, int n,
                                 float alpha,
                                 float *x, int offx, int incx);
ViennaCLStatus ViennaCLCUDADscal(ViennaCLCUDABackend backend, int n,
                                 double alpha,
                                 double *x, int offx, int incx);
#endif

#ifdef VIENNACL_WITH_OPENCL
ViennaCLStatus ViennaCLOpenCLSscal(ViennaCLOpenCLBackend backend, size_t n,
                                   float alpha,
                                   cl_mem x, size_t offx, int incx);
ViennaCLStatus ViennaCLOpenCLDscal(ViennaCLOpenCLBackend backend, size_t n,
                                   double alpha,
                                   cl_mem x, size_t offx, int incx);
#endif

ViennaCLStatus ViennaCLHostSscal(ViennaCLHostBackend backend, size_t n,
                                 float alpha,
                                 float *x, size_t offx, int incx);
ViennaCLStatus ViennaCLHostDscal(ViennaCLHostBackend backend, size_t n,
                                 double alpha,
                                 double *x, size_t offx, int incx);




#ifdef __cplusplus
}
#endif


#endif
