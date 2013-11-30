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

typedef int ViennaCLInt;

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
  ViennaCLInt context_id;
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

  ViennaCLInt   offset;
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

  ViennaCLInt   offset;
  ViennaCLInt   inc;
  ViennaCLInt   size;
};

typedef ViennaCLVector_impl*    ViennaCLVector;


struct ViennaCLMatrix_impl
{
  ViennaCLBackend    backend;
  ViennaCLPrecision  precision;
  ViennaCLOrder      order;
  ViennaCLTranspose  trans;

  // buffer:
#ifdef VIENNACL_WITH_CUDA
  char * cuda_mem;
#endif
#ifdef VIENNACL_WITH_OPENCL
  cl_mem opencl_mem;
#endif
  char * host_mem;

  ViennaCLInt   size1;
  ViennaCLInt   start1;
  ViennaCLInt   stride1;
  ViennaCLInt   internal_size1;

  ViennaCLInt   size2;
  ViennaCLInt   start2;
  ViennaCLInt   stride2;
  ViennaCLInt   internal_size2;
};

typedef ViennaCLMatrix_impl*    ViennaCLMatrix;


/******************** BLAS Level 1 ***********************/

// IxASUM

ViennaCLStatus ViennaCLiamax(ViennaCLInt *alpha, ViennaCLVector x);

#ifdef VIENNACL_WITH_CUDA
ViennaCLStatus ViennaCLCUDAiSamax(ViennaCLCUDABackend backend, ViennaCLInt n,
                                  ViennaCLInt *alpha,
                                  float *x, ViennaCLInt offx, ViennaCLInt incx);
ViennaCLStatus ViennaCLCUDAiDamax(ViennaCLCUDABackend backend, ViennaCLInt n,
                                  ViennaCLInt *alpha,
                                  double *x, ViennaCLInt offx, ViennaCLInt incx);
#endif

#ifdef VIENNACL_WITH_OPENCL
ViennaCLStatus ViennaCLOpenCLiSamax(ViennaCLOpenCLBackend backend, ViennaCLInt n,
                                    ViennaCLInt *alpha,
                                    cl_mem x, ViennaCLInt offx, ViennaCLInt incx);
ViennaCLStatus ViennaCLOpenCLiDamax(ViennaCLOpenCLBackend backend, ViennaCLInt n,
                                    ViennaCLInt *alpha,
                                    cl_mem x, ViennaCLInt offx, ViennaCLInt incx);
#endif

ViennaCLStatus ViennaCLHostiSamax(ViennaCLHostBackend backend, ViennaCLInt n,
                                  ViennaCLInt *alpha,
                                  float *x, ViennaCLInt offx, ViennaCLInt incx);
ViennaCLStatus ViennaCLHostiDamax(ViennaCLHostBackend backend, ViennaCLInt n,
                                  ViennaCLInt *alpha,
                                  double *x, ViennaCLInt offx, ViennaCLInt incx);


// xASUM

ViennaCLStatus ViennaCLasum(ViennaCLHostScalar *alpha, ViennaCLVector x);

#ifdef VIENNACL_WITH_CUDA
ViennaCLStatus ViennaCLCUDASasum(ViennaCLCUDABackend backend, ViennaCLInt n,
                                 float *alpha,
                                 float *x, ViennaCLInt offx, ViennaCLInt incx);
ViennaCLStatus ViennaCLCUDADasum(ViennaCLCUDABackend backend, ViennaCLInt n,
                                 double *alpha,
                                 double *x, ViennaCLInt offx, ViennaCLInt incx);
#endif

#ifdef VIENNACL_WITH_OPENCL
ViennaCLStatus ViennaCLOpenCLSasum(ViennaCLOpenCLBackend backend, ViennaCLInt n,
                                   float *alpha,
                                   cl_mem x, ViennaCLInt offx, ViennaCLInt incx);
ViennaCLStatus ViennaCLOpenCLDasum(ViennaCLOpenCLBackend backend, ViennaCLInt n,
                                   double *alpha,
                                   cl_mem x, ViennaCLInt offx, ViennaCLInt incx);
#endif

ViennaCLStatus ViennaCLHostSasum(ViennaCLHostBackend backend, ViennaCLInt n,
                                 float *alpha,
                                 float *x, ViennaCLInt offx, ViennaCLInt incx);
ViennaCLStatus ViennaCLHostDasum(ViennaCLHostBackend backend, ViennaCLInt n,
                                 double *alpha,
                                 double *x, ViennaCLInt offx, ViennaCLInt incx);



// xAXPY

ViennaCLStatus ViennaCLaxpy(ViennaCLHostScalar alpha, ViennaCLVector x, ViennaCLVector y);

#ifdef VIENNACL_WITH_CUDA
ViennaCLStatus ViennaCLCUDASaxpy(ViennaCLCUDABackend backend, ViennaCLInt n,
                                 float alpha,
                                 float *x, ViennaCLInt offx, ViennaCLInt incx,
                                 float *y, ViennaCLInt offy, ViennaCLInt incy);
ViennaCLStatus ViennaCLCUDADaxpy(ViennaCLCUDABackend backend, ViennaCLInt n,
                                 double alpha,
                                 double *x, ViennaCLInt offx, ViennaCLInt incx,
                                 double *y, ViennaCLInt offy, ViennaCLInt incy);
#endif

#ifdef VIENNACL_WITH_OPENCL
ViennaCLStatus ViennaCLOpenCLSaxpy(ViennaCLOpenCLBackend backend, ViennaCLInt n,
                                   float alpha,
                                   cl_mem x, ViennaCLInt offx, ViennaCLInt incx,
                                   cl_mem y, ViennaCLInt offy, ViennaCLInt incy);
ViennaCLStatus ViennaCLOpenCLDaxpy(ViennaCLOpenCLBackend backend, ViennaCLInt n,
                                   double alpha,
                                   cl_mem x, ViennaCLInt offx, ViennaCLInt incx,
                                   cl_mem y, ViennaCLInt offy, ViennaCLInt incy);
#endif

ViennaCLStatus ViennaCLHostSaxpy(ViennaCLHostBackend backend, ViennaCLInt n,
                                 float alpha,
                                 float *x, ViennaCLInt offx, ViennaCLInt incx,
                                 float *y, ViennaCLInt offy, ViennaCLInt incy);
ViennaCLStatus ViennaCLHostDaxpy(ViennaCLHostBackend backend, ViennaCLInt n,
                                 double alpha,
                                 double *x, ViennaCLInt offx, ViennaCLInt incx,
                                 double *y, ViennaCLInt offy, ViennaCLInt incy);


// xCOPY

ViennaCLStatus ViennaCLcopy(ViennaCLVector x, ViennaCLVector y);

#ifdef VIENNACL_WITH_CUDA
ViennaCLStatus ViennaCLCUDAScopy(ViennaCLCUDABackend backend, ViennaCLInt n,
                                 float *x, ViennaCLInt offx, ViennaCLInt incx,
                                 float *y, ViennaCLInt offy, ViennaCLInt incy);
ViennaCLStatus ViennaCLCUDADcopy(ViennaCLCUDABackend backend, ViennaCLInt n,
                                 double *x, ViennaCLInt offx, ViennaCLInt incx,
                                 double *y, ViennaCLInt offy, ViennaCLInt incy);
#endif

#ifdef VIENNACL_WITH_OPENCL
ViennaCLStatus ViennaCLOpenCLScopy(ViennaCLOpenCLBackend backend, ViennaCLInt n,
                                   cl_mem x, ViennaCLInt offx, ViennaCLInt incx,
                                   cl_mem y, ViennaCLInt offy, ViennaCLInt incy);
ViennaCLStatus ViennaCLOpenCLDcopy(ViennaCLOpenCLBackend backend, ViennaCLInt n,
                                   cl_mem x, ViennaCLInt offx, ViennaCLInt incx,
                                   cl_mem y, ViennaCLInt offy, ViennaCLInt incy);
#endif

ViennaCLStatus ViennaCLHostScopy(ViennaCLHostBackend backend, ViennaCLInt n,
                                 float *x, ViennaCLInt offx, ViennaCLInt incx,
                                 float *y, ViennaCLInt offy, ViennaCLInt incy);
ViennaCLStatus ViennaCLHostDcopy(ViennaCLHostBackend backend, ViennaCLInt n,
                                 double *x, ViennaCLInt offx, ViennaCLInt incx,
                                 double *y, ViennaCLInt offy, ViennaCLInt incy);

// xDOT

ViennaCLStatus ViennaCLdot(ViennaCLHostScalar *alpha, ViennaCLVector x, ViennaCLVector y);

#ifdef VIENNACL_WITH_CUDA
ViennaCLStatus ViennaCLCUDASdot(ViennaCLCUDABackend backend, ViennaCLInt n,
                                float *alpha,
                                float *x, ViennaCLInt offx, ViennaCLInt incx,
                                float *y, ViennaCLInt offy, ViennaCLInt incy);
ViennaCLStatus ViennaCLCUDADdot(ViennaCLCUDABackend backend, ViennaCLInt n,
                                double *alpha,
                                double *x, ViennaCLInt offx, ViennaCLInt incx,
                                double *y, ViennaCLInt offy, ViennaCLInt incy);
#endif

#ifdef VIENNACL_WITH_OPENCL
ViennaCLStatus ViennaCLOpenCLSdot(ViennaCLOpenCLBackend backend, ViennaCLInt n,
                                  float *alpha,
                                  cl_mem x, ViennaCLInt offx, ViennaCLInt incx,
                                  cl_mem y, ViennaCLInt offy, ViennaCLInt incy);
ViennaCLStatus ViennaCLOpenCLDdot(ViennaCLOpenCLBackend backend, ViennaCLInt n,
                                  double *alpha,
                                  cl_mem x, ViennaCLInt offx, ViennaCLInt incx,
                                  cl_mem y, ViennaCLInt offy, ViennaCLInt incy);
#endif

ViennaCLStatus ViennaCLHostSdot(ViennaCLHostBackend backend, ViennaCLInt n,
                                float *alpha,
                                float *x, ViennaCLInt offx, ViennaCLInt incx,
                                float *y, ViennaCLInt offy, ViennaCLInt incy);
ViennaCLStatus ViennaCLHostDdot(ViennaCLHostBackend backend, ViennaCLInt n,
                                double *alpha,
                                double *x, ViennaCLInt offx, ViennaCLInt incx,
                                double *y, ViennaCLInt offy, ViennaCLInt incy);

// xNRM2

ViennaCLStatus ViennaCLnrm2(ViennaCLHostScalar *alpha, ViennaCLVector x);

#ifdef VIENNACL_WITH_CUDA
ViennaCLStatus ViennaCLCUDASnrm2(ViennaCLCUDABackend backend, ViennaCLInt n,
                                 float *alpha,
                                 float *x, ViennaCLInt offx, ViennaCLInt incx);
ViennaCLStatus ViennaCLCUDADnrm2(ViennaCLCUDABackend backend, ViennaCLInt n,
                                 double *alpha,
                                 double *x, ViennaCLInt offx, ViennaCLInt incx);
#endif

#ifdef VIENNACL_WITH_OPENCL
ViennaCLStatus ViennaCLOpenCLSnrm2(ViennaCLOpenCLBackend backend, ViennaCLInt n,
                                   float *alpha,
                                   cl_mem x, ViennaCLInt offx, ViennaCLInt incx);
ViennaCLStatus ViennaCLOpenCLDnrm2(ViennaCLOpenCLBackend backend, ViennaCLInt n,
                                   double *alpha,
                                   cl_mem x, ViennaCLInt offx, ViennaCLInt incx);
#endif

ViennaCLStatus ViennaCLHostSnrm2(ViennaCLHostBackend backend, ViennaCLInt n,
                                 float *alpha,
                                 float *x, ViennaCLInt offx, ViennaCLInt incx);
ViennaCLStatus ViennaCLHostDnrm2(ViennaCLHostBackend backend, ViennaCLInt n,
                                 double *alpha,
                                 double *x, ViennaCLInt offx, ViennaCLInt incx);


// xROT

ViennaCLStatus ViennaCLrot(ViennaCLVector     x,     ViennaCLVector y,
                           ViennaCLHostScalar c, ViennaCLHostScalar s);

#ifdef VIENNACL_WITH_CUDA
ViennaCLStatus ViennaCLCUDASrot(ViennaCLCUDABackend backend, ViennaCLInt n,
                                float *x, ViennaCLInt offx, ViennaCLInt incx,
                                float *y, ViennaCLInt offy, ViennaCLInt incy,
                                float c, float s);
ViennaCLStatus ViennaCLCUDADrot(ViennaCLCUDABackend backend, ViennaCLInt n,
                                double *x, ViennaCLInt offx, ViennaCLInt incx,
                                double *y, ViennaCLInt offy, ViennaCLInt incy,
                                double c, double s);
#endif

#ifdef VIENNACL_WITH_OPENCL
ViennaCLStatus ViennaCLOpenCLSrot(ViennaCLOpenCLBackend backend, ViennaCLInt n,
                                  cl_mem x, ViennaCLInt offx, ViennaCLInt incx,
                                  cl_mem y, ViennaCLInt offy, ViennaCLInt incy,
                                  float c, float s);
ViennaCLStatus ViennaCLOpenCLDrot(ViennaCLOpenCLBackend backend, ViennaCLInt n,
                                  cl_mem x, ViennaCLInt offx, ViennaCLInt incx,
                                  cl_mem y, ViennaCLInt offy, ViennaCLInt incy,
                                  double c, double s);
#endif

ViennaCLStatus ViennaCLHostSrot(ViennaCLHostBackend backend, ViennaCLInt n,
                                float *x, ViennaCLInt offx, ViennaCLInt incx,
                                float *y, ViennaCLInt offy, ViennaCLInt incy,
                                float c, float s);
ViennaCLStatus ViennaCLHostDrot(ViennaCLHostBackend backend, ViennaCLInt n,
                                double *x, ViennaCLInt offx, ViennaCLInt incx,
                                double *y, ViennaCLInt offy, ViennaCLInt incy,
                                double c, double s);



// xSCAL

ViennaCLStatus ViennaCLscal(ViennaCLHostScalar alpha, ViennaCLVector x);

#ifdef VIENNACL_WITH_CUDA
ViennaCLStatus ViennaCLCUDASscal(ViennaCLCUDABackend backend, ViennaCLInt n,
                                 float alpha,
                                 float *x, ViennaCLInt offx, ViennaCLInt incx);
ViennaCLStatus ViennaCLCUDADscal(ViennaCLCUDABackend backend, ViennaCLInt n,
                                 double alpha,
                                 double *x, ViennaCLInt offx, ViennaCLInt incx);
#endif

#ifdef VIENNACL_WITH_OPENCL
ViennaCLStatus ViennaCLOpenCLSscal(ViennaCLOpenCLBackend backend, ViennaCLInt n,
                                   float alpha,
                                   cl_mem x, ViennaCLInt offx, ViennaCLInt incx);
ViennaCLStatus ViennaCLOpenCLDscal(ViennaCLOpenCLBackend backend, ViennaCLInt n,
                                   double alpha,
                                   cl_mem x, ViennaCLInt offx, ViennaCLInt incx);
#endif

ViennaCLStatus ViennaCLHostSscal(ViennaCLHostBackend backend, ViennaCLInt n,
                                 float alpha,
                                 float *x, ViennaCLInt offx, ViennaCLInt incx);
ViennaCLStatus ViennaCLHostDscal(ViennaCLHostBackend backend, ViennaCLInt n,
                                 double alpha,
                                 double *x, ViennaCLInt offx, ViennaCLInt incx);


// xSWAP

ViennaCLStatus ViennaCLswap(ViennaCLVector x, ViennaCLVector y);

#ifdef VIENNACL_WITH_CUDA
ViennaCLStatus ViennaCLCUDASswap(ViennaCLCUDABackend backend, ViennaCLInt n,
                                 float *x, ViennaCLInt offx, ViennaCLInt incx,
                                 float *y, ViennaCLInt offy, ViennaCLInt incy);
ViennaCLStatus ViennaCLCUDADswap(ViennaCLCUDABackend backend, ViennaCLInt n,
                                 double *x, ViennaCLInt offx, ViennaCLInt incx,
                                 double *y, ViennaCLInt offy, ViennaCLInt incy);
#endif

#ifdef VIENNACL_WITH_OPENCL
ViennaCLStatus ViennaCLOpenCLSswap(ViennaCLOpenCLBackend backend, ViennaCLInt n,
                                   cl_mem x, ViennaCLInt offx, ViennaCLInt incx,
                                   cl_mem y, ViennaCLInt offy, ViennaCLInt incy);
ViennaCLStatus ViennaCLOpenCLDswap(ViennaCLOpenCLBackend backend, ViennaCLInt n,
                                   cl_mem x, ViennaCLInt offx, ViennaCLInt incx,
                                   cl_mem y, ViennaCLInt offy, ViennaCLInt incy);
#endif

ViennaCLStatus ViennaCLHostSswap(ViennaCLHostBackend backend, ViennaCLInt n,
                                 float *x, ViennaCLInt offx, ViennaCLInt incx,
                                 float *y, ViennaCLInt offy, ViennaCLInt incy);
ViennaCLStatus ViennaCLHostDswap(ViennaCLHostBackend backend, ViennaCLInt n,
                                 double *x, ViennaCLInt offx, ViennaCLInt incx,
                                 double *y, ViennaCLInt offy, ViennaCLInt incy);



/******************** BLAS Level 2 ***********************/

// xGEMV: y <- alpha * Ax + beta * y

ViennaCLStatus ViennaCLgemv(ViennaCLHostScalar alpha, ViennaCLMatrix A, ViennaCLVector x, ViennaCLHostScalar beta, ViennaCLVector y);

#ifdef VIENNACL_WITH_CUDA
ViennaCLStatus ViennaCLCUDASgemv(ViennaCLCUDABackend backend,
                                 ViennaCLOrder order, ViennaCLTranspose transA,
                                 ViennaCLInt m, ViennaCLInt n, float alpha, float *A, ViennaCLInt offA_row, ViennaCLInt offA_col, ViennaCLInt incA_row, ViennaCLInt incA_col, ViennaCLInt lda,
                                 float *x, ViennaCLInt offx, ViennaCLInt incx,
                                 float beta,
                                 float *y, ViennaCLInt offy, ViennaCLInt incy);
ViennaCLStatus ViennaCLCUDADgemv(ViennaCLCUDABackend backend,
                                 ViennaCLOrder order, ViennaCLTranspose transA,
                                 ViennaCLInt m, ViennaCLInt n, double alpha, double *A, ViennaCLInt offA_row, ViennaCLInt offA_col, ViennaCLInt incA_row, ViennaCLInt incA_col, ViennaCLInt lda,
                                 double *x, ViennaCLInt offx, ViennaCLInt incx,
                                 double beta,
                                 double *y, ViennaCLInt offy, ViennaCLInt incy);
#endif

#ifdef VIENNACL_WITH_OPENCL
ViennaCLStatus ViennaCLOpenCLSgemv(ViennaCLOpenCLBackend backend,
                                   ViennaCLOrder order, ViennaCLTranspose transA,
                                   ViennaCLInt m, ViennaCLInt n, float alpha, cl_mem A, ViennaCLInt offA_row, ViennaCLInt offA_col, ViennaCLInt incA_row, ViennaCLInt incA_col, ViennaCLInt lda,
                                   cl_mem x, ViennaCLInt offx, ViennaCLInt incx,
                                   float beta,
                                   cl_mem y, ViennaCLInt offy, ViennaCLInt incy);
ViennaCLStatus ViennaCLOpenCLDgemv(ViennaCLOpenCLBackend backend,
                                   ViennaCLOrder order, ViennaCLTranspose transA,
                                   ViennaCLInt m, ViennaCLInt n, double alpha, cl_mem A, ViennaCLInt offA_row, ViennaCLInt offA_col, ViennaCLInt incA_row, ViennaCLInt incA_col, ViennaCLInt lda,
                                   cl_mem x, ViennaCLInt offx, ViennaCLInt incx,
                                   double beta,
                                   cl_mem y, ViennaCLInt offy, ViennaCLInt incy);
#endif

ViennaCLStatus ViennaCLHostSgemv(ViennaCLHostBackend backend,
                                 ViennaCLOrder order, ViennaCLTranspose transA,
                                 ViennaCLInt m, ViennaCLInt n, float alpha, float *A, ViennaCLInt offA_row, ViennaCLInt offA_col, ViennaCLInt incA_row, ViennaCLInt incA_col, ViennaCLInt lda,
                                 float *x, ViennaCLInt offx, ViennaCLInt incx,
                                 float beta,
                                 float *y, ViennaCLInt offy, ViennaCLInt incy);
ViennaCLStatus ViennaCLHostDgemv(ViennaCLHostBackend backend,
                                 ViennaCLOrder order, ViennaCLTranspose transA,
                                 ViennaCLInt m, ViennaCLInt n, double alpha, double *A, ViennaCLInt offA_row, ViennaCLInt offA_col, ViennaCLInt incA_row, ViennaCLInt incA_col, ViennaCLInt lda,
                                 double *x, ViennaCLInt offx, ViennaCLInt incx,
                                 double beta,
                                 double *y, ViennaCLInt offy, ViennaCLInt incy);

// xTRSV: Ax <- x

ViennaCLStatus ViennaCLtrsv(ViennaCLMatrix A, ViennaCLVector x, ViennaCLUplo uplo);

#ifdef VIENNACL_WITH_CUDA
ViennaCLStatus ViennaCLCUDAStrsv(ViennaCLCUDABackend backend,
                                 ViennaCLUplo uplo, ViennaCLOrder order, ViennaCLTranspose transA, ViennaCLDiag diag,
                                 ViennaCLInt n, float *A, ViennaCLInt offA_row, ViennaCLInt offA_col, ViennaCLInt incA_row, ViennaCLInt incA_col, ViennaCLInt lda,
                                 float *x, ViennaCLInt offx, ViennaCLInt incx);
ViennaCLStatus ViennaCLCUDADtrsv(ViennaCLCUDABackend backend,
                                 ViennaCLUplo uplo, ViennaCLOrder order, ViennaCLTranspose transA, ViennaCLDiag diag,
                                 ViennaCLInt n, double *A, ViennaCLInt offA_row, ViennaCLInt offA_col, ViennaCLInt incA_row, ViennaCLInt incA_col, ViennaCLInt lda,
                                 double *x, ViennaCLInt offx, ViennaCLInt incx);
#endif

#ifdef VIENNACL_WITH_OPENCL
ViennaCLStatus ViennaCLOpenCLStrsv(ViennaCLOpenCLBackend backend,
                                   ViennaCLUplo uplo, ViennaCLOrder order, ViennaCLTranspose transA, ViennaCLDiag diag,
                                   ViennaCLInt n, cl_mem A, ViennaCLInt offA_row, ViennaCLInt offA_col, ViennaCLInt incA_row, ViennaCLInt incA_col, ViennaCLInt lda,
                                   cl_mem x, ViennaCLInt offx, ViennaCLInt incx);
ViennaCLStatus ViennaCLOpenCLDtrsv(ViennaCLOpenCLBackend backend,
                                   ViennaCLUplo uplo, ViennaCLOrder order, ViennaCLTranspose transA, ViennaCLDiag diag,
                                   ViennaCLInt n, cl_mem A, ViennaCLInt offA_row, ViennaCLInt offA_col, ViennaCLInt incA_row, ViennaCLInt incA_col, ViennaCLInt lda,
                                   cl_mem x, ViennaCLInt offx, ViennaCLInt incx);
#endif

ViennaCLStatus ViennaCLHostStrsv(ViennaCLHostBackend backend,
                                 ViennaCLUplo uplo, ViennaCLOrder order, ViennaCLTranspose transA, ViennaCLDiag diag,
                                 ViennaCLInt n, float *A, ViennaCLInt offA_row, ViennaCLInt offA_col, ViennaCLInt incA_row, ViennaCLInt incA_col, ViennaCLInt lda,
                                 float *x, ViennaCLInt offx, ViennaCLInt incx);
ViennaCLStatus ViennaCLHostDtrsv(ViennaCLHostBackend backend,
                                 ViennaCLUplo uplo, ViennaCLOrder order, ViennaCLTranspose transA, ViennaCLDiag diag,
                                 ViennaCLInt n, double *A, ViennaCLInt offA_row, ViennaCLInt offA_col, ViennaCLInt incA_row, ViennaCLInt incA_col, ViennaCLInt lda,
                                 double *x, ViennaCLInt offx, ViennaCLInt incx);


// xGER: A <- alpha * x * y + A

ViennaCLStatus ViennaCLger(ViennaCLHostScalar alpha, ViennaCLVector x, ViennaCLVector y, ViennaCLMatrix A);

#ifdef VIENNACL_WITH_CUDA
ViennaCLStatus ViennaCLCUDASger(ViennaCLCUDABackend backend,
                                ViennaCLOrder order,
                                ViennaCLInt m, ViennaCLInt n,
                                float alpha,
                                float *x, ViennaCLInt offx, ViennaCLInt incx,
                                float *y, ViennaCLInt offy, ViennaCLInt incy,
                                float *A, ViennaCLInt offA_row, ViennaCLInt offA_col, ViennaCLInt incA_row, ViennaCLInt incA_col, ViennaCLInt lda);
ViennaCLStatus ViennaCLCUDADger(ViennaCLCUDABackend backend,
                                ViennaCLOrder order,
                                ViennaCLInt m,  ViennaCLInt n,
                                double alpha,
                                double *x, ViennaCLInt offx, ViennaCLInt incx,
                                double *y, ViennaCLInt offy, ViennaCLInt incy,
                                double *A, ViennaCLInt offA_row, ViennaCLInt offA_col, ViennaCLInt incA_row, ViennaCLInt incA_col, ViennaCLInt lda);
#endif

#ifdef VIENNACL_WITH_OPENCL
ViennaCLStatus ViennaCLOpenCLSger(ViennaCLOpenCLBackend backend,
                                  ViennaCLOrder order,
                                  ViennaCLInt m, ViennaCLInt n,
                                  float alpha,
                                  cl_mem x, ViennaCLInt offx, ViennaCLInt incx,
                                  cl_mem y, ViennaCLInt offy, ViennaCLInt incy,
                                  cl_mem A, ViennaCLInt offA_row, ViennaCLInt offA_col, ViennaCLInt incA_row, ViennaCLInt incA_col, ViennaCLInt lda);
ViennaCLStatus ViennaCLOpenCLDger(ViennaCLOpenCLBackend backend,
                                  ViennaCLOrder order,
                                  ViennaCLInt m, ViennaCLInt n,
                                  cl_mem x, ViennaCLInt offx, ViennaCLInt incx,
                                  cl_mem y, ViennaCLInt offy, ViennaCLInt incy,
                                  double alpha, cl_mem A, ViennaCLInt offA_row, ViennaCLInt offA_col, ViennaCLInt incA_row, ViennaCLInt incA_col, ViennaCLInt lda);
#endif

ViennaCLStatus ViennaCLHostSger(ViennaCLHostBackend backend,
                                ViennaCLOrder order,
                                ViennaCLInt m, ViennaCLInt n,
                                float alpha,
                                float *x, ViennaCLInt offx, ViennaCLInt incx,
                                float *y, ViennaCLInt offy, ViennaCLInt incy,
                                float *A, ViennaCLInt offA_row, ViennaCLInt offA_col, ViennaCLInt incA_row, ViennaCLInt incA_col, ViennaCLInt lda);
ViennaCLStatus ViennaCLHostDger(ViennaCLHostBackend backend,
                                ViennaCLOrder order,
                                ViennaCLInt m, ViennaCLInt n,
                                double alpha,
                                double *x, ViennaCLInt offx, ViennaCLInt incx,
                                double *y, ViennaCLInt offy, ViennaCLInt incy,
                                double *A, ViennaCLInt offA_row, ViennaCLInt offA_col, ViennaCLInt incA_row, ViennaCLInt incA_col, ViennaCLInt lda);



/******************** BLAS Level 3 ***********************/

// xGEMM: C <- alpha * AB + beta * C

ViennaCLStatus ViennaCLgemm(ViennaCLHostScalar alpha, ViennaCLMatrix A, ViennaCLMatrix B, ViennaCLHostScalar beta, ViennaCLMatrix C);

#ifdef VIENNACL_WITH_CUDA
ViennaCLStatus ViennaCLCUDASgemm(ViennaCLCUDABackend backend,
                                 ViennaCLOrder order, ViennaCLTranspose transA, ViennaCLTranspose transB,
                                 ViennaCLInt m, ViennaCLInt n, ViennaCLInt k,
                                 float alpha,
                                 float *A, ViennaCLInt offA_row, ViennaCLInt offA_col, ViennaCLInt incA_row, ViennaCLInt incA_col, ViennaCLInt lda,
                                 float *B, ViennaCLInt offB_row, ViennaCLInt offB_col, ViennaCLInt incB_row, ViennaCLInt incB_col, ViennaCLInt ldb,
                                 float beta,
                                 float *C, ViennaCLInt offC_row, ViennaCLInt offC_col, ViennaCLInt incC_row, ViennaCLInt incC_col, ViennaCLInt ldc);
ViennaCLStatus ViennaCLCUDADgemm(ViennaCLCUDABackend backend,
                                 ViennaCLOrder order, ViennaCLTranspose transA, ViennaCLTranspose transB,
                                 ViennaCLInt m, ViennaCLInt n, ViennaCLInt k,
                                 double alpha,
                                 double *A, ViennaCLInt offA_row, ViennaCLInt offA_col, ViennaCLInt incA_row, ViennaCLInt incA_col, ViennaCLInt lda,
                                 double *B, ViennaCLInt offB_row, ViennaCLInt offB_col, ViennaCLInt incB_row, ViennaCLInt incB_col, ViennaCLInt ldb,
                                 double beta,
                                 double *C, ViennaCLInt offC_row, ViennaCLInt offC_col, ViennaCLInt incC_row, ViennaCLInt incC_col, ViennaCLInt ldc);
#endif

#ifdef VIENNACL_WITH_OPENCL
ViennaCLStatus ViennaCLOpenCLSgemm(ViennaCLOpenCLBackend backend,
                                   ViennaCLOrder order, ViennaCLTranspose transA, ViennaCLTranspose transB,
                                   ViennaCLInt m, ViennaCLInt n, ViennaCLInt k,
                                   float alpha,
                                   cl_mem *A, ViennaCLInt offA_row, ViennaCLInt offA_col, ViennaCLInt incA_row, ViennaCLInt incA_col, ViennaCLInt lda,
                                   cl_mem *B, ViennaCLInt offB_row, ViennaCLInt offB_col, ViennaCLInt incB_row, ViennaCLInt incB_col, ViennaCLInt ldb,
                                   float beta,
                                   cl_mem *C, ViennaCLInt offC_row, ViennaCLInt offC_col, ViennaCLInt incC_row, ViennaCLInt incC_col, ViennaCLInt ldc);
ViennaCLStatus ViennaCLOpenCLDgemm(ViennaCLOpenCLBackend backend,
                                   ViennaCLOrder order, ViennaCLTranspose transA, ViennaCLTranspose transB,
                                   ViennaCLInt m, ViennaCLInt n, ViennaCLInt k,
                                   double alpha,
                                   cl_mem *A, ViennaCLInt offA_row, ViennaCLInt offA_col, ViennaCLInt incA_row, ViennaCLInt incA_col, ViennaCLInt lda,
                                   cl_mem *B, ViennaCLInt offB_row, ViennaCLInt offB_col, ViennaCLInt incB_row, ViennaCLInt incB_col, ViennaCLInt ldb,
                                   double beta,
                                   cl_mem *C, ViennaCLInt offC_row, ViennaCLInt offC_col, ViennaCLInt incC_row, ViennaCLInt incC_col, ViennaCLInt ldc);
#endif

ViennaCLStatus ViennaCLHostSgemm(ViennaCLHostBackend backend,
                                 ViennaCLOrder order, ViennaCLTranspose transA, ViennaCLTranspose transB,
                                 ViennaCLInt m, ViennaCLInt n, ViennaCLInt k,
                                 float alpha,
                                 float *A, ViennaCLInt offA_row, ViennaCLInt offA_col, ViennaCLInt incA_row, ViennaCLInt incA_col, ViennaCLInt lda,
                                 float *B, ViennaCLInt offB_row, ViennaCLInt offB_col, ViennaCLInt incB_row, ViennaCLInt incB_col, ViennaCLInt ldb,
                                 float beta,
                                 float *C, ViennaCLInt offC_row, ViennaCLInt offC_col, ViennaCLInt incC_row, ViennaCLInt incC_col, ViennaCLInt ldc);
ViennaCLStatus ViennaCLHostDgemm(ViennaCLHostBackend backend,
                                 ViennaCLOrder order, ViennaCLTranspose transA, ViennaCLTranspose transB,
                                 ViennaCLInt m, ViennaCLInt n, ViennaCLInt k,
                                 double alpha,
                                 double *A, ViennaCLInt offA_row, ViennaCLInt offA_col, ViennaCLInt incA_row, ViennaCLInt incA_col, ViennaCLInt lda,
                                 double *B, ViennaCLInt offB_row, ViennaCLInt offB_col, ViennaCLInt incB_row, ViennaCLInt incB_col, ViennaCLInt ldb,
                                 double beta,
                                 double *C, ViennaCLInt offC_row, ViennaCLInt offC_col, ViennaCLInt incC_row, ViennaCLInt incC_col, ViennaCLInt ldc);


// xTRSM: B <- alpha * A^{-1} B

ViennaCLStatus ViennaCLtrsm(ViennaCLHostScalar alpha, ViennaCLMatrix A, ViennaCLMatrix B);

#ifdef VIENNACL_WITH_CUDA
ViennaCLStatus ViennaCLCUDAStrsm(ViennaCLCUDABackend backend,
                                 ViennaCLOrder order, ViennaCLUplo uplo, ViennaCLTranspose transA, ViennaCLDiag diag, ViennaCLTranspose transB,
                                 ViennaCLInt m, ViennaCLInt n,
                                 float alpha,
                                 float *A, ViennaCLInt offA_row, ViennaCLInt offA_col, ViennaCLInt incA_row, ViennaCLInt incA_col, ViennaCLInt lda,
                                 float *B, ViennaCLInt offB_row, ViennaCLInt offB_col, ViennaCLInt incB_row, ViennaCLInt incB_col, ViennaCLInt ldb);
ViennaCLStatus ViennaCLCUDADtrsm(ViennaCLCUDABackend backend,
                                 ViennaCLOrder order, ViennaCLUplo uplo, ViennaCLTranspose transA, ViennaCLDiag diag, ViennaCLTranspose transB,
                                 ViennaCLInt m, ViennaCLInt n,
                                 double alpha,
                                 double *A, ViennaCLInt offA_row, ViennaCLInt offA_col, ViennaCLInt incA_row, ViennaCLInt incA_col, ViennaCLInt lda,
                                 double *B, ViennaCLInt offB_row, ViennaCLInt offB_col, ViennaCLInt incB_row, ViennaCLInt incB_col, ViennaCLInt ldb);
#endif

#ifdef VIENNACL_WITH_OPENCL
ViennaCLStatus ViennaCLOpenCLStrsm(ViennaCLOpenCLBackend backend,
                                   ViennaCLOrder order, ViennaCLUplo uplo, ViennaCLTranspose transA, ViennaCLDiag diag, ViennaCLTranspose transB,
                                   ViennaCLInt m, ViennaCLInt n,
                                   float alpha,
                                   cl_mem *A, ViennaCLInt offA_row, ViennaCLInt offA_col, ViennaCLInt incA_row, ViennaCLInt incA_col, ViennaCLInt lda,
                                   cl_mem *B, ViennaCLInt offB_row, ViennaCLInt offB_col, ViennaCLInt incB_row, ViennaCLInt incB_col, ViennaCLInt ldb);
ViennaCLStatus ViennaCLOpenCLDtrsm(ViennaCLOpenCLBackend backend,
                                   ViennaCLOrder order, ViennaCLUplo uplo, ViennaCLTranspose transA, ViennaCLDiag diag, ViennaCLTranspose transB,
                                   ViennaCLInt m, ViennaCLInt n,
                                   double alpha,
                                   cl_mem *A, ViennaCLInt offA_row, ViennaCLInt offA_col, ViennaCLInt incA_row, ViennaCLInt incA_col, ViennaCLInt lda,
                                   cl_mem *B, ViennaCLInt offB_row, ViennaCLInt offB_col, ViennaCLInt incB_row, ViennaCLInt incB_col, ViennaCLInt ldb);
#endif

ViennaCLStatus ViennaCLHostStrsm(ViennaCLHostBackend backend,
                                 ViennaCLOrder order, ViennaCLUplo uplo, ViennaCLTranspose transA, ViennaCLDiag diag, ViennaCLTranspose transB,
                                 ViennaCLInt m, ViennaCLInt n,
                                 float alpha,
                                 float *A, ViennaCLInt offA_row, ViennaCLInt offA_col, ViennaCLInt incA_row, ViennaCLInt incA_col, ViennaCLInt lda,
                                 float *B, ViennaCLInt offB_row, ViennaCLInt offB_col, ViennaCLInt incB_row, ViennaCLInt incB_col, ViennaCLInt ldb);
ViennaCLStatus ViennaCLHostDtrsm(ViennaCLHostBackend backend,
                                 ViennaCLOrder order, ViennaCLUplo uplo, ViennaCLTranspose transA, ViennaCLDiag diag, ViennaCLTranspose transB,
                                 ViennaCLInt m, ViennaCLInt n,
                                 double alpha,
                                 double *A, ViennaCLInt offA_row, ViennaCLInt offA_col, ViennaCLInt incA_row, ViennaCLInt incA_col, ViennaCLInt lda,
                                 double *B, ViennaCLInt offB_row, ViennaCLInt offB_col, ViennaCLInt incB_row, ViennaCLInt incB_col, ViennaCLInt ldb);

#ifdef __cplusplus
}
#endif


#endif
