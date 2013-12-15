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
  ViennaCLFloat,
  ViennaCLDouble
} ViennaCLPrecision;

// Error codes:
typedef enum
{
  ViennaCLSuccess = 0,
  ViennaCLGenericFailure
} ViennaCLStatus;


/************* Backend Management ******************/

/** @brief Generic backend for CUDA, OpenCL, host-based stuff */
struct ViennaCLBackend_impl;
typedef ViennaCLBackend_impl*   ViennaCLBackend;

ViennaCLStatus ViennaCLBackendCreate(ViennaCLBackend * backend);
ViennaCLStatus ViennaCLBackendSetOpenCLContextID(ViennaCLBackend backend, ViennaCLInt context_id);
ViennaCLStatus ViennaCLBackendDestroy(ViennaCLBackend * backend);

/******** User Types **********/

struct ViennaCLHostScalar_impl;
typedef ViennaCLHostScalar_impl*    ViennaCLHostScalar;

struct ViennaCLScalar_impl;
typedef ViennaCLScalar_impl*        ViennaCLScalar;

struct ViennaCLVector_impl;
typedef ViennaCLVector_impl*        ViennaCLVector;

struct ViennaCLMatrix_impl;
typedef ViennaCLMatrix_impl*        ViennaCLMatrix;


/******************** BLAS Level 1 ***********************/

// IxASUM

ViennaCLStatus ViennaCLiamax(ViennaCLInt *alpha, ViennaCLVector x);

ViennaCLStatus ViennaCLCUDAiSamax(ViennaCLBackend backend, ViennaCLInt n,
                                  ViennaCLInt *alpha,
                                  float *x, ViennaCLInt offx, ViennaCLInt incx);
ViennaCLStatus ViennaCLCUDAiDamax(ViennaCLBackend backend, ViennaCLInt n,
                                  ViennaCLInt *alpha,
                                  double *x, ViennaCLInt offx, ViennaCLInt incx);

#ifdef VIENNACL_WITH_OPENCL
ViennaCLStatus ViennaCLOpenCLiSamax(ViennaCLBackend backend, ViennaCLInt n,
                                    ViennaCLInt *alpha,
                                    cl_mem x, ViennaCLInt offx, ViennaCLInt incx);
ViennaCLStatus ViennaCLOpenCLiDamax(ViennaCLBackend backend, ViennaCLInt n,
                                    ViennaCLInt *alpha,
                                    cl_mem x, ViennaCLInt offx, ViennaCLInt incx);
#endif

ViennaCLStatus ViennaCLHostiSamax(ViennaCLBackend backend, ViennaCLInt n,
                                  ViennaCLInt *alpha,
                                  float *x, ViennaCLInt offx, ViennaCLInt incx);
ViennaCLStatus ViennaCLHostiDamax(ViennaCLBackend backend, ViennaCLInt n,
                                  ViennaCLInt *alpha,
                                  double *x, ViennaCLInt offx, ViennaCLInt incx);


// xASUM

ViennaCLStatus ViennaCLasum(ViennaCLHostScalar *alpha, ViennaCLVector x);

ViennaCLStatus ViennaCLCUDASasum(ViennaCLBackend backend, ViennaCLInt n,
                                 float *alpha,
                                 float *x, ViennaCLInt offx, ViennaCLInt incx);
ViennaCLStatus ViennaCLCUDADasum(ViennaCLBackend backend, ViennaCLInt n,
                                 double *alpha,
                                 double *x, ViennaCLInt offx, ViennaCLInt incx);

#ifdef VIENNACL_WITH_OPENCL
ViennaCLStatus ViennaCLOpenCLSasum(ViennaCLBackend backend, ViennaCLInt n,
                                   float *alpha,
                                   cl_mem x, ViennaCLInt offx, ViennaCLInt incx);
ViennaCLStatus ViennaCLOpenCLDasum(ViennaCLBackend backend, ViennaCLInt n,
                                   double *alpha,
                                   cl_mem x, ViennaCLInt offx, ViennaCLInt incx);
#endif

ViennaCLStatus ViennaCLHostSasum(ViennaCLBackend backend, ViennaCLInt n,
                                 float *alpha,
                                 float *x, ViennaCLInt offx, ViennaCLInt incx);
ViennaCLStatus ViennaCLHostDasum(ViennaCLBackend backend, ViennaCLInt n,
                                 double *alpha,
                                 double *x, ViennaCLInt offx, ViennaCLInt incx);



// xAXPY

ViennaCLStatus ViennaCLaxpy(ViennaCLHostScalar alpha, ViennaCLVector x, ViennaCLVector y);

ViennaCLStatus ViennaCLCUDASaxpy(ViennaCLBackend backend, ViennaCLInt n,
                                 float alpha,
                                 float *x, ViennaCLInt offx, ViennaCLInt incx,
                                 float *y, ViennaCLInt offy, ViennaCLInt incy);
ViennaCLStatus ViennaCLCUDADaxpy(ViennaCLBackend backend, ViennaCLInt n,
                                 double alpha,
                                 double *x, ViennaCLInt offx, ViennaCLInt incx,
                                 double *y, ViennaCLInt offy, ViennaCLInt incy);

#ifdef VIENNACL_WITH_OPENCL
ViennaCLStatus ViennaCLOpenCLSaxpy(ViennaCLBackend backend, ViennaCLInt n,
                                   float alpha,
                                   cl_mem x, ViennaCLInt offx, ViennaCLInt incx,
                                   cl_mem y, ViennaCLInt offy, ViennaCLInt incy);
ViennaCLStatus ViennaCLOpenCLDaxpy(ViennaCLBackend backend, ViennaCLInt n,
                                   double alpha,
                                   cl_mem x, ViennaCLInt offx, ViennaCLInt incx,
                                   cl_mem y, ViennaCLInt offy, ViennaCLInt incy);
#endif

ViennaCLStatus ViennaCLHostSaxpy(ViennaCLBackend backend, ViennaCLInt n,
                                 float alpha,
                                 float *x, ViennaCLInt offx, ViennaCLInt incx,
                                 float *y, ViennaCLInt offy, ViennaCLInt incy);
ViennaCLStatus ViennaCLHostDaxpy(ViennaCLBackend backend, ViennaCLInt n,
                                 double alpha,
                                 double *x, ViennaCLInt offx, ViennaCLInt incx,
                                 double *y, ViennaCLInt offy, ViennaCLInt incy);


// xCOPY

ViennaCLStatus ViennaCLcopy(ViennaCLVector x, ViennaCLVector y);

ViennaCLStatus ViennaCLCUDAScopy(ViennaCLBackend backend, ViennaCLInt n,
                                 float *x, ViennaCLInt offx, ViennaCLInt incx,
                                 float *y, ViennaCLInt offy, ViennaCLInt incy);
ViennaCLStatus ViennaCLCUDADcopy(ViennaCLBackend backend, ViennaCLInt n,
                                 double *x, ViennaCLInt offx, ViennaCLInt incx,
                                 double *y, ViennaCLInt offy, ViennaCLInt incy);

#ifdef VIENNACL_WITH_OPENCL
ViennaCLStatus ViennaCLOpenCLScopy(ViennaCLBackend backend, ViennaCLInt n,
                                   cl_mem x, ViennaCLInt offx, ViennaCLInt incx,
                                   cl_mem y, ViennaCLInt offy, ViennaCLInt incy);
ViennaCLStatus ViennaCLOpenCLDcopy(ViennaCLBackend backend, ViennaCLInt n,
                                   cl_mem x, ViennaCLInt offx, ViennaCLInt incx,
                                   cl_mem y, ViennaCLInt offy, ViennaCLInt incy);
#endif

ViennaCLStatus ViennaCLHostScopy(ViennaCLBackend backend, ViennaCLInt n,
                                 float *x, ViennaCLInt offx, ViennaCLInt incx,
                                 float *y, ViennaCLInt offy, ViennaCLInt incy);
ViennaCLStatus ViennaCLHostDcopy(ViennaCLBackend backend, ViennaCLInt n,
                                 double *x, ViennaCLInt offx, ViennaCLInt incx,
                                 double *y, ViennaCLInt offy, ViennaCLInt incy);

// xDOT

ViennaCLStatus ViennaCLdot(ViennaCLHostScalar *alpha, ViennaCLVector x, ViennaCLVector y);

ViennaCLStatus ViennaCLCUDASdot(ViennaCLBackend backend, ViennaCLInt n,
                                float *alpha,
                                float *x, ViennaCLInt offx, ViennaCLInt incx,
                                float *y, ViennaCLInt offy, ViennaCLInt incy);
ViennaCLStatus ViennaCLCUDADdot(ViennaCLBackend backend, ViennaCLInt n,
                                double *alpha,
                                double *x, ViennaCLInt offx, ViennaCLInt incx,
                                double *y, ViennaCLInt offy, ViennaCLInt incy);

#ifdef VIENNACL_WITH_OPENCL
ViennaCLStatus ViennaCLOpenCLSdot(ViennaCLBackend backend, ViennaCLInt n,
                                  float *alpha,
                                  cl_mem x, ViennaCLInt offx, ViennaCLInt incx,
                                  cl_mem y, ViennaCLInt offy, ViennaCLInt incy);
ViennaCLStatus ViennaCLOpenCLDdot(ViennaCLBackend backend, ViennaCLInt n,
                                  double *alpha,
                                  cl_mem x, ViennaCLInt offx, ViennaCLInt incx,
                                  cl_mem y, ViennaCLInt offy, ViennaCLInt incy);
#endif

ViennaCLStatus ViennaCLHostSdot(ViennaCLBackend backend, ViennaCLInt n,
                                float *alpha,
                                float *x, ViennaCLInt offx, ViennaCLInt incx,
                                float *y, ViennaCLInt offy, ViennaCLInt incy);
ViennaCLStatus ViennaCLHostDdot(ViennaCLBackend backend, ViennaCLInt n,
                                double *alpha,
                                double *x, ViennaCLInt offx, ViennaCLInt incx,
                                double *y, ViennaCLInt offy, ViennaCLInt incy);

// xNRM2

ViennaCLStatus ViennaCLnrm2(ViennaCLHostScalar *alpha, ViennaCLVector x);

ViennaCLStatus ViennaCLCUDASnrm2(ViennaCLBackend backend, ViennaCLInt n,
                                 float *alpha,
                                 float *x, ViennaCLInt offx, ViennaCLInt incx);
ViennaCLStatus ViennaCLCUDADnrm2(ViennaCLBackend backend, ViennaCLInt n,
                                 double *alpha,
                                 double *x, ViennaCLInt offx, ViennaCLInt incx);

#ifdef VIENNACL_WITH_OPENCL
ViennaCLStatus ViennaCLOpenCLSnrm2(ViennaCLBackend backend, ViennaCLInt n,
                                   float *alpha,
                                   cl_mem x, ViennaCLInt offx, ViennaCLInt incx);
ViennaCLStatus ViennaCLOpenCLDnrm2(ViennaCLBackend backend, ViennaCLInt n,
                                   double *alpha,
                                   cl_mem x, ViennaCLInt offx, ViennaCLInt incx);
#endif

ViennaCLStatus ViennaCLHostSnrm2(ViennaCLBackend backend, ViennaCLInt n,
                                 float *alpha,
                                 float *x, ViennaCLInt offx, ViennaCLInt incx);
ViennaCLStatus ViennaCLHostDnrm2(ViennaCLBackend backend, ViennaCLInt n,
                                 double *alpha,
                                 double *x, ViennaCLInt offx, ViennaCLInt incx);


// xROT

ViennaCLStatus ViennaCLrot(ViennaCLVector     x,     ViennaCLVector y,
                           ViennaCLHostScalar c, ViennaCLHostScalar s);

ViennaCLStatus ViennaCLCUDASrot(ViennaCLBackend backend, ViennaCLInt n,
                                float *x, ViennaCLInt offx, ViennaCLInt incx,
                                float *y, ViennaCLInt offy, ViennaCLInt incy,
                                float c, float s);
ViennaCLStatus ViennaCLCUDADrot(ViennaCLBackend backend, ViennaCLInt n,
                                double *x, ViennaCLInt offx, ViennaCLInt incx,
                                double *y, ViennaCLInt offy, ViennaCLInt incy,
                                double c, double s);

#ifdef VIENNACL_WITH_OPENCL
ViennaCLStatus ViennaCLOpenCLSrot(ViennaCLBackend backend, ViennaCLInt n,
                                  cl_mem x, ViennaCLInt offx, ViennaCLInt incx,
                                  cl_mem y, ViennaCLInt offy, ViennaCLInt incy,
                                  float c, float s);
ViennaCLStatus ViennaCLOpenCLDrot(ViennaCLBackend backend, ViennaCLInt n,
                                  cl_mem x, ViennaCLInt offx, ViennaCLInt incx,
                                  cl_mem y, ViennaCLInt offy, ViennaCLInt incy,
                                  double c, double s);
#endif

ViennaCLStatus ViennaCLHostSrot(ViennaCLBackend backend, ViennaCLInt n,
                                float *x, ViennaCLInt offx, ViennaCLInt incx,
                                float *y, ViennaCLInt offy, ViennaCLInt incy,
                                float c, float s);
ViennaCLStatus ViennaCLHostDrot(ViennaCLBackend backend, ViennaCLInt n,
                                double *x, ViennaCLInt offx, ViennaCLInt incx,
                                double *y, ViennaCLInt offy, ViennaCLInt incy,
                                double c, double s);



// xSCAL

ViennaCLStatus ViennaCLscal(ViennaCLHostScalar alpha, ViennaCLVector x);

ViennaCLStatus ViennaCLCUDASscal(ViennaCLBackend backend, ViennaCLInt n,
                                 float alpha,
                                 float *x, ViennaCLInt offx, ViennaCLInt incx);
ViennaCLStatus ViennaCLCUDADscal(ViennaCLBackend backend, ViennaCLInt n,
                                 double alpha,
                                 double *x, ViennaCLInt offx, ViennaCLInt incx);

#ifdef VIENNACL_WITH_OPENCL
ViennaCLStatus ViennaCLOpenCLSscal(ViennaCLBackend backend, ViennaCLInt n,
                                   float alpha,
                                   cl_mem x, ViennaCLInt offx, ViennaCLInt incx);
ViennaCLStatus ViennaCLOpenCLDscal(ViennaCLBackend backend, ViennaCLInt n,
                                   double alpha,
                                   cl_mem x, ViennaCLInt offx, ViennaCLInt incx);
#endif

ViennaCLStatus ViennaCLHostSscal(ViennaCLBackend backend, ViennaCLInt n,
                                 float alpha,
                                 float *x, ViennaCLInt offx, ViennaCLInt incx);
ViennaCLStatus ViennaCLHostDscal(ViennaCLBackend backend, ViennaCLInt n,
                                 double alpha,
                                 double *x, ViennaCLInt offx, ViennaCLInt incx);


// xSWAP

ViennaCLStatus ViennaCLswap(ViennaCLVector x, ViennaCLVector y);

ViennaCLStatus ViennaCLCUDASswap(ViennaCLBackend backend, ViennaCLInt n,
                                 float *x, ViennaCLInt offx, ViennaCLInt incx,
                                 float *y, ViennaCLInt offy, ViennaCLInt incy);
ViennaCLStatus ViennaCLCUDADswap(ViennaCLBackend backend, ViennaCLInt n,
                                 double *x, ViennaCLInt offx, ViennaCLInt incx,
                                 double *y, ViennaCLInt offy, ViennaCLInt incy);

#ifdef VIENNACL_WITH_OPENCL
ViennaCLStatus ViennaCLOpenCLSswap(ViennaCLBackend backend, ViennaCLInt n,
                                   cl_mem x, ViennaCLInt offx, ViennaCLInt incx,
                                   cl_mem y, ViennaCLInt offy, ViennaCLInt incy);
ViennaCLStatus ViennaCLOpenCLDswap(ViennaCLBackend backend, ViennaCLInt n,
                                   cl_mem x, ViennaCLInt offx, ViennaCLInt incx,
                                   cl_mem y, ViennaCLInt offy, ViennaCLInt incy);
#endif

ViennaCLStatus ViennaCLHostSswap(ViennaCLBackend backend, ViennaCLInt n,
                                 float *x, ViennaCLInt offx, ViennaCLInt incx,
                                 float *y, ViennaCLInt offy, ViennaCLInt incy);
ViennaCLStatus ViennaCLHostDswap(ViennaCLBackend backend, ViennaCLInt n,
                                 double *x, ViennaCLInt offx, ViennaCLInt incx,
                                 double *y, ViennaCLInt offy, ViennaCLInt incy);



/******************** BLAS Level 2 ***********************/

// xGEMV: y <- alpha * Ax + beta * y

ViennaCLStatus ViennaCLgemv(ViennaCLHostScalar alpha, ViennaCLMatrix A, ViennaCLVector x, ViennaCLHostScalar beta, ViennaCLVector y);

ViennaCLStatus ViennaCLCUDASgemv(ViennaCLBackend backend,
                                 ViennaCLOrder order, ViennaCLTranspose transA,
                                 ViennaCLInt m, ViennaCLInt n, float alpha, float *A, ViennaCLInt offA_row, ViennaCLInt offA_col, ViennaCLInt incA_row, ViennaCLInt incA_col, ViennaCLInt lda,
                                 float *x, ViennaCLInt offx, ViennaCLInt incx,
                                 float beta,
                                 float *y, ViennaCLInt offy, ViennaCLInt incy);
ViennaCLStatus ViennaCLCUDADgemv(ViennaCLBackend backend,
                                 ViennaCLOrder order, ViennaCLTranspose transA,
                                 ViennaCLInt m, ViennaCLInt n, double alpha, double *A, ViennaCLInt offA_row, ViennaCLInt offA_col, ViennaCLInt incA_row, ViennaCLInt incA_col, ViennaCLInt lda,
                                 double *x, ViennaCLInt offx, ViennaCLInt incx,
                                 double beta,
                                 double *y, ViennaCLInt offy, ViennaCLInt incy);

#ifdef VIENNACL_WITH_OPENCL
ViennaCLStatus ViennaCLOpenCLSgemv(ViennaCLBackend backend,
                                   ViennaCLOrder order, ViennaCLTranspose transA,
                                   ViennaCLInt m, ViennaCLInt n, float alpha, cl_mem A, ViennaCLInt offA_row, ViennaCLInt offA_col, ViennaCLInt incA_row, ViennaCLInt incA_col, ViennaCLInt lda,
                                   cl_mem x, ViennaCLInt offx, ViennaCLInt incx,
                                   float beta,
                                   cl_mem y, ViennaCLInt offy, ViennaCLInt incy);
ViennaCLStatus ViennaCLOpenCLDgemv(ViennaCLBackend backend,
                                   ViennaCLOrder order, ViennaCLTranspose transA,
                                   ViennaCLInt m, ViennaCLInt n, double alpha, cl_mem A, ViennaCLInt offA_row, ViennaCLInt offA_col, ViennaCLInt incA_row, ViennaCLInt incA_col, ViennaCLInt lda,
                                   cl_mem x, ViennaCLInt offx, ViennaCLInt incx,
                                   double beta,
                                   cl_mem y, ViennaCLInt offy, ViennaCLInt incy);
#endif

ViennaCLStatus ViennaCLHostSgemv(ViennaCLBackend backend,
                                 ViennaCLOrder order, ViennaCLTranspose transA,
                                 ViennaCLInt m, ViennaCLInt n, float alpha, float *A, ViennaCLInt offA_row, ViennaCLInt offA_col, ViennaCLInt incA_row, ViennaCLInt incA_col, ViennaCLInt lda,
                                 float *x, ViennaCLInt offx, ViennaCLInt incx,
                                 float beta,
                                 float *y, ViennaCLInt offy, ViennaCLInt incy);
ViennaCLStatus ViennaCLHostDgemv(ViennaCLBackend backend,
                                 ViennaCLOrder order, ViennaCLTranspose transA,
                                 ViennaCLInt m, ViennaCLInt n, double alpha, double *A, ViennaCLInt offA_row, ViennaCLInt offA_col, ViennaCLInt incA_row, ViennaCLInt incA_col, ViennaCLInt lda,
                                 double *x, ViennaCLInt offx, ViennaCLInt incx,
                                 double beta,
                                 double *y, ViennaCLInt offy, ViennaCLInt incy);

// xTRSV: Ax <- x

ViennaCLStatus ViennaCLtrsv(ViennaCLMatrix A, ViennaCLVector x, ViennaCLUplo uplo);

ViennaCLStatus ViennaCLCUDAStrsv(ViennaCLBackend backend,
                                 ViennaCLUplo uplo, ViennaCLOrder order, ViennaCLTranspose transA, ViennaCLDiag diag,
                                 ViennaCLInt n, float *A, ViennaCLInt offA_row, ViennaCLInt offA_col, ViennaCLInt incA_row, ViennaCLInt incA_col, ViennaCLInt lda,
                                 float *x, ViennaCLInt offx, ViennaCLInt incx);
ViennaCLStatus ViennaCLCUDADtrsv(ViennaCLBackend backend,
                                 ViennaCLUplo uplo, ViennaCLOrder order, ViennaCLTranspose transA, ViennaCLDiag diag,
                                 ViennaCLInt n, double *A, ViennaCLInt offA_row, ViennaCLInt offA_col, ViennaCLInt incA_row, ViennaCLInt incA_col, ViennaCLInt lda,
                                 double *x, ViennaCLInt offx, ViennaCLInt incx);

#ifdef VIENNACL_WITH_OPENCL
ViennaCLStatus ViennaCLOpenCLStrsv(ViennaCLBackend backend,
                                   ViennaCLUplo uplo, ViennaCLOrder order, ViennaCLTranspose transA, ViennaCLDiag diag,
                                   ViennaCLInt n, cl_mem A, ViennaCLInt offA_row, ViennaCLInt offA_col, ViennaCLInt incA_row, ViennaCLInt incA_col, ViennaCLInt lda,
                                   cl_mem x, ViennaCLInt offx, ViennaCLInt incx);
ViennaCLStatus ViennaCLOpenCLDtrsv(ViennaCLBackend backend,
                                   ViennaCLUplo uplo, ViennaCLOrder order, ViennaCLTranspose transA, ViennaCLDiag diag,
                                   ViennaCLInt n, cl_mem A, ViennaCLInt offA_row, ViennaCLInt offA_col, ViennaCLInt incA_row, ViennaCLInt incA_col, ViennaCLInt lda,
                                   cl_mem x, ViennaCLInt offx, ViennaCLInt incx);
#endif

ViennaCLStatus ViennaCLHostStrsv(ViennaCLBackend backend,
                                 ViennaCLUplo uplo, ViennaCLOrder order, ViennaCLTranspose transA, ViennaCLDiag diag,
                                 ViennaCLInt n, float *A, ViennaCLInt offA_row, ViennaCLInt offA_col, ViennaCLInt incA_row, ViennaCLInt incA_col, ViennaCLInt lda,
                                 float *x, ViennaCLInt offx, ViennaCLInt incx);
ViennaCLStatus ViennaCLHostDtrsv(ViennaCLBackend backend,
                                 ViennaCLUplo uplo, ViennaCLOrder order, ViennaCLTranspose transA, ViennaCLDiag diag,
                                 ViennaCLInt n, double *A, ViennaCLInt offA_row, ViennaCLInt offA_col, ViennaCLInt incA_row, ViennaCLInt incA_col, ViennaCLInt lda,
                                 double *x, ViennaCLInt offx, ViennaCLInt incx);


// xGER: A <- alpha * x * y + A

ViennaCLStatus ViennaCLger(ViennaCLHostScalar alpha, ViennaCLVector x, ViennaCLVector y, ViennaCLMatrix A);

ViennaCLStatus ViennaCLCUDASger(ViennaCLBackend backend,
                                ViennaCLOrder order,
                                ViennaCLInt m, ViennaCLInt n,
                                float alpha,
                                float *x, ViennaCLInt offx, ViennaCLInt incx,
                                float *y, ViennaCLInt offy, ViennaCLInt incy,
                                float *A, ViennaCLInt offA_row, ViennaCLInt offA_col, ViennaCLInt incA_row, ViennaCLInt incA_col, ViennaCLInt lda);
ViennaCLStatus ViennaCLCUDADger(ViennaCLBackend backend,
                                ViennaCLOrder order,
                                ViennaCLInt m,  ViennaCLInt n,
                                double alpha,
                                double *x, ViennaCLInt offx, ViennaCLInt incx,
                                double *y, ViennaCLInt offy, ViennaCLInt incy,
                                double *A, ViennaCLInt offA_row, ViennaCLInt offA_col, ViennaCLInt incA_row, ViennaCLInt incA_col, ViennaCLInt lda);

#ifdef VIENNACL_WITH_OPENCL
ViennaCLStatus ViennaCLOpenCLSger(ViennaCLBackend backend,
                                  ViennaCLOrder order,
                                  ViennaCLInt m, ViennaCLInt n,
                                  float alpha,
                                  cl_mem x, ViennaCLInt offx, ViennaCLInt incx,
                                  cl_mem y, ViennaCLInt offy, ViennaCLInt incy,
                                  cl_mem A, ViennaCLInt offA_row, ViennaCLInt offA_col, ViennaCLInt incA_row, ViennaCLInt incA_col, ViennaCLInt lda);
ViennaCLStatus ViennaCLOpenCLDger(ViennaCLBackend backend,
                                  ViennaCLOrder order,
                                  ViennaCLInt m, ViennaCLInt n,
                                  cl_mem x, ViennaCLInt offx, ViennaCLInt incx,
                                  cl_mem y, ViennaCLInt offy, ViennaCLInt incy,
                                  double alpha, cl_mem A, ViennaCLInt offA_row, ViennaCLInt offA_col, ViennaCLInt incA_row, ViennaCLInt incA_col, ViennaCLInt lda);
#endif

ViennaCLStatus ViennaCLHostSger(ViennaCLBackend backend,
                                ViennaCLOrder order,
                                ViennaCLInt m, ViennaCLInt n,
                                float alpha,
                                float *x, ViennaCLInt offx, ViennaCLInt incx,
                                float *y, ViennaCLInt offy, ViennaCLInt incy,
                                float *A, ViennaCLInt offA_row, ViennaCLInt offA_col, ViennaCLInt incA_row, ViennaCLInt incA_col, ViennaCLInt lda);
ViennaCLStatus ViennaCLHostDger(ViennaCLBackend backend,
                                ViennaCLOrder order,
                                ViennaCLInt m, ViennaCLInt n,
                                double alpha,
                                double *x, ViennaCLInt offx, ViennaCLInt incx,
                                double *y, ViennaCLInt offy, ViennaCLInt incy,
                                double *A, ViennaCLInt offA_row, ViennaCLInt offA_col, ViennaCLInt incA_row, ViennaCLInt incA_col, ViennaCLInt lda);



/******************** BLAS Level 3 ***********************/

// xGEMM: C <- alpha * AB + beta * C

ViennaCLStatus ViennaCLgemm(ViennaCLHostScalar alpha, ViennaCLMatrix A, ViennaCLMatrix B, ViennaCLHostScalar beta, ViennaCLMatrix C);

ViennaCLStatus ViennaCLCUDASgemm(ViennaCLBackend backend,
                                 ViennaCLOrder orderA, ViennaCLTranspose transA,
                                 ViennaCLOrder orderB, ViennaCLTranspose transB,
                                 ViennaCLOrder orderC,
                                 ViennaCLInt m, ViennaCLInt n, ViennaCLInt k,
                                 float alpha,
                                 float *A, ViennaCLInt offA_row, ViennaCLInt offA_col, ViennaCLInt incA_row, ViennaCLInt incA_col, ViennaCLInt lda,
                                 float *B, ViennaCLInt offB_row, ViennaCLInt offB_col, ViennaCLInt incB_row, ViennaCLInt incB_col, ViennaCLInt ldb,
                                 float beta,
                                 float *C, ViennaCLInt offC_row, ViennaCLInt offC_col, ViennaCLInt incC_row, ViennaCLInt incC_col, ViennaCLInt ldc);
ViennaCLStatus ViennaCLCUDADgemm(ViennaCLBackend backend,
                                 ViennaCLOrder orderA, ViennaCLTranspose transA,
                                 ViennaCLOrder orderB, ViennaCLTranspose transB,
                                 ViennaCLOrder orderC,
                                 ViennaCLInt m, ViennaCLInt n, ViennaCLInt k,
                                 double alpha,
                                 double *A, ViennaCLInt offA_row, ViennaCLInt offA_col, ViennaCLInt incA_row, ViennaCLInt incA_col, ViennaCLInt lda,
                                 double *B, ViennaCLInt offB_row, ViennaCLInt offB_col, ViennaCLInt incB_row, ViennaCLInt incB_col, ViennaCLInt ldb,
                                 double beta,
                                 double *C, ViennaCLInt offC_row, ViennaCLInt offC_col, ViennaCLInt incC_row, ViennaCLInt incC_col, ViennaCLInt ldc);

#ifdef VIENNACL_WITH_OPENCL
ViennaCLStatus ViennaCLOpenCLSgemm(ViennaCLBackend backend,
                                   ViennaCLOrder orderA, ViennaCLTranspose transA,
                                   ViennaCLOrder orderB, ViennaCLTranspose transB,
                                   ViennaCLOrder orderC,
                                   ViennaCLInt m, ViennaCLInt n, ViennaCLInt k,
                                   float alpha,
                                   cl_mem A, ViennaCLInt offA_row, ViennaCLInt offA_col, ViennaCLInt incA_row, ViennaCLInt incA_col, ViennaCLInt lda,
                                   cl_mem B, ViennaCLInt offB_row, ViennaCLInt offB_col, ViennaCLInt incB_row, ViennaCLInt incB_col, ViennaCLInt ldb,
                                   float beta,
                                   cl_mem C, ViennaCLInt offC_row, ViennaCLInt offC_col, ViennaCLInt incC_row, ViennaCLInt incC_col, ViennaCLInt ldc);
ViennaCLStatus ViennaCLOpenCLDgemm(ViennaCLBackend backend,
                                   ViennaCLOrder orderA, ViennaCLTranspose transA,
                                   ViennaCLOrder orderB, ViennaCLTranspose transB,
                                   ViennaCLOrder orderC,
                                   ViennaCLInt m, ViennaCLInt n, ViennaCLInt k,
                                   double alpha,
                                   cl_mem A, ViennaCLInt offA_row, ViennaCLInt offA_col, ViennaCLInt incA_row, ViennaCLInt incA_col, ViennaCLInt lda,
                                   cl_mem B, ViennaCLInt offB_row, ViennaCLInt offB_col, ViennaCLInt incB_row, ViennaCLInt incB_col, ViennaCLInt ldb,
                                   double beta,
                                   cl_mem C, ViennaCLInt offC_row, ViennaCLInt offC_col, ViennaCLInt incC_row, ViennaCLInt incC_col, ViennaCLInt ldc);
#endif

ViennaCLStatus ViennaCLHostSgemm(ViennaCLBackend backend,
                                 ViennaCLOrder orderA, ViennaCLTranspose transA,
                                 ViennaCLOrder orderB, ViennaCLTranspose transB,
                                 ViennaCLOrder orderC,
                                 ViennaCLInt m, ViennaCLInt n, ViennaCLInt k,
                                 float alpha,
                                 float *A, ViennaCLInt offA_row, ViennaCLInt offA_col, ViennaCLInt incA_row, ViennaCLInt incA_col, ViennaCLInt lda,
                                 float *B, ViennaCLInt offB_row, ViennaCLInt offB_col, ViennaCLInt incB_row, ViennaCLInt incB_col, ViennaCLInt ldb,
                                 float beta,
                                 float *C, ViennaCLInt offC_row, ViennaCLInt offC_col, ViennaCLInt incC_row, ViennaCLInt incC_col, ViennaCLInt ldc);
ViennaCLStatus ViennaCLHostDgemm(ViennaCLBackend backend,
                                 ViennaCLOrder orderA, ViennaCLTranspose transA,
                                 ViennaCLOrder orderB, ViennaCLTranspose transB,
                                 ViennaCLOrder orderC,
                                 ViennaCLInt m, ViennaCLInt n, ViennaCLInt k,
                                 double alpha,
                                 double *A, ViennaCLInt offA_row, ViennaCLInt offA_col, ViennaCLInt incA_row, ViennaCLInt incA_col, ViennaCLInt lda,
                                 double *B, ViennaCLInt offB_row, ViennaCLInt offB_col, ViennaCLInt incB_row, ViennaCLInt incB_col, ViennaCLInt ldb,
                                 double beta,
                                 double *C, ViennaCLInt offC_row, ViennaCLInt offC_col, ViennaCLInt incC_row, ViennaCLInt incC_col, ViennaCLInt ldc);


#ifdef __cplusplus
}
#endif


#endif
