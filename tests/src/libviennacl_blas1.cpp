/* =========================================================================
   Copyright (c) 2010-2016, Institute for Microelectronics,
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


/** \file tests/src/libviennacl_blas1.cpp  Testing the BLAS level 1 routines in the ViennaCL BLAS-like shared library
*   \test Testing the BLAS level 1 routines in the ViennaCL BLAS-like shared library
**/


// include necessary system headers
#include <iostream>
#include <vector>
#include <cmath>

// Some helper functions for this tutorial:
#include "viennacl.hpp"

#include "viennacl/vector.hpp"

template<typename ScalarType>
ScalarType diff(ScalarType const & s1, ScalarType const & s2)
{
   if (s1 > s2 || s1 < s2)
      return (s1 - s2) / std::max(static_cast<ScalarType>(std::fabs(static_cast<double>(s1))),
                                  static_cast<ScalarType>(std::fabs(static_cast<double>(s2))));
   return ScalarType(0);
}

template<typename ScalarType, typename ViennaCLVectorType>
ScalarType diff(std::vector<ScalarType> const & v1, ViennaCLVectorType const & vcl_vec)
{
   std::vector<ScalarType> v2_cpu(vcl_vec.size());
   viennacl::backend::finish();
   viennacl::copy(vcl_vec, v2_cpu);

   ScalarType inf_norm = 0;
   for (unsigned int i=0;i<v1.size(); ++i)
   {
      if ( std::max( std::fabs(v2_cpu[i]), std::fabs(v1[i]) ) > 0 )
         v2_cpu[i] = std::fabs(v2_cpu[i] - v1[i]) / std::max( std::fabs(v2_cpu[i]), std::fabs(v1[i]) );
      else
         v2_cpu[i] = 0.0;

      if (v2_cpu[i] > inf_norm)
        inf_norm = v2_cpu[i];
   }

   return inf_norm;
}

template<typename T, typename U, typename EpsilonT>
void check(T const & t, U const & u, EpsilonT eps)
{
  EpsilonT rel_error = std::fabs(static_cast<EpsilonT>(diff(t,u)));
  if (rel_error > eps)
  {
    std::cerr << "Relative error: " << rel_error << std::endl;
    std::cerr << "Aborting!" << std::endl;
    exit(EXIT_FAILURE);
  }
  std::cout << "SUCCESS ";
}

int main()
{
  std::size_t size  = 10; // at least 7
  float  eps_float  = 1e-5f;
  double eps_double = 1e-12;

  float  ref_float_alpha;
  double ref_double_alpha;

  std::vector<float> ref_float_x(size, 1.0f);
  std::vector<float> ref_float_y(size, 2.0f);

  std::vector<double> ref_double_x(size, 1.0);
  std::vector<double> ref_double_y(size, 2.0);

  ViennaCLBackend my_backend;
  ViennaCLBackendCreate(&my_backend);

  // Host setup
  float host_float_alpha = 0;
  viennacl::vector<float> host_float_x = viennacl::scalar_vector<float>(size, 1.0f, viennacl::context(viennacl::MAIN_MEMORY));
  viennacl::vector<float> host_float_y = viennacl::scalar_vector<float>(size, 2.0f, viennacl::context(viennacl::MAIN_MEMORY));

  double host_double_alpha = 0;
  viennacl::vector<double> host_double_x = viennacl::scalar_vector<double>(size, 1.0, viennacl::context(viennacl::MAIN_MEMORY));
  viennacl::vector<double> host_double_y = viennacl::scalar_vector<double>(size, 2.0, viennacl::context(viennacl::MAIN_MEMORY));

  // CUDA setup
#ifdef VIENNACL_WITH_CUDA
  float cuda_float_alpha = 0;
  viennacl::vector<float> cuda_float_x = viennacl::scalar_vector<float>(size, 1.0f, viennacl::context(viennacl::CUDA_MEMORY));
  viennacl::vector<float> cuda_float_y = viennacl::scalar_vector<float>(size, 2.0f, viennacl::context(viennacl::CUDA_MEMORY));

  double cuda_double_alpha = 0;
  viennacl::vector<double> cuda_double_x = viennacl::scalar_vector<double>(size, 1.0, viennacl::context(viennacl::CUDA_MEMORY));
  viennacl::vector<double> cuda_double_y = viennacl::scalar_vector<double>(size, 2.0, viennacl::context(viennacl::CUDA_MEMORY));
#endif

  // OpenCL setup
#ifdef VIENNACL_WITH_OPENCL
  ViennaCLInt context_id = 0;
  float opencl_float_alpha = 0;
  viennacl::vector<float> opencl_float_x = viennacl::scalar_vector<float>(size, 1.0f, viennacl::context(viennacl::ocl::get_context(context_id)));
  viennacl::vector<float> opencl_float_y = viennacl::scalar_vector<float>(size, 2.0f, viennacl::context(viennacl::ocl::get_context(context_id)));

  double opencl_double_alpha = 0;
  viennacl::vector<double> *opencl_double_x = NULL;
  viennacl::vector<double> *opencl_double_y = NULL;
  if ( viennacl::ocl::current_device().double_support() )
  {
    opencl_double_x = new viennacl::vector<double>(viennacl::scalar_vector<double>(size, 1.0, viennacl::context(viennacl::ocl::get_context(context_id))));
    opencl_double_y = new viennacl::vector<double>(viennacl::scalar_vector<double>(size, 2.0, viennacl::context(viennacl::ocl::get_context(context_id))));
  }

  ViennaCLBackendSetOpenCLContextID(my_backend, context_id);
#endif

  // consistency checks:
  check(ref_float_x, host_float_x, eps_float);
  check(ref_float_y, host_float_y, eps_float);
  check(ref_double_x, host_double_x, eps_double);
  check(ref_double_y, host_double_y, eps_double);
#ifdef VIENNACL_WITH_CUDA
  check(ref_float_x, cuda_float_x, eps_float);
  check(ref_float_y, cuda_float_y, eps_float);
  check(ref_double_x, cuda_double_x, eps_double);
  check(ref_double_y, cuda_double_y, eps_double);
#endif
#ifdef VIENNACL_WITH_OPENCL
  check(ref_float_x, opencl_float_x, eps_float);
  check(ref_float_y, opencl_float_y, eps_float);
  if ( viennacl::ocl::current_device().double_support() )
  {
    check(ref_double_x, *opencl_double_x, eps_double);
    check(ref_double_y, *opencl_double_y, eps_double);
  }
#endif

  // ASUM
  std::cout << std::endl << "-- Testing xASUM...";
  ref_float_alpha  = 0;
  ref_double_alpha = 0;
  for (std::size_t i=0; i<size/4; ++i)
  {
    ref_float_alpha  += std::fabs(ref_float_x[2 + 3*i]);
    ref_double_alpha += std::fabs(ref_double_x[2 + 3*i]);
  }

  std::cout << std::endl << "Host: ";
  ViennaCLHostSasum(my_backend, ViennaCLInt(size/4),
                    &host_float_alpha,
                    viennacl::linalg::host_based::detail::extract_raw_pointer<float>(host_float_x), 2, 3);
  check(ref_float_alpha, host_float_alpha, eps_float);
  ViennaCLHostDasum(my_backend, ViennaCLInt(size/4),
                    &host_double_alpha,
                    viennacl::linalg::host_based::detail::extract_raw_pointer<double>(host_double_x), 2, 3);
  check(ref_double_alpha, host_double_alpha, eps_double);


#ifdef VIENNACL_WITH_CUDA
  std::cout << std::endl << "CUDA: ";
  ViennaCLCUDASasum(my_backend, ViennaCLInt(size/4),
                    &cuda_float_alpha,
                    viennacl::cuda_arg(cuda_float_x), 2, 3);
  check(ref_float_alpha, cuda_float_alpha, eps_float);
  ViennaCLCUDADasum(my_backend, ViennaCLInt(size/4),
                    &cuda_double_alpha,
                    viennacl::cuda_arg(cuda_double_x), 2, 3);
  check(ref_double_alpha, cuda_double_alpha, eps_double);
#endif

#ifdef VIENNACL_WITH_OPENCL
  std::cout << std::endl << "OpenCL: ";
  ViennaCLOpenCLSasum(my_backend, ViennaCLInt(size/4),
                      &opencl_float_alpha,
                      viennacl::traits::opencl_handle(opencl_float_x).get(), 2, 3);
  check(ref_float_alpha, opencl_float_alpha, eps_float);
  if ( viennacl::ocl::current_device().double_support() )
  {
    ViennaCLOpenCLDasum(my_backend, ViennaCLInt(size/4),
                        &opencl_double_alpha,
                        viennacl::traits::opencl_handle(*opencl_double_x).get(), 2, 3);
    check(ref_double_alpha, opencl_double_alpha, eps_double);
  }
#endif



  // AXPY
  std::cout << std::endl << "-- Testing xAXPY...";
  for (std::size_t i=0; i<size/3; ++i)
  {
    ref_float_y[1 + 2*i]  += 2.0f * ref_float_x[0 + 2*i];
    ref_double_y[1 + 2*i] += 2.0  * ref_double_x[0 + 2*i];
  }

  std::cout << std::endl << "Host: ";
  ViennaCLHostSaxpy(my_backend, ViennaCLInt(size/3),
                    2.0f,
                    viennacl::linalg::host_based::detail::extract_raw_pointer<float>(host_float_x), 0, 2,
                    viennacl::linalg::host_based::detail::extract_raw_pointer<float>(host_float_y), 1, 2);
  check(ref_float_x, host_float_x, eps_float);
  check(ref_float_y, host_float_y, eps_float);
  ViennaCLHostDaxpy(my_backend, ViennaCLInt(size/3),
                    2.0,
                    viennacl::linalg::host_based::detail::extract_raw_pointer<double>(host_double_x), 0, 2,
                    viennacl::linalg::host_based::detail::extract_raw_pointer<double>(host_double_y), 1, 2);
  check(ref_double_x, host_double_x, eps_double);
  check(ref_double_y, host_double_y, eps_double);


#ifdef VIENNACL_WITH_CUDA
  std::cout << std::endl << "CUDA: ";
  ViennaCLCUDASaxpy(my_backend, ViennaCLInt(size/3),
                    2.0f,
                    viennacl::cuda_arg(cuda_float_x), 0, 2,
                    viennacl::cuda_arg(cuda_float_y), 1, 2);
  check(ref_float_x, cuda_float_x, eps_float);
  check(ref_float_y, cuda_float_y, eps_float);
  ViennaCLCUDADaxpy(my_backend, ViennaCLInt(size/3),
                    2.0,
                    viennacl::cuda_arg(cuda_double_x), 0, 2,
                    viennacl::cuda_arg(cuda_double_y), 1, 2);
  check(ref_double_x, cuda_double_x, eps_double);
  check(ref_double_y, cuda_double_y, eps_double);
#endif

#ifdef VIENNACL_WITH_OPENCL
  std::cout << std::endl << "OpenCL: ";
  ViennaCLOpenCLSaxpy(my_backend, ViennaCLInt(size/3),
                      2.0f,
                      viennacl::traits::opencl_handle(opencl_float_x).get(), 0, 2,
                      viennacl::traits::opencl_handle(opencl_float_y).get(), 1, 2);
  check(ref_float_x, opencl_float_x, eps_float);
  check(ref_float_y, opencl_float_y, eps_float);
  if ( viennacl::ocl::current_device().double_support() )
  {
    ViennaCLOpenCLDaxpy(my_backend, ViennaCLInt(size/3),
                        2.0,
                        viennacl::traits::opencl_handle(*opencl_double_x).get(), 0, 2,
                        viennacl::traits::opencl_handle(*opencl_double_y).get(), 1, 2);
    check(ref_double_x, *opencl_double_x, eps_double);
    check(ref_double_y, *opencl_double_y, eps_double);
  }
#endif



  // COPY
  std::cout << std::endl << "-- Testing xCOPY...";
  for (std::size_t i=0; i<size/3; ++i)
  {
    ref_float_y[0 + 2*i]  = ref_float_x[1 + 2*i];
    ref_double_y[0 + 2*i] = ref_double_x[1 + 2*i];
  }

  std::cout << std::endl << "Host: ";
  ViennaCLHostScopy(my_backend, ViennaCLInt(size/3),
                    viennacl::linalg::host_based::detail::extract_raw_pointer<float>(host_float_x), 1, 2,
                    viennacl::linalg::host_based::detail::extract_raw_pointer<float>(host_float_y), 0, 2);
  check(ref_float_x, host_float_x, eps_float);
  check(ref_float_y, host_float_y, eps_float);
  ViennaCLHostDcopy(my_backend, ViennaCLInt(size/3),
                    viennacl::linalg::host_based::detail::extract_raw_pointer<double>(host_double_x), 1, 2,
                    viennacl::linalg::host_based::detail::extract_raw_pointer<double>(host_double_y), 0, 2);
  check(ref_double_x, host_double_x, eps_double);
  check(ref_double_y, host_double_y, eps_double);


#ifdef VIENNACL_WITH_CUDA
  std::cout << std::endl << "CUDA: ";
  ViennaCLCUDAScopy(my_backend, ViennaCLInt(size/3),
                    viennacl::cuda_arg(cuda_float_x), 1, 2,
                    viennacl::cuda_arg(cuda_float_y), 0, 2);
  check(ref_float_x, cuda_float_x, eps_float);
  check(ref_float_y, cuda_float_y, eps_float);
  ViennaCLCUDADcopy(my_backend, ViennaCLInt(size/3),
                    viennacl::cuda_arg(cuda_double_x), 1, 2,
                    viennacl::cuda_arg(cuda_double_y), 0, 2);
  check(ref_double_x, cuda_double_x, eps_double);
  check(ref_double_y, cuda_double_y, eps_double);
#endif

#ifdef VIENNACL_WITH_OPENCL
  std::cout << std::endl << "OpenCL: ";
  ViennaCLOpenCLScopy(my_backend, ViennaCLInt(size/3),
                      viennacl::traits::opencl_handle(opencl_float_x).get(), 1, 2,
                      viennacl::traits::opencl_handle(opencl_float_y).get(), 0, 2);
  check(ref_float_x, opencl_float_x, eps_float);
  check(ref_float_y, opencl_float_y, eps_float);
  if ( viennacl::ocl::current_device().double_support() )
  {
    ViennaCLOpenCLDcopy(my_backend, ViennaCLInt(size/3),
                        viennacl::traits::opencl_handle(*opencl_double_x).get(), 1, 2,
                        viennacl::traits::opencl_handle(*opencl_double_y).get(), 0, 2);
    check(ref_double_x, *opencl_double_x, eps_double);
    check(ref_double_y, *opencl_double_y, eps_double);
  }
#endif



  // DOT
  std::cout << std::endl << "-- Testing xDOT...";
  ref_float_alpha  = 0;
  ref_double_alpha = 0;
  for (std::size_t i=0; i<size/2; ++i)
  {
    ref_float_alpha  += ref_float_y[3 + i]  * ref_float_x[2 + i];
    ref_double_alpha += ref_double_y[3 + i] * ref_double_x[2 + i];
  }

  std::cout << std::endl << "Host: ";
  ViennaCLHostSdot(my_backend, ViennaCLInt(size/2),
                   &host_float_alpha,
                   viennacl::linalg::host_based::detail::extract_raw_pointer<float>(host_float_x), 2, 1,
                   viennacl::linalg::host_based::detail::extract_raw_pointer<float>(host_float_y), 3, 1);
  check(ref_float_alpha, host_float_alpha, eps_float);
  ViennaCLHostDdot(my_backend, ViennaCLInt(size/2),
                   &host_double_alpha,
                   viennacl::linalg::host_based::detail::extract_raw_pointer<double>(host_double_x), 2, 1,
                   viennacl::linalg::host_based::detail::extract_raw_pointer<double>(host_double_y), 3, 1);
  check(ref_double_alpha, host_double_alpha, eps_double);


#ifdef VIENNACL_WITH_CUDA
  std::cout << std::endl << "CUDA: ";
  ViennaCLCUDASdot(my_backend, ViennaCLInt(size/2),
                   &cuda_float_alpha,
                   viennacl::cuda_arg(cuda_float_x), 2, 1,
                   viennacl::cuda_arg(cuda_float_y), 3, 1);
  check(ref_float_alpha, cuda_float_alpha, eps_float);
  ViennaCLCUDADdot(my_backend, ViennaCLInt(size/2),
                   &cuda_double_alpha,
                   viennacl::cuda_arg(cuda_double_x), 2, 1,
                   viennacl::cuda_arg(cuda_double_y), 3, 1);
  check(ref_double_alpha, cuda_double_alpha, eps_double);
#endif

#ifdef VIENNACL_WITH_OPENCL
  std::cout << std::endl << "OpenCL: ";
  ViennaCLOpenCLSdot(my_backend, ViennaCLInt(size/2),
                     &opencl_float_alpha,
                     viennacl::traits::opencl_handle(opencl_float_x).get(), 2, 1,
                     viennacl::traits::opencl_handle(opencl_float_y).get(), 3, 1);
  check(ref_float_alpha, opencl_float_alpha, eps_float);
  if ( viennacl::ocl::current_device().double_support() )
  {
    ViennaCLOpenCLDdot(my_backend, ViennaCLInt(size/2),
                       &opencl_double_alpha,
                       viennacl::traits::opencl_handle(*opencl_double_x).get(), 2, 1,
                       viennacl::traits::opencl_handle(*opencl_double_y).get(), 3, 1);
    check(ref_double_alpha, opencl_double_alpha, eps_double);
  }
#endif



  // NRM2
  std::cout << std::endl << "-- Testing xNRM2...";
  ref_float_alpha  = 0;
  ref_double_alpha = 0;
  for (std::size_t i=0; i<size/3; ++i)
  {
    ref_float_alpha  += ref_float_x[1 + 2*i]  * ref_float_x[1 + 2*i];
    ref_double_alpha += ref_double_x[1 + 2*i] * ref_double_x[1 + 2*i];
  }
  ref_float_alpha = std::sqrt(ref_float_alpha);
  ref_double_alpha = std::sqrt(ref_double_alpha);

  std::cout << std::endl << "Host: ";
  ViennaCLHostSnrm2(my_backend, ViennaCLInt(size/3),
                    &host_float_alpha,
                    viennacl::linalg::host_based::detail::extract_raw_pointer<float>(host_float_x), 1, 2);
  check(ref_float_alpha, host_float_alpha, eps_float);
  ViennaCLHostDnrm2(my_backend, ViennaCLInt(size/3),
                    &host_double_alpha,
                    viennacl::linalg::host_based::detail::extract_raw_pointer<double>(host_double_x), 1, 2);
  check(ref_double_alpha, host_double_alpha, eps_double);


#ifdef VIENNACL_WITH_CUDA
  std::cout << std::endl << "CUDA: ";
  ViennaCLCUDASnrm2(my_backend, ViennaCLInt(size/3),
                    &cuda_float_alpha,
                    viennacl::cuda_arg(cuda_float_x), 1, 2);
  check(ref_float_alpha, cuda_float_alpha, eps_float);
  ViennaCLCUDADnrm2(my_backend, ViennaCLInt(size/3),
                    &cuda_double_alpha,
                    viennacl::cuda_arg(cuda_double_x), 1, 2);
  check(ref_double_alpha, cuda_double_alpha, eps_double);
#endif

#ifdef VIENNACL_WITH_OPENCL
  std::cout << std::endl << "OpenCL: ";
  ViennaCLOpenCLSnrm2(my_backend, ViennaCLInt(size/3),
                      &opencl_float_alpha,
                      viennacl::traits::opencl_handle(opencl_float_x).get(), 1, 2);
  check(ref_float_alpha, opencl_float_alpha, eps_float);
  if ( viennacl::ocl::current_device().double_support() )
  {
    ViennaCLOpenCLDnrm2(my_backend, ViennaCLInt(size/3),
                        &opencl_double_alpha,
                        viennacl::traits::opencl_handle(*opencl_double_x).get(), 1, 2);
    check(ref_double_alpha, opencl_double_alpha, eps_double);
  }
#endif




  // ROT
  std::cout << std::endl << "-- Testing xROT...";
  for (std::size_t i=0; i<size/4; ++i)
  {
    float tmp            =  0.6f * ref_float_x[2 + 3*i] + 0.8f * ref_float_y[1 + 2*i];
    ref_float_y[1 + 2*i] = -0.8f * ref_float_x[2 + 3*i] + 0.6f * ref_float_y[1 + 2*i];;
    ref_float_x[2 + 3*i] = tmp;

    double tmp2           =  0.6 * ref_double_x[2 + 3*i] + 0.8 * ref_double_y[1 + 2*i];
    ref_double_y[1 + 2*i] = -0.8 * ref_double_x[2 + 3*i] + 0.6 * ref_double_y[1 + 2*i];;
    ref_double_x[2 + 3*i] = tmp2;
  }

  std::cout << std::endl << "Host: ";
  ViennaCLHostSrot(my_backend, ViennaCLInt(size/4),
                   viennacl::linalg::host_based::detail::extract_raw_pointer<float>(host_float_x), 2, 3,
                   viennacl::linalg::host_based::detail::extract_raw_pointer<float>(host_float_y), 1, 2,
                   0.6f, 0.8f);
  check(ref_float_x, host_float_x, eps_float);
  check(ref_float_y, host_float_y, eps_float);
  ViennaCLHostDrot(my_backend, ViennaCLInt(size/4),
                   viennacl::linalg::host_based::detail::extract_raw_pointer<double>(host_double_x), 2, 3,
                   viennacl::linalg::host_based::detail::extract_raw_pointer<double>(host_double_y), 1, 2,
                   0.6, 0.8);
  check(ref_double_x, host_double_x, eps_double);
  check(ref_double_y, host_double_y, eps_double);


#ifdef VIENNACL_WITH_CUDA
  std::cout << std::endl << "CUDA: ";
  ViennaCLCUDASrot(my_backend, ViennaCLInt(size/4),
                   viennacl::cuda_arg(cuda_float_x), 2, 3,
                   viennacl::cuda_arg(cuda_float_y), 1, 2,
                   0.6f, 0.8f);
  check(ref_float_x, cuda_float_x, eps_float);
  check(ref_float_y, cuda_float_y, eps_float);
  ViennaCLCUDADrot(my_backend, ViennaCLInt(size/4),
                   viennacl::cuda_arg(cuda_double_x), 2, 3,
                   viennacl::cuda_arg(cuda_double_y), 1, 2,
                   0.6, 0.8);
  check(ref_double_x, cuda_double_x, eps_double);
  check(ref_double_y, cuda_double_y, eps_double);
#endif

#ifdef VIENNACL_WITH_OPENCL
  std::cout << std::endl << "OpenCL: ";
  ViennaCLOpenCLSrot(my_backend, ViennaCLInt(size/4),
                     viennacl::traits::opencl_handle(opencl_float_x).get(), 2, 3,
                     viennacl::traits::opencl_handle(opencl_float_y).get(), 1, 2,
                     0.6f, 0.8f);
  check(ref_float_x, opencl_float_x, eps_float);
  check(ref_float_y, opencl_float_y, eps_float);
  if ( viennacl::ocl::current_device().double_support() )
  {
    ViennaCLOpenCLDrot(my_backend, ViennaCLInt(size/4),
                       viennacl::traits::opencl_handle(*opencl_double_x).get(), 2, 3,
                       viennacl::traits::opencl_handle(*opencl_double_y).get(), 1, 2,
                       0.6, 0.8);
    check(ref_double_x, *opencl_double_x, eps_double);
    check(ref_double_y, *opencl_double_y, eps_double);
  }
#endif



  // SCAL
  std::cout << std::endl << "-- Testing xSCAL...";
  for (std::size_t i=0; i<size/4; ++i)
  {
    ref_float_x[1 + 3*i]  *= 2.0f;
    ref_double_x[1 + 3*i] *= 2.0;
  }

  std::cout << std::endl << "Host: ";
  ViennaCLHostSscal(my_backend, ViennaCLInt(size/4),
                    2.0f,
                    viennacl::linalg::host_based::detail::extract_raw_pointer<float>(host_float_x), 1, 3);
  check(ref_float_x, host_float_x, eps_float);
  ViennaCLHostDscal(my_backend, ViennaCLInt(size/4),
                    2.0,
                    viennacl::linalg::host_based::detail::extract_raw_pointer<double>(host_double_x), 1, 3);
  check(ref_double_x, host_double_x, eps_double);

#ifdef VIENNACL_WITH_CUDA
  std::cout << std::endl << "CUDA: ";
  ViennaCLCUDASscal(my_backend, ViennaCLInt(size/4),
                    2.0f,
                    viennacl::cuda_arg(cuda_float_x), 1, 3);
  check(ref_float_x, cuda_float_x, eps_float);
  ViennaCLCUDADscal(my_backend, ViennaCLInt(size/4),
                    2.0,
                    viennacl::cuda_arg(cuda_double_x), 1, 3);
  check(ref_double_x, cuda_double_x, eps_double);
#endif

#ifdef VIENNACL_WITH_OPENCL
  std::cout << std::endl << "OpenCL: ";
  ViennaCLOpenCLSscal(my_backend, ViennaCLInt(size/4),
                      2.0f,
                      viennacl::traits::opencl_handle(opencl_float_x).get(), 1, 3);
  check(ref_float_x, opencl_float_x, eps_float);
  if ( viennacl::ocl::current_device().double_support() )
  {
    ViennaCLOpenCLDscal(my_backend, ViennaCLInt(size/4),
                        2.0,
                        viennacl::traits::opencl_handle(*opencl_double_x).get(), 1, 3);
    check(ref_double_x, *opencl_double_x, eps_double);
  }
#endif


  // SWAP
  std::cout << std::endl << "-- Testing xSWAP...";
  for (std::size_t i=0; i<size/3; ++i)
  {
    float tmp = ref_float_x[2 + 2*i];
    ref_float_x[2 + 2*i] = ref_float_y[1 + 2*i];
    ref_float_y[1 + 2*i] = tmp;

    double tmp2 = ref_double_x[2 + 2*i];
    ref_double_x[2 + 2*i] = ref_double_y[1 + 2*i];
    ref_double_y[1 + 2*i] = tmp2;
  }

  std::cout << std::endl << "Host: ";
  ViennaCLHostSswap(my_backend, ViennaCLInt(size/3),
                    viennacl::linalg::host_based::detail::extract_raw_pointer<float>(host_float_x), 2, 2,
                    viennacl::linalg::host_based::detail::extract_raw_pointer<float>(host_float_y), 1, 2);
  check(ref_float_y, host_float_y, eps_float);
  ViennaCLHostDswap(my_backend, ViennaCLInt(size/3),
                    viennacl::linalg::host_based::detail::extract_raw_pointer<double>(host_double_x), 2, 2,
                    viennacl::linalg::host_based::detail::extract_raw_pointer<double>(host_double_y), 1, 2);
  check(ref_double_y, host_double_y, eps_double);


#ifdef VIENNACL_WITH_CUDA
  std::cout << std::endl << "CUDA: ";
  ViennaCLCUDASswap(my_backend, ViennaCLInt(size/3),
                    viennacl::cuda_arg(cuda_float_x), 2, 2,
                    viennacl::cuda_arg(cuda_float_y), 1, 2);
  check(ref_float_y, cuda_float_y, eps_float);
  ViennaCLCUDADswap(my_backend, ViennaCLInt(size/3),
                    viennacl::cuda_arg(cuda_double_x), 2, 2,
                    viennacl::cuda_arg(cuda_double_y), 1, 2);
  check(ref_double_y, cuda_double_y, eps_double);
#endif

#ifdef VIENNACL_WITH_OPENCL
  std::cout << std::endl << "OpenCL: ";
  ViennaCLOpenCLSswap(my_backend, ViennaCLInt(size/3),
                      viennacl::traits::opencl_handle(opencl_float_x).get(), 2, 2,
                      viennacl::traits::opencl_handle(opencl_float_y).get(), 1, 2);
  check(ref_float_y, opencl_float_y, eps_float);
  if ( viennacl::ocl::current_device().double_support() )
  {
    ViennaCLOpenCLDswap(my_backend, ViennaCLInt(size/3),
                        viennacl::traits::opencl_handle(*opencl_double_x).get(), 2, 2,
                        viennacl::traits::opencl_handle(*opencl_double_y).get(), 1, 2);
    check(ref_double_y, *opencl_double_y, eps_double);
  }
#endif


  // IAMAX
  std::cout << std::endl << "-- Testing IxASUM...";
  ViennaCLInt ref_index = 0;
  ref_float_alpha = 0;
  for (std::size_t i=0; i<size/3; ++i)
  {
    if (ref_float_x[0 + 2*i] > std::fabs(ref_float_alpha))
    {
      ref_index = ViennaCLInt(i);
      ref_float_alpha = std::fabs(ref_float_x[0 + 2*i]);
    }
  }

  std::cout << std::endl << "Host: ";
  ViennaCLInt idx = 0;
  ViennaCLHostiSamax(my_backend, ViennaCLInt(size/3),
                     &idx,
                     viennacl::linalg::host_based::detail::extract_raw_pointer<float>(host_float_x), 0, 2);
  check(static_cast<float>(ref_index), static_cast<float>(idx), eps_float);
  idx = 0;
  ViennaCLHostiDamax(my_backend, ViennaCLInt(size/3),
                     &idx,
                     viennacl::linalg::host_based::detail::extract_raw_pointer<double>(host_double_x), 0, 2);
  check(ref_index, idx, eps_double);

#ifdef VIENNACL_WITH_CUDA
  std::cout << std::endl << "CUDA: ";
  idx = 0;
  ViennaCLCUDAiSamax(my_backend, ViennaCLInt(size/3),
                     &idx,
                     viennacl::cuda_arg(cuda_float_x), 0, 2);
  check(ref_float_x[2*ref_index], ref_float_x[2*idx], eps_float);
  idx = 0;
  ViennaCLCUDAiDamax(my_backend, ViennaCLInt(size/3),
                     &idx,
                     viennacl::cuda_arg(cuda_double_x), 0, 2);
  check(ref_double_x[2*ref_index], ref_double_x[2*idx], eps_double);
#endif

#ifdef VIENNACL_WITH_OPENCL
  std::cout << std::endl << "OpenCL: ";
  idx = 0;
  ViennaCLOpenCLiSamax(my_backend, ViennaCLInt(size/3),
                       &idx,
                       viennacl::traits::opencl_handle(opencl_float_x).get(), 0, 2);
  check(ref_float_x[2*static_cast<std::size_t>(ref_index)], ref_float_x[2*static_cast<std::size_t>(idx)], eps_float);
  idx = 0;
  if ( viennacl::ocl::current_device().double_support() )
  {
    ViennaCLOpenCLiDamax(my_backend, ViennaCLInt(size/3),
                         &idx,
                         viennacl::traits::opencl_handle(*opencl_double_x).get(), 0, 2);
    check(ref_double_x[2*static_cast<std::size_t>(ref_index)], ref_double_x[2*static_cast<std::size_t>(idx)], eps_double);
  }
#endif

#ifdef VIENNACL_WITH_OPENCL
  //cleanup
  if ( viennacl::ocl::current_device().double_support() )
  {
    delete opencl_double_x;
    delete opencl_double_y;
  }
#endif

  ViennaCLBackendDestroy(&my_backend);

  //
  //  That's it.
  //
  std::cout << std::endl << "!!!! TEST COMPLETED SUCCESSFULLY !!!!" << std::endl;

  return EXIT_SUCCESS;
}

