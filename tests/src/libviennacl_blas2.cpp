/* =========================================================================
   Copyright (c) 2010-2015, Institute for Microelectronics,
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

/** \file tests/src/libviennacl_blas2.cpp  Testing the BLAS level 2 routines in the ViennaCL BLAS-like shared library
*   \test Testing the BLAS level 2 routines in the ViennaCL BLAS-like shared library
**/


// include necessary system headers
#include <iostream>
#include <vector>

// Some helper functions for this tutorial:
#include "viennacl.hpp"

#include "viennacl/vector.hpp"

template<typename ScalarType>
ScalarType diff(ScalarType const & s1, ScalarType const & s2)
{
  if (s1 > s2 || s1 < s2)
    return (s1 - s2) / std::max(std::fabs(s1), std::fabs(s2));
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
  std::size_t size1  = 13; // at least 7
  std::size_t size2  = 11; // at least 7
  float  eps_float  = 1e-5f;
  double eps_double = 1e-12;

  ViennaCLBackend my_backend;
  ViennaCLBackendCreate(&my_backend);

  std::vector<float> ref_float_x(size1); for (std::size_t i=0; i<size1; ++i) ref_float_x[i] = static_cast<float>(i);
  std::vector<float> ref_float_y(size2); for (std::size_t i=0; i<size2; ++i) ref_float_y[i] = static_cast<float>(size2 - i);
  std::vector<float> ref_float_A(size1*size2); for (std::size_t i=0; i<size1*size2; ++i) ref_float_A[i] = static_cast<float>(3*i);
  std::vector<float> ref_float_B(size1*size2); for (std::size_t i=0; i<size1*size2; ++i) ref_float_B[i] = static_cast<float>(2*i);

  std::vector<double> ref_double_x(size1, 1.0); for (std::size_t i=0; i<size1; ++i) ref_double_x[i] = static_cast<double>(i);
  std::vector<double> ref_double_y(size2, 2.0); for (std::size_t i=0; i<size2; ++i) ref_double_y[i] = static_cast<double>(size2 - i);
  std::vector<double> ref_double_A(size1*size2, 3.0); for (std::size_t i=0; i<size1*size2; ++i) ref_double_A[i] = static_cast<double>(3*i);
  std::vector<double> ref_double_B(size1*size2, 4.0); for (std::size_t i=0; i<size1*size2; ++i) ref_double_B[i] = static_cast<double>(2*i);

  // Host setup
  viennacl::vector<float> host_float_x = viennacl::scalar_vector<float>(size1, 1.0f, viennacl::context(viennacl::MAIN_MEMORY)); for (std::size_t i=0; i<size1; ++i) host_float_x[i] = float(i);
  viennacl::vector<float> host_float_y = viennacl::scalar_vector<float>(size2, 2.0f, viennacl::context(viennacl::MAIN_MEMORY)); for (std::size_t i=0; i<size2; ++i) host_float_y[i] = float(size2 - i);
  viennacl::vector<float> host_float_A = viennacl::scalar_vector<float>(size1*size2, 3.0f, viennacl::context(viennacl::MAIN_MEMORY)); for (std::size_t i=0; i<size1*size2; ++i) host_float_A[i] = float(3*i);
  viennacl::vector<float> host_float_B = viennacl::scalar_vector<float>(size1*size2, 4.0f, viennacl::context(viennacl::MAIN_MEMORY)); for (std::size_t i=0; i<size1*size2; ++i) host_float_B[i] = float(2*i);

  viennacl::vector<double> host_double_x = viennacl::scalar_vector<double>(size1, 1.0, viennacl::context(viennacl::MAIN_MEMORY)); for (std::size_t i=0; i<size1; ++i) host_double_x[i] = double(i);
  viennacl::vector<double> host_double_y = viennacl::scalar_vector<double>(size2, 2.0, viennacl::context(viennacl::MAIN_MEMORY)); for (std::size_t i=0; i<size2; ++i) host_double_y[i] = double(size2 - i);
  viennacl::vector<double> host_double_A = viennacl::scalar_vector<double>(size1*size2, 3.0, viennacl::context(viennacl::MAIN_MEMORY)); for (std::size_t i=0; i<size1*size2; ++i) host_double_A[i] = double(3*i);
  viennacl::vector<double> host_double_B = viennacl::scalar_vector<double>(size1*size2, 4.0, viennacl::context(viennacl::MAIN_MEMORY)); for (std::size_t i=0; i<size1*size2; ++i) host_double_B[i] = double(2*i);

  // CUDA setup
#ifdef VIENNACL_WITH_CUDA
  viennacl::vector<float> cuda_float_x = viennacl::scalar_vector<float>(size1, 1.0f, viennacl::context(viennacl::CUDA_MEMORY)); for (std::size_t i=0; i<size1; ++i) cuda_float_x[i] = float(i);
  viennacl::vector<float> cuda_float_y = viennacl::scalar_vector<float>(size2, 2.0f, viennacl::context(viennacl::CUDA_MEMORY)); for (std::size_t i=0; i<size2; ++i) cuda_float_y[i] = float(size2 - i);
  viennacl::vector<float> cuda_float_A = viennacl::scalar_vector<float>(size1*size2, 3.0f, viennacl::context(viennacl::CUDA_MEMORY)); for (std::size_t i=0; i<size1*size2; ++i) cuda_float_A[i] = float(3*i);
  viennacl::vector<float> cuda_float_B = viennacl::scalar_vector<float>(size1*size2, 4.0f, viennacl::context(viennacl::CUDA_MEMORY)); for (std::size_t i=0; i<size1*size2; ++i) cuda_float_B[i] = float(2*i);

  viennacl::vector<double> cuda_double_x = viennacl::scalar_vector<double>(size1, 1.0, viennacl::context(viennacl::CUDA_MEMORY)); for (std::size_t i=0; i<size1; ++i) cuda_double_x[i] = double(i);
  viennacl::vector<double> cuda_double_y = viennacl::scalar_vector<double>(size2, 2.0, viennacl::context(viennacl::CUDA_MEMORY)); for (std::size_t i=0; i<size2; ++i) cuda_double_y[i] = double(size2 - i);
  viennacl::vector<double> cuda_double_A = viennacl::scalar_vector<double>(size1*size2, 3.0, viennacl::context(viennacl::CUDA_MEMORY)); for (std::size_t i=0; i<size1*size2; ++i) cuda_double_A[i] = double(3*i);
  viennacl::vector<double> cuda_double_B = viennacl::scalar_vector<double>(size1*size2, 4.0, viennacl::context(viennacl::CUDA_MEMORY)); for (std::size_t i=0; i<size1*size2; ++i) cuda_double_B[i] = double(2*i);
#endif

  // OpenCL setup
#ifdef VIENNACL_WITH_OPENCL
  ViennaCLInt context_id = 0;
  viennacl::vector<float> opencl_float_x = viennacl::scalar_vector<float>(size1, 1.0f, viennacl::context(viennacl::ocl::get_context(context_id))); for (std::size_t i=0; i<size1; ++i) opencl_float_x[i] = float(i);
  viennacl::vector<float> opencl_float_y = viennacl::scalar_vector<float>(size2, 2.0f, viennacl::context(viennacl::ocl::get_context(context_id))); for (std::size_t i=0; i<size2; ++i) opencl_float_y[i] = float(size2 - i);
  viennacl::vector<float> opencl_float_A = viennacl::scalar_vector<float>(size1*size2, 3.0f, viennacl::context(viennacl::ocl::get_context(context_id))); for (std::size_t i=0; i<size1*size2; ++i) opencl_float_A[i] = float(3*i);
  viennacl::vector<float> opencl_float_B = viennacl::scalar_vector<float>(size1*size2, 4.0f, viennacl::context(viennacl::ocl::get_context(context_id))); for (std::size_t i=0; i<size1*size2; ++i) opencl_float_B[i] = float(2*i);

  viennacl::vector<double> *opencl_double_x = NULL;
  viennacl::vector<double> *opencl_double_y = NULL;
  viennacl::vector<double> *opencl_double_A = NULL;
  viennacl::vector<double> *opencl_double_B = NULL;
  if ( viennacl::ocl::current_device().double_support() )
  {
    opencl_double_x = new viennacl::vector<double>(viennacl::scalar_vector<double>(size1, 1.0, viennacl::context(viennacl::ocl::get_context(context_id)))); for (std::size_t i=0; i<size1; ++i) (*opencl_double_x)[i] = double(i);
    opencl_double_y = new viennacl::vector<double>(viennacl::scalar_vector<double>(size2, 2.0, viennacl::context(viennacl::ocl::get_context(context_id)))); for (std::size_t i=0; i<size2; ++i) (*opencl_double_y)[i] = double(size2 - i);
    opencl_double_A = new viennacl::vector<double>(viennacl::scalar_vector<double>(size1*size2, 3.0, viennacl::context(viennacl::ocl::get_context(context_id)))); for (std::size_t i=0; i<size1*size2; ++i) (*opencl_double_A)[i] = double(3*i);
    opencl_double_B = new viennacl::vector<double>(viennacl::scalar_vector<double>(size1*size2, 4.0, viennacl::context(viennacl::ocl::get_context(context_id)))); for (std::size_t i=0; i<size1*size2; ++i) (*opencl_double_B)[i] = double(2*i);
  }

  ViennaCLBackendSetOpenCLContextID(my_backend, context_id);
#endif

  // consistency checks:
  check(ref_float_x, host_float_x, eps_float);
  check(ref_float_y, host_float_y, eps_float);
  check(ref_float_A, host_float_A, eps_float);
  check(ref_float_B, host_float_B, eps_float);
  check(ref_double_x, host_double_x, eps_double);
  check(ref_double_y, host_double_y, eps_double);
  check(ref_double_A, host_double_A, eps_double);
  check(ref_double_B, host_double_B, eps_double);
#ifdef VIENNACL_WITH_CUDA
  check(ref_float_x, cuda_float_x, eps_float);
  check(ref_float_y, cuda_float_y, eps_float);
  check(ref_float_A, cuda_float_A, eps_float);
  check(ref_float_B, cuda_float_B, eps_float);
  check(ref_double_x, cuda_double_x, eps_double);
  check(ref_double_y, cuda_double_y, eps_double);
  check(ref_double_A, cuda_double_A, eps_double);
  check(ref_double_B, cuda_double_B, eps_double);
#endif
#ifdef VIENNACL_WITH_OPENCL
  check(ref_float_x, opencl_float_x, eps_float);
  check(ref_float_y, opencl_float_y, eps_float);
  check(ref_float_A, opencl_float_A, eps_float);
  check(ref_float_B, opencl_float_B, eps_float);
  if ( viennacl::ocl::current_device().double_support() )
  {
    check(ref_double_x, *opencl_double_x, eps_double);
    check(ref_double_y, *opencl_double_y, eps_double);
    check(ref_double_A, *opencl_double_A, eps_double);
    check(ref_double_B, *opencl_double_B, eps_double);
  }
#endif

  // GEMV
  std::cout << std::endl << "-- Testing xGEMV...";
  for (std::size_t i=0; i<size1/3; ++i)
  {
    ref_float_x[i * 2 + 1] *= 0.1234f;
    ref_double_x[i * 2 + 1] *= 0.1234;
    for (std::size_t j=0; j<size2/4; ++j)
    {
      ref_float_x[i * 2 + 1]  += 3.1415f * ref_float_A[(2*i+2) * size2 + 3 * j + 1] * ref_float_y[j * 3 + 1];
      ref_double_x[i * 2 + 1] += 3.1415  * ref_double_A[(2*i+2) * size2 + 3 * j + 1] * ref_double_y[j * 3 + 1];
    }
  }

  std::cout << std::endl << "Host: ";
  ViennaCLHostSgemv(my_backend,
                    ViennaCLRowMajor, ViennaCLNoTrans,
                    ViennaCLInt(size1/3), ViennaCLInt(size2/4), 3.1415f, viennacl::linalg::host_based::detail::extract_raw_pointer<float>(host_float_A), 2, 1, 2, 3, ViennaCLInt(size2),
                    viennacl::linalg::host_based::detail::extract_raw_pointer<float>(host_float_y), 1, 3,
                    0.1234f,
                    viennacl::linalg::host_based::detail::extract_raw_pointer<float>(host_float_x), 1, 2);
  check(ref_float_x, host_float_x, eps_float);
  ViennaCLHostDgemv(my_backend,
                    ViennaCLRowMajor, ViennaCLNoTrans,
                    ViennaCLInt(size1/3), ViennaCLInt(size2/4), 3.1415, viennacl::linalg::host_based::detail::extract_raw_pointer<double>(host_double_A), 2, 1, 2, 3, ViennaCLInt(size2),
                    viennacl::linalg::host_based::detail::extract_raw_pointer<double>(host_double_y), 1, 3,
                    0.1234,
                    viennacl::linalg::host_based::detail::extract_raw_pointer<double>(host_double_x), 1, 2);
  check(ref_double_x, host_double_x, eps_double);


#ifdef VIENNACL_WITH_CUDA
  std::cout << std::endl << "CUDA: ";
  ViennaCLCUDASgemv(my_backend,
                    ViennaCLRowMajor, ViennaCLNoTrans,
                    ViennaCLInt(size1/3), ViennaCLInt(size2/4), 3.1415f, viennacl::cuda_arg(cuda_float_A), 2, 1, 2, 3, size2,
                    viennacl::cuda_arg(cuda_float_y), 1, 3,
                    0.1234f,
                    viennacl::cuda_arg(cuda_float_x), 1, 2);
  check(ref_float_x, cuda_float_x, eps_float);
  ViennaCLCUDADgemv(my_backend,
                    ViennaCLRowMajor, ViennaCLNoTrans,
                    ViennaCLInt(size1/3), ViennaCLInt(size2/4), 3.1415, viennacl::cuda_arg(cuda_double_A), 2, 1, 2, 3, size2,
                    viennacl::cuda_arg(cuda_double_y), 1, 3,
                    0.1234,
                    viennacl::cuda_arg(cuda_double_x), 1, 2);
  check(ref_double_x, cuda_double_x, eps_double);
#endif

#ifdef VIENNACL_WITH_OPENCL
  std::cout << std::endl << "OpenCL: ";
  ViennaCLOpenCLSgemv(my_backend,
                      ViennaCLRowMajor, ViennaCLNoTrans,
                      ViennaCLInt(size1/3), ViennaCLInt(size2/4), 3.1415f, viennacl::traits::opencl_handle(opencl_float_A), 2, 1, 2, 3, ViennaCLInt(size2),
                      viennacl::traits::opencl_handle(opencl_float_y), 1, 3,
                      0.1234f,
                      viennacl::traits::opencl_handle(opencl_float_x), 1, 2);
  check(ref_float_x, opencl_float_x, eps_float);
  if ( viennacl::ocl::current_device().double_support() )
  {
    ViennaCLOpenCLDgemv(my_backend,
                        ViennaCLRowMajor, ViennaCLNoTrans,
                        ViennaCLInt(size1/3), ViennaCLInt(size2/4), 3.1415, viennacl::traits::opencl_handle(*opencl_double_A), 2, 1, 2, 3, ViennaCLInt(size2),
                        viennacl::traits::opencl_handle(*opencl_double_y), 1, 3,
                        0.1234,
                        viennacl::traits::opencl_handle(*opencl_double_x), 1, 2);
    check(ref_double_x, *opencl_double_x, eps_double);
  }
#endif



#ifdef VIENNACL_WITH_OPENCL
  delete opencl_double_x;
  delete opencl_double_y;
  delete opencl_double_A;
  delete opencl_double_B;
#endif

  ViennaCLBackendDestroy(&my_backend);

  //
  //  That's it.
  //
  std::cout << std::endl << "!!!! TEST COMPLETED SUCCESSFULLY !!!!" << std::endl;

  return EXIT_SUCCESS;
}

