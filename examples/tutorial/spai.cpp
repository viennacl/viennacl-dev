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

/** \example spai.cpp
*
*   This tutorial shows how to use the sparse approximate inverse (SPAI) preconditioner.
*
*   \warning SPAI is currently only available with the OpenCL backend and is experimental. API-changes may happen any time in the future.
*
*   We start with including the necessary headers:
**/

// enable Boost.uBLAS support
#define VIENNACL_WITH_UBLAS

#ifndef NDEBUG
 #define BOOST_UBLAS_NDEBUG
#endif

// System headers:
#include <utility>
#include <iostream>
#include <fstream>
#include <string>
#include <cmath>
#include <algorithm>
#include <stdio.h>

// ViennaCL headers:
#include "viennacl/scalar.hpp"
#include "viennacl/matrix.hpp"
#include "viennacl/compressed_matrix.hpp"
#include "viennacl/linalg/cg.hpp"
#include "viennacl/linalg/bicgstab.hpp"
#include "viennacl/linalg/prod.hpp"
#include "viennacl/linalg/inner_prod.hpp"
#include "viennacl/linalg/ilu.hpp"
#include "viennacl/linalg/norm_2.hpp"
#include "viennacl/io/matrix_market.hpp"
#include "viennacl/linalg/spai.hpp"

// Boost headers:
#include "boost/numeric/ublas/vector.hpp"
#include "boost/numeric/ublas/matrix.hpp"
#include "boost/numeric/ublas/io.hpp"

// auxiliary functionality:
#include "vector-io.hpp"

/**
*  The following helper routine is used to run a solver with the provided preconditioner and to print the resulting residual norm.
**/
template<typename MatrixT, typename VectorT, typename SolverTagT, typename PreconditionerT>
void run_solver(MatrixT const & A, VectorT const & b, SolverTagT const & solver_tag, PreconditionerT const & precond)
{
  VectorT result = viennacl::linalg::solve(A, b, solver_tag, precond);
  std::cout << " * Solver iterations: " << solver_tag.iters() << std::endl;
  VectorT residual = viennacl::linalg::prod(A, result);
  residual -= b;
  std::cout << " * Rel. Residual: " << viennacl::linalg::norm_2(residual) / viennacl::linalg::norm_2(b) << std::endl;
}

/**
*  The main steps in this tutorial are the following:
*  - Setup the systems
*  - Run solvers without preconditioner and with ILUT preconditioner for comparison
*  - Run solver with SPAI preconditioner on CPU
*  - Run solver with SPAI preconditioner on GPU
*  - Run solver with factored SPAI preconditioner on CPU
*  - Run solver with factored SPAI preconditioner on GPU
*
**/
int main (int, const char **)
{
  typedef float               ScalarType;
  typedef boost::numeric::ublas::compressed_matrix<ScalarType>        MatrixType;
  typedef boost::numeric::ublas::vector<ScalarType>                   VectorType;
  typedef viennacl::compressed_matrix<ScalarType>                     GPUMatrixType;
  typedef viennacl::vector<ScalarType>                                GPUVectorType;

  /**
  *  If you have multiple OpenCL-capable devices in your system, we pick the second device for this tutorial.
  **/
#ifdef VIENNACL_WITH_OPENCL
  // Optional: Customize OpenCL backend
  viennacl::ocl::platform pf = viennacl::ocl::get_platforms()[0];
  std::vector<viennacl::ocl::device> const & devices = pf.devices();

  // Optional: Set first device to first context:
  viennacl::ocl::setup_context(0, devices[0]);

  // Optional: Set second device for second context (use the same device for the second context if only one device available):
  if (devices.size() > 1)
    viennacl::ocl::setup_context(1, devices[1]);
  else
    viennacl::ocl::setup_context(1, devices[0]);

  std::cout << viennacl::ocl::current_device().info() << std::endl;
  viennacl::context ctx(viennacl::ocl::get_context(1));
#else
  viennacl::context ctx;
#endif

  /**
  * Create uBLAS-based sparse matrix and read system matrix from file
  **/
  MatrixType M;

  if (!viennacl::io::read_matrix_market_file(M, "../examples/testdata/mat65k.mtx"))
  {
    std::cerr<<"ERROR: Could not read matrix file " << std::endl;
    exit(EXIT_FAILURE);
  }

  std::cout << "Size of matrix: " << M.size1() << std::endl;
  std::cout << "Avg. Entries per row: " << double(M.nnz()) / static_cast<double>(M.size1()) << std::endl;

  /**
  *   Use a constant load vector for simplicity
  **/
  VectorType rhs(M.size2());
  for (std::size_t i=0; i<rhs.size(); ++i)
    rhs(i) = ScalarType(1);

  /**
  *   Create the ViennaCL matrix and vector and initialize with uBLAS data:
  **/
  GPUMatrixType  gpu_M(M.size1(), M.size2(), ctx);
  GPUVectorType  gpu_rhs(M.size1(), ctx);
  viennacl::copy(M, gpu_M);
  viennacl::copy(rhs, gpu_rhs);

  /**
  *  <h2>Solver Runs</h2>
  *  We use a relative tolerance of \f$ 10^{-10} \f$ with a maximum of 50 iterations for each use case.
  *  Usually more than 50 solver iterations are required for convergence, but this choice ensures shorter execution times and suffices for this tutorial.
  **/

  viennacl::linalg::bicgstab_tag solver_tag(1e-10, 50); //for simplicity and reasonably short execution times we use only 50 iterations here

  /**
  *  The first reference is to use no preconditioner (CPU and GPU):
  **/
  std::cout << "--- Reference 1: Pure BiCGStab on CPU ---" << std::endl;
  VectorType result = viennacl::linalg::solve(M, rhs, solver_tag);
  std::cout << " * Solver iterations: " << solver_tag.iters() << std::endl;
  VectorType residual = viennacl::linalg::prod(M, result) - rhs;
  std::cout << " * Rel. Residual: " << viennacl::linalg::norm_2(residual) / viennacl::linalg::norm_2(rhs) << std::endl;

  std::cout << "--- Reference 2: Pure BiCGStab on GPU ---" << std::endl;
  GPUVectorType gpu_result = viennacl::linalg::solve(gpu_M, gpu_rhs, solver_tag);
  std::cout << " * Solver iterations: " << solver_tag.iters() << std::endl;
  GPUVectorType gpu_residual = viennacl::linalg::prod(gpu_M, gpu_result);
  gpu_residual -= gpu_rhs;
  std::cout << " * Rel. Residual: " << viennacl::linalg::norm_2(gpu_residual) / viennacl::linalg::norm_2(gpu_rhs) << std::endl;


  /**
  * The second reference is a standard ILUT preconditioner (only CPU):
  **/
  std::cout << "--- Reference 2: BiCGStab with ILUT on CPU ---" << std::endl;
  std::cout << " * Preconditioner setup..." << std::endl;
  viennacl::linalg::ilut_precond<MatrixType> ilut(M, viennacl::linalg::ilut_tag());
  std::cout << " * Iterative solver run..." << std::endl;
  run_solver(M, rhs, solver_tag, ilut);


  /**
  * <h2>Step 1: SPAI with CPU</h2>
  **/
  std::cout << "--- Test 1: CPU-based SPAI ---" << std::endl;
  std::cout << " * Preconditioner setup..." << std::endl;
  viennacl::linalg::spai_precond<MatrixType> spai_cpu(M, viennacl::linalg::spai_tag(1e-3, 3, 5e-2));
  std::cout << " * Iterative solver run..." << std::endl;
  run_solver(M, rhs, solver_tag, spai_cpu);

  /**
  * <h2>Step 2: FSPAI with CPU</h2>
  **/
  std::cout << "--- Test 2: CPU-based FSPAI ---" << std::endl;
  std::cout << " * Preconditioner setup..." << std::endl;
  viennacl::linalg::fspai_precond<MatrixType> fspai_cpu(M, viennacl::linalg::fspai_tag());
  std::cout << " * Iterative solver run..." << std::endl;
  run_solver(M, rhs, solver_tag, fspai_cpu);

  /**
  * <h2>Step 3: SPAI with GPU</h2>
  **/
  std::cout << "--- Test 3: GPU-based SPAI ---" << std::endl;
  std::cout << " * Preconditioner setup..." << std::endl;
  viennacl::linalg::spai_precond<GPUMatrixType> spai_gpu(gpu_M, viennacl::linalg::spai_tag(1e-3, 3, 5e-2));
  std::cout << " * Iterative solver run..." << std::endl;
  run_solver(gpu_M, gpu_rhs, solver_tag, spai_gpu);

  /**
  * <h2>Step 4: FSPAI with GPU</h2>
  **/
  std::cout << "--- Test 4: GPU-based FSPAI ---" << std::endl;
  std::cout << " * Preconditioner setup..." << std::endl;
  viennacl::linalg::fspai_precond<GPUMatrixType> fspai_gpu(gpu_M, viennacl::linalg::fspai_tag());
  std::cout << " * Iterative solver run..." << std::endl;
  run_solver(gpu_M, gpu_rhs, solver_tag, fspai_gpu);

  /**
  *   That's it! Print success message and exit.
  **/
  std::cout << "!!!! TUTORIAL COMPLETED SUCCESSFULLY !!!!" << std::endl;

  return EXIT_SUCCESS;
}

