/* =========================================================================
   Copyright (c) 2010-2014, Institute for Microelectronics,
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


/** \example amg.cpp
*
*   This tutorial shows the use of algebraic multigrid (AMG) preconditioners.
*   \warning AMG is currently only experimentally available with the OpenCL backend and depends on Boost.uBLAS
*
*   We start with some rather general includes and preprocessor variables:
**/


#ifndef NDEBUG     //without NDEBUG the performance of sparse ublas matrices is poor.
 #define BOOST_UBLAS_NDEBUG
#endif

#include <boost/numeric/ublas/matrix_sparse.hpp>
#include <boost/numeric/ublas/operation_sparse.hpp>

#define VIENNACL_WITH_UBLAS 1

#include "viennacl/vector.hpp"
#include "viennacl/coordinate_matrix.hpp"
#include "viennacl/compressed_matrix.hpp"
#include "viennacl/linalg/ilu.hpp"
#include "viennacl/linalg/cg.hpp"
#include "viennacl/linalg/bicgstab.hpp"
#include "viennacl/io/matrix_market.hpp"
#include "viennacl/linalg/norm_2.hpp"
#include "viennacl/tools/matrix_generation.hpp"

/**
* Import the AMG functionality:
**/
#include "viennacl/linalg/amg.hpp"

/**
* Some more includes:
**/
#include <iostream>
#include <vector>
#include <ctime>
#include "vector-io.hpp"


/** <h2>Part 1: Worker routines</h2>
*
*  <h3>Run the Solver</h3>
*   Runs the provided solver specified in the `solver` object with the provided preconditioner `precond`
**/
template<typename MatrixType, typename VectorType, typename SolverTag, typename PrecondTag>
void run_solver(MatrixType const & matrix, VectorType const & rhs, VectorType const & ref_result, SolverTag const & solver, PrecondTag const & precond)
{
  VectorType result(rhs);
  VectorType residual(rhs);

  result = viennacl::linalg::solve(matrix, rhs, solver, precond);
  residual -= viennacl::linalg::prod(matrix, result);
  std::cout << "  > Relative residual: " << viennacl::linalg::norm_2(residual) / viennacl::linalg::norm_2(rhs) << std::endl;
  std::cout << "  > Iterations: " << solver.iters() << std::endl;
  result -= ref_result;
  std::cout << "  > Relative deviation from result: " << viennacl::linalg::norm_2(result) / viennacl::linalg::norm_2(ref_result) << std::endl;
}

/** <h3>Compare AMG preconditioner for uBLAS and ViennaCL types</h3>
*
*  The AMG implementations in ViennaCL can be used with uBLAS types as well as ViennaCL types.
*  This function compares the two in terms of execution time.
**/
template<typename ScalarType>
void run_amg(viennacl::linalg::cg_tag & cg_solver,
             boost::numeric::ublas::vector<ScalarType> & /*ublas_vec*/,
             boost::numeric::ublas::vector<ScalarType> & /*ublas_result*/,
             boost::numeric::ublas::compressed_matrix<ScalarType> & ublas_matrix,
             viennacl::vector<ScalarType> & vcl_vec,
             viennacl::vector<ScalarType> & vcl_result,
             viennacl::compressed_matrix<ScalarType> & vcl_compressed_matrix,
             std::string info,
             viennacl::linalg::amg_tag & amg_tag)
{

  //boost::numeric::ublas::vector<ScalarType> avgstencil;
  unsigned int coarselevels = amg_tag.get_coarselevels();

  std::cout << "-- CG with AMG preconditioner, " << info << " --" << std::endl;

  amg_tag.set_coarselevels(coarselevels);
  viennacl::linalg::amg_precond<viennacl::compressed_matrix<ScalarType> > vcl_amg = viennacl::linalg::amg_precond<viennacl::compressed_matrix<ScalarType> > (vcl_compressed_matrix, amg_tag);
  std::cout << " * Setup phase (ViennaCL types)..." << std::endl;
  vcl_amg.tag().set_coarselevels(coarselevels);
  vcl_amg.setup();

  std::cout << " * CG solver (ViennaCL types)..." << std::endl;
  run_solver(vcl_compressed_matrix, vcl_vec, vcl_result, cg_solver, vcl_amg);

}

/**
*  <h2>Part 2: Run Solvers with AMG Preconditioners</h2>
*
*  In this
**/
int main()
{
  /**
  * Print some device info at the beginning. If there is more than one OpenCL device available, use the second device.
  **/
  std::cout << std::endl;
  std::cout << "----------------------------------------------" << std::endl;
  std::cout << "               Device Info" << std::endl;
  std::cout << "----------------------------------------------" << std::endl;

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

  typedef float    ScalarType;  // feel free to change this to double if supported by your device


  /**
  * Set up the matrices and vectors for the iterative solvers (cf. iterative.cpp)
  **/
  boost::numeric::ublas::vector<ScalarType> ublas_vec, ublas_result;
  boost::numeric::ublas::compressed_matrix<ScalarType> ublas_matrix;

  //viennacl::tools::generate_fdm_laplace(ublas_matrix, 3, 3);

  // Read matrix
  if (!viennacl::io::read_matrix_market_file(ublas_matrix, "../examples/testdata/mat65k.mtx"))
  {
    std::cout << "Error reading Matrix file" << std::endl;
    return EXIT_FAILURE;
  }

  // rhs and result vector:
  ublas_vec.resize(ublas_matrix.size1());
  ublas_result.resize(ublas_matrix.size1());
  for (std::size_t i=0; i<ublas_result.size(); ++i)
    ublas_result[i] = ScalarType(1);

  ublas_vec = prod(ublas_matrix, ublas_result);

  viennacl::vector<ScalarType> vcl_vec(ublas_vec.size(), ctx);
  viennacl::vector<ScalarType> vcl_result(ublas_vec.size(), ctx);
  viennacl::compressed_matrix<ScalarType> vcl_compressed_matrix(ublas_vec.size(), ublas_vec.size(), ctx);

  // Copy to GPU
  viennacl::copy(ublas_matrix, vcl_compressed_matrix);
  viennacl::copy(ublas_vec, vcl_vec);
  viennacl::copy(ublas_result, vcl_result);

  /**
  * Instantiate a tag for the conjugate gradient solver, the AMG preconditioner tag, and create an AMG preconditioner object:
  **/
  viennacl::linalg::cg_tag cg_solver;
  viennacl::linalg::amg_tag amg_tag;

  /**
  * Run solver without preconditioner. This serves as a baseline for comparison.
  * Note that iterative solvers without preconditioner on GPUs can be very efficient because they map well to the massively parallel hardware.
  **/
  std::cout << "-- CG solver (CPU, no preconditioner) --" << std::endl;
  run_solver(ublas_matrix, ublas_vec, ublas_result, cg_solver, viennacl::linalg::no_precond());

  std::cout << "-- CG solver (GPU, no preconditioner) --" << std::endl;
  run_solver(vcl_compressed_matrix, vcl_vec, vcl_result, cg_solver, viennacl::linalg::no_precond());

  /**
  * Generate the setup for an AMG preconditioner of Ruge-Stueben type with direct interpolation (RS+DIRECT) and run the solver:
  **/
  amg_tag = viennacl::linalg::amg_tag(VIENNACL_AMG_COARSE_RS,       // coarsening strategy
                                      VIENNACL_AMG_INTERPOL_DIRECT, // interpolation strategy
                                      0.25, // strength of dependence threshold
                                      0.2,  // interpolation weight
                                      0.67, // jacobi smoother weight
                                      3,    // presmoothing steps
                                      3,    // postsmoothing steps
                                      0);   // number of coarse levels to be used (0: automatically use as many as reasonable)
  //run_amg (cg_solver, ublas_vec, ublas_result, ublas_matrix, vcl_vec, vcl_result, vcl_compressed_matrix, "RS COARSENING, DIRECT INTERPOLATION", amg_tag);

  /**
  * Generate the setup for an AMG preconditioner of Ruge-Stueben type with only one pass and direct interpolation (ONEPASS+DIRECT)
  **/
  amg_tag = viennacl::linalg::amg_tag(VIENNACL_AMG_COARSE_ONEPASS, VIENNACL_AMG_INTERPOL_DIRECT,0.25, 0.2, 0.67, 3, 3, 0);
  run_amg (cg_solver, ublas_vec, ublas_result, ublas_matrix, vcl_vec, vcl_result, vcl_compressed_matrix, "ONEPASS COARSENING, DIRECT INTERPOLATION", amg_tag);


  /**
  * Generate the setup for an AMG preconditioner of Ruge-Stueben type with classic interpolation (RS+CLASSIC) and run the solver:
  **/
  amg_tag = viennacl::linalg::amg_tag(VIENNACL_AMG_COARSE_RS, VIENNACL_AMG_INTERPOL_CLASSIC, 0.25, 0.2, 0.67, 3, 3, 0);
  run_amg ( cg_solver, ublas_vec, ublas_result, ublas_matrix, vcl_vec, vcl_result, vcl_compressed_matrix, "RS COARSENING, CLASSIC INTERPOLATION", amg_tag);

  /**
  * Generate the setup for an AMG preconditioner of parallel Ruge-Stueben type with direct interpolation (RS0+DIRECT)
  **/
  amg_tag = viennacl::linalg::amg_tag(VIENNACL_AMG_COARSE_RS0, VIENNACL_AMG_INTERPOL_DIRECT, 0.25, 0.2, 0.67, 3, 3, 0);
  run_amg (cg_solver, ublas_vec, ublas_result, ublas_matrix, vcl_vec, vcl_result, vcl_compressed_matrix, "RS0 COARSENING, DIRECT INTERPOLATION", amg_tag);

  /**
  * Generate the setup for an AMG preconditioner of parallel Ruge-Stueben type (with communication across domains) and use direct interpolation (RS3+DIRECT)
  **/
  amg_tag = viennacl::linalg::amg_tag(VIENNACL_AMG_COARSE_RS3, VIENNACL_AMG_INTERPOL_DIRECT, 0.25, 0.2, 0.67, 3, 3, 0);
  run_amg (cg_solver, ublas_vec, ublas_result, ublas_matrix, vcl_vec, vcl_result, vcl_compressed_matrix, "RS3 COARSENING, DIRECT INTERPOLATION", amg_tag);

  /**
  * Generate the setup for an AMG preconditioner which as aggregation-based (AG)
  **/
  amg_tag = viennacl::linalg::amg_tag(VIENNACL_AMG_COARSE_AG, VIENNACL_AMG_INTERPOL_AG, 0.08, 0, 0.67, 3, 3, 0);
  run_amg (cg_solver, ublas_vec, ublas_result, ublas_matrix, vcl_vec, vcl_result, vcl_compressed_matrix, "AG COARSENING, AG INTERPOLATION", amg_tag);

  /**
  * Generate the setup for an AMG preconditioner with smoothed aggregation (SA)
  **/
  amg_tag = viennacl::linalg::amg_tag(VIENNACL_AMG_COARSE_AG, VIENNACL_AMG_INTERPOL_SA, 0.08, 0.67, 0.67, 3, 3, 0);
  run_amg (cg_solver, ublas_vec, ublas_result, ublas_matrix, vcl_vec, vcl_result, vcl_compressed_matrix, "AG COARSENING, SA INTERPOLATION",amg_tag);


  /**
  *  That's it.
  **/
  std::cout << "!!!! TUTORIAL COMPLETED SUCCESSFULLY !!!!" << std::endl;

  return EXIT_SUCCESS;
}

