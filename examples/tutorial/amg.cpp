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


/** \example amg.cpp
*
*   This tutorial shows the use of algebraic multigrid (AMG) preconditioners.
*   \warning AMG is currently only experimentally available with the OpenCL backend and depends on Boost.uBLAS
*
*   We start with some rather general includes and preprocessor variables:
**/

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
#include "viennacl/tools/timer.hpp"


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

  viennacl::tools::timer timer;
  timer.start();
  result = viennacl::linalg::solve(matrix, rhs, solver, precond);
  viennacl::backend::finish();
  std::cout << "  > Solver time: " << timer.get() << std::endl;
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
             viennacl::vector<ScalarType> & vcl_vec,
             viennacl::vector<ScalarType> & vcl_result,
             viennacl::compressed_matrix<ScalarType> & vcl_compressed_matrix,
             std::string info,
             viennacl::linalg::amg_tag & amg_tag)
{
  std::cout << "-- CG with AMG preconditioner, " << info << " --" << std::endl;

  viennacl::linalg::amg_precond<viennacl::compressed_matrix<ScalarType> > vcl_amg(vcl_compressed_matrix, amg_tag);
  std::cout << " * Setup phase (ViennaCL types)..." << std::endl;
  viennacl::tools::timer timer;
  timer.start();
  vcl_amg.setup();
  viennacl::backend::finish();
  std::cout << "  > Setup time: " << timer.get() << std::endl;

  std::cout << " * CG solver (ViennaCL types)..." << std::endl;
  run_solver(vcl_compressed_matrix, vcl_vec, vcl_result, cg_solver, vcl_amg);
}

/**
*  <h2>Part 2: Run Solvers with AMG Preconditioners</h2>
*
*  In this
**/
int main(int argc, char **argv)
{
  std::string filename("../examples/testdata/mat65k.mtx");
  if (argc == 2)
    filename = argv[1];

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
  viennacl::context ctx(viennacl::ocl::get_context(0));
#else
  viennacl::context ctx;
#endif

  typedef double    ScalarType;  // feel free to change this to double if supported by your device


  /**
  * Set up the matrices and vectors for the iterative solvers (cf. iterative.cpp)
  **/
  viennacl::compressed_matrix<ScalarType> vcl_compressed_matrix(ctx);

  //viennacl::tools::generate_fdm_laplace(vcl_compressed_matrix, points_per_dim, points_per_dim);
  // Read matrix
  std::cout << "Reading matrix..." << std::endl;
  std::vector< std::map<unsigned int, ScalarType> > read_in_matrix;
  if (!viennacl::io::read_matrix_market_file(read_in_matrix, filename))
  {
    std::cout << "Error reading Matrix file" << std::endl;
    return EXIT_FAILURE;
  }
  viennacl::copy(read_in_matrix, vcl_compressed_matrix);
  std::cout << "Reading matrix completed." << std::endl;

  viennacl::vector<ScalarType> vcl_vec(vcl_compressed_matrix.size1(), ctx);
  viennacl::vector<ScalarType> vcl_result(vcl_compressed_matrix.size1(), ctx);

  std::vector<ScalarType> std_vec, std_result;


  // rhs and result vector:
  std_vec.resize(vcl_compressed_matrix.size1());
  std_result.resize(vcl_compressed_matrix.size1());
  for (std::size_t i=0; i<std_result.size(); ++i)
    std_result[i] = ScalarType(1);

  // Copy to GPU
  viennacl::copy(std_vec, vcl_vec);
  viennacl::copy(std_result, vcl_result);

  vcl_vec = viennacl::linalg::prod(vcl_compressed_matrix, vcl_result);


  /**
  * Instantiate a tag for the conjugate gradient solver, the AMG preconditioner tag, and create an AMG preconditioner object:
  **/
  viennacl::linalg::cg_tag cg_solver(1e-8, 10000);

  viennacl::context host_ctx(viennacl::MAIN_MEMORY);
  viennacl::context target_ctx = viennacl::traits::context(vcl_compressed_matrix);

  /**
  * Run solver without preconditioner. This serves as a baseline for comparison.
  * Note that iterative solvers without preconditioner on GPUs can be very efficient because they map well to the massively parallel hardware.
  **/
  std::cout << "-- CG solver (no preconditioner, warmup) --" << std::endl;
  run_solver(vcl_compressed_matrix, vcl_vec, vcl_result, cg_solver, viennacl::linalg::no_precond());

  /**
  * Generate the setup for an AMG preconditioner of Ruge-Stueben type with only one pass and direct interpolation (ONEPASS+DIRECT)
  **/
  viennacl::linalg::amg_tag amg_tag_direct;
  amg_tag_direct.set_coarsening_method(viennacl::linalg::AMG_COARSENING_METHOD_ONEPASS);
  amg_tag_direct.set_interpolation_method(viennacl::linalg::AMG_INTERPOLATION_METHOD_DIRECT);
  amg_tag_direct.set_strong_connection_threshold(0.25);
  amg_tag_direct.set_jacobi_weight(0.67);
  amg_tag_direct.set_presmooth_steps(1);
  amg_tag_direct.set_postsmooth_steps(1);
  amg_tag_direct.set_setup_context(host_ctx);    // run setup on host
  amg_tag_direct.set_target_context(target_ctx); // run solver cycles on device
  run_amg(cg_solver, vcl_vec, vcl_result, vcl_compressed_matrix, "ONEPASS COARSENING, DIRECT INTERPOLATION", amg_tag_direct);

  /**
  * Generate the setup for an aggregation-based AMG preconditioner with unsmoothed aggregation
  **/
  viennacl::linalg::amg_tag amg_tag_agg_pmis;
  amg_tag_agg_pmis.set_coarsening_method(viennacl::linalg::AMG_COARSENING_METHOD_MIS2_AGGREGATION);
  amg_tag_agg_pmis.set_interpolation_method(viennacl::linalg::AMG_INTERPOLATION_METHOD_AGGREGATION);
  run_amg(cg_solver, vcl_vec, vcl_result, vcl_compressed_matrix, "AG COARSENING (PMIS), AG INTERPOLATION", amg_tag_agg_pmis);

  /**
  * Generate the setup for a smoothed aggregation AMG preconditioner
  **/
  viennacl::linalg::amg_tag amg_tag_sa_pmis;
  amg_tag_sa_pmis.set_coarsening_method(viennacl::linalg::AMG_COARSENING_METHOD_MIS2_AGGREGATION);
  amg_tag_sa_pmis.set_interpolation_method(viennacl::linalg::AMG_INTERPOLATION_METHOD_SMOOTHED_AGGREGATION);
  run_amg (cg_solver, vcl_vec, vcl_result, vcl_compressed_matrix, "AG COARSENING (PMIS), SA INTERPOLATION", amg_tag_sa_pmis);

  std::cout << std::endl;
  std::cout << " -------------- Benchmark runs -------------- " << std::endl;
  std::cout << std::endl;

  std::cout << "-- CG solver (no preconditioner) --" << std::endl;
  run_solver(vcl_compressed_matrix, vcl_vec, vcl_result, cg_solver, viennacl::linalg::no_precond());
  run_amg(cg_solver, vcl_vec, vcl_result, vcl_compressed_matrix, "ONEPASS COARSENING, DIRECT INTERPOLATION", amg_tag_direct);
  run_amg(cg_solver, vcl_vec, vcl_result, vcl_compressed_matrix, "AG COARSENING (PMIS), AG INTERPOLATION", amg_tag_agg_pmis);
  run_amg (cg_solver, vcl_vec, vcl_result, vcl_compressed_matrix, "AG COARSENING (PMIS), SA INTERPOLATION", amg_tag_sa_pmis);

  /**
  *  That's it.
  **/
  std::cout << "!!!! TUTORIAL COMPLETED SUCCESSFULLY !!!!" << std::endl;

  return EXIT_SUCCESS;
}

