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

  // note: direct assignment runs into issues with OpenCL (kernels for char poor)
  amg_tag.set_coarselevels(vcl_amg.tag().get_coarselevels());
  for (std::size_t i=0; i<amg_tag.get_coarselevels(); ++i)
  {
    if (vcl_amg.tag().get_coarse_information(i).size() > 0)
    {
      std::vector<char> tmp(vcl_amg.tag().get_coarse_information(i).size());
      viennacl::copy(vcl_amg.tag().get_coarse_information(i), tmp);
      viennacl::copy(tmp, amg_tag.get_coarse_information(i));
    }
  }
}

/**
*  <h2>Part 2: Run Solvers with AMG Preconditioners</h2>
*
*  In this
**/
int main(int argc, char **argv)
{
  if (argc < 2)
  {
    std::cout << "Missing argument: filename" << std::endl;
    exit(EXIT_FAILURE);
  }

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

  std::cout << "Number of grid points per dimension: " << std::endl;
  std::size_t points_per_dim;
  std::cin >> points_per_dim;

  //viennacl::tools::generate_fdm_laplace(vcl_compressed_matrix, points_per_dim, points_per_dim);
  // Read matrix
  if (!viennacl::io::read_matrix_market_file(vcl_compressed_matrix, argv[1]))
  {
    std::cout << "Error reading Matrix file" << std::endl;
    return EXIT_FAILURE;
  }

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
  viennacl::linalg::amg_tag amg_tag;

  viennacl::context host_ctx(viennacl::MAIN_MEMORY);
  viennacl::context target_ctx = viennacl::traits::context(vcl_compressed_matrix);

  /**
  * Run solver without preconditioner. This serves as a baseline for comparison.
  * Note that iterative solvers without preconditioner on GPUs can be very efficient because they map well to the massively parallel hardware.
  **/
  std::cout << "-- CG solver (no preconditioner, warmup) --" << std::endl;
  run_solver(vcl_compressed_matrix, vcl_vec, vcl_result, cg_solver, viennacl::linalg::no_precond());

  std::cout << "-- CG solver (no preconditioner) --" << std::endl;
  run_solver(vcl_compressed_matrix, vcl_vec, vcl_result, cg_solver, viennacl::linalg::no_precond());

  /**
  * Generate the setup for an AMG preconditioner of Ruge-Stueben type with direct interpolation (RS+DIRECT) and run the solver:
  **/
  amg_tag = viennacl::linalg::amg_tag(VIENNACL_AMG_COARSE_RS,       // coarsening strategy
                                      VIENNACL_AMG_INTERPOL_DIRECT, // interpolation strategy
                                      0.25, // strength of dependence threshold
                                      0.2,  // interpolation weight
                                      0.67, // jacobi smoother weight
                                      1,    // presmoothing steps
                                      1,    // postsmoothing steps
                                      0);   // number of coarse levels to be used (0: automatically use as many as reasonable)
  //run_amg (cg_solver, ublas_vec, ublas_result, ublas_matrix, vcl_vec, vcl_result, vcl_compressed_matrix, "RS COARSENING, DIRECT INTERPOLATION", amg_tag);

  /**
  * Generate the setup for an AMG preconditioner of Ruge-Stueben type with only one pass and direct interpolation (ONEPASS+DIRECT)
  **/
  amg_tag = viennacl::linalg::amg_tag(VIENNACL_AMG_COARSE_ONEPASS, VIENNACL_AMG_INTERPOL_DIRECT,0.25, 0.2, 0.67, 1, 1, 0);
  amg_tag.set_setup_context(host_ctx);
  amg_tag.set_target_context(target_ctx);
  run_amg(cg_solver, vcl_vec, vcl_result, vcl_compressed_matrix, "ONEPASS COARSENING, DIRECT INTERPOLATION", amg_tag);


  /**
  * Generate the setup for an AMG preconditioner of Ruge-Stueben type with classic interpolation (RS+CLASSIC) and run the solver:
  **/
  //amg_tag = viennacl::linalg::amg_tag(VIENNACL_AMG_COARSE_RS, VIENNACL_AMG_INTERPOL_CLASSIC, 0.25, 0.2, 0.67, 3, 3, 0);
  //run_amg ( cg_solver, ublas_vec, ublas_result, ublas_matrix, vcl_vec, vcl_result, vcl_compressed_matrix, "RS COARSENING, CLASSIC INTERPOLATION", amg_tag);

  /**
  * Generate the setup for an AMG preconditioner of parallel Ruge-Stueben type with direct interpolation (RS0+DIRECT)
  **/
  //amg_tag = viennacl::linalg::amg_tag(VIENNACL_AMG_COARSE_RS0, VIENNACL_AMG_INTERPOL_DIRECT, 0.25, 0.2, 0.67, 3, 3, 0);
  //run_amg (cg_solver, ublas_vec, ublas_result, ublas_matrix, vcl_vec, vcl_result, vcl_compressed_matrix, "RS0 COARSENING, DIRECT INTERPOLATION", amg_tag);

  /**
  * Generate the setup for an AMG preconditioner of parallel Ruge-Stueben type (with communication across domains) and use direct interpolation (RS3+DIRECT)
  **/
  //amg_tag = viennacl::linalg::amg_tag(VIENNACL_AMG_COARSE_RS3, VIENNACL_AMG_INTERPOL_DIRECT, 0.25, 0.2, 0.67, 3, 3, 0);
  //run_amg (cg_solver, ublas_vec, ublas_result, ublas_matrix, vcl_vec, vcl_result, vcl_compressed_matrix, "RS3 COARSENING, DIRECT INTERPOLATION", amg_tag);

  /**
  * Generate the setup for an AMG preconditioner which as aggregation-based (AG)
  **/
  viennacl::linalg::amg_tag amg_tag_host(VIENNACL_AMG_COARSE_AG, VIENNACL_AMG_INTERPOL_AG, 0.002, 0, 0.67, 2, 2, 0);
#ifdef VIENNACL_WITH_OPENCL
  amg_tag_host.set_setup_context(host_ctx);
  amg_tag_host.set_target_context(target_ctx);
  amg_tag_host.save_coarse_information(true);
#endif
  run_amg(cg_solver, vcl_vec, vcl_result, vcl_compressed_matrix, "AG COARSENING, AG INTERPOLATION (host)", amg_tag_host);

#ifdef VIENNACL_WITH_OPENCL
  amg_tag = viennacl::linalg::amg_tag(VIENNACL_AMG_COARSE_AG, VIENNACL_AMG_INTERPOL_AG, 0.002, 0, 0.67, 2, 2, 0);
  amg_tag.set_setup_context(target_ctx);
  amg_tag.set_target_context(target_ctx);
  for (std::size_t i=0; i < amg_tag_host.get_coarselevels(); ++i)
  {
    if (amg_tag_host.get_coarse_information(i).size() > 0)
    {
      std::vector<char> tmp(amg_tag_host.get_coarse_information(i).size());
      viennacl::copy(amg_tag_host.get_coarse_information(i), tmp);
      viennacl::copy(tmp, amg_tag.get_coarse_information(i));
    }
    //amg_tag.set_coarse_information(i, amg_tag_host.get_coarse_information(i));
  }
  amg_tag.set_coarselevels(amg_tag_host.get_coarselevels());
  amg_tag.use_coarse_information(true);
  run_amg(cg_solver, vcl_vec, vcl_result, vcl_compressed_matrix, "AG COARSENING, AG INTERPOLATION (device)", amg_tag);
#endif

  /**
  * Generate the setup for an AMG preconditioner with smoothed aggregation (SA)
  **/
  //amg_tag = viennacl::linalg::amg_tag(VIENNACL_AMG_COARSE_AG, VIENNACL_AMG_INTERPOL_SA, 0.08, 0.67, 0.67, 3, 3, 0);
  //run_amg (cg_solver, ublas_vec, ublas_result, ublas_matrix, vcl_vec, vcl_result, vcl_compressed_matrix, "AG COARSENING, SA INTERPOLATION",amg_tag);


  /**
  *  That's it.
  **/
  std::cout << "!!!! TUTORIAL COMPLETED SUCCESSFULLY !!!!" << std::endl;

  return EXIT_SUCCESS;
}

