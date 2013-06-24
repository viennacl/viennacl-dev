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


/*
*
*   Tutorial: Algebraic multigrid preconditioner (only available with the OpenCL backend, experimental)
*
*/



#ifndef NDEBUG     //without NDEBUG the performance of sparse ublas matrices is poor.
 #define NDEBUG
#endif

#include <boost/numeric/ublas/matrix_sparse.hpp>
#include <boost/numeric/ublas/operation_sparse.hpp>

#define VIENNACL_WITH_UBLAS 1

#define SOLVER_ITERS 2500
//#define SCALAR float
#define SCALAR double

//#define SOLVER_TOLERANCE 1e-5
#define SOLVER_TOLERANCE 1e-9

#include "viennacl/vector.hpp"
#include "viennacl/coordinate_matrix.hpp"
#include "viennacl/compressed_matrix.hpp"
#include "viennacl/linalg/ilu.hpp"
#include "viennacl/linalg/cg.hpp"
#include "viennacl/linalg/bicgstab.hpp"
#include "viennacl/io/matrix_market.hpp"
#include "viennacl/linalg/norm_2.hpp"

#include "viennacl/linalg/amg.hpp"

#include <iostream>
#include <vector>
#include <ctime>
#include "vector-io.hpp"


template <typename MatrixType, typename VectorType, typename SolverTag, typename PrecondTag>
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

template <typename ScalarType>
void run_amg(viennacl::linalg::cg_tag & cg_solver,
             boost::numeric::ublas::vector<ScalarType> & ublas_vec,
             boost::numeric::ublas::vector<ScalarType> & ublas_result,
             boost::numeric::ublas::compressed_matrix<ScalarType> & ublas_matrix,
             viennacl::vector<ScalarType> & vcl_vec,
             viennacl::vector<ScalarType> & vcl_result,
             viennacl::compressed_matrix<ScalarType> & vcl_compressed_matrix,
             std::string info,
             viennacl::linalg::amg_tag & amg_tag)
{

  viennacl::linalg::amg_precond<boost::numeric::ublas::compressed_matrix<ScalarType> > ublas_amg = viennacl::linalg::amg_precond<boost::numeric::ublas::compressed_matrix<ScalarType> > (ublas_matrix, amg_tag);
  boost::numeric::ublas::vector<ScalarType> avgstencil;
  unsigned int coarselevels = amg_tag.get_coarselevels();

  std::cout << "-- CG with AMG preconditioner, " << info << " --" << std::endl;

  std::cout << " * Setup phase (ublas types)..." << std::endl;

  // Coarse level measure might have been changed during setup. Reload!
  ublas_amg.tag().set_coarselevels(coarselevels);
  ublas_amg.setup();

  std::cout << " * Operator complexity: " << ublas_amg.calc_complexity(avgstencil) << std::endl;

  amg_tag.set_coarselevels(coarselevels);
  viennacl::linalg::amg_precond<viennacl::compressed_matrix<ScalarType> > vcl_amg = viennacl::linalg::amg_precond<viennacl::compressed_matrix<ScalarType> > (vcl_compressed_matrix, amg_tag);
  std::cout << " * Setup phase (ViennaCL types)..." << std::endl;
  vcl_amg.tag().set_coarselevels(coarselevels);
  vcl_amg.setup();

  std::cout << " * CG solver (ublas types)..." << std::endl;
  run_solver(ublas_matrix, ublas_vec, ublas_result, cg_solver, ublas_amg);

  std::cout << " * CG solver (ViennaCL types)..." << std::endl;
  run_solver(vcl_compressed_matrix, vcl_vec, vcl_result, cg_solver, vcl_amg);

}

int main()
{
  //
  // Print some device info
  //
  std::cout << std::endl;
  std::cout << "----------------------------------------------" << std::endl;
  std::cout << "               Device Info" << std::endl;
  std::cout << "----------------------------------------------" << std::endl;

#ifdef VIENNACL_WITH_OPENCL
  std::cout << viennacl::ocl::current_device().info() << std::endl;
#endif

  typedef float    ScalarType;  // feel free to change this to double if supported by your device


  //
  // Set up the matrices and vectors for the iterative solvers (cf. iterative.cpp)
  //
  boost::numeric::ublas::vector<ScalarType> ublas_vec, ublas_result;
  boost::numeric::ublas::compressed_matrix<ScalarType> ublas_matrix;

  viennacl::linalg::cg_tag cg_solver;
  viennacl::linalg::amg_tag amg_tag;
  viennacl::linalg::amg_precond<boost::numeric::ublas::compressed_matrix<ScalarType> > ublas_amg;

  // Read matrix
  if (!viennacl::io::read_matrix_market_file(ublas_matrix, "../examples/testdata/mat65k.mtx"))
  {
    std::cout << "Error reading Matrix file" << std::endl;
    return EXIT_FAILURE;
  }

  // Set up rhs and result vector
  if (!readVectorFromFile("../examples/testdata/rhs65025.txt", ublas_vec))
  {
    std::cout << "Error reading RHS file" << std::endl;
    return 0;
  }

  if (!readVectorFromFile("../examples/testdata/result65025.txt", ublas_result))
  {
    std::cout << "Error reading Result file" << std::endl;
    return 0;
  }

  viennacl::vector<ScalarType> vcl_vec(ublas_vec.size());
  viennacl::vector<ScalarType> vcl_result(ublas_vec.size());
  viennacl::compressed_matrix<ScalarType> vcl_compressed_matrix(ublas_vec.size(), ublas_vec.size());

  // Copy to GPU
  viennacl::copy(ublas_matrix, vcl_compressed_matrix);
  viennacl::copy(ublas_vec, vcl_vec);
  viennacl::copy(ublas_result, vcl_result);

  //
  // Run solver without preconditioner
  //
  std::cout << "-- CG solver (CPU, no preconditioner) --" << std::endl;
  run_solver(ublas_matrix, ublas_vec, ublas_result, cg_solver, viennacl::linalg::no_precond());

  std::cout << "-- CG solver (GPU, no preconditioner) --" << std::endl;
  run_solver(vcl_compressed_matrix, vcl_vec, vcl_result, cg_solver, viennacl::linalg::no_precond());

  //
  // With AMG Preconditioner RS+DIRECT
  //
  amg_tag = viennacl::linalg::amg_tag(VIENNACL_AMG_COARSE_RS,       // coarsening strategy
                                      VIENNACL_AMG_INTERPOL_DIRECT, // interpolation strategy
                                      0.25, // strength of dependence threshold
                                      0.2,  // interpolation weight
                                      0.67, // jacobi smoother weight
                                      3,    // presmoothing steps
                                      3,    // postsmoothing steps
                                      0);   // number of coarse levels to be used (0: automatically use as many as reasonable)
  run_amg (cg_solver, ublas_vec, ublas_result, ublas_matrix, vcl_vec, vcl_result, vcl_compressed_matrix, "RS COARSENING, DIRECT INTERPOLATION", amg_tag);

  //
  // With AMG Preconditioner RS+CLASSIC
  //
  amg_tag = viennacl::linalg::amg_tag(VIENNACL_AMG_COARSE_RS, VIENNACL_AMG_INTERPOL_CLASSIC, 0.25, 0.2, 0.67, 3, 3, 0);
  run_amg ( cg_solver, ublas_vec, ublas_result, ublas_matrix, vcl_vec, vcl_result, vcl_compressed_matrix, "RS COARSENING, CLASSIC INTERPOLATION", amg_tag);

  //
  // With AMG Preconditioner ONEPASS+DIRECT
  //
  amg_tag = viennacl::linalg::amg_tag(VIENNACL_AMG_COARSE_ONEPASS, VIENNACL_AMG_INTERPOL_DIRECT,0.25, 0.2, 0.67, 3, 3, 0);
  run_amg (cg_solver, ublas_vec, ublas_result, ublas_matrix, vcl_vec, vcl_result, vcl_compressed_matrix, "ONEPASS COARSENING, DIRECT INTERPOLATION", amg_tag);

  //
  // With AMG Preconditioner RS0+DIRECT
  //
  amg_tag = viennacl::linalg::amg_tag(VIENNACL_AMG_COARSE_RS0, VIENNACL_AMG_INTERPOL_DIRECT, 0.25, 0.2, 0.67, 3, 3, 0);
  run_amg (cg_solver, ublas_vec, ublas_result, ublas_matrix, vcl_vec, vcl_result, vcl_compressed_matrix, "RS0 COARSENING, DIRECT INTERPOLATION", amg_tag);

  //
  // With AMG Preconditioner RS3+DIRECT
  //
  amg_tag = viennacl::linalg::amg_tag(VIENNACL_AMG_COARSE_RS3, VIENNACL_AMG_INTERPOL_DIRECT, 0.25, 0.2, 0.67, 3, 3, 0);
  run_amg (cg_solver, ublas_vec, ublas_result, ublas_matrix, vcl_vec, vcl_result, vcl_compressed_matrix, "RS3 COARSENING, DIRECT INTERPOLATION", amg_tag);

  //
  // With AMG Preconditioner AG
  //
  amg_tag = viennacl::linalg::amg_tag(VIENNACL_AMG_COARSE_AG, VIENNACL_AMG_INTERPOL_AG, 0.08, 0, 0.67, 3, 3, 0);
  run_amg (cg_solver, ublas_vec, ublas_result, ublas_matrix, vcl_vec, vcl_result, vcl_compressed_matrix, "AG COARSENING, AG INTERPOLATION", amg_tag);

  //
  // With AMG Preconditioner SA
  //
  amg_tag = viennacl::linalg::amg_tag(VIENNACL_AMG_COARSE_AG, VIENNACL_AMG_INTERPOL_SA, 0.08, 0.67, 0.67, 3, 3, 0);
  run_amg (cg_solver, ublas_vec, ublas_result, ublas_matrix, vcl_vec, vcl_result, vcl_compressed_matrix, "AG COARSENING, SA INTERPOLATION",amg_tag);


  //
  //  That's it.
  //
  std::cout << "!!!! TUTORIAL COMPLETED SUCCESSFULLY !!!!" << std::endl;

  return EXIT_SUCCESS;
}

