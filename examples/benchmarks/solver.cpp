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

/*
*
*   Benchmark:  Iterative solver tests (solver.cpp and solver.cu are identical, the latter being required for compilation using CUDA nvcc)
*
*/


#ifndef NDEBUG
 #define NDEBUG
#endif

#include <boost/numeric/ublas/matrix_sparse.hpp>
#include <boost/numeric/ublas/io.hpp>
#include <boost/numeric/ublas/operation_sparse.hpp>

#define VIENNACL_WITH_UBLAS 1

#include "viennacl/scalar.hpp"
#include "viennacl/vector.hpp"
#include "viennacl/coordinate_matrix.hpp"
#include "viennacl/compressed_matrix.hpp"
#include "viennacl/ell_matrix.hpp"
#include "viennacl/sliced_ell_matrix.hpp"
#include "viennacl/hyb_matrix.hpp"
#include "viennacl/context.hpp"

#include "viennacl/linalg/cg.hpp"
#include "viennacl/linalg/bicgstab.hpp"
#include "viennacl/linalg/gmres.hpp"
#include "viennacl/linalg/mixed_precision_cg.hpp"

#include "viennacl/linalg/ilu.hpp"
#include "viennacl/linalg/ichol.hpp"
#include "viennacl/linalg/jacobi_precond.hpp"
#include "viennacl/linalg/row_scaling.hpp"

#include "viennacl/io/matrix_market.hpp"
#include "viennacl/tools/timer.hpp"


#include <iostream>
#include <vector>


using namespace boost::numeric;

#define BENCHMARK_RUNS          1


inline void printOps(double num_ops, double exec_time)
{
  std::cout << "GFLOPs: " << num_ops / (1000000 * exec_time * 1000) << std::endl;
}


template<typename ScalarType>
ScalarType diff_inf(ublas::vector<ScalarType> & v1, viennacl::vector<ScalarType> & v2)
{
   ublas::vector<ScalarType> v2_cpu(v2.size());
   viennacl::copy(v2.begin(), v2.end(), v2_cpu.begin());

   for (unsigned int i=0;i<v1.size(); ++i)
   {
      if ( std::max( fabs(v2_cpu[i]), fabs(v1[i]) ) > 0 )
         v2_cpu[i] = fabs(v2_cpu[i] - v1[i]) / std::max( fabs(v2_cpu[i]), fabs(v1[i]) );
      else
         v2_cpu[i] = 0.0;
   }

   return norm_inf(v2_cpu);
}

template<typename ScalarType>
ScalarType diff_2(ublas::vector<ScalarType> & v1, viennacl::vector<ScalarType> & v2)
{
   ublas::vector<ScalarType> v2_cpu(v2.size());
   viennacl::copy(v2.begin(), v2.end(), v2_cpu.begin());

   return norm_2(v1 - v2_cpu) / norm_2(v1);
}


template<typename MatrixType, typename VectorType, typename SolverTag, typename PrecondTag>
void run_solver(MatrixType const & matrix, VectorType const & rhs, VectorType const & ref_result, SolverTag const & solver, PrecondTag const & precond, long ops)
{
  viennacl::tools::timer timer;
  VectorType result(rhs);
  VectorType residual(rhs);
  viennacl::backend::finish();

  timer.start();
  for (int runs=0; runs<BENCHMARK_RUNS; ++runs)
  {
    result = viennacl::linalg::solve(matrix, rhs, solver, precond);
  }
  viennacl::backend::finish();
  double exec_time = timer.get();
  std::cout << "Exec. time: " << exec_time << std::endl;
  std::cout << "Est. "; printOps(static_cast<double>(ops), exec_time / BENCHMARK_RUNS);
  residual -= viennacl::linalg::prod(matrix, result);
  std::cout << "Relative residual: " << viennacl::linalg::norm_2(residual) / viennacl::linalg::norm_2(rhs) << std::endl;
  std::cout << "Estimated rel. residual: " << solver.error() << std::endl;
  std::cout << "Iterations: " << solver.iters() << std::endl;
  result -= ref_result;
  std::cout << "Relative deviation from result: " << viennacl::linalg::norm_2(result) / viennacl::linalg::norm_2(ref_result) << std::endl;
}


template<typename ScalarType>
int run_benchmark(viennacl::context ctx)
{
  viennacl::tools::timer timer;
  double exec_time;

  ublas::vector<ScalarType> ublas_vec1;
  ublas::vector<ScalarType> ublas_vec2;
  ublas::vector<ScalarType> ublas_result;
  unsigned int solver_iters = 100;
  unsigned int solver_krylov_dim = 20;
  double solver_tolerance = 1e-6;

  ublas::compressed_matrix<ScalarType> ublas_matrix;
  if (!viennacl::io::read_matrix_market_file(ublas_matrix, "../examples/testdata/mat65k.mtx"))
  {
    std::cout << "Error reading Matrix file" << std::endl;
    return EXIT_FAILURE;
  }
  std::cout << "done reading matrix" << std::endl;

  ublas_result = ublas::scalar_vector<ScalarType>(ublas_matrix.size1(), ScalarType(1.0));
  ublas_vec1 = ublas::prod(ublas_matrix, ublas_result);
  ublas_vec2 = ublas_vec1;

  viennacl::compressed_matrix<ScalarType> vcl_compressed_matrix(ublas_vec1.size(), ublas_vec1.size(), ctx);
  viennacl::coordinate_matrix<ScalarType> vcl_coordinate_matrix(ublas_vec1.size(), ublas_vec1.size(), ctx);
  viennacl::ell_matrix<ScalarType> vcl_ell_matrix(ctx);
  viennacl::sliced_ell_matrix<ScalarType> vcl_sliced_ell_matrix(ctx);
  viennacl::hyb_matrix<ScalarType> vcl_hyb_matrix(ctx);

  viennacl::vector<ScalarType> vcl_vec1(ublas_vec1.size(), ctx);
  viennacl::vector<ScalarType> vcl_vec2(ublas_vec1.size(), ctx);
  viennacl::vector<ScalarType> vcl_result(ublas_vec1.size(), ctx);


  //cpu to gpu:
  viennacl::copy(ublas_matrix, vcl_compressed_matrix);
  viennacl::copy(ublas_matrix, vcl_coordinate_matrix);
  viennacl::copy(ublas_matrix, vcl_ell_matrix);
  viennacl::copy(ublas_matrix, vcl_sliced_ell_matrix);
  viennacl::copy(ublas_matrix, vcl_hyb_matrix);
  viennacl::copy(ublas_vec1, vcl_vec1);
  viennacl::copy(ublas_vec2, vcl_vec2);
  viennacl::copy(ublas_result, vcl_result);


  std::cout << "------- Jacobi preconditioner ----------" << std::endl;
  viennacl::linalg::jacobi_precond< ublas::compressed_matrix<ScalarType> >    ublas_jacobi(ublas_matrix, viennacl::linalg::jacobi_tag());
  viennacl::linalg::jacobi_precond< viennacl::compressed_matrix<ScalarType> > vcl_jacobi_csr(vcl_compressed_matrix, viennacl::linalg::jacobi_tag());
  viennacl::linalg::jacobi_precond< viennacl::coordinate_matrix<ScalarType> > vcl_jacobi_coo(vcl_coordinate_matrix, viennacl::linalg::jacobi_tag());

  std::cout << "------- Row-Scaling preconditioner ----------" << std::endl;
  viennacl::linalg::row_scaling< ublas::compressed_matrix<ScalarType> >    ublas_row_scaling(ublas_matrix, viennacl::linalg::row_scaling_tag(1));
  viennacl::linalg::row_scaling< viennacl::compressed_matrix<ScalarType> > vcl_row_scaling_csr(vcl_compressed_matrix, viennacl::linalg::row_scaling_tag(1));
  viennacl::linalg::row_scaling< viennacl::coordinate_matrix<ScalarType> > vcl_row_scaling_coo(vcl_coordinate_matrix, viennacl::linalg::row_scaling_tag(1));

  ///////////////////////////////////////////////////////////////////////////////
  //////////////////////  Incomplete Cholesky preconditioner   //////////////////
  ///////////////////////////////////////////////////////////////////////////////
  std::cout << "------- ICHOL0 on CPU (ublas) ----------" << std::endl;

  timer.start();
  viennacl::linalg::ichol0_precond< ublas::compressed_matrix<ScalarType> >    ublas_ichol0(ublas_matrix, viennacl::linalg::ichol0_tag());
  exec_time = timer.get();
  std::cout << "Setup time: " << exec_time << std::endl;

  timer.start();
  for (int runs=0; runs<BENCHMARK_RUNS; ++runs)
    ublas_ichol0.apply(ublas_vec1);
  exec_time = timer.get();
  std::cout << "ublas time: " << exec_time << std::endl;

  std::cout << "------- ICHOL0 with ViennaCL ----------" << std::endl;

  timer.start();
  viennacl::linalg::ichol0_precond< viennacl::compressed_matrix<ScalarType> > vcl_ichol0(vcl_compressed_matrix, viennacl::linalg::ichol0_tag());
  exec_time = timer.get();
  std::cout << "Setup time: " << exec_time << std::endl;

  viennacl::backend::finish();
  timer.start();
  for (int runs=0; runs<BENCHMARK_RUNS; ++runs)
    vcl_ichol0.apply(vcl_vec1);
  viennacl::backend::finish();
  exec_time = timer.get();
  std::cout << "ViennaCL time: " << exec_time << std::endl;

  std::cout << "------- Chow-Patel parallel ICC with ViennaCL ----------" << std::endl;

  timer.start();
  viennacl::linalg::chow_patel_icc_precond< viennacl::compressed_matrix<ScalarType> > vcl_chow_patel_icc(vcl_compressed_matrix, viennacl::linalg::chow_patel_tag());
  viennacl::backend::finish();
  std::cout << "Setup time: " << timer.get() << std::endl;

  timer.start();
  for (int runs=0; runs<BENCHMARK_RUNS; ++runs)
    vcl_chow_patel_icc.apply(vcl_vec1);
  viennacl::backend::finish();
  std::cout << "ViennaCL Chow-Patel-ILU substitution time: " << timer.get() << std::endl;


  ///////////////////////////////////////////////////////////////////////////////
  //////////////////////           ILU preconditioner         //////////////////
  ///////////////////////////////////////////////////////////////////////////////
  std::cout << "------- ILU0 on with ublas ----------" << std::endl;

  timer.start();
  viennacl::linalg::ilu0_precond< ublas::compressed_matrix<ScalarType> >    ublas_ilu0(ublas_matrix, viennacl::linalg::ilu0_tag());
  exec_time = timer.get();
  std::cout << "Setup time (no level scheduling): " << exec_time << std::endl;
  timer.start();
  for (int runs=0; runs<BENCHMARK_RUNS; ++runs)
    ublas_ilu0.apply(ublas_vec1);
  exec_time = timer.get();
  std::cout << "ublas ILU0 substitution time (no level scheduling): " << exec_time << std::endl;

  std::cout << "------- ILU0 with ViennaCL ----------" << std::endl;

  timer.start();
  viennacl::linalg::ilu0_precond< viennacl::compressed_matrix<ScalarType> > vcl_ilu0(vcl_compressed_matrix, viennacl::linalg::ilu0_tag());
  exec_time = timer.get();
  std::cout << "Setup time (no level scheduling): " << exec_time << std::endl;

  viennacl::backend::finish();
  timer.start();
  for (int runs=0; runs<BENCHMARK_RUNS; ++runs)
    vcl_ilu0.apply(vcl_vec1);
  viennacl::backend::finish();
  exec_time = timer.get();
  std::cout << "ViennaCL ILU0 substitution time (no level scheduling): " << exec_time << std::endl;

  timer.start();
  viennacl::linalg::ilu0_tag ilu0_with_level_scheduling; ilu0_with_level_scheduling.use_level_scheduling(true);
  viennacl::linalg::ilu0_precond< viennacl::compressed_matrix<ScalarType> > vcl_ilu0_level_scheduling(vcl_compressed_matrix, ilu0_with_level_scheduling);
  exec_time = timer.get();
  std::cout << "Setup time (with level scheduling): " << exec_time << std::endl;

  viennacl::backend::finish();
  timer.start();
  for (int runs=0; runs<BENCHMARK_RUNS; ++runs)
    vcl_ilu0_level_scheduling.apply(vcl_vec1);
  viennacl::backend::finish();
  exec_time = timer.get();
  std::cout << "ViennaCL ILU0 substitution time (with level scheduling): " << exec_time << std::endl;



  ////////////////////////////////////////////

  std::cout << "------- Block-ILU0 with ublas ----------" << std::endl;

  ublas_vec1 = ublas_vec2;
  viennacl::copy(ublas_vec1, vcl_vec1);

  timer.start();
  viennacl::linalg::block_ilu_precond< ublas::compressed_matrix<ScalarType>,
                                       viennacl::linalg::ilu0_tag>          ublas_block_ilu0(ublas_matrix, viennacl::linalg::ilu0_tag());
  exec_time = timer.get();
  std::cout << "Setup time: " << exec_time << std::endl;

  timer.start();
  for (int runs=0; runs<BENCHMARK_RUNS; ++runs)
    ublas_block_ilu0.apply(ublas_vec1);
  exec_time = timer.get();
  std::cout << "ublas time: " << exec_time << std::endl;

  std::cout << "------- Block-ILU0 with ViennaCL ----------" << std::endl;

  timer.start();
  viennacl::linalg::block_ilu_precond< viennacl::compressed_matrix<ScalarType>,
                                       viennacl::linalg::ilu0_tag>          vcl_block_ilu0(vcl_compressed_matrix, viennacl::linalg::ilu0_tag());
  exec_time = timer.get();
  std::cout << "Setup time: " << exec_time << std::endl;

  //vcl_block_ilu0.apply(vcl_vec1);  //warm-up
  viennacl::backend::finish();
  timer.start();
  for (int runs=0; runs<BENCHMARK_RUNS; ++runs)
    vcl_block_ilu0.apply(vcl_vec1);
  viennacl::backend::finish();
  exec_time = timer.get();
  std::cout << "ViennaCL time: " << exec_time << std::endl;

  ////////////////////////////////////////////

  std::cout << "------- ILUT with ublas ----------" << std::endl;

  ublas_vec1 = ublas_vec2;
  viennacl::copy(ublas_vec1, vcl_vec1);

  timer.start();
  viennacl::linalg::ilut_precond< ublas::compressed_matrix<ScalarType> >    ublas_ilut(ublas_matrix, viennacl::linalg::ilut_tag());
  exec_time = timer.get();
  std::cout << "Setup time (no level scheduling): " << exec_time << std::endl;
  timer.start();
  for (int runs=0; runs<BENCHMARK_RUNS; ++runs)
    ublas_ilut.apply(ublas_vec1);
  exec_time = timer.get();
  std::cout << "ublas ILUT substitution time (no level scheduling): " << exec_time << std::endl;


  std::cout << "------- ILUT with ViennaCL ----------" << std::endl;

  timer.start();
  viennacl::linalg::ilut_precond< viennacl::compressed_matrix<ScalarType> > vcl_ilut(vcl_compressed_matrix, viennacl::linalg::ilut_tag());
  exec_time = timer.get();
  std::cout << "Setup time (no level scheduling): " << exec_time << std::endl;

  viennacl::backend::finish();
  timer.start();
  for (int runs=0; runs<BENCHMARK_RUNS; ++runs)
    vcl_ilut.apply(vcl_vec1);
  viennacl::backend::finish();
  exec_time = timer.get();
  std::cout << "ViennaCL ILUT substitution time (no level scheduling): " << exec_time << std::endl;

  timer.start();
  viennacl::linalg::ilut_tag ilut_with_level_scheduling; ilut_with_level_scheduling.use_level_scheduling(true);
  viennacl::linalg::ilut_precond< viennacl::compressed_matrix<ScalarType> > vcl_ilut_level_scheduling(vcl_compressed_matrix, ilut_with_level_scheduling);
  exec_time = timer.get();
  std::cout << "Setup time (with level scheduling): " << exec_time << std::endl;

  viennacl::backend::finish();
  timer.start();
  for (int runs=0; runs<BENCHMARK_RUNS; ++runs)
    vcl_ilut_level_scheduling.apply(vcl_vec1);
  viennacl::backend::finish();
  exec_time = timer.get();
  std::cout << "ViennaCL ILUT substitution time (with level scheduling): " << exec_time << std::endl;


  ////////////////////////////////////////////

  std::cout << "------- Block-ILUT with ublas ----------" << std::endl;

  ublas_vec1 = ublas_vec2;
  viennacl::copy(ublas_vec1, vcl_vec1);

  timer.start();
  viennacl::linalg::block_ilu_precond< ublas::compressed_matrix<ScalarType>,
                                       viennacl::linalg::ilut_tag>          ublas_block_ilut(ublas_matrix, viennacl::linalg::ilut_tag());
  exec_time = timer.get();
  std::cout << "Setup time: " << exec_time << std::endl;

  //ublas_block_ilut.apply(ublas_vec1);
  timer.start();
  for (int runs=0; runs<BENCHMARK_RUNS; ++runs)
    ublas_block_ilut.apply(ublas_vec1);
  exec_time = timer.get();
  std::cout << "ublas time: " << exec_time << std::endl;

  std::cout << "------- Block-ILUT with ViennaCL ----------" << std::endl;

  timer.start();
  viennacl::linalg::block_ilu_precond< viennacl::compressed_matrix<ScalarType>,
                                       viennacl::linalg::ilut_tag>          vcl_block_ilut(vcl_compressed_matrix, viennacl::linalg::ilut_tag());
  exec_time = timer.get();
  std::cout << "Setup time: " << exec_time << std::endl;

  //vcl_block_ilut.apply(vcl_vec1);  //warm-up
  viennacl::backend::finish();
  timer.start();
  for (int runs=0; runs<BENCHMARK_RUNS; ++runs)
    vcl_block_ilut.apply(vcl_vec1);
  viennacl::backend::finish();
  exec_time = timer.get();
  std::cout << "ViennaCL time: " << exec_time << std::endl;

  std::cout << "------- Chow-Patel parallel ILU with ViennaCL ----------" << std::endl;

  timer.start();
  viennacl::linalg::chow_patel_ilu_precond< viennacl::compressed_matrix<ScalarType> > vcl_chow_patel_ilu(vcl_compressed_matrix, viennacl::linalg::chow_patel_tag());
  viennacl::backend::finish();
  std::cout << "Setup time: " << timer.get() << std::endl;

  timer.start();
  for (int runs=0; runs<BENCHMARK_RUNS; ++runs)
    vcl_chow_patel_ilu.apply(vcl_vec1);
  viennacl::backend::finish();
  std::cout << "ViennaCL Chow-Patel-ILU substitution time: " << timer.get() << std::endl;

  ///////////////////////////////////////////////////////////////////////////////
  //////////////////////              CG solver                //////////////////
  ///////////////////////////////////////////////////////////////////////////////
  long cg_ops = static_cast<long>(solver_iters * (ublas_matrix.nnz() + 6 * ublas_vec2.size()));

  viennacl::linalg::cg_tag cg_solver(solver_tolerance, solver_iters);

  std::cout << "------- CG solver (no preconditioner) using ublas ----------" << std::endl;
  run_solver(ublas_matrix, ublas_vec2, ublas_result, cg_solver, viennacl::linalg::no_precond(), cg_ops);

  std::cout << "------- CG solver (no preconditioner) via ViennaCL, compressed_matrix ----------" << std::endl;
  run_solver(vcl_compressed_matrix, vcl_vec2, vcl_result, cg_solver, viennacl::linalg::no_precond(), cg_ops);

  bool is_double = (sizeof(ScalarType) == sizeof(double));
  if (is_double)
  {
    std::cout << "------- CG solver, mixed precision (no preconditioner) via ViennaCL, compressed_matrix ----------" << std::endl;
    viennacl::linalg::mixed_precision_cg_tag mixed_precision_cg_solver(solver_tolerance, solver_iters);

    run_solver(vcl_compressed_matrix, vcl_vec2, vcl_result, mixed_precision_cg_solver, viennacl::linalg::no_precond(), cg_ops);
  }

  std::cout << "------- CG solver (no preconditioner) via ViennaCL, coordinate_matrix ----------" << std::endl;
  run_solver(vcl_coordinate_matrix, vcl_vec2, vcl_result, cg_solver, viennacl::linalg::no_precond(), cg_ops);

  std::cout << "------- CG solver (no preconditioner) via ViennaCL, ell_matrix ----------" << std::endl;
  run_solver(vcl_ell_matrix, vcl_vec2, vcl_result, cg_solver, viennacl::linalg::no_precond(), cg_ops);

  std::cout << "------- CG solver (no preconditioner) via ViennaCL, sliced_ell_matrix ----------" << std::endl;
  run_solver(vcl_sliced_ell_matrix, vcl_vec2, vcl_result, cg_solver, viennacl::linalg::no_precond(), cg_ops);

  std::cout << "------- CG solver (no preconditioner) via ViennaCL, hyb_matrix ----------" << std::endl;
  run_solver(vcl_hyb_matrix, vcl_vec2, vcl_result, cg_solver, viennacl::linalg::no_precond(), cg_ops);

  std::cout << "------- CG solver (ICHOL0 preconditioner) using ublas ----------" << std::endl;
  run_solver(ublas_matrix, ublas_vec2, ublas_result, cg_solver, ublas_ichol0, cg_ops);

  std::cout << "------- CG solver (ICHOL0 preconditioner) via ViennaCL, compressed_matrix ----------" << std::endl;
  run_solver(vcl_compressed_matrix, vcl_vec2, vcl_result, cg_solver, vcl_ichol0, cg_ops);

  std::cout << "------- CG solver (Chow-Patel ICHOL0 preconditioner) via ViennaCL, compressed_matrix ----------" << std::endl;
  run_solver(vcl_compressed_matrix, vcl_vec2, vcl_result, cg_solver, vcl_chow_patel_icc, cg_ops);

  std::cout << "------- CG solver (ILU0 preconditioner) using ublas ----------" << std::endl;
  run_solver(ublas_matrix, ublas_vec2, ublas_result, cg_solver, ublas_ilu0, cg_ops);

  std::cout << "------- CG solver (ILU0 preconditioner) via ViennaCL, compressed_matrix ----------" << std::endl;
  run_solver(vcl_compressed_matrix, vcl_vec2, vcl_result, cg_solver, vcl_ilu0, cg_ops);


  std::cout << "------- CG solver (Block-ILU0 preconditioner) using ublas ----------" << std::endl;
  run_solver(ublas_matrix, ublas_vec2, ublas_result, cg_solver, ublas_block_ilu0, cg_ops);

  std::cout << "------- CG solver (Block-ILU0 preconditioner) via ViennaCL, compressed_matrix ----------" << std::endl;
  run_solver(vcl_compressed_matrix, vcl_vec2, vcl_result, cg_solver, vcl_block_ilu0, cg_ops);

  std::cout << "------- CG solver (ILUT preconditioner) using ublas ----------" << std::endl;
  run_solver(ublas_matrix, ublas_vec2, ublas_result, cg_solver, ublas_ilut, cg_ops);

  std::cout << "------- CG solver (ILUT preconditioner) via ViennaCL, compressed_matrix ----------" << std::endl;
  run_solver(vcl_compressed_matrix, vcl_vec2, vcl_result, cg_solver, vcl_ilut, cg_ops);

  std::cout << "------- CG solver (ILUT preconditioner) via ViennaCL, coordinate_matrix ----------" << std::endl;
  run_solver(vcl_coordinate_matrix, vcl_vec2, vcl_result, cg_solver, vcl_ilut, cg_ops);

  std::cout << "------- CG solver (Block-ILUT preconditioner) using ublas ----------" << std::endl;
  run_solver(ublas_matrix, ublas_vec2, ublas_result, cg_solver, ublas_block_ilut, cg_ops);

  std::cout << "------- CG solver (Block-ILUT preconditioner) via ViennaCL, compressed_matrix ----------" << std::endl;
  run_solver(vcl_compressed_matrix, vcl_vec2, vcl_result, cg_solver, vcl_block_ilut, cg_ops);

  std::cout << "------- CG solver (Jacobi preconditioner) using ublas ----------" << std::endl;
  run_solver(ublas_matrix, ublas_vec2, ublas_result, cg_solver, ublas_jacobi, cg_ops);

  std::cout << "------- CG solver (Jacobi preconditioner) via ViennaCL, compressed_matrix ----------" << std::endl;
  run_solver(vcl_compressed_matrix, vcl_vec2, vcl_result, cg_solver, vcl_jacobi_csr, cg_ops);

  std::cout << "------- CG solver (Jacobi preconditioner) via ViennaCL, coordinate_matrix ----------" << std::endl;
  run_solver(vcl_coordinate_matrix, vcl_vec2, vcl_result, cg_solver, vcl_jacobi_coo, cg_ops);


  std::cout << "------- CG solver (row scaling preconditioner) using ublas ----------" << std::endl;
  run_solver(ublas_matrix, ublas_vec2, ublas_result, cg_solver, ublas_row_scaling, cg_ops);

  std::cout << "------- CG solver (row scaling preconditioner) via ViennaCL, compressed_matrix ----------" << std::endl;
  run_solver(vcl_compressed_matrix, vcl_vec2, vcl_result, cg_solver, vcl_row_scaling_csr, cg_ops);

  std::cout << "------- CG solver (row scaling preconditioner) via ViennaCL, coordinate_matrix ----------" << std::endl;
  run_solver(vcl_coordinate_matrix, vcl_vec2, vcl_result, cg_solver, vcl_row_scaling_coo, cg_ops);

  ///////////////////////////////////////////////////////////////////////////////
  //////////////////////           BiCGStab solver             //////////////////
  ///////////////////////////////////////////////////////////////////////////////

  long bicgstab_ops = static_cast<long>(solver_iters * (2 * ublas_matrix.nnz() + 13 * ublas_vec2.size()));

  viennacl::linalg::bicgstab_tag bicgstab_solver(solver_tolerance, solver_iters);

  std::cout << "------- BiCGStab solver (no preconditioner) using ublas ----------" << std::endl;
  run_solver(ublas_matrix, ublas_vec2, ublas_result, bicgstab_solver, viennacl::linalg::no_precond(), bicgstab_ops);

  std::cout << "------- BiCGStab solver (no preconditioner) via ViennaCL, compressed_matrix ----------" << std::endl;
  run_solver(vcl_compressed_matrix, vcl_vec2, vcl_result, bicgstab_solver, viennacl::linalg::no_precond(), bicgstab_ops);

  std::cout << "------- BiCGStab solver (ILU0 preconditioner) via ViennaCL, compressed_matrix ----------" << std::endl;
  run_solver(vcl_compressed_matrix, vcl_vec2, vcl_result, bicgstab_solver, vcl_ilu0, bicgstab_ops);

  std::cout << "------- BiCGStab solver (Chow-Patel-ILU preconditioner) via ViennaCL, compressed_matrix ----------" << std::endl;
  run_solver(vcl_compressed_matrix, vcl_vec2, vcl_result, bicgstab_solver, vcl_chow_patel_ilu, bicgstab_ops);

  std::cout << "------- BiCGStab solver (no preconditioner) via ViennaCL, coordinate_matrix ----------" << std::endl;
  run_solver(vcl_coordinate_matrix, vcl_vec2, vcl_result, bicgstab_solver, viennacl::linalg::no_precond(), bicgstab_ops);

  std::cout << "------- BiCGStab solver (no preconditioner) via ViennaCL, ell_matrix ----------" << std::endl;
  run_solver(vcl_ell_matrix, vcl_vec2, vcl_result, bicgstab_solver, viennacl::linalg::no_precond(), bicgstab_ops);

  std::cout << "------- BiCGStab solver (no preconditioner) via ViennaCL, sliced_ell_matrix ----------" << std::endl;
  run_solver(vcl_sliced_ell_matrix, vcl_vec2, vcl_result, bicgstab_solver, viennacl::linalg::no_precond(), bicgstab_ops);

  std::cout << "------- BiCGStab solver (no preconditioner) via ViennaCL, hyb_matrix ----------" << std::endl;
  run_solver(vcl_hyb_matrix, vcl_vec2, vcl_result, bicgstab_solver, viennacl::linalg::no_precond(), bicgstab_ops);

  std::cout << "------- BiCGStab solver (ILUT preconditioner) using ublas ----------" << std::endl;
  run_solver(ublas_matrix, ublas_vec2, ublas_result, bicgstab_solver, ublas_ilut, bicgstab_ops);

  std::cout << "------- BiCGStab solver (ILUT preconditioner) via ViennaCL, compressed_matrix ----------" << std::endl;
  run_solver(vcl_compressed_matrix, vcl_vec2, vcl_result, bicgstab_solver, vcl_ilut, bicgstab_ops);

  std::cout << "------- BiCGStab solver (Block-ILUT preconditioner) using ublas ----------" << std::endl;
  run_solver(ublas_matrix, ublas_vec2, ublas_result, bicgstab_solver, ublas_block_ilut, bicgstab_ops);

#ifdef VIENNACL_WITH_OPENCL
  std::cout << "------- BiCGStab solver (Block-ILUT preconditioner) via ViennaCL, compressed_matrix ----------" << std::endl;
  run_solver(vcl_compressed_matrix, vcl_vec2, vcl_result, bicgstab_solver, vcl_block_ilut, bicgstab_ops);
#endif

//  std::cout << "------- BiCGStab solver (ILUT preconditioner) via ViennaCL, coordinate_matrix ----------" << std::endl;
//  run_solver(vcl_coordinate_matrix, vcl_vec2, vcl_result, bicgstab_solver, vcl_ilut, bicgstab_ops);

  std::cout << "------- BiCGStab solver (Jacobi preconditioner) using ublas ----------" << std::endl;
  run_solver(ublas_matrix, ublas_vec2, ublas_result, bicgstab_solver, ublas_jacobi, bicgstab_ops);

  std::cout << "------- BiCGStab solver (Jacobi preconditioner) via ViennaCL, compressed_matrix ----------" << std::endl;
  run_solver(vcl_compressed_matrix, vcl_vec2, vcl_result, bicgstab_solver, vcl_jacobi_csr, bicgstab_ops);

  std::cout << "------- BiCGStab solver (Jacobi preconditioner) via ViennaCL, coordinate_matrix ----------" << std::endl;
  run_solver(vcl_coordinate_matrix, vcl_vec2, vcl_result, bicgstab_solver, vcl_jacobi_coo, bicgstab_ops);


  std::cout << "------- BiCGStab solver (row scaling preconditioner) using ublas ----------" << std::endl;
  run_solver(ublas_matrix, ublas_vec2, ublas_result, bicgstab_solver, ublas_row_scaling, bicgstab_ops);

  std::cout << "------- BiCGStab solver (row scaling preconditioner) via ViennaCL, compressed_matrix ----------" << std::endl;
  run_solver(vcl_compressed_matrix, vcl_vec2, vcl_result, bicgstab_solver, vcl_row_scaling_csr, bicgstab_ops);

  std::cout << "------- BiCGStab solver (row scaling preconditioner) via ViennaCL, coordinate_matrix ----------" << std::endl;
  run_solver(vcl_coordinate_matrix, vcl_vec2, vcl_result, bicgstab_solver, vcl_row_scaling_coo, bicgstab_ops);


  ///////////////////////////////////////////////////////////////////////////////
  ///////////////////////            GMRES solver             ///////////////////
  ///////////////////////////////////////////////////////////////////////////////

  long gmres_ops = static_cast<long>(solver_iters * (ublas_matrix.nnz() + (solver_iters * 2 + 7) * ublas_vec2.size()));

  viennacl::linalg::gmres_tag gmres_solver(solver_tolerance, solver_iters, solver_krylov_dim);

  std::cout << "------- GMRES solver (no preconditioner) using ublas ----------" << std::endl;
  run_solver(ublas_matrix, ublas_vec2, ublas_result, gmres_solver, viennacl::linalg::no_precond(), gmres_ops);

  std::cout << "------- GMRES solver (no preconditioner) via ViennaCL, compressed_matrix ----------" << std::endl;
  run_solver(vcl_compressed_matrix, vcl_vec2, vcl_result, gmres_solver, viennacl::linalg::no_precond(), gmres_ops);

  std::cout << "------- GMRES solver (no preconditioner) on GPU, coordinate_matrix ----------" << std::endl;
  run_solver(vcl_coordinate_matrix, vcl_vec2, vcl_result, gmres_solver, viennacl::linalg::no_precond(), gmres_ops);

  std::cout << "------- GMRES solver (no preconditioner) on GPU, ell_matrix ----------" << std::endl;
  run_solver(vcl_ell_matrix, vcl_vec2, vcl_result, gmres_solver, viennacl::linalg::no_precond(), gmres_ops);

  std::cout << "------- GMRES solver (no preconditioner) on GPU, sliced_ell_matrix ----------" << std::endl;
  run_solver(vcl_sliced_ell_matrix, vcl_vec2, vcl_result, gmres_solver, viennacl::linalg::no_precond(), gmres_ops);

  std::cout << "------- GMRES solver (no preconditioner) on GPU, hyb_matrix ----------" << std::endl;
  run_solver(vcl_hyb_matrix, vcl_vec2, vcl_result, gmres_solver, viennacl::linalg::no_precond(), gmres_ops);

  std::cout << "------- GMRES solver (ILUT preconditioner) using ublas ----------" << std::endl;
  run_solver(ublas_matrix, ublas_vec2, ublas_result, gmres_solver, ublas_ilut, gmres_ops);

  std::cout << "------- GMRES solver (ILUT preconditioner) via ViennaCL, compressed_matrix ----------" << std::endl;
  run_solver(vcl_compressed_matrix, vcl_vec2, vcl_result, gmres_solver, vcl_ilut, gmres_ops);

  std::cout << "------- GMRES solver (ILUT preconditioner) via ViennaCL, coordinate_matrix ----------" << std::endl;
  run_solver(vcl_coordinate_matrix, vcl_vec2, vcl_result, gmres_solver, vcl_ilut, gmres_ops);


  std::cout << "------- GMRES solver (Jacobi preconditioner) using ublas ----------" << std::endl;
  run_solver(ublas_matrix, ublas_vec2, ublas_result, gmres_solver, ublas_jacobi, gmres_ops);

  std::cout << "------- GMRES solver (Jacobi preconditioner) via ViennaCL, compressed_matrix ----------" << std::endl;
  run_solver(vcl_compressed_matrix, vcl_vec2, vcl_result, gmres_solver, vcl_jacobi_csr, gmres_ops);

  std::cout << "------- GMRES solver (Jacobi preconditioner) via ViennaCL, coordinate_matrix ----------" << std::endl;
  run_solver(vcl_coordinate_matrix, vcl_vec2, vcl_result, gmres_solver, vcl_jacobi_coo, gmres_ops);


  std::cout << "------- GMRES solver (row scaling preconditioner) using ublas ----------" << std::endl;
  run_solver(ublas_matrix, ublas_vec2, ublas_result, gmres_solver, ublas_row_scaling, gmres_ops);

  std::cout << "------- GMRES solver (row scaling preconditioner) via ViennaCL, compressed_matrix ----------" << std::endl;
  run_solver(vcl_compressed_matrix, vcl_vec2, vcl_result, gmres_solver, vcl_row_scaling_csr, gmres_ops);

  std::cout << "------- GMRES solver (row scaling preconditioner) via ViennaCL, coordinate_matrix ----------" << std::endl;
  run_solver(vcl_coordinate_matrix, vcl_vec2, vcl_result, gmres_solver, vcl_row_scaling_coo, gmres_ops);

  return EXIT_SUCCESS;
}

int main()
{
  std::cout << std::endl;
  std::cout << "----------------------------------------------" << std::endl;
  std::cout << "               Device Info" << std::endl;
  std::cout << "----------------------------------------------" << std::endl;

#ifdef VIENNACL_WITH_OPENCL
  viennacl::ocl::platform pf = viennacl::ocl::get_platforms()[0];
  std::vector<viennacl::ocl::device> const & devices = pf.devices();

  // Set first device to first context:
  viennacl::ocl::setup_context(0, devices[0]);

  // Set second device for second context (use the same device for the second context if only one device available):
  if (devices.size() > 1)
    viennacl::ocl::setup_context(1, devices[1]);
  else
    viennacl::ocl::setup_context(1, devices[0]);

  std::cout << viennacl::ocl::current_device().info() << std::endl;
  viennacl::context ctx(viennacl::ocl::get_context(1));
#else
  viennacl::context ctx;
#endif

  std::cout << "---------------------------------------------------------------------------" << std::endl;
  std::cout << "---------------------------------------------------------------------------" << std::endl;
  std::cout << " Benchmark for Execution Times of Iterative Solvers provided with ViennaCL " << std::endl;
  std::cout << "---------------------------------------------------------------------------" << std::endl;
  std::cout << " Note that the purpose of this benchmark is not to run solvers until" << std::endl;
  std::cout << " convergence. Instead, only the execution times of a few iterations are" << std::endl;
  std::cout << " recorded. Residual errors are only printed for information." << std::endl << std::endl;


  std::cout << std::endl;
  std::cout << "----------------------------------------------" << std::endl;
  std::cout << "----------------------------------------------" << std::endl;
  std::cout << "## Benchmark :: Solver" << std::endl;
  std::cout << "----------------------------------------------" << std::endl;
  std::cout << std::endl;
  std::cout << "   -------------------------------" << std::endl;
  std::cout << "   # benchmarking single-precision" << std::endl;
  std::cout << "   -------------------------------" << std::endl;
  run_benchmark<float>(ctx);
#ifdef VIENNACL_WITH_OPENCL
  if ( viennacl::ocl::current_device().double_support() )
#endif
  {
    std::cout << std::endl;
    std::cout << "   -------------------------------" << std::endl;
    std::cout << "   # benchmarking double-precision" << std::endl;
    std::cout << "   -------------------------------" << std::endl;
    run_benchmark<double>(ctx);
  }
  return 0;
}

