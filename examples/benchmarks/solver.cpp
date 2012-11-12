/* =========================================================================
   Copyright (c) 2010-2012, Institute for Microelectronics,
                            Institute for Analysis and Scientific Computing,
                            TU Wien.

                            -----------------
                  ViennaCL - The Vienna Computing Library
                            -----------------

   Project Head:    Karl Rupp                   rupp@iue.tuwien.ac.at
               
   (A list of authors and contributors can be found in the PDF manual)

   License:         MIT (X11), see file LICENSE in the base directory
============================================================================= */

#ifndef NDEBUG
 #define NDEBUG
#endif

#include <boost/numeric/ublas/matrix_sparse.hpp>
#include <boost/numeric/ublas/operation_sparse.hpp>

#define VIENNACL_WITH_UBLAS 1

#include "viennacl/scalar.hpp"
#include "viennacl/vector.hpp"
#include "viennacl/coordinate_matrix.hpp"
#include "viennacl/compressed_matrix.hpp"
#include "viennacl/ell_matrix.hpp"
#include "viennacl/hyb_matrix.hpp"
#include "viennacl/linalg/ilu.hpp"
#include "viennacl/linalg/cg.hpp"
#include "viennacl/linalg/bicgstab.hpp"
#include "viennacl/linalg/gmres.hpp"
#include "viennacl/io/matrix_market.hpp"

#ifdef VIENNACL_WITH_OPENCL
  #include "viennacl/linalg/mixed_precision_cg.hpp"
  #include "viennacl/linalg/detail/ilu/opencl_block_ilu.hpp"  //preview feature: OpenCL accelerated block-ILU
  #include "viennacl/linalg/jacobi_precond.hpp"
  #include "viennacl/linalg/row_scaling.hpp"
#endif  


#include <iostream>
#include <vector>
#include "benchmark-utils.hpp"
#include "io.hpp"


using namespace boost::numeric;

/*
*   Benchmark:
*   Iterative solver tests
*   
*/

#define BENCHMARK_RUNS          1


template <typename ScalarType>
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

template <typename ScalarType>
ScalarType diff_2(ublas::vector<ScalarType> & v1, viennacl::vector<ScalarType> & v2)
{
   ublas::vector<ScalarType> v2_cpu(v2.size());
   viennacl::copy(v2.begin(), v2.end(), v2_cpu.begin());

   return norm_2(v1 - v2_cpu) / norm_2(v1);
}


template <typename MatrixType, typename VectorType, typename SolverTag, typename PrecondTag>
void run_solver(MatrixType const & matrix, VectorType const & rhs, VectorType const & ref_result, SolverTag const & solver, PrecondTag const & precond, long ops)
{
  Timer timer;
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
  std::cout << "Est. "; printOps(ops, exec_time / BENCHMARK_RUNS);
  residual -= viennacl::linalg::prod(matrix, result);
  std::cout << "Relative residual: " << viennacl::linalg::norm_2(residual) / viennacl::linalg::norm_2(rhs) << std::endl;
  std::cout << "Estimated rel. residual: " << solver.error() << std::endl;
  std::cout << "Iterations: " << solver.iters() << std::endl;
  result -= ref_result;
  std::cout << "Relative deviation from result: " << viennacl::linalg::norm_2(result) / viennacl::linalg::norm_2(ref_result) << std::endl;
}


template<typename ScalarType>
int run_benchmark()
{
  
  Timer timer;
  double exec_time;
   
  ScalarType std_factor1 = static_cast<ScalarType>(3.1415);
  ScalarType std_factor2 = static_cast<ScalarType>(42.0);
  viennacl::scalar<ScalarType> vcl_factor1(std_factor1);
  viennacl::scalar<ScalarType> vcl_factor2(std_factor2);
  
  ublas::vector<ScalarType> ublas_vec1;
  ublas::vector<ScalarType> ublas_vec2;
  ublas::vector<ScalarType> ublas_result;
  unsigned int solver_iters = 100;
  unsigned int solver_krylov_dim = 20;
  double solver_tolerance = 1e-6;

  #ifdef _MSC_VER
  if (!readVectorFromFile<ScalarType>("../../examples/testdata/rhs65025.txt", ublas_vec1))
  #else
  if (!readVectorFromFile<ScalarType>("../examples/testdata/rhs65025.txt", ublas_vec1))
  #endif
  {
    std::cout << "Error reading RHS file" << std::endl;
    return 0;
  }
  std::cout << "done reading rhs" << std::endl;
  ublas_vec2 = ublas_vec1;
  #ifdef _MSC_VER
  if (!readVectorFromFile<ScalarType>("../../examples/testdata/result65025.txt", ublas_result))
  #else
  if (!readVectorFromFile<ScalarType>("../examples/testdata/result65025.txt", ublas_result))
  #endif
  {
    std::cout << "Error reading result file" << std::endl;
    return 0;
  }
  std::cout << "done reading result" << std::endl;
  
  viennacl::compressed_matrix<ScalarType> vcl_compressed_matrix(ublas_vec1.size(), ublas_vec1.size());
  viennacl::coordinate_matrix<ScalarType> vcl_coordinate_matrix(ublas_vec1.size(), ublas_vec1.size());
  viennacl::ell_matrix<ScalarType> vcl_ell_matrix;
  viennacl::hyb_matrix<ScalarType> vcl_hyb_matrix;

  viennacl::vector<ScalarType> vcl_vec1(ublas_vec1.size());
  viennacl::vector<ScalarType> vcl_vec2(ublas_vec1.size()); 
  viennacl::vector<ScalarType> vcl_result(ublas_vec1.size()); 
  

  ublas::compressed_matrix<ScalarType> ublas_matrix;
  #ifdef _MSC_VER
  if (!viennacl::io::read_matrix_market_file(ublas_matrix, "../../examples/testdata/mat65k.mtx"))
  #else
  if (!viennacl::io::read_matrix_market_file(ublas_matrix, "../examples/testdata/mat65k.mtx"))
  #endif
  {
    std::cout << "Error reading Matrix file" << std::endl;
    return EXIT_FAILURE;
  }
  //unsigned int cg_mat_size = cg_mat.size(); 
  std::cout << "done reading matrix" << std::endl;
  
  //cpu to gpu:
  viennacl::copy(ublas_matrix, vcl_compressed_matrix);
  viennacl::copy(ublas_matrix, vcl_coordinate_matrix);
  viennacl::copy(ublas_matrix, vcl_ell_matrix);
  viennacl::copy(ublas_matrix, vcl_hyb_matrix);
  viennacl::copy(ublas_vec1, vcl_vec1);
  viennacl::copy(ublas_vec2, vcl_vec2);
  viennacl::copy(ublas_result, vcl_result);
  
  
#ifdef VIENNACL_WITH_OPENCL
  viennacl::linalg::jacobi_precond< ublas::compressed_matrix<ScalarType> >    ublas_jacobi(ublas_matrix, viennacl::linalg::jacobi_tag());
  viennacl::linalg::jacobi_precond< viennacl::compressed_matrix<ScalarType> > vcl_jacobi(vcl_compressed_matrix, viennacl::linalg::jacobi_tag());
  
  viennacl::linalg::row_scaling< ublas::compressed_matrix<ScalarType> >    ublas_row_scaling(ublas_matrix, viennacl::linalg::row_scaling_tag(1));
  viennacl::linalg::row_scaling< viennacl::compressed_matrix<ScalarType> > vcl_row_scaling(vcl_compressed_matrix, viennacl::linalg::row_scaling_tag(1));
#endif
  
  ///////////////////////////////////////////////////////////////////////////////
  //////////////////////           ILU preconditioner         //////////////////
  ///////////////////////////////////////////////////////////////////////////////
  std::cout << "------- ILU0 on CPU (ublas) ----------" << std::endl;

  timer.start();
  viennacl::linalg::ilu0_precond< ublas::compressed_matrix<ScalarType> >    ublas_ilu0(ublas_matrix, viennacl::linalg::ilu0_tag());
  exec_time = timer.get();
  std::cout << "Setup time: " << exec_time << std::endl;
  
  timer.start();
  for (int runs=0; runs<BENCHMARK_RUNS; ++runs)
    ublas_ilu0.apply(ublas_vec1);
  exec_time = timer.get();
  std::cout << "ublas time: " << exec_time << std::endl;
  
  std::cout << "------- ILU0 with ViennaCL ----------" << std::endl;

  timer.start();
  viennacl::linalg::ilu0_precond< viennacl::compressed_matrix<ScalarType> > vcl_ilu0(vcl_compressed_matrix, viennacl::linalg::ilu0_tag());
  exec_time = timer.get();
  std::cout << "Setup time: " << exec_time << std::endl;
  
  viennacl::backend::finish();
  timer.start();
  for (int runs=0; runs<BENCHMARK_RUNS; ++runs)
    vcl_ilu0.apply(vcl_vec1);
  viennacl::backend::finish();
  exec_time = timer.get();
  std::cout << "ViennaCL time: " << exec_time << std::endl;
  
  ////////////////////////////////////////////

  std::cout << "------- Block-ILU0 on CPU (ublas) ----------" << std::endl;

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
  
#ifdef VIENNACL_WITH_OPENCL
  std::cout << "------- Block-ILU0 with ViennaCL ----------" << std::endl;

  timer.start();
  viennacl::linalg::block_ilu_precond< viennacl::compressed_matrix<ScalarType>,
                                       viennacl::linalg::ilu0_tag>          vcl_block_ilu0(vcl_compressed_matrix, viennacl::linalg::ilu0_tag());
  exec_time = timer.get();
  std::cout << "Setup time: " << exec_time << std::endl;
  
  //vcl_block_ilu0.apply(vcl_vec1);  //warm-up
  viennacl::ocl::get_queue().finish();
  timer.start();
  for (int runs=0; runs<BENCHMARK_RUNS; ++runs)
    vcl_block_ilu0.apply(vcl_vec1);
  viennacl::ocl::get_queue().finish();
  exec_time = timer.get();
  std::cout << "ViennaCL time: " << exec_time << std::endl;
#endif
  
  ////////////////////////////////////////////
  
  std::cout << "------- ILUT on CPU (ublas) ----------" << std::endl;

  ublas_vec1 = ublas_vec2;
  viennacl::copy(ublas_vec1, vcl_vec1);
  
  timer.start();
  viennacl::linalg::ilut_precond< ublas::compressed_matrix<ScalarType> >    ublas_ilut(ublas_matrix, viennacl::linalg::ilut_tag());
  exec_time = timer.get();
  std::cout << "Setup time: " << exec_time << std::endl;
  
  ublas_ilut.apply(ublas_vec1);
  timer.start();
  for (int runs=0; runs<BENCHMARK_RUNS; ++runs)
    ublas_ilut.apply(ublas_vec1);
  exec_time = timer.get();
  std::cout << "ublas time: " << exec_time << std::endl;

  std::cout << "------- ILUT with ViennaCL ----------" << std::endl;

  timer.start();
  viennacl::linalg::ilut_precond< viennacl::compressed_matrix<ScalarType> > vcl_ilut(vcl_compressed_matrix, viennacl::linalg::ilut_tag());
  exec_time = timer.get();
  std::cout << "Setup time: " << exec_time << std::endl;
  
  vcl_ilut.apply(vcl_vec1);
  viennacl::backend::finish();
  timer.start();
  for (int runs=0; runs<BENCHMARK_RUNS; ++runs)
    vcl_ilut.apply(vcl_vec1);
  viennacl::backend::finish();
  exec_time = timer.get();
  std::cout << "ViennaCL time: " << exec_time << std::endl;

  ////////////////////////////////////////////

  std::cout << "------- Block-ILUT on CPU (ublas) ----------" << std::endl;

  ublas_vec1 = ublas_vec2;
  viennacl::copy(ublas_vec1, vcl_vec1);

  timer.start();
  viennacl::linalg::block_ilu_precond< ublas::compressed_matrix<ScalarType>,
                                       viennacl::linalg::ilut_tag>          ublas_block_ilut(ublas_matrix, viennacl::linalg::ilut_tag());
  exec_time = timer.get();
  std::cout << "Setup time: " << exec_time << std::endl;
  
  ublas_block_ilut.apply(ublas_vec1);
  timer.start();
  for (int runs=0; runs<BENCHMARK_RUNS; ++runs)
    ublas_block_ilut.apply(ublas_vec1);
  exec_time = timer.get();
  std::cout << "ublas time: " << exec_time << std::endl;
  
#ifdef VIENNACL_WITH_OPENCL
  std::cout << "------- Block-ILUT with ViennaCL ----------" << std::endl;

  timer.start();
  viennacl::linalg::block_ilu_precond< viennacl::compressed_matrix<ScalarType>,
                                       viennacl::linalg::ilut_tag>          vcl_block_ilut(vcl_compressed_matrix, viennacl::linalg::ilut_tag());
  exec_time = timer.get();
  std::cout << "Setup time: " << exec_time << std::endl;
  
  vcl_block_ilut.apply(vcl_vec1);  //warm-up
  viennacl::ocl::get_queue().finish();
  timer.start();
  for (int runs=0; runs<BENCHMARK_RUNS; ++runs)
    vcl_block_ilut.apply(vcl_vec1);
  viennacl::ocl::get_queue().finish();
  exec_time = timer.get();
  std::cout << "ViennaCL time: " << exec_time << std::endl;
#endif  
  
  ///////////////////////////////////////////////////////////////////////////////
  //////////////////////              CG solver                //////////////////
  ///////////////////////////////////////////////////////////////////////////////
  long cg_ops = static_cast<long>(solver_iters * (ublas_matrix.nnz() + 6 * ublas_vec2.size()));
  
  viennacl::linalg::cg_tag cg_solver(solver_tolerance, solver_iters);
  
  std::cout << "------- CG solver (no preconditioner) using ublas ----------" << std::endl;
  run_solver(ublas_matrix, ublas_vec2, ublas_result, cg_solver, viennacl::linalg::no_precond(), cg_ops);
  
  std::cout << "------- CG solver (no preconditioner) via ViennaCL, compressed_matrix ----------" << std::endl;
  run_solver(vcl_compressed_matrix, vcl_vec2, vcl_result, cg_solver, viennacl::linalg::no_precond(), cg_ops);

#ifdef VIENNACL_WITH_OPENCL
  std::cout << "------- CG solver, mixed precision (no preconditioner) via ViennaCL, compressed_matrix ----------" << std::endl;
  viennacl::linalg::mixed_precision_cg_tag mixed_precision_cg_solver(solver_tolerance, solver_iters);
  
  run_solver(vcl_compressed_matrix, vcl_vec2, vcl_result, mixed_precision_cg_solver, viennacl::linalg::no_precond(), cg_ops);
  run_solver(vcl_compressed_matrix, vcl_vec2, vcl_result, mixed_precision_cg_solver, viennacl::linalg::no_precond(), cg_ops);
#endif 
  
  std::cout << "------- CG solver (no preconditioner) via ViennaCL, coordinate_matrix ----------" << std::endl;
  run_solver(vcl_coordinate_matrix, vcl_vec2, vcl_result, cg_solver, viennacl::linalg::no_precond(), cg_ops);

  std::cout << "------- CG solver (no preconditioner) via ViennaCL, ell_matrix ----------" << std::endl;
  run_solver(vcl_ell_matrix, vcl_vec2, vcl_result, cg_solver, viennacl::linalg::no_precond(), cg_ops);

  std::cout << "------- CG solver (no preconditioner) via ViennaCL, hyb_matrix ----------" << std::endl;
  run_solver(vcl_hyb_matrix, vcl_vec2, vcl_result, cg_solver, viennacl::linalg::no_precond(), cg_ops);
  

  std::cout << "------- CG solver (ILU0 preconditioner) using ublas ----------" << std::endl;
  run_solver(ublas_matrix, ublas_vec2, ublas_result, cg_solver, ublas_ilu0, cg_ops);

  std::cout << "------- CG solver (ILU0 preconditioner) via ViennaCL, compressed_matrix ----------" << std::endl;
  run_solver(vcl_compressed_matrix, vcl_vec2, vcl_result, cg_solver, vcl_ilu0, cg_ops);

  
  std::cout << "------- CG solver (Block-ILU0 preconditioner) using ublas ----------" << std::endl;
  run_solver(ublas_matrix, ublas_vec2, ublas_result, cg_solver, ublas_block_ilu0, cg_ops);

#ifdef VIENNACL_WITH_OPENCL
  std::cout << "------- CG solver (Block-ILU0 preconditioner) via ViennaCL, compressed_matrix ----------" << std::endl;
  run_solver(vcl_compressed_matrix, vcl_vec2, vcl_result, cg_solver, vcl_block_ilu0, cg_ops);
#endif
  
  std::cout << "------- CG solver (ILUT preconditioner) using ublas ----------" << std::endl;
  run_solver(ublas_matrix, ublas_vec2, ublas_result, cg_solver, ublas_ilut, cg_ops);
  
  std::cout << "------- CG solver (ILUT preconditioner) via ViennaCL, compressed_matrix ----------" << std::endl;
  run_solver(vcl_compressed_matrix, vcl_vec2, vcl_result, cg_solver, vcl_ilut, cg_ops);

  std::cout << "------- CG solver (Block-ILUT preconditioner) using ublas ----------" << std::endl;
  run_solver(ublas_matrix, ublas_vec2, ublas_result, cg_solver, ublas_block_ilut, cg_ops);

#ifdef VIENNACL_WITH_OPENCL
  std::cout << "------- CG solver (Block-ILUT preconditioner) via ViennaCL, compressed_matrix ----------" << std::endl;
  run_solver(vcl_compressed_matrix, vcl_vec2, vcl_result, cg_solver, vcl_block_ilut, cg_ops);

  
//  std::cout << "------- CG solver (ILUT preconditioner) via ViennaCL, coordinate_matrix ----------" << std::endl;
//  run_solver(vcl_coordinate_matrix, vcl_vec2, vcl_result, cg_solver, vcl_ilut, cg_ops);
  
  
  std::cout << "------- CG solver (Jacobi preconditioner) using ublas ----------" << std::endl;
  run_solver(ublas_matrix, ublas_vec2, ublas_result, cg_solver, ublas_jacobi, cg_ops);
  
  std::cout << "------- CG solver (Jacobi preconditioner) via ViennaCL, compressed_matrix ----------" << std::endl;
  run_solver(vcl_compressed_matrix, vcl_vec2, vcl_result, cg_solver, vcl_jacobi, cg_ops);
  
//  std::cout << "------- CG solver (Jacobi preconditioner) via ViennaCL, coordinate_matrix ----------" << std::endl;
//  run_solver(vcl_coordinate_matrix, vcl_vec2, vcl_result, cg_solver, vcl_jacobi, cg_ops);
  
  
  std::cout << "------- CG solver (row scaling preconditioner) using ublas ----------" << std::endl;
  run_solver(ublas_matrix, ublas_vec2, ublas_result, cg_solver, ublas_row_scaling, cg_ops);
  
  std::cout << "------- CG solver (row scaling preconditioner) via ViennaCL, compressed_matrix ----------" << std::endl;
  run_solver(vcl_compressed_matrix, vcl_vec2, vcl_result, cg_solver, vcl_row_scaling, cg_ops);
#endif  
  
//  std::cout << "------- CG solver (row scaling preconditioner) via ViennaCL, coordinate_matrix ----------" << std::endl;
//  run_solver(vcl_coordinate_matrix, vcl_vec2, vcl_result, cg_solver, vcl_row_scaling, cg_ops);
  
  ///////////////////////////////////////////////////////////////////////////////
  //////////////////////           BiCGStab solver             //////////////////
  ///////////////////////////////////////////////////////////////////////////////
  
  long bicgstab_ops = static_cast<long>(solver_iters * (2 * ublas_matrix.nnz() + 13 * ublas_vec2.size()));
  
  viennacl::linalg::bicgstab_tag bicgstab_solver(solver_tolerance, solver_iters);
                                                                             
  std::cout << "------- BiCGStab solver (no preconditioner) using ublas ----------" << std::endl;
  run_solver(ublas_matrix, ublas_vec2, ublas_result, bicgstab_solver, viennacl::linalg::no_precond(), bicgstab_ops);
  
  std::cout << "------- BiCGStab solver (no preconditioner) via ViennaCL, compressed_matrix ----------" << std::endl;
  run_solver(vcl_compressed_matrix, vcl_vec2, vcl_result, bicgstab_solver, viennacl::linalg::no_precond(), bicgstab_ops);
  
//  std::cout << "------- BiCGStab solver (no preconditioner) on GPU, coordinate_matrix ----------" << std::endl;
//  run_solver(vcl_coordinate_matrix, vcl_vec2, vcl_result, bicgstab_solver, bicgstab_ops);

  
  std::cout << "------- BiCGStab solver (ILUT preconditioner) using ublas ----------" << std::endl;
  run_solver(ublas_matrix, ublas_vec2, ublas_result, bicgstab_solver, ublas_ilut, bicgstab_ops);
  
  std::cout << "------- BiCGStab solver (ILUT preconditioner) via ViennaCL, compressed_matrix ----------" << std::endl;
  run_solver(vcl_compressed_matrix, vcl_vec2, vcl_result, bicgstab_solver, vcl_ilut, bicgstab_ops);

  std::cout << "------- BiCGStab solver (Block-ILUT preconditioner) using ublas ----------" << std::endl;
  run_solver(ublas_matrix, ublas_vec2, ublas_result, bicgstab_solver, ublas_block_ilut, bicgstab_ops);

#ifdef VIENNACL_WITH_OPENCL
  std::cout << "------- BiCGStab solver (Block-ILUT preconditioner) via ViennaCL, compressed_matrix ----------" << std::endl;
  run_solver(vcl_compressed_matrix, vcl_vec2, vcl_result, bicgstab_solver, vcl_block_ilut, bicgstab_ops);
  
//  std::cout << "------- BiCGStab solver (ILUT preconditioner) via ViennaCL, coordinate_matrix ----------" << std::endl;
//  run_solver(vcl_coordinate_matrix, vcl_vec2, vcl_result, bicgstab_solver, vcl_ilut, bicgstab_ops);
  
  std::cout << "------- BiCGStab solver (Jacobi preconditioner) using ublas ----------" << std::endl;
  run_solver(ublas_matrix, ublas_vec2, ublas_result, bicgstab_solver, ublas_jacobi, bicgstab_ops);
  
  std::cout << "------- BiCGStab solver (Jacobi preconditioner) via ViennaCL, compressed_matrix ----------" << std::endl;
  run_solver(vcl_compressed_matrix, vcl_vec2, vcl_result, bicgstab_solver, vcl_jacobi, bicgstab_ops);
  
//  std::cout << "------- CG solver (Jacobi preconditioner) via ViennaCL, coordinate_matrix ----------" << std::endl;
//  run_solver(vcl_coordinate_matrix, vcl_vec2, vcl_result, bicgstab_solver, vcl_jacobi, bicgstab_ops);
  
  std::cout << "------- BiCGStab solver (row scaling preconditioner) using ublas ----------" << std::endl;
  run_solver(ublas_matrix, ublas_vec2, ublas_result, bicgstab_solver, ublas_row_scaling, bicgstab_ops);
  
  std::cout << "------- BiCGStab solver (row scaling preconditioner) via ViennaCL, compressed_matrix ----------" << std::endl;
  run_solver(vcl_compressed_matrix, vcl_vec2, vcl_result, bicgstab_solver, vcl_row_scaling, bicgstab_ops);
#endif
  
//  std::cout << "------- CG solver row scaling preconditioner) via ViennaCL, coordinate_matrix ----------" << std::endl;
//  run_solver(vcl_coordinate_matrix, vcl_vec2, vcl_result, bicgstab_solver, vcl_row_scaling, bicgstab_ops);

  ///////////////////////////////////////////////////////////////////////////////
  ///////////////////////            GMRES solver             ///////////////////
  ///////////////////////////////////////////////////////////////////////////////
  
  long gmres_ops = static_cast<long>(solver_iters * (ublas_matrix.nnz() + (solver_iters * 2 + 7) * ublas_vec2.size()));
  
  viennacl::linalg::gmres_tag gmres_solver(solver_tolerance, solver_iters, solver_krylov_dim);
  
  std::cout << "------- GMRES solver (no preconditioner) using ublas ----------" << std::endl;
  run_solver(ublas_matrix, ublas_vec2, ublas_result, gmres_solver, viennacl::linalg::no_precond(), gmres_ops);
  
  std::cout << "------- GMRES solver (no preconditioner) via ViennaCL, compressed_matrix ----------" << std::endl;
  run_solver(vcl_compressed_matrix, vcl_vec2, vcl_result, gmres_solver, viennacl::linalg::no_precond(), gmres_ops);
  
//  std::cout << "------- GMRES solver (no preconditioner) on GPU, coordinate_matrix ----------" << std::endl;
//  run_solver(vcl_coordinate_matrix, vcl_vec2, vcl_result, gmres_solver, bicgstab_ops);

  
  std::cout << "------- GMRES solver (ILUT preconditioner) using ublas ----------" << std::endl;
  run_solver(ublas_matrix, ublas_vec2, ublas_result, gmres_solver, ublas_ilut, gmres_ops);
  
  std::cout << "------- GMRES solver (ILUT preconditioner) via ViennaCL, compressed_matrix ----------" << std::endl;
  run_solver(vcl_compressed_matrix, vcl_vec2, vcl_result, gmres_solver, vcl_ilut, gmres_ops);
  
//  std::cout << "------- GMRES solver (ILUT preconditioner) via ViennaCL, coordinate_matrix ----------" << std::endl;
//  run_solver(vcl_coordinate_matrix, vcl_vec2, vcl_result, gmres_solver, vcl_ilut, gmres_ops);


#ifdef VIENNACL_WITH_OPENCL
  std::cout << "------- GMRES solver (Jacobi preconditioner) using ublas ----------" << std::endl;
  run_solver(ublas_matrix, ublas_vec2, ublas_result, gmres_solver, ublas_jacobi, gmres_ops);
  
  std::cout << "------- GMRES solver (Jacobi preconditioner) via ViennaCL, compressed_matrix ----------" << std::endl;
  run_solver(vcl_compressed_matrix, vcl_vec2, vcl_result, gmres_solver, vcl_jacobi, gmres_ops);
  
//  std::cout << "------- GMRES solver (Jacobi preconditioner) via ViennaCL, coordinate_matrix ----------" << std::endl;
//  run_solver(vcl_coordinate_matrix, vcl_vec2, vcl_result, gmres_solver, vcl_jacobi, gmres_ops);
  
  
  std::cout << "------- GMRES solver (row scaling preconditioner) using ublas ----------" << std::endl;
  run_solver(ublas_matrix, ublas_vec2, ublas_result, gmres_solver, ublas_row_scaling, gmres_ops);
  
  std::cout << "------- GMRES solver (row scaling preconditioner) via ViennaCL, compressed_matrix ----------" << std::endl;
  run_solver(vcl_compressed_matrix, vcl_vec2, vcl_result, gmres_solver, vcl_row_scaling, gmres_ops);
#endif
  
//  std::cout << "------- GMRES solver (row scaling preconditioner) via ViennaCL, coordinate_matrix ----------" << std::endl;
//  run_solver(vcl_coordinate_matrix, vcl_vec2, vcl_result, gmres_solver, vcl_row_scaling, gmres_ops);
  
  return 0;
}

int main()
{
  std::cout << std::endl;
  std::cout << "----------------------------------------------" << std::endl;
  std::cout << "               Device Info" << std::endl;
  std::cout << "----------------------------------------------" << std::endl;
  
#ifdef VIENNACL_WITH_OPENCL
  std::cout << viennacl::ocl::current_device().info() << std::endl;
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
  run_benchmark<float>();
#ifdef VIENNACL_WITH_OPENCL
  if( viennacl::ocl::current_device().double_support() )
#endif
  {
    std::cout << std::endl;
    std::cout << "   -------------------------------" << std::endl;
    std::cout << "   # benchmarking double-precision" << std::endl;
    std::cout << "   -------------------------------" << std::endl;
    run_benchmark<double>();
  }
  return 0;
}

