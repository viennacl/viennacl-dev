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


/*
*   Benchmark:  Direct solve matrix-matrix and matrix-vecotor
*
*/
#include <iostream>


#include "viennacl/scalar.hpp"
#include "viennacl/matrix.hpp"
#include "viennacl/matrix_proxy.hpp"
#include "viennacl/vector.hpp"
#include "viennacl/linalg/prod.hpp"
#include "viennacl/linalg/norm_2.hpp"
#include "viennacl/linalg/direct_solve.hpp"
#include "viennacl/tools/random.hpp"
#include "viennacl/tools/timer.hpp"

#define BENCHMARK_RUNS 10


inline void printOps(double num_ops, double exec_time)
{
  std::cout << "GFLOPs: " << num_ops / (1000000 * exec_time * 1000) << std::endl;
}


template<typename NumericT>
void fill_matrix(viennacl::matrix<NumericT> & mat)
{
  viennacl::tools::uniform_random_numbers<NumericT> randomNumber;

  for (std::size_t i = 0; i < mat.size1(); ++i)
  {
    for (std::size_t j = 0; j < mat.size2(); ++j)
      mat(i, j) = static_cast<NumericT>(-0.5) * randomNumber();
    mat(i, i) = NumericT(1.0) + NumericT(2.0) * randomNumber(); //some extra weight on diagonal for stability
  }
}

template<typename NumericT>
void fill_vector(viennacl::vector<NumericT> & vec)
{
  viennacl::tools::uniform_random_numbers<NumericT> randomNumber;

  for (std::size_t i = 0; i < vec.size(); ++i)
    vec(i) = NumericT(1.0) + NumericT(2.0) * randomNumber(); //some extra weight on diagonal for stability
}

template<typename NumericT,typename MatrixT1, typename MatrixT2,typename MatrixT3, typename SolverTag>
void run_solver_matrix(MatrixT1 const & matrix1, MatrixT2 const & matrix2,MatrixT3 & result, SolverTag)
{
  std::cout << "------- Solver tag: " <<SolverTag::name()<<" ----------" << std::endl;
  result = viennacl::linalg::solve(matrix1, matrix2, SolverTag());

  viennacl::tools::timer timer;
  viennacl::backend::finish();

  timer.start();
  for (int runs=0; runs<BENCHMARK_RUNS; ++runs)
    result = viennacl::linalg::solve(matrix1, matrix2, SolverTag());

  double exec_time = timer.get();
  viennacl::backend::finish();
  std::cout << "GPU: ";printOps(double(matrix1.size1() * matrix1.size1() * matrix2.size2()),(static_cast<double>(exec_time) / static_cast<double>(BENCHMARK_RUNS)));
  std::cout << "GPU: " << double(matrix1.size1() * matrix1.size1() * matrix2.size2() * sizeof(NumericT)) / (static_cast<double>(exec_time) / static_cast<double>(BENCHMARK_RUNS)) / 1e9 << " GB/sec" << std::endl;
  std::cout << "Execution time: " << exec_time/BENCHMARK_RUNS << std::endl;
  std::cout << "------- Finnished: " << SolverTag::name() << " ----------" << std::endl;
}

template<typename NumericT,typename VectorT, typename VectorT2,typename MatrixT, typename SolverTag>
void run_solver_vector(MatrixT const & matrix, VectorT2 const & vector2,VectorT & result, SolverTag)
{
  std::cout << "------- Solver tag: " <<SolverTag::name()<<" ----------" << std::endl;
  result = viennacl::linalg::solve(matrix, vector2, SolverTag());

  viennacl::tools::timer timer;
  viennacl::backend::finish();

  timer.start();
  for (int runs=0; runs<BENCHMARK_RUNS; ++runs)
  {
    result = viennacl::linalg::solve(matrix, vector2, SolverTag());
  }
  double exec_time = timer.get();
  viennacl::backend::finish();
  std::cout << "GPU: ";printOps(double(matrix.size1() * matrix.size1()),(static_cast<double>(exec_time) / static_cast<double>(BENCHMARK_RUNS)));
  std::cout << "GPU: "<< double(matrix.size1() * matrix.size1() * sizeof(NumericT)) / (static_cast<double>(exec_time) / static_cast<double>(BENCHMARK_RUNS)) / 1e9 << " GB/sec" << std::endl;
  std::cout << "Execution time: " << exec_time/BENCHMARK_RUNS << std::endl;
  std::cout << "------- Finished: " << SolverTag::name() << " ----------" << std::endl;
}

template<typename NumericT,typename F_A, typename F_B>
void run_benchmark()
{
  std::size_t matrix_size = 1500;  //some odd number, not too large
  std::size_t rhs_num = 153;

  viennacl::matrix<NumericT, F_A> vcl_A(matrix_size, matrix_size);
  viennacl::matrix<NumericT, F_B> vcl_B(matrix_size, rhs_num);
  viennacl::matrix<NumericT, F_B> result(matrix_size, rhs_num);

  viennacl::vector<NumericT> vcl_vec_B(matrix_size);
  viennacl::vector<NumericT> vcl_vec_result(matrix_size);

  fill_matrix(vcl_A);
  fill_matrix(vcl_B);

  fill_vector(vcl_vec_B);
  std::cout << "------- Solve Matrix-Matrix: ----------\n" << std::endl;
  run_solver_matrix<NumericT>(vcl_A,vcl_B,result,viennacl::linalg::lower_tag());
  run_solver_matrix<NumericT>(vcl_A,vcl_B,result,viennacl::linalg::unit_lower_tag());
  run_solver_matrix<NumericT>(vcl_A,vcl_B,result,viennacl::linalg::upper_tag());
  run_solver_matrix<NumericT>(vcl_A,vcl_B,result,viennacl::linalg::unit_upper_tag());
  std::cout << "------- End Matrix-Matrix: ----------\n" << std::endl;

  std::cout << "------- Solve Matrix-Vector: ----------\n" << std::endl;
  run_solver_vector<NumericT>(vcl_A,vcl_vec_B,vcl_vec_result,viennacl::linalg::lower_tag());
  run_solver_vector<NumericT>(vcl_A,vcl_vec_B,vcl_vec_result,viennacl::linalg::unit_lower_tag());
  run_solver_vector<NumericT>(vcl_A,vcl_vec_B,vcl_vec_result,viennacl::linalg::upper_tag());
  run_solver_vector<NumericT>(vcl_A,vcl_vec_B,vcl_vec_result,viennacl::linalg::unit_upper_tag());
  std::cout << "------- End Matrix-Vector: ----------\n" << std::endl;
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
  std::cout << std::endl;
  std::cout << "----------------------------------------------" << std::endl;
  std::cout << "----------------------------------------------" << std::endl;
  std::cout << "## Benchmark :: Direct solve" << std::endl;
  std::cout << "----------------------------------------------" << std::endl;
  std::cout << std::endl;
  std::cout << "   -------------------------------" << std::endl;
  std::cout << "   # benchmarking single-precision" << std::endl;
  std::cout << "   -------------------------------" << std::endl;
  run_benchmark<float,viennacl::row_major,viennacl::row_major>();
#ifdef VIENNACL_WITH_OPENCL
  if ( viennacl::ocl::current_device().double_support() )
#endif
  {
    std::cout << std::endl;
    std::cout << "   -------------------------------" << std::endl;
    std::cout << "   # benchmarking double-precision" << std::endl;
    std::cout << "   -------------------------------" << std::endl;
    run_benchmark<double,viennacl::row_major,viennacl::row_major>();
  }
  return 0;
}
