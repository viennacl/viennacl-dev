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

/** \example iterative-custom.cpp
*
*   This tutorial explains the use of iterative solvers in ViennaCL with custom monitors and initial guesses.
*
*   \note iterative-custom.cpp and iterative-custom.cu are identical, the latter being required for compilation using CUDA nvcc
*
*   We start with including the necessary system headers:
**/

//
// include necessary system headers
//
#include <iostream>

//
// ViennaCL includes
//
#include "viennacl/scalar.hpp"
#include "viennacl/vector.hpp"
#include "viennacl/compressed_matrix.hpp"
#include "viennacl/linalg/prod.hpp"
#include "viennacl/linalg/jacobi_precond.hpp"
#include "viennacl/linalg/cg.hpp"
#include "viennacl/linalg/bicgstab.hpp"
#include "viennacl/linalg/gmres.hpp"
#include "viennacl/io/matrix_market.hpp"

/**
 *  <h1>Defining Custom Monitors Functions for Iterative Solvers</h1>
 *  Custom monitors for the iterative solvers require two ingredients:
 *  First, a structure holding all the auxiliary data we want to reuse in the monitor.
 *  Second, a callback function called by the solver in each iteration.
 *
 *  In this example we define a callback-routine for printing the current estimate for the residual and compare it with the true residual.
 *  To do so, we need to pass the system matrix, the right hand side, and the initial guess to the monitor routine, which we achieve by packing pointers to these objects into a struct:
 **/
template<typename MatrixT, typename VectorT>
struct monitor_user_data
{
  monitor_user_data(MatrixT const & A, VectorT const & b, VectorT const & guess) : A_ptr(&A), b_ptr(&b), guess_ptr(&guess) {}

  MatrixT const *A_ptr;
  VectorT const *b_ptr;
  VectorT const *guess_ptr;
};

/**
 *  The actual callback-routine takes the current approximation to the result as the first parameter, and the current estimate for the relative residual norm as second argument.
 *  The third argument is a pointer to our user-data, which in a first step cast to the correct type.
 *  If the monitor returns true, the iterative solver stops. This is handy for defining custom termination criteria, e.g. one-norms for the result change.
 *  Since we do not want to terminate the iterative solver with a custom criterion here, we always return 'false' at the end of the function.
 *
 *  Note to type-safety evangelists: This void*-interface is designed to be later exposed through a shared library ('libviennacl').
 *  Thus, user types may not be known at the point of compilation, requiring a void*-approach.
 **/
template<typename VectorT, typename NumericT, typename MatrixT>
bool my_custom_monitor(VectorT const & current_approx, NumericT residual_estimate, void *user_data)
{
  // Extract residual:
  monitor_user_data<MatrixT, VectorT> const *data = reinterpret_cast<monitor_user_data<MatrixT, VectorT> const*>(user_data);

  // Form residual r = b - A*x, taking an initial guess into account: r = b - A * (current_approx + x_initial)
  VectorT x = current_approx + *data->guess_ptr;
  VectorT residual = *data->b_ptr - viennacl::linalg::prod(*data->A_ptr, x);
  VectorT initial_residual = *data->b_ptr - viennacl::linalg::prod(*data->A_ptr, *data->guess_ptr);

  std::cout << "Residual estimate vs. true residual: " << residual_estimate << " vs. " << viennacl::linalg::norm_2(residual) / viennacl::linalg::norm_2(initial_residual) << std::endl;

  return false; // no termination of iteration
}


/**
*  <h1>The main Program</h1>
*  In the main() routine we create matrices and vectors and fill them with data.
**/
int main()
{
  typedef float       ScalarType;


  /**
  * Read system matrix from a matrix-market file
  **/
  std::vector<std::map<unsigned int, ScalarType> > stl_A;
  if (!viennacl::io::read_matrix_market_file(stl_A, "../examples/testdata/mat65k.mtx"))
  {
    std::cout << "Error reading Matrix file" << std::endl;
    return EXIT_FAILURE;
  }
  viennacl::compressed_matrix<ScalarType> A;
  viennacl::copy(stl_A, A);

  /**
  * Set up right hand side and reference solution consisting of all ones:
  **/
  viennacl::vector<ScalarType> ref_result = viennacl::scalar_vector<ScalarType>(A.size2(), ScalarType(1.0));
  viennacl::vector<ScalarType> result(A.size2());

  viennacl::vector<ScalarType> b = viennacl::linalg::prod(A, ref_result);

  /**
  * As initial guess we take a vector consisting of all 0.9s, except for the first entry, which we set to zero:
  **/
  viennacl::vector<ScalarType> init_guess = viennacl::scalar_vector<ScalarType>(ref_result.size(), ScalarType(0.9));
  init_guess[0] = 0;

  /**
   * Set up the monitor data, holding the system matrix, the right hand side, and the initial guess:
   **/
  monitor_user_data<viennacl::compressed_matrix<ScalarType>, viennacl::vector<ScalarType> > my_monitor_data(A, b, init_guess);


  /**
  * set up Jacobi preconditioners (just for demonstration purposes, can be any other preconditioner here):
  **/
  viennacl::linalg::jacobi_precond< viennacl::compressed_matrix<ScalarType> > jacobi(A, viennacl::linalg::jacobi_tag());


  /**
  * <h2>Conjugate Gradient Solver</h2>
  **/
  std::cout << "----- CG Method -----" << std::endl;

  /**
  * Run the CG method with a relative tolerance of 1e-5 and a maximum of 20 iterations.
  * We instantiate the CG solver object, register the monitor callback (with data), set the initial guess, and launch the solver.
  **/
  viennacl::linalg::cg_tag my_cg_tag(1e-5, 20);
  viennacl::linalg::cg_solver<viennacl::vector<ScalarType> > my_cg_solver(my_cg_tag);

  my_cg_solver.set_monitor(my_custom_monitor<viennacl::vector<ScalarType>, ScalarType, viennacl::compressed_matrix<ScalarType> >, &my_monitor_data);
  my_cg_solver.set_initial_guess(init_guess);

  my_cg_solver(A, b); // without preconditioner


  /**
  * <h2>Stabilized BiConjugate Gradient Solver</h2>
  **/
  std::cout << "----- BiCGStab Method -----" << std::endl;

  /**
  * Run the Jacobi-preconditioned BiCGStab method with a relative tolerance of 1e-5 and a maximum of 20 iterations.
  * We instantiate the BiCGStab solver object, register the monitor callback (with data), set the initial guess, and launch the solver.
  **/
  viennacl::linalg::bicgstab_tag my_bicgstab_tag(1e-5, 20);
  viennacl::linalg::bicgstab_solver<viennacl::vector<ScalarType> > my_bicgstab_solver(my_bicgstab_tag);

  my_bicgstab_solver.set_monitor(my_custom_monitor<viennacl::vector<ScalarType>, ScalarType, viennacl::compressed_matrix<ScalarType> >, &my_monitor_data);
  my_bicgstab_solver.set_initial_guess(init_guess);

  my_bicgstab_solver(A, b, jacobi); // with Jacobi preconditioner


  /**
  * <h2>GMRES Solver</h2>
  **/
  std::cout << "----- GMRES Method -----" << std::endl;

  /**
  * Run the unpreconditioned GMRES method with a relative tolerance of 1e-5 and a maximum of 30 iterations for a Krylov size of 10 (i.e. restart every 10 iterations).
  * We instantiate the GMRES solver object, register the monitor callback (with data), set the initial guess, and launch the solver.
  *
  * Note that the monitor in the GMRES method is only called after each restart, but not in every (inner) iteration.
  **/
  viennacl::linalg::gmres_tag my_gmres_tag(1e-5, 30, 10);
  viennacl::linalg::gmres_solver<viennacl::vector<ScalarType> > my_gmres_solver(my_gmres_tag);

  my_gmres_solver.set_monitor(my_custom_monitor<viennacl::vector<ScalarType>, ScalarType, viennacl::compressed_matrix<ScalarType> >, &my_monitor_data);
  my_gmres_solver.set_initial_guess(init_guess);

  my_gmres_solver(A, b);

  /**
  *  That's it, the tutorial is completed.
  **/
  std::cout << "!!!! TUTORIAL COMPLETED SUCCESSFULLY !!!!" << std::endl;

  return EXIT_SUCCESS;
}

