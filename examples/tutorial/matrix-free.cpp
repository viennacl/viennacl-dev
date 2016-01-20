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

/** \example matrix-free.cpp
*
*   This tutorial explains how to use the iterative solvers in ViennaCL in a matrix-free manner, i.e. without explicitly assembling a matrix.
*
*   We consider the solution of the Poisson equation \f$ \Delta \varphi = -1 \f$ on the unit square \f$ [0,1] \times [0,1] \f$ with homogeneous Dirichlet boundary conditions using a finite-difference discretization.
*   A \f$ N \times N \f$ grid is used, where the first and the last points per dimensions represent the boundary.
*   For simplicity we only consider the host-backend here. Have a look at custom-kernels.hpp and custom-cuda.cu on how to use custom kernels in such a matrix-free setting.
*
*   \note matrix-free.cpp and matrix-free.cu are identical, the latter being required for compilation using CUDA nvcc
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
#include "viennacl/linalg/prod.hpp"
#include "viennacl/linalg/cg.hpp"
#include "viennacl/linalg/bicgstab.hpp"
#include "viennacl/linalg/gmres.hpp"

/**
  * ViennaCL imposes two type requirements on a user-provided operator to compute `y = prod(A, x)` for the iterative solvers:
  *   - A member function `apply()`, taking two ViennaCL base vectors `x` and `y` as arguments. This member function carries out the action of the matrix to the vector.
  *   - A member function `size1()` returning the length of the result vectors.
  * Keep in mind that you can always wrap your existing classes accordingly to fit ViennaCL's interface requirements.
  *
  * We define a simple class for dealing with the \f$ N \times N \f$ grid for solving Poisson's equation.
  * It only holds the number of grid points per coordinate direction and implements the `apply()` and `size1()` routines.
  * Depending on whether the host, OpenCL, or CUDA is used for the computation, the respective implementation is called.
  * We skip the details for now and discuss (and implement) them at the end of this tutorial.
  **/
template<typename NumericT>
class MyOperator
{
public:
  MyOperator(std::size_t N) : N_(N) {}

  // Dispatcher for y = Ax
  void apply(viennacl::vector_base<NumericT> const & x, viennacl::vector_base<NumericT> & y) const
  {
#if defined(VIENNACL_WITH_CUDA)
    if (viennacl::traits::active_handle_id(x) == viennacl::CUDA_MEMORY)
      apply_cuda(x, y);
#endif

#if defined(VIENNACL_WITH_OPENCL)
    if (viennacl::traits::active_handle_id(x) == viennacl::OPENCL_MEMORY)
      apply_opencl(x, y);
#endif

    if (viennacl::traits::active_handle_id(x) == viennacl::MAIN_MEMORY)
      apply_host(x, y);
  }

  std::size_t size1() const { return N_ * N_; }

private:

#if defined(VIENNACL_WITH_CUDA)
  void apply_cuda(viennacl::vector_base<NumericT> const & x, viennacl::vector_base<NumericT> & y) const;
#endif

#if defined(VIENNACL_WITH_OPENCL)
  void apply_opencl(viennacl::vector_base<NumericT> const & x, viennacl::vector_base<NumericT> & y) const;
#endif

  void apply_host(viennacl::vector_base<NumericT> const & x, viennacl::vector_base<NumericT> & y) const;

  std::size_t N_;
};


/**
* <h2>Main Program</h2>
*
*  In the `main()` routine we create the right hand side vector, instantiate the operator, and then call the solver.
**/
int main()
{
  typedef float       ScalarType;  // feel free to change to double (and change OpenCL kernel argument types accordingly)

  std::size_t N = 10;
  viennacl::vector<ScalarType> rhs = viennacl::scalar_vector<ScalarType>(N*N, ScalarType(-1));
  MyOperator<ScalarType> op(N);

  /**
  * Run the CG method with our on-the-fly operator.
  * Use `viennacl::linalg::bicgstab_tag()` or `viennacl::linalg::gmres_tag()` instead of `viennacl::linalg::cg_tag()` to solve using BiCGStab or GMRES, respectively.
  **/
  viennacl::vector<ScalarType> result = viennacl::linalg::solve(op, rhs, viennacl::linalg::cg_tag());

  /**
   * Pretty-Print solution vector to verify solution.
   * (We use a slow direct element-access via `operator[]` here for convenience.)
   **/
  std::cout.precision(3);
  std::cout << std::fixed;
  std::cout << "Result value map: " << std::endl;
  std::cout << std::endl << "^ y " << std::endl;
  for (std::size_t i=0; i<N; ++i)
  {
    std::cout << "|  ";
    for (std::size_t j=0; j<N; ++j)
      std::cout << result[i * N + j] << "  ";
    std::cout << std::endl;
  }
  std::cout << "*---------------------------------------------> x" << std::endl;

  /**
  *  That's it, print a completion message. Read on for details on how to implement the actual compute kernels.
  **/
  std::cout << "!!!! TUTORIAL COMPLETED SUCCESSFULLY !!!!" << std::endl;

  return EXIT_SUCCESS;
}




/**
  * <h2> Implementation Details </h2>
  *
  *  So far we have only looked at the code for the control logic.
  *  In the following we define the actual 'worker' code for the matrix-free implementations.
  *
  *  <h3> Execution on Host </h3>
  *
  *  Since the execution on the host has the smallest amount of boilerplate code surrounding it, we use this case as a starting point.
  **/
template<typename NumericT>
void MyOperator<NumericT>::apply_host(viennacl::vector_base<NumericT> const & x, viennacl::vector_base<NumericT> & y) const
{
  NumericT const * values_x = viennacl::linalg::host_based::detail::extract_raw_pointer<NumericT>(x.handle());
  NumericT       * values_y = viennacl::linalg::host_based::detail::extract_raw_pointer<NumericT>(y.handle());

  NumericT dx = NumericT(1) / NumericT(N_ + 1);
  NumericT dy = NumericT(1) / NumericT(N_ + 1);

/**
  *  In the following we iterate over all \f$ N \times N \f$ points and apply the five-point stencil directly.
  *  This is done in a straightforward manner for illustration purposes.
  *  Multi-threaded execution via OpenMP can be obtained by uncommenting the pragma below.
  *
  *  Feel free to apply additional optimizations with respect to data access patterns and the like.
  **/

  // feel free to use
  //  #pragma omp parallel for
  // here
  for (std::size_t i=0; i<N_; ++i)
    for (std::size_t j=0; j<N_; ++j)
    {
      NumericT value_right  = (j < N_ - 1) ? values_x[ i   *N_ + j + 1] : 0;
      NumericT value_left   = (j > 0     ) ? values_x[ i   *N_ + j - 1] : 0;
      NumericT value_top    = (i < N_ - 1) ? values_x[(i+1)*N_ + j    ] : 0;
      NumericT value_bottom = (i > 0     ) ? values_x[(i-1)*N_ + j    ] : 0;
      NumericT value_center = values_x[i*N_ + j];

      values_y[i*N_ + j] =   ((value_right - value_center) / dx - (value_center - value_left)   / dx) / dx
                           + ((value_top   - value_center) / dy - (value_center - value_bottom) / dy) / dy;
    }
}


/**
  * <h3> Execution via CUDA </h3>
  *
  *  The host-based kernel code serves as a basis for the CUDA kernel.
  *  The only thing we have to adjust are the array bounds:
  *  We assign one CUDA threadblock per index `i`.
  *  For a fixed `i`, parallelization across all threads in the block is obtained with respect to `j`.
  *
  *  Again, feel free to apply additional optimizations with respect to data access patterns and the like...
  **/

#if defined(VIENNACL_WITH_CUDA)
template<typename NumericT>
__global__ void apply_cuda_kernel(NumericT const * values_x,
                                  NumericT       * values_y,
                                  std::size_t N)
{
  NumericT dx = NumericT(1) / (N + 1);
  NumericT dy = NumericT(1) / (N + 1);

  for (std::size_t i = blockIdx.x; i < N; i += gridDim.x)
    for (std::size_t j = threadIdx.x; j < N; j += blockDim.x)
    {
      NumericT value_right  = (j < N - 1) ? values_x[ i   *N + j + 1] : 0;
      NumericT value_left   = (j > 0    ) ? values_x[ i   *N + j - 1] : 0;
      NumericT value_top    = (i < N - 1) ? values_x[(i+1)*N + j    ] : 0;
      NumericT value_bottom = (i > 0    ) ? values_x[(i-1)*N + j    ] : 0;
      NumericT value_center = values_x[i*N + j];

      values_y[i*N + j] =   ((value_right - value_center) / dx - (value_center - value_left)   / dx) / dx
                          + ((value_top   - value_center) / dy - (value_center - value_bottom) / dy) / dy;
    }
}
#endif


#if defined(VIENNACL_WITH_CUDA)
template<typename NumericT>
void MyOperator<NumericT>::apply_cuda(viennacl::vector_base<NumericT> const & x, viennacl::vector_base<NumericT> & y) const
{
  apply_cuda_kernel<<<128, 128>>>(viennacl::cuda_arg(x), viennacl::cuda_arg(y), N_);
}
#endif




/**
  *  <h3> Execution via OpenCL </h3>
  *
  *  The OpenCL kernel is almost identical to the CUDA kernel: Only a couple of keywords need to be replaced.
  *  Also, the sources need to be packed into a string:
  **/

#if defined(VIENNACL_WITH_OPENCL)
static const char * my_compute_program =
"typedef float NumericT; \n"
"__kernel void apply_opencl_kernel(__global NumericT const * values_x, \n"
"                                  __global NumericT       * values_y, \n"
"                                  unsigned int N) {\n"

"      NumericT dx = (NumericT)1 / (N + 1); \n"
"      NumericT dy = (NumericT)1 / (N + 1); \n"

"      for (unsigned int i = get_group_id(0); i < N; i += get_num_groups(0)) \n"
"        for (unsigned int j = get_local_id(0); j < N; j += get_local_size(0)) { \n"

"          NumericT value_right  = (j < N - 1) ? values_x[ i   *N + j + 1] : 0; \n"
"          NumericT value_left   = (j > 0    ) ? values_x[ i   *N + j - 1] : 0; \n"
"          NumericT value_top    = (i < N - 1) ? values_x[(i+1)*N + j    ] : 0; \n"
"          NumericT value_bottom = (i > 0    ) ? values_x[(i-1)*N + j    ] : 0; \n"
"          NumericT value_center = values_x[i*N + j]; \n"

"          values_y[i*N + j] =   ((value_right - value_center) / dx - (value_center - value_left)   / dx) / dx  \n"
"                              + ((value_top   - value_center) / dy - (value_center - value_bottom) / dy) / dy; \n"
"        }  \n"
"    } \n";
#endif

  /**
    *  Before the kernel is called for the first time, the OpenCL program containing the kernel needs to be compiled.
    *  We use a simple singleton using a static variable to achieve that.
    *
    *  Except for the kernel compilation at the first invocation, the OpenCL kernel launch is just one line of code just like for CUDA.
    *  Refer to custom-kernels.cpp for some more details.
    **/
#if defined(VIENNACL_WITH_OPENCL)
template<typename NumericT>
void MyOperator<NumericT>::apply_opencl(viennacl::vector_base<NumericT> const & x, viennacl::vector_base<NumericT> & y) const
  {
    viennacl::ocl::context & ctx = const_cast<viennacl::ocl::context &>(viennacl::traits::opencl_handle(x).context());
    static bool first_run = true;
    if (first_run) {
      ctx.add_program(my_compute_program, "my_compute_program");
      first_run = false;
    }
    viennacl::ocl::kernel & my_kernel = ctx.get_kernel("my_compute_program", "apply_opencl_kernel");

    viennacl::ocl::enqueue(my_kernel(x, y, static_cast<cl_uint>(N_)));
  }
#endif


