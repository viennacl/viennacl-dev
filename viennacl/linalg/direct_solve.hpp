#ifndef VIENNACL_LINALG_DIRECT_SOLVE_HPP_
#define VIENNACL_LINALG_DIRECT_SOLVE_HPP_

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

/** @file viennacl/linalg/direct_solve.hpp
    @brief Implementations of dense direct solvers are found here.
*/

#include "viennacl/forwards.h"
#include "viennacl/meta/enable_if.hpp"
#include "viennacl/vector.hpp"
#include "viennacl/vector_proxy.hpp"
#include "viennacl/matrix.hpp"
#include "viennacl/matrix_proxy.hpp"
#include "viennacl/linalg/prod.hpp"
#include "viennacl/linalg/host_based/direct_solve.hpp"

#ifdef VIENNACL_WITH_OPENCL
  #include "viennacl/linalg/opencl/direct_solve.hpp"
#endif

#ifdef VIENNACL_WITH_CUDA
  #include "viennacl/linalg/cuda/direct_solve.hpp"
#endif

#define VIENNACL_DIRECT_SOLVE_BLOCKSIZE 64

namespace viennacl
{
namespace linalg
{

//
// A \ B:
//

/** @brief Direct inplace solver for dense triangular systems. Matlab notation: A \ B
*
* @param A    The system matrix
* @param B    The matrix of row vectors, where the solution is directly written to
*/
template<typename NumericT, typename SolverTagT>
void inplace_solve_kernel(const matrix_base<NumericT>  & A, const matrix_base<NumericT> & B, SolverTagT)
{
  assert( (viennacl::traits::size1(A) == viennacl::traits::size2(A)) && bool("Size check failed in inplace_solve(): size1(A) != size2(A)"));
  assert( (viennacl::traits::size1(A) == viennacl::traits::size1(B)) && bool("Size check failed in inplace_solve(): size1(A) != size1(B)"));
  switch (viennacl::traits::handle(A).get_active_handle_id())
  {
    case viennacl::MAIN_MEMORY:
      viennacl::linalg::host_based::inplace_solve(A, false, const_cast<matrix_base<NumericT> &>(B), false, SolverTagT());
      break;
#ifdef VIENNACL_WITH_OPENCL
    case viennacl::OPENCL_MEMORY:
      viennacl::linalg::opencl::inplace_solve(A, false, const_cast<matrix_base<NumericT> &>(B), false, SolverTagT());
      break;
#endif
#ifdef VIENNACL_WITH_CUDA
    case viennacl::CUDA_MEMORY:
      viennacl::linalg::cuda::inplace_solve(A, false, const_cast<matrix_base<NumericT> &>(B), false, SolverTagT());
      break;
#endif
    case viennacl::MEMORY_NOT_INITIALIZED:
      throw memory_exception("not initialised!");
    default:
      throw memory_exception("not implemented");
  }
}

/** @brief Direct inplace solver for dense triangular systems with transposed right hand side
*
* @param A       The system matrix
* @param proxy_B The transposed matrix of row vectors, where the solution is directly written to
*/
template<typename NumericT, typename SolverTagT>
void inplace_solve_kernel(matrix_base<NumericT> const & A,
                          matrix_expression<const matrix_base<NumericT>, const matrix_base<NumericT>, op_trans>  & proxy_B,
                          SolverTagT)
{
  assert( (viennacl::traits::size1(A) == viennacl::traits::size2(A))       && bool("Size check failed in inplace_solve(): size1(A) != size2(A)"));
  assert( (viennacl::traits::size1(A) == viennacl::traits::size1(proxy_B)) && bool("Size check failed in inplace_solve(): size1(A) != size1(B^T)"));

  switch (viennacl::traits::handle(A).get_active_handle_id())
  {
    case viennacl::MAIN_MEMORY:
      viennacl::linalg::host_based::inplace_solve(A, false, const_cast<matrix_base<NumericT> &>(proxy_B.lhs()), true, SolverTagT());
      break;
#ifdef VIENNACL_WITH_OPENCL
    case viennacl::OPENCL_MEMORY:
      viennacl::linalg::opencl::inplace_solve(A, false, const_cast<matrix_base<NumericT> &>(proxy_B.lhs()), true, SolverTagT());
      break;
#endif
#ifdef VIENNACL_WITH_CUDA
    case viennacl::CUDA_MEMORY:
      viennacl::linalg::cuda::inplace_solve(A, false, const_cast<matrix_base<NumericT> &>(proxy_B.lhs()), true, SolverTagT());
      break;
#endif
    case viennacl::MEMORY_NOT_INITIALIZED:
      throw memory_exception("not initialised!");
    default:
      throw memory_exception("not implemented");
  }
}

//upper triangular solver for transposed lower triangular matrices
/** @brief Direct inplace solver for dense triangular systems that stem from transposed triangular systems
*
* @param proxy_A  The system matrix proxy
* @param B        The matrix holding the load vectors, where the solution is directly written to
*/
template<typename NumericT, typename SolverTagT>
void inplace_solve_kernel(matrix_expression<const matrix_base<NumericT>, const matrix_base<NumericT>, op_trans> const & proxy_A,
                          matrix_base<NumericT> & B,
                          SolverTagT)
{
  assert( (viennacl::traits::size1(proxy_A) == viennacl::traits::size2(proxy_A)) && bool("Size check failed in inplace_solve(): size1(A) != size2(A)"));
  assert( (viennacl::traits::size1(proxy_A) == viennacl::traits::size1(B))       && bool("Size check failed in inplace_solve(): size1(A^T) != size1(B)"));

  switch (viennacl::traits::handle(proxy_A.lhs()).get_active_handle_id())
  {
    case viennacl::MAIN_MEMORY:
      viennacl::linalg::host_based::inplace_solve(const_cast<matrix_base<NumericT> &>(proxy_A.lhs()), true, B, false, SolverTagT());
      break;
#ifdef VIENNACL_WITH_OPENCL
    case viennacl::OPENCL_MEMORY:
      viennacl::linalg::opencl::inplace_solve(const_cast<matrix_base<NumericT> &>(proxy_A.lhs()), true, B, false, SolverTagT());
      break;
#endif
#ifdef VIENNACL_WITH_CUDA
    case viennacl::CUDA_MEMORY:
      viennacl::linalg::cuda::inplace_solve(const_cast<matrix_base<NumericT> &>(proxy_A.lhs()), true, B, false, SolverTagT());
      break;
#endif
    case viennacl::MEMORY_NOT_INITIALIZED:
      throw memory_exception("not initialised!");
    default:
      throw memory_exception("not implemented");
  }
}

/** @brief Direct inplace solver for dense transposed triangular systems with transposed right hand side. Matlab notation: A' \ B'
*
* @param proxy_A  The system matrix proxy
* @param proxy_B  The matrix holding the load vectors, where the solution is directly written to
*/
template<typename NumericT, typename SolverTagT>
void inplace_solve_kernel(matrix_expression< const matrix_base<NumericT>, const matrix_base<NumericT>, op_trans> const & proxy_A,
                          matrix_expression< const matrix_base<NumericT>, const matrix_base<NumericT>, op_trans>       & proxy_B,
                          SolverTagT)
{
  assert( (viennacl::traits::size1(proxy_A) == viennacl::traits::size2(proxy_A)) && bool("Size check failed in inplace_solve(): size1(A) != size2(A)"));
  assert( (viennacl::traits::size1(proxy_A) == viennacl::traits::size1(proxy_B)) && bool("Size check failed in inplace_solve(): size1(A^T) != size1(B^T)"));

  switch (viennacl::traits::handle(proxy_A.lhs()).get_active_handle_id())
  {
    case viennacl::MAIN_MEMORY:
      viennacl::linalg::host_based::inplace_solve(const_cast<matrix_base<NumericT> &>(proxy_A.lhs()), true,
                                                  const_cast<matrix_base<NumericT> &>(proxy_B.lhs()), true, SolverTagT());

      break;
#ifdef VIENNACL_WITH_OPENCL
    case viennacl::OPENCL_MEMORY:
      viennacl::linalg::opencl::inplace_solve(const_cast<matrix_base<NumericT> &>(proxy_A.lhs()), true,
                                              const_cast<matrix_base<NumericT> &>(proxy_B.lhs()), true, SolverTagT());
      break;
#endif
#ifdef VIENNACL_WITH_CUDA
    case viennacl::CUDA_MEMORY:
      viennacl::linalg::cuda::inplace_solve(const_cast<matrix_base<NumericT> &>(proxy_A.lhs()), true,
                                            const_cast<matrix_base<NumericT> &>(proxy_B.lhs()), true, SolverTagT());
      break;
#endif
    case viennacl::MEMORY_NOT_INITIALIZED:
      throw memory_exception("not initialised!");
    default:
      throw memory_exception("not implemented");
  }
}

//
// A \ b
//

template<typename NumericT, typename SolverTagT>
void inplace_solve_vec_kernel(const matrix_base<NumericT> & mat,
                              const vector_base<NumericT> & vec,
                              SolverTagT)
{
  assert( (mat.size1() == vec.size()) && bool("Size check failed in inplace_solve(): size1(A) != size(b)"));
  assert( (mat.size2() == vec.size()) && bool("Size check failed in inplace_solve(): size2(A) != size(b)"));

  switch (viennacl::traits::handle(mat).get_active_handle_id())
  {
    case viennacl::MAIN_MEMORY:
      viennacl::linalg::host_based::inplace_solve(mat, false, const_cast<vector_base<NumericT> &>(vec), SolverTagT());
      break;
#ifdef VIENNACL_WITH_OPENCL
    case viennacl::OPENCL_MEMORY:
      viennacl::linalg::opencl::inplace_solve(mat, false, const_cast<vector_base<NumericT> &>(vec), SolverTagT());
      break;
#endif
#ifdef VIENNACL_WITH_CUDA
    case viennacl::CUDA_MEMORY:
      viennacl::linalg::cuda::inplace_solve(mat, false, const_cast<vector_base<NumericT> &>(vec), SolverTagT());
      break;
#endif
    case viennacl::MEMORY_NOT_INITIALIZED:
      throw memory_exception("not initialised!");
    default:
      throw memory_exception("not implemented");
  }
}

/** @brief Direct inplace solver for dense upper triangular systems that stem from transposed lower triangular systems
*
* @param proxy    The system matrix proxy
* @param vec    The load vector, where the solution is directly written to
*/
template<typename NumericT, typename SolverTagT>
void inplace_solve_vec_kernel(matrix_expression<const matrix_base<NumericT>, const matrix_base<NumericT>, op_trans> const & proxy,
                              const vector_base<NumericT> & vec,
                              SolverTagT)
{
  assert( (proxy.lhs().size1() == vec.size()) && bool("Size check failed in inplace_solve(): size1(A) != size(b)"));
  assert( (proxy.lhs().size2() == vec.size()) && bool("Size check failed in inplace_solve(): size2(A) != size(b)"));

  switch (viennacl::traits::handle(proxy.lhs()).get_active_handle_id())
  {
    case viennacl::MAIN_MEMORY:
      viennacl::linalg::host_based::inplace_solve(proxy.lhs(), true, const_cast<vector_base<NumericT> &>(vec), SolverTagT());
      break;
#ifdef VIENNACL_WITH_OPENCL
    case viennacl::OPENCL_MEMORY:
      viennacl::linalg::opencl::inplace_solve(proxy.lhs(), true, const_cast<vector_base<NumericT> &>(vec), SolverTagT());
      break;
#endif
#ifdef VIENNACL_WITH_CUDA
    case viennacl::CUDA_MEMORY:
      viennacl::linalg::cuda::inplace_solve(proxy.lhs(), true, const_cast<vector_base<NumericT> &>(vec), SolverTagT());
      break;
#endif
    case viennacl::MEMORY_NOT_INITIALIZED:
      throw memory_exception("not initialised!");
    default:
      throw memory_exception("not implemented");
  }
}


template<typename MatrixT1, typename MatrixT2, typename SolverTagT>
void inplace_solve_lower_impl(MatrixT1 const & A, MatrixT2 & B, SolverTagT)
{
  vcl_size_t blockSize = VIENNACL_DIRECT_SOLVE_BLOCKSIZE;
  if (A.size1() <= blockSize)
    inplace_solve_kernel(A, B, SolverTagT());
  else
  {
    for (vcl_size_t i = 0; i < A.size1(); i = i + blockSize)
    {
      vcl_size_t Apos1 = i;
      vcl_size_t Apos2 = i + blockSize;
      vcl_size_t Bpos = B.size2();
      if (Apos2 > A.size1())
      {
        inplace_solve_kernel(viennacl::project(A, viennacl::range(Apos1, A.size1()), viennacl::range(Apos1, A.size2())),
                             viennacl::project(B, viennacl::range(Apos1, A.size1()), viennacl::range(0, Bpos)),
                             SolverTagT());
        break;
      }
      inplace_solve_kernel(viennacl::project(A, viennacl::range(Apos1, Apos2), viennacl::range(Apos1, Apos2)),
                           viennacl::project(B, viennacl::range(Apos1, Apos2), viennacl::range(0, Bpos)),
                           SolverTagT());
      if (Apos2 < A.size1())
      {
        viennacl::project(B, viennacl::range(Apos2, B.size1()), viennacl::range(0, Bpos)) -=
                          viennacl::linalg::prod(viennacl::project(const_cast<MatrixT1 &>(A), viennacl::range(Apos2, A.size1()), viennacl::range(Apos1, Apos2)),
                                                 viennacl::project(B, viennacl::range(Apos1, Apos2), viennacl::range(0, Bpos)));
      }
    }
  }
}

template<typename MatrixT1, typename MatrixT2>
void inplace_solve_impl(MatrixT1 const & A, MatrixT2 & B, viennacl::linalg::lower_tag)
{
  inplace_solve_lower_impl(A, B, viennacl::linalg::lower_tag());
}

template<typename MatrixT1, typename MatrixT2>
void inplace_solve_impl(MatrixT1 const & A, MatrixT2 & B, viennacl::linalg::unit_lower_tag)
{
  inplace_solve_lower_impl(A, B, viennacl::linalg::unit_lower_tag());
}

template<typename MatrixT1, typename MatrixT2, typename SolverTagT>
void inplace_solve_upper_impl(MatrixT1 const & A, MatrixT2 & B, SolverTagT)
{
  int blockSize = VIENNACL_DIRECT_SOLVE_BLOCKSIZE;
  if (static_cast<int>(A.size1()) <= blockSize)
    inplace_solve_kernel(A, B, SolverTagT());
  else
  {
    for (int i = static_cast<int>(A.size1()); i > 0; i = i - blockSize)
    {
      int Apos1 = i - blockSize;
      vcl_size_t Apos2 = i;
      vcl_size_t Bpos = B.size2();
      if (Apos1 < 0)
      {
        inplace_solve_kernel(viennacl::project(A, viennacl::range(0, Apos2), viennacl::range(0, Apos2)),
                             viennacl::project(B, viennacl::range(0, Apos2), viennacl::range(0, Bpos)),
                             SolverTagT());
        break;
      }
      inplace_solve_kernel(viennacl::project(A, viennacl::range(vcl_size_t(Apos1), Apos2), viennacl::range(vcl_size_t(Apos1), Apos2)),
                           viennacl::project(B, viennacl::range(vcl_size_t(Apos1), Apos2), viennacl::range(0, Bpos)),
                           SolverTagT());
      if (Apos1 > 0)
      {
        viennacl::project(B, viennacl::range(0, vcl_size_t(Apos1)), viennacl::range(0, Bpos)) -=
                          viennacl::linalg::prod(viennacl::project(const_cast<MatrixT1 &>(A), viennacl::range(0, vcl_size_t(Apos1)), viennacl::range(vcl_size_t(Apos1), Apos2)),
                                                 viennacl::project(B, viennacl::range(vcl_size_t(Apos1), Apos2), viennacl::range(0, Bpos)));
      }
    }
  }
}

template<typename MatrixT1, typename MatrixT2>
void inplace_solve_impl(MatrixT1 const & A, MatrixT2 & B, viennacl::linalg::upper_tag)
{
  inplace_solve_upper_impl(A, B, viennacl::linalg::upper_tag());
}

template<typename MatrixT1, typename MatrixT2>
void inplace_solve_impl(MatrixT1 const & A, MatrixT2 & B, viennacl::linalg::unit_upper_tag)
{
  inplace_solve_upper_impl(A, B, viennacl::linalg::unit_upper_tag());
}

/** @brief Direct inplace solver for triangular systems with multiple right hand sides, i.e. A \ B   (MATLAB notation)
*
* @param A      The system matrix
* @param B      The matrix of row vectors, where the solution is directly written to
*/
template<typename NumericT, typename SolverTagT>
void inplace_solve(const matrix_base<NumericT> & A,
                   matrix_base<NumericT> & B,
                   SolverTagT)
{
  inplace_solve_impl(A,B,SolverTagT());
}

/** @brief Direct inplace solver for triangular systems with multiple transposed right hand sides, i.e. A \ B^T   (MATLAB notation)
*
* @param A       The system matrix
* @param proxy_B The proxy for the transposed matrix of row vectors, where the solution is directly written to
*/
template<typename NumericT, typename SolverTagT>
void inplace_solve(const matrix_base<NumericT> & A,
                   matrix_expression<const matrix_base<NumericT>, const matrix_base<NumericT>, op_trans> proxy_B,
                   SolverTagT)
{
  matrix_base<NumericT> B(proxy_B);
  inplace_solve_impl(A,B,SolverTagT());
  B=trans(B);
  const_cast<matrix_base<NumericT> &>(proxy_B.lhs()) = B;
}

//upper triangular solver for transposed lower triangular matrices
/** @brief Direct inplace solver for transposed triangular systems with multiple right hand sides, i.e. A^T \ B   (MATLAB notation)
*
* @param proxy_A  The transposed system matrix proxy
* @param B        The matrix holding the load vectors, where the solution is directly written to
*/
template<typename NumericT, typename SolverTagT>
void inplace_solve(const matrix_expression< const matrix_base<NumericT>, const matrix_base<NumericT>, op_trans>  & proxy_A,
                   matrix_base<NumericT> & B,
                   SolverTagT)
{
  matrix_base<NumericT> A(proxy_A);
  inplace_solve_impl(A,B,SolverTagT());
}

/** @brief Direct inplace solver for transposed triangular systems with multiple transposed right hand sides, i.e. A^T \ B^T   (MATLAB notation)
*
* @param proxy_A    The transposed system matrix proxy
* @param proxy_B    The transposed matrix holding the load vectors, where the solution is directly written to
*/
template<typename NumericT, typename SolverTagT>
void inplace_solve(matrix_expression< const matrix_base<NumericT>, const matrix_base<NumericT>, op_trans> const & proxy_A,
                   matrix_expression< const matrix_base<NumericT>, const matrix_base<NumericT>, op_trans>         proxy_B,
                   SolverTagT)
{
  matrix_base<NumericT> A(proxy_A);
  matrix_base<NumericT> B(proxy_B);
  inplace_solve_impl(A,B,SolverTagT());
  B=trans(B);
  const_cast<matrix_base<NumericT> &>(proxy_B.lhs()) = B;
}


/////////////////// general wrappers for non-inplace solution //////////////////////


/** @brief Convenience functions for C = solve(A, B, some_tag()); Creates a temporary result matrix and forwards the request to inplace_solve()
*
* @param A    The system matrix
* @param B    The matrix of load vectors
* @param tag    Dispatch tag
*/
template<typename NumericT, typename SolverTagT>
matrix_base<NumericT> solve(const matrix_base<NumericT> & A,
                            const matrix_base<NumericT> & B,
                            SolverTagT tag)
{
  // do an inplace solve on the result vector:
  matrix_base<NumericT> result(B);
  inplace_solve(A, result, tag);
  return result;
}

/** @brief Convenience functions for C = solve(A, B^T, some_tag()); Creates a temporary result matrix and forwards the request to inplace_solve()
*
* @param A    The system matrix
* @param proxy  The transposed load vector
* @param tag    Dispatch tag
*/
template<typename NumericT, typename SolverTagT>
matrix_base<NumericT> solve(const matrix_base<NumericT> & A,
                            const matrix_expression< const matrix_base<NumericT>, const matrix_base<NumericT>, op_trans> & proxy,
                            SolverTagT tag)
{
  // do an inplace solve on the result vector:
  matrix_base<NumericT> result(proxy);
  inplace_solve(A, result, tag);
  return result;
}

/** @brief Convenience functions for result = solve(trans(mat), B, some_tag()); Creates a temporary result matrix and forwards the request to inplace_solve()
*
* @param proxy  The transposed system matrix proxy
* @param B      The matrix of load vectors
* @param tag    Dispatch tag
*/
template<typename NumericT, typename SolverTagT>
matrix_base<NumericT> solve(const matrix_expression< const matrix_base<NumericT>, const matrix_base<NumericT>, op_trans> & proxy,
                            const matrix_base<NumericT> & B,
                            SolverTagT tag)
{
  // do an inplace solve on the result vector:
  matrix_base<NumericT> result(B);
  inplace_solve(proxy, result, tag);
  return result;
}

/** @brief Convenience functions for result = solve(trans(mat), vec, some_tag()); Creates a temporary result vector and forwards the request to inplace_solve()
*
* @param proxy_A  The transposed system matrix proxy
* @param proxy_B  The transposed matrix of load vectors, where the solution is directly written to
* @param tag    Dispatch tag
*/
template<typename NumericT, typename SolverTagT>
matrix_base<NumericT> solve(const matrix_expression< const matrix_base<NumericT>, const matrix_base<NumericT>, op_trans> & proxy_A,
                            const matrix_expression< const matrix_base<NumericT>, const matrix_base<NumericT>, op_trans> & proxy_B,
                            SolverTagT tag)
{
  // run an inplace solve on the result vector:
  matrix_base<NumericT> result(proxy_B);
  inplace_solve(proxy_A, result, tag);
  return result;
}

////vector Rutine:

template<typename MatrixT1, typename VectorT, typename SolverTagT>
void inplace_solve_lower_vec_impl(MatrixT1 const & A, VectorT & B, SolverTagT)
{
  vcl_size_t blockSize = VIENNACL_DIRECT_SOLVE_BLOCKSIZE;
  if (A.size1() < blockSize)
    inplace_solve_vec_kernel(A, B, SolverTagT());
  else
  {
    for (vcl_size_t i = 0; i < A.size1(); i = i + blockSize)
    {
      vcl_size_t Apos1 = i;
      vcl_size_t Apos2 = i + blockSize;
      if (i > A.size2())
      {
        inplace_solve_vec_kernel(viennacl::project(A, viennacl::range(Apos1, A.size1()), viennacl::range(Apos1, A.size2())),
                                 viennacl::project(B, viennacl::range(Apos1, A.size1())),
                                 SolverTagT());
        break;
      }
      inplace_solve_vec_kernel(viennacl::project(A, viennacl::range(Apos1, Apos2), viennacl::range(Apos1, Apos2)),
                               viennacl::project(B, viennacl::range(Apos1, Apos2)),
                               SolverTagT());
      if (Apos2 < A.size1())
      {
        VectorT temp(viennacl::linalg::prod(viennacl::project(A, viennacl::range(Apos2, A.size1()), viennacl::range(Apos1, Apos2)),
                     viennacl::project(B, viennacl::range(Apos1, Apos2))));
        viennacl::project(B, viennacl::range(Apos2, A.size1())) -= temp;
      }
    }
  }
}

template<typename MatrixT1, typename VectorT>
void inplace_solve_vec_impl(MatrixT1 const & A, VectorT & B, viennacl::linalg::lower_tag)
{
  inplace_solve_lower_vec_impl(A, B, viennacl::linalg::lower_tag());
}

template<typename MatrixT1, typename VectorT>
void inplace_solve_vec_impl(MatrixT1 const & A, VectorT & B, viennacl::linalg::unit_lower_tag)
{
  inplace_solve_lower_vec_impl(A, B, viennacl::linalg::unit_lower_tag());
}

template<typename MatrixT1, typename VectorT, typename SolverTagT>
void inplace_solve_upper_vec_impl(MatrixT1 const & A, VectorT & B, SolverTagT)
{
  unsigned int blockSize = VIENNACL_DIRECT_SOLVE_BLOCKSIZE;
  if (A.size1() < blockSize)
    inplace_solve_vec_kernel(A, B, SolverTagT());
  else
  {
    for (int i = static_cast<int>(A.size1()); i > 0; i = i - blockSize)
    {
      int Apos1 = i - blockSize;
      vcl_size_t Apos2 = vcl_size_t(i);
      if (Apos1 < 0)
      {
        inplace_solve_vec_kernel(viennacl::project(A, viennacl::range(0, Apos2), viennacl::range(0, Apos2)),
                                 viennacl::project(B, viennacl::range(0, Apos2)),
                                 SolverTagT());
        break;
      }
      inplace_solve_vec_kernel(viennacl::project(A, viennacl::range(vcl_size_t(Apos1), Apos2), viennacl::range(vcl_size_t(Apos1), Apos2)),
                               viennacl::project(B, viennacl::range(vcl_size_t(Apos1), Apos2)),
                               SolverTagT());
      if (Apos1 > 0)
      {
        VectorT temp(viennacl::linalg::prod(viennacl::project(A, viennacl::range(0, vcl_size_t(Apos1)), viennacl::range(vcl_size_t(Apos1), Apos2)),
                     viennacl::project(B, viennacl::range(vcl_size_t(Apos1), Apos2))));
        viennacl::project(B, viennacl::range(0, vcl_size_t(Apos1))) -= temp;
      }
    }
  }
}

template<typename MatrixT1, typename VectorT>
void inplace_solve_vec_impl(MatrixT1 const & A, VectorT & B, viennacl::linalg::upper_tag)
{
  inplace_solve_upper_vec_impl(A, B, viennacl::linalg::upper_tag());
}

template<typename MatrixT1, typename VectorT>
void inplace_solve_vec_impl(MatrixT1 const & A, VectorT & B, viennacl::linalg::unit_upper_tag)
{
  inplace_solve_upper_vec_impl(A, B, viennacl::linalg::unit_upper_tag());
}

template<typename NumericT, typename SolverTagT>
void inplace_solve(const matrix_base<NumericT> & mat,
                   vector_base<NumericT> & vec,
                   SolverTagT)
{
  inplace_solve_vec_impl(mat, vec, SolverTagT());
}

template<typename NumericT, typename SolverTagT>
void inplace_solve(matrix_expression<const matrix_base<NumericT>, const matrix_base<NumericT>, op_trans> const & proxy,
                   vector_base<NumericT> & vec,
                   SolverTagT)
{
  matrix_base<NumericT> mat(proxy);
  inplace_solve_vec_impl(mat,vec,SolverTagT());
}

/** @brief Convenience functions for result = solve(mat, vec, some_tag()); Creates a temporary result vector and forwards the request to inplace_solve()
*
* @param mat    The system matrix
* @param vec    The load vector
* @param tag    Dispatch tag
*/
template<typename NumericT, typename SolverTagT>
vector<NumericT> solve(const matrix_base<NumericT> & mat,
                       const vector_base<NumericT> & vec,
                       SolverTagT const & tag)
{
// do an inplace solve on the result vector:
  vector<NumericT> result(vec);
  inplace_solve(mat, result, tag);
  return result;
}

/** @brief Convenience functions for result = solve(trans(mat), vec, some_tag()); Creates a temporary result vector and forwards the request to inplace_solve()
*
* @param proxy  The transposed system matrix proxy
* @param vec    The load vector, where the solution is directly written to
* @param tag    Dispatch tag
*/
template<typename NumericT, typename SolverTagT>
vector<NumericT> solve(const matrix_expression< const matrix_base<NumericT>, const matrix_base<NumericT>, op_trans> & proxy,
                       const vector_base<NumericT> & vec,
                       SolverTagT const & tag)
{
  // run an inplace solve on the result vector:
  vector<NumericT> result(vec);
  inplace_solve(proxy, result, tag);
  return result;
}


}
}

#endif
