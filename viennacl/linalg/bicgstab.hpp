#ifndef VIENNACL_LINALG_BICGSTAB_HPP_
#define VIENNACL_LINALG_BICGSTAB_HPP_

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

/** @file bicgstab.hpp
    @brief The stabilized bi-conjugate gradient method is implemented here
*/

#include <vector>
#include <cmath>
#include <numeric>

#include "viennacl/forwards.h"
#include "viennacl/tools/tools.hpp"
#include "viennacl/linalg/prod.hpp"
#include "viennacl/linalg/inner_prod.hpp"
#include "viennacl/linalg/norm_2.hpp"
#include "viennacl/traits/clear.hpp"
#include "viennacl/traits/size.hpp"
#include "viennacl/traits/context.hpp"
#include "viennacl/meta/result_of.hpp"
#include "viennacl/linalg/iterative_operations.hpp"

namespace viennacl
{
namespace linalg
{

/** @brief A tag for the stabilized Bi-conjugate gradient solver. Used for supplying solver parameters and for dispatching the solve() function
*/
class bicgstab_tag
{
public:
  /** @brief The constructor
  *
  * @param tol              Relative tolerance for the residual (solver quits if ||r|| < tol * ||r_initial||)
  * @param max_iters        The maximum number of iterations
  * @param max_iters_before_restart   The maximum number of iterations before BiCGStab is reinitialized (to avoid accumulation of round-off errors)
  */
  bicgstab_tag(double tol = 1e-8, vcl_size_t max_iters = 400, vcl_size_t max_iters_before_restart = 200)
    : tol_(tol), iterations_(max_iters), iterations_before_restart_(max_iters_before_restart) {}

  /** @brief Returns the relative tolerance */
  double tolerance() const { return tol_; }
  /** @brief Returns the maximum number of iterations */
  vcl_size_t max_iterations() const { return iterations_; }
  /** @brief Returns the maximum number of iterations before a restart*/
  vcl_size_t max_iterations_before_restart() const { return iterations_before_restart_; }

  /** @brief Return the number of solver iterations: */
  vcl_size_t iters() const { return iters_taken_; }
  void iters(vcl_size_t i) const { iters_taken_ = i; }

  /** @brief Returns the estimated relative error at the end of the solver run */
  double error() const { return last_error_; }
  /** @brief Sets the estimated relative error at the end of the solver run */
  void error(double e) const { last_error_ = e; }

private:
  double tol_;
  vcl_size_t iterations_;
  vcl_size_t iterations_before_restart_;

  //return values from solver
  mutable vcl_size_t iters_taken_;
  mutable double last_error_;
};



namespace detail
{
  /** @brief Implementation of a pipelined stabilized Bi-conjugate gradient solver */
  template<typename MatrixT, typename NumericT>
  viennacl::vector<NumericT> pipelined_solve(MatrixT const & A, //MatrixType const & A,
                                             viennacl::vector_base<NumericT> const & rhs,
                                             bicgstab_tag const & tag,
                                             viennacl::linalg::no_precond)
  {
    viennacl::vector<NumericT> result = viennacl::zero_vector<NumericT>(rhs.size(), viennacl::traits::context(rhs));

    viennacl::vector<NumericT> residual = rhs;
    viennacl::vector<NumericT> p = rhs;
    viennacl::vector<NumericT> r0star = rhs;
    viennacl::vector<NumericT> Ap = rhs;
    viennacl::vector<NumericT> s  = rhs;
    viennacl::vector<NumericT> As = rhs;

    // Layout of temporary buffer:
    //  chunk 0: <residual, r_0^*>
    //  chunk 1: <As, As>
    //  chunk 2: <As, s>
    //  chunk 3: <Ap, r_0^*>
    //  chunk 4: <As, r_0^*>
    //  chunk 5: <s, s>
    vcl_size_t buffer_size_per_vector = 256;
    vcl_size_t num_buffer_chunks = 6;
    viennacl::vector<NumericT> inner_prod_buffer = viennacl::zero_vector<NumericT>(num_buffer_chunks*buffer_size_per_vector, viennacl::traits::context(rhs)); // temporary buffer
    std::vector<NumericT>      host_inner_prod_buffer(inner_prod_buffer.size());

    NumericT norm_rhs_host = viennacl::linalg::norm_2(residual);
    NumericT beta;
    NumericT alpha;
    NumericT omega;
    NumericT residual_norm = norm_rhs_host;
    inner_prod_buffer[0] = norm_rhs_host * norm_rhs_host;

    NumericT  r_dot_r0 = 0;
    NumericT As_dot_As = 0;
    NumericT As_dot_s  = 0;
    NumericT Ap_dot_r0 = 0;
    NumericT As_dot_r0 = 0;
    NumericT  s_dot_s  = 0;

    if (norm_rhs_host <= 0) //solution is zero if RHS norm is zero
      return result;

    for (vcl_size_t i = 0; i < tag.max_iterations(); ++i)
    {
      tag.iters(i+1);
      // Ap = A*p_j
      // Ap_dot_r0 = <Ap, r_0^*>
      viennacl::linalg::pipelined_bicgstab_prod(A, p, Ap, r0star,
                                                inner_prod_buffer, buffer_size_per_vector, 3*buffer_size_per_vector);

      //////// first (weak) synchronization point ////

      ///// method 1: compute alpha on host:
      //
      //// we only need the second chunk of the buffer for computing Ap_dot_r0:
      //viennacl::fast_copy(inner_prod_buffer.begin(), inner_prod_buffer.end(), host_inner_prod_buffer.begin());
      //Ap_dot_r0 = std::accumulate(host_inner_prod_buffer.begin() +     buffer_size_per_vector, host_inner_prod_buffer.begin() + 2 * buffer_size_per_vector, ScalarType(0));

      //alpha = residual_dot_r0 / Ap_dot_r0;

      //// s_j = r_j - alpha_j q_j
      //s = residual - alpha * Ap;

      ///// method 2: compute alpha on device:
      // s = r - alpha * Ap
      // <s, s> first stage
      // dump alpha at end of inner_prod_buffer
      viennacl::linalg::pipelined_bicgstab_update_s(s, residual, Ap,
                                                    inner_prod_buffer, buffer_size_per_vector, 5*buffer_size_per_vector);

      // As = A*s_j
      // As_dot_As = <As, As>
      // As_dot_s  = <As, s>
      // As_dot_r0 = <As, r_0^*>
      viennacl::linalg::pipelined_bicgstab_prod(A, s, As, r0star,
                                                inner_prod_buffer, buffer_size_per_vector, 4*buffer_size_per_vector);

      //////// second (strong) synchronization point ////

      viennacl::fast_copy(inner_prod_buffer.begin(), inner_prod_buffer.end(), host_inner_prod_buffer.begin());

       r_dot_r0 = std::accumulate(host_inner_prod_buffer.begin(),                              host_inner_prod_buffer.begin() +     buffer_size_per_vector, NumericT(0));
      As_dot_As = std::accumulate(host_inner_prod_buffer.begin() +     buffer_size_per_vector, host_inner_prod_buffer.begin() + 2 * buffer_size_per_vector, NumericT(0));
      As_dot_s  = std::accumulate(host_inner_prod_buffer.begin() + 2 * buffer_size_per_vector, host_inner_prod_buffer.begin() + 3 * buffer_size_per_vector, NumericT(0));
      Ap_dot_r0 = std::accumulate(host_inner_prod_buffer.begin() + 3 * buffer_size_per_vector, host_inner_prod_buffer.begin() + 4 * buffer_size_per_vector, NumericT(0));
      As_dot_r0 = std::accumulate(host_inner_prod_buffer.begin() + 4 * buffer_size_per_vector, host_inner_prod_buffer.begin() + 5 * buffer_size_per_vector, NumericT(0));
       s_dot_s  = std::accumulate(host_inner_prod_buffer.begin() + 5 * buffer_size_per_vector, host_inner_prod_buffer.begin() + 6 * buffer_size_per_vector, NumericT(0));

      alpha =   r_dot_r0 / Ap_dot_r0;
      beta  = - As_dot_r0 / Ap_dot_r0;
      omega =   As_dot_s  / As_dot_As;

      residual_norm = std::sqrt(s_dot_s - NumericT(2.0) * omega * As_dot_s + omega * omega *  As_dot_As);
      if (std::fabs(residual_norm / norm_rhs_host) < tag.tolerance())
        break;

      // x_{j+1} = x_j + alpha * p_j + omega * s_j
      // r_{j+1} = s_j - omega * t_j
      // p_{j+1} = r_{j+1} + beta * (p_j - omega * q_j)
      // and compute first stage of r_dot_r0 = <r_{j+1}, r_o^*> for use in next iteration
       viennacl::linalg::pipelined_bicgstab_vector_update(result, alpha, p, omega, s,
                                                          residual, As,
                                                          beta, Ap,
                                                          r0star, inner_prod_buffer, buffer_size_per_vector);
    }

    //store last error estimate:
    tag.error(residual_norm / norm_rhs_host);

    return result;
  }
}

// compressed_matrix

/** @brief Overload for the pipelined BiCGStab implementation for the ViennaCL sparse matrix types */
template<typename NumericT>
viennacl::vector<NumericT> solve(viennacl::compressed_matrix<NumericT> const & A,
                                 viennacl::vector_base<NumericT> const & rhs,
                                 bicgstab_tag const & tag,
                                 viennacl::linalg::no_precond)
{
  return detail::pipelined_solve(A, rhs, tag, viennacl::linalg::no_precond());
}

/** @brief Overload for the pipelined BiCGStab implementation for the ViennaCL sparse matrix types */
template<typename NumericT>
viennacl::vector<NumericT> solve(viennacl::compressed_matrix<NumericT> const & A,
                                 viennacl::vector<NumericT> const & rhs,
                                 bicgstab_tag const & tag,
                                 viennacl::linalg::no_precond)
{
  return detail::pipelined_solve(A, rhs, tag, viennacl::linalg::no_precond());
}

/** @brief Overload for the pipelined BiCGStab implementation for the ViennaCL sparse matrix types */
template<typename NumericT>
viennacl::vector<NumericT> solve(viennacl::compressed_matrix<NumericT> const & A,
                                 viennacl::vector_range<NumericT> const & rhs,
                                 bicgstab_tag const & tag,
                                 viennacl::linalg::no_precond)
{
  return detail::pipelined_solve(A, rhs, tag, viennacl::linalg::no_precond());
}

/** @brief Overload for the pipelined BiCGStab implementation for the ViennaCL sparse matrix types */
template<typename NumericT>
viennacl::vector<NumericT> solve(viennacl::compressed_matrix<NumericT> const & A,
                                 viennacl::vector_slice<NumericT> const & rhs,
                                 bicgstab_tag const & tag,
                                 viennacl::linalg::no_precond)
{
  return detail::pipelined_solve(A, rhs, tag, viennacl::linalg::no_precond());
}


// coordinate_matrix

/** @brief Overload for the pipelined BiCGStab implementation for the ViennaCL sparse matrix types */
template<typename NumericT>
viennacl::vector<NumericT> solve(viennacl::coordinate_matrix<NumericT> const & A,
                                 viennacl::vector_base<NumericT> const & rhs,
                                 bicgstab_tag const & tag,
                                 viennacl::linalg::no_precond)
{
  return detail::pipelined_solve(A, rhs, tag, viennacl::linalg::no_precond());
}

/** @brief Overload for the pipelined BiCGStab implementation for the ViennaCL sparse matrix types */
template<typename NumericT>
viennacl::vector<NumericT> solve(viennacl::coordinate_matrix<NumericT> const & A,
                                 viennacl::vector<NumericT> const & rhs,
                                 bicgstab_tag const & tag,
                                 viennacl::linalg::no_precond)
{
  return detail::pipelined_solve(A, rhs, tag, viennacl::linalg::no_precond());
}

/** @brief Overload for the pipelined BiCGStab implementation for the ViennaCL sparse matrix types */
template<typename NumericT>
viennacl::vector<NumericT> solve(viennacl::coordinate_matrix<NumericT> const & A,
                                 viennacl::vector_range<NumericT> const & rhs,
                                 bicgstab_tag const & tag,
                                 viennacl::linalg::no_precond)
{
  return detail::pipelined_solve(A, rhs, tag, viennacl::linalg::no_precond());
}

/** @brief Overload for the pipelined BiCGStab implementation for the ViennaCL sparse matrix types */
template<typename NumericT>
viennacl::vector<NumericT> solve(viennacl::coordinate_matrix<NumericT> const & A,
                                 viennacl::vector_slice<NumericT> const & rhs,
                                 bicgstab_tag const & tag,
                                 viennacl::linalg::no_precond)
{
  return detail::pipelined_solve(A, rhs, tag, viennacl::linalg::no_precond());
}

// ell_matrix

/** @brief Overload for the pipelined BiCGStab implementation for the ViennaCL sparse matrix types */
template<typename NumericT>
viennacl::vector<NumericT> solve(viennacl::ell_matrix<NumericT> const & A,
                                 viennacl::vector_base<NumericT> const & rhs,
                                 bicgstab_tag const & tag,
                                 viennacl::linalg::no_precond)
{
  return detail::pipelined_solve(A, rhs, tag, viennacl::linalg::no_precond());
}

/** @brief Overload for the pipelined BiCGStab implementation for the ViennaCL sparse matrix types */
template<typename NumericT>
viennacl::vector<NumericT> solve(viennacl::ell_matrix<NumericT> const & A,
                                 viennacl::vector<NumericT> const & rhs,
                                 bicgstab_tag const & tag,
                                 viennacl::linalg::no_precond)
{
  return detail::pipelined_solve(A, rhs, tag, viennacl::linalg::no_precond());
}

/** @brief Overload for the pipelined BiCGStab implementation for the ViennaCL sparse matrix types */
template<typename NumericT>
viennacl::vector<NumericT> solve(viennacl::ell_matrix<NumericT> const & A,
                                 viennacl::vector_range<NumericT> const & rhs,
                                 bicgstab_tag const & tag,
                                 viennacl::linalg::no_precond)
{
  return detail::pipelined_solve(A, rhs, tag, viennacl::linalg::no_precond());
}

/** @brief Overload for the pipelined BiCGStab implementation for the ViennaCL sparse matrix types */
template<typename NumericT>
viennacl::vector<NumericT> solve(viennacl::ell_matrix<NumericT> const & A,
                                 viennacl::vector_slice<NumericT> const & rhs,
                                 bicgstab_tag const & tag,
                                 viennacl::linalg::no_precond)
{
  return detail::pipelined_solve(A, rhs, tag, viennacl::linalg::no_precond());
}

// sliced_ell_matrix

/** @brief Overload for the pipelined BiCGStab implementation for the ViennaCL sparse matrix types */
template<typename NumericT>
viennacl::vector<NumericT> solve(viennacl::sliced_ell_matrix<NumericT> const & A,
                                 viennacl::vector_base<NumericT> const & rhs,
                                 bicgstab_tag const & tag,
                                 viennacl::linalg::no_precond)
{
  return detail::pipelined_solve(A, rhs, tag, viennacl::linalg::no_precond());
}

/** @brief Overload for the pipelined BiCGStab implementation for the ViennaCL sparse matrix types */
template<typename NumericT>
viennacl::vector<NumericT> solve(viennacl::sliced_ell_matrix<NumericT> const & A,
                                 viennacl::vector<NumericT> const & rhs,
                                 bicgstab_tag const & tag,
                                 viennacl::linalg::no_precond)
{
  return detail::pipelined_solve(A, rhs, tag, viennacl::linalg::no_precond());
}

/** @brief Overload for the pipelined BiCGStab implementation for the ViennaCL sparse matrix types */
template<typename NumericT>
viennacl::vector<NumericT> solve(viennacl::sliced_ell_matrix<NumericT> const & A,
                                 viennacl::vector_range<NumericT> const & rhs,
                                 bicgstab_tag const & tag,
                                 viennacl::linalg::no_precond)
{
  return detail::pipelined_solve(A, rhs, tag, viennacl::linalg::no_precond());
}

/** @brief Overload for the pipelined BiCGStab implementation for the ViennaCL sparse matrix types */
template<typename NumericT>
viennacl::vector<NumericT> solve(viennacl::sliced_ell_matrix<NumericT> const & A,
                                 viennacl::vector_slice<NumericT> const & rhs,
                                 bicgstab_tag const & tag,
                                 viennacl::linalg::no_precond)
{
  return detail::pipelined_solve(A, rhs, tag, viennacl::linalg::no_precond());
}


// hyb_matrix

/** @brief Overload for the pipelined BiCGStab implementation for the ViennaCL sparse matrix types */
template<typename NumericT>
viennacl::vector<NumericT> solve(viennacl::hyb_matrix<NumericT> const & A,
                                 viennacl::vector_base<NumericT> const & rhs,
                                 bicgstab_tag const & tag,
                                 viennacl::linalg::no_precond)
{
  return detail::pipelined_solve(A, rhs, tag, viennacl::linalg::no_precond());
}

/** @brief Overload for the pipelined BiCGStab implementation for the ViennaCL sparse matrix types */
template<typename NumericT>
viennacl::vector<NumericT> solve(viennacl::hyb_matrix<NumericT> const & A,
                                 viennacl::vector<NumericT> const & rhs,
                                 bicgstab_tag const & tag,
                                 viennacl::linalg::no_precond)
{
  return detail::pipelined_solve(A, rhs, tag, viennacl::linalg::no_precond());
}

/** @brief Overload for the pipelined BiCGStab implementation for the ViennaCL sparse matrix types */
template<typename NumericT>
viennacl::vector<NumericT> solve(viennacl::hyb_matrix<NumericT> const & A,
                                 viennacl::vector_range<NumericT> const & rhs,
                                 bicgstab_tag const & tag,
                                 viennacl::linalg::no_precond)
{
  return detail::pipelined_solve(A, rhs, tag, viennacl::linalg::no_precond());
}

/** @brief Overload for the pipelined BiCGStab implementation for the ViennaCL sparse matrix types */
template<typename NumericT>
viennacl::vector<NumericT> solve(viennacl::hyb_matrix<NumericT> const & A,
                                 viennacl::vector_slice<NumericT> const & rhs,
                                 bicgstab_tag const & tag,
                                 viennacl::linalg::no_precond)
{
  return detail::pipelined_solve(A, rhs, tag, viennacl::linalg::no_precond());
}





/** @brief Implementation of the stabilized Bi-conjugate gradient solver
*
* Following the description in "Iterative Methods for Sparse Linear Systems" by Y. Saad
*
* @param matrix     The system matrix
* @param rhs        The load vector
* @param tag        Solver configuration tag
* @return The result vector
*/
template<typename MatrixT, typename VectorT>
VectorT solve(MatrixT const & matrix, VectorT const & rhs, bicgstab_tag const & tag)
{
  typedef typename viennacl::result_of::value_type<VectorT>::type            NumericType;
  typedef typename viennacl::result_of::cpu_value_type<NumericType>::type    CPU_NumericType;
  VectorT result = rhs;
  viennacl::traits::clear(result);

  VectorT residual = rhs;
  VectorT p = rhs;
  VectorT r0star = rhs;
  VectorT tmp0 = rhs;
  VectorT tmp1 = rhs;
  VectorT s = rhs;

  CPU_NumericType norm_rhs_host = viennacl::linalg::norm_2(residual);
  CPU_NumericType ip_rr0star = norm_rhs_host * norm_rhs_host;
  CPU_NumericType beta;
  CPU_NumericType alpha;
  CPU_NumericType omega;
  //ScalarType inner_prod_temp; //temporary variable for inner product computation
  CPU_NumericType new_ip_rr0star = 0;
  CPU_NumericType residual_norm = norm_rhs_host;

  if (norm_rhs_host <= 0) //solution is zero if RHS norm is zero
    return result;

  bool restart_flag = true;
  vcl_size_t last_restart = 0;
  for (vcl_size_t i = 0; i < tag.max_iterations(); ++i)
  {
    if (restart_flag)
    {
      residual = rhs;
      residual -= viennacl::linalg::prod(matrix, result);
      p = residual;
      r0star = residual;
      ip_rr0star = viennacl::linalg::norm_2(residual);
      ip_rr0star *= ip_rr0star;
      restart_flag = false;
      last_restart = i;
    }

    tag.iters(i+1);
    tmp0 = viennacl::linalg::prod(matrix, p);
    alpha = ip_rr0star / viennacl::linalg::inner_prod(tmp0, r0star);

    s = residual - alpha*tmp0;

    tmp1 = viennacl::linalg::prod(matrix, s);
    CPU_NumericType norm_tmp1 = viennacl::linalg::norm_2(tmp1);
    omega = viennacl::linalg::inner_prod(tmp1, s) / (norm_tmp1 * norm_tmp1);

    result += alpha * p + omega * s;
    residual = s - omega * tmp1;

    new_ip_rr0star = viennacl::linalg::inner_prod(residual, r0star);
    residual_norm = viennacl::linalg::norm_2(residual);
    if (std::fabs(residual_norm / norm_rhs_host) < tag.tolerance())
      break;

    beta = new_ip_rr0star / ip_rr0star * alpha/omega;
    ip_rr0star = new_ip_rr0star;

    if (    (ip_rr0star <= 0 && ip_rr0star >= 0)
         || (omega <= 0 && omega >= 0)
         || (i - last_restart > tag.max_iterations_before_restart())
       ) //search direction degenerate. A restart might help
      restart_flag = true;

    // Execution of
    //  p = residual + beta * (p - omega*tmp0);
    // without introducing temporary vectors:
    p -= omega * tmp0;
    p = residual + beta * p;
  }

  //store last error estimate:
  tag.error(residual_norm / norm_rhs_host);

  return result;
}

template<typename MatrixT, typename VectorT>
VectorT solve(MatrixT const & matrix, VectorT const & rhs, bicgstab_tag const & tag, viennacl::linalg::no_precond)
{
  return solve(matrix, rhs, tag);
}

/** @brief Implementation of the preconditioned stabilized Bi-conjugate gradient solver
*
* Following the description of the unpreconditioned case in "Iterative Methods for Sparse Linear Systems" by Y. Saad
*
* @param matrix     The system matrix
* @param rhs        The load vector
* @param tag        Solver configuration tag
* @param precond    A preconditioner. Precondition operation is done via member function apply()
* @return The result vector
*/
template<typename MatrixT, typename VectorT, typename PreconditionerT>
VectorT solve(MatrixT const & matrix, VectorT const & rhs, bicgstab_tag const & tag, PreconditionerT const & precond)
{
  typedef typename viennacl::result_of::value_type<VectorT>::type            NumericType;
  typedef typename viennacl::result_of::cpu_value_type<NumericType>::type    CPU_NumericType;
  VectorT result = rhs;
  viennacl::traits::clear(result);

  VectorT residual = rhs;
  VectorT r0star = residual;  //can be chosen arbitrarily in fact
  VectorT tmp0 = rhs;
  VectorT tmp1 = rhs;
  VectorT s = rhs;

  VectorT p = residual;

  CPU_NumericType ip_rr0star = viennacl::linalg::norm_2(residual);
  CPU_NumericType norm_rhs_host = viennacl::linalg::norm_2(residual);
  CPU_NumericType beta;
  CPU_NumericType alpha;
  CPU_NumericType omega;
  CPU_NumericType new_ip_rr0star = 0;
  CPU_NumericType residual_norm = norm_rhs_host;

  if (!norm_rhs_host) //solution is zero if RHS norm is zero
    return result;

  bool restart_flag = true;
  vcl_size_t last_restart = 0;
  for (unsigned int i = 0; i < tag.max_iterations(); ++i)
  {
    if (restart_flag)
    {
      residual = rhs;
      residual -= viennacl::linalg::prod(matrix, result);
      precond.apply(residual);
      p = residual;
      r0star = residual;
      ip_rr0star = viennacl::linalg::norm_2(residual);
      ip_rr0star *= ip_rr0star;
      restart_flag = false;
      last_restart = i;
    }

    tag.iters(i+1);
    tmp0 = viennacl::linalg::prod(matrix, p);
    precond.apply(tmp0);
    alpha = ip_rr0star / viennacl::linalg::inner_prod(tmp0, r0star);

    s = residual - alpha*tmp0;

    tmp1 = viennacl::linalg::prod(matrix, s);
    precond.apply(tmp1);
    CPU_NumericType norm_tmp1 = viennacl::linalg::norm_2(tmp1);
    omega = viennacl::linalg::inner_prod(tmp1, s) / (norm_tmp1 * norm_tmp1);

    result += alpha * p + omega * s;
    residual = s - omega * tmp1;

    residual_norm = viennacl::linalg::norm_2(residual);
    if (residual_norm / norm_rhs_host < tag.tolerance())
      break;

    new_ip_rr0star = viennacl::linalg::inner_prod(residual, r0star);

    beta = new_ip_rr0star / ip_rr0star * alpha/omega;
    ip_rr0star = new_ip_rr0star;

    if (!ip_rr0star || !omega || i - last_restart > tag.max_iterations_before_restart()) //search direction degenerate. A restart might help
      restart_flag = true;

    // Execution of
    //  p = residual + beta * (p - omega*tmp0);
    // without introducing temporary vectors:
    p -= omega * tmp0;
    p = residual + beta * p;

    //std::cout << "Rel. Residual in current step: " << std::sqrt(std::fabs(viennacl::linalg::inner_prod(residual, residual) / norm_rhs_host)) << std::endl;
  }

  //store last error estimate:
  tag.error(residual_norm / norm_rhs_host);

  return result;
}

}
}

#endif
