#ifndef VIENNACL_LINALG_CG_HPP_
#define VIENNACL_LINALG_CG_HPP_

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

/** @file viennacl/linalg/cg.hpp
    @brief The conjugate gradient method is implemented here
*/

#include <vector>
#include <map>
#include <cmath>
#include <numeric>

#include "viennacl/forwards.h"
#include "viennacl/tools/tools.hpp"
#include "viennacl/linalg/ilu.hpp"
#include "viennacl/linalg/prod.hpp"
#include "viennacl/linalg/inner_prod.hpp"
#include "viennacl/linalg/norm_2.hpp"
#include "viennacl/traits/clear.hpp"
#include "viennacl/traits/size.hpp"
#include "viennacl/meta/result_of.hpp"
#include "viennacl/linalg/iterative_operations.hpp"

namespace viennacl
{
namespace linalg
{

/** @brief A tag for the conjugate gradient Used for supplying solver parameters and for dispatching the solve() function
*/
class cg_tag
{
public:
  /** @brief The constructor
  *
  * @param tol              Relative tolerance for the residual (solver quits if ||r|| < tol * ||r_initial||)
  * @param max_iterations   The maximum number of iterations
  */
  cg_tag(double tol = 1e-8, unsigned int max_iterations = 300) : tol_(tol), iterations_(max_iterations) {}

  /** @brief Returns the relative tolerance */
  double tolerance() const { return tol_; }
  /** @brief Returns the maximum number of iterations */
  unsigned int max_iterations() const { return iterations_; }

  /** @brief Return the number of solver iterations: */
  unsigned int iters() const { return iters_taken_; }
  void iters(unsigned int i) const { iters_taken_ = i; }

  /** @brief Returns the estimated relative error at the end of the solver run */
  double error() const { return last_error_; }
  /** @brief Sets the estimated relative error at the end of the solver run */
  void error(double e) const { last_error_ = e; }


private:
  double tol_;
  unsigned int iterations_;

  //return values from solver
  mutable unsigned int iters_taken_;
  mutable double last_error_;
};

namespace detail
{

  /** @brief handles the no_precond case at minimal overhead */
  template<typename VectorT, typename PreconditionerT>
  class z_handler{
  public:
    z_handler(VectorT & residual) : z_(residual){ }
    VectorT & get() { return z_; }
  private:
    VectorT z_;
  };

  template<typename VectorT>
  class z_handler<VectorT, viennacl::linalg::no_precond>{
  public:
    z_handler(VectorT & residual) : presidual_(&residual){ }
    VectorT & get() { return *presidual_; }
  private:
    VectorT * presidual_;
  };

}

namespace detail
{

  /** @brief Implementation of a pipelined conjugate gradient algorithm (no preconditioner), specialized for ViennaCL types.
  *
  * Pipelined version from A. T. Chronopoulos and C. W. Gear, J. Comput. Appl. Math. 25(2), 153–168 (1989)
  *
  * @param A          The system matrix
  * @param rhs        The load vector
  * @param tag        Solver configuration tag
  * @return The result vector
  */
  //template<typename MatrixType, typename ScalarType>
  template<typename MatrixT, typename NumericT>
  viennacl::vector<NumericT> pipelined_solve(MatrixT const & A, //MatrixType const & A,
                                             viennacl::vector<NumericT> const & rhs,
                                             cg_tag const & tag,
                                             viennacl::linalg::no_precond)
  {
    typedef typename viennacl::vector<NumericT>::difference_type   difference_type;

    viennacl::vector<NumericT> result(rhs);
    viennacl::traits::clear(result);

    viennacl::vector<NumericT> residual(rhs);
    viennacl::vector<NumericT> p(rhs);
    viennacl::vector<NumericT> Ap = viennacl::linalg::prod(A, p);
    viennacl::vector<NumericT> inner_prod_buffer = viennacl::zero_vector<NumericT>(3*256, viennacl::traits::context(rhs)); // temporary buffer
    std::vector<NumericT>      host_inner_prod_buffer(inner_prod_buffer.size());
    vcl_size_t                 buffer_size_per_vector = inner_prod_buffer.size() / 3;
    difference_type            buffer_offset_per_vector = static_cast<difference_type>(buffer_size_per_vector);

    NumericT norm_rhs_squared = viennacl::linalg::norm_2(residual); norm_rhs_squared *= norm_rhs_squared;

    if (!norm_rhs_squared) //check for early convergence of A*x = 0
      return result;

    NumericT inner_prod_rr = norm_rhs_squared;
    NumericT alpha = inner_prod_rr / viennacl::linalg::inner_prod(p, Ap);
    NumericT beta  = viennacl::linalg::norm_2(Ap); beta = (alpha * alpha * beta * beta - inner_prod_rr) / inner_prod_rr;
    NumericT inner_prod_ApAp = 0;
    NumericT inner_prod_pAp  = 0;

    for (unsigned int i = 0; i < tag.max_iterations(); ++i)
    {
      tag.iters(i+1);

      viennacl::linalg::pipelined_cg_vector_update(result, alpha, p, residual, Ap, beta, inner_prod_buffer);
      viennacl::linalg::pipelined_cg_prod(A, p, Ap, inner_prod_buffer);

      // bring back the partial results to the host:
      viennacl::fast_copy(inner_prod_buffer.begin(), inner_prod_buffer.end(), host_inner_prod_buffer.begin());

      inner_prod_rr   = std::accumulate(host_inner_prod_buffer.begin(),                                host_inner_prod_buffer.begin() +     buffer_offset_per_vector, NumericT(0));
      inner_prod_ApAp = std::accumulate(host_inner_prod_buffer.begin() +     buffer_offset_per_vector, host_inner_prod_buffer.begin() + 2 * buffer_offset_per_vector, NumericT(0));
      inner_prod_pAp  = std::accumulate(host_inner_prod_buffer.begin() + 2 * buffer_offset_per_vector, host_inner_prod_buffer.begin() + 3 * buffer_offset_per_vector, NumericT(0));

      if (std::fabs(inner_prod_rr / norm_rhs_squared) < tag.tolerance() *  tag.tolerance())    //squared norms involved here
        break;

      alpha = inner_prod_rr / inner_prod_pAp;
      beta  = (alpha*alpha*inner_prod_ApAp - inner_prod_rr) / inner_prod_rr;
    }

    //store last error estimate:
    tag.error(std::sqrt(std::fabs(inner_prod_rr) / norm_rhs_squared));

    return result;
  }
}

// compressed_matrix

/** @brief Overload for the pipelined BiCGStab implementation for the ViennaCL sparse matrix types */
template<typename NumericT>
viennacl::vector<NumericT> solve(viennacl::compressed_matrix<NumericT> const & A,
                                 viennacl::vector_base<NumericT> const & rhs,
                                 cg_tag const & tag,
                                 viennacl::linalg::no_precond)
{
  return detail::pipelined_solve(A, rhs, tag, viennacl::linalg::no_precond());
}

/** @brief Overload for the pipelined BiCGStab implementation for the ViennaCL sparse matrix types */
template<typename NumericT>
viennacl::vector<NumericT> solve(viennacl::compressed_matrix<NumericT> const & A,
                                 viennacl::vector<NumericT> const & rhs,
                                 cg_tag const & tag,
                                 viennacl::linalg::no_precond)
{
  return detail::pipelined_solve(A, rhs, tag, viennacl::linalg::no_precond());
}

/** @brief Overload for the pipelined BiCGStab implementation for the ViennaCL sparse matrix types */
template<typename NumericT>
viennacl::vector<NumericT> solve(viennacl::compressed_matrix<NumericT> const & A,
                                 viennacl::vector_range<NumericT> const & rhs,
                                 cg_tag const & tag,
                                 viennacl::linalg::no_precond)
{
  return detail::pipelined_solve(A, rhs, tag, viennacl::linalg::no_precond());
}

/** @brief Overload for the pipelined BiCGStab implementation for the ViennaCL sparse matrix types */
template<typename NumericT>
viennacl::vector<NumericT> solve(viennacl::compressed_matrix<NumericT> const & A,
                                 viennacl::vector_slice<NumericT> const & rhs,
                                 cg_tag const & tag,
                                 viennacl::linalg::no_precond)
{
  return detail::pipelined_solve(A, rhs, tag, viennacl::linalg::no_precond());
}

// coordinate_matrix

/** @brief Overload for the pipelined BiCGStab implementation for the ViennaCL sparse matrix types */
template<typename NumericT>
viennacl::vector<NumericT> solve(viennacl::coordinate_matrix<NumericT> const & A,
                                 viennacl::vector_base<NumericT> const & rhs,
                                 cg_tag const & tag,
                                 viennacl::linalg::no_precond)
{
  return detail::pipelined_solve(A, rhs, tag, viennacl::linalg::no_precond());
}

/** @brief Overload for the pipelined BiCGStab implementation for the ViennaCL sparse matrix types */
template<typename NumericT>
viennacl::vector<NumericT> solve(viennacl::coordinate_matrix<NumericT> const & A,
                                 viennacl::vector<NumericT> const & rhs,
                                 cg_tag const & tag,
                                 viennacl::linalg::no_precond)
{
  return detail::pipelined_solve(A, rhs, tag, viennacl::linalg::no_precond());
}

/** @brief Overload for the pipelined BiCGStab implementation for the ViennaCL sparse matrix types */
template<typename NumericT>
viennacl::vector<NumericT> solve(viennacl::coordinate_matrix<NumericT> const & A,
                                 viennacl::vector_range<NumericT> const & rhs,
                                 cg_tag const & tag,
                                 viennacl::linalg::no_precond)
{
  return detail::pipelined_solve(A, rhs, tag, viennacl::linalg::no_precond());
}

/** @brief Overload for the pipelined BiCGStab implementation for the ViennaCL sparse matrix types */
template<typename NumericT>
viennacl::vector<NumericT> solve(viennacl::coordinate_matrix<NumericT> const & A,
                                 viennacl::vector_slice<NumericT> const & rhs,
                                 cg_tag const & tag,
                                 viennacl::linalg::no_precond)
{
  return detail::pipelined_solve(A, rhs, tag, viennacl::linalg::no_precond());
}

// ell_matrix

/** @brief Overload for the pipelined BiCGStab implementation for the ViennaCL sparse matrix types */
template<typename NumericT>
viennacl::vector<NumericT> solve(viennacl::ell_matrix<NumericT> const & A,
                                 viennacl::vector_base<NumericT> const & rhs,
                                 cg_tag const & tag,
                                 viennacl::linalg::no_precond)
{
  return detail::pipelined_solve(A, rhs, tag, viennacl::linalg::no_precond());
}

/** @brief Overload for the pipelined BiCGStab implementation for the ViennaCL sparse matrix types */
template<typename NumericT>
viennacl::vector<NumericT> solve(viennacl::ell_matrix<NumericT> const & A,
                                 viennacl::vector<NumericT> const & rhs,
                                 cg_tag const & tag,
                                 viennacl::linalg::no_precond)
{
  return detail::pipelined_solve(A, rhs, tag, viennacl::linalg::no_precond());
}

/** @brief Overload for the pipelined BiCGStab implementation for the ViennaCL sparse matrix types */
template<typename NumericT>
viennacl::vector<NumericT> solve(viennacl::ell_matrix<NumericT> const & A,
                                 viennacl::vector_range<NumericT> const & rhs,
                                 cg_tag const & tag,
                                 viennacl::linalg::no_precond)
{
  return detail::pipelined_solve(A, rhs, tag, viennacl::linalg::no_precond());
}

/** @brief Overload for the pipelined BiCGStab implementation for the ViennaCL sparse matrix types */
template<typename NumericT>
viennacl::vector<NumericT> solve(viennacl::ell_matrix<NumericT> const & A,
                                 viennacl::vector_slice<NumericT> const & rhs,
                                 cg_tag const & tag,
                                 viennacl::linalg::no_precond)
{
  return detail::pipelined_solve(A, rhs, tag, viennacl::linalg::no_precond());
}



// sliced_ell_matrix

/** @brief Overload for the pipelined BiCGStab implementation for the ViennaCL sparse matrix types */
template<typename NumericT>
viennacl::vector<NumericT> solve(viennacl::sliced_ell_matrix<NumericT> const & A,
                                 viennacl::vector_base<NumericT> const & rhs,
                                 cg_tag const & tag,
                                 viennacl::linalg::no_precond)
{
  return detail::pipelined_solve(A, rhs, tag, viennacl::linalg::no_precond());
}

/** @brief Overload for the pipelined BiCGStab implementation for the ViennaCL sparse matrix types */
template<typename NumericT>
viennacl::vector<NumericT> solve(viennacl::sliced_ell_matrix<NumericT> const & A,
                                 viennacl::vector<NumericT> const & rhs,
                                 cg_tag const & tag,
                                 viennacl::linalg::no_precond)
{
  return detail::pipelined_solve(A, rhs, tag, viennacl::linalg::no_precond());
}

/** @brief Overload for the pipelined BiCGStab implementation for the ViennaCL sparse matrix types */
template<typename NumericT>
viennacl::vector<NumericT> solve(viennacl::sliced_ell_matrix<NumericT> const & A,
                                 viennacl::vector_range<NumericT> const & rhs,
                                 cg_tag const & tag,
                                 viennacl::linalg::no_precond)
{
  return detail::pipelined_solve(A, rhs, tag, viennacl::linalg::no_precond());
}

/** @brief Overload for the pipelined BiCGStab implementation for the ViennaCL sparse matrix types */
template<typename NumericT>
viennacl::vector<NumericT> solve(viennacl::sliced_ell_matrix<NumericT> const & A,
                                 viennacl::vector_slice<NumericT> const & rhs,
                                 cg_tag const & tag,
                                 viennacl::linalg::no_precond)
{
  return detail::pipelined_solve(A, rhs, tag, viennacl::linalg::no_precond());
}


// hyb_matrix

/** @brief Overload for the pipelined BiCGStab implementation for the ViennaCL sparse matrix types */
template<typename NumericT>
viennacl::vector<NumericT> solve(viennacl::hyb_matrix<NumericT> const & A,
                                 viennacl::vector_base<NumericT> const & rhs,
                                 cg_tag const & tag,
                                 viennacl::linalg::no_precond)
{
  return detail::pipelined_solve(A, rhs, tag, viennacl::linalg::no_precond());
}

/** @brief Overload for the pipelined BiCGStab implementation for the ViennaCL sparse matrix types */
template<typename NumericT>
viennacl::vector<NumericT> solve(viennacl::hyb_matrix<NumericT> const & A,
                                 viennacl::vector<NumericT> const & rhs,
                                 cg_tag const & tag,
                                 viennacl::linalg::no_precond)
{
  return detail::pipelined_solve(A, rhs, tag, viennacl::linalg::no_precond());
}

/** @brief Overload for the pipelined BiCGStab implementation for the ViennaCL sparse matrix types */
template<typename NumericT>
viennacl::vector<NumericT> solve(viennacl::hyb_matrix<NumericT> const & A,
                                 viennacl::vector_range<NumericT> const & rhs,
                                 cg_tag const & tag,
                                 viennacl::linalg::no_precond)
{
  return detail::pipelined_solve(A, rhs, tag, viennacl::linalg::no_precond());
}

/** @brief Overload for the pipelined BiCGStab implementation for the ViennaCL sparse matrix types */
template<typename NumericT>
viennacl::vector<NumericT> solve(viennacl::hyb_matrix<NumericT> const & A,
                                 viennacl::vector_slice<NumericT> const & rhs,
                                 cg_tag const & tag,
                                 viennacl::linalg::no_precond)
{
  return detail::pipelined_solve(A, rhs, tag, viennacl::linalg::no_precond());
}








/** @brief Implementation of the preconditioned conjugate gradient solver, generic implementation for non-ViennaCL types.
*
* Following Algorithm 9.1 in "Iterative Methods for Sparse Linear Systems" by Y. Saad
*
* @param matrix     The system matrix
* @param rhs        The load vector
* @param tag        Solver configuration tag
* @param precond    A preconditioner. Precondition operation is done via member function apply()
* @return The result vector
*/
template<typename MatrixT, typename VectorT, typename PreconditionerT>
VectorT solve(MatrixT const & matrix, VectorT const & rhs, cg_tag const & tag, PreconditionerT const & precond)
{
  typedef typename viennacl::result_of::value_type<VectorT>::type           NumericType;
  typedef typename viennacl::result_of::cpu_value_type<NumericType>::type   CPU_NumericType;

  VectorT result = rhs;
  viennacl::traits::clear(result);

  VectorT residual = rhs;
  VectorT tmp = rhs;
  detail::z_handler<VectorT, PreconditionerT> zhandler(residual);
  VectorT & z = zhandler.get();

  precond.apply(z);
  VectorT p = z;

  CPU_NumericType ip_rr = viennacl::linalg::inner_prod(residual, z);
  CPU_NumericType alpha;
  CPU_NumericType new_ip_rr = 0;
  CPU_NumericType beta;
  CPU_NumericType norm_rhs_squared = ip_rr;
  CPU_NumericType new_ipp_rr_over_norm_rhs;

  if (norm_rhs_squared <= 0) //solution is zero if RHS norm is zero
    return result;

  for (unsigned int i = 0; i < tag.max_iterations(); ++i)
  {
    tag.iters(i+1);
    tmp = viennacl::linalg::prod(matrix, p);

    alpha = ip_rr / viennacl::linalg::inner_prod(tmp, p);

    result += alpha * p;
    residual -= alpha * tmp;
    z = residual;
    precond.apply(z);

    if (static_cast<VectorT*>(&residual)==static_cast<VectorT*>(&z))
      new_ip_rr = std::pow(viennacl::linalg::norm_2(residual),2);
    else
      new_ip_rr = viennacl::linalg::inner_prod(residual, z);

    new_ipp_rr_over_norm_rhs = new_ip_rr / norm_rhs_squared;
    if (std::fabs(new_ipp_rr_over_norm_rhs) < tag.tolerance() *  tag.tolerance())    //squared norms involved here
      break;

    beta = new_ip_rr / ip_rr;
    ip_rr = new_ip_rr;

    p = z + beta*p;
  }

  //store last error estimate:
  tag.error(std::sqrt(std::fabs(new_ip_rr / norm_rhs_squared)));

  return result;
}

template<typename MatrixT, typename VectorT>
VectorT solve(MatrixT const & matrix, VectorT const & rhs, cg_tag const & tag)
{
  return solve(matrix, rhs, tag, viennacl::linalg::no_precond());
}

}
}

#endif
