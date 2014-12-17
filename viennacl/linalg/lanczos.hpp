#ifndef VIENNACL_LINALG_LANCZOS_HPP_
#define VIENNACL_LINALG_LANCZOS_HPP_

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

/** @file viennacl/linalg/lanczos.hpp
*   @brief Generic interface for the Lanczos algorithm.
*
*   Contributed by Guenther Mader and Astrid Rupp.
*/

#include <cmath>
#include <vector>
#include "viennacl/vector.hpp"
#include "viennacl/compressed_matrix.hpp"
#include "viennacl/linalg/prod.hpp"
#include "viennacl/linalg/inner_prod.hpp"
#include "viennacl/linalg/norm_2.hpp"
#include "viennacl/io/matrix_market.hpp"
#include "viennacl/linalg/bisect.hpp"
#include <boost/random.hpp>
#include <boost/random/mersenne_twister.hpp>
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/matrix_proxy.hpp>
#include <boost/numeric/ublas/matrix_expression.hpp>
#include <boost/numeric/ublas/matrix_sparse.hpp>
#include <boost/numeric/ublas/vector.hpp>
#include <boost/numeric/ublas/operation.hpp>
#include <boost/numeric/ublas/vector_expression.hpp>
#include <boost/numeric/ublas/io.hpp>

namespace viennacl
{
namespace linalg
{

/** @brief A tag for the lanczos algorithm.
*/
class lanczos_tag
{
public:

  enum
  {
    partial_reorthogonalization = 0,
    full_reorthogonalization,
    no_reorthogonalization
  };

  /** @brief The constructor
  *
  * @param factor                 Exponent of epsilon - tolerance for batches of Reorthogonalization
  * @param numeig                 Number of eigenvalues to be returned
  * @param met                    Method for Lanczos-Algorithm: 0 for partial Reorthogonalization, 1 for full Reorthogonalization and 2 for Lanczos without Reorthogonalization
  * @param krylov                 Maximum krylov-space size
  */

  lanczos_tag(double factor = 0.75,
              vcl_size_t numeig = 10,
              int met = 0,
              vcl_size_t krylov = 100) : factor_(factor), num_eigenvalues_(numeig), method_(met), krylov_size_(krylov) {}

  /** @brief Sets the number of eigenvalues */
  void num_eigenvalues(vcl_size_t numeig){ num_eigenvalues_ = numeig; }

    /** @brief Returns the number of eigenvalues */
  vcl_size_t num_eigenvalues() const { return num_eigenvalues_; }

    /** @brief Sets the exponent of epsilon */
  void factor(double fct) { factor_ = fct; }

  /** @brief Returns the exponent */
  double factor() const { return factor_; }

  /** @brief Sets the size of the kylov space */
  void krylov_size(vcl_size_t max) { krylov_size_ = max; }

  /** @brief Returns the size of the kylov space */
  vcl_size_t  krylov_size() const { return krylov_size_; }

  /** @brief Sets the reorthogonalization method */
  void method(int met){ method_ = met; }

  /** @brief Returns the reorthogonalization method */
  int method() const { return method_; }


private:
  double factor_;
  vcl_size_t num_eigenvalues_;
  int method_; // see enum defined above for possible values
  vcl_size_t krylov_size_;
};


namespace detail
{
  /**
  *   @brief Implementation of the Lanczos PRO algorithm
  *
  *   @param A            The system matrix
  *   @param r            Random start vector
  *   @param size         Size of krylov-space
  *   @param tag          Lanczos_tag with several options for the algorithm
  *   @return             Returns the eigenvalues (number of eigenvalues equals size of krylov-space)
  */

  template< typename MatrixT, typename VectorT >
  std::vector<
          typename viennacl::result_of::cpu_value_type<typename MatrixT::value_type>::type
          >
  lanczosPRO (MatrixT const& A, VectorT & r, vcl_size_t size, lanczos_tag const & tag)
  {
    typedef typename viennacl::result_of::value_type<MatrixT>::type        ScalarType;
    typedef typename viennacl::result_of::cpu_value_type<ScalarType>::type    CPU_ScalarType;


    // generation of some random numbers, used for lanczos PRO algorithm
    boost::mt11213b mt;
    boost::normal_distribution<CPU_ScalarType> N(0, 1);
    boost::bernoulli_distribution<CPU_ScalarType> B(0.5);
    boost::triangle_distribution<CPU_ScalarType> T(-1, 0, 1);

    boost::variate_generator<boost::mt11213b&, boost::normal_distribution<CPU_ScalarType> >     get_N(mt, N);
    boost::variate_generator<boost::mt11213b&, boost::bernoulli_distribution<CPU_ScalarType> >  get_B(mt, B);
    boost::variate_generator<boost::mt11213b&, boost::triangle_distribution<CPU_ScalarType> >   get_T(mt, T);


    long i, k, retry, reorths;
    std::vector<long> l_bound(size/2), u_bound(size/2);
    bool second_step;
    CPU_ScalarType squ_eps, eta, temp, eps, retry_th;
    vcl_size_t n = r.size();
    std::vector< std::vector<CPU_ScalarType> > w(2, std::vector<CPU_ScalarType>(size));
    CPU_ScalarType cpu_beta;

    boost::numeric::ublas::vector<CPU_ScalarType> s(n);

    VectorT t(n);
    CPU_ScalarType inner_rt;
    ScalarType vcl_beta;
    ScalarType vcl_alpha;
    std::vector<CPU_ScalarType> alphas, betas;
    boost::numeric::ublas::matrix<CPU_ScalarType> Q(n, size);

    second_step = false;
    eps = std::numeric_limits<CPU_ScalarType>::epsilon();
    squ_eps = std::sqrt(eps);
    retry_th = 1e-2;
    eta = std::exp(std::log(eps) * tag.factor());
    reorths = 0;
    retry = 0;

    vcl_beta = viennacl::linalg::norm_2(r);

    r /= vcl_beta;

    detail::copy_vec_to_vec(r,s);
    boost::numeric::ublas::column(Q, 0) = s;

    VectorT u = viennacl::linalg::prod(A, r);
    vcl_alpha = viennacl::linalg::inner_prod(u, r);
    alphas.push_back(vcl_alpha);
    w[0][0] = 1;
    betas.push_back(vcl_beta);

    long batches = 0;
    for (i = 1;i < static_cast<long>(size); i++)
    {
      r = u - vcl_alpha * r;
      vcl_beta = viennacl::linalg::norm_2(r);

      betas.push_back(vcl_beta);
      r = r / vcl_beta;

      vcl_size_t index = vcl_size_t(i % 2);
      w[index][vcl_size_t(i)] = 1;
      k = (i + 1) % 2;
      w[index][0] = (betas[1] * w[vcl_size_t(k)][1] + (alphas[0] - vcl_alpha) * w[vcl_size_t(k)][0] - betas[vcl_size_t(i) - 1] * w[index][0]) / vcl_beta + eps * 0.3 * get_N() * (betas[1] + vcl_beta);

      for (vcl_size_t j = 1; j < vcl_size_t(i - 1); j++)
      {
              w[index][j] = (betas[j + 1] * w[vcl_size_t(k)][j + 1] + (alphas[j] - vcl_alpha) * w[vcl_size_t(k)][j] + betas[j] * w[vcl_size_t(k)][j - 1] - betas[vcl_size_t(i) - 1] * w[index][j]) / vcl_beta + eps * 0.3 * get_N() * (betas[j + 1] + vcl_beta);
      }
      w[index][vcl_size_t(i) - 1] = 0.6 * eps * CPU_ScalarType(n) * get_N() * betas[1] / vcl_beta;

      if (second_step)
      {
        for (vcl_size_t j = 0; j < vcl_size_t(batches); j++)
        {
          l_bound[vcl_size_t(j)]++;
          u_bound[vcl_size_t(j)]--;

          for (k = l_bound[j];k < u_bound[j];k++)
          {
            detail::copy_vec_to_vec(boost::numeric::ublas::column(Q, vcl_size_t(k)), t);
            inner_rt = viennacl::linalg::inner_prod(r,t);
            r = r - inner_rt * t;
            w[index][vcl_size_t(k)] = 1.5 * eps * get_N();
            reorths++;
          }
        }
        temp = viennacl::linalg::norm_2(r);
        r = r / temp;
        vcl_beta = vcl_beta * temp;
        second_step = false;
      }
      batches = 0;

      for (vcl_size_t j = 0; j < vcl_size_t(i); j++)
      {
        if (std::fabs(w[index][j]) >= squ_eps)
        {
          detail::copy_vec_to_vec(boost::numeric::ublas::column(Q, j), t);
          inner_rt = viennacl::linalg::inner_prod(r,t);
          r = r - inner_rt * t;
          w[index][j] = 1.5 * eps * get_N();
          k = long(j) - 1;
          reorths++;
          while (k >= 0 && std::fabs(w[index][vcl_size_t(k)]) > eta)
          {
            detail::copy_vec_to_vec(boost::numeric::ublas::column(Q, vcl_size_t(k)), t);
            inner_rt = viennacl::linalg::inner_prod(r,t);
            r = r - inner_rt * t;
            w[index][vcl_size_t(k)] = 1.5 * eps * get_N();
            k--;
            reorths++;
          }
          l_bound[vcl_size_t(batches)] = k + 1;
          k = long(j) + 1;

          while (k < i && std::fabs(w[index][vcl_size_t(k)]) > eta)
          {
            detail::copy_vec_to_vec(boost::numeric::ublas::column(Q, vcl_size_t(k)), t);
            inner_rt = viennacl::linalg::inner_prod(r,t);
            r = r - inner_rt * t;
            w[index][vcl_size_t(k)] = 1.5 * eps * get_N();
            k++;
            reorths++;
          }
          u_bound[vcl_size_t(batches)] = k - 1;
          batches++;
          j = vcl_size_t(k);
        }
      }

      if (batches > 0)
      {
        temp = viennacl::linalg::norm_2(r);
        r = r / temp;
        vcl_beta = vcl_beta * temp;
        second_step = true;

        while (temp < retry_th)
        {
          for (vcl_size_t j = 0; j < vcl_size_t(i); j++)
          {
            detail::copy_vec_to_vec(boost::numeric::ublas::column(Q, vcl_size_t(k)), t);
            inner_rt = viennacl::linalg::inner_prod(r,t);
            r = r - inner_rt * t;
            reorths++;
          }
          retry++;
          temp = viennacl::linalg::norm_2(r);
          r = r / temp;
          vcl_beta = vcl_beta * temp;
        }
      }

      detail::copy_vec_to_vec(r,s);
      boost::numeric::ublas::column(Q, vcl_size_t(i)) = s;

      cpu_beta = vcl_beta;
      s = - cpu_beta * boost::numeric::ublas::column(Q, vcl_size_t(i - 1));
      detail::copy_vec_to_vec(s, u);
      u += viennacl::linalg::prod(A, r);
      vcl_alpha = viennacl::linalg::inner_prod(u, r);
      alphas.push_back(vcl_alpha);
    }

    return bisect(alphas, betas);
  }

  template<typename NumericT>
  void test_eigenvector(std::vector<NumericT> const & alphas, std::vector<NumericT> const & betas,
                        NumericT eigenvalue, std::vector<NumericT> const & eigenvector)
  {
    std::vector<NumericT> result_vector(eigenvector);
    result_vector[0] = alphas[0] * eigenvector[0] + betas[1] * eigenvector[1];
    for (vcl_size_t i=1; i<alphas.size() - 1; ++i)
      result_vector[i] = betas[i] * eigenvector[i-1] + alphas[i] * eigenvector[i] + betas[i+1] * eigenvector[i+1];
    result_vector[alphas.size() - 1] = betas[alphas.size() - 1] * eigenvector[alphas.size() - 2] + alphas[alphas.size() - 1] * eigenvector[alphas.size() - 1];

    // check:
    for (vcl_size_t i=0; i<eigenvector.size(); ++i)
    {
      NumericT rel_error = std::fabs(result_vector[i] - eigenvalue * eigenvector[i]) / std::max(std::fabs(eigenvalue * eigenvector[i]), std::fabs(result_vector[i]));
      if (rel_error > 1e-3)
        std::cerr << "FAILED at entry " << i << " (rel error: " << rel_error << "): A*v = " << result_vector[i] << ", but eigenvalue * eigenvector is " << eigenvalue << " * " << eigenvector[i] << " = " << eigenvalue * eigenvector[i] << std::endl;
      //else
      //  std::cout << "PASSED: " << rel_error << std::endl;
    }
  }

  /** @brief Inverse iteration for finding an eigenvector for an eigenvalue.
   *
   *  beta[0] to be ignored for consistency.
   */
  template<typename NumericT>
  void inverse_iteration(std::vector<NumericT> const & alphas, std::vector<NumericT> const & betas,
                         NumericT & eigenvalue, std::vector<NumericT> & eigenvector)
  {
    std::vector<NumericT> alpha_sweeped = alphas;
    for (vcl_size_t i=0; i<alpha_sweeped.size(); ++i)
      alpha_sweeped[i] -= eigenvalue;
    for (vcl_size_t row=1; row < alpha_sweeped.size(); ++row)
      alpha_sweeped[row] -= betas[row] * betas[row] / alpha_sweeped[row-1];

    // starting guess: ignore last equation
    eigenvector[alphas.size() - 1] = 1.0;

    for (vcl_size_t iter=0; iter<1; ++iter)
    {
      // solve first n-1 equations (A - \lambda I) y = -beta[n]
      eigenvector[alphas.size() - 1] /= alpha_sweeped[alphas.size() - 1];
      for (vcl_size_t row2=1; row2 < alphas.size(); ++row2)
      {
        vcl_size_t row = alphas.size() - row2 - 1;
        eigenvector[row] -= eigenvector[row+1] * betas[row+1];
        eigenvector[row] /= alpha_sweeped[row];
      }

      // normalize eigenvector:
      NumericT norm_vector = 0;
      for (vcl_size_t i=0; i<eigenvector.size(); ++i)
        norm_vector += eigenvector[i] * eigenvector[i];
      norm_vector = std::sqrt(norm_vector);
      for (vcl_size_t i=0; i<eigenvector.size(); ++i)
        eigenvector[i] /= norm_vector;
    }

    //eigenvalue = (alphas[0] * eigenvector[0] + betas[1] * eigenvector[1]) / eigenvector[0];
  }

  /**
  *   @brief Implementation of the Lanczos FRO algorithm
  *
  *   @param A            The system matrix
  *   @param r            Random start vector
  *   @param eigenvectors_A  A dense matrix in which the eigenvectors of A will be stored. Both row- and column-major matrices are supported.
  *   @param krylov_dim   Size of krylov-space
  *   @return             Returns the eigenvalues (number of eigenvalues equals size of krylov-space)
  */
  template< typename MatrixT, typename DenseMatrixT, typename NumericT>
  std::vector<NumericT>
  lanczos(MatrixT const& A, vector_base<NumericT> & r, DenseMatrixT & eigenvectors_A, vcl_size_t krylov_dim, lanczos_tag const & tag)
  {
    std::vector<NumericT> alphas, betas;
    viennacl::vector<NumericT> Aq(r.size());
    viennacl::matrix<NumericT, viennacl::column_major> Q(r.size(), krylov_dim + 1);  // Krylov basis (each Krylov vector is one column)

    NumericT norm_r = norm_2(r);
    NumericT beta = norm_r;
    r /= norm_r;

    // first Krylov vector:
    viennacl::vector_base<NumericT> q0(Q.handle(), Q.size1(), 0, 1);
    q0 = r;

    //
    // Step 1: Run Lanczos' method to obtain tridiagonal matrix
    //
    for (vcl_size_t i = 0; i < krylov_dim; i++)
    {
      betas.push_back(beta);
      // last available vector from Krylov basis:
      viennacl::vector_base<NumericT> q_i(Q.handle(), Q.size1(), i * Q.internal_size1(), 1);

      // Lanczos algorithm:
      // - Compute A * q:
      Aq = viennacl::linalg::prod(A, q_i);

      // - Form Aq <- Aq - <Aq, q_i> * q_i - beta * q_{i-1}, where beta is ||q_i|| before normalization in previous iteration
      NumericT alpha = viennacl::linalg::inner_prod(Aq, q_i);
      Aq -= alpha * q_i;

      if (i > 0)
      {
        viennacl::vector_base<NumericT> q_iminus1(Q.handle(), Q.size1(), (i-1) * Q.internal_size1(), 1);
        Aq -= beta * q_iminus1;

        // Extra measures for improved numerical stability?
        if (tag.method() == lanczos_tag::full_reorthogonalization)
        {
          // Gram-Schmidt (re-)orthogonalization:
          // TODO: Reuse fast (pipelined) routines from GMRES or GEMV
          for (vcl_size_t j = 0; j < i; j++)
          {
            viennacl::vector_base<NumericT> q_j(Q.handle(), Q.size1(), j * Q.internal_size1(), 1);
            NumericT inner_rq = viennacl::linalg::inner_prod(Aq, q_j);
            Aq -= inner_rq * q_j;
          }
        }
      }

      // normalize Aq and add to Krylov basis at column i+1 in Q:
      beta = viennacl::linalg::norm_2(Aq);
      viennacl::vector_base<NumericT> q_iplus1(Q.handle(), Q.size1(), (i+1) * Q.internal_size1(), 1);
      q_iplus1 = Aq / beta;

      alphas.push_back(alpha);
    }

    //
    // Step 2: Compute eigenvalues of tridiagonal matrix obtained during Lanczos iterations:
    //
    std::vector<NumericT> eigenvalues = bisect(alphas, betas);

    //
    // Step 3: Compute eigenvectors via inverse iteration. Does not update eigenvalues, so only approximate by nature.
    //
    bool compute_eigenvectors = true;
    if (compute_eigenvectors)
    {
      std::vector<NumericT> eigenvector_tridiag(alphas.size());
      for (std::size_t i=0; i < tag.num_eigenvalues(); ++i)
      {
        // compute eigenvector of tridiagonal matrix via inverse:
        inverse_iteration(alphas, betas, eigenvalues[eigenvalues.size() - i - 1], eigenvector_tridiag);

        // eigenvector w of full matrix A. Given as w = Q * u, where u is the eigenvector of the tridiagonal matrix
        viennacl::vector<NumericT> eigenvector_u(eigenvector_tridiag.size());
        viennacl::copy(eigenvector_tridiag, eigenvector_u);

        viennacl::vector_base<NumericT> eigenvector_A(eigenvectors_A.handle(),
                                                      eigenvectors_A.size1(),
                                                      eigenvectors_A.row_major() ? i : i * eigenvectors_A.internal_size1(),
                                                      eigenvectors_A.row_major() ? eigenvectors_A.internal_size2() : 1);
        eigenvector_A = viennacl::linalg::prod(project(Q,
                                                       range(0, Q.size1()),
                                                       range(0, eigenvector_u.size())),
                                               eigenvector_u);
      }
    }

    return eigenvalues;
  }

} // end namespace detail

/**
*   @brief Implementation of the calculation of eigenvalues using lanczos (with and without reorthogonalization).
*
*   Implementation of Lanczos with partial reorthogonalization is implemented separately.
*
*   @param matrix        The system matrix
*   @param tag           Tag with several options for the lanczos algorithm
*   @return              Returns the n largest eigenvalues (n defined in the lanczos_tag)
*/
template<typename MatrixT, typename DenseMatrixT>
std::vector< typename viennacl::result_of::cpu_value_type<typename MatrixT::value_type>::type >
eig(MatrixT const & matrix, DenseMatrixT & eigenvalues_A, lanczos_tag const & tag)
{
  typedef typename viennacl::result_of::value_type<MatrixT>::type           NumericType;
  typedef typename viennacl::result_of::cpu_value_type<NumericType>::type   CPU_NumericType;
  typedef typename viennacl::result_of::vector_for_matrix<MatrixT>::type    VectorT;

  boost::mt11213b mt;
  boost::normal_distribution<CPU_NumericType>    N(0, 1);
  boost::bernoulli_distribution<CPU_NumericType> B(0.5);
  boost::triangle_distribution<CPU_NumericType>  T(-1, 0, 1);

  boost::variate_generator<boost::mt11213b&, boost::normal_distribution<CPU_NumericType> >     get_N(mt, N);
  boost::variate_generator<boost::mt11213b&, boost::bernoulli_distribution<CPU_NumericType> >  get_B(mt, B);
  boost::variate_generator<boost::mt11213b&, boost::triangle_distribution<CPU_NumericType> >   get_T(mt, T);

  std::vector<CPU_NumericType> eigenvalues;
  vcl_size_t matrix_size = matrix.size1();
  VectorT r(matrix_size);
  std::vector<CPU_NumericType> s(matrix_size);

  for (vcl_size_t i=0; i<s.size(); ++i)
    s[i] = 3.0 * get_B() + get_T() - 1.5;

  detail::copy_vec_to_vec(s,r);

  vcl_size_t size_krylov = (matrix_size < tag.krylov_size()) ? matrix_size
                                                              : tag.krylov_size();

  switch (tag.method())
  {
  case lanczos_tag::partial_reorthogonalization:
    eigenvalues = detail::lanczosPRO(matrix, r, size_krylov, tag);
    break;
  case lanczos_tag::full_reorthogonalization:
  case lanczos_tag::no_reorthogonalization:
    eigenvalues = detail::lanczos(matrix, r, eigenvalues_A, size_krylov, tag);
    break;
  }

  std::vector<CPU_NumericType> largest_eigenvalues;

  for (vcl_size_t i = 1; i<=tag.num_eigenvalues(); i++)
    largest_eigenvalues.push_back(eigenvalues[size_krylov-i]);


  return largest_eigenvalues;
}


/**
*   @brief Implementation of the calculation of eigenvalues using lanczos (with and without reorthogonalization).
*
*   Implementation of Lanczos with partial reorthogonalization is implemented separately.
*
*   @param matrix        The system matrix
*   @param tag           Tag with several options for the lanczos algorithm
*   @return              Returns the n largest eigenvalues (n defined in the lanczos_tag)
*/
template<typename MatrixT>
std::vector< typename viennacl::result_of::cpu_value_type<typename MatrixT::value_type>::type >
eig(MatrixT const & matrix, lanczos_tag const & tag)
{
  typedef typename viennacl::result_of::cpu_value_type<typename MatrixT::value_type>::type  NumericType;

  viennacl::matrix<NumericType> eigenvectors(matrix.size1(), tag.num_eigenvalues());
  return eig(matrix, eigenvectors, tag);
}

} // end namespace linalg
} // end namespace viennacl
#endif
