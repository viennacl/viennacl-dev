#ifndef VIENNACL_LINALG_AMG_HPP_
#define VIENNACL_LINALG_AMG_HPP_

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

/** @file viennacl/linalg/amg.hpp
    @brief Main include file for algebraic multigrid (AMG) preconditioners.  Experimental.

    Implementation contributed by Markus Wagner
*/

#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/lu.hpp>
#include <boost/numeric/ublas/operation.hpp>
#include <boost/numeric/ublas/vector_proxy.hpp>
#include <boost/numeric/ublas/matrix_proxy.hpp>
#include <boost/numeric/ublas/vector.hpp>
#include <boost/numeric/ublas/triangular.hpp>
#include <vector>
#include <cmath>
#include "viennacl/forwards.h"
#include "viennacl/tools/tools.hpp"
#include "viennacl/linalg/prod.hpp"
#include "viennacl/linalg/direct_solve.hpp"

#include "viennacl/linalg/detail/amg/amg_base.hpp"
#include "viennacl/linalg/detail/amg/amg_coarse.hpp"
#include "viennacl/linalg/detail/amg/amg_interpol.hpp"

#include <map>

#ifdef VIENNACL_WITH_OPENMP
 #include <omp.h>
#endif

#include "viennacl/linalg/detail/amg/amg_debug.hpp"

#define VIENNACL_AMG_COARSE_LIMIT 50
#define VIENNACL_AMG_MAX_LEVELS 100

namespace viennacl
{
namespace linalg
{

typedef detail::amg::amg_tag          amg_tag;


/** @brief Setup AMG preconditioner
*
* @param A            Operator matrices on all levels
* @param P            Prolongation/Interpolation operators on all levels
* @param pointvector  Vector of points on all levels
* @param tag          AMG preconditioner tag
*/
template<typename InternalT1, typename InternalT2>
void amg_setup(InternalT1 & A, InternalT1 & P, InternalT2 & pointvector, amg_tag & tag)
{
  typedef typename InternalT2::value_type      PointVectorType;

  unsigned int i, iterations, c_points, f_points;
  detail::amg::amg_slicing<InternalT1,InternalT2> slicing;

  // Set number of iterations. If automatic coarse grid construction is chosen (0), then set a maximum size and stop during the process.
  iterations = tag.get_coarselevels();
  if (iterations == 0)
    iterations = VIENNACL_AMG_MAX_LEVELS;

  // For parallel coarsenings build data structures (number of threads set automatically).
  if (tag.get_coarse() == VIENNACL_AMG_COARSE_RS0 || tag.get_coarse() == VIENNACL_AMG_COARSE_RS3)
    slicing.init(iterations);

  for (i=0; i<iterations; ++i)
  {
    // Initialize Pointvector on level i and construct points.
    pointvector[i] = PointVectorType(static_cast<unsigned int>(A[i].size1()));
    pointvector[i].init_points();

    // Construct C and F points on coarse level (i is fine level, i+1 coarse level).
    detail::amg::amg_coarse (i, A, pointvector, slicing, tag);

    // Calculate number of C and F points on level i.
    c_points = pointvector[i].get_cpoints();
    f_points = pointvector[i].get_fpoints();

    #if defined (VIENNACL_AMG_DEBUG) //or defined(VIENNACL_AMG_DEBUGBENCH)
    std::cout << "Level " << i << ": ";
    std::cout << "No of C points = " << c_points << ", ";
    std::cout << "No of F points = " << f_points << std::endl;
    #endif

    // Stop routine when the maximal coarse level is found (no C or F point). Coarsest level is level i.
    if (c_points == 0 || f_points == 0)
      break;

    // Construct interpolation matrix for level i.
    detail::amg::amg_interpol (i, A, P, pointvector, tag);

    // Compute coarse grid operator (A[i+1] = R * A[i] * P) with R = trans(P).
    detail::amg::amg_galerkin_prod(A[i], P[i], A[i+1]);

    // Test triple matrix product. Very slow for large matrix sizes (ublas).
    // test_triplematprod(A[i],P[i],A[i+1]);

    pointvector[i].delete_points();

    #ifdef VIENNACL_AMG_DEBUG
    std::cout << "Coarse Grid Operator Matrix:" << std::endl;
    printmatrix (A[i+1]);
    #endif

    // If Limit of coarse points is reached then stop. Coarsest level is level i+1.
    if (tag.get_coarselevels() == 0 && c_points <= VIENNACL_AMG_COARSE_LIMIT)
    {
      tag.set_coarselevels(i+1);
      return;
    }
  }
  tag.set_coarselevels(i);
}

/** @brief Initialize AMG preconditioner
*
* @param mat          System matrix
* @param A            Operator matrices on all levels
* @param P            Prolongation/Interpolation operators on all levels
* @param pointvector  Vector of points on all levels
* @param tag          AMG preconditioner tag
*/
template<typename MatrixT, typename InternalT1, typename InternalT2>
void amg_init(MatrixT const & mat, InternalT1 & A, InternalT1 & P, InternalT2 & pointvector, amg_tag & tag)
{
  //typedef typename MatrixType::value_type ScalarType;
  typedef typename InternalT1::value_type SparseMatrixType;

  if (tag.get_coarselevels() > 0)
  {
    A.resize(tag.get_coarselevels()+1);
    P.resize(tag.get_coarselevels());
    pointvector.resize(tag.get_coarselevels());
  }
  else
  {
    A.resize(VIENNACL_AMG_MAX_LEVELS+1);
    P.resize(VIENNACL_AMG_MAX_LEVELS);
    pointvector.resize(VIENNACL_AMG_MAX_LEVELS);
  }

  // Insert operator matrix as operator for finest level.
  SparseMatrixType A0(mat);
  A.insert_element(0, A0);
}

/** @brief Save operators after setup phase for CPU computation.
*
* @param A      Operator matrices on all levels on the CPU
* @param P      Prolongation/Interpolation operators on all levels on the CPU
* @param R      Restriction operators on all levels on the CPU
* @param A_setup    Operators matrices on all levels from setup phase
* @param P_setup    Prolongation/Interpolation operators on all levels from setup phase
* @param tag    AMG preconditioner tag
*/
template<typename InternalT1, typename InternalT2>
void amg_transform_cpu(InternalT1 & A, InternalT1 & P, InternalT1 & R, InternalT2 & A_setup, InternalT2 & P_setup, amg_tag & tag)
{
  //typedef typename InternalType1::value_type MatrixType;

  // Resize internal data structures to actual size.
  A.resize(tag.get_coarselevels()+1);
  P.resize(tag.get_coarselevels());
  R.resize(tag.get_coarselevels());

  // Transform into matrix type.
  for (unsigned int i=0; i<tag.get_coarselevels()+1; ++i)
  {
    A[i].resize(A_setup[i].size1(),A_setup[i].size2(),false);
    A[i] = A_setup[i];
  }
  for (unsigned int i=0; i<tag.get_coarselevels(); ++i)
  {
    P[i].resize(P_setup[i].size1(),P_setup[i].size2(),false);
    P[i] = P_setup[i];
  }
  for (unsigned int i=0; i<tag.get_coarselevels(); ++i)
  {
    R[i].resize(P_setup[i].size2(),P_setup[i].size1(),false);
    P_setup[i].set_trans(true);
    R[i] = P_setup[i];
    P_setup[i].set_trans(false);
  }
}

/** @brief Save operators after setup phase for GPU computation.
*
* @param A          Operator matrices on all levels on the GPU
* @param P          Prolongation/Interpolation operators on all levels on the GPU
* @param R          Restriction operators on all levels on the GPU
* @param A_setup    Operators matrices on all levels from setup phase
* @param P_setup    Prolongation/Interpolation operators on all levels from setup phase
* @param tag        AMG preconditioner tag
* @param ctx        Optional context in which the auxiliary objects are created (one out of multiple OpenCL contexts, CUDA, host)
*/
template<typename InternalT1, typename InternalT2>
void amg_transform_gpu(InternalT1 & A, InternalT1 & P, InternalT1 & R, InternalT2 & A_setup, InternalT2 & P_setup, amg_tag & tag, viennacl::context ctx)
{
  typedef typename InternalT2::value_type::value_type    NumericType;

  // Resize internal data structures to actual size.
  A.resize(tag.get_coarselevels()+1);
  P.resize(tag.get_coarselevels());
  R.resize(tag.get_coarselevels());

  // Copy to GPU using the internal sparse matrix structure: std::vector<std::map>.
  for (unsigned int i=0; i<tag.get_coarselevels()+1; ++i)
  {
    viennacl::switch_memory_context(A[i], ctx);
    //A[i].resize(A_setup[i].size1(),A_setup[i].size2(),false);
    viennacl::copy(*(A_setup[i].get_internal_pointer()),A[i]);
  }
  for (unsigned int i=0; i<tag.get_coarselevels(); ++i)
  {
    // find number of nonzeros in P:
    vcl_size_t nonzeros = 0;
    for (vcl_size_t j=0; j<P_setup[i].get_internal_pointer()->size(); ++j)
      nonzeros += (*P_setup[i].get_internal_pointer())[j].size();

    viennacl::switch_memory_context(P[i], ctx);
    //P[i].resize(P_setup[i].size1(),P_setup[i].size2(),false);
    viennacl::detail::copy_impl(tools::const_sparse_matrix_adapter<NumericType>(*(P_setup[i].get_internal_pointer()), P_setup[i].size1(), P_setup[i].size2()), P[i], nonzeros);
    //viennacl::copy((boost::numeric::ublas::compressed_matrix<ScalarType>)P_setup[i],P[i]);

    viennacl::switch_memory_context(R[i], ctx);
    //R[i].resize(P_setup[i].size2(),P_setup[i].size1(),false);
    P_setup[i].set_trans(true);
    viennacl::detail::copy_impl(tools::const_sparse_matrix_adapter<NumericType>(*(P_setup[i].get_internal_pointer()), P_setup[i].size1(), P_setup[i].size2()), R[i], nonzeros);
    P_setup[i].set_trans(false);
  }
}

/** @brief Setup data structures for precondition phase.
*
* @param result      Result vector on all levels
* @param rhs         RHS vector on all levels
* @param residual    Residual vector on all levels
* @param A           Operators matrices on all levels from setup phase
* @param tag         AMG preconditioner tag
*/
template<typename InternalVectorT, typename SparseMatrixT>
void amg_setup_apply(InternalVectorT & result, InternalVectorT & rhs, InternalVectorT & residual, SparseMatrixT const & A, amg_tag const & tag)
{
  typedef typename InternalVectorT::value_type VectorType;

  result.resize(tag.get_coarselevels()+1);
  rhs.resize(tag.get_coarselevels()+1);
  residual.resize(tag.get_coarselevels());

  for (unsigned int level=0; level < tag.get_coarselevels()+1; ++level)
  {
    result[level] = VectorType(A[level].size1());
    result[level].clear();
    rhs[level] = VectorType(A[level].size1());
    rhs[level].clear();
  }
  for (unsigned int level=0; level < tag.get_coarselevels(); ++level)
  {
    residual[level] = VectorType(A[level].size1());
    residual[level].clear();
  }
}


/** @brief Setup data structures for precondition phase for later use on the GPU
*
* @param result      Result vector on all levels
* @param rhs         RHS vector on all levels
* @param residual    Residual vector on all levels
* @param A           Operators matrices on all levels from setup phase
* @param tag         AMG preconditioner tag
* @param ctx         Optional context in which the auxiliary objects are created (one out of multiple OpenCL contexts, CUDA, host)
*/
template<typename InternalVectorT, typename SparseMatrixT>
void amg_setup_apply(InternalVectorT & result, InternalVectorT & rhs, InternalVectorT & residual, SparseMatrixT const & A, amg_tag const & tag, viennacl::context ctx)
{
  typedef typename InternalVectorT::value_type VectorType;

  result.resize(tag.get_coarselevels()+1);
  rhs.resize(tag.get_coarselevels()+1);
  residual.resize(tag.get_coarselevels());

  for (unsigned int level=0; level < tag.get_coarselevels()+1; ++level)
  {
    result[level] = VectorType(A[level].size1(), ctx);
      rhs[level] = VectorType(A[level].size1(), ctx);
  }
  for (unsigned int level=0; level < tag.get_coarselevels(); ++level)
  {
    residual[level] = VectorType(A[level].size1(), ctx);
  }
}


/** @brief Pre-compute LU factorization for direct solve (ublas library).
 *  @brief Speeds up precondition phase as this is computed only once overall instead of once per iteration.
*
* @param op           Operator matrix for direct solve
* @param permutation  Permutation matrix which saves the factorization result
* @param A            Operator matrix on coarsest level
*/
template<typename NumericT, typename SparseMatrixT>
void amg_lu(boost::numeric::ublas::compressed_matrix<NumericT> & op, boost::numeric::ublas::permutation_matrix<> & permutation, SparseMatrixT const & A)
{
  typedef typename SparseMatrixT::const_iterator1 ConstRowIterator;
  typedef typename SparseMatrixT::const_iterator2 ConstColIterator;

  // Copy to operator matrix. Needed
  op.resize(A.size1(),A.size2(),false);
  for (ConstRowIterator row_iter = A.begin1(); row_iter != A.end1(); ++row_iter)
    for (ConstColIterator col_iter = row_iter.begin(); col_iter != row_iter.end(); ++col_iter)
      op (col_iter.index1(), col_iter.index2()) = *col_iter;

  // Permutation matrix has to be reinitialized with actual size. Do not clear() or resize()!
  permutation = boost::numeric::ublas::permutation_matrix<> (op.size1());
  boost::numeric::ublas::lu_factorize(op, permutation);
}

/** @brief AMG preconditioner class, can be supplied to solve()-routines
*/
template<typename MatrixT>
class amg_precond
{
  typedef typename MatrixT::value_type                NumericType;
  typedef boost::numeric::ublas::vector<NumericType>  VectorType;
  typedef detail::amg::amg_sparsematrix<NumericType>  SparseMatrixType;
  typedef detail::amg::amg_pointvector                PointVectorType;

  typedef typename SparseMatrixType::const_iterator1  InternalConstRowIterator;
  typedef typename SparseMatrixType::const_iterator2  InternalConstColIterator;
  typedef typename SparseMatrixType::iterator1        InternalRowIterator;
  typedef typename SparseMatrixType::iterator2        InternalColIterator;

  boost::numeric::ublas::vector<SparseMatrixType> A_setup_;
  boost::numeric::ublas::vector<SparseMatrixType> P_setup_;
  boost::numeric::ublas::vector<MatrixT>          A_;
  boost::numeric::ublas::vector<MatrixT>          P_;
  boost::numeric::ublas::vector<MatrixT>          R_;
  boost::numeric::ublas::vector<PointVectorType>  pointvector_;

  mutable boost::numeric::ublas::compressed_matrix<NumericType> op_;
  mutable boost::numeric::ublas::permutation_matrix<>           permutation_;

  mutable boost::numeric::ublas::vector<VectorType> result_;
  mutable boost::numeric::ublas::vector<VectorType> rhs_;
  mutable boost::numeric::ublas::vector<VectorType> residual_;

  mutable bool done_init_apply_;

  amg_tag tag_;
public:

  amg_precond(): permutation_(0) {}
  /** @brief The constructor. Saves system matrix, tag and builds data structures for setup.
  *
  * @param mat  System matrix
  * @param tag  The AMG tag
  */
  amg_precond(MatrixT const & mat, amg_tag const & tag): permutation_(0)
  {
    tag_ = tag;
    // Initialize data structures.
    amg_init (mat, A_setup_, P_setup_, pointvector_, tag_);

    done_init_apply_ = false;
  }

  /** @brief Start setup phase for this class and copy data structures.
  */
  void setup()
  {
    // Start setup phase.
    amg_setup(A_setup_, P_setup_, pointvector_, tag_);
    // Transform to CPU-Matrixtype for precondition phase.
    amg_transform_cpu(A_, P_, R_, A_setup_, P_setup_, tag_);

    done_init_apply_ = false;
  }

  /** @brief Prepare data structures for preconditioning:
   *  Build data structures for precondition phase.
   *  Do LU factorization on coarsest level.
  */
  void init_apply() const
  {
    // Setup precondition phase (Data structures).
    amg_setup_apply(result_, rhs_, residual_, A_setup_, tag_);
    // Do LU factorization for direct solve.
    amg_lu(op_, permutation_, A_setup_[tag_.get_coarselevels()]);

    done_init_apply_ = true;
  }

  /** @brief Returns complexity measures.
  *
  * @param avgstencil  Average stencil sizes on all levels
  * @return            Operator complexity of AMG method
  */
  template<typename VectorT>
  NumericType calc_complexity(VectorT & avgstencil)
  {
    avgstencil = VectorT(tag_.get_coarselevels()+1);
    unsigned int nonzero=0, systemmat_nonzero=0, level_coefficients=0;

    for (unsigned int level=0; level < tag_.get_coarselevels()+1; ++level)
    {
      level_coefficients = 0;
      for (InternalRowIterator row_iter = A_setup_[level].begin1(); row_iter != A_setup_[level].end1(); ++row_iter)
      {
        for (InternalColIterator col_iter = row_iter.begin(); col_iter != row_iter.end(); ++col_iter)
        {
          if (level == 0)
            systemmat_nonzero++;
          nonzero++;
          level_coefficients++;
        }
      }
      avgstencil[level] = static_cast<NumericType>(level_coefficients)/static_cast<NumericType>(A_setup_[level].size1());
    }
    return static_cast<NumericType>(nonzero) / static_cast<NumericType>(systemmat_nonzero);
  }

  /** @brief Precondition Operation
  *
  * @param vec The vector to which preconditioning is applied to (ublas version)
  */
  template<typename VectorT>
  void apply(VectorT & vec) const
  {
    // Build data structures and do lu factorization before first iteration step.
    if (!done_init_apply_)
      init_apply();

    int level;

    // Precondition operation (Yang, p.3)
    rhs_[0] = vec;
    for (level=0; level<static_cast<int>(tag_.get_coarselevels()); level++)
    {
      result_[level].clear();

      // Apply Smoother presmooth_ times.
      smooth_jacobi (level, tag_.get_presmooth(), result_[level], rhs_[level]);

      #ifdef VIENNACL_AMG_DEBUG
      std::cout << "After presmooth:" << std::endl;
      printvector(result_[level]);
      #endif

      // Compute residual.
      residual_[level] = rhs_[level] - boost::numeric::ublas::prod(A_[level], result_[level]);

      #ifdef VIENNACL_AMG_DEBUG
      std::cout << "Residual:" << std::endl;
      printvector(residual_[level]);
      #endif

      // Restrict to coarse level. Restricted residual is RHS of coarse level.
      rhs_[level+1] = boost::numeric::ublas::prod(R_[level], residual_[level]);

      #ifdef VIENNACL_AMG_DEBUG
      std::cout << "Restricted Residual: " << std::endl;
      printvector(rhs_[level+1]);
      #endif
    }

    // On highest level use direct solve to solve equation.
    result_[level] = rhs_[level];
    boost::numeric::ublas::lu_substitute(op_, permutation_, result_[level]);

    #ifdef VIENNACL_AMG_DEBUG
    std::cout << "After direct solve: " << std::endl;
    printvector(result_[level]);
    #endif

    for (level=tag_.get_coarselevels()-1; level >= 0; level--)
    {
      #ifdef VIENNACL_AMG_DEBUG
      std::cout << "Coarse Error: " << std::endl;
      printvector(result_[level+1]);
      #endif

      // Interpolate error to fine level. Correct solution by adding error.
      result_[level] += boost::numeric::ublas::prod(P_[level], result_[level+1]);

      #ifdef VIENNACL_AMG_DEBUG
      std::cout << "Corrected Result: " << std::endl;
      printvector(result_[level]);
      #endif

      // Apply Smoother postsmooth_ times.
      smooth_jacobi(level, tag_.get_postsmooth(), result_[level], rhs_[level]);

      #ifdef VIENNACL_AMG_DEBUG
      std::cout << "After postsmooth: " << std::endl;
      printvector(result_[level]);
      #endif
    }
    vec = result_[0];
  }

  /** @brief (Weighted) Jacobi Smoother (CPU version)
  * @param level       Coarse level to which smoother is applied to
  * @param iterations  Number of smoother iterations
  * @param x           The vector smoothing is applied to
  * @param rhs_smooth  The right hand side of the equation for the smoother
  */
  template<typename VectorT>
  void smooth_jacobi(int level, int const iterations, VectorT & x, VectorT const & rhs_smooth) const
  {
    VectorT old_result(x.size());
    long index;

    for (int i=0; i<iterations; ++i)
    {
      old_result = x;
      x.clear();
#ifdef VIENNACL_WITH_OPENMP
      #pragma omp parallel for
#endif
      for (index=0; index < static_cast<long>(A_setup_[level].size1()); ++index)
      {
        InternalConstRowIterator row_iter = A_setup_[level].begin1();
        row_iter += index;
        NumericType sum  = 0;
        NumericType diag = 1;
        for (InternalConstColIterator col_iter = row_iter.begin(); col_iter != row_iter.end(); ++col_iter)
        {
          if (col_iter.index1() == col_iter.index2())
            diag = *col_iter;
          else
            sum += *col_iter * old_result[col_iter.index2()];
        }
        x[index]= static_cast<NumericType>(tag_.get_jacobiweight()) * (rhs_smooth[index] - sum) / diag + (1-static_cast<NumericType>(tag_.get_jacobiweight())) * old_result[index];
      }
    }
  }

  amg_tag & tag() { return tag_; }
};

/** @brief AMG preconditioner class, can be supplied to solve()-routines.
*
*  Specialization for compressed_matrix
*/
template<typename NumericT, unsigned int AlignmentV>
class amg_precond< compressed_matrix<NumericT, AlignmentV> >
{
  typedef viennacl::compressed_matrix<NumericT, AlignmentV> MatrixType;
  typedef viennacl::vector<NumericT>                        VectorType;
  typedef detail::amg::amg_sparsematrix<NumericT>           SparseMatrixType;
  typedef detail::amg::amg_pointvector                      PointVectorType;

  typedef typename SparseMatrixType::const_iterator1   InternalConstRowIterator;
  typedef typename SparseMatrixType::const_iterator2   InternalConstColIterator;
  typedef typename SparseMatrixType::iterator1         InternalRowIterator;
  typedef typename SparseMatrixType::iterator2         InternalColIterator;

  boost::numeric::ublas::vector<SparseMatrixType> A_setup_;
  boost::numeric::ublas::vector<SparseMatrixType> P_setup_;
  boost::numeric::ublas::vector<MatrixType>       A_;
  boost::numeric::ublas::vector<MatrixType>       P_;
  boost::numeric::ublas::vector<MatrixType>       R_;
  boost::numeric::ublas::vector<PointVectorType>  pointvector_;

  mutable boost::numeric::ublas::compressed_matrix<NumericT>  op_;
  mutable boost::numeric::ublas::permutation_matrix<>         permutation_;

  mutable boost::numeric::ublas::vector<VectorType> result_;
  mutable boost::numeric::ublas::vector<VectorType> rhs_;
  mutable boost::numeric::ublas::vector<VectorType> residual_;

  viennacl::context ctx_;

  mutable bool done_init_apply_;

  amg_tag tag_;

public:

  amg_precond(): permutation_(0) {}

  /** @brief The constructor. Builds data structures.
  *
  * @param mat  System matrix
  * @param tag  The AMG tag
  */
  amg_precond(compressed_matrix<NumericT, AlignmentV> const & mat, amg_tag const & tag): permutation_(0), ctx_(viennacl::traits::context(mat))
  {
    tag_ = tag;

    // Copy to CPU. Internal structure of sparse matrix is used for copy operation.
    std::vector<std::map<unsigned int, NumericT> > mat2 = std::vector<std::map<unsigned int, NumericT> >(mat.size1());
    viennacl::copy(mat, mat2);

    // Initialize data structures.
    amg_init (mat2, A_setup_, P_setup_, pointvector_, tag_);

    done_init_apply_ = false;
  }

  /** @brief Start setup phase for this class and copy data structures.
  */
  void setup()
  {
    // Start setup phase.
    amg_setup(A_setup_, P_setup_, pointvector_, tag_);
    // Transform to GPU-Matrixtype for precondition phase.
    amg_transform_gpu(A_, P_, R_, A_setup_, P_setup_, tag_, ctx_);

    done_init_apply_ = false;
  }

  /** @brief Prepare data structures for preconditioning:
   *  Build data structures for precondition phase.
   *  Do LU factorization on coarsest level.
  */
  void init_apply() const
  {
    // Setup precondition phase (Data structures).
    amg_setup_apply(result_, rhs_, residual_, A_setup_, tag_, ctx_);
    // Do LU factorization for direct solve.
    amg_lu(op_, permutation_, A_setup_[tag_.get_coarselevels()]);

    done_init_apply_ = true;
  }

  /** @brief Returns complexity measures
  *
  * @param avgstencil  Average stencil sizes on all levels
  * @return     Operator complexity of AMG method
  */
  template<typename VectorT>
  NumericT calc_complexity(VectorT & avgstencil)
  {
    avgstencil = VectorT(tag_.get_coarselevels()+1);
    unsigned int nonzero=0, systemmat_nonzero=0, level_coefficients=0;

    for (unsigned int level=0; level < tag_.get_coarselevels()+1; ++level)
    {
      level_coefficients = 0;
      for (InternalRowIterator row_iter = A_setup_[level].begin1(); row_iter != A_setup_[level].end1(); ++row_iter)
      {
        for (InternalColIterator col_iter = row_iter.begin(); col_iter != row_iter.end(); ++col_iter)
        {
          if (level == 0)
            systemmat_nonzero++;
          nonzero++;
          level_coefficients++;
        }
      }
      avgstencil[level] = level_coefficients/static_cast<double>(A_[level].size1());
    }
    return nonzero/static_cast<double>(systemmat_nonzero);
  }

  /** @brief Precondition Operation
  *
  * @param vec The vector to which preconditioning is applied to
  */
  template<typename VectorT>
  void apply(VectorT & vec) const
  {
    if (!done_init_apply_)
      init_apply();

    vcl_size_t level;

    // Precondition operation (Yang, p.3).
    rhs_[0] = vec;
    for (level=0; level < tag_.get_coarselevels(); level++)
    {
      result_[level].clear();

      // Apply Smoother presmooth_ times.
      smooth_jacobi(level, tag_.get_presmooth(), result_[level], rhs_[level]);

      #ifdef VIENNACL_AMG_DEBUG
      std::cout << "After presmooth: " << std::endl;
      printvector(result_[level]);
      #endif

      // Compute residual.
      //residual[level] = rhs_[level] - viennacl::linalg::prod(A_[level], result_[level]);
      residual_[level] = viennacl::linalg::prod(A_[level], result_[level]);
      residual_[level] = rhs_[level] - residual_[level];

      #ifdef VIENNACL_AMG_DEBUG
      std::cout << "Residual: " << std::endl;
      printvector(residual_[level]);
      #endif

      // Restrict to coarse level. Result is RHS of coarse level equation.
      //residual_coarse[level] = viennacl::linalg::prod(R[level],residual[level]);
      rhs_[level+1] = viennacl::linalg::prod(R_[level], residual_[level]);

      #ifdef VIENNACL_AMG_DEBUG
      std::cout << "Restricted Residual: " << std::endl;
      printvector(rhs_[level+1]);
      #endif
    }

    // On highest level use direct solve to solve equation (on the CPU)
    //TODO: Use GPU direct solve!
    result_[level] = rhs_[level];
    boost::numeric::ublas::vector<NumericT> result_cpu(result_[level].size());

    viennacl::copy(result_[level], result_cpu);
    boost::numeric::ublas::lu_substitute(op_, permutation_, result_cpu);
    viennacl::copy(result_cpu, result_[level]);

    #ifdef VIENNACL_AMG_DEBUG
    std::cout << "After direct solve: " << std::endl;
    printvector (result[level]);
    #endif

    for (int level2 = static_cast<int>(tag_.get_coarselevels()-1); level2 >= 0; level2--)
    {
      level = static_cast<vcl_size_t>(level2);

      #ifdef VIENNACL_AMG_DEBUG
      std::cout << "Coarse Error: " << std::endl;
      printvector(result[level+1]);
      #endif

      // Interpolate error to fine level and correct solution.
      result_[level] += viennacl::linalg::prod(P_[level], result_[level+1]);

      #ifdef VIENNACL_AMG_DEBUG
      std::cout << "Corrected Result: " << std::endl;
      printvector(result_[level]);
      #endif

      // Apply Smoother postsmooth_ times.
      smooth_jacobi(level, tag_.get_postsmooth(), result_[level], rhs_[level]);

      #ifdef VIENNACL_AMG_DEBUG
      std::cout << "After postsmooth: " << std::endl;
      printvector(result_[level]);
      #endif
    }
    vec = result_[0];
  }

  /** @brief Jacobi Smoother (GPU version)
  * @param level       Coarse level to which smoother is applied to
  * @param iterations  Number of smoother iterations
  * @param x           The vector smoothing is applied to
  * @param rhs_smooth  The right hand side of the equation for the smoother
  */
  template<typename VectorT>
  void smooth_jacobi(vcl_size_t level, unsigned int iterations, VectorT & x, VectorT const & rhs_smooth) const
  {
    VectorType old_result = x;

    viennacl::ocl::context & ctx = const_cast<viennacl::ocl::context &>(viennacl::traits::opencl_handle(x).context());
    viennacl::linalg::opencl::kernels::compressed_matrix<NumericT>::init(ctx);
    viennacl::ocl::kernel & k = ctx.get_kernel(viennacl::linalg::opencl::kernels::compressed_matrix<NumericT>::program_name(), "jacobi");

    for (unsigned int i=0; i<iterations; ++i)
    {
      if (i > 0)
        old_result = x;
      x.clear();
      viennacl::ocl::enqueue(k(A_[level].handle1().opencl_handle(), A_[level].handle2().opencl_handle(), A_[level].handle().opencl_handle(),
                              static_cast<NumericT>(tag_.get_jacobiweight()),
                              viennacl::traits::opencl_handle(old_result),
                              viennacl::traits::opencl_handle(x),
                              viennacl::traits::opencl_handle(rhs_smooth),
                              static_cast<cl_uint>(rhs_smooth.size())));

    }
  }

  amg_tag & tag() { return tag_; }
};

}
}



#endif

