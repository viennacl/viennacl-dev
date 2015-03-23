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

#include <vector>
#include <cmath>
#include "viennacl/forwards.h"
#include "viennacl/tools/tools.hpp"
#include "viennacl/linalg/prod.hpp"
#include "viennacl/linalg/direct_solve.hpp"

#include "viennacl/linalg/detail/amg/amg_base.hpp"
#include "viennacl/linalg/detail/amg/amg_coarse.hpp"
#include "viennacl/linalg/detail/amg/amg_interpol.hpp"
#include "viennacl/tools/timer.hpp"
#include "viennacl/linalg/direct_solve.hpp"
#include "viennacl/linalg/lu.hpp"

#include <map>

#ifdef VIENNACL_WITH_OPENMP
 #include <omp.h>
#endif

#define VIENNACL_AMG_COARSE_LIMIT 50
#define VIENNACL_AMG_MAX_LEVELS 20

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
template<typename NumericT, typename PointListT>
void amg_setup(std::vector<compressed_matrix<NumericT> > & list_of_A,
               std::vector<compressed_matrix<NumericT> > & list_of_P,
               std::vector<compressed_matrix<NumericT> > & list_of_R,
               PointListT & list_of_pointvectors,
               amg_tag & tag)
{
  viennacl::tools::timer timer;

  // Set number of iterations. If automatic coarse grid construction is chosen (0), then set a maximum size and stop during the process.
  unsigned int iterations = tag.get_coarselevels();
  if (iterations == 0)
    iterations = VIENNACL_AMG_MAX_LEVELS;

  for (unsigned int i=0; i<iterations; ++i)
  {
    std::cout << "Working on Level " << i << std::endl;

    unsigned int max_nnz_per_row = (list_of_A[i].nnz() / list_of_A[i].size1()) + 5; // crude estimate
    //std::cout << "Resizing for " << max_nnz_per_row << " nonzeros per row" << std::endl;
    list_of_pointvectors[i].resize(list_of_A[i].size1(), max_nnz_per_row);

    // Construct C and F points on coarse level (i is fine level, i+1 coarse level).
    timer.start();
    detail::amg::amg_coarse(list_of_A[i], list_of_pointvectors[i], tag);
    std::cout << " Coarse grid construction time: " << timer.get() << std::endl;

    // Calculate number of C and F points on level i.
    unsigned int c_points = list_of_pointvectors[i].num_coarse_points();
    unsigned int f_points = list_of_A[i].size1() - c_points;

    //std::cout << "Level " << i << ": ";
    std::cout << " No of C points = " << c_points << ", ";
    std::cout << " No of F points = " << f_points << std::endl;

    // Stop routine when the maximal coarse level is found (no C or F point). Coarsest level is level i.
    if (c_points == 0 || f_points == 0)
      break;

    // Construct interpolation matrix for level i.
    timer.start();
    detail::amg::amg_interpol(list_of_A[i], list_of_P[i], list_of_pointvectors[i], tag);
    std::cout << " Interpolation construction time: " << timer.get() << std::endl;

    // Compute coarse grid operator (A[i+1] = R * A[i] * P) with R = trans(P).
    timer.start();
    detail::amg::amg_galerkin_prod(list_of_A[i], list_of_P[i], list_of_R[i], list_of_A[i+1]);
    std::cout << " Galerkin product time: " << timer.get() << std::endl;


    // If Limit of coarse points is reached then stop. Coarsest level is level i+1.
    if (tag.get_coarselevels() == 0 && c_points <= VIENNACL_AMG_COARSE_LIMIT)
    {
      tag.set_coarselevels(i+1);
      return;
    }
  }
}




/** @brief Setup AMG preconditioner
*
* @param A            Operator matrices on all levels
* @param P            Prolongation/Interpolation operators on all levels
* @param pointvector  Vector of points on all levels
* @param tag          AMG preconditioner tag
*
template<typename InternalT1, typename InternalT2>
void amg_setup_old(InternalT1 & A, InternalT1 & P, InternalT2 & pointvector, amg_tag & tag)
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
}*/

/** @brief Initialize AMG preconditioner
*
* @param mat          System matrix
* @param A            Operator matrices on all levels
* @param P            Prolongation/Interpolation operators on all levels
* @param pointvector  Vector of points on all levels
* @param tag          AMG preconditioner tag
*/
template<typename MatrixT, typename InternalT1, typename InternalT2>
void amg_init(MatrixT const & mat, InternalT1 & A, InternalT1 & P, InternalT1 & R, InternalT2 & pointvector, amg_tag & tag)
{
  //typedef typename MatrixType::value_type ScalarType;
  typedef typename InternalT1::value_type SparseMatrixType;

  std::size_t num_levels = (tag.get_coarselevels() > 0) ? tag.get_coarselevels() : VIENNACL_AMG_MAX_LEVELS;

  A.resize(num_levels+1);
  P.resize(num_levels);
  R.resize(num_levels);
  pointvector.resize(num_levels);

  // Insert operator matrix as operator for finest level.
  //SparseMatrixType A0(mat);
  //A.insert_element(0, A0);
  A[0] = mat;
}

/** @brief Save operators after setup phase for CPU computation.
*
* @param A      Operator matrices on all levels on the CPU
* @param P      Prolongation/Interpolation operators on all levels on the CPU
* @param R      Restriction operators on all levels on the CPU
* @param A_setup    Operators matrices on all levels from setup phase
* @param P_setup    Prolongation/Interpolation operators on all levels from setup phase
* @param tag    AMG preconditioner tag
*
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
} */

/** @brief Save operators after setup phase for GPU computation.
*
* @param A          Operator matrices on all levels on the GPU
* @param P          Prolongation/Interpolation operators on all levels on the GPU
* @param R          Restriction operators on all levels on the GPU
* @param A_setup    Operators matrices on all levels from setup phase
* @param P_setup    Prolongation/Interpolation operators on all levels from setup phase
* @param tag        AMG preconditioner tag
* @param ctx        Optional context in which the auxiliary objects are created (one out of multiple OpenCL contexts, CUDA, host)
*
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
} */

/** @brief Setup data structures for precondition phase.
*
* @param result      Result vector on all levels
* @param rhs         RHS vector on all levels
* @param residual    Residual vector on all levels
* @param A           Operators matrices on all levels from setup phase
* @param tag         AMG preconditioner tag
*
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
} */


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
void amg_setup_apply(InternalVectorT & result,
                     InternalVectorT & result_backup,
                     InternalVectorT & rhs,
                     InternalVectorT & residual,
                     SparseMatrixT const & A,
                     amg_tag const & tag,
                     viennacl::context ctx)
{
  typedef typename InternalVectorT::value_type VectorType;

  result.resize(tag.get_coarselevels()+1);
  result_backup.resize(tag.get_coarselevels()+1);
  rhs.resize(tag.get_coarselevels()+1);
  residual.resize(tag.get_coarselevels());

  for (unsigned int level=0; level < tag.get_coarselevels()+1; ++level)
  {
    result[level] = VectorType(A[level].size1(), ctx);
    result_backup[level] = VectorType(A[level].size1(), ctx);
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
void amg_lu(viennacl::matrix<NumericT> & op,
            SparseMatrixT const & A)
{
  op.resize(A.size1(), A.size2(), false);
  viennacl::linalg::assign_to_dense(A, op);

  viennacl::linalg::lu_factorize(op);
}

/** @brief AMG preconditioner class, can be supplied to solve()-routines
*/
template<typename MatrixT>
class amg_precond;


/** @brief AMG preconditioner class, can be supplied to solve()-routines.
*
*  Specialization for compressed_matrix
*/
template<typename NumericT, unsigned int AlignmentV>
class amg_precond< compressed_matrix<NumericT, AlignmentV> >
{
  typedef viennacl::compressed_matrix<NumericT, AlignmentV> SparseMatrixType;
  typedef viennacl::vector<NumericT>                        VectorType;
  typedef detail::amg::amg_pointlist                        PointListType;

  std::vector<SparseMatrixType> A_list_;
  std::vector<SparseMatrixType> P_list_;
  std::vector<SparseMatrixType> R_list_;
  std::vector<PointListType>  pointvector_list_;

  viennacl::matrix<NumericT>        coarsest_op_;

  mutable std::vector<VectorType> result_list_;
  mutable std::vector<VectorType> result_backup_list_;
  mutable std::vector<VectorType> rhs_list_;
  mutable std::vector<VectorType> residual_list_;

  viennacl::context ctx_;

  mutable bool done_init_apply_;

  amg_tag tag_;

public:

  amg_precond() {}

  /** @brief The constructor. Builds data structures.
  *
  * @param mat  System matrix
  * @param tag  The AMG tag
  */
  amg_precond(compressed_matrix<NumericT, AlignmentV> const & mat,
              amg_tag const & tag)
    : ctx_(viennacl::traits::context(mat))
  {
    tag_ = tag;

    // Initialize data structures.
    amg_init(mat, A_list_, P_list_, R_list_, pointvector_list_, tag_);
  }

  /** @brief Start setup phase for this class and copy data structures.
  */
  void setup()
  {
    // Start setup phase.
    amg_setup(A_list_, P_list_, R_list_, pointvector_list_, tag_);

    // Setup precondition phase (Data structures).
    amg_setup_apply(result_list_, result_backup_list_, rhs_list_, residual_list_, A_list_, tag_, ctx_);

    // LU factorization for direct solve.
    amg_lu(coarsest_op_, A_list_[tag_.get_coarselevels()]);
  }


  /** @brief Returns complexity measures
  *
  * @param avgstencil  Average stencil sizes on all levels
  * @return     Operator complexity of AMG method
  */
  /*
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
  } */

  /** @brief Precondition Operation
  *
  * @param vec The vector to which preconditioning is applied to
  */
  template<typename VectorT>
  void apply(VectorT & vec) const
  {
    vcl_size_t level;

    // Precondition operation (Yang, p.3).
    rhs_list_[0] = vec;

    // Part 1: Restrict down to coarsest level
    for (level=0; level < tag_.get_coarselevels(); level++)
    {
      result_list_[level].clear();

      // Apply Smoother presmooth_ times.
      viennacl::linalg::smooth_jacobi(tag_.get_presmooth(),
                                      A_list_[level],
                                      result_list_[level],
                                      result_backup_list_[level],
                                      rhs_list_[level],
                                      static_cast<NumericT>(tag_.get_jacobiweight()));

      // Compute residual.
      //residual[level] = rhs_[level] - viennacl::linalg::prod(A_[level], result_[level]);
      residual_list_[level] = viennacl::linalg::prod(A_list_[level], result_list_[level]);
      residual_list_[level] = rhs_list_[level] - residual_list_[level];

      // Restrict to coarse level. Result is RHS of coarse level equation.
      //residual_coarse[level] = viennacl::linalg::prod(R[level],residual[level]);
      rhs_list_[level+1] = viennacl::linalg::prod(R_list_[level], residual_list_[level]);
    }

    // Part 2: On highest level use direct solve to solve equation (on the CPU)
    result_list_[level] = rhs_list_[level];
    viennacl::linalg::lu_substitute(coarsest_op_, result_list_[level]);

    // Part 3: Prolongation to finest level
    for (int level2 = static_cast<int>(tag_.get_coarselevels()-1); level2 >= 0; level2--)
    {
      level = static_cast<vcl_size_t>(level2);

      // Interpolate error to fine level and correct solution.
      result_backup_list_[level] = viennacl::linalg::prod(P_list_[level], result_list_[level+1]);
      result_list_[level] += result_backup_list_[level];

      // Apply Smoother postsmooth_ times.
      viennacl::linalg::smooth_jacobi(tag_.get_postsmooth(),
                                      A_list_[level],
                                      result_list_[level],
                                      result_backup_list_[level],
                                      rhs_list_[level],
                                      static_cast<NumericT>(tag_.get_jacobiweight()));
    }
    vec = result_list_[0];
  }



  amg_tag & tag() { return tag_; }
};

}
}



#endif

