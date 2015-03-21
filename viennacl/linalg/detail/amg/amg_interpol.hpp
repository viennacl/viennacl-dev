#ifndef VIENNACL_LINALG_DETAIL_AMG_AMG_INTERPOL_HPP
#define VIENNACL_LINALG_DETAIL_AMG_AMG_INTERPOL_HPP

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

/** @file amg_interpol.hpp
    @brief Implementations of several variants of the AMG interpolation operators (setup phase). Experimental.
*/

#include <boost/numeric/ublas/vector.hpp>
#include <cmath>
#include "viennacl/linalg/detail/amg/amg_base.hpp"

#include <map>
#ifdef VIENNACL_WITH_OPENMP
#include <omp.h>
#endif

#include "viennacl/linalg/detail/amg/amg_debug.hpp"

namespace viennacl
{
namespace linalg
{
namespace detail
{
namespace amg
{

/** @brief Calls the right function to build interpolation matrix
 * @param level        Coarse level identifier
 * @param A            Operator matrix on all levels
 * @param P            Prolongation matrices. P[level] is constructed
 * @param pointvector  Vector of points on all levels
 * @param tag          AMG preconditioner tag
*/
template<typename InternalT1, typename InternalT2>
void amg_interpol(InternalT1 & A, InternalT1 & P, InternalT2 & pointvector, amg_tag & tag)
{
  switch (tag.get_interpol())
  {
  case VIENNACL_AMG_INTERPOL_DIRECT:  amg_interpol_direct (A, P, pointvector, tag); break;
  //case VIENNACL_AMG_INTERPOL_CLASSIC: amg_interpol_classic(level, A, P, pointvector, tag); break;
  case VIENNACL_AMG_INTERPOL_AG:      amg_interpol_ag     (A, P, pointvector, tag); break;
  //case VIENNACL_AMG_INTERPOL_SA:      amg_interpol_sa     (level, A, P, pointvector, tag); break;
  default: throw std::runtime_error("Not implemented yet!");
  }
}

/** @brief Direct interpolation. Multi-threaded! (VIENNACL_AMG_INTERPOL_DIRECT)
 * @param level        Coarse level identifier
 * @param A            Operator matrix on all levels
 * @param P            Prolongation matrices. P[level] is constructed
 * @param pointvector  Vector of points on all levels
 * @param tag          AMG preconditioner tag
*/
template<typename NumericT, typename PointListT>
void amg_interpol_direct(compressed_matrix<NumericT> const & A,
                         compressed_matrix<NumericT> & P,
                         PointListT const & pointvector,
                         amg_tag & tag)
{
  typedef typename PointListT::influence_const_iterator  InfluenceIteratorType;

  NumericT     const * A_elements   = viennacl::linalg::host_based::detail::extract_raw_pointer<NumericT>(A.handle());
  unsigned int const * A_row_buffer = viennacl::linalg::host_based::detail::extract_raw_pointer<unsigned int>(A.handle1());
  unsigned int const * A_col_buffer = viennacl::linalg::host_based::detail::extract_raw_pointer<unsigned int>(A.handle2());

  unsigned int num_coarse = pointvector.num_coarse_points();

  P.resize(A.size1(), num_coarse);

  std::vector<std::map<unsigned int, NumericT> > P_setup(A.size1());

  // Iterate over all points to build the interpolation matrix row-by-row
  // Interpolation for coarse points is immediate using '1'.
  // Interpolation for fine points is set up via corresponding row weights (cf. Yang paper, p. 14)
#ifdef VIENNACL_WITH_OPENMP
  #pragma omp parallel for
#endif
  for (unsigned int row = 0; row<A.size1(); ++row)
  {
    std::map<unsigned int, NumericT> & P_setup_row = P_setup[row];
    if (pointvector.is_coarse(row))
      P_setup_row[pointvector.get_coarse_index(row)] = NumericT(1);
    else if (pointvector.is_fine(row))
    {
      //std::cout << "Building interpolant for fine point " << row << std::endl;

      NumericT row_sum = 0;
      NumericT row_coarse_sum = 0;
      NumericT diag = 0;

      // Row sum of coefficients (without diagonal) and sum of influencing coarse point coefficients has to be computed
      unsigned int row_A_start = A_row_buffer[row];
      unsigned int row_A_end   = A_row_buffer[row + 1];
      InfluenceIteratorType influence_iter = pointvector.influence_cbegin(row);
      InfluenceIteratorType influence_end = pointvector.influence_cend(row);
      for (unsigned int index = row_A_start; index < row_A_end; ++index)
      {
        unsigned int col = A_col_buffer[index];
        NumericT value = A_elements[index];

        if (col == row)
        {
          diag = value;
          continue;
        }
        else if (pointvector.is_coarse(col))
        {
          // Note: One increment is sufficient, because influence_iter traverses an ordered subset of the column indices in this row
          while (influence_iter != influence_end && *influence_iter < col)
            ++influence_iter;

          if (influence_iter != influence_end && *influence_iter == col)
            row_coarse_sum += value;
        }

        row_sum += value;
      }

      NumericT temp_res = -row_sum/(row_coarse_sum*diag);

      if (temp_res > 0 || temp_res < 0)
      {
        // Iterate over all strongly influencing points to build the interpolant
        influence_iter = pointvector.influence_cbegin(row);
        for (unsigned int index = row_A_start; index < row_A_end; ++index)
        {
          unsigned int col = A_col_buffer[index];
          if (!pointvector.is_coarse(col))
            continue;
          NumericT value = A_elements[index];

          // Advance to correct influence metric:
          while (influence_iter != influence_end && *influence_iter < col)
            ++influence_iter;

          if (influence_iter != influence_end && *influence_iter == col)
          {
            //std::cout << " Setting entry "  << temp_res * value << " at " << pointvector.get_coarse_index(col) << " for point " << col << std::endl;
            P_setup_row[pointvector.get_coarse_index(col)] = temp_res * value;
          }
        }
      }

      // TODO truncate interpolation if specified by the user.
      (void)tag;
    }
    else
      throw std::runtime_error("Logic error in direct interpolation: Point is neither coarse-point nor fine-point!");
  }

  // TODO: P_setup can be avoided without sacrificing parallelism.
  viennacl::tools::sparse_matrix_adapter<NumericT> P_adapter(P_setup, P.size1(), P.size2());
  viennacl::copy(P_adapter, P);
}




/** @brief Direct interpolation. Multi-threaded! (VIENNACL_AMG_INTERPOL_DIRECT)
 * @param level        Coarse level identifier
 * @param A            Operator matrix on all levels
 * @param P            Prolongation matrices. P[level] is constructed
 * @param pointvector  Vector of points on all levels
 * @param tag          AMG preconditioner tag
*
template<typename InternalT1, typename InternalT2>
void amg_interpol_direct_old(unsigned int level, InternalT1 & A, InternalT1 & P, InternalT2 & pointvector, amg_tag & tag)
{
  typedef typename InternalT1::value_type         SparseMatrixType;
  //typedef typename InternalType2::value_type    PointVectorType;
  typedef typename SparseMatrixType::value_type   ScalarType;
  typedef typename SparseMatrixType::iterator1    InternalRowIterator;
  typedef typename SparseMatrixType::iterator2    InternalColIterator;

  unsigned int c_points = pointvector[level].get_cpoints();

  // Setup Prolongation/Interpolation matrix
  P[level] = SparseMatrixType(static_cast<unsigned int>(A[level].size1()), c_points);
  P[level].clear();

  // Assign indices to C points
  pointvector[level].build_index();

  // Direct Interpolation (Yang, p.14)
#ifdef VIENNACL_WITH_OPENMP
  #pragma omp parallel for
#endif
  for (long x=0; x < static_cast<long>(pointvector[level].size()); ++x)
  {
    amg_point *pointx = pointvector[level][static_cast<unsigned int>(x)];
    //if (A[level](x,x) > 0)
    //  diag_sign = 1;
    //else
    //  diag_sign = -1;

    // When the current line corresponds to a C point then the diagonal coefficient is 1 and the rest 0
    if (pointx->is_cpoint())
      P[level](static_cast<unsigned int>(x),pointx->get_coarse_index()) = 1;

    // When the current line corresponds to a F point then the diagonal is 0 and the rest has to be computed (Yang, p.14)
    if (pointx->is_fpoint())
    {
      // Jump to row x
      InternalRowIterator row_iter = A[level].begin1();
      row_iter += vcl_size_t(x);

      // Row sum of coefficients (without diagonal) and sum of influencing C point coefficients has to be computed
      ScalarType row_sum = 0;
      ScalarType c_sum   = 0;
      ScalarType diag    = 0;
      for (InternalColIterator col_iter = row_iter.begin(); col_iter != row_iter.end(); ++col_iter)
      {
        long y = static_cast<long>(col_iter.index2());
        if (x == y)// || *col_iter * diag_sign > 0)
        {
          diag += *col_iter;
          continue;
        }

        // Sum all other coefficients in line x
        row_sum += *col_iter;

        amg_point *pointy = pointvector[level][static_cast<unsigned int>(y)];
        // Sum all coefficients that correspond to a strongly influencing C point
        if (pointy->is_cpoint())
          if (pointx->is_influencing(pointy))
            c_sum += *col_iter;
      }
      ScalarType temp_res = -row_sum/(c_sum*diag);

      // Iterate over all strongly influencing points of point x
      for (amg_point::iterator iter = pointx->begin_influencing(); iter != pointx->end_influencing(); ++iter)
      {
        amg_point *pointy = *iter;
        // The value is only non-zero for columns that correspond to a C point
        if (pointy->is_cpoint())
        {
          if (temp_res > 0 || temp_res < 0)
            P[level](static_cast<unsigned int>(x), pointy->get_coarse_index()) = temp_res * A[level](static_cast<unsigned int>(x),pointy->get_index());
        }
      }

      //Truncate interpolation if chosen
      if (tag.get_interpolweight() > 0)
        amg_truncate_row(P[level], static_cast<unsigned int>(x), tag);
    }
  }

  // P test
  //test_interpolation(A[level], P[level], Pointvector[level]);

  #ifdef VIENNACL_AMG_DEBUG
  std::cout << "Prolongation Matrix:" << std::endl;
  printmatrix (P[level]);
  #endif
} */

/** @brief Classical interpolation. Don't use with onepass classical coarsening or RS0 (Yang, p.14)! Multi-threaded! (VIENNACL_AMG_INTERPOL_CLASSIC)
 * @param level        Coarse level identifier
 * @param A            Operator matrix on all levels
 * @param P            Prolongation matrices. P[level] is constructed
 * @param pointvector  Vector of points on all levels
 * @param tag          AMG preconditioner tag
*
template<typename InternalT1, typename InternalT2>
void amg_interpol_classic(unsigned int level, InternalT1 & A, InternalT1 & P, InternalT2 & pointvector, amg_tag & tag)
{
  typedef typename InternalT1::value_type           SparseMatrixType;
  //typedef typename InternalType2::value_type      PointVectorType;
  typedef typename SparseMatrixType::value_type     ScalarType;
  typedef typename SparseMatrixType::iterator1      InternalRowIterator;
  typedef typename SparseMatrixType::iterator2      InternalColIterator;

  unsigned int c_points = pointvector[level].get_cpoints();

  // Setup Prolongation/Interpolation matrix
  P[level] = SparseMatrixType(static_cast<unsigned int>(A[level].size1()), c_points);
  P[level].clear();

  // Assign indices to C points
  pointvector[level].build_index();

  // Classical Interpolation (Yang, p.13-14)
#ifdef VIENNACL_WITH_OPENMP
  #pragma omp parallel for
#endif
  for (long x=0; x < static_cast<long>(pointvector[level].size()); ++x)
  {
    amg_point *pointx = pointvector[level][static_cast<unsigned int>(x)];
    int diag_sign = (A[level](static_cast<unsigned int>(x),static_cast<unsigned int>(x)) > 0) ? 1 : -1;

    // When the current line corresponds to a C point then the diagonal coefficient is 1 and the rest 0
    if (pointx->is_cpoint())
      P[level](static_cast<unsigned int>(x),pointx->get_coarse_index()) = 1;

    // When the current line corresponds to a F point then the diagonal is 0 and the rest has to be computed (Yang, p.14)
    if (pointx->is_fpoint())
    {
      // Jump to row x
      InternalRowIterator row_iter = A[level].begin1();
      row_iter += vcl_size_t(x);

      ScalarType weak_sum = 0;
      amg_sparsevector<ScalarType> c_sum_row(static_cast<unsigned int>(A[level].size1()));
      c_sum_row.clear();
      for (InternalColIterator col_iter = row_iter.begin(); col_iter != row_iter.end(); ++col_iter)
      {
        long k = static_cast<unsigned int>(col_iter.index2());
        amg_point *pointk = pointvector[level][static_cast<unsigned int>(k)];

        // Sum of weakly influencing neighbors + diagonal coefficient
        if (x == k || !pointx->is_influencing(pointk))// || *col_iter * diag_sign > 0)
        {
          weak_sum += *col_iter;
          continue;
        }

        // Sums of coefficients in row k (strongly influening F neighbors) of C point neighbors of x are calculated
        if (pointk->is_fpoint() && pointx->is_influencing(pointk))
        {
          for (amg_point::iterator iter = pointx->begin_influencing(); iter != pointx->end_influencing(); ++iter)
          {
            amg_point *pointm = *iter;
            long m = pointm->get_index();

            if (pointm->is_cpoint())
              // Only use coefficients that have opposite sign of diagonal.
              if (A[level](static_cast<unsigned int>(k),static_cast<unsigned int>(m)) * ScalarType(diag_sign) < 0)
                c_sum_row[static_cast<unsigned int>(k)] += A[level](static_cast<unsigned int>(k), static_cast<unsigned int>(m));
          }
          continue;
        }
      }

      // Iterate over all strongly influencing points of point x
      for (amg_point::iterator iter = pointx->begin_influencing(); iter != pointx->end_influencing(); ++iter)
      {
        amg_point *pointy = *iter;
        long y = pointy->get_index();

        // The value is only non-zero for columns that correspond to a C point
        if (pointy->is_cpoint())
        {
          ScalarType strong_sum = 0;
          // Calculate term for strongly influencing F neighbors
          for (typename amg_sparsevector<ScalarType>::iterator iter2 = c_sum_row.begin(); iter2 != c_sum_row.end(); ++iter2)
          {
            long k = iter2.index();
            // Only use coefficients that have opposite sign of diagonal.
            if (A[level](static_cast<unsigned int>(k),static_cast<unsigned int>(y)) * ScalarType(diag_sign) < 0)
              strong_sum += (A[level](static_cast<unsigned int>(x),static_cast<unsigned int>(k)) * A[level](static_cast<unsigned int>(k),static_cast<unsigned int>(y))) / (*iter2);
          }

          // Calculate coefficient
          ScalarType temp_res = - (A[level](static_cast<unsigned int>(x),static_cast<unsigned int>(y)) + strong_sum) / (weak_sum);
          if (temp_res < 0 || temp_res > 0)
            P[level](static_cast<unsigned int>(x),pointy->get_coarse_index()) = temp_res;
        }
      }

      //Truncate iteration if chosen
      if (tag.get_interpolweight() > 0)
        amg_truncate_row(P[level], static_cast<unsigned int>(x), tag);
    }
  }

  #ifdef VIENNACL_AMG_DEBUG
  std::cout << "Prolongation Matrix:" << std::endl;
  printmatrix (P[level]);
  #endif
} */

/** @brief Interpolation truncation (for VIENNACL_AMG_INTERPOL_DIRECT and VIENNACL_AMG_INTERPOL_CLASSIC)
*
* @param P    Interpolation matrix
* @param row  Row which has to be truncated
* @param tag  AMG preconditioner tag
*/
template<typename SparseMatrixT>
void amg_truncate_row(SparseMatrixT & P, unsigned int row, amg_tag & tag)
{
  typedef typename SparseMatrixT::value_type   ScalarType;
  typedef typename SparseMatrixT::iterator1    InternalRowIterator;
  typedef typename SparseMatrixT::iterator2    InternalColIterator;

  ScalarType row_max, row_min, row_sum_pos, row_sum_neg, row_sum_pos_scale, row_sum_neg_scale;

  InternalRowIterator row_iter = P.begin1();
  row_iter += row;

  row_max = 0;
  row_min = 0;
  row_sum_pos = 0;
  row_sum_neg = 0;

  // Truncate interpolation by making values to zero that are a lot smaller than the biggest value in a row
  // Determine max entry and sum of row (seperately for negative and positive entries)
  for (InternalColIterator col_iter = row_iter.begin(); col_iter != row_iter.end(); ++col_iter)
  {
    if (*col_iter > row_max)
      row_max = *col_iter;
    if (*col_iter < row_min)
      row_min = *col_iter;
    if (*col_iter > 0)
      row_sum_pos += *col_iter;
    if (*col_iter < 0)
      row_sum_neg += *col_iter;
  }

  row_sum_pos_scale = row_sum_pos;
  row_sum_neg_scale = row_sum_neg;

  // Make certain values to zero (seperately for negative and positive entries)
  for (InternalColIterator col_iter = row_iter.begin(); col_iter != row_iter.end(); ++col_iter)
  {
    if (*col_iter > 0 && *col_iter < tag.get_interpolweight() * row_max)
    {
      row_sum_pos_scale -= *col_iter;
      *col_iter = 0;
    }
    if (*col_iter < 0 && *col_iter > tag.get_interpolweight() * row_min)
    {
      row_sum_pos_scale -= *col_iter;
      *col_iter = 0;
    }
  }

  // Scale remaining values such that row sum is unchanged
  for (InternalColIterator col_iter = row_iter.begin(); col_iter != row_iter.end(); ++col_iter)
  {
    if (*col_iter > 0)
      *col_iter = *col_iter *(row_sum_pos/row_sum_pos_scale);
    if (*col_iter < 0)
      *col_iter = *col_iter *(row_sum_neg/row_sum_neg_scale);
  }
}

template<typename NumericT, typename PointListT>
void amg_interpol_ag(compressed_matrix<NumericT> const & A,
                     compressed_matrix<NumericT> & P,
                     PointListT & pointvector,
                     amg_tag & tag)
{
  typedef typename PointListT::influence_const_iterator  InfluenceIteratorType;

  (void)tag;
  P.resize(A.size1(), pointvector.num_coarse_points());

  std::vector<std::map<unsigned int, NumericT> > P_setup(A.size1());

  // Build interpolation matrix:
#ifdef VIENNACL_WITH_OPENMP
  #pragma omp parallel for
#endif
  for (unsigned int row = 0; row<A.size1(); ++row)
  {
    P_setup[row][pointvector.get_coarse_aggregate(row)] = NumericT(1);
  }

  viennacl::tools::sparse_matrix_adapter<NumericT> P_adapter(P_setup, P.size1(), P.size2());
  viennacl::copy(P_adapter, P);
}

/** @brief AG (aggregation based) interpolation. Multi-Threaded! (VIENNACL_INTERPOL_SA)
 * @param level        Coarse level identifier
 * @param A            Operator matrix on all levels
 * @param P            Prolongation matrices. P[level] is constructed
 * @param pointvector  Vector of points on all levels
*
template<typename InternalT1, typename InternalT2>
void amg_interpol_ag(unsigned int level, InternalT1 & A, InternalT1 & P, InternalT2 & pointvector, amg_tag)
{
  typedef typename InternalT1::value_type SparseMatrixType;
  //typedef typename InternalType2::value_type PointVectorType;
  //typedef typename SparseMatrixType::value_type ScalarType;
  //typedef typename SparseMatrixType::iterator1 InternalRowIterator;
  //typedef typename SparseMatrixType::iterator2 InternalColIterator;

  unsigned int c_points = pointvector[level].get_cpoints();

  P[level] = SparseMatrixType(static_cast<unsigned int>(A[level].size1()), c_points);
  P[level].clear();

  // Assign indices to C points
  pointvector[level].build_index();

  // Set prolongation such that F point is interpolated (weight=1) by the aggregate it belongs to (Vanek et al p.6)
#ifdef VIENNACL_WITH_OPENMP
  #pragma omp parallel for
#endif
  for (long x=0; x<static_cast<long>(pointvector[level].size()); ++x)
  {
    amg_point *pointx = pointvector[level][static_cast<unsigned int>(x)];
    amg_point *pointy = pointvector[level][pointx->get_aggregate()];
    // Point x belongs to aggregate y.
    P[level](static_cast<unsigned int>(x), pointy->get_coarse_index()) = 1;
  }

  #ifdef VIENNACL_AMG_DEBUG
  std::cout << "Aggregation based Prolongation:" << std::endl;
  printmatrix(P[level]);
  #endif
} */

/** @brief SA (smoothed aggregate) interpolation. Multi-Threaded! (VIENNACL_INTERPOL_SA)
 * @param level        Coarse level identifier
 * @param A            Operator matrix on all levels
 * @param P            Prolongation matrices. P[level] is constructed
 * @param pointvector  Vector of points on all levels
 * @param tag          AMG preconditioner tag
*
template<typename InternalT1, typename InternalT2>
void amg_interpol_sa(unsigned int level, InternalT1 & A, InternalT1 & P, InternalT2 & pointvector, amg_tag & tag)
{
  typedef typename InternalT1::value_type         SparseMatrixType;
  //typedef typename InternalType2::value_type    PointVectorType;
  typedef typename SparseMatrixType::value_type   ScalarType;
  typedef typename SparseMatrixType::iterator1    InternalRowIterator;
  typedef typename SparseMatrixType::iterator2    InternalColIterator;

  unsigned int c_points = pointvector[level].get_cpoints();

  InternalT1 P_tentative = InternalT1(P.size());
  SparseMatrixType Jacobi = SparseMatrixType(static_cast<unsigned int>(A[level].size1()), static_cast<unsigned int>(A[level].size2()));
  Jacobi.clear();
  P[level] = SparseMatrixType(static_cast<unsigned int>(A[level].size1()), c_points);
  P[level].clear();

  // Build Jacobi Matrix via filtered A matrix (Vanek et al. p.6)
#ifdef VIENNACL_WITH_OPENMP
  #pragma omp parallel for
#endif
  for (long x=0; x<static_cast<long>(A[level].size1()); ++x)
  {
    ScalarType diag = 0;
    InternalRowIterator row_iter = A[level].begin1();
    row_iter += vcl_size_t(x);
    for (InternalColIterator col_iter = row_iter.begin(); col_iter != row_iter.end(); ++col_iter)
    {
      long y = static_cast<long>(col_iter.index2());
      // Determine the structure of the Jacobi matrix by using a filtered matrix of A:
      // The diagonal consists of the diagonal coefficient minus all coefficients of points not in the neighborhood of x.
      // All other coefficients are the same as in A.
      // Already use Jacobi matrix to save filtered A matrix to speed up computation.
      if (x == y)
        diag += *col_iter;
      else if (!pointvector[level][static_cast<unsigned int>(x)]->is_influencing(pointvector[level][static_cast<unsigned int>(y)]))
        diag += -*col_iter;
      else
        Jacobi (static_cast<unsigned int>(x), static_cast<unsigned int>(y)) = *col_iter;
    }
    InternalRowIterator row_iter2 = Jacobi.begin1();
    row_iter2 += vcl_size_t(x);
    // Traverse through filtered A matrix and compute the Jacobi filtering
    for (InternalColIterator col_iter2 = row_iter2.begin(); col_iter2 != row_iter2.end(); ++col_iter2)
    {
        *col_iter2 = - static_cast<ScalarType>(tag.get_interpolweight())/diag * *col_iter2;
    }
    // Diagonal can be computed seperately.
    Jacobi (static_cast<unsigned int>(x), static_cast<unsigned int>(x)) = 1 - static_cast<ScalarType>(tag.get_interpolweight());
  }

  #ifdef VIENNACL_AMG_DEBUG
  std::cout << "Jacobi Matrix:" << std::endl;
  printmatrix(Jacobi);
  #endif

  // Use AG interpolation as tentative prolongation
  amg_interpol_ag(level, A, P_tentative, pointvector, tag);

  #ifdef VIENNACL_AMG_DEBUG
  std::cout << "Tentative Prolongation:" << std::endl;
  printmatrix(P_tentative[level]);
  #endif

  // Multiply Jacobi matrix with tentative prolongation to get actual prolongation
  amg_mat_prod(Jacobi,P_tentative[level],P[level]);

  #ifdef VIENNACL_AMG_DEBUG
  std::cout << "Prolongation Matrix:" << std::endl;
  printmatrix(P[level]);
  #endif
} */

} //namespace amg
} //namespace detail
} //namespace linalg
} //namespace viennacl

#endif
