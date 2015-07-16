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

#include <cmath>
#include "viennacl/linalg/detail/amg/amg_base.hpp"
#include "viennacl/traits/context.hpp"

#include <map>
#ifdef VIENNACL_WITH_OPENMP
#include <omp.h>
#endif

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
  P = compressed_matrix<NumericT>(A.size1(), pointvector.num_coarse_points(), A.size1(), viennacl::traits::context(A));

  NumericT     * P_elements   = viennacl::linalg::host_based::detail::extract_raw_pointer<NumericT>(P.handle());
  unsigned int * P_row_buffer = viennacl::linalg::host_based::detail::extract_raw_pointer<unsigned int>(P.handle1());
  unsigned int * P_col_buffer = viennacl::linalg::host_based::detail::extract_raw_pointer<unsigned int>(P.handle2());

  // Build interpolation matrix:
#ifdef VIENNACL_WITH_OPENMP
  #pragma omp parallel for
#endif
  for (unsigned int row = 0; row<A.size1(); ++row)
  {
    P_elements[row]   = NumericT(1);
    P_row_buffer[row] = row;
    P_col_buffer[row] = pointvector.get_coarse_aggregate(row);
  }
  P_row_buffer[A.size1()] = A.size1(); // don't forget finalizer

  P.generate_row_block_information();
}

} //namespace amg
} //namespace detail
} //namespace linalg
} //namespace viennacl

#endif
