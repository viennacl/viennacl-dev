#ifndef VIENNACL_LINALG_DETAIL_AMG_AMG_COARSE_HPP
#define VIENNACL_LINALG_DETAIL_AMG_AMG_COARSE_HPP

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

/** @file amg_coarse.hpp
    @brief Implementations of several variants of the AMG coarsening procedure (setup phase). Experimental.
*/

#include <cmath>
#include "viennacl/linalg/detail/amg/amg_base.hpp"

#include <map>
#include <set>
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

/** @brief Calls the right coarsening procedure
* @param level        Coarse level identifier
* @param A            Operator matrix on all levels
* @param pointvector  Vector of points on all levels
* @param slicing      Partitioning of the system matrix to different processors (only used in RS0 and RS3)
* @param tag          AMG preconditioner tag
*/
template<typename InternalT1, typename InternalT2>
void amg_coarse(InternalT1 & A, InternalT2 & pointvector, amg_tag & tag)
{
  switch (tag.get_coarse())
  {
  case VIENNACL_AMG_COARSE_ONEPASS: amg_coarse_classic_onepass(A, pointvector, tag); break;
  /*case VIENNACL_AMG_COARSE_RS:      amg_coarse_classic(A, pointvector, tag); break;
  case VIENNACL_AMG_COARSE_ONEPASS: amg_coarse_classic_onepass(level, A, pointvector, tag); break;
  case VIENNACL_AMG_COARSE_RS0:     amg_coarse_rs0(level, A, pointvector, slicing, tag); break;
  case VIENNACL_AMG_COARSE_RS3:     amg_coarse_rs3(level, A, pointvector, slicing, tag); break;
  */
  case VIENNACL_AMG_COARSE_AG:      amg_coarse_ag(A, pointvector, tag); break;
  default: throw std::runtime_error("not implemented yet");
  }
}


template<typename NumericT, typename PointListT>
void amg_influence(compressed_matrix<NumericT> const & A, PointListT & pointvector, amg_tag & tag)
{
  NumericT     const * A_elements   = viennacl::linalg::host_based::detail::extract_raw_pointer<NumericT>(A.handle());
  unsigned int const * A_row_buffer = viennacl::linalg::host_based::detail::extract_raw_pointer<unsigned int>(A.handle1());
  unsigned int const * A_col_buffer = viennacl::linalg::host_based::detail::extract_raw_pointer<unsigned int>(A.handle2());

  //std::cout << "Calculating influences..." << std::endl;
#ifdef VIENNACL_WITH_OPENMP
  #pragma omp parallel for
#endif
  for (std::size_t i=0; i<A.size1(); ++i)
  {
    unsigned int row_start = A_row_buffer[i];
    unsigned int row_stop  = A_row_buffer[i+1];
    NumericT diag = 0;
    NumericT largest_positive = 0;
    NumericT largest_negative = 0;

    // obtain diagonal element as well as maximum positive and negative off-diagonal entries
    for (unsigned int nnz_index = row_start; nnz_index < row_stop; ++nnz_index)
    {
      unsigned int col = A_col_buffer[nnz_index];
      NumericT value   = A_elements[nnz_index];

      if (col == i)
        diag = value;
      else if (value > largest_positive)
        largest_positive = value;
      else if (value < largest_negative)
        largest_negative = value;
    }

    if (largest_positive <= 0 && largest_negative >= 0) // no offdiagonal entries
      continue;

    // Find all points that strongly influence current point (Yang, p.5)
    //std::cout << "Looking for strongly influencing points for point " << i << std::endl;
    for (unsigned int nnz_index = row_start; nnz_index < row_stop; ++nnz_index)
    {
      unsigned int col = A_col_buffer[nnz_index];

      if (i == col)
        continue;

      NumericT value   = A_elements[nnz_index];

      if (   (diag > 0 && diag * value <= tag.get_threshold() * diag * largest_negative)
          || (diag < 0 && diag * value <= tag.get_threshold() * diag * largest_positive))
      {
        //std::cout << " - Adding influence from point " << col << std::endl;
        pointvector.add_influence(i, col);
      }
    }
  }
}



struct amg_id_influence
{
  amg_id_influence(std::size_t id2, std::size_t influences2) : id(id2), influences(influences2) {}

  unsigned int  id;
  unsigned int  influences;
};

bool operator>(amg_id_influence const & a, amg_id_influence const & b)
{
  if (a.influences > b.influences)
    return true;
  if (a.influences == b.influences)
    return a.id > b.id;
  return false;
}

/** @brief Classical (RS) one-pass coarsening. Single-Threaded! (VIENNACL_AMG_COARSE_CLASSIC_ONEPASS)
* @param level         Course level identifier
* @param A             Operator matrix on all levels
* @param pointvector   Vector of points on all levels
* @param tag           AMG preconditioner tag
*/
template<typename NumericT, typename PointListT>
void amg_coarse_classic_onepass(compressed_matrix<NumericT> const & A, PointListT & pointvector, amg_tag & tag)
{
  typedef typename PointListT::influence_const_iterator  InfluenceIteratorType;

  //get influences:
  amg_influence(A, pointvector, tag);

  std::set<amg_id_influence, std::greater<amg_id_influence> > points_by_influences;

  for (std::size_t i=0; i<pointvector.size(); ++i)
    points_by_influences.insert(amg_id_influence(i, pointvector.influence_count(i)));

  //std::cout << "Starting coarsening process..." << std::endl;

  while (!points_by_influences.empty())
  {
    amg_id_influence point = *(points_by_influences.begin());

    // remove point from queue:
    points_by_influences.erase(points_by_influences.begin());

    //std::cout << "Working on point " << point.id << std::endl;

    // point is already coarse or fine point, continue;
    if (!pointvector.is_undecided(point.id))
      continue;

    //std::cout << " Setting point " << point.id << " to a coarse point." << std::endl;
    // make this a coarse point:
    pointvector.set_coarse(point.id);

    // Set strongly influenced points to fine points:
    InfluenceIteratorType influence_end = pointvector.influence_cend(point.id);
    for (InfluenceIteratorType influence_iter = pointvector.influence_cbegin(point.id); influence_iter != influence_end; ++influence_iter)
    {
      unsigned int influenced_point_id = *influence_iter;

      //std::cout << "Checking point " << influenced_point_id << std::endl;
      if (!pointvector.is_undecided(influenced_point_id))
        continue;

      //std::cout << " Setting point " << influenced_point_id << " to a fine point." << std::endl;
      pointvector.set_fine(influenced_point_id);

      // add one to influence measure for all undecided points strongly influencing this fine point.
      InfluenceIteratorType influence2_end = pointvector.influence_cend(influenced_point_id);
      for (InfluenceIteratorType influence2_iter = pointvector.influence_cbegin(influenced_point_id); influence2_iter != influence2_end; ++influence2_iter)
      {
        if (pointvector.is_undecided(*influence2_iter))
        {
          // grab and remove from set, increase influence counter, store back:
          amg_id_influence point_to_find(*influence2_iter, pointvector.influence_count(*influence2_iter));
          points_by_influences.erase(point_to_find);

          ++point_to_find.influences;
          pointvector.augment_influence(point_to_find.id); // for consistency

          points_by_influences.insert(point_to_find);
        }
      }
    }

  }

  pointvector.enumerate_coarse_points();

}



/** @brief AG (aggregation based) coarsening. Single-Threaded for now (VIENNACL_AMG_COARSE_CLASSIC_ONEPASS)
* @param level         Course level identifier
* @param A             Operator matrix on all levels
* @param pointvector   Vector of points on all levels
* @param tag           AMG preconditioner tag
*/
template<typename NumericT, typename PointListT>
void amg_coarse_ag(compressed_matrix<NumericT> const & A, PointListT & pointvector, amg_tag & tag)
{
  typedef typename PointListT::influence_const_iterator  InfluenceIteratorType;

  //get influences:
  amg_influence(A, pointvector, tag);

  // Stage 1: Build aggregates:
  for (std::size_t i=0; i<A.size1(); ++i)
  {
    // check if node has no aggregates next to it (MIS-2)
    bool is_new_coarse_node = true;
    // Set strongly influenced points to fine points:
    InfluenceIteratorType influence_end = pointvector.influence_cend(i);
    for (InfluenceIteratorType influence_iter = pointvector.influence_cbegin(i); influence_iter != influence_end; ++influence_iter)
    {
      if (!pointvector.is_undecided(*influence_iter)) // either coarse or fine point
      {
        is_new_coarse_node = false;
        break;
      }
    }

    if (is_new_coarse_node)
    {
      //std::cout << "Setting new coarse node: " << i << std::endl;
      pointvector.set_coarse(i);

      // make all strongly influenced neighbors fine points:
      for (InfluenceIteratorType influence_iter = pointvector.influence_cbegin(i); influence_iter != influence_end; ++influence_iter)
      {
        //std::cout << "Setting new fine node: " << *influence_iter << std::endl;
        pointvector.set_fine(*influence_iter);
      }
    }
  }

  pointvector.enumerate_coarse_points();

  //
  // Stage 2: Propagate coarse aggregate indices to neighbors:
  //
#ifdef VIENNACL_WITH_OPENMP
  #pragma omp parallel for
#endif
  for (unsigned int row = 0; row<A.size1(); ++row)
  {
    if (pointvector.is_coarse(row))
    {
      unsigned int coarse_index = pointvector.get_coarse_index(row);

      InfluenceIteratorType influence_end = pointvector.influence_cend(row);
      for (InfluenceIteratorType influence_iter = pointvector.influence_cbegin(row); influence_iter != influence_end; ++influence_iter)
      {
        //std::cout << "Setting fine node " << *influence_iter << " to be aggregated with node " << row << "/" << coarse_index << std::endl;
        pointvector.set_coarse_aggregate(*influence_iter, coarse_index); // Set aggregate index for fine point
      }
    }
  }


  //
  // Stage 3: Merge remaining undecided points (merging to first aggregate found when cycling over the hierarchy
  //
  for (std::size_t i=0; i<A.size1(); ++i)
  {
    if (pointvector.is_undecided(i))
    {
      InfluenceIteratorType influence_end = pointvector.influence_cend(i);
      for (InfluenceIteratorType influence_iter = pointvector.influence_cbegin(i); influence_iter != influence_end; ++influence_iter)
      {
        if (!pointvector.is_undecided(*influence_iter)) // either coarse or fine point
        {
          //std::cout << "Setting fine node " << i << " to be aggregated with node " << *influence_iter << "/" << pointvector.get_coarse_index(*influence_iter) << std::endl;
          pointvector.set_coarse_aggregate(i, pointvector.get_coarse_index(*influence_iter)); // Set aggregate index for fine point
          break;
        }
      }
      //std::cout << "Setting new fine node (merging): " << i << std::endl;
      pointvector.set_fine(i);
    }
  }


}



} //namespace amg
} //namespace detail
} //namespace linalg
} //namespace viennacl

#endif
