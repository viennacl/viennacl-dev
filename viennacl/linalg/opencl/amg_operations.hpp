#ifndef VIENNACL_LINALG_OPENCL_AMG_OPERATIONS_HPP
#define VIENNACL_LINALG_OPENCL_AMG_OPERATIONS_HPP

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

/** @file opencl/amg_operations.hpp
    @brief Implementations of routines for AMG in OpenCL.
*/

#include <cstdlib>
#include <cmath>
#include "viennacl/linalg/detail/amg/amg_base.hpp"

#include <map>
#include <set>

namespace viennacl
{
namespace linalg
{
namespace opencl
{
namespace amg
{


///////////////////////////////////////////

/** @brief Routine for taking all connections in the matrix as strong */
template<typename NumericT>
void amg_influence_trivial(compressed_matrix<NumericT> const & A,
                           viennacl::linalg::detail::amg::amg_level_context & amg_context,
                           viennacl::linalg::detail::amg::amg_tag & tag)
{
}


/** @brief Routine for extracting strongly connected points considering a user-provided threshold value */
template<typename NumericT>
void amg_influence_advanced(compressed_matrix<NumericT> const & A,
                            viennacl::linalg::detail::amg::amg_level_context & amg_context,
                            viennacl::linalg::detail::amg::amg_tag & tag)
{
}


/** @brief Dispatcher for influence processing */
template<typename NumericT>
void amg_influence(compressed_matrix<NumericT> const & A,
                   viennacl::linalg::detail::amg::amg_level_context & amg_context,
                   viennacl::linalg::detail::amg::amg_tag & tag)
{
  // TODO: dispatch based on influence tolerance provided
  amg_influence_trivial(A, amg_context, tag);
}

/** @brief Assign IDs to coarse points */
inline void enumerate_coarse_points(viennacl::linalg::detail::amg::amg_level_context & amg_context)
{

}


//////////////////////////////////////



/** @brief AG (aggregation based) coarsening, single-threaded version of stage 1
*
* @param A             Operator matrix on all levels
* @param amg_context   AMG hierarchy datastructures
* @param tag           AMG preconditioner tag
*/
template<typename NumericT>
void amg_coarse_ag_stage1_mis2(compressed_matrix<NumericT> const & A,
                               viennacl::linalg::detail::amg::amg_level_context & amg_context,
                               viennacl::linalg::detail::amg::amg_tag & tag)
{
  std::vector<unsigned int> random_weights(A.size1());
  for (std::size_t i=0; i<random_weights.size(); ++i)
    random_weights[i] = static_cast<unsigned int>(rand()) % static_cast<unsigned int>(A.size1());

  unsigned int num_undecided = static_cast<unsigned int>(A.size1());
  while (num_undecided > 0)
  {
    //
    // init temporary work data:
    //


    //
    // Propagate maximum tuple twice
    //
    for (unsigned int r = 0; r < 2; ++r)
    {
      // max operation

      // copy work array
    }

    //
    // mark MIS and non-MIS nodes:
    //

  } //while
}



/** @brief AG (aggregation based) coarsening. Partially single-threaded version (VIENNACL_AMG_COARSE_AG)
*
* @param A             Operator matrix
* @param amg_context   AMG hierarchy datastructures
* @param tag           AMG preconditioner tag
*/
template<typename NumericT>
void amg_coarse_ag(compressed_matrix<NumericT> const & A,
                   viennacl::linalg::detail::amg::amg_level_context & amg_context,
                   viennacl::linalg::detail::amg::amg_tag & tag)
{

  //
  // Stage 1: Build aggregates:
  //
  if (tag.get_coarse() == VIENNACL_AMG_COARSE_AG_MIS2) amg_coarse_ag_stage1_mis2(A, amg_context, tag);

  viennacl::linalg::opencl::amg::enumerate_coarse_points(amg_context);

  //
  // Stage 2: Propagate coarse aggregate indices to neighbors:
  //


  //
  // Stage 3: Merge remaining undecided points (merging to first aggregate found when cycling over the hierarchy
  //

  //
  // Stage 4: Set undecided points to fine points (coarse ID already set in Stage 3)
  //          Note: Stage 3 and Stage 4 were initially fused, but are now split in order to avoid race conditions (or a fallback to sequential execution).
  //

}




/** @brief Calls the right coarsening procedure
*
* @param A            Operator matrix on all levels
* @param amg_context  AMG hierarchy datastructures
* @param tag          AMG preconditioner tag
*/
template<typename InternalT1>
void amg_coarse(InternalT1 & A,
                viennacl::linalg::detail::amg::amg_level_context & amg_context,
                viennacl::linalg::detail::amg::amg_tag & tag)
{
  switch (tag.get_coarse())
  {
  case VIENNACL_AMG_COARSE_AG_MIS2: amg_coarse_ag(A, amg_context, tag); break;
  default: throw std::runtime_error("not implemented yet");
  }
}




////////////////////////////////////// Interpolation /////////////////////////////


/** @brief AG (aggregation based) interpolation. Multi-Threaded! (VIENNACL_INTERPOL_SA)
 *
 * @param A            Operator matrix
 * @param P            Prolongation matrix
 * @param amg_context  AMG hierarchy datastructures
 * @param tag          AMG configuration tag
*/
template<typename NumericT>
void amg_interpol_ag(compressed_matrix<NumericT> const & A,
                     compressed_matrix<NumericT> & P,
                     viennacl::linalg::detail::amg::amg_level_context & amg_context,
                     viennacl::linalg::detail::amg::amg_tag & tag)
{
  (void)tag;
  P = compressed_matrix<NumericT>(A.size1(), amg_context.num_coarse_, A.size1(), viennacl::traits::context(A));

  // build matrix here

  P.generate_row_block_information();
}



/** @brief Dispatcher for building the interpolation matrix
 *
 * @param A            Operator matrix
 * @param P            Prolongation matrix
 * @param amg_context  AMG hierarchy datastructures
 * @param tag          AMG configuration tag
*/
template<typename MatrixT>
void amg_interpol(MatrixT const & A,
                  MatrixT & P,
                  viennacl::linalg::detail::amg::amg_level_context & amg_context,
                  viennacl::linalg::detail::amg::amg_tag & tag)
{
  switch (tag.get_interpol())
  {
  case VIENNACL_AMG_INTERPOL_AG:      amg_interpol_ag     (A, P, amg_context, tag); break;
  //case VIENNACL_AMG_INTERPOL_SA:      amg_interpol_sa     (level, A, P, pointvector, tag); break;
  default: throw std::runtime_error("Not implemented yet!");
  }
}

} //namespace amg
} //namespace host_based
} //namespace linalg
} //namespace viennacl

#endif
