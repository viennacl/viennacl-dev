#ifndef VIENNACL_LINALG_HOST_BASED_AMG_OPERATIONS_HPP
#define VIENNACL_LINALG_HOST_BASED_AMG_OPERATIONS_HPP

/* =========================================================================
   Copyright (c) 2010-2016, Institute for Microelectronics,
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

/** @file host_based/amg_operations.hpp
    @brief Implementations of routines for AMG using the CPU on the host (with OpenMP if enabled).
*/

#include <cstdlib>
#include <cmath>
#include "viennacl/linalg/detail/amg/amg_base.hpp"

#include <map>
#include <set>
#include <functional>
#ifdef VIENNACL_WITH_OPENMP
#include <omp.h>
#endif

namespace viennacl
{
namespace linalg
{
namespace host_based
{
namespace amg
{


///////////////////////////////////////////

/** @brief Routine for taking all connections in the matrix as strong */
template<typename NumericT>
void amg_influence_trivial(compressed_matrix<NumericT> const & A,
                           viennacl::linalg::detail::amg::amg_level_context & amg_context,
                           viennacl::linalg::amg_tag & tag)
{
  (void)tag;

  unsigned int const * A_row_buffer = viennacl::linalg::host_based::detail::extract_raw_pointer<unsigned int>(A.handle1());
  unsigned int const * A_col_buffer = viennacl::linalg::host_based::detail::extract_raw_pointer<unsigned int>(A.handle2());

  unsigned int *influences_row_ptr = viennacl::linalg::host_based::detail::extract_raw_pointer<unsigned int>(amg_context.influence_jumper_.handle());
  unsigned int *influences_id_ptr  = viennacl::linalg::host_based::detail::extract_raw_pointer<unsigned int>(amg_context.influence_ids_.handle());
  unsigned int *influences_values_ptr  = viennacl::linalg::host_based::detail::extract_raw_pointer<unsigned int>(amg_context.influence_values_.handle());

#ifdef VIENNACL_WITH_OPENMP
  #pragma omp parallel for
#endif
  for (long i2=0; i2<static_cast<long>(A.size1()); ++i2)
  {
    vcl_size_t i = vcl_size_t(i2);
    influences_row_ptr[i] = A_row_buffer[i];
    influences_values_ptr[i] = A_row_buffer[i+1] - A_row_buffer[i];
  }
  influences_row_ptr[A.size1()] = A_row_buffer[A.size1()];

#ifdef VIENNACL_WITH_OPENMP
  #pragma omp parallel for
#endif
  for (long i=0; i<long(A.nnz()); ++i)
    influences_id_ptr[i] = A_col_buffer[i];
}


/** @brief Routine for extracting strongly connected points considering a user-provided threshold value */
template<typename NumericT>
void amg_influence_advanced(compressed_matrix<NumericT> const & A,
                            viennacl::linalg::detail::amg::amg_level_context & amg_context,
                            viennacl::linalg::amg_tag & tag)
{
  NumericT     const * A_elements   = viennacl::linalg::host_based::detail::extract_raw_pointer<NumericT>(A.handle());
  unsigned int const * A_row_buffer = viennacl::linalg::host_based::detail::extract_raw_pointer<unsigned int>(A.handle1());
  unsigned int const * A_col_buffer = viennacl::linalg::host_based::detail::extract_raw_pointer<unsigned int>(A.handle2());

  unsigned int *influences_row_ptr = viennacl::linalg::host_based::detail::extract_raw_pointer<unsigned int>(amg_context.influence_jumper_.handle());
  unsigned int *influences_id_ptr  = viennacl::linalg::host_based::detail::extract_raw_pointer<unsigned int>(amg_context.influence_ids_.handle());

  //
  // Step 1: Scan influences in order to allocate the necessary memory
  //
#ifdef VIENNACL_WITH_OPENMP
  #pragma omp parallel for
#endif
  for (long i2=0; i2<static_cast<long>(A.size1()); ++i2)
  {
    vcl_size_t i = vcl_size_t(i2);
    unsigned int row_start = A_row_buffer[i];
    unsigned int row_stop  = A_row_buffer[i+1];
    NumericT diag = 0;
    NumericT largest_positive = 0;
    NumericT largest_negative = 0;
    unsigned int num_influences = 0;

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
    {
      influences_row_ptr[i] = 0;
      continue;
    }

    // Find all points that strongly influence current point (Yang, p.5)
    //std::cout << "Looking for strongly influencing points for point " << i << std::endl;
    for (unsigned int nnz_index = row_start; nnz_index < row_stop; ++nnz_index)
    {
      unsigned int col = A_col_buffer[nnz_index];

      if (i == col)
        continue;

      NumericT value   = A_elements[nnz_index];

      if (   (diag > 0 && diag * value <= tag.get_strong_connection_threshold() * diag * largest_negative)
          || (diag < 0 && diag * value <= tag.get_strong_connection_threshold() * diag * largest_positive))
      {
        ++num_influences;
      }
    }

    influences_row_ptr[i] = num_influences;
  }

  //
  // Step 2: Exclusive scan on number of influences to obtain CSR-like datastructure
  //
  unsigned int current_entry = 0;
  for (std::size_t i=0; i<A.size1(); ++i)
  {
    unsigned int tmp = influences_row_ptr[i];
    influences_row_ptr[i] = current_entry;
    current_entry += tmp;
  }
  influences_row_ptr[A.size1()] = current_entry;


  //
  // Step 3: Write actual influences
  //
#ifdef VIENNACL_WITH_OPENMP
  #pragma omp parallel for
#endif
  for (long i2=0; i2<static_cast<long>(A.size1()); ++i2)
  {
    unsigned int i = static_cast<unsigned int>(i2);
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
    unsigned int *influences_id_write_ptr = influences_id_ptr + influences_row_ptr[i];
    for (unsigned int nnz_index = row_start; nnz_index < row_stop; ++nnz_index)
    {
      unsigned int col = A_col_buffer[nnz_index];

      if (i == col)
        continue;

      NumericT value   = A_elements[nnz_index];

      if (   (diag > 0 && diag * value <= tag.get_strong_connection_threshold() * diag * largest_negative)
          || (diag < 0 && diag * value <= tag.get_strong_connection_threshold() * diag * largest_positive))
      {
        //std::cout << " - Adding influence from point " << col << std::endl;
        *influences_id_write_ptr = col;
        ++influences_id_write_ptr;
      }
    }
  }

}


/** @brief Dispatcher for influence processing */
template<typename NumericT>
void amg_influence(compressed_matrix<NumericT> const & A,
                   viennacl::linalg::detail::amg::amg_level_context & amg_context,
                   viennacl::linalg::amg_tag & tag)
{
  // TODO: dispatch based on influence tolerance provided
  amg_influence_trivial(A, amg_context, tag);
}



/** @brief Assign IDs to coarse points */
inline void enumerate_coarse_points(viennacl::linalg::detail::amg::amg_level_context & amg_context)
{
  unsigned int *point_types_ptr  = viennacl::linalg::host_based::detail::extract_raw_pointer<unsigned int>(amg_context.point_types_.handle());
  unsigned int *coarse_id_ptr    = viennacl::linalg::host_based::detail::extract_raw_pointer<unsigned int>(amg_context.coarse_id_.handle());

  unsigned int num_coarse = 0;
#ifdef VIENNACL_WITH_OPENMP
#pragma omp parallel for reduction(+: num_coarse)
#endif
  for (vcl_size_t i=0; i<amg_context.coarse_id_.size(); ++i)
  {
    if (point_types_ptr[i] == viennacl::linalg::detail::amg::amg_level_context::POINT_TYPE_COARSE)
    {
      coarse_id_ptr[i] = 1;
      num_coarse += 1;
    }
    else
      coarse_id_ptr[i] = 0;
  }

  // possible improvement: fuse for-loop above with first stage of exclusive scan
  viennacl::linalg::exclusive_scan(amg_context.coarse_id_, amg_context.coarse_id_);

  amg_context.num_coarse_ = num_coarse;
}




//////////////////////////////////////


/** @brief Helper struct for sequential classical one-pass coarsening */
struct amg_id_influence
{
  amg_id_influence(std::size_t id2, std::size_t influences2) : id(static_cast<unsigned int>(id2)), influences(static_cast<unsigned int>(influences2)) {}

  unsigned int  id;
  unsigned int  influences;
};

inline bool operator>(amg_id_influence const & a, amg_id_influence const & b)
{
  if (a.influences > b.influences)
    return true;
  if (a.influences == b.influences)
    return a.id > b.id;
  return false;
}

/** @brief Classical (RS) one-pass coarsening. Single-Threaded! (VIENNACL_AMG_COARSE_CLASSIC_ONEPASS)
*
* @param A             Operator matrix for the respective level
* @param amg_context   AMG datastructure object for the grid hierarchy
* @param tag           AMG preconditioner tag
*/
template<typename NumericT>
void amg_coarse_classic_onepass(compressed_matrix<NumericT> const & A,
                                viennacl::linalg::detail::amg::amg_level_context & amg_context,
                                viennacl::linalg::amg_tag & tag)
{
  unsigned int *point_types_ptr       = viennacl::linalg::host_based::detail::extract_raw_pointer<unsigned int>(amg_context.point_types_.handle());
  unsigned int *influences_row_ptr    = viennacl::linalg::host_based::detail::extract_raw_pointer<unsigned int>(amg_context.influence_jumper_.handle());
  unsigned int *influences_id_ptr     = viennacl::linalg::host_based::detail::extract_raw_pointer<unsigned int>(amg_context.influence_ids_.handle());
  unsigned int *influences_values_ptr = viennacl::linalg::host_based::detail::extract_raw_pointer<unsigned int>(amg_context.influence_values_.handle());

  std::set<amg_id_influence, std::greater<amg_id_influence> > points_by_influences;

  amg_influence_advanced(A, amg_context, tag);

  for (std::size_t i=0; i<A.size1(); ++i)
    points_by_influences.insert(amg_id_influence(i, influences_values_ptr[i]));

  //std::cout << "Starting coarsening process..." << std::endl;

  while (!points_by_influences.empty())
  {
    amg_id_influence point = *(points_by_influences.begin());

    // remove point from queue:
    points_by_influences.erase(points_by_influences.begin());

    //std::cout << "Working on point " << point.id << std::endl;

    // point is already coarse or fine point, continue;
    if (point_types_ptr[point.id] != viennacl::linalg::detail::amg::amg_level_context::POINT_TYPE_UNDECIDED)
      continue;

    //std::cout << " Setting point " << point.id << " to a coarse point." << std::endl;
    // make this a coarse point:
    point_types_ptr[point.id] = viennacl::linalg::detail::amg::amg_level_context::POINT_TYPE_COARSE;

    // Set strongly influenced points to fine points:
    unsigned int j_stop = influences_row_ptr[point.id + 1];
    for (unsigned int j = influences_row_ptr[point.id]; j < j_stop; ++j)
    {
      unsigned int influenced_point_id = influences_id_ptr[j];

      //std::cout << "Checking point " << influenced_point_id << std::endl;
      if (point_types_ptr[influenced_point_id] != viennacl::linalg::detail::amg::amg_level_context::POINT_TYPE_UNDECIDED)
        continue;

      //std::cout << " Setting point " << influenced_point_id << " to a fine point." << std::endl;
      point_types_ptr[influenced_point_id] = viennacl::linalg::detail::amg::amg_level_context::POINT_TYPE_FINE;

      // add one to influence measure for all undecided points strongly influencing this fine point.
      unsigned int k_stop = influences_row_ptr[influenced_point_id + 1];
      for (unsigned int k = influences_row_ptr[influenced_point_id]; k < k_stop; ++k)
      {
        unsigned int influenced_influenced_point_id = influences_id_ptr[k];
        if (point_types_ptr[influenced_influenced_point_id] == viennacl::linalg::detail::amg::amg_level_context::POINT_TYPE_UNDECIDED)
        {
          // grab and remove from set, increase influence counter, store back:
          amg_id_influence point_to_find(influenced_influenced_point_id, influences_values_ptr[influenced_influenced_point_id]);
          points_by_influences.erase(point_to_find);

          point_to_find.influences += 1;
          influences_values_ptr[influenced_influenced_point_id] += 1; // for consistency

          points_by_influences.insert(point_to_find);
        }
      } //for
    } // for

  } // while

  viennacl::linalg::host_based::amg::enumerate_coarse_points(amg_context);
}


/** @brief Classical coarsening using parallel maximum independent sets
*
* Algorithm follows PMIS as outlined in "Reducing Complexity in Parallel Algebraic Multigrid Preconditioners" by De Sterck, Meyer-Yang, Heys (2004)
*
* @param A             Operator matrix for the respective level
* @param amg_context   AMG datastructure object for the grid hierarchy
* @param tag           AMG preconditioner tag
*/
template<typename NumericT>
void amg_coarse_classic_pmis1(compressed_matrix<NumericT> const & A,
                              viennacl::linalg::detail::amg::amg_level_context & amg_context,
                              viennacl::linalg::amg_tag & tag)
{
  (void)tag;

  amg_influence_trivial(A, amg_context, tag);
  //amg_influence_advanced(A, amg_context, tag);

  unsigned int  *point_types_ptr       = viennacl::linalg::host_based::detail::extract_raw_pointer<unsigned int>(amg_context.point_types_.handle());
  unsigned int *influences_row_ptr    = viennacl::linalg::host_based::detail::extract_raw_pointer<unsigned int>(amg_context.influence_jumper_.handle());
  unsigned int *influences_id_ptr     = viennacl::linalg::host_based::detail::extract_raw_pointer<unsigned int>(amg_context.influence_ids_.handle());

  std::vector<float> random_weights(A.size1());
  for (std::size_t i=0; i<random_weights.size(); ++i)
    random_weights[i] = float(influences_row_ptr[i+1] - influences_row_ptr[i]) + (float(rand()) / float(RAND_MAX));

  std::size_t num_threads = 1;
#ifdef VIENNACL_WITH_OPENMP
  num_threads = omp_get_max_threads();
#endif

  viennacl::vector<float>        work_random2(A.size1(), viennacl::traits::context(A));
  viennacl::vector<unsigned int> work_index2(A.size1(), viennacl::traits::context(A));

  float        *work_random2_ptr    = viennacl::linalg::host_based::detail::extract_raw_pointer<float>(work_random2.handle());
  unsigned int *work_index2_ptr     = viennacl::linalg::host_based::detail::extract_raw_pointer<unsigned int>(work_index2.handle());


  unsigned int num_undecided = static_cast<unsigned int>(A.size1());
  unsigned int pmis_iters = 0;

  //
  // Setup: Set orphaned points with no strong connection to fine points:
  //
  for (long i=0; i<static_cast<long>(A.size1()); ++i)
  {
    if (influences_row_ptr[i] == influences_row_ptr[i + 1])
    {
      point_types_ptr[i] = viennacl::linalg::detail::amg::amg_level_context::POINT_TYPE_FINE;
    }
  }


  //
  // Repeat PMIS iteration until all nodes have been decided
  //
  while (num_undecided > 0)
  {
    ++pmis_iters;

    //
    // Propagate maximum tuple once (only required for undecided points)
    //
#ifdef VIENNACL_WITH_OPENMP
    #pragma omp parallel for
#endif
    for (long i=0; i<static_cast<long>(A.size1()); ++i)
    {
      if (point_types_ptr[i] != viennacl::linalg::detail::amg::amg_level_context::POINT_TYPE_UNDECIDED)
        continue;

      // load
      float random = random_weights[i];
      unsigned int index  = i;

      // max
      unsigned int j_stop = influences_row_ptr[i + 1];
      for (unsigned int j = influences_row_ptr[i]; j < j_stop; ++j)
      {
        unsigned int influenced_point_id = influences_id_ptr[j];

        if (point_types_ptr[influenced_point_id] != viennacl::linalg::detail::amg::amg_level_context::POINT_TYPE_UNDECIDED)
          continue;

        float other_random = random_weights[influenced_point_id];

        if (    random < other_random
            || (random <= other_random && index < influenced_point_id))
        {
          random = other_random;
          index  = influenced_point_id;
        }
      } // for

      // store
      work_random2_ptr[i] = random;
      work_index2_ptr[i]  = index;
    }


    //
    // mark coarse nodes:
    //

// OpenMP acceleration for this loop disabled because of troubles with NVCC in CUDA 6.0.
//#if defined(VIENNACL_WITH_OPENMP)
//  #pragma omp parallel for
//#endif
    for (long i=0; i<static_cast<long>(A.size1()); ++i)
    {
      if (point_types_ptr[i] == viennacl::linalg::detail::amg::amg_level_context::POINT_TYPE_UNDECIDED)
      {
        //if (work_has_undecided_neighbor[i] && i == work_index2_ptr[i]) // this is a local maximum, so turn this into a coarse node
        if (i == work_index2_ptr[i]) // this is a local maximum, so turn this into a coarse node
        {
          point_types_ptr[i] = viennacl::linalg::detail::amg::amg_level_context::POINT_TYPE_COARSE;

          // set neighbors to fine nodes (race conditions are harmless here, because the same value is set to all strongly influenced undecided neighbor nodes)
          unsigned int j_stop = influences_row_ptr[i + 1];
          for (unsigned int j = influences_row_ptr[i]; j < j_stop; ++j)
          {
            unsigned int influenced_point_id = influences_id_ptr[j];
            if (point_types_ptr[influenced_point_id] == viennacl::linalg::detail::amg::amg_level_context::POINT_TYPE_UNDECIDED)
              point_types_ptr[influenced_point_id] = viennacl::linalg::detail::amg::amg_level_context::POINT_TYPE_FINE;
          }
        }
      }
    }

    //
    // count undecided nodes
    //
    std::vector<unsigned int> thread_buffer(num_threads);

#ifdef VIENNACL_WITH_OPENMP
    #pragma omp parallel for
#endif
    for (long i2=0; i2<static_cast<long>(A.size1()); ++i2)
    {
      unsigned int i = static_cast<unsigned int>(i2);

      if (point_types_ptr[i] == viennacl::linalg::detail::amg::amg_level_context::POINT_TYPE_UNDECIDED)
      {
#ifdef VIENNACL_WITH_OPENMP
        thread_buffer[omp_get_thread_num()] += 1;
#else
        thread_buffer[0] += 1;
#endif
      }
    }

    num_undecided = 0;
    for (std::size_t i=0; i<thread_buffer.size(); ++i)
      num_undecided += thread_buffer[i];
  } // while

  viennacl::linalg::host_based::amg::enumerate_coarse_points(amg_context);
}




//////////////////////////


/** @brief AG (aggregation based) coarsening, single-threaded version of stage 1
*
* @param A             Operator matrix for the respective level
* @param amg_context   AMG datastructure object for the grid hierarchy
* @param tag           AMG preconditioner tag
*/
template<typename NumericT>
void amg_coarse_ag_stage1_sequential(compressed_matrix<NumericT> const & A,
                                     viennacl::linalg::detail::amg::amg_level_context & amg_context,
                                     viennacl::linalg::amg_tag & tag)
{
  (void)tag;
  unsigned int *point_types_ptr       = viennacl::linalg::host_based::detail::extract_raw_pointer<unsigned int>(amg_context.point_types_.handle());
  unsigned int *influences_row_ptr    = viennacl::linalg::host_based::detail::extract_raw_pointer<unsigned int>(amg_context.influence_jumper_.handle());
  unsigned int *influences_id_ptr     = viennacl::linalg::host_based::detail::extract_raw_pointer<unsigned int>(amg_context.influence_ids_.handle());

  for (unsigned int i=0; i<static_cast<unsigned int>(A.size1()); ++i)
  {
    // check if node has no aggregates next to it (MIS-2)
    bool is_new_coarse_node = true;

    // Set strongly influenced points to fine points:
    unsigned int j_stop = influences_row_ptr[i + 1];
    for (unsigned int j = influences_row_ptr[i]; j < j_stop; ++j)
    {
      unsigned int influenced_point_id = influences_id_ptr[j];
      if (point_types_ptr[influenced_point_id] != viennacl::linalg::detail::amg::amg_level_context::POINT_TYPE_UNDECIDED) // either coarse or fine point
      {
        is_new_coarse_node = false;
        break;
      }
    }

    if (is_new_coarse_node)
    {
      // make all strongly influenced neighbors fine points:
      for (unsigned int j = influences_row_ptr[i]; j < j_stop; ++j)
      {
        unsigned int influenced_point_id = influences_id_ptr[j];
        point_types_ptr[influenced_point_id] = viennacl::linalg::detail::amg::amg_level_context::POINT_TYPE_FINE;
      }

      //std::cout << "Setting new coarse node: " << i << std::endl;
      // Note: influences may include diagonal element, so it's important to *first* set fine points above before setting the coarse information here
      point_types_ptr[i] = viennacl::linalg::detail::amg::amg_level_context::POINT_TYPE_COARSE;
    }
  }
}


/** @brief Propagates the work state information a distance of two, using a maximum operation to overwrite 'smaller' work state */
template<typename IndexT>
void amg_coarse_ag_stage1_mis2_propagate(IndexT const *work_state_ptr,  IndexT const *work_random_ptr,  IndexT const *work_index_ptr,
                                         IndexT       *work_state2_ptr, IndexT       *work_random2_ptr, IndexT       *work_index2_ptr,
                                         IndexT const *influences_row_ptr, IndexT const *influences_id_ptr, long num_points)
{

#ifdef VIENNACL_WITH_OPENMP
  #pragma omp parallel for
#endif
  for (long i=0; i<num_points; ++i)
  {
    // load
    IndexT state  = work_state_ptr[i];
    IndexT random = work_random_ptr[i];
    IndexT index  = work_index_ptr[i];

    // max
    IndexT j_stop = influences_row_ptr[i + 1];
    for (IndexT j = influences_row_ptr[i]; j < j_stop; ++j)
    {
      IndexT influenced_point_id = influences_id_ptr[j];

      IndexT other_state = work_state_ptr[influenced_point_id];


      // lexigraphical triple-max (not particularly pretty, but does the job):
      if (state < other_state)
      {
        state  = other_state;
        random = work_random_ptr[influenced_point_id];
        index  = work_index_ptr[influenced_point_id];
      }
      else if (state == other_state)
      {
        IndexT other_random = work_random_ptr[influenced_point_id];

        if (random < other_random)
        {
          state  = other_state;
          random = other_random;
          index  = work_index_ptr[influenced_point_id];
        }
        else if (random == other_random)
        {
          IndexT other_index = work_index_ptr[influenced_point_id];

          if (index < other_index)
          {
            state  = other_state;
            random = other_random;
            index  = other_index;
          }
        }
      }
    } // for

    // store
    work_state2_ptr[i]  = state;
    work_random2_ptr[i] = random;
    work_index2_ptr[i]  = index;
  }

}


/** @brief AG (aggregation based) coarsening, multi-threaded version of stage 1 using parallel maximum independent sets
*
* @param A             Operator matrix for the respective level
* @param amg_context   AMG datastructure object for the grid hierarchy
* @param tag           AMG preconditioner tag
*/
template<typename NumericT>
void amg_coarse_ag_stage1_mis2(compressed_matrix<NumericT> const & A,
                               viennacl::linalg::detail::amg::amg_level_context & amg_context,
                               viennacl::linalg::amg_tag & tag)
{
  (void)tag;
  unsigned int  *point_types_ptr       = viennacl::linalg::host_based::detail::extract_raw_pointer<unsigned int>(amg_context.point_types_.handle());
  unsigned int *influences_row_ptr    = viennacl::linalg::host_based::detail::extract_raw_pointer<unsigned int>(amg_context.influence_jumper_.handle());
  unsigned int *influences_id_ptr     = viennacl::linalg::host_based::detail::extract_raw_pointer<unsigned int>(amg_context.influence_ids_.handle());

  std::vector<unsigned int> random_weights(A.size1());
  for (std::size_t i=0; i<random_weights.size(); ++i)
    random_weights[i] = static_cast<unsigned int>(rand()) % static_cast<unsigned int>(A.size1());

  std::size_t num_threads = 1;
#ifdef VIENNACL_WITH_OPENMP
  num_threads = omp_get_max_threads();
#endif

  viennacl::vector<unsigned int> work_state(A.size1(), viennacl::traits::context(A));
  viennacl::vector<unsigned int> work_random(A.size1(), viennacl::traits::context(A));
  viennacl::vector<unsigned int> work_index(A.size1(), viennacl::traits::context(A));

  unsigned int *work_state_ptr     = viennacl::linalg::host_based::detail::extract_raw_pointer<unsigned int>(work_state.handle());
  unsigned int *work_random_ptr    = viennacl::linalg::host_based::detail::extract_raw_pointer<unsigned int>(work_random.handle());
  unsigned int *work_index_ptr     = viennacl::linalg::host_based::detail::extract_raw_pointer<unsigned int>(work_index.handle());

  viennacl::vector<unsigned int> work_state2(A.size1(), viennacl::traits::context(A));
  viennacl::vector<unsigned int> work_random2(A.size1(), viennacl::traits::context(A));
  viennacl::vector<unsigned int> work_index2(A.size1(), viennacl::traits::context(A));

  unsigned int *work_state2_ptr     = viennacl::linalg::host_based::detail::extract_raw_pointer<unsigned int>(work_state2.handle());
  unsigned int *work_random2_ptr    = viennacl::linalg::host_based::detail::extract_raw_pointer<unsigned int>(work_random2.handle());
  unsigned int *work_index2_ptr     = viennacl::linalg::host_based::detail::extract_raw_pointer<unsigned int>(work_index2.handle());


  unsigned int num_undecided = static_cast<unsigned int>(A.size1());
  unsigned int pmis_iters = 0;

  //
  // init temporary work data:
  //
#ifdef VIENNACL_WITH_OPENMP
  #pragma omp parallel for
#endif
  for (long i2=0; i2<static_cast<long>(A.size1()); ++i2)
  {
    unsigned int i = static_cast<unsigned int>(i2);
    switch (point_types_ptr[i])
    {
    case viennacl::linalg::detail::amg::amg_level_context::POINT_TYPE_UNDECIDED: work_state_ptr[i] = 1; break;
    case viennacl::linalg::detail::amg::amg_level_context::POINT_TYPE_FINE:      work_state_ptr[i] = 0; break;
    case viennacl::linalg::detail::amg::amg_level_context::POINT_TYPE_COARSE:    work_state_ptr[i] = 2; break;
    default:
      throw std::runtime_error("Unexpected state encountered in MIS2 setup for AMG.");
    }

    //work_random_ptr[i] = random_weights[i]; // note: optimized out. Passing random_weights directly to first call to amg_coarse_ag_stage1_mis2_propagate()
    work_index_ptr[i]  = i;
  }


  //
  // Repeat PMIS iteration until all nodes have been decided
  //
  while (num_undecided > 0)
  {
    ++pmis_iters;

    //
    // Propagate maximum tuple twice
    // (first call writes state from work_*_ptr to work_*2_ptr, second call from work_*2_ptr back to work_*_ptr)
    //
    amg_coarse_ag_stage1_mis2_propagate(work_state_ptr,  &(random_weights[0]),  work_index_ptr,
                                        work_state2_ptr, work_random2_ptr, work_index2_ptr,
                                        influences_row_ptr, influences_id_ptr, long(A.size1()));
    amg_coarse_ag_stage1_mis2_propagate(work_state2_ptr, work_random2_ptr, work_index2_ptr,
                                        work_state_ptr,  work_random_ptr,  work_index_ptr,
                                        influences_row_ptr, influences_id_ptr, long(A.size1()));

    //
    // mark MIS and non-MIS nodes:
    //
    std::vector<unsigned int> thread_buffer(num_threads);

#ifdef VIENNACL_WITH_OPENMP
    #pragma omp parallel for
#endif
    for (long i2=0; i2<static_cast<long>(A.size1()); ++i2)
    {
      unsigned int i = static_cast<unsigned int>(i2);
      unsigned int max_state  = work_state_ptr[i];
      unsigned int max_index  = work_index_ptr[i];
      work_index_ptr[i] = i; // reset for next iteration

      switch (point_types_ptr[i])
      {
      case viennacl::linalg::detail::amg::amg_level_context::POINT_TYPE_UNDECIDED:
        if (i == max_index) // make this a MIS node
        {
          point_types_ptr[i] = viennacl::linalg::detail::amg::amg_level_context::POINT_TYPE_COARSE;
          work_state_ptr[i] = 2;
        }
        else if (max_state == 2) // mind the mapping of viennacl::linalg::detail::amg::amg_level_context::POINT_TYPE_COARSE above!
        {
          point_types_ptr[i] = viennacl::linalg::detail::amg::amg_level_context::POINT_TYPE_FINE;
          work_state_ptr[i] = 0;
        }
        else
        {
          work_state_ptr[i] = 1;
#ifdef VIENNACL_WITH_OPENMP
          thread_buffer[omp_get_thread_num()] += 1;
#else
          thread_buffer[0] += 1;
#endif
        }
        break;

      case viennacl::linalg::detail::amg::amg_level_context::POINT_TYPE_FINE:      work_state_ptr[i] = 0; break;
      case viennacl::linalg::detail::amg::amg_level_context::POINT_TYPE_COARSE:    work_state_ptr[i] = 2; break;
      default:
        throw std::runtime_error("Unexpected state encountered in MIS2 setup for AMG.");
      }
    }

    num_undecided = 0;
    for (std::size_t i=0; i<thread_buffer.size(); ++i)
      num_undecided += thread_buffer[i];
  } // while

  // consistency with sequential MIS: reset state for non-coarse points, so that coarse indices are correctly picked up later
#ifdef VIENNACL_WITH_OPENMP
  #pragma omp parallel for
#endif
  for (long i=0; i<static_cast<long>(A.size1()); ++i)
    if (point_types_ptr[i] != viennacl::linalg::detail::amg::amg_level_context::POINT_TYPE_COARSE)
      point_types_ptr[i] = viennacl::linalg::detail::amg::amg_level_context::POINT_TYPE_UNDECIDED;

}



/** @brief AG (aggregation based) coarsening. Partially single-threaded version (VIENNACL_AMG_COARSE_AG)
*
* @param A             Operator matrix for the respective level
* @param amg_context   AMG datastructure object for the grid hierarchy
* @param tag           AMG preconditioner tag
*/
template<typename NumericT>
void amg_coarse_ag(compressed_matrix<NumericT> const & A,
                   viennacl::linalg::detail::amg::amg_level_context & amg_context,
                   viennacl::linalg::amg_tag & tag)
{
  unsigned int *point_types_ptr       = viennacl::linalg::host_based::detail::extract_raw_pointer<unsigned int>(amg_context.point_types_.handle());
  unsigned int *influences_row_ptr    = viennacl::linalg::host_based::detail::extract_raw_pointer<unsigned int>(amg_context.influence_jumper_.handle());
  unsigned int *influences_id_ptr     = viennacl::linalg::host_based::detail::extract_raw_pointer<unsigned int>(amg_context.influence_ids_.handle());
  unsigned int *coarse_id_ptr         = viennacl::linalg::host_based::detail::extract_raw_pointer<unsigned int>(amg_context.coarse_id_.handle());

  amg_influence_trivial(A, amg_context, tag);

  //
  // Stage 1: Build aggregates:
  //
  if (tag.get_coarsening_method() == viennacl::linalg::AMG_COARSENING_METHOD_AGGREGATION)      amg_coarse_ag_stage1_sequential(A, amg_context, tag);
  if (tag.get_coarsening_method() == viennacl::linalg::AMG_COARSENING_METHOD_MIS2_AGGREGATION) amg_coarse_ag_stage1_mis2(A, amg_context, tag);

  viennacl::linalg::host_based::amg::enumerate_coarse_points(amg_context);

  //
  // Stage 2: Propagate coarse aggregate indices to neighbors:
  //
#ifdef VIENNACL_WITH_OPENMP
  #pragma omp parallel for
#endif
  for (long i2=0; i2<static_cast<long>(A.size1()); ++i2)
  {
    unsigned int i = static_cast<unsigned int>(i2);
    if (point_types_ptr[i] == viennacl::linalg::detail::amg::amg_level_context::POINT_TYPE_COARSE)
    {
      unsigned int coarse_index = coarse_id_ptr[i];

      unsigned int j_stop = influences_row_ptr[i + 1];
      for (unsigned int j = influences_row_ptr[i]; j < j_stop; ++j)
      {
        unsigned int influenced_point_id = influences_id_ptr[j];
        coarse_id_ptr[influenced_point_id] = coarse_index; // Set aggregate index for fine point

        if (influenced_point_id != i) // Note: Any write races between threads are harmless here
          point_types_ptr[influenced_point_id] = viennacl::linalg::detail::amg::amg_level_context::POINT_TYPE_FINE;
      }
    }
  }


  //
  // Stage 3: Merge remaining undecided points (merging to first aggregate found when cycling over the hierarchy
  //
#ifdef VIENNACL_WITH_OPENMP
  #pragma omp parallel for
#endif
  for (long i2=0; i2<static_cast<long>(A.size1()); ++i2)
  {
    unsigned int i = static_cast<unsigned int>(i2);
    if (point_types_ptr[i] == viennacl::linalg::detail::amg::amg_level_context::POINT_TYPE_UNDECIDED)
    {
      unsigned int j_stop = influences_row_ptr[i + 1];
      for (unsigned int j = influences_row_ptr[i]; j < j_stop; ++j)
      {
        unsigned int influenced_point_id = influences_id_ptr[j];
        if (point_types_ptr[influenced_point_id] != viennacl::linalg::detail::amg::amg_level_context::POINT_TYPE_UNDECIDED) // either coarse or fine point
        {
          //std::cout << "Setting fine node " << i << " to be aggregated with node " << *influence_iter << "/" << pointvector.get_coarse_index(*influence_iter) << std::endl;
          coarse_id_ptr[i] = coarse_id_ptr[influenced_point_id];
          break;
        }
      }
    }
  }

  //
  // Stage 4: Set undecided points to fine points (coarse ID already set in Stage 3)
  //          Note: Stage 3 and Stage 4 were initially fused, but are now split in order to avoid race conditions (or a fallback to sequential execution).
  //
#ifdef VIENNACL_WITH_OPENMP
  #pragma omp parallel for
#endif
  for (long i=0; i<static_cast<long>(A.size1()); ++i)
    if (point_types_ptr[i] == viennacl::linalg::detail::amg::amg_level_context::POINT_TYPE_UNDECIDED)
      point_types_ptr[i] = viennacl::linalg::detail::amg::amg_level_context::POINT_TYPE_FINE;

}




/** @brief Entry point and dispatcher for coarsening procedures
*
* @param A             Operator matrix for the respective level
* @param amg_context   AMG datastructure object for the grid hierarchy
* @param tag           AMG preconditioner tag
*/
template<typename MatrixT>
void amg_coarse(MatrixT & A,
                viennacl::linalg::detail::amg::amg_level_context & amg_context,
                viennacl::linalg::amg_tag & tag)
{
  switch (tag.get_coarsening_method())
  {
  case viennacl::linalg::AMG_COARSENING_METHOD_ONEPASS: amg_coarse_classic_onepass(A, amg_context, tag); break;
  case viennacl::linalg::AMG_COARSENING_METHOD_AGGREGATION:
  case viennacl::linalg::AMG_COARSENING_METHOD_MIS2_AGGREGATION: amg_coarse_ag(A, amg_context, tag); break;
  //default: throw std::runtime_error("not implemented yet");
  }
}




////////////////////////////////////// Interpolation /////////////////////////////


/** @brief Direct interpolation. Multi-threaded! (VIENNACL_AMG_INTERPOL_DIRECT)
 *
 * @param A            Operator matrix
 * @param P            Prolongation matrix
 * @param amg_context  AMG hierarchy datastructures
 * @param tag          AMG preconditioner tag
*/
template<typename NumericT>
void amg_interpol_direct(compressed_matrix<NumericT> const & A,
                         compressed_matrix<NumericT> & P,
                         viennacl::linalg::detail::amg::amg_level_context & amg_context,
                         viennacl::linalg::amg_tag & tag)
{
  NumericT     const * A_elements   = viennacl::linalg::host_based::detail::extract_raw_pointer<NumericT>(A.handle());
  unsigned int const * A_row_buffer = viennacl::linalg::host_based::detail::extract_raw_pointer<unsigned int>(A.handle1());
  unsigned int const * A_col_buffer = viennacl::linalg::host_based::detail::extract_raw_pointer<unsigned int>(A.handle2());

  unsigned int *point_types_ptr       = viennacl::linalg::host_based::detail::extract_raw_pointer<unsigned int>(amg_context.point_types_.handle());
  unsigned int *influences_row_ptr    = viennacl::linalg::host_based::detail::extract_raw_pointer<unsigned int>(amg_context.influence_jumper_.handle());
  unsigned int *influences_id_ptr     = viennacl::linalg::host_based::detail::extract_raw_pointer<unsigned int>(amg_context.influence_ids_.handle());
  unsigned int *coarse_id_ptr         = viennacl::linalg::host_based::detail::extract_raw_pointer<unsigned int>(amg_context.coarse_id_.handle());

  NumericT threshold = 1e-2;

  //
  // Step 1: Determine sparsity pattern of P:
  //
  unsigned int *P_nonzero_helper = (unsigned int*)malloc((A.size1()+1) * sizeof(unsigned int));
  NumericT *interpolation_helper = (NumericT *)malloc(A.size1() * sizeof(NumericT));

#ifdef VIENNACL_WITH_OPENMP
  #pragma omp parallel for
#endif
  for (long row2=0; row2<static_cast<long>(A.size1()); ++row2)
  {
    unsigned int row = static_cast<unsigned int>(row2);

    if (point_types_ptr[row] == viennacl::linalg::detail::amg::amg_level_context::POINT_TYPE_COARSE)
    {
      P_nonzero_helper[row] = 1;
    }
    else if (point_types_ptr[row] == viennacl::linalg::detail::amg::amg_level_context::POINT_TYPE_FINE)
    {
      NumericT row_sum = 0;
      NumericT row_coarse_sum = 0;
      NumericT diag = 0;

      // Row sum of coefficients (without diagonal) and sum of influencing coarse point coefficients has to be computed
      unsigned int row_A_start = A_row_buffer[row];
      unsigned int row_A_end   = A_row_buffer[row + 1];
      unsigned int const *influence_iter = influences_id_ptr + influences_row_ptr[row];
      unsigned int const *influence_end  = influences_id_ptr + influences_row_ptr[row + 1];
      for (unsigned int index = row_A_start; index < row_A_end; ++index)
      {
        unsigned int col = A_col_buffer[index];
        NumericT value = A_elements[index];

        if (col == row)
        {
          diag = value;
          continue;
        }
        else if (point_types_ptr[col] == viennacl::linalg::detail::amg::amg_level_context::POINT_TYPE_COARSE)
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

      if (std::fabs(temp_res) > threshold * std::fabs(diag))
      {
        unsigned int num_nonzeros_in_row = 0;

        unsigned int row_A_start = A_row_buffer[row];
        unsigned int row_A_end   = A_row_buffer[row + 1];
        unsigned int const *influence_iter = influences_id_ptr + influences_row_ptr[row];
        unsigned int const *influence_end  = influences_id_ptr + influences_row_ptr[row + 1];

        for (unsigned int index = row_A_start; index < row_A_end; ++index)
        {
          unsigned int col = A_col_buffer[index];
          if (point_types_ptr[col] != viennacl::linalg::detail::amg::amg_level_context::POINT_TYPE_COARSE)
            continue;
          NumericT value = A_elements[index];

          // Advance to correct influence metric:
          while (influence_iter != influence_end && *influence_iter < col)
            ++influence_iter;

          if (influence_iter != influence_end && *influence_iter == col && std::fabs(value) > 0)
            ++num_nonzeros_in_row;
        }

        interpolation_helper[row] = temp_res;
        P_nonzero_helper[row] = num_nonzeros_in_row; // as many nonzero entries as there are coarse points
      }
      else
      {
        interpolation_helper[row] = 0; // this ensures that the above if-statement is never valid if tested against zero
        P_nonzero_helper[row] = 0;
      }
    }
  }
  P_nonzero_helper[A.size1()] = 0;

  viennacl::vector_base<unsigned int> P_nonzero_helper_wrapper(P_nonzero_helper, viennacl::MAIN_MEMORY, A.size1() + 1);
  viennacl::linalg::exclusive_scan(P_nonzero_helper_wrapper);


  //
  // Step 2: Populate P
  //
  P = compressed_matrix<NumericT>(A.size1(), amg_context.num_coarse_, P_nonzero_helper[A.size1()], viennacl::traits::context(A));

  NumericT     * P_elements   = viennacl::linalg::host_based::detail::extract_raw_pointer<NumericT>(P.handle());
  unsigned int * P_row_buffer = viennacl::linalg::host_based::detail::extract_raw_pointer<unsigned int>(P.handle1());
  unsigned int * P_col_buffer = viennacl::linalg::host_based::detail::extract_raw_pointer<unsigned int>(P.handle2());


  // Iterate over all points to build the interpolation matrix row-by-row
  // Interpolation for coarse points is immediate using '1'.
  // Interpolation for fine points is set up via corresponding row weights (cf. Yang paper, p. 14)
#ifdef VIENNACL_WITH_OPENMP
  #pragma omp parallel for
#endif
  for (long row2=0; row2<static_cast<long>(A.size1()); ++row2)
  {
    unsigned int row = static_cast<unsigned int>(row2);

    unsigned int row_start = P_nonzero_helper[row];
    P_row_buffer[row] = row_start;

    if (point_types_ptr[row] == viennacl::linalg::detail::amg::amg_level_context::POINT_TYPE_COARSE)
    {
      P_col_buffer[row_start] = coarse_id_ptr[row];
      P_elements[row_start]   = NumericT(1);
    }
    else if (point_types_ptr[row] == viennacl::linalg::detail::amg::amg_level_context::POINT_TYPE_FINE)
    {
      NumericT temp_res = interpolation_helper[row];

      unsigned int row_A_start = A_row_buffer[row];
      unsigned int row_A_end   = A_row_buffer[row + 1];
      unsigned int const *influence_iter = influences_id_ptr + influences_row_ptr[row];
      unsigned int const *influence_end  = influences_id_ptr + influences_row_ptr[row + 1];

      if (std::fabs(temp_res) > 0)
      {
        // Iterate over all strongly influencing points to build the interpolant
        influence_iter = influences_id_ptr + influences_row_ptr[row];
        for (unsigned int index = row_A_start; index < row_A_end; ++index)
        {
          unsigned int col = A_col_buffer[index];
          if (point_types_ptr[col] != viennacl::linalg::detail::amg::amg_level_context::POINT_TYPE_COARSE)
            continue;
          NumericT value = A_elements[index];

          // Advance to correct influence metric:
          while (influence_iter != influence_end && *influence_iter < col)
            ++influence_iter;

          if (influence_iter != influence_end && *influence_iter == col && std::fabs(value) > 0)
          {
            P_col_buffer[row_start] = coarse_id_ptr[col];
            P_elements[row_start]   = temp_res * value;
            ++row_start;
          }
        }
      }

      // TODO truncate interpolation if specified by the user.
      (void)tag;
    }
    else
      throw std::runtime_error("Logic error in direct interpolation: Point is neither coarse-point nor fine-point!");
  }
  P_row_buffer[A.size1()] = P_nonzero_helper[A.size1()];

  P.generate_row_block_information();

  free(P_nonzero_helper);
  free(interpolation_helper);
}


/** @brief Extended+i interpolation. Multi-threaded! (VIENNACL_AMG_INTERPOL_EXTENDED_I)
 *
 * See "Distance-Two Interpolation for Parallel Algebraic Multigrid" by H. De Sterck et al. (2007) for description
 *
 * @param A            Operator matrix
 * @param P            Prolongation matrix
 * @param amg_context  AMG hierarchy datastructures
 * @param tag          AMG preconditioner tag
*/
template<typename NumericT>
void amg_interpol_extended_i_mis1(compressed_matrix<NumericT> const & A,
                                compressed_matrix<NumericT> & P,
                                viennacl::linalg::detail::amg::amg_level_context & amg_context,
                                viennacl::linalg::amg_tag & tag)
{
  (void)tag;

  NumericT     const * A_elements   = viennacl::linalg::host_based::detail::extract_raw_pointer<NumericT>(A.handle());
  unsigned int const * A_row_buffer = viennacl::linalg::host_based::detail::extract_raw_pointer<unsigned int>(A.handle1());
  unsigned int const * A_col_buffer = viennacl::linalg::host_based::detail::extract_raw_pointer<unsigned int>(A.handle2());

  unsigned int *point_types_ptr       = viennacl::linalg::host_based::detail::extract_raw_pointer<unsigned int>(amg_context.point_types_.handle());
  unsigned int *influences_row_ptr    = viennacl::linalg::host_based::detail::extract_raw_pointer<unsigned int>(amg_context.influence_jumper_.handle());
  unsigned int *influences_id_ptr     = viennacl::linalg::host_based::detail::extract_raw_pointer<unsigned int>(amg_context.influence_ids_.handle());
  unsigned int *coarse_id_ptr         = viennacl::linalg::host_based::detail::extract_raw_pointer<unsigned int>(amg_context.coarse_id_.handle());

  P.resize(A.size1(), amg_context.num_coarse_, false);

  std::vector<std::map<unsigned int, NumericT> > P_setup(A.size1());

  // Iterate over all points to build the interpolation matrix row-by-row
  // Interpolation for coarse points is immediate using '1'.
  // Interpolation for fine points is set up via (4.10) and (4.11) in "Distance-Two Interpolation for Parallel Algebraic Multigrid"
#ifdef VIENNACL_WITH_OPENMP
  #pragma omp parallel for
#endif
  for (long row2=0; row2<static_cast<long>(A.size1()); ++row2)
  {
    unsigned int row = static_cast<unsigned int>(row2);
    std::map<unsigned int, NumericT> & P_setup_row = P_setup[row]; // w_ij (first assembled as w_ij * \tilde{a}_ii)

    if (point_types_ptr[row] == viennacl::linalg::detail::amg::amg_level_context::POINT_TYPE_COARSE)
    {
      P_setup_row[coarse_id_ptr[row]] = NumericT(1);
    }
    else if (point_types_ptr[row] == viennacl::linalg::detail::amg::amg_level_context::POINT_TYPE_FINE)
    {
      // TODO: Add consideration of strong connections (currently all connections are considered strong)
      //unsigned int const *influence_iter = influences_id_ptr + influences_row_ptr[row];
      //unsigned int const *influence_end  = influences_id_ptr + influences_row_ptr[row + 1];

      NumericT atilde_ii = 0;

      unsigned int row_A_start = A_row_buffer[row];
      unsigned int row_A_end   = A_row_buffer[row + 1];
      for (unsigned int index = row_A_start; index < row_A_end; ++index)
      {
        unsigned int col = A_col_buffer[index];

        if (col == row)
        {
          atilde_ii += A_elements[index];  // a_ii contribution to \tilde{a}_ii
        }
        else if (point_types_ptr[col] == viennacl::linalg::detail::amg::amg_level_context::POINT_TYPE_COARSE)
        {
          P_setup_row[coarse_id_ptr[col]] += A_elements[index]; // a_ij

          // contributions from fine nodes influencing this coarse node follow below
        }
        else if (point_types_ptr[col] == viennacl::linalg::detail::amg::amg_level_context::POINT_TYPE_FINE) // reach out for coarse nodes at distance 2
        {
          NumericT aik = A_elements[index];
          NumericT akl = 0;

          unsigned int distance_2_start = A_row_buffer[col];
          unsigned int distance_2_end   = A_row_buffer[col + 1];

          // determine akl expression in denominator first:
          for (unsigned int index_2 = distance_2_start; index_2 < distance_2_end; ++index_2)
          {
            unsigned int col_2 = A_col_buffer[index_2];

            if (col_2 == row || point_types_ptr[col_2] == viennacl::linalg::detail::amg::amg_level_context::POINT_TYPE_COARSE)
              akl += A_elements[index_2];
          }

          // compute numerator and denominator of w_ij
          for (unsigned int index_2 = distance_2_start; index_2 < distance_2_end; ++index_2)
          {
            unsigned int col_2 = A_col_buffer[index_2];
            NumericT akj = 0;
            NumericT aki = 0;

            if (point_types_ptr[col_2] == viennacl::linalg::detail::amg::amg_level_context::POINT_TYPE_COARSE)
              akj  = A_elements[index_2];

            if (col_2 == row)
              aki  = A_elements[index_2];

            if (std::fabs(akj) > 0)
              P_setup_row[coarse_id_ptr[col_2]] += aik * akj / akl;
            if (std::fabs(aki) > 0)
              atilde_ii += aik * aki / akl;
          }
        }
      }

      if (atilde_ii <= 0 && atilde_ii >= 0)
        throw std::runtime_error("amg_interpol_extended_i_mis1(): Denominator zero, interpolation invalid!");

      // now divide the entries in P_setup_row by \tilde{a}_ii to form w_ij:
      // Note: A minus sign is missing in Eq. 4.10 in "Distance-Two Interpolation for Parallel Algebraic Multigrid".
      // Compare with Eq. 4.6, a minus sign appears there
      for (typename std::map<unsigned int, NumericT>::iterator it  = P_setup_row.begin();
                                                               it != P_setup_row.end();
                                                             ++it)
        it->second /= NumericT(-1.0) * atilde_ii;
    }
    else
      throw std::runtime_error("Logic error in direct interpolation: Point is neither coarse-point nor fine-point!");
  }

  // TODO: P_setup can be avoided without sacrificing parallelism.
  viennacl::tools::sparse_matrix_adapter<NumericT> P_adapter(P_setup, P.size1(), P.size2());
  viennacl::copy(P_adapter, P);

}




/** @brief AG (aggregation based) interpolation. Multi-Threaded! (VIENNACL_INTERPOL_AG)
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
                     viennacl::linalg::amg_tag & tag)
{
  (void)tag;
  P = compressed_matrix<NumericT>(A.size1(), amg_context.num_coarse_, A.size1(), viennacl::traits::context(A));

  NumericT     * P_elements   = viennacl::linalg::host_based::detail::extract_raw_pointer<NumericT>(P.handle());
  unsigned int * P_row_buffer = viennacl::linalg::host_based::detail::extract_raw_pointer<unsigned int>(P.handle1());
  unsigned int * P_col_buffer = viennacl::linalg::host_based::detail::extract_raw_pointer<unsigned int>(P.handle2());

  unsigned int *coarse_id_ptr = viennacl::linalg::host_based::detail::extract_raw_pointer<unsigned int>(amg_context.coarse_id_.handle());

  // Build interpolation matrix:
#ifdef VIENNACL_WITH_OPENMP
  #pragma omp parallel for
#endif
  for (long row2 = 0; row2 < long(A.size1()); ++row2)
  {
    unsigned int row = static_cast<unsigned int>(row2);
    P_elements[row]   = NumericT(1);
    P_row_buffer[row] = row;
    P_col_buffer[row] = coarse_id_ptr[row];
  }
  P_row_buffer[A.size1()] = static_cast<unsigned int>(A.size1()); // don't forget finalizer

  P.generate_row_block_information();
}


/** @brief Smoothed aggregation interpolation. Multi-Threaded! (VIENNACL_INTERPOL_SA)
 *
 * @param A            Operator matrix
 * @param P            Prolongation matrix
 * @param amg_context  AMG hierarchy datastructures
 * @param tag          AMG configuration tag
*/
template<typename NumericT>
void amg_interpol_sa(compressed_matrix<NumericT> const & A,
                     compressed_matrix<NumericT> & P,
                     viennacl::linalg::detail::amg::amg_level_context & amg_context,
                     viennacl::linalg::amg_tag & tag)
{
  (void)tag;
  viennacl::compressed_matrix<NumericT> P_tentative(A.size1(), amg_context.num_coarse_, A.size1(), viennacl::traits::context(A));

  // form tentative operator:
  amg_interpol_ag(A, P_tentative, amg_context, tag);

  unsigned int const * A_row_buffer = viennacl::linalg::host_based::detail::extract_raw_pointer<unsigned int>(A.handle1());
  unsigned int const * A_col_buffer = viennacl::linalg::host_based::detail::extract_raw_pointer<unsigned int>(A.handle2());
  NumericT     const * A_elements   = viennacl::linalg::host_based::detail::extract_raw_pointer<NumericT>(A.handle());

  viennacl::compressed_matrix<NumericT> Jacobi(A.size1(), A.size1(), A.nnz(), viennacl::traits::context(A));
  unsigned int * Jacobi_row_buffer = viennacl::linalg::host_based::detail::extract_raw_pointer<unsigned int>(Jacobi.handle1());
  unsigned int * Jacobi_col_buffer = viennacl::linalg::host_based::detail::extract_raw_pointer<unsigned int>(Jacobi.handle2());
  NumericT     * Jacobi_elements   = viennacl::linalg::host_based::detail::extract_raw_pointer<NumericT>(Jacobi.handle());


  // Build Jacobi matrix:
#ifdef VIENNACL_WITH_OPENMP
  #pragma omp parallel for
#endif
  for (long row2=0; row2<static_cast<long>(A.size1()); ++row2)
  {
    unsigned int row = static_cast<unsigned int>(row2);
    unsigned int row_begin = A_row_buffer[row];
    unsigned int row_end   = A_row_buffer[row+1];

    Jacobi_row_buffer[row] = row_begin;

    // Step 1: Extract diagonal:
    NumericT diag = 0;
    for (unsigned int j = row_begin; j < row_end; ++j)
    {
      if (A_col_buffer[j] == row)
      {
        diag = A_elements[j];
        break;
      }
    }

    // Step 2: Write entries:
    for (unsigned int j = row_begin; j < row_end; ++j)
    {
      unsigned int col_index = A_col_buffer[j];
      Jacobi_col_buffer[j] = col_index;

      if (col_index == row)
        Jacobi_elements[j] = NumericT(1) - NumericT(tag.get_jacobi_weight());
      else
        Jacobi_elements[j] = - NumericT(tag.get_jacobi_weight()) * A_elements[j] / diag;
    }
  }
  Jacobi_row_buffer[A.size1()] = static_cast<unsigned int>(Jacobi.nnz()); // don't forget finalizer

  P = viennacl::linalg::prod(Jacobi, P_tentative);

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
                  viennacl::linalg::amg_tag & tag)
{
  switch (tag.get_interpolation_method())
  {
  case viennacl::linalg::AMG_INTERPOLATION_METHOD_DIRECT:               amg_interpol_direct (A, P, amg_context, tag); break;
  case viennacl::linalg::AMG_INTERPOLATION_METHOD_AGGREGATION:          amg_interpol_ag     (A, P, amg_context, tag); break;
  case viennacl::linalg::AMG_INTERPOLATION_METHOD_SMOOTHED_AGGREGATION: amg_interpol_sa     (A, P, amg_context, tag); break;
  default: throw std::runtime_error("Not implemented yet!");
  }
}


/** @brief Computes B = trans(A).
  *
  * To be replaced by native functionality in ViennaCL.
  */
template<typename NumericT>
void amg_transpose(compressed_matrix<NumericT> const & A,
                   compressed_matrix<NumericT> & B)
{
  NumericT     const * A_elements   = viennacl::linalg::host_based::detail::extract_raw_pointer<NumericT>(A.handle());
  unsigned int const * A_row_buffer = viennacl::linalg::host_based::detail::extract_raw_pointer<unsigned int>(A.handle1());
  unsigned int const * A_col_buffer = viennacl::linalg::host_based::detail::extract_raw_pointer<unsigned int>(A.handle2());

  // initialize datastructures for B:
  B = compressed_matrix<NumericT>(A.size2(), A.size1(), A.nnz(), viennacl::traits::context(A));

  NumericT     * B_elements   = viennacl::linalg::host_based::detail::extract_raw_pointer<NumericT>(B.handle());
  unsigned int * B_row_buffer = viennacl::linalg::host_based::detail::extract_raw_pointer<unsigned int>(B.handle1());
  unsigned int * B_col_buffer = viennacl::linalg::host_based::detail::extract_raw_pointer<unsigned int>(B.handle2());

#ifdef VIENNACL_WITH_OPENMP
  int threads = 8;
#else
  int threads = 1;
#endif
  std::size_t scratchpad_size = threads * (B.size1()+1); //column-oriented matrix with 'threads' columns
  unsigned int * scratchpad = (unsigned int *)malloc(sizeof(unsigned int) * scratchpad_size);

  // prepare uninitialized scratchpad:
#ifdef VIENNACL_WITH_OPENMP
  #pragma omp parallel for
#endif
  for (std::size_t i = 0; i < scratchpad_size; ++i)
    scratchpad[i] = 0;

  //
  // Stage 1: Compute pattern for B
  //
#ifdef VIENNACL_WITH_OPENMP
  #pragma omp parallel for num_threads(threads)
#endif
  for (std::size_t row = 0; row < A.size1(); ++row)
  {
    unsigned int thread_id = 0;
#ifdef VIENNACL_WITH_OPENMP
    thread_id = omp_get_thread_num();
#endif

    unsigned int row_start = A_row_buffer[row];
    unsigned int row_stop  = A_row_buffer[row+1];

    for (unsigned int nnz_index = row_start; nnz_index < row_stop; ++nnz_index)
      scratchpad[thread_id * (B.size1()+1) + A_col_buffer[nnz_index]] += 1;
  }

#ifdef VIENNACL_WITH_OPENMP
  #pragma omp parallel for
#endif
  for (std::size_t row = 0; row < B.size1(); ++row)
  {
    unsigned int offset = scratchpad[row];
    for (std::size_t i = 1; i<static_cast<std::size_t>(threads); ++i)
    {
      unsigned int tmp = scratchpad[i*(B.size1()+1) + row];
      scratchpad[i*(B.size1()+1) + row] = offset;
      offset += tmp;
    }
    scratchpad[row] = offset;
  }

  //
  // Stage 2: Bring row-start array in place using exclusive-scan:
  //
  viennacl::vector_base<unsigned int> helper_vec(scratchpad, viennacl::MAIN_MEMORY, B.size1()+1);
  viennacl::linalg::host_based::exclusive_scan(helper_vec, helper_vec);

  // propagate offsets and copy CSR datastructure over to B:
#ifdef VIENNACL_WITH_OPENMP
  #pragma omp parallel for
#endif
  for (std::size_t row = 0; row < B.size1(); ++row)
  {
    unsigned int row_offset = scratchpad[row];
    B_row_buffer[row] = row_offset;
    for (std::size_t i = 1; i<std::size_t(threads); ++i)
      scratchpad[i*(B.size1()+1) + row] += row_offset;
  }
  B_row_buffer[B.size1()] = scratchpad[B.size1()];

  //
  // Stage 3: Fill with data
  //
#ifdef VIENNACL_WITH_OPENMP
  #pragma omp parallel for num_threads(threads)
#endif
  for (std::size_t row = 0; row < A.size1(); ++row)
  {
    unsigned int thread_id = 0;
#ifdef VIENNACL_WITH_OPENMP
    thread_id = omp_get_thread_num();
#endif
    //std::cout << "Row " << row << ": ";
    unsigned int row_start = A_row_buffer[row];
    unsigned int row_stop  = A_row_buffer[row+1];

    for (unsigned int nnz_index = row_start; nnz_index < row_stop; ++nnz_index)
    {
      unsigned int col_in_A = A_col_buffer[nnz_index];
      unsigned int array_index = thread_id * static_cast<unsigned int>(B.size1()+1) + col_in_A;
      unsigned int B_nnz_index = scratchpad[array_index];
      scratchpad[array_index] += 1;
      B_col_buffer[B_nnz_index] = static_cast<unsigned int>(row);
      B_elements[B_nnz_index] = A_elements[nnz_index];
    }
  }

  // Step 3: Make datastructure consistent (row blocks!)
  B.generate_row_block_information();

  free(scratchpad);
}

/** Assign sparse matrix A to dense matrix B */
template<typename NumericT, unsigned int AlignmentV>
void assign_to_dense(viennacl::compressed_matrix<NumericT, AlignmentV> const & A,
                     viennacl::matrix_base<NumericT> & B)
{
  NumericT     const * A_elements   = detail::extract_raw_pointer<NumericT>(A.handle());
  unsigned int const * A_row_buffer = detail::extract_raw_pointer<unsigned int>(A.handle1());
  unsigned int const * A_col_buffer = detail::extract_raw_pointer<unsigned int>(A.handle2());

  NumericT           * B_elements   = detail::extract_raw_pointer<NumericT>(B.handle());

#ifdef VIENNACL_WITH_OPENMP
  #pragma omp parallel for
#endif
  for (long row = 0; row < static_cast<long>(A.size1()); ++row)
  {
    unsigned int row_stop  = A_row_buffer[row+1];

    for (unsigned int nnz_index = A_row_buffer[row]; nnz_index < row_stop; ++nnz_index)
      B_elements[static_cast<unsigned int>(row) * static_cast<unsigned int>(B.internal_size2()) + A_col_buffer[nnz_index]] = A_elements[nnz_index];
  }

}

/** @brief Damped Jacobi Smoother (CUDA version)
*
* @param iterations  Number of smoother iterations
* @param A           Operator matrix for the smoothing
* @param x           The vector smoothing is applied to
* @param x_backup    (Different) Vector holding the same values as x
* @param rhs_smooth  The right hand side of the equation for the smoother
* @param weight      Damping factor. 0: No effect of smoother. 1: Undamped Jacobi iteration
*/
template<typename NumericT>
void smooth_jacobi(unsigned int iterations,
                   compressed_matrix<NumericT> const & A,
                   vector<NumericT> & x,
                   vector<NumericT> & x_backup,
                   vector<NumericT> const & rhs_smooth,
                   NumericT weight)
{

  NumericT     const * A_elements   = viennacl::linalg::host_based::detail::extract_raw_pointer<NumericT>(A.handle());
  unsigned int const * A_row_buffer = viennacl::linalg::host_based::detail::extract_raw_pointer<unsigned int>(A.handle1());
  unsigned int const * A_col_buffer = viennacl::linalg::host_based::detail::extract_raw_pointer<unsigned int>(A.handle2());
  NumericT     const * rhs_elements = viennacl::linalg::host_based::detail::extract_raw_pointer<NumericT>(rhs_smooth.handle());

  NumericT           * x_elements     = viennacl::linalg::host_based::detail::extract_raw_pointer<NumericT>(x.handle());
  NumericT     const * x_old_elements = viennacl::linalg::host_based::detail::extract_raw_pointer<NumericT>(x_backup.handle());

  for (unsigned int i=0; i<iterations; ++i)
  {
    x_backup = x;

    #ifdef VIENNACL_WITH_OPENMP
    #pragma omp parallel for
    #endif
    for (long row2 = 0; row2 < static_cast<long>(A.size1()); ++row2)
    {
      unsigned int row = static_cast<unsigned int>(row2);
      unsigned int col_end   = A_row_buffer[row+1];

      NumericT sum  = NumericT(0);
      NumericT diag = NumericT(1);
      for (unsigned int index = A_row_buffer[row]; index != col_end; ++index)
      {
        unsigned int col = A_col_buffer[index];
        if (col == row)
          diag = A_elements[index];
        else
          sum += A_elements[index] * x_old_elements[col];
      }

      x_elements[row] = weight * (rhs_elements[row] - sum) / diag + (NumericT(1) - weight) * x_old_elements[row];
    }
  }
}

} //namespace amg
} //namespace host_based
} //namespace linalg
} //namespace viennacl

#endif
