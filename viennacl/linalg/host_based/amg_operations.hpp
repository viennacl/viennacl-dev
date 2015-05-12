#ifndef VIENNACL_LINALG_HOST_BASED_AMG_OPERATIONS_HPP
#define VIENNACL_LINALG_HOST_BASED_AMG_OPERATIONS_HPP

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

/** @file host_based/amg_operations.hpp
    @brief Implementations of routines for AMG using the CPU on the host (with OpenMP if enabled).
*/

#include <cstdlib>
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
namespace host_based
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
  (void)tag;

  unsigned int const * A_row_buffer = viennacl::linalg::host_based::detail::extract_raw_pointer<unsigned int>(A.handle1());
  unsigned int const * A_col_buffer = viennacl::linalg::host_based::detail::extract_raw_pointer<unsigned int>(A.handle2());

  unsigned int *influences_row_ptr = viennacl::linalg::host_based::detail::extract_raw_pointer<unsigned int>(amg_context.influence_jumper_.handle());
  unsigned int *influences_id_ptr  = viennacl::linalg::host_based::detail::extract_raw_pointer<unsigned int>(amg_context.influence_ids_.handle());
  unsigned int *influences_values_ptr  = viennacl::linalg::host_based::detail::extract_raw_pointer<unsigned int>(amg_context.influence_values_.handle());

  for (std::size_t i=0; i<=A.size1(); ++i)
    influences_row_ptr[i] = A_row_buffer[i];

  for (std::size_t i=0; i<A.size1(); ++i)
    influences_values_ptr[i] = influences_row_ptr[i+1] - influences_row_ptr[i];

#ifdef VIENNACL_WITH_OPENMP
  #pragma omp parallel for
#endif
  for (std::size_t i=0; i<A.nnz(); ++i)
    influences_id_ptr[i] = A_col_buffer[i];
}


/** @brief Routine for extracting strongly connected points considering a user-provided threshold value */
template<typename NumericT>
void amg_influence_advanced(compressed_matrix<NumericT> const & A,
                            viennacl::linalg::detail::amg::amg_level_context & amg_context,
                            viennacl::linalg::detail::amg::amg_tag & tag)
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
  for (std::size_t i=0; i<A.size1(); ++i)
  {
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

      if (   (diag > 0 && diag * value <= tag.get_threshold() * diag * largest_negative)
          || (diag < 0 && diag * value <= tag.get_threshold() * diag * largest_positive))
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
    unsigned int *influences_id_write_ptr = influences_id_ptr + influences_row_ptr[i];
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
                   viennacl::linalg::detail::amg::amg_tag & tag)
{
  // TODO: dispatch based on influence tolerance provided
  amg_influence_trivial(A, amg_context, tag);
}



/** @brief Assign IDs to coarse points */
inline void enumerate_coarse_points(viennacl::linalg::detail::amg::amg_level_context & amg_context)
{
  char         *point_types_ptr  = viennacl::linalg::host_based::detail::extract_raw_pointer<char>(amg_context.point_types_.handle());
  unsigned int *coarse_id_ptr    = viennacl::linalg::host_based::detail::extract_raw_pointer<unsigned int>(amg_context.coarse_id_.handle());

  unsigned int coarse_id = 0;
  for (vcl_size_t i=0; i<amg_context.coarse_id_.size(); ++i)
  {
    //assert(point_types_ptr[i] != viennacl::linalg::detail::amg::amg_level_context::POINT_TYPE_UNDECIDED && bool("Logic error in enumerate_coarse_points(): Undecided points detected!"));

    if (point_types_ptr[i] == viennacl::linalg::detail::amg::amg_level_context::POINT_TYPE_COARSE)
      coarse_id_ptr[i] = coarse_id++;
  }

  //std::cout << "Coarse nodes after enumerate_coarse_points(): " << coarse_id << std::endl;
  amg_context.num_coarse_ = coarse_id;
}




//////////////////////////////////////


/** @brief Helper struct for sequential classical one-pass coarsening */
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
template<typename NumericT>
void amg_coarse_classic_onepass(compressed_matrix<NumericT> const & A,
                                viennacl::linalg::detail::amg::amg_level_context & amg_context,
                                viennacl::linalg::detail::amg::amg_tag & tag)
{
  char         *point_types_ptr       = viennacl::linalg::host_based::detail::extract_raw_pointer<char>(amg_context.point_types_.handle());
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


//////////////////////////


/** @brief AG (aggregation based) coarsening, single-threaded version of stage 1
* @param level         Course level identifier
* @param A             Operator matrix on all levels
* @param pointvector   Vector of points on all levels
* @param tag           AMG preconditioner tag
*/
template<typename NumericT>
void amg_coarse_ag_stage1_sequential(compressed_matrix<NumericT> const & A,
                                     viennacl::linalg::detail::amg::amg_level_context & amg_context,
                                     viennacl::linalg::detail::amg::amg_tag & tag)
{
  char         *point_types_ptr       = viennacl::linalg::host_based::detail::extract_raw_pointer<char>(amg_context.point_types_.handle());
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



/** @brief AG (aggregation based) coarsening, single-threaded version of stage 1
* @param level         Course level identifier
* @param A             Operator matrix on all levels
* @param pointvector   Vector of points on all levels
* @param tag           AMG preconditioner tag
*/
template<typename NumericT>
void amg_coarse_ag_stage1_mis2(compressed_matrix<NumericT> const & A,
                               viennacl::linalg::detail::amg::amg_level_context & amg_context,
                               viennacl::linalg::detail::amg::amg_tag & tag)
{
  char         *point_types_ptr       = viennacl::linalg::host_based::detail::extract_raw_pointer<char>(amg_context.point_types_.handle());
  unsigned int *influences_row_ptr    = viennacl::linalg::host_based::detail::extract_raw_pointer<unsigned int>(amg_context.influence_jumper_.handle());
  unsigned int *influences_id_ptr     = viennacl::linalg::host_based::detail::extract_raw_pointer<unsigned int>(amg_context.influence_ids_.handle());

  std::vector<unsigned int> random_weights(A.size1());
  for (std::size_t i=0; i<random_weights.size(); ++i)
    random_weights[i] = static_cast<unsigned int>(rand()) % static_cast<unsigned int>(A.size1());

  std::size_t num_threads = 1;
#ifdef VIENNACL_WITH_OPENMP
  num_threads = omp_get_max_threads();
#endif

  viennacl::vector<char>         work_state(A.size1(), viennacl::traits::context(A));
  viennacl::vector<unsigned int> work_random(A.size1(), viennacl::traits::context(A));
  viennacl::vector<unsigned int> work_index(A.size1(), viennacl::traits::context(A));

  char         *work_state_ptr     = viennacl::linalg::host_based::detail::extract_raw_pointer<char>(work_state.handle());
  unsigned int *work_random_ptr    = viennacl::linalg::host_based::detail::extract_raw_pointer<unsigned int>(work_random.handle());
  unsigned int *work_index_ptr     = viennacl::linalg::host_based::detail::extract_raw_pointer<unsigned int>(work_index.handle());

  viennacl::vector<char>         work_state2(A.size1(), viennacl::traits::context(A));
  viennacl::vector<unsigned int> work_random2(A.size1(), viennacl::traits::context(A));
  viennacl::vector<unsigned int> work_index2(A.size1(), viennacl::traits::context(A));

  char         *work_state2_ptr     = viennacl::linalg::host_based::detail::extract_raw_pointer<char>(work_state2.handle());
  unsigned int *work_random2_ptr    = viennacl::linalg::host_based::detail::extract_raw_pointer<unsigned int>(work_random2.handle());
  unsigned int *work_index2_ptr     = viennacl::linalg::host_based::detail::extract_raw_pointer<unsigned int>(work_index2.handle());


  unsigned int num_undecided = static_cast<unsigned int>(A.size1());
  while (num_undecided > 0)
  {
    //
    // init temporary work data:
    //
#ifdef VIENNACL_WITH_OPENMP
  #pragma omp parallel for
#endif
    for (unsigned int i = 0; i<static_cast<unsigned int>(A.size1()); ++i)
    {
      switch (point_types_ptr[i])
      {
      case viennacl::linalg::detail::amg::amg_level_context::POINT_TYPE_UNDECIDED: work_state_ptr[i] = 1; break;
      case viennacl::linalg::detail::amg::amg_level_context::POINT_TYPE_FINE:      work_state_ptr[i] = 0; break;
      case viennacl::linalg::detail::amg::amg_level_context::POINT_TYPE_COARSE:    work_state_ptr[i] = 2; break;
      default:
        throw std::runtime_error("Unexpected state encountered in MIS2 setup for AMG.");
      }

      work_random_ptr[i] = random_weights[i];
      work_index_ptr[i]  = i;
    }


    //
    // Propagate maximum tuple twice
    //
    for (unsigned int r = 0; r < 2; ++r)
    {
      // max operation
#ifdef VIENNACL_WITH_OPENMP
      #pragma omp parallel for
#endif
      for (unsigned int i = 0; i<static_cast<unsigned int>(A.size1()); ++i)
      {
        // load
        char         state  = work_state_ptr[i];
        unsigned int random = work_random_ptr[i];
        unsigned int index  = work_index_ptr[i];

        // max
        unsigned int j_stop = influences_row_ptr[i + 1];
        for (unsigned int j = influences_row_ptr[i]; j < j_stop; ++j)
        {
          unsigned int influenced_point_id = influences_id_ptr[j];

          // lexigraphical triple-max (not particularly pretty, but does the job):
          if (state < work_state_ptr[influenced_point_id])
          {
            state  = work_state_ptr[influenced_point_id];
            random = work_random_ptr[influenced_point_id];
            index  = work_index_ptr[influenced_point_id];
          }
          else if (state == work_state_ptr[influenced_point_id])
          {
            if (random < work_random_ptr[influenced_point_id])
            {
              state  = work_state_ptr[influenced_point_id];
              random = work_random_ptr[influenced_point_id];
              index  = work_index_ptr[influenced_point_id];
            }
            else if (random == work_random_ptr[influenced_point_id])
            {
              if (index < work_index_ptr[influenced_point_id])
              {
                state  = work_state_ptr[influenced_point_id];
                random = work_random_ptr[influenced_point_id];
                index  = work_index_ptr[influenced_point_id];
              }
            } // max(random)
          } // max(state)
        } // for

        // store
        work_state2_ptr[i]  = state;
        work_random2_ptr[i] = random;
        work_index2_ptr[i]  = index;
      }

      // copy work array
#ifdef VIENNACL_WITH_OPENMP
      #pragma omp parallel for
#endif
      for (unsigned int i = 0; i<static_cast<unsigned int>(A.size1()); ++i)
      {
        work_state_ptr[i]  = work_state2_ptr[i];
        work_random_ptr[i] = work_random2_ptr[i];
        work_index_ptr[i]  = work_index2_ptr[i];
      }
    }

    //
    // mark MIS and non-MIS nodes:
    //
    std::vector<unsigned int> thread_buffer(num_threads);

#ifdef VIENNACL_WITH_OPENMP
    #pragma omp parallel for
#endif
    for (unsigned int i = 0; i<static_cast<unsigned int>(A.size1()); ++i)
    {
      char         max_state  = work_state_ptr[i];
      unsigned int max_index  = work_index_ptr[i];

      if (point_types_ptr[i] == viennacl::linalg::detail::amg::amg_level_context::POINT_TYPE_UNDECIDED)
      {
        if (i == max_index) // make this a MIS node
          point_types_ptr[i] = viennacl::linalg::detail::amg::amg_level_context::POINT_TYPE_COARSE;
        else if (max_state == viennacl::linalg::detail::amg::amg_level_context::POINT_TYPE_COARSE)
          point_types_ptr[i] = viennacl::linalg::detail::amg::amg_level_context::POINT_TYPE_FINE;
        else
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
  }
}



/** @brief AG (aggregation based) coarsening. Partially single-threaded version (VIENNACL_AMG_COARSE_AG)
* @param level         Course level identifier
* @param A             Operator matrix on all levels
* @param pointvector   Vector of points on all levels
* @param tag           AMG preconditioner tag
*/
template<typename NumericT>
void amg_coarse_ag(compressed_matrix<NumericT> const & A,
                   viennacl::linalg::detail::amg::amg_level_context & amg_context,
                   viennacl::linalg::detail::amg::amg_tag & tag)
{
  char         *point_types_ptr       = viennacl::linalg::host_based::detail::extract_raw_pointer<char>(amg_context.point_types_.handle());
  unsigned int *influences_row_ptr    = viennacl::linalg::host_based::detail::extract_raw_pointer<unsigned int>(amg_context.influence_jumper_.handle());
  unsigned int *influences_id_ptr     = viennacl::linalg::host_based::detail::extract_raw_pointer<unsigned int>(amg_context.influence_ids_.handle());
  unsigned int *coarse_id_ptr         = viennacl::linalg::host_based::detail::extract_raw_pointer<unsigned int>(amg_context.coarse_id_.handle());

  amg_influence_trivial(A, amg_context, tag);

  //
  // Stage 1: Build aggregates:
  //
  if (tag.get_coarse() == VIENNACL_AMG_COARSE_AG)      amg_coarse_ag_stage1_sequential(A, amg_context, tag);
  if (tag.get_coarse() == VIENNACL_AMG_COARSE_AG_MIS2) amg_coarse_ag_stage1_mis2(A, amg_context, tag);

  viennacl::linalg::host_based::amg::enumerate_coarse_points(amg_context);

  //
  // Stage 2: Propagate coarse aggregate indices to neighbors:
  //
#ifdef VIENNACL_WITH_OPENMP
  #pragma omp parallel for
#endif
  for (unsigned int i = 0; i<static_cast<unsigned int>(A.size1()); ++i)
  {
    if (point_types_ptr[i] == viennacl::linalg::detail::amg::amg_level_context::POINT_TYPE_COARSE)
    {
      unsigned int coarse_index = coarse_id_ptr[i];

      unsigned int j_stop = influences_row_ptr[i + 1];
      for (unsigned int j = influences_row_ptr[i]; j < j_stop; ++j)
      {
        unsigned int influenced_point_id = influences_id_ptr[j];
        coarse_id_ptr[influenced_point_id] = coarse_index; // Set aggregate index for fine point
      }
    }
  }


  //
  // Stage 3: Merge remaining undecided points (merging to first aggregate found when cycling over the hierarchy
  //
#ifdef VIENNACL_WITH_OPENMP
  #pragma omp parallel for
#endif
  for (unsigned int i = 0; i<static_cast<unsigned int>(A.size1()); ++i)
  {
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
  for (unsigned int i = 0; i<static_cast<unsigned int>(A.size1()); ++i)
    if (point_types_ptr[i] == viennacl::linalg::detail::amg::amg_level_context::POINT_TYPE_UNDECIDED)
      point_types_ptr[i] = viennacl::linalg::detail::amg::amg_level_context::POINT_TYPE_FINE;



}




/** @brief Calls the right coarsening procedure
* @param level        Coarse level identifier
* @param A            Operator matrix on all levels
* @param pointvector  Vector of points on all levels
* @param slicing      Partitioning of the system matrix to different processors (only used in RS0 and RS3)
* @param tag          AMG preconditioner tag
*/
template<typename InternalT1>
void amg_coarse(InternalT1 & A,
                viennacl::linalg::detail::amg::amg_level_context & amg_context,
                viennacl::linalg::detail::amg::amg_tag & tag)
{
  switch (tag.get_coarse())
  {
  case VIENNACL_AMG_COARSE_ONEPASS: amg_coarse_classic_onepass(A, amg_context, tag); break;
  case VIENNACL_AMG_COARSE_AG:
  case VIENNACL_AMG_COARSE_AG_MIS2: amg_coarse_ag(A, amg_context, tag); break;
  default: throw std::runtime_error("not implemented yet");
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
                         viennacl::linalg::detail::amg::amg_tag & tag)
{
  NumericT     const * A_elements   = viennacl::linalg::host_based::detail::extract_raw_pointer<NumericT>(A.handle());
  unsigned int const * A_row_buffer = viennacl::linalg::host_based::detail::extract_raw_pointer<unsigned int>(A.handle1());
  unsigned int const * A_col_buffer = viennacl::linalg::host_based::detail::extract_raw_pointer<unsigned int>(A.handle2());

  char         *point_types_ptr       = viennacl::linalg::host_based::detail::extract_raw_pointer<char>(amg_context.point_types_.handle());
  unsigned int *influences_row_ptr    = viennacl::linalg::host_based::detail::extract_raw_pointer<unsigned int>(amg_context.influence_jumper_.handle());
  unsigned int *influences_id_ptr     = viennacl::linalg::host_based::detail::extract_raw_pointer<unsigned int>(amg_context.influence_ids_.handle());
  unsigned int *coarse_id_ptr         = viennacl::linalg::host_based::detail::extract_raw_pointer<unsigned int>(amg_context.coarse_id_.handle());

  P.resize(A.size1(), amg_context.num_coarse_, false);

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
    //std::cout << "Row " << row << ": " << std::endl;
    if (point_types_ptr[row] == viennacl::linalg::detail::amg::amg_level_context::POINT_TYPE_COARSE)
    {
      //std::cout << "  Setting value 1.0 at " << coarse_id_ptr[row] << std::endl;
      P_setup_row[coarse_id_ptr[row]] = NumericT(1);
    }
    else if (point_types_ptr[row] == viennacl::linalg::detail::amg::amg_level_context::POINT_TYPE_FINE)
    {
      //std::cout << "Building interpolant for fine point " << row << std::endl;

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
      //std::cout << "row_sum: " << row_sum << ", row_coarse_sum: " << row_coarse_sum << ", diag: " << diag << std::endl;

      if (std::fabs(temp_res) > 1e-2 * std::fabs(diag))
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

          if (influence_iter != influence_end && *influence_iter == col)
          {
            //std::cout << " Setting entry "  << temp_res * value << " at " << coarse_id_ptr[col] << " for point " << col << std::endl;
            P_setup_row[coarse_id_ptr[col]] = temp_res * value;
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

  NumericT     * P_elements   = viennacl::linalg::host_based::detail::extract_raw_pointer<NumericT>(P.handle());
  unsigned int * P_row_buffer = viennacl::linalg::host_based::detail::extract_raw_pointer<unsigned int>(P.handle1());
  unsigned int * P_col_buffer = viennacl::linalg::host_based::detail::extract_raw_pointer<unsigned int>(P.handle2());

  unsigned int *coarse_id_ptr = viennacl::linalg::host_based::detail::extract_raw_pointer<unsigned int>(amg_context.coarse_id_.handle());

  // Build interpolation matrix:
#ifdef VIENNACL_WITH_OPENMP
  #pragma omp parallel for
#endif
  for (unsigned int row = 0; row<A.size1(); ++row)
  {
    P_elements[row]   = NumericT(1);
    P_row_buffer[row] = row;
    P_col_buffer[row] = coarse_id_ptr[row];
  }
  P_row_buffer[A.size1()] = A.size1(); // don't forget finalizer

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
  case VIENNACL_AMG_INTERPOL_DIRECT:  amg_interpol_direct (A, P, amg_context, tag); break;
  //case VIENNACL_AMG_INTERPOL_CLASSIC: amg_interpol_classic(level, A, P, pointvector, tag); break;
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
