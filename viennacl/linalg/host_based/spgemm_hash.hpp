#ifndef VIENNACL_LINALG_HOST_BASED_SPGEMM_HASH_HPP_
#define VIENNACL_LINALG_HOST_BASED_SPGEMM_HASH_HPP_

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

/** @file viennacl/linalg/host_based/sparse_matrix_operations.hpp
    @brief Implementations of operations using sparse matrices on the CPU using a single thread or OpenMP.
*/

#include "viennacl/forwards.h"
#include "viennacl/linalg/host_based/common.hpp"

namespace viennacl
{
namespace linalg
{
namespace host_based
{

/** @brief Non-monotonic hash function based on modulo operation with respect to the number of elements expected times a safety factor
*
*  Although very simple, it's fast and likely to perform well for most structured and unstructured matrices.
*  Few corner-cases where this hash works poorly are possible, but can be addressed via an adjustment of the expansion factor.
*/
template<typename IndexT>
class spgemm_unordered_hash
{
public:
  spgemm_unordered_hash() {}

  spgemm_unordered_hash(IndexT max_elements, IndexT exp_factor = 3)
    : hash_max_(exp_factor * max_elements),
      min_map_size_(hash_max_ + max_elements)
  {}

  IndexT min_map_size() const { return min_map_size_; }

  IndexT operator()(IndexT in) const { return in % hash_max_; }

private:
  IndexT hash_max_;
  IndexT min_map_size_;
};


/** @brief Monotonic hash function for an a-priori known number of elements from a known range of indices
*
*  Best performance if the indices are evenly distributed.
*  Corner-case with poor performance: One outlier, all other indices clustered.
*  Since this hash is only used for merging a large number of rows from B, it is unlikely that all rows are clustered around the same hash bucket.
*/
template<typename IndexT>
class spgemm_ordered_hash
{
public:
  spgemm_ordered_hash() {}

  spgemm_ordered_hash(IndexT max_elements, IndexT min_index, IndexT max_index, IndexT exp_factor = 3)
    : hash_max_(exp_factor * max_elements),
      min_map_size_(hash_max_ + max_elements),
      min_element_(min_index),
      value_range_(double(max_index - min_index))
  {}

  IndexT min_map_size() const { return min_map_size_; }

  // Note: integer arithmetic is likely to overflow, hence using floating point arithmetic for hash function
  IndexT operator()(IndexT in) const
  {
    assert(in >= min_element_ && bool("Calculation of minimum element for hash incorrect!"));
    assert(in - min_element_ <= value_range_ && bool("Calculation of maximum element for hash incorrect!"));
    return static_cast<IndexT>(double(hash_max_) * double(in - min_element_) / value_range_); }

private:
  IndexT hash_max_;
  IndexT min_map_size_;
  IndexT min_element_;
  double value_range_;
};




template<typename IndexT>
struct hash_element_index
{
  typedef IndexT   IndexType;

  IndexT index;
};

template<typename IndexT>
static inline void spgemm_hash_merge(hash_element_index<IndexT> &, hash_element_index<IndexT> const &) {}


template<typename IndexT, typename NumericT>
struct hash_element_index_value
{
  typedef IndexT   IndexType;

  IndexT   index;
  NumericT value;
};

template<typename IndexT, typename NumericT>
static inline void spgemm_hash_merge(hash_element_index_value<IndexT, NumericT> & a, hash_element_index_value<IndexT, NumericT> const & b) { a.value += b.value; }

/** @brief Performance-optimized hash-map for spGEMM.
*
*  Flat in memory, using knowledge of the maximum number of elements in the map and their index range.
*/
template<typename HashElementT, typename HashFunctorT>
class spgemm_hash_map
{

public:
  typedef HashElementT                        value_type;
  typedef HashFunctorT                        hash_type;
  typedef typename HashElementT::IndexType    IndexType;

  spgemm_hash_map(unsigned int max_element_index) : buffer_(NULL), size_(0), invalid_item_index_(max_element_index), alloc_size_(0) {}

  ~spgemm_hash_map() { free(buffer_); }

  void reset(HashFunctorT new_hash)
  {
    hash_ = new_hash;

    // check hash size and reallocate if necessary:
    if (alloc_size_ < hash_.min_map_size() + 1)
    {
      free(buffer_);
      alloc_size_ = static_cast<unsigned int>(static_cast<double>(hash_.min_map_size()) * 1.5); // some extra space in order to avoid reallocations if max_num_elements grows slowly
      if (alloc_size_ < 100) // we don't want buffers which are too small, because they are likely to reallocated later anyway
        alloc_size_ = 100;
      buffer_ = (HashElementT*)malloc(sizeof(HashElementT)*alloc_size_);
    }

    // set all hash entries to invalid:
    HashElementT* buffer_end = buffer_ + hash_.min_map_size() + 1;
    for (HashElementT* buffer_ptr=buffer_; buffer_ptr != buffer_end; ++buffer_ptr)
      buffer_ptr->index = invalid_item_index_;
    size_ = 0;
  }

  void insert(HashElementT new_item)
  {
    IndexType map_index = hash_(new_item.index);

    assert(map_index < hash_.min_map_size() && bool("Hash function buggy: Computed index too large!"));

    HashElementT *buffer_element_ptr = buffer_ + map_index;
    while (   buffer_element_ptr->index != invalid_item_index_
           && buffer_element_ptr->index != new_item.index) // search for match or next free element
      ++buffer_element_ptr;

    // merge if already in map:
    if (buffer_element_ptr->index == new_item.index)
    {
      spgemm_hash_merge(*buffer_element_ptr, new_item);
    }
    else // insert new element
    {
      *buffer_element_ptr = new_item;
      ++size_;
    }
  }

  unsigned int       size() const { return size_; }
  unsigned int alloc_size() const { return alloc_size_; }
  IndexType    invalid_item_index() const { return invalid_item_index_; }

  HashElementT const * raw_data() const { return buffer_; }

private:
  HashElementT *buffer_;
  unsigned int size_;
  HashFunctorT hash_;
  IndexType    invalid_item_index_;
  unsigned int alloc_size_;
};


template<typename HashMapT>
unsigned int row_C_scan_hash(unsigned int max_entries_C,
                             unsigned int row_start_A, unsigned int row_end_A, unsigned int const *A_col_buffer,
                             unsigned int const *B_row_buffer, unsigned int const *B_col_buffer,
                             HashMapT & hash_map)
{
  typedef typename HashMapT::value_type  HashMapElement;
  typedef typename HashMapT::hash_type   HashType;

  // Trivial case: row length 0:
  if (row_start_A == row_end_A)
    return 0;

  // Trivial case: row length 1:
  if (row_end_A - row_start_A == 1)
  {
    unsigned int A_col = A_col_buffer[row_start_A];
    return B_row_buffer[A_col + 1] - B_row_buffer[A_col];
  }

  // all other row lengths:
  HashMapElement item;
  hash_map.reset(HashType(max_entries_C));
  while (row_end_A > row_start_A)     // process row by row:
  {
    unsigned int A_col = A_col_buffer[row_start_A];
    unsigned int row_B_end = B_row_buffer[A_col + 1];
    for (unsigned int j = B_row_buffer[A_col]; j < row_B_end; ++j)
    {
      item.index = B_col_buffer[j];
      hash_map.insert(item);
    }

    ++row_start_A;
  }

  return hash_map.size();
}

//////////////////////////////

template<typename NumericT, typename HashMapT>
void row_C_compute_hash(unsigned int row_start_A, unsigned int row_end_A, unsigned int const *A_col_buffer, NumericT const *A_elements,
                        unsigned int const *B_row_buffer, unsigned int const *B_col_buffer, NumericT const *B_elements, unsigned int B_size2,
                        unsigned int row_start_C, unsigned int row_end_C, unsigned int *C_col_buffer, NumericT *C_elements,
                        HashMapT & hash_map)
{
  typedef typename HashMapT::value_type  HashMapElement;
  typedef typename HashMapT::hash_type   HashType;

  // Trivial case: row length 0:
  if (row_start_A == row_end_A)
    return;

  // Trivial case: row length 1:
  if (row_end_A - row_start_A == 1)
  {
    unsigned int A_col = A_col_buffer[row_start_A];
    unsigned int B_end = B_row_buffer[A_col + 1];
    NumericT A_value   = A_elements[row_start_A];
    C_col_buffer += row_start_C;
    C_elements += row_start_C;
    for (unsigned int j = B_row_buffer[A_col]; j < B_end; ++j, ++C_col_buffer, ++C_elements)
    {
      *C_col_buffer = B_col_buffer[j];
      *C_elements = A_value * B_elements[j];
    }
    return;
  }

  // all other row lengths:

  // Step 1: Determine minimum and maximum index (it's essential that indices in B are ordered)
  unsigned int min_index = B_size2;
  unsigned int max_index = 0;
  for (unsigned int j = row_start_A; j < row_end_A; ++j)
  {
    unsigned int A_col   = A_col_buffer[j];
    unsigned int B_first = B_row_buffer[A_col];
    unsigned int B_last  = B_row_buffer[A_col+1];
    if (B_last > B_first)
    {
      min_index = std::min(min_index, B_col_buffer[B_first]);
      max_index = std::max(max_index, B_col_buffer[B_last-1]);
    }
  }

  if (min_index > max_index)  // all relevant rows in B are empty, so there's nothing to do
    return;

  // Step 2:
  HashMapElement item;
  hash_map.reset(HashType(row_end_C - row_start_C, min_index, max_index));
  while (row_end_A > row_start_A)     // process row by row:
  {
    unsigned int A_col = A_col_buffer[row_start_A];
    NumericT   A_value = A_elements[row_start_A];
    unsigned int row_B_end = B_row_buffer[A_col + 1];
    for (unsigned int j = B_row_buffer[A_col]; j < row_B_end; ++j)
    {
      item.index = B_col_buffer[j];
      item.value = A_value * B_elements[j];
      hash_map.insert(item);
    }

    ++row_start_A;
  }

  // copy from hash to result:
  assert(hash_map.size() == row_end_C - row_start_C && bool("Number of items in hashmap does not match preallocated size of row in C!"));
  unsigned int items_copied = 0;
  HashMapElement const *hash_ptr = hash_map.raw_data();
  C_col_buffer += row_start_C;
  C_elements   += row_start_C;
  while (items_copied < hash_map.size())
  {
    if (hash_ptr->index < hash_map.invalid_item_index())
    {
      *C_col_buffer = hash_ptr->index; ++C_col_buffer;
      *C_elements   = hash_ptr->value; ++C_elements;
      ++items_copied;
    }
    ++hash_ptr;
  }
}

} // namespace host_based
} //namespace linalg
} //namespace viennacl


#endif
