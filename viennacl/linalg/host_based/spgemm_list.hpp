#ifndef VIENNACL_LINALG_HOST_BASED_SPGEMM_LIST_HPP_
#define VIENNACL_LINALG_HOST_BASED_SPGEMM_LIST_HPP_

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


/** @brief Highly optimized single-linked list implementation for SpGEMM.
 *
 *  Don't try to 'generalize' unless you know what you're doing. Performance is important here.
 */
template<typename T>
struct sp_gemm_list_value_item
{
  sp_gemm_list_value_item(unsigned int next, unsigned int col, T val) : next_idx(next), col_idx(col), value(val) {}

  unsigned int next_idx;
  unsigned int col_idx;
  T value;
};

struct sp_gemm_list_item
{
  template<typename T>
  sp_gemm_list_item(unsigned int next, unsigned int col, T) : next_idx(next), col_idx(col) {}
  sp_gemm_list_item(unsigned int next, unsigned int col) : next_idx(next), col_idx(col) {}

  unsigned int next_idx;
  unsigned int col_idx;
};

template<typename ItemT>
class sp_gemm_list
{
public:
  sp_gemm_list() : buffer_(NULL), buffer_alloc_size_(0) {}

  ~sp_gemm_list() { free(buffer_); }

  void reserve_and_reset(std::size_t new_size)
  {
    if (new_size > buffer_alloc_size_)
    {
      free(buffer_);
      buffer_ = (ItemT*)malloc(sizeof(ItemT)*new_size);
      buffer_alloc_size_ = static_cast<unsigned int>(new_size);
    }
    buffer_head_ = buffer_alloc_size_;
    buffer_size_ = 0;
    previous_index_ = buffer_alloc_size_;
    current_index_  = buffer_head_;
  }

  unsigned int size() const { return buffer_size_; }
  unsigned int alloc_size() const { return buffer_alloc_size_; }
  unsigned int current_index() const { return current_index_; }
  bool is_valid() const { return current_index_ < buffer_alloc_size_; }

  void rewind()
  {
    previous_index_ = buffer_alloc_size_;
    current_index_  = buffer_head_;
  }

  void advance()
  {
    assert(is_valid() && bool("Incrementing spgemm_list beyond end!"));
    previous_index_ = current_index_;
    current_index_ = buffer_[previous_index_].next_idx;
  }

  /** @brief Insert an element in front of the current location */
  template<typename T>
  void insert_before_current(unsigned int new_index, T val)
  {
    if (current_index_ == buffer_alloc_size_) // insert at end of list
    {
      if (previous_index_ == buffer_alloc_size_) // make it first element in list
        buffer_head_ = buffer_size_;
      else
        buffer_[previous_index_].next_idx = buffer_size_;
      current_index_ = buffer_size_;
      buffer_[current_index_] = ItemT(buffer_alloc_size_, new_index, val);
    }
    else //insert inside list
    {
      if (previous_index_ == buffer_alloc_size_) // insert at beginning
      {
        buffer_head_ = buffer_size_;
        buffer_[buffer_head_] = ItemT(current_index_, new_index, val);
        current_index_ = buffer_head_;
      }
      else
      {
        buffer_[previous_index_].next_idx = buffer_size_;
        buffer_[buffer_size_] = ItemT(current_index_, new_index, val);
        current_index_ = buffer_size_;
      }
    }
    ++buffer_size_;
  }

  void insert_before_current(unsigned int new_index) { insert_before_current(new_index, double(0)); }


  ItemT const & item() const { return buffer_[current_index_]; }
  ItemT       & item()       { return buffer_[current_index_]; }

private:
  ItemT *buffer_;
  unsigned int buffer_alloc_size_;
  unsigned int buffer_head_;
  unsigned int buffer_size_;

  unsigned int previous_index_;
  unsigned int current_index_;
};

template<typename RowListT>
void merge_two_rows(unsigned int j,
                    unsigned int const *A_col_buffer,
                    unsigned int const *B_row_buffer,
                    unsigned int const *B_col_buffer,
                    RowListT & row_C_list,
                    unsigned int C_size2)
{
  unsigned int row_index_B_0 = A_col_buffer[j];
  unsigned int row_start_B_0 = B_row_buffer[row_index_B_0];
  unsigned int row_end_B_0   = B_row_buffer[row_index_B_0+1];

  unsigned int row_index_B_1 = A_col_buffer[j+1];
  unsigned int row_start_B_1 = B_row_buffer[row_index_B_1];
  unsigned int row_end_B_1   = B_row_buffer[row_index_B_1+1];

  row_C_list.rewind();
  unsigned int new_index_0 = (row_start_B_0 < row_end_B_0) ? B_col_buffer[row_start_B_0] : C_size2;
  unsigned int new_index_1 = (row_start_B_1 < row_end_B_1) ? B_col_buffer[row_start_B_1] : C_size2;

  while (1)
  {
    // search:
    unsigned int current_list_index = C_size2;
    while (row_C_list.is_valid())
    {
      current_list_index = row_C_list.item().col_idx;
      if (current_list_index >= new_index_0 || current_list_index >= new_index_1) // one of the indices needs to be inserted or updated
        break;

      // advance in list:
      row_C_list.advance();
    }

    // insert into list and/or move to next entry in row of B:
    if (new_index_0 < new_index_1 && new_index_0 != C_size2)
    {
      if (current_list_index != new_index_0)
        row_C_list.insert_before_current(new_index_0);

      ++row_start_B_0;
      new_index_0 = (row_start_B_0 < row_end_B_0) ? B_col_buffer[row_start_B_0] : C_size2;
    }
    else if (new_index_0 == new_index_1 && new_index_0 != C_size2)
    {
      if (current_list_index != new_index_0)
        row_C_list.insert_before_current(new_index_0);

      ++row_start_B_0;
      new_index_0 = (row_start_B_0 < row_end_B_0) ? B_col_buffer[row_start_B_0] : C_size2;
      ++row_start_B_1;
      new_index_1 = (row_start_B_1 < row_end_B_1) ? B_col_buffer[row_start_B_1] : C_size2;
    }
    else if (new_index_0 > new_index_1 && new_index_1 != C_size2)
    {
      if (current_list_index != new_index_1)
        row_C_list.insert_before_current(new_index_1);

      ++row_start_B_1;
      new_index_1 = (row_start_B_1 < row_end_B_1) ? B_col_buffer[row_start_B_1] : C_size2;
    }

    // all entries processed:
    if (new_index_0 == C_size2 && new_index_1 == C_size2)
      break;
  }

}

/** Second stage */
template<typename RowListT, typename NumericT>
void merge_two_rows(unsigned int j,
                    unsigned int const *A_col_buffer,
                    NumericT const * A_elements,
                    unsigned int const *B_row_buffer,
                    unsigned int const *B_col_buffer,
                    NumericT const * B_elements,
                    RowListT & row_C_list,
                    unsigned int C_size2)
{
  NumericT     val_A_0       = A_elements[j];
  unsigned int row_index_B_0 = A_col_buffer[j];
  unsigned int row_start_B_0 = B_row_buffer[row_index_B_0];
  unsigned int row_end_B_0   = B_row_buffer[row_index_B_0+1];

  NumericT     val_A_1       = A_elements[j+1];
  unsigned int row_index_B_1 = A_col_buffer[j+1];
  unsigned int row_start_B_1 = B_row_buffer[row_index_B_1];
  unsigned int row_end_B_1   = B_row_buffer[row_index_B_1+1];

  row_C_list.rewind();
  unsigned int new_index_0 = (row_start_B_0 < row_end_B_0) ? B_col_buffer[row_start_B_0] : C_size2;
  unsigned int new_index_1 = (row_start_B_1 < row_end_B_1) ? B_col_buffer[row_start_B_1] : C_size2;

  while (1)
  {
    // search:
    unsigned int current_list_index = C_size2;
    while (row_C_list.is_valid())
    {
      current_list_index = row_C_list.item().col_idx;
      if (current_list_index >= new_index_0 || current_list_index >= new_index_1) // one of the indices needs to be inserted or updated
        break;

      // advance in list:
      row_C_list.advance();
    }

    // insert into list and/or move to next entry in row of B:
    if (new_index_0 < new_index_1 && new_index_0 != C_size2)
    {
      if (current_list_index != new_index_0)
        row_C_list.insert_before_current(new_index_0, val_A_0 * B_elements[row_start_B_0]);
      else
        row_C_list.item().value += val_A_0 * B_elements[row_start_B_0];

      ++row_start_B_0;
      new_index_0 = (row_start_B_0 < row_end_B_0) ? B_col_buffer[row_start_B_0] : C_size2;
    }
    else if (new_index_0 == new_index_1 && new_index_0 != C_size2)
    {
      if (current_list_index != new_index_0)
        row_C_list.insert_before_current(new_index_0, val_A_0 * B_elements[row_start_B_0] + val_A_1 * B_elements[row_start_B_1]);
      else
        row_C_list.item().value += val_A_0 * B_elements[row_start_B_0] + val_A_1 * B_elements[row_start_B_1];

      ++row_start_B_0;
      new_index_0 = (row_start_B_0 < row_end_B_0) ? B_col_buffer[row_start_B_0] : C_size2;
      ++row_start_B_1;
      new_index_1 = (row_start_B_1 < row_end_B_1) ? B_col_buffer[row_start_B_1] : C_size2;
    }
    else if (new_index_0 > new_index_1 && new_index_1 != C_size2)
    {
      if (current_list_index != new_index_1)
        row_C_list.insert_before_current(new_index_1, val_A_1 * B_elements[row_start_B_1]);
      else
        row_C_list.item().value += val_A_1 * B_elements[row_start_B_1];

      ++row_start_B_1;
      new_index_1 = (row_start_B_1 < row_end_B_1) ? B_col_buffer[row_start_B_1] : C_size2;
    }

    // all entries processed:
    if (new_index_0 == C_size2 && new_index_1 == C_size2)
      break;
  }

}


template<typename ListT>
unsigned int row_C_scan_symbolic_list(unsigned int max_entries_C,
                                      unsigned int row_start_A, unsigned int row_end_A, unsigned int const *A_col_buffer,
                                      unsigned int const *B_row_buffer, unsigned int const *B_col_buffer, unsigned int B_size2,
                                      ListT & row_C_list)
{
  row_C_list.reserve_and_reset(max_entries_C);

  for (unsigned int j=row_start_A; j<row_end_A; ++j)
  {
    if (row_end_A - j > 1) // merge two rows of B concurrently
    {
      merge_two_rows(j, A_col_buffer, B_row_buffer, B_col_buffer, row_C_list, B_size2);
      ++j;
      continue;
    }
    unsigned int row_index_B = A_col_buffer[j];
    unsigned int row_start_B = B_row_buffer[row_index_B];
    unsigned int row_end_B   = B_row_buffer[row_index_B+1];

    row_C_list.rewind();
    for (std::size_t k=row_start_B; k<row_end_B; ++k)
    {
      unsigned int new_index = B_col_buffer[k];
      // search:
      bool found = false;
      while (row_C_list.is_valid())
      {
        sp_gemm_list_item const & C_item = row_C_list.item();
        if (C_item.col_idx == new_index)
        {
          found = true;
          break;
        }

        if (C_item.col_idx > new_index) // no need to traverse further
          break;

        // advance in list:
        row_C_list.advance();
      }

      if (!found) // insert into list:
        row_C_list.insert_before_current(new_index);
    }
  }

  return row_C_list.size();
}

template<typename NumericT, typename ListT>
void row_C_scan_numeric_list(unsigned int row_start_A, unsigned int row_end_A, unsigned int const *A_col_buffer, NumericT const *A_elements,
                             unsigned int const *B_row_buffer, unsigned int const *B_col_buffer, NumericT const *B_elements, unsigned int B_size2,
                             unsigned int row_start_C, unsigned int row_end_C, unsigned int *C_col_buffer, NumericT *C_elements,
                             ListT & row_C_list)
{
  row_C_list.reserve_and_reset(row_end_C - row_start_C + 1);

  for (unsigned int j=row_start_A; j<row_end_A; ++j)
  {
    if (row_end_A - j > 1) // merge two rows of B concurrently
    {
      merge_two_rows(j, A_col_buffer, A_elements, B_row_buffer, B_col_buffer, B_elements, row_C_list, B_size2);
      ++j;
      continue;
    }
    NumericT val_A = A_elements[j];
    unsigned int row_index_B = A_col_buffer[j];
    unsigned int row_start_B = B_row_buffer[row_index_B];
    unsigned int row_end_B   = B_row_buffer[row_index_B+1];

    row_C_list.rewind();
    for (std::size_t k=row_start_B; k<row_end_B; ++k)
    {
      unsigned int new_index = B_col_buffer[k];

      // search:
      bool found = false;
      while (row_C_list.is_valid())
      {
        sp_gemm_list_value_item<NumericT> & C_item = row_C_list.item();
        if (C_item.col_idx == new_index)
        {
          found = true;
          C_item.value += val_A * B_elements[k];
          break;
        }

        if (C_item.col_idx > new_index) // no need to traverse further
          break;

        // advance in list:
        row_C_list.advance();
      }

      if (!found) // insert into list:
        row_C_list.insert_before_current(new_index, val_A * B_elements[k]);
    }
  }

  // copy to output array:
  row_C_list.rewind();
  for (unsigned int j = row_start_C; j<row_end_C; ++j)
  {
    sp_gemm_list_value_item<NumericT> const & C_item = row_C_list.item();
    C_col_buffer[j] = C_item.col_idx;
    C_elements[j]   = C_item.value;
    row_C_list.advance();
  }

}


} // namespace host_based
} //namespace linalg
} //namespace viennacl


#endif
