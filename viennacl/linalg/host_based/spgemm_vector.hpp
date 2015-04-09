#ifndef VIENNACL_LINALG_HOST_BASED_SPGEMM_VECTOR_HPP_
#define VIENNACL_LINALG_HOST_BASED_SPGEMM_VECTOR_HPP_

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

inline
unsigned int row_C_scan_symbolic_vector_1(unsigned int row_index_B,
                                          unsigned int const *B_row_buffer, unsigned int const *B_col_buffer, unsigned int B_size2,
                                          unsigned int const *row_C_vector_input, unsigned int const *row_C_vector_input_end,
                                          unsigned int *row_C_vector_output)
{
  unsigned int *output_ptr = row_C_vector_output;
  unsigned int current_col_input = (row_C_vector_input < row_C_vector_input_end) ? *row_C_vector_input : B_size2;

  unsigned int row_B_end = B_row_buffer[row_index_B + 1];
  for (unsigned int j = B_row_buffer[row_index_B]; j < row_B_end; ++j)
  {
    unsigned int col_B = B_col_buffer[j];

    // advance row_C_vector_input as needed:
    while (current_col_input < col_B)
    {
      *output_ptr = current_col_input;
      ++output_ptr;

      ++row_C_vector_input;
      current_col_input = (row_C_vector_input < row_C_vector_input_end) ? *row_C_vector_input : B_size2;
    }

    // write current entry:
    *output_ptr = col_B;
    ++output_ptr;

    // skip input if same as col_B:
    if (current_col_input == col_B)
    {
      ++row_C_vector_input;
      current_col_input = (row_C_vector_input < row_C_vector_input_end) ? *row_C_vector_input : B_size2;
    }
  }

  // write remaining entries:
  for (; row_C_vector_input < row_C_vector_input_end; ++row_C_vector_input, ++output_ptr)
    *output_ptr = *row_C_vector_input;

  return output_ptr - row_C_vector_output;
}

inline
unsigned int row_C_scan_symbolic_vector(unsigned int row_start_A, unsigned int row_end_A, unsigned int const *A_col_buffer,
                                        unsigned int const *B_row_buffer, unsigned int const *B_col_buffer, unsigned int B_size2,
                                        unsigned int *row_C_vector_1, unsigned int *row_C_vector_2)
{
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
  unsigned int row_C_len = 0;
  while (row_end_A > row_start_A)
  {
    // process single row:
    row_C_len = row_C_scan_symbolic_vector_1(A_col_buffer[row_start_A],
                                             B_row_buffer, B_col_buffer, B_size2,
                                             row_C_vector_1, row_C_vector_1 + row_C_len,
                                             row_C_vector_2);

    ++row_start_A;
    std::swap(row_C_vector_1, row_C_vector_2);
  }

  return row_C_len;
}

//////////////////////////////

template<typename NumericT>
unsigned int row_C_scan_numeric_vector_1(unsigned int row_index_B, NumericT val_A,
                                         unsigned int const *B_row_buffer, unsigned int const *B_col_buffer, NumericT const *B_elements, unsigned int B_size2,
                                         unsigned int const *row_C_vector_input, unsigned int const *row_C_vector_input_end, NumericT *row_C_vector_input_values,
                                         unsigned int *row_C_vector_output, NumericT *row_C_vector_output_values)
{
  unsigned int *output_ptr        = row_C_vector_output;
  NumericT     *output_ptr_values = row_C_vector_output_values;

  unsigned int current_col_input       = (row_C_vector_input < row_C_vector_input_end) ? *row_C_vector_input        : B_size2;
  NumericT     current_col_input_value = (row_C_vector_input < row_C_vector_input_end) ? *row_C_vector_input_values : NumericT(0);

  unsigned int row_B_end = B_row_buffer[row_index_B + 1];
  for (unsigned int j = B_row_buffer[row_index_B]; j < row_B_end; ++j)
  {
    unsigned int col_B = B_col_buffer[j];

    // advance row_C_vector_input as needed:
    while (current_col_input < col_B)
    {
      *output_ptr = current_col_input;
      ++output_ptr;
      *output_ptr_values = current_col_input_value;
      ++output_ptr_values;

      ++row_C_vector_input;
      ++row_C_vector_input_values;
      current_col_input       = (row_C_vector_input < row_C_vector_input_end) ? *row_C_vector_input        : B_size2;
      current_col_input_value = (row_C_vector_input < row_C_vector_input_end) ? *row_C_vector_input_values : NumericT(0);
    }

    // write current entry:
    *output_ptr = col_B;
    ++output_ptr;

    // skip input if same as col_B:
    if (current_col_input == col_B)
    {
      *output_ptr_values = val_A * B_elements[j] + current_col_input_value;

      ++row_C_vector_input;
      ++row_C_vector_input_values;
      current_col_input       = (row_C_vector_input < row_C_vector_input_end) ? *row_C_vector_input        : B_size2;
      current_col_input_value = (row_C_vector_input < row_C_vector_input_end) ? *row_C_vector_input_values : NumericT(0);
    }
    else
      *output_ptr_values = val_A * B_elements[j];

    ++output_ptr_values;
  }

  // write remaining entries:
  for (; row_C_vector_input < row_C_vector_input_end; ++row_C_vector_input, ++row_C_vector_input_values, ++output_ptr, ++output_ptr_values)
  {
    *output_ptr        = *row_C_vector_input;
    *output_ptr_values = *row_C_vector_input_values;
  }

  return output_ptr - row_C_vector_output;
}

template<typename NumericT>
void row_C_scan_numeric_vector(unsigned int row_start_A, unsigned int row_end_A, unsigned int const *A_col_buffer, NumericT const *A_elements,
                               unsigned int const *B_row_buffer, unsigned int const *B_col_buffer, NumericT const *B_elements, unsigned int B_size2,
                               unsigned int row_start_C, unsigned int row_end_C, unsigned int *C_col_buffer, NumericT *C_elements,
                               unsigned int *row_C_vector_1, NumericT *row_C_vector_1_values,
                               unsigned int *row_C_vector_2, NumericT *row_C_vector_2_values)
{
  (void)row_end_C;

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
  unsigned int row_C_len = 0;
  while (row_end_A > row_start_A)
  {
    // process single row:
    row_C_len = row_C_scan_numeric_vector_1(A_col_buffer[row_start_A], A_elements[row_start_A],
                                            B_row_buffer, B_col_buffer, B_elements, B_size2,
                                            row_C_vector_1, row_C_vector_1 + row_C_len, row_C_vector_1_values,
                                            row_C_vector_2, row_C_vector_2_values);

    ++row_start_A;
    std::swap(row_C_vector_1,        row_C_vector_2);
    std::swap(row_C_vector_1_values, row_C_vector_2_values);
  }

  // copy to output:
  C_col_buffer += row_start_C;
  C_elements += row_start_C;
  for (unsigned int i=0; i<row_C_len; ++i, ++C_col_buffer, ++C_elements)
  {
    *C_col_buffer = row_C_vector_1[i];
    *C_elements   = row_C_vector_1_values[i];
  }
}


} // namespace host_based
} //namespace linalg
} //namespace viennacl


#endif
