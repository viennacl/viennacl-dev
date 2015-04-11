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


#ifdef VIENNACL_WITH_AVX2
#include "immintrin.h"
#endif


namespace viennacl
{
namespace linalg
{
namespace host_based
{



#ifdef VIENNACL_WITH_AVX2
inline
unsigned int row_C_scan_symbolic_vector_AVX2(int const *row_indices_B,
                                             int const *B_row_buffer, int const *B_col_buffer, int B_size2,
                                             int const *row_C_vector_input, int const *row_C_vector_input_end,
                                             int *row_C_vector_output)
{
  __m256i avx_row_indices = _mm256_loadu_si256((__m256i const *)row_indices_B);
  __m256i avx_row_start   = _mm256_i32gather_epi32(B_row_buffer,   avx_row_indices, 4);
  __m256i avx_row_end     = _mm256_i32gather_epi32(B_row_buffer+1, avx_row_indices, 4);

  __m256i avx_all_ones    = _mm256_set_epi32(1, 1, 1, 1, 1, 1, 1, 1);
  __m256i avx_all_bsize2  = _mm256_set_epi32(B_size2, B_size2, B_size2, B_size2, B_size2, B_size2, B_size2, B_size2);
  __m256i avx_load_mask   = _mm256_cmpgt_epi32(avx_row_end, avx_row_start);
  __m256i avx_index_front = avx_all_bsize2;
  avx_index_front         = _mm256_mask_i32gather_epi32(avx_index_front, B_col_buffer, avx_row_start, avx_load_mask, 4);

  int *output_ptr = row_C_vector_output;

  while (1)
  {
    // get minimum index in current front:
    __m256i avx_index_min1 = avx_index_front;
    __m256i avx_temp       = _mm256_permutevar8x32_epi32(avx_index_min1, _mm256_set_epi32(3, 2, 1, 0, 7, 6, 5, 4));
    avx_index_min1 = _mm256_min_epi32(avx_index_min1, avx_temp); // first four elements compared against last four elements

    avx_temp       = _mm256_shuffle_epi32(avx_index_min1, int(78));    // 0b01001110 = 78, using shuffle instead of permutevar here because of lower latency
    avx_index_min1 = _mm256_min_epi32(avx_index_min1, avx_temp); // first two elements compared against elements three and four (same for upper half of register)

    avx_temp       = _mm256_shuffle_epi32(avx_index_min1, int(177));    // 0b10110001 = 177, using shuffle instead of permutevar here because of lower latency
    avx_index_min1 = _mm256_min_epi32(avx_index_min1, avx_temp); // now all entries of avx_index_min1 hold the minimum

    int min_index_in_front = ((int*)&avx_index_min1)[0];
    // check for end of merge operation:
    if (min_index_in_front == B_size2)
      break;

    // write current entry:
    *output_ptr = min_index_in_front;
    ++output_ptr;

    // advance index front where equal to minimum index:
    avx_load_mask   = _mm256_cmpeq_epi32(avx_index_front, avx_index_min1);
    // first part: set index to B_size2 if equal to minimum index:
    avx_temp        = _mm256_and_si256(avx_all_bsize2, avx_load_mask);
    avx_index_front = _mm256_max_epi32(avx_index_front, avx_temp);
    // second part: increment row_start registers where minimum found:
    avx_temp        = _mm256_and_si256(avx_all_ones, avx_load_mask); //ones only where the minimum was found
    avx_row_start   = _mm256_add_epi32(avx_row_start, avx_temp);
    // third part part: load new data where more entries available:
    avx_load_mask   = _mm256_cmpgt_epi32(avx_row_end, avx_row_start);
    avx_index_front = _mm256_mask_i32gather_epi32(avx_index_front, B_col_buffer, avx_row_start, avx_load_mask, 4);
  }

  return static_cast<unsigned int>(output_ptr - row_C_vector_output);
}
#endif

/** @brief Merges up to IndexNum rows from B into the result buffer.
*
* Because the input buffer also needs to be considered, this routine actually works on an index front of length (IndexNum+1)
**/
template<unsigned int IndexNum>
unsigned int row_C_scan_symbolic_vector_N(unsigned int const *row_indices_B,
                                          unsigned int const *B_row_buffer, unsigned int const *B_col_buffer, unsigned int B_size2,
                                          unsigned int const *row_C_vector_input, unsigned int const *row_C_vector_input_end,
                                          unsigned int *row_C_vector_output)
{
  unsigned int index_front[IndexNum+1];
  unsigned int const *index_front_start[IndexNum+1];
  unsigned int const *index_front_end[IndexNum+1];

  // Set up pointers for loading the indices:
  for (unsigned int i=0; i<IndexNum; ++i, ++row_indices_B)
  {
    index_front_start[i] = B_col_buffer + B_row_buffer[*row_indices_B];
    index_front_end[i]   = B_col_buffer + B_row_buffer[*row_indices_B + 1];
  }
  index_front_start[IndexNum] = row_C_vector_input;
  index_front_end[IndexNum]   = row_C_vector_input_end;

  // load indices:
  for (unsigned int i=0; i<=IndexNum; ++i)
    index_front[i] = (index_front_start[i] < index_front_end[i]) ? *index_front_start[i] : B_size2;

  unsigned int *output_ptr = row_C_vector_output;

  while (1)
  {
    // get minimum index in current front:
    unsigned int min_index_in_front = B_size2;
    for (unsigned int i=0; i<=IndexNum; ++i)
      min_index_in_front = std::min(min_index_in_front, index_front[i]);

    if (min_index_in_front == B_size2) // we're done
      break;

    // advance index front where equal to minimum index:
    for (unsigned int i=0; i<=IndexNum; ++i)
    {
      if (index_front[i] == min_index_in_front)
      {
        index_front_start[i] += 1;
        index_front[i] = (index_front_start[i] < index_front_end[i]) ? *index_front_start[i] : B_size2;
      }
    }

    // write current entry:
    *output_ptr = min_index_in_front;
    ++output_ptr;
  }

  return static_cast<unsigned int>(output_ptr - row_C_vector_output);
}

inline
unsigned int row_C_scan_symbolic_vector_1(unsigned int row_index_B,
                                          unsigned int const *B_row_buffer, unsigned int const *B_col_buffer, unsigned int B_size2,
                                          unsigned int const *row_C_vector_input, unsigned int const *row_C_vector_input_end,
                                          unsigned int *row_C_vector_output)
{
  unsigned int *output_ptr = row_C_vector_output;
  unsigned int col_C = (row_C_vector_input < row_C_vector_input_end) ? *row_C_vector_input : B_size2;

  unsigned int const *row_B_start = B_col_buffer + B_row_buffer[row_index_B];
  unsigned int const *row_B_end   = B_col_buffer + B_row_buffer[row_index_B + 1];
  unsigned int col_B = (row_B_start < row_B_end) ? *row_B_start : B_size2;
  while (1)
  {
    unsigned int min_index = std::min(col_B, col_C);

    if (min_index == B_size2)
      break;

    if (min_index == col_B)
    {
      ++row_B_start;
      col_B = (row_B_start < row_B_end) ? *row_B_start : B_size2;
    }

    if (min_index == col_C)
    {
      ++row_C_vector_input;
      col_C = (row_C_vector_input < row_C_vector_input_end) ? *row_C_vector_input : B_size2;
    }

    // write current entry:
    *output_ptr = min_index;
    ++output_ptr;
  }

  return static_cast<unsigned int>(output_ptr - row_C_vector_output);
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
#ifdef VIENNACL_WITH_AVX2
    if (row_end_A - row_start_A >= 8 && row_C_len == 0)
    {
      row_C_len = row_C_scan_symbolic_vector_AVX2((const int*)(A_col_buffer + row_start_A),
                                                  (const int*)B_row_buffer, (const int*)B_col_buffer, int(B_size2),
                                                  (const int*)row_C_vector_1, (const int*)(row_C_vector_1 + row_C_len),
                                                  (int*)row_C_vector_2);
      row_start_A += 8;
    }
    else
#endif
    /*if (row_end_A - row_start_A > 3)
    {
      row_C_len = row_C_scan_symbolic_vector_N<3>(A_col_buffer + row_start_A,
                                                  B_row_buffer, B_col_buffer, B_size2,
                                                  row_C_vector_1, row_C_vector_1 + row_C_len,
                                                  row_C_vector_2);
      row_start_A += 3;
    }
    else*/
    {
      // process single row:
      row_C_len = row_C_scan_symbolic_vector_1(A_col_buffer[row_start_A],
                                               B_row_buffer, B_col_buffer, B_size2,
                                               row_C_vector_1, row_C_vector_1 + row_C_len,
                                               row_C_vector_2);
      ++row_start_A;
    }

    std::swap(row_C_vector_1, row_C_vector_2);
  }

  return row_C_len;
}

//////////////////////////////

/** @brief Merges up to IndexNum rows from B into the result buffer.
*
* Because the input buffer also needs to be considered, this routine actually works on an index front of length (IndexNum+1)
**/
template<unsigned int IndexNum, typename NumericT>
unsigned int row_C_scan_numeric_vector_N(unsigned int const *row_indices_B, NumericT const *val_A,
                                          unsigned int const *B_row_buffer, unsigned int const *B_col_buffer, NumericT const *B_elements, unsigned int B_size2,
                                          unsigned int const *row_C_vector_input, unsigned int const *row_C_vector_input_end, NumericT *row_C_vector_input_values,
                                          unsigned int *row_C_vector_output, NumericT *row_C_vector_output_values)
{
  unsigned int index_front[IndexNum+1];
  unsigned int const *index_front_start[IndexNum+1];
  unsigned int const *index_front_end[IndexNum+1];
  NumericT const * value_front_start[IndexNum+1];
  NumericT values_A[IndexNum+1];

  // Set up pointers for loading the indices:
  for (unsigned int i=0; i<IndexNum; ++i, ++row_indices_B)
  {
    unsigned int row_B = *row_indices_B;

    index_front_start[i] = B_col_buffer + B_row_buffer[row_B];
    index_front_end[i]   = B_col_buffer + B_row_buffer[row_B + 1];
    value_front_start[i] = B_elements   + B_row_buffer[row_B];
    values_A[i]          = val_A[i];
  }
  index_front_start[IndexNum] = row_C_vector_input;
  index_front_end[IndexNum]   = row_C_vector_input_end;
  value_front_start[IndexNum] = row_C_vector_input_values;
  values_A[IndexNum]          = NumericT(1);

  // load indices:
  for (unsigned int i=0; i<=IndexNum; ++i)
    index_front[i] = (index_front_start[i] < index_front_end[i]) ? *index_front_start[i] : B_size2;

  unsigned int *output_ptr = row_C_vector_output;

  while (1)
  {
    // get minimum index in current front:
    unsigned int min_index_in_front = B_size2;
    for (unsigned int i=0; i<=IndexNum; ++i)
      min_index_in_front = std::min(min_index_in_front, index_front[i]);

    if (min_index_in_front == B_size2) // we're done
      break;

    // advance index front where equal to minimum index:
    NumericT row_C_value = 0;
    for (unsigned int i=0; i<=IndexNum; ++i)
    {
      if (index_front[i] == min_index_in_front)
      {
        index_front_start[i] += 1;
        index_front[i] = (index_front_start[i] < index_front_end[i]) ? *index_front_start[i] : B_size2;

        row_C_value += values_A[i] * *value_front_start[i];
        value_front_start[i] += 1;
      }
    }

    // write current entry:
    *output_ptr = min_index_in_front;
    ++output_ptr;
    *row_C_vector_output_values = row_C_value;
    ++row_C_vector_output_values;
  }

  return static_cast<unsigned int>(output_ptr - row_C_vector_output);
}



template<typename NumericT>
unsigned int row_C_scan_numeric_vector_1(unsigned int row_index_B, NumericT val_A,
                                         unsigned int const *B_row_buffer, unsigned int const *B_col_buffer, NumericT const *B_elements, unsigned int B_size2,
                                         unsigned int const *row_C_vector_input, unsigned int const *row_C_vector_input_end, NumericT *row_C_vector_input_values,
                                         unsigned int *row_C_vector_output, NumericT *row_C_vector_output_values)
{
  unsigned int *output_ptr        = row_C_vector_output;
  NumericT     *output_ptr_values = row_C_vector_output_values;

  unsigned int col_C       = (row_C_vector_input < row_C_vector_input_end) ? *row_C_vector_input        : B_size2;

  unsigned int row_B_offset       = B_row_buffer[row_index_B];
  unsigned int const *row_B_start = B_col_buffer + row_B_offset;
  unsigned int const *row_B_end   = B_col_buffer + B_row_buffer[row_index_B + 1];
  NumericT const * row_B_values   = B_elements   + row_B_offset;
  unsigned int col_B = (row_B_start < row_B_end) ? *row_B_start : B_size2;
  while (1)
  {
    unsigned int min_index = std::min(col_B, col_C);
    NumericT value = 0;

    if (min_index == B_size2)
      break;

    if (min_index == col_B)
    {
      ++row_B_start;
      col_B = (row_B_start < row_B_end) ? *row_B_start : B_size2;

      value += val_A * *row_B_values;
      ++row_B_values;
    }

    if (min_index == col_C)
    {
      ++row_C_vector_input;
      col_C = (row_C_vector_input < row_C_vector_input_end) ? *row_C_vector_input : B_size2;

      value += *row_C_vector_input_values;
      ++row_C_vector_input_values;
    }

    // write current entry:
    *output_ptr = min_index;
    ++output_ptr;
    *output_ptr_values = value;
    ++output_ptr_values;
  }

  return static_cast<unsigned int>(output_ptr - row_C_vector_output);
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
    /*if (row_end_A - row_start_A > 3)
    {
      row_C_len = row_C_scan_numeric_vector_N<3>(A_col_buffer + row_start_A , A_elements + row_start_A,
                                                 B_row_buffer, B_col_buffer, B_elements, B_size2,
                                                 row_C_vector_1, row_C_vector_1 + row_C_len, row_C_vector_1_values,
                                                 row_C_vector_2, row_C_vector_2_values);
      row_start_A += 3;
    }
    else */ // process single row:
    {
      row_C_len = row_C_scan_numeric_vector_1(A_col_buffer[row_start_A], A_elements[row_start_A],
                                              B_row_buffer, B_col_buffer, B_elements, B_size2,
                                              row_C_vector_1, row_C_vector_1 + row_C_len, row_C_vector_1_values,
                                              row_C_vector_2, row_C_vector_2_values);
      ++row_start_A;
    }

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
