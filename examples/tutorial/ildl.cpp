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

/** \example ildl.cpp
*
*   Experiments with ILDL factorizations
*
*   We start with including the necessary system headers:
**/

//
// include necessary system headers
//
#include <iostream>
#include <cmath>

//
// ViennaCL includes
//
#include "viennacl/scalar.hpp"
#include "viennacl/vector.hpp"
#include "viennacl/compressed_matrix.hpp"
#include "viennacl/linalg/prod.hpp"
#include "viennacl/linalg/ilu.hpp"
#include "viennacl/io/matrix_market.hpp"

/** @brief Sets up a SPD test system, no 2x2-blocks. Used to verify convergence of standard Chow-Patel. */
template<typename NumericT, typename IndexT>
void setup_system_2(viennacl::compressed_matrix<NumericT> & A,
                    viennacl::vector<NumericT> & scaling,
                    viennacl::vector<IndexT> & permutation,
                    viennacl::vector<IndexT> & inv_permutation,
                    viennacl::vector<IndexT> & blocking)
{
  std::vector<std::map<unsigned int, NumericT> > B(5);

  B[0][0] =  2.0; B[0][1] = -1.0; B[0][2] = 1e-5; B[0][3] = 1e-5; B[0][4] = 1e-5;
  B[1][0] = -1.0; B[1][1] =  3.0; B[1][2] = -1.0; B[1][3] = 1e-5; B[1][4] = 1e-5;
  B[2][0] = 1e-5; B[2][1] = -1.0; B[2][2] =  4.0; B[2][3] = -1.0; B[2][4] = 1e-5;
  B[3][0] = 1e-5; B[3][1] = 1e-5; B[3][2] = -1.0; B[3][3] =  5.0; B[3][4] = -1.0;
  B[4][0] = 1e-5; B[4][1] = 1e-5; B[4][2] = 1e-5; B[4][3] = -1.0; B[4][4] =  6.0;

  viennacl::copy(B, A);

  //
  // Scaling
  //
  std::vector<NumericT> scal(A.size1());
  for (std::size_t i=0; i<scal.size(); ++i)
    scal[i] = 1.0;

  viennacl::copy(scal, scaling);

  //
  // Permutation
  //
  std::vector<IndexT> perm(A.size1());
  //perm[0] = perm.size() - 1;
  //for (std::size_t i=1; i<perm.size(); ++i)
  //  perm[i] = i-1;
  for (std::size_t i=0; i<perm.size(); ++i)
    perm[i] = i;

  viennacl::copy(perm, permutation);

  //
  // Inverse permutation
  //
  std::vector<IndexT> inv_perm(perm.size());
  for (std::size_t i=0; i<perm.size(); ++i)
    inv_perm[perm[i]] = i;

  viennacl::copy(inv_perm, inv_permutation);

  //
  // blocking
  //
  std::vector<IndexT> blocks(perm.size());
  for (std::size_t i=0; i<blocks.size(); ++i)
    blocks[i] = 1;

  viennacl::copy(blocks, blocking);

}

/** @brief Test system with a 2x2 block on the diagonal. Used to test ILDL with matching. */
template<typename NumericT, typename IndexT>
void setup_system_3(viennacl::compressed_matrix<NumericT> & A,
                    viennacl::vector<NumericT> & scaling,
                    viennacl::vector<IndexT> & permutation,
                    viennacl::vector<IndexT> & inv_permutation,
                    viennacl::vector<IndexT> & blocking)
{
  std::vector<std::map<unsigned int, NumericT> > B(5);

  B[0][0] =  1.0; B[0][1] = -1.0; B[0][2] = 1e-2; B[0][3] = 1e-2; B[0][4] = 1e-2;
  B[1][0] = -1.0; B[1][1] =  2.0; B[1][2] = -1.0; B[1][3] = 1e-2; B[1][4] = 1e-2;
  B[2][0] = 1e-2; B[2][1] = -1.0; B[2][2] =  0.0; B[2][3] = -1.0; B[2][4] = 1e-2;
  B[3][0] = 1e-2; B[3][1] = 1e-2; B[3][2] = -1.0; B[3][3] =  0.0; B[3][4] = -1.0;
  B[4][0] = 1e-2; B[4][1] = 1e-2; B[4][2] = 1e-2; B[4][3] = -1.0; B[4][4] =  6.0;

  viennacl::copy(B, A);

  //
  // Scaling
  //
  std::vector<NumericT> scal(A.size1());
  for (std::size_t i=0; i<scal.size(); ++i)
    scal[i] = 1.0;

  viennacl::copy(scal, scaling);

  //
  // Permutation
  //
  std::vector<IndexT> perm(A.size1());
  for (std::size_t i=0; i<perm.size(); ++i)
    perm[i] = i; //perm.size() - 1 - i;

  viennacl::copy(perm, permutation);

  //
  // Inverse permutation
  //
  std::vector<IndexT> inv_perm(perm.size());
  for (std::size_t i=0; i<perm.size(); ++i)
    inv_perm[perm[i]] = i;

  viennacl::copy(inv_perm, inv_permutation);

  //
  // blocking (indices after permutation)
  //
  std::vector<IndexT> blocks(perm.size());
  blocks[0] = 1; blocks[1] = 1;
  blocks[2] = 2; blocks[3] = 0;
  blocks[4] = 1;

  viennacl::copy(blocks, blocking);

}

/** @brief Extracts a lower triangular matrix L of A with augmented nonzero pattern, and extracts diagonal blocks.

  L is stored in standard CSR format with a few tweaks:
   - If the blocksize for certain row is 2 instead of one, that row will store 2x1 and 2x2 blocks (row-major) directly.
     All block entries that are not part of the original nonzero pattern are added as zero.
   - diagonal entries/blocks are not stored

  Example:
   A has 2x2-blocks in rows {0, 1} and rows {2, 3}. L will not have any entries in rows 0 and 1, but will have a 2x2 block (4 values, 4 column indices) stored as the first entries in row 2.
   Column indices for each numerical entry are stored as usual.

  @param A               The system matrix
  @param L               The lower triangular matrix with augmented sparsity pattern
  @param D               Block-diagonal matrix, initialized with the entries from A
  @param scaling         Numerical scaling vector. Entry a_ij in A is scaled by scaling[i] times scaling[j]
  @param permutation     Permutation vector. permutation[i] returns the index of i after permutation
  @param inv_permutation Inverse permutation vector. inv_permutation[i] returns the index of i in the original matrix (before permutation)
  @param blocksize       Vector of block sizes. 2: first row/column of a 2x2-block. 1: row/column is not blocked. 0: Second row/column of a 2x2-block.
**/
template<typename NumericT, typename IndexT>
void init_blocked_L_from_A(viennacl::compressed_matrix<NumericT> const & A,
                           viennacl::compressed_matrix<NumericT> & L,
                           viennacl::compressed_matrix<NumericT> & D,
                           viennacl::vector<NumericT> const & scaling,
                           viennacl::vector<IndexT> const & permutation,
                           viennacl::vector<IndexT> const & inv_permutation,
                           viennacl::vector<IndexT> const & blocksize)
{
  NumericT     const * A_elements  = viennacl::linalg::host_based::detail::extract_raw_pointer<NumericT>(A.handle());
  unsigned int const * A_rows      = viennacl::linalg::host_based::detail::extract_raw_pointer<unsigned int>(A.handle1());
  unsigned int const * A_cols      = viennacl::linalg::host_based::detail::extract_raw_pointer<unsigned int>(A.handle2());

  NumericT     const * scal      = viennacl::linalg::host_based::detail::extract_raw_pointer<NumericT>(scaling.handle());
  unsigned int const * perm      = viennacl::linalg::host_based::detail::extract_raw_pointer<unsigned int>(permutation.handle());
  unsigned int const * inv_perm  = viennacl::linalg::host_based::detail::extract_raw_pointer<unsigned int>(inv_permutation.handle());
  unsigned int const * bsize     = viennacl::linalg::host_based::detail::extract_raw_pointer<unsigned int>(blocksize.handle());

  //
  // Step 1: (Over-)estimate number of elements for L with augmented nonzero pattern
  //
  unsigned int cnt = 0;
  for (std::size_t i=0; i<A.size1(); ++i)
  {
    unsigned int bsize_row = bsize[i];
    if (bsize[i] < 1)
      continue;

    unsigned int row_in_A = inv_perm[i];
    for (unsigned int j = A_rows[row_in_A]; j < A_rows[row_in_A+1]; ++j)
    {
      unsigned int col_index = perm[A_cols[j]];
      if (col_index < i)
      {
        unsigned int bsize_col = bsize[col_index];
        if (bsize_col < 1)
          bsize_col = 2;

        cnt += bsize_row * bsize_col;
      }
    }
  }


  //
  // Step 2: Set up initial guess for blocked L and D from A
  //

  L.resize(A.size1(), A.size2(), false);
  L.reserve(cnt, false);

  D.resize(A.size1(), A.size2(), false);
  D.reserve(2*A.size2(), false);

  NumericT     * L_elements  = viennacl::linalg::host_based::detail::extract_raw_pointer<NumericT>(L.handle());
  unsigned int * L_rows      = viennacl::linalg::host_based::detail::extract_raw_pointer<unsigned int>(L.handle1());
  unsigned int * L_cols      = viennacl::linalg::host_based::detail::extract_raw_pointer<unsigned int>(L.handle2());

  NumericT     * D_elements  = viennacl::linalg::host_based::detail::extract_raw_pointer<NumericT>(D.handle());
  unsigned int * D_rows      = viennacl::linalg::host_based::detail::extract_raw_pointer<unsigned int>(D.handle1());
  unsigned int * D_cols      = viennacl::linalg::host_based::detail::extract_raw_pointer<unsigned int>(D.handle2());

  unsigned int cnt_L = 0;
  unsigned int cnt_D = 0;
  for (std::size_t i=0; i<A.size1(); ++i)
  {
    L_rows[i] = cnt_L;
    D_rows[i] = cnt_D;

    if (bsize[i] == 1) // deal with a 'normal' row
    {
      //
      // Step a) Permute row of A (note: STL map is not the best pick in terms of performance.)
      //
      typedef std::map<unsigned int, unsigned int> MapType;
      MapType reordered_row_A;

      unsigned int row_in_A = inv_perm[i];
      for (unsigned int j = A_rows[row_in_A]; j < A_rows[row_in_A+1]; ++j)
        reordered_row_A[perm[A_cols[j]]] = j;

      //
      // Step b) Iterate over permuted row and write to L
      //
      unsigned int last_in_row_L = A.size2();
      for (typename MapType::const_iterator it = reordered_row_A.begin(); it != reordered_row_A.end(); ++it)
      {
        unsigned int col = it->first;
        if (col < i)
        {
          if (bsize[col] == 0 && last_in_row_L != col - 1) // check if preceding entry has been written. If not, augment nonzero pattern
          {
            L_cols[cnt_L] = col - 1;
            L_elements[cnt_L] = 0;
            ++cnt_L;
          }

          // write current entry to L
          L_cols[cnt_L] = col;
          last_in_row_L = col;
          NumericT aij = scal[perm[i]] * A_elements[it->second] * scal[inv_perm[col]];
          L_elements[cnt_L] = aij;
          ++cnt_L;
        }
        else if (col == i)
        {
          D_cols[cnt_D] = col;
          NumericT aij = scal[perm[i]] * A_elements[it->second] * scal[inv_perm[col]];
          D_elements[cnt_D] = aij;
          ++cnt_D;
        }
      }
    }
    else if (bsize[i] == 2) // deal with two rows concurrently (blocking)
    {
      //
      // Step a) Permute the two rows of A (note: STL map is not the best pick in terms of performance.)
      //
      typedef std::map<unsigned int, unsigned int> MapType;
      MapType reordered_row_A_0;
      MapType reordered_row_A_1;

      unsigned int row_in_A = inv_perm[i];
      for (unsigned int j = A_rows[row_in_A]; j < A_rows[row_in_A+1]; ++j)
        reordered_row_A_0[perm[A_cols[j]]] = j;
      row_in_A = inv_perm[i+1];
      for (unsigned int j = A_rows[row_in_A]; j < A_rows[row_in_A+1]; ++j)
        reordered_row_A_1[perm[A_cols[j]]] = j;


      //
      // Step b) Iterate over permuted row and write to L
      //
      typename MapType::const_iterator it0 = reordered_row_A_0.begin();
      typename MapType::const_iterator it1 = reordered_row_A_1.begin();

      while (1)
      {
        // pick next entry for each of the two rows:
        unsigned int col0 = (it0 != reordered_row_A_0.end()) ? it0->first : A.size2();
        unsigned int col1 = (it1 != reordered_row_A_1.end()) ? it1->first : A.size2();

        unsigned int col_min = std::min(col0, col1);

        if (col_min < i) // entries for L
        {
          if (bsize[col_min] == 1) // 2x1 block
          {
            L_cols[cnt_L] = col_min;
            NumericT aij0 = 0;
            if (col0 == col_min)
            {
              aij0 = (scal[perm[i]] * A_elements[it0->second] * scal[inv_perm[col0]]);
              ++it0;
            }
            L_elements[cnt_L] = aij0;
            ++cnt_L;

            L_cols[cnt_L] = col_min;
            NumericT aij1 = 0;
            if (col1 == col_min)
            {
              aij1 = (scal[perm[i]] * A_elements[it1->second] * scal[inv_perm[col1]]);
              ++it1;
            }
            L_elements[cnt_L] = aij1;
            ++cnt_L;
          }
          else // 2x2 block
          {
            if (bsize[col_min] == 0)
              col_min -= 1;

            // first row:
            NumericT aij00 = 0, aij01 = 0;
            if (col0 == col_min)
            {
              aij00 = (scal[perm[i]] * A_elements[it0->second] * scal[inv_perm[col0]]);
              ++it0;
              col0 = (it0 != reordered_row_A_0.end()) ? it0->first : A.size2();
            }

            if (col0 == col_min + 1)
            {
              aij01 = (scal[perm[i]] * A_elements[it0->second] * scal[inv_perm[col0]]);
              ++it0;
              col0 = (it0 != reordered_row_A_0.end()) ? it0->first : A.size2();
            }

            // second row:
            NumericT aij10 = 0, aij11 = 0;
            if (col1 == col_min)
            {
              aij10 = (scal[perm[i]] * A_elements[it1->second] * scal[inv_perm[col1]]);
              ++it1;
              col1 = (it1 != reordered_row_A_1.end()) ? it1->first : A.size2();
            }

            if (col1 == col_min + 1)
            {
              aij11 = (scal[perm[i]] * A_elements[it1->second] * scal[inv_perm[col1]]);
              ++it1;
              col1 = (it1 != reordered_row_A_1.end()) ? it1->first : A.size2();
            }

            // write to L:
            L_cols[cnt_L] = col_min;
            L_elements[cnt_L] = aij00;
            ++cnt_L;

            L_cols[cnt_L] = col_min + 1;
            L_elements[cnt_L] = aij01;
            ++cnt_L;

            L_cols[cnt_L] = col_min;
            L_elements[cnt_L] = aij10;
            ++cnt_L;

            L_cols[cnt_L] = col_min + 1;
            L_elements[cnt_L] = aij11;
            ++cnt_L;
          }
        }
        else if (col_min == i) // entries for D
        {
          NumericT d00 = 0.0, d01 = 0.0;
          NumericT d10 = 0.0, d11 = 0.0;

          // check for nonzero entries in first row of diagonal
          if (col0 == col_min)
          {
            d00 = scal[perm[i]] * A_elements[it0->second] * scal[inv_perm[col0]];
            ++it0;
            col0 = (it0 != reordered_row_A_0.end()) ? it0->first : A.size2();
          }
          if (col0 == col_min+1)
            d01 = scal[perm[i]] * A_elements[it0->second] * scal[inv_perm[col0]];

          // check for nonzero entries in second row of diagonal block
          if (col1 == col_min)
          {
            d10 = scal[perm[i]] * A_elements[it1->second] * scal[inv_perm[col1]];
            ++it1;
            col1 = (it1 != reordered_row_A_1.end()) ? it1->first : A.size2();
          }
          if (col1 == col_min+1)
            d11 = scal[perm[i]] * A_elements[it1->second] * scal[inv_perm[col1]];

          // write to D
          D_cols[cnt_D] = i;
          D_elements[cnt_D] = d00;
          ++cnt_D;

          D_cols[cnt_D] = i+1;
          D_elements[cnt_D] = d01;
          ++cnt_D;

          D_rows[i+1] = cnt_D; // new row

          D_cols[cnt_D] = i;
          D_elements[cnt_D] = d10;
          ++cnt_D;

          D_cols[cnt_D] = i+1;
          D_elements[cnt_D] = d11;
          ++cnt_D;

          break; // no more work left in this row pair
        }
        else
          throw std::logic_error("invalid column index encountered!");
      }

      ++i; // skip next row because we dealt with it just now
      L_rows[i] = cnt_L;
    }
    else
      throw std::logic_error("bsize[i] == 0 encountered!");

  } // for all rows of A

  L_rows[L.size1()] = cnt_L;
  D_rows[D.size1()] = cnt_D;
}



/** @brief Converted a matrix in blocked CSR-storage (without diagonal) back to a conventional CSR matrix (including diagonal entries) */
template<typename NumericT, typename IndexT>
void convert_blocked_L_to_L(viennacl::compressed_matrix<NumericT> const & blocked_L,
                            viennacl::compressed_matrix<NumericT> & L,
                            viennacl::vector<IndexT> const & blocksize)
{
  NumericT     const * blocked_L_elements  = viennacl::linalg::host_based::detail::extract_raw_pointer<NumericT    >(blocked_L.handle());
  unsigned int const * blocked_L_rows      = viennacl::linalg::host_based::detail::extract_raw_pointer<unsigned int>(blocked_L.handle1());
  unsigned int const * blocked_L_cols      = viennacl::linalg::host_based::detail::extract_raw_pointer<unsigned int>(blocked_L.handle2());

  L.resize(blocked_L.size1(), blocked_L.size2(), false);
  L.reserve(blocked_L_rows[blocked_L.size1()] + blocked_L.size1(), false);

  NumericT     * L_elements  = viennacl::linalg::host_based::detail::extract_raw_pointer<NumericT    >(L.handle());
  unsigned int * L_rows      = viennacl::linalg::host_based::detail::extract_raw_pointer<unsigned int>(L.handle1());
  unsigned int * L_cols      = viennacl::linalg::host_based::detail::extract_raw_pointer<unsigned int>(L.handle2());

  unsigned int const * bsize     = viennacl::linalg::host_based::detail::extract_raw_pointer<unsigned int>(blocksize.handle());

  unsigned int cnt_L = 0;
  for (std::size_t i=0; i<L.size1(); ++i)
  {
    L_rows[i] = cnt_L;

    if (bsize[i] == 1)
    {
      for (IndexT j=blocked_L_rows[i]; j != blocked_L_rows[i+1]; ++j)
      {
        L_cols[cnt_L]     = blocked_L_cols[j];
        L_elements[cnt_L] = blocked_L_elements[j];
        ++cnt_L;
      }
    }
    else if (bsize[i] == 2)
    {
      for (IndexT j=blocked_L_rows[i]; j != blocked_L_rows[i+1]; ++j)
      {
        IndexT start_col = blocked_L_cols[j];
        L_cols[cnt_L]     = blocked_L_cols[j];
        L_elements[cnt_L] = blocked_L_elements[j];
        ++cnt_L;
        ++j;

        if (bsize[start_col] == 2)
        {
          L_cols[cnt_L]     = blocked_L_cols[j];
          L_elements[cnt_L] = blocked_L_elements[j];
          ++cnt_L;
          j += 2; //
        }
      }

      // add unit diagonal:
      L_cols[cnt_L] = i;
      L_elements[cnt_L] = 1.0;
      ++cnt_L;

      L_rows[i+1] = cnt_L; // start next row

      for (IndexT j=blocked_L_rows[i]; j != blocked_L_rows[i+1]; ++j)
      {
        IndexT start_col = blocked_L_cols[j];

        if (bsize[start_col] == 2)
        {
          j += 2; // second row of 2x2-block
          L_cols[cnt_L]     = blocked_L_cols[j];
          L_elements[cnt_L] = blocked_L_elements[j];
          ++cnt_L;
        }

        ++j;
        L_cols[cnt_L]     = blocked_L_cols[j];
        L_elements[cnt_L] = blocked_L_elements[j];
        ++cnt_L;
      }

      ++i;
    }

    // add unit diagonal:
    L_cols[cnt_L] = i;
    L_elements[cnt_L] = 1.0;
    ++cnt_L;
  }
  L_rows[L.size1()] = cnt_L;
}

/** @brief Computes s = a_ij - sum_k L_ik D_kk L_jk for a given index pair (i,j) for the Chow-Patel version of ILDL
*
* @param row          Row index i
* @param col          Column index j
* @param L_rows       Row-array for blocked CSR-storage of L
* @param L_cols       Column-array for blocked CSR-storage of L
* @param L_elements   Entry-array for blocked CSR-storage of L
* @param L_rows       Row-array for blocked CSR-storage of L
* @param L_cols       Column-array for blocked CSR-storage of L
* @param bsize        Vector of block sizes (1: no blocking, 2/0: blocking)
* @param s            Output vector
*/
template<typename NumericT, typename IndexT>
void compute_s(IndexT row, IndexT col,
               IndexT const * L_rows, IndexT const * L_cols, NumericT const * L_elements,
               IndexT const * D_rows, NumericT const * D_elements,
               IndexT const * bsize,
               NumericT * s)
{
  IndexT bsize_row = bsize[row];
  IndexT bsize_col = bsize[col];

  IndexT const *row_it    = L_cols + L_rows[row];
  IndexT const *L_row_end = L_cols + L_rows[row+1];

  IndexT const *col_it    = L_cols + L_rows[col];
  IndexT const *L_col_end = L_cols + L_rows[col+1];

  for (; row_it != L_row_end; ++row_it)
  {
    IndexT current_col = (col_it != L_col_end) ? *col_it : 99999999;  //TODO: Fix magic value indicating end of row
    IndexT current_row = *row_it;

    while (current_col < current_row)
    {
      ++col_it;
      current_col = (col_it != L_col_end) ? *col_it : 99999999;  //TODO: Fix magic value indicating end of row
    }

    if (current_col != current_row)
      continue;

    IndexT bsize_block = bsize[current_col];

    // Extract pointers to D_kk, L_ik, L_jk
    NumericT const * D_block = D_elements + D_rows[current_row];
    NumericT const * L_row_block = L_elements + (row_it - L_cols);
    NumericT const * L_col_block = L_elements + (col_it - L_cols);

    //
    // Compute the block update: s -= L_ik D_kk L_jk
    // (explicitly coding all 8 cases is probably more maintainable than trying to 'smartly' save a few lines of codes with tricky if-statements)
    //
    if (bsize_row == 1 && bsize_col == 1) // s is of dimension 1x1
    {
      if (bsize_block == 1)
      {
        s[0] -= L_row_block[0] * D_block[0] * L_col_block[0];
      }
      else
      {
        // D * L^T
        NumericT D_times_L[2] = {D_block[0] * L_col_block[0] + D_block[1] * L_col_block[1],
                                 D_block[2] * L_col_block[0] + D_block[3] * L_col_block[1]};
        s[0] -= L_row_block[0] * D_times_L[0] + L_row_block[1] * D_times_L[1];

        ++row_it; // 2 entries in row processed
      }
    }
    else if (bsize_row == 1 && bsize_col == 2) // s is of dimension 1x2
    {
      if (bsize_block == 1) // D is 1x1
      {
        s[0] -= L_row_block[0] * D_block[0] * L_col_block[0];
        s[1] -= L_row_block[0] * D_block[0] * L_col_block[1];
      }
      else // D is 2x2
      {
        // temp = D * L^T
        NumericT D_times_L[4] = {D_block[0] * L_col_block[0] + D_block[1] * L_col_block[1], D_block[0] * L_col_block[2] + D_block[1] * L_col_block[3],
                                 D_block[2] * L_col_block[0] + D_block[3] * L_col_block[1], D_block[2] * L_col_block[2] + D_block[3] * L_col_block[3] };

        s[0] -= L_row_block[0] * D_times_L[0] + L_row_block[1] * D_times_L[2];
        s[1] -= L_row_block[0] * D_times_L[1] + L_row_block[1] * D_times_L[3];

        ++row_it; // 2 entries in row processed
      }
    }
    else if (bsize_row == 2 && bsize_col == 1)  // s is of dimension 2x1
    {
      if (bsize_block == 1) // D is 1x1
      {
        s[0] -= L_row_block[0] * D_block[0] * L_col_block[0];
        s[1] -= L_row_block[1] * D_block[0] * L_col_block[0];

        ++row_it; // 2 entries in row processed
      }
      else // D is 2x2
      {
        // temp = D * L^T
        NumericT D_times_L[2] = {D_block[0] * L_col_block[0] + D_block[1] * L_col_block[1],
                                 D_block[2] * L_col_block[0] + D_block[3] * L_col_block[1] };

        s[0] -= L_row_block[0] * D_times_L[0] + L_row_block[1] * D_times_L[1];
        s[1] -= L_row_block[2] * D_times_L[0] + L_row_block[3] * D_times_L[1];;

        row_it += 3; // 4 entries in row processed
      }
    }
    else // (bsize_row == 2 && bsize_col == 2)   // s is of dimension 2x2
    {
      if (bsize_block == 1) // D is 1x1
      {
        s[0] -= L_row_block[0] * D_block[0] * L_col_block[0];
        s[1] -= L_row_block[0] * D_block[0] * L_col_block[1];
        s[2] -= L_row_block[1] * D_block[0] * L_col_block[0];
        s[3] -= L_row_block[1] * D_block[0] * L_col_block[1];

        ++row_it; // 2 entries in row processed
      }
      else // D is 2x2
      {
        // temp = D * L^T
        NumericT D_times_L[4] = {D_block[0] * L_col_block[0] + D_block[1] * L_col_block[1], D_block[0] * L_col_block[2] + D_block[1] * L_col_block[3],
                                 D_block[2] * L_col_block[0] + D_block[3] * L_col_block[1], D_block[2] * L_col_block[2] + D_block[3] * L_col_block[3] };

        s[0] -= L_row_block[0] * D_times_L[0] + L_row_block[1] * D_times_L[2];
        s[1] -= L_row_block[0] * D_times_L[1] + L_row_block[1] * D_times_L[3];
        s[2] -= L_row_block[2] * D_times_L[0] + L_row_block[3] * D_times_L[2];
        s[3] -= L_row_block[2] * D_times_L[1] + L_row_block[3] * D_times_L[3];

        row_it += 3; // 4 entries in row processed
      }
    }
  }

}

/** @brief Updates the triangular factor matrix according to L_ij = s * D_jj^{-1}
*
*  L_ij if of dimension 'bsize_row'-times-'bsize_col' and hence may be 1x1, 1x2, 2x1, or 2x2.
*/
template<typename NumericT, typename IndexT>
void update_L(IndexT bsize_row, IndexT bsize_col,
              NumericT const *D, NumericT const *s,
              NumericT *L)
{
  if (bsize_col == 1) // D is 1x1
  {
    L[0] = s[0] / D[0];
    if (bsize_row == 2)
      L[1] = s[1] / D[0];
  }
  else
  {
    NumericT detD = D[0] * D[3] - D[1] * D[2];
    NumericT invD[4] = {  D[3] / detD, -D[1] / detD,
                         -D[2] / detD,  D[0] / detD };

    // L = s * D^{-1}   for first row of L (either 1x2 or 2x2)
    L[0] = s[0] * invD[0] + s[1] * invD[2];
    L[1] = s[0] * invD[1] + s[1] * invD[3];

    if (bsize_row == 2) // Second row of L for the 2x2 case
    {
      L[2] = s[2] * invD[0] + s[3] * invD[2];
      L[3] = s[2] * invD[1] + s[3] * invD[3];
    }
  }
}

/** @brief Entry point for a Chow-Patel-like incomplete LDL factorization for symmetric matrices.
*
*  Supports permutations, scaling, and matching (2x2-blocks).
*  All input and output matrices in standard CSR storage.
*
*  @param A               Symmetric system matrix to be incompletely factored
*  @param L               Resulting incomplete lower-triangular factor
*  @param D               Block-diagonal (1x1 or 2x2 blocks) matrix L
*  @param scaling         Scaling vector
*  @param permutation     Permutation vector. Entry i returns the index of i after permutation
*  @param inv_permutation Inverse permutation vector
*  @param blocksize       Blocking vector for matchings of each index. 2: First row/column of a 2x2 matching, 1: No blocking for that row, 0: Second row/column of a 2x2 blocking.
*
*  @tparam NumericT       Floating point type to be used. Either 'float' or 'double'.
*  @tparam IndexT         Index (integer) type. Currently only 'unsigned int' supported.
*/
template<typename NumericT, typename IndexT>
void chow_patel_factor(viennacl::compressed_matrix<NumericT> const & A,
                       viennacl::compressed_matrix<NumericT> & L,
                       viennacl::compressed_matrix<NumericT> & D,
                       viennacl::vector<NumericT> const & scaling,
                       viennacl::vector<IndexT> const & permutation,
                       viennacl::vector<IndexT> const & inv_permutation,
                       viennacl::vector<IndexT> const & blocksize)
{
  IndexT const * bsize   = viennacl::linalg::host_based::detail::extract_raw_pointer<unsigned int>(blocksize.handle());

  viennacl::compressed_matrix<NumericT> blocked_L;
  init_blocked_L_from_A(A, blocked_L, D, scaling, permutation, inv_permutation, blocksize);

  IndexT   * D_rows      = viennacl::linalg::host_based::detail::extract_raw_pointer<unsigned int>(D.handle1());
  NumericT * D_elements  = viennacl::linalg::host_based::detail::extract_raw_pointer<NumericT    >(D.handle());

  IndexT   * L_rows      = viennacl::linalg::host_based::detail::extract_raw_pointer<unsigned int>(blocked_L.handle1());
  IndexT   * L_cols      = viennacl::linalg::host_based::detail::extract_raw_pointer<unsigned int>(blocked_L.handle2());
  NumericT * L_elements  = viennacl::linalg::host_based::detail::extract_raw_pointer<NumericT    >(blocked_L.handle());

  // hold a copy of the values of A:
  std::vector<NumericT> init_A_elements(L_rows[A.size1()]);
  for (std::size_t i=0; i<init_A_elements.size(); ++i)
    init_A_elements[i] = L_elements[i];

  // hold a copy of the values of D:
  std::vector<NumericT> init_D_elements(D_rows[A.size1()]);
  for (std::size_t i=0; i<init_D_elements.size(); ++i)
    init_D_elements[i] = D_elements[i];

  //
  // The actual work: Run several Chow-Patel sweeps:
  //
  for (std::size_t sweeps = 0; sweeps < 2; ++sweeps)
  {
    for (std::size_t i=0; i<A.size1(); ++i)
    {
      IndexT bsize_row = bsize[i];

      for (IndexT j = L_rows[i]; j < L_rows[i+1]; ++j)
      {
        IndexT col = L_cols[j];

        IndexT bsize_col = bsize[col];

        if (bsize_col == 0)
          continue;

        NumericT s[4] = {0, 0, 0, 0};

        // load values to s:
        for (IndexT k=0; k<bsize_row * bsize_col; ++k)
          s[k] = init_A_elements[j + k];

        // compute s - sum L_jk D_kl L_lk
        compute_s(IndexT(i), col,
                  L_rows, L_cols, L_elements,
                  D_rows, D_elements,
                  bsize, s);

        update_L(bsize_row, bsize_col, D_elements + D_rows[col], s, L_elements + j);
      }

      // diagonal update:
      NumericT s[4] = {0, 0, 0, 0};

      for (IndexT k=0; k<bsize_row * bsize_row; ++k)
        s[k] = init_D_elements[D_rows[i] + k];

      // compute s - sum L_jk D_kl L_lk
      compute_s(IndexT(i), IndexT(i),
                L_rows, L_cols, L_elements,
                D_rows, D_elements,
                bsize, s);

      NumericT *D = D_elements + D_rows[i]; // diagonal block

      for (IndexT k=0; k<bsize_row * bsize_row; ++k) // just copy entries over
        D[k] = s[k];

      if (bsize_row == 2) // two rows updated simultaneously
        ++i;
    } // for entries in row
  } // for i

  //
  // Final step: Convert blocked L back to CSR-L
  //
  convert_blocked_L_to_L(blocked_L, L, blocksize);
}


/**
*  Tests
**/
int main()
{
  typedef double       NumericT;

  /**
  * Set up necessary ViennaCL objects
  **/
  viennacl::compressed_matrix<NumericT> A;
  viennacl::vector<NumericT> scaling;
  viennacl::vector<unsigned int> permutation;
  viennacl::vector<unsigned int> inv_permutation;
  viennacl::vector<unsigned int> blocking;

  setup_system_3(A, scaling, permutation, inv_permutation, blocking);

  std::cout << "Matrix A:" << std::endl;
  std::cout << A << std::endl;

  std::cout << "Factorization of A... " << std::endl;
  {
    viennacl::compressed_matrix<NumericT> L;
    viennacl::compressed_matrix<NumericT> D;
    chow_patel_factor(A, L, D, scaling, permutation, inv_permutation, blocking);

    std::cout << "L: " << std::endl;
    std::cout << L << std::endl;

    std::cout << "D: " << std::endl;
    std::cout << D << std::endl;

    viennacl::compressed_matrix<NumericT> L_trans;
    viennacl::linalg::ilu_transpose(L, L_trans);
    viennacl::compressed_matrix<NumericT> temp = viennacl::linalg::prod(D, L_trans);

    viennacl::compressed_matrix<NumericT> A_approx = viennacl::linalg::prod(L, temp);
    std::cout << "L*D*L^T: " << std::endl;
    std::cout << A_approx << std::endl;
  }

  return EXIT_SUCCESS;
}

