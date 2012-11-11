#ifndef VIENNACL_LINALG_DETAIL_ILU_COMMON_HPP_
#define VIENNACL_LINALG_DETAIL_ILU_COMMON_HPP_

/* =========================================================================
   Copyright (c) 2010-2012, Institute for Microelectronics,
                            Institute for Analysis and Scientific Computing,
                            TU Wien.

                            -----------------
                  ViennaCL - The Vienna Computing Library
                            -----------------

   Project Head:    Karl Rupp                   rupp@iue.tuwien.ac.at
               
   (A list of authors and contributors can be found in the PDF manual)

   License:         MIT (X11), see file LICENSE in the base directory
============================================================================= */

/** @file viennacl/linalg/detail/ilu/common.hpp
    @brief Common routines used within ILU-type preconditioners
*/

#include <vector>
#include <cmath>
#include <iostream>
#include <map>
#include <list>

#include "viennacl/forwards.h"
#include "viennacl/tools/tools.hpp"
#include "viennacl/backend/memory.hpp"


namespace viennacl
{
  namespace linalg
  {
    namespace detail
    {
    
      /** @brief Increments a row iterator (iteration along increasing row indices) up to a certain row index k.
      * 
      * Generic implementation using the iterator concept from boost::numeric::ublas. Could not find a better way for sparse matrices...
      *
      * @param row_iter   The row iterator
      * @param k      The final row index
      */
      template <typename T>
      void ilu_inc_row_iterator_to_row_index(T & row_iter, unsigned int k)
      {
        while (row_iter.index1() < k)
          ++row_iter;
      }
      
      /** @brief Increments a row iterator (iteration along increasing row indices) up to a certain row index k.
      * 
      * Specialization for the sparse matrix adapter shipped with ViennaCL
      *
      * @param row_iter   The row iterator
      * @param k      The final row index
      */
      template <typename ScalarType>
      void ilu_inc_row_iterator_to_row_index(viennacl::tools::sparse_matrix_adapter<ScalarType> & row_iter, unsigned int k)
      {
        row_iter += k - row_iter.index1();
      }
      
      /** @brief Increments a row iterator (iteration along increasing row indices) up to a certain row index k.
      * 
      * Specialization for the const sparse matrix adapter shipped with ViennaCL
      *
      * @param row_iter   The row iterator
      * @param k      The final row index
      */
      template <typename ScalarType>
      void ilu_inc_row_iterator_to_row_index(viennacl::tools::const_sparse_matrix_adapter<ScalarType> & row_iter, unsigned int k)
      {
        row_iter += k - row_iter.index1();
      }

      /** @brief Generic inplace solution of a unit lower triangular system
      *   
      * @param mat  The system matrix
      * @param vec  The right hand side vector
      */
      template<typename MatrixType, typename VectorType>
      void ilu_inplace_solve(MatrixType const & mat, VectorType & vec, viennacl::linalg::unit_lower_tag)
      {
        typedef typename MatrixType::const_iterator1    InputRowIterator;  //iterate along increasing row index
        typedef typename MatrixType::const_iterator2    InputColIterator;  //iterate along increasing column index
        
        for (InputRowIterator row_iter = mat.begin1(); row_iter != mat.end1(); ++row_iter)
        {
          for (InputColIterator col_iter = row_iter.begin(); col_iter != row_iter.end(); ++col_iter)
          {
            if (col_iter.index2() < col_iter.index1())
              vec[col_iter.index1()] -= *col_iter * vec[col_iter.index2()];
          }
        }
      }

      /** @brief Generic inplace solution of a upper triangular system
      *   
      * @param mat  The system matrix
      * @param vec  The right hand side vector
      */
      template<typename MatrixType, typename VectorType>
      void ilu_inplace_solve(MatrixType const & mat, VectorType & vec, viennacl::linalg::upper_tag)
      {
        typedef typename MatrixType::const_reverse_iterator1    InputRowIterator;  //iterate along increasing row index
        typedef typename MatrixType::const_iterator2            InputColIterator;  //iterate along increasing column index
        typedef typename VectorType::value_type                 ScalarType;
        
        ScalarType diagonal_entry = 1.0;
        
        for (InputRowIterator row_iter = mat.rbegin1(); row_iter != mat.rend1(); ++row_iter)
        {
          for (InputColIterator col_iter = row_iter.begin(); col_iter != row_iter.end(); ++col_iter)
          {
            if (col_iter.index2() > col_iter.index1())
              vec[col_iter.index1()] -= *col_iter * vec[col_iter.index2()];
            if (col_iter.index2() == col_iter.index1())
              diagonal_entry = *col_iter;
          }
          vec[row_iter.index1()] /= diagonal_entry;
        }
      }

      /** @brief Generic LU substitution
      *   
      * @param mat  The system matrix
      * @param vec  The right hand side vector
      */
      template<typename MatrixType, typename VectorType>
      void ilu_lu_substitute(MatrixType const & mat, VectorType & vec)
      {
        ilu_inplace_solve(mat, vec, unit_lower_tag());
        ilu_inplace_solve(mat, vec, upper_tag());
      }

      
      
      /** @brief Builds a dependency graph for fast LU substitutions of sparse factors on GPU */
      template<typename ScalarType, unsigned int MAT_ALIGNMENT, unsigned int VEC_ALIGNMENT>
      void ilu_inplace_solve(matrix_expression< const compressed_matrix<ScalarType, MAT_ALIGNMENT>,
                                                const compressed_matrix<ScalarType, MAT_ALIGNMENT>,
                                                op_trans> const & proxy_L,
                             vector<ScalarType, VEC_ALIGNMENT> & vec,
                             viennacl::linalg::unit_lower_tag)
      {
        ScalarType         * vec_buf    = NULL;//detail::extract_raw_pointer<ScalarType>(vec.handle());
        ScalarType   const * elements   = NULL;//detail::extract_raw_pointer<ScalarType>(proxy_L.lhs().handle());
        unsigned int const * row_buffer = NULL;//detail::extract_raw_pointer<unsigned int>(proxy_L.lhs().handle1());
        unsigned int const * col_buffer = NULL;//detail::extract_raw_pointer<unsigned int>(proxy_L.lhs().handle2());
        
        std::cout << "Building dependency graph..." << std::endl;
        std::vector<std::size_t> row_elimination(proxy_L.lhs().size1());
        
        std::list< viennacl::backend::mem_handle > row_index_arrays;
        std::list< viennacl::backend::mem_handle > row_buffers;
        std::list< viennacl::backend::mem_handle > col_buffers;
        std::list< viennacl::backend::mem_handle > element_buffers;
        std::list< std::size_t > row_elimination_num_list;
        
        std::size_t summed_rows = 0;
        for (std::size_t elimination_run = 0; elimination_run < proxy_L.lhs().size1(); ++elimination_run)
        {
          std::size_t eliminated_rows_in_run = 0;
          std::vector<std::map<unsigned int, ScalarType> > transposed_elimination_matrix(proxy_L.lhs().size1());
          
          // tag columns which depend on current elimination run:
          for (std::size_t col = 0; col < proxy_L.lhs().size1(); ++col)
          {
            if (row_elimination[col] < elimination_run) //row already eliminated or dependent on existing run
              continue;
            
            std::size_t col_end = row_buffer[col+1];
            for (std::size_t i = row_buffer[col]; i < col_end; ++i)
            {
              unsigned int row_index = col_buffer[i];
              if (row_index > col)
              {
                row_elimination[row_index] = row_elimination[col] + 1; //col needs to wait
                
                if (row_elimination[col] == elimination_run)
                  transposed_elimination_matrix[row_index][col] = elements[i];
              }
            }
          }

          // count cols to be eliminated in this run
          for (std::size_t col = 0; col < proxy_L.lhs().size1(); ++col)
          {
            if (row_elimination[col] == elimination_run)
              ++eliminated_rows_in_run;
          }
          
          std::size_t num_tainted_cols = 0;
          std::size_t num_entries = 0;
          for (std::size_t i=0; i<transposed_elimination_matrix.size(); ++i)
          {
            num_entries += transposed_elimination_matrix[i].size();
            if (transposed_elimination_matrix[i].size() > 0)
              ++num_tainted_cols;
          }
          //std::cout << "num_entries: " << num_entries << std::endl;
          //std::cout << "num_tainted_cols: " << num_tainted_cols << std::endl;
          
          if (num_tainted_cols > 0)
          {
            row_index_arrays.push_back(viennacl::backend::mem_handle());
            viennacl::backend::integral_type_host_array<unsigned int> row_index_array(row_index_arrays.back(), num_tainted_cols);
            
            row_buffers.push_back(viennacl::backend::mem_handle());
            viennacl::backend::integral_type_host_array<unsigned int> row_buffer(row_buffers.back(), num_tainted_cols + 1);
            
            col_buffers.push_back(viennacl::backend::mem_handle());
            viennacl::backend::integral_type_host_array<unsigned int> col_buffer(col_buffers.back(), num_entries);
            
            element_buffers.push_back(viennacl::backend::mem_handle());
            std::vector<ScalarType> elements_buffer(num_entries);
            
            row_elimination_num_list.push_back(num_tainted_cols);
            
            std::size_t k=0;
            std::size_t nnz_index = 0;
            row_buffer.set(0, 0);
            for (std::size_t i=0; i<transposed_elimination_matrix.size(); ++i)
            {
              if (transposed_elimination_matrix[i].size() > 0)
              {
                row_index_array.set(k, i);
                for (typename std::map<unsigned int, ScalarType>::const_iterator it = transposed_elimination_matrix[i].begin();
                                                                                it != transposed_elimination_matrix[i].end();
                                                                              ++it)
                {
                  col_buffer.set(nnz_index, it->first);
                  elements_buffer[nnz_index] = it->second;
                  ++nnz_index;
                }
                row_buffer.set(++k, nnz_index);
              }
            }
          
            //
            // Wrap in memory_handles:
            //
            viennacl::backend::memory_create(row_index_arrays.back(), row_index_array.raw_size(),                  row_index_array.get());
            viennacl::backend::memory_create(row_buffers.back(),      row_buffer.raw_size(),                       row_buffer.get());
            viennacl::backend::memory_create(col_buffers.back(),      col_buffer.raw_size(),                       col_buffer.get());
            viennacl::backend::memory_create(element_buffers.back(),  sizeof(ScalarType) * elements_buffer.size(), &(elements_buffer[0]));
          }

          // Print some info:
          std::cout << "Eliminated columns in run " << elimination_run << ": " << eliminated_rows_in_run << " (tainted columns: " << num_tainted_cols << ")" << std::endl;
          summed_rows += eliminated_rows_in_run;
          if (eliminated_rows_in_run == 0)
            break;
        }
        std::cout << "Eliminated rows: " << summed_rows << " out of " << row_elimination.size() << std::endl;
        
        
        //
        // Multifrontal substitution:
        //
        typedef typename std::list< viennacl::backend::mem_handle >::iterator  ListIterator;
        ListIterator row_index_array_it = row_index_arrays.begin();
        ListIterator row_buffers_it = row_buffers.begin();
        ListIterator col_buffers_it = col_buffers.begin();
        ListIterator element_buffers_it = element_buffers.begin();
        typename std::list< std::size_t>::iterator row_elimination_num_it = row_elimination_num_list.begin();
        for (std::size_t i=0; i<row_index_arrays.size(); ++i)
        {
          unsigned int const * elim_row_index = NULL;//detail::extract_raw_pointer<unsigned int>(*row_index_array_it);
          unsigned int const * elim_row_buffer = NULL;//detail::extract_raw_pointer<unsigned int>(*row_buffers_it);
          unsigned int const * elim_col_buffer = NULL;//detail::extract_raw_pointer<unsigned int>(*col_buffers_it);
          ScalarType   const * elim_elements   = NULL;//detail::extract_raw_pointer<ScalarType>(*element_buffers_it);
          
          for (std::size_t row=0; row < *row_elimination_num_it; ++row)
          {
            ScalarType vec_entry = vec_buf[elim_row_index[row]];
            unsigned int row_end = elim_row_buffer[row+1];
            for (std::size_t j = elim_row_buffer[row]; j < row_end; ++j)
              vec_entry -= vec_buf[elim_col_buffer[j]] * elim_elements[j];
            vec_buf[elim_row_index[row]] = vec_entry;
          }
          
          ++row_index_array_it;
          ++row_buffers_it;
          ++col_buffers_it;
          ++element_buffers_it;
          ++row_elimination_num_it;
        }
      }
      
      
    } // namespace detail
  } // namespace linalg
} // namespace viennacl




#endif



