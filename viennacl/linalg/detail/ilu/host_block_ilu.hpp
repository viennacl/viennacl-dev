#ifndef VIENNACL_LINALG_DETAIL_HOST_BLOCK_ILU_HPP_
#define VIENNACL_LINALG_DETAIL_HOST_BLOCK_ILU_HPP_

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

/** @file viennacl/linalg/detail/ilu/host_block_ilu.hpp
    @brief Implementations of incomplete block factorization preconditioners (host-based, no OpenCL)
*/

#include <vector>
#include <cmath>
#include "viennacl/forwards.h"
#include "viennacl/tools/tools.hpp"
#include "viennacl/linalg/detail/ilu/common.hpp"
#include "viennacl/linalg/detail/ilu/ilu0.hpp"
#include "viennacl/linalg/detail/ilu/ilut.hpp"

#include <map>

namespace viennacl
{
  namespace linalg
  {
    namespace detail
    {
      template <typename VectorType>
      class ilu_vector_range
      {
        public:
          typedef typename VectorType::value_type      value_type;
          typedef typename VectorType::size_type       size_type;
          
          ilu_vector_range(VectorType & v,
                           size_type start_index,
                           size_type vec_size
                          ) : vec_(v), start_(start_index), size_(vec_size) {}
          
          value_type & operator()(size_type index)
          {
            assert(index < size_ && bool("Index out of bounds!"));
            
            return vec_[start_ + index];  
          }
          
          value_type & operator[](size_type index)
          {
            return this->operator()(index);
          }
          
          size_type size() const { return size_; }
          
        private:
          VectorType & vec_;
          size_type start_;
          size_type size_;
      };
      
      /** @brief Extracts a diagonal block from a larger system matrix
        *
        * @param compressed_matrix   The full matrix
        * @param block_matrix        The output matrix, to which the extracted block is written to
        * @param start_index         First row- and column-index of the block
        * @param stop_index          First row- and column-index beyond the block
        */
      template <typename MatrixType, typename STLMatrixType>
      void extract_block_matrix(MatrixType const & compressed_matrix,
                                STLMatrixType & block_matrix,
                                std::size_t start_index,
                                std::size_t stop_index
                                )
      {
        typedef typename MatrixType::const_iterator1     RowIterator;
        typedef typename MatrixType::const_iterator2     ColumnIterator;

        for (RowIterator row_iter = compressed_matrix.begin1();
                        row_iter != compressed_matrix.end1();
                      ++row_iter)
        {
          if (row_iter.index1() < start_index)
            continue;

          if (row_iter.index1() >= stop_index)
            break;

          for (ColumnIterator col_iter = row_iter.begin();
                              col_iter != row_iter.end();
                            ++col_iter)
          {
            if (col_iter.index2() < start_index)
              continue;

            if (col_iter.index2() >= static_cast<std::size_t>(stop_index))
              continue;

            block_matrix[col_iter.index1() - start_index][col_iter.index2() - start_index] = *col_iter;
          }
        }
      }
          
      
    }

    /** @brief A block ILU preconditioner class, can be supplied to solve()-routines
     * 
     * @tparam MatrixType   Type of the system matrix
     * @tparam ILUTag       Type of the tag identifiying the ILU preconditioner to be used on each block.
    */
    template <typename MatrixType, typename ILUTag>
    class block_ilu_precond
    {
      typedef typename MatrixType::value_type      ScalarType;
      
      public:
        typedef std::vector<std::pair<std::size_t, std::size_t> >    index_vector_type;   //the pair refers to index range [a, b) of each block
        
        
        block_ilu_precond(MatrixType const & mat,
                          ILUTag const & tag,
                          std::size_t num_blocks = 4
                         ) : tag_(tag), LU_blocks(num_blocks)
        {
          
          // Set up vector of block indices:
          block_indices_.resize(num_blocks);
          for (std::size_t i=0; i<num_blocks; ++i)
          {
            std::size_t start_index = (   i  * mat.size1()) / num_blocks;
            std::size_t stop_index  = ((i+1) * mat.size1()) / num_blocks;
            
            block_indices_[i] = std::pair<std::size_t, std::size_t>(start_index, stop_index);
          }
          
          //initialize preconditioner:
          //std::cout << "Start CPU precond" << std::endl;
          init(mat);          
          //std::cout << "End CPU precond" << std::endl;
        }

        block_ilu_precond(MatrixType const & mat,
                          ILUTag const & tag,
                          index_vector_type const & block_boundaries
                         ) : tag_(tag), block_indices_(block_boundaries), LU_blocks(block_boundaries.size())
        {
          //initialize preconditioner:
          //std::cout << "Start CPU precond" << std::endl;
          init(mat);          
          //std::cout << "End CPU precond" << std::endl;
        }
        
        
        template <typename VectorType>
        void apply(VectorType & vec) const
        {
          for (std::size_t i=0; i<block_indices_.size(); ++i)
          {
            /*viennacl::tools::const_sparse_matrix_adapter<ScalarType> LU_const_adapter(LU_blocks[i],
                                                                                      LU_blocks[i].size(),
                                                                                      LU_blocks[i].size());
            viennacl::linalg::detail::ilu_lu_substitute(LU_const_adapter, vec_range);*/
            detail::ilu_vector_range<VectorType>  vec_range(vec,
                                                            block_indices_[i].first,
                                                            LU_blocks[i].size2());
            
            unsigned int const * row_buffer = viennacl::linalg::single_threaded::detail::extract_raw_pointer<unsigned int>(LU_blocks[i].handle1());
            unsigned int const * col_buffer = viennacl::linalg::single_threaded::detail::extract_raw_pointer<unsigned int>(LU_blocks[i].handle2());
            ScalarType   const * elements   = viennacl::linalg::single_threaded::detail::extract_raw_pointer<ScalarType>(LU_blocks[i].handle());
            
            viennacl::linalg::single_threaded::detail::csr_inplace_solve<ScalarType>(row_buffer, col_buffer, elements, vec_range, LU_blocks[i].size2(), unit_lower_tag());
            viennacl::linalg::single_threaded::detail::csr_inplace_solve<ScalarType>(row_buffer, col_buffer, elements, vec_range, LU_blocks[i].size2(), upper_tag());
            
          }
        }
        
      private:
        void init(MatrixType const & mat)
        {
#ifdef VIENNACL_WITH_OPENMP
          #pragma omp parallel for
#endif
          for (std::size_t i=0; i<block_indices_.size(); ++i)
          {
            // Step 1: Extract blocks
            std::size_t block_size = block_indices_[i].second - block_indices_[i].first;
            std::vector< std::map<unsigned int, ScalarType> > mat_block(block_size);
            detail::extract_block_matrix(mat, mat_block, block_indices_[i].first, block_indices_[i].second);
            
            
            // Step 2: Precondition blocks:
            viennacl::switch_memory_domain(LU_blocks[i], viennacl::MAIN_MEMORY);
            preconditioner_dispatch(mat_block, LU_blocks[i], tag_);
          }
          
        }
        
        void preconditioner_dispatch(std::vector< std::map<unsigned int, ScalarType> > const & mat_block,
                                     viennacl::compressed_matrix<ScalarType> & LU,
                                     viennacl::linalg::ilu0_tag)
        {

          viennacl::copy(mat_block, LU);
          viennacl::linalg::precondition(LU, tag_);
        }

        void preconditioner_dispatch(std::vector< std::map<unsigned int, ScalarType> > const & mat_block,
                                     viennacl::compressed_matrix<ScalarType> & LU,
                                     viennacl::linalg::ilut_tag)
        {
          std::vector< std::map<unsigned int, ScalarType> > temp(mat_block.size());
          
          viennacl::linalg::precondition(mat_block, temp, tag_);
          
          viennacl::copy(temp, LU);
        }
        
        ILUTag const & tag_;
        index_vector_type block_indices_;
        std::vector< viennacl::compressed_matrix<ScalarType> > LU_blocks;
    };


  }
}




#endif



