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
      template <typename VectorType, typename ValueType, typename SizeType = std::size_t>
      class ilu_vector_range
      {
        public:
          //typedef typename VectorType::value_type      value_type;
          //typedef typename VectorType::size_type       size_type;
          
          ilu_vector_range(VectorType & v,
                           SizeType start_index,
                           SizeType vec_size
                          ) : vec_(v), start_(start_index), size_(vec_size) {}
          
          ValueType & operator()(SizeType index)
          {
            assert(index < size_ && bool("Index out of bounds!"));
            return vec_[start_ + index];  
          }
          
          ValueType & operator[](SizeType index)
          {
            assert(index < size_ && bool("Index out of bounds!"));
            return vec_[start_ + index];  
          }
          
          SizeType size() const { return size_; }
          
        private:
          VectorType & vec_;
          SizeType start_;
          SizeType size_;
      };
      
      /** @brief Extracts a diagonal block from a larger system matrix
        *
        * @param compressed_matrix   The full matrix
        * @param block_matrix        The output matrix, to which the extracted block is written to
        * @param start_index         First row- and column-index of the block
        * @param stop_index          First row- and column-index beyond the block
        */
      template <typename ScalarType>
      void extract_block_matrix(viennacl::compressed_matrix<ScalarType> const & A,
                                viennacl::compressed_matrix<ScalarType> & diagonal_block_A,
                                std::size_t start_index,
                                std::size_t stop_index
                                )
      {
        
        assert( (A.handle1().get_active_handle_id() == viennacl::MAIN_MEMORY) && bool("System matrix must reside in main memory for ILU0") );
        assert( (A.handle2().get_active_handle_id() == viennacl::MAIN_MEMORY) && bool("System matrix must reside in main memory for ILU0") );
        assert( (A.handle().get_active_handle_id() == viennacl::MAIN_MEMORY) && bool("System matrix must reside in main memory for ILU0") );
        
        ScalarType   const * A_elements   = viennacl::linalg::single_threaded::detail::extract_raw_pointer<ScalarType>(A.handle());
        unsigned int const * A_row_buffer = viennacl::linalg::single_threaded::detail::extract_raw_pointer<unsigned int>(A.handle1());
        unsigned int const * A_col_buffer = viennacl::linalg::single_threaded::detail::extract_raw_pointer<unsigned int>(A.handle2());

        ScalarType   * output_elements   = viennacl::linalg::single_threaded::detail::extract_raw_pointer<ScalarType>(diagonal_block_A.handle());
        unsigned int * output_row_buffer = viennacl::linalg::single_threaded::detail::extract_raw_pointer<unsigned int>(diagonal_block_A.handle1());
        unsigned int * output_col_buffer = viennacl::linalg::single_threaded::detail::extract_raw_pointer<unsigned int>(diagonal_block_A.handle2());
        
        std::size_t output_counter = 0;
        for (std::size_t row = start_index; row < stop_index; ++row)
        {
          unsigned int buffer_col_start = A_row_buffer[row];
          unsigned int buffer_col_end   = A_row_buffer[row+1];
          
          output_row_buffer[row - start_index] = output_counter;
          
          for (unsigned int buf_index = buffer_col_start; buf_index < buffer_col_end; ++buf_index)
          {
            unsigned int col = A_col_buffer[buf_index];
            if (col < start_index)
              continue;

            if (col >= static_cast<unsigned int>(stop_index))
              continue;

            output_col_buffer[output_counter] = col - start_index;
            output_elements[output_counter] = A_elements[buf_index];
            ++output_counter;
          }
          output_row_buffer[row - start_index + 1] = output_counter;
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
            detail::ilu_vector_range<VectorType, ScalarType>  vec_range(vec, block_indices_[i].first, LU_blocks[i].size2());
            
            unsigned int const * row_buffer = viennacl::linalg::single_threaded::detail::extract_raw_pointer<unsigned int>(LU_blocks[i].handle1());
            unsigned int const * col_buffer = viennacl::linalg::single_threaded::detail::extract_raw_pointer<unsigned int>(LU_blocks[i].handle2());
            ScalarType   const * elements   = viennacl::linalg::single_threaded::detail::extract_raw_pointer<ScalarType>(LU_blocks[i].handle());
            
            viennacl::linalg::single_threaded::detail::csr_inplace_solve<ScalarType>(row_buffer, col_buffer, elements, vec_range, LU_blocks[i].size2(), unit_lower_tag());
            viennacl::linalg::single_threaded::detail::csr_inplace_solve<ScalarType>(row_buffer, col_buffer, elements, vec_range, LU_blocks[i].size2(), upper_tag());
            
          }
        }
        
      private:
        void init(MatrixType const & A)
        {
          
          viennacl::compressed_matrix<ScalarType> mat(A.size1(), A.size2());
          viennacl::switch_memory_domain(mat, viennacl::MAIN_MEMORY);
          
          viennacl::copy(A, mat);
          
          unsigned int const * row_buffer = viennacl::linalg::single_threaded::detail::extract_raw_pointer<unsigned int>(mat.handle1());
          
#ifdef VIENNACL_WITH_OPENMP
          #pragma omp parallel for
#endif
          for (std::size_t i=0; i<block_indices_.size(); ++i)
          {
            // Step 1: Extract blocks
            std::size_t block_size = block_indices_[i].second - block_indices_[i].first;
            std::size_t block_nnz  = row_buffer[block_indices_[i].second] - row_buffer[block_indices_[i].first];
            viennacl::compressed_matrix<ScalarType> mat_block(block_size, block_size, block_nnz);
            viennacl::switch_memory_domain(mat_block, viennacl::MAIN_MEMORY);
            
            detail::extract_block_matrix(mat, mat_block, block_indices_[i].first, block_indices_[i].second);
            
            // Step 2: Precondition blocks:
            viennacl::switch_memory_domain(LU_blocks[i], viennacl::MAIN_MEMORY);
            preconditioner_dispatch(mat_block, LU_blocks[i], tag_);
          }
          
        }
        
        void preconditioner_dispatch(viennacl::compressed_matrix<ScalarType> const & mat_block,
                                     viennacl::compressed_matrix<ScalarType> & LU,
                                     viennacl::linalg::ilu0_tag)
        {
          LU = mat_block;
          viennacl::linalg::precondition(LU, tag_);
        }

        void preconditioner_dispatch(viennacl::compressed_matrix<ScalarType> const & mat_block,
                                     viennacl::compressed_matrix<ScalarType> & LU,
                                     viennacl::linalg::ilut_tag)
        {
          std::vector< std::map<unsigned int, ScalarType> > temp(mat_block.size1());
          
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



