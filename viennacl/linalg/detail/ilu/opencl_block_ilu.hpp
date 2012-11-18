#ifndef VIENNACL_LINALG_DETAIL_OPENCL_BLOCK_ILU_HPP_
#define VIENNACL_LINALG_DETAIL_OPENCL_BLOCK_ILU_HPP_

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

/** @file viennacl/linalg/detail/ilu/opencl_block_ilu.hpp
    @brief Implementations of incomplete block factorization preconditioners using OpenCL
*/

#include <vector>
#include <cmath>
#include "viennacl/forwards.h"
#include "viennacl/tools/tools.hpp"
#include "viennacl/linalg/detail/ilu/common.hpp"
#include "viennacl/linalg/detail/ilu/ilu0.hpp"
#include "viennacl/linalg/detail/ilu/ilut.hpp"
#include "viennacl/linalg/detail/ilu/host_block_ilu.hpp"

#include "viennacl/linalg/kernels/ilu_kernels.h"

#include <map>

namespace viennacl
{
  namespace linalg
  {
    
    /** @brief ILUT preconditioner class, can be supplied to solve()-routines.
    *
    *  Specialization for compressed_matrix
    */
    template <typename ScalarType, unsigned int MAT_ALIGNMENT, typename ILUTag>
    class block_ilu_precond< compressed_matrix<ScalarType, MAT_ALIGNMENT>, ILUTag >
    {
        typedef compressed_matrix<ScalarType, MAT_ALIGNMENT>        MatrixType;
        //typedef std::vector<ScalarType>                             STLVectorType;
      
      public:
        typedef std::vector<std::pair<std::size_t, std::size_t> >    index_vector_type;   //the pair refers to index range [a, b) of each block
          
        
        block_ilu_precond(MatrixType const & mat,
                          ILUTag const & tag,
                          std::size_t num_blocks = 4
                         ) : tag_(tag), block_indices_(num_blocks),
                             gpu_block_indices(0),
                             gpu_L_trans(0,0),
                             gpu_U_trans(0,0),
                             gpu_D(mat.size1()),
                             LU_blocks(num_blocks)
        {
          viennacl::linalg::kernels::ilu<ScalarType, 1>::init();
          
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
                         ) : tag_(tag), block_indices_(block_boundaries),
                             gpu_block_indices(0),
                             gpu_L_trans(0,0),
                             gpu_U_trans(0,0),
                             gpu_D(mat.size1()),
                             LU_blocks(block_boundaries.size())
        {
          viennacl::linalg::kernels::ilu<ScalarType, 1>::init();
          
          //initialize preconditioner:
          //std::cout << "Start CPU precond" << std::endl;
          init(mat);          
          //std::cout << "End CPU precond" << std::endl;
        }
        
        
        void apply(vector<ScalarType> & vec) const
        {
          apply_gpu(vec);
        }

        // GPU application (default)
        void apply_gpu(vector<ScalarType> & vec) const
        {
          viennacl::ocl::kernel & block_ilut_kernel =
               viennacl::ocl::get_kernel(viennacl::linalg::kernels::ilu<ScalarType, 1>::program_name(), "block_ilu_substitute");

          viennacl::ocl::enqueue(block_ilut_kernel(gpu_L_trans.handle1().opencl_handle(),  //L
                                                   gpu_L_trans.handle2().opencl_handle(),
                                                   gpu_L_trans.handle().opencl_handle(),
                                                   gpu_U_trans.handle1().opencl_handle(),  //U
                                                   gpu_U_trans.handle2().opencl_handle(),
                                                   gpu_U_trans.handle().opencl_handle(),
                                                   gpu_D,                  //D
                                                   gpu_block_indices,
                                                   vec,
                                                   static_cast<cl_uint>(vec.size())));
        }

        // CPU fallback:
        void apply_cpu(vector<ScalarType> & vec) const
        {
          if (vec.handle().get_active_handle_id() != viennacl::MAIN_MEMORY)
          {
            viennacl::memory_types old_memory_location = viennacl::memory_domain(vec);
            viennacl::switch_memory_domain(vec, viennacl::MAIN_MEMORY);
            
            ScalarType * vector_entries = viennacl::linalg::single_threaded::detail::extract_raw_pointer<ScalarType>(vec);
            
            for (std::size_t i=0; i<block_indices_.size(); ++i)
            {
              detail::ilu_vector_range<ScalarType *, ScalarType>  vec_range(vector_entries, block_indices_[i].first, LU_blocks[i].size2());
              
              unsigned int const * row_buffer = viennacl::linalg::single_threaded::detail::extract_raw_pointer<unsigned int>(LU_blocks[i].handle1());
              unsigned int const * col_buffer = viennacl::linalg::single_threaded::detail::extract_raw_pointer<unsigned int>(LU_blocks[i].handle2());
              ScalarType   const * elements   = viennacl::linalg::single_threaded::detail::extract_raw_pointer<ScalarType>(LU_blocks[i].handle());
              
              viennacl::linalg::single_threaded::detail::csr_inplace_solve<ScalarType>(row_buffer, col_buffer, elements, vec_range, LU_blocks[i].size2(), unit_lower_tag());
              viennacl::linalg::single_threaded::detail::csr_inplace_solve<ScalarType>(row_buffer, col_buffer, elements, vec_range, LU_blocks[i].size2(), upper_tag());
            }
            
            viennacl::switch_memory_domain(vec, old_memory_location);
          }
          else //apply directly:
          {
            ScalarType * vector_entries = viennacl::linalg::single_threaded::detail::extract_raw_pointer<ScalarType>(vec);
            
            for (std::size_t i=0; i<block_indices_.size(); ++i)
            {
              detail::ilu_vector_range<ScalarType *, ScalarType>  vec_range(vector_entries, block_indices_[i].first, LU_blocks[i].size2());
              
              unsigned int const * row_buffer = viennacl::linalg::single_threaded::detail::extract_raw_pointer<unsigned int>(LU_blocks[i].handle1());
              unsigned int const * col_buffer = viennacl::linalg::single_threaded::detail::extract_raw_pointer<unsigned int>(LU_blocks[i].handle2());
              ScalarType   const * elements   = viennacl::linalg::single_threaded::detail::extract_raw_pointer<ScalarType>(LU_blocks[i].handle());
              
              viennacl::linalg::single_threaded::detail::csr_inplace_solve<ScalarType>(row_buffer, col_buffer, elements, vec_range, LU_blocks[i].size2(), unit_lower_tag());
              viennacl::linalg::single_threaded::detail::csr_inplace_solve<ScalarType>(row_buffer, col_buffer, elements, vec_range, LU_blocks[i].size2(), upper_tag());
            }
          }
          
        }
        
      private:
        
        void init(MatrixType const & A)
        {
          viennacl::compressed_matrix<ScalarType> mat;
          viennacl::switch_memory_domain(mat, viennacl::MAIN_MEMORY);
          
          mat = A;
          
          unsigned int const * row_buffer = viennacl::linalg::single_threaded::detail::extract_raw_pointer<unsigned int>(mat.handle1());
          
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
          
          /*
           * copy resulting preconditioner back to GPU:
           */
          
          std::vector<cl_uint> block_indices_uint(2 * block_indices_.size());
          for (std::size_t i=0; i<block_indices_.size(); ++i)
          {
            block_indices_uint[2*i]     = static_cast<cl_uint>(block_indices_[i].first);
            block_indices_uint[2*i + 1] = static_cast<cl_uint>(block_indices_[i].second);
          }

          gpu_block_indices = viennacl::ocl::current_context().create_memory(CL_MEM_READ_WRITE,
                                                                             sizeof(cl_uint) * block_indices_uint.size(),
                                                                             &(block_indices_uint[0]) );
          
          blocks_to_GPU(mat.size1());

          //set kernel parameters:
          viennacl::ocl::kernel & block_ilut_kernel =
               viennacl::ocl::get_kernel(viennacl::linalg::kernels::ilu<ScalarType, 1>::program_name(), "block_ilu_substitute");
          
          block_ilut_kernel.global_work_size(0, 128 * block_indices_.size() );
          block_ilut_kernel.local_work_size(0, 128);
        }
        
        // Copy computed preconditioned blocks to OpenCL device
        void blocks_to_GPU(std::size_t matrix_size)
        {
          std::vector< std::map<cl_uint, ScalarType> > L_transposed(matrix_size);
          std::vector< std::map<cl_uint, ScalarType> > U_transposed(matrix_size);
          std::vector<ScalarType> entries_D(matrix_size);
          
          //
          // Transpose individual blocks into a single large matrix:
          //
          for (std::size_t block_index = 0; block_index < LU_blocks.size(); ++block_index)
          {
            MatrixType const & current_block = LU_blocks[block_index];
            
            unsigned int const * row_buffer = viennacl::linalg::single_threaded::detail::extract_raw_pointer<unsigned int>(current_block.handle1());
            unsigned int const * col_buffer = viennacl::linalg::single_threaded::detail::extract_raw_pointer<unsigned int>(current_block.handle2());
            ScalarType   const * elements   = viennacl::linalg::single_threaded::detail::extract_raw_pointer<ScalarType>(current_block.handle());
            
            std::size_t block_start = block_indices_[block_index].first;
            
            //transpose L and U:
            for (std::size_t row = 0; row < current_block.size1(); ++row)
            {
              unsigned int buffer_col_start = row_buffer[row];
              unsigned int buffer_col_end   = row_buffer[row+1];
              
              for (unsigned int buf_index = buffer_col_start; buf_index < buffer_col_end; ++buf_index)
              {
                unsigned int col = col_buffer[buf_index];
                
                if (row > col) //entry for L
                  L_transposed[col + block_start][row + block_start] = elements[buf_index];
                else if (row == col)
                  entries_D[row + block_start] = elements[buf_index];
                else //entry for U
                  U_transposed[col + block_start][row + block_start] = elements[buf_index];
              }
            }
          }
          
          //
          // Move data to GPU:
          //
          viennacl::copy(L_transposed, gpu_L_trans);
          viennacl::copy(U_transposed, gpu_U_trans);
          viennacl::copy(entries_D, gpu_D);
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
        viennacl::ocl::handle<cl_mem> gpu_block_indices;
        viennacl::compressed_matrix<ScalarType> gpu_L_trans;
        viennacl::compressed_matrix<ScalarType> gpu_U_trans;
        viennacl::vector<ScalarType> gpu_D;
        
        std::vector< MatrixType > LU_blocks;
    };

  }
}




#endif



