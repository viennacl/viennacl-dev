
#ifndef VIENNACL_LINALG_ICHOL_HPP_
#define VIENNACL_LINALG_ICHOL_HPP_

/* =========================================================================
   Copyright (c) 2010-2011, Institute for Microelectronics,
   Institute for Analysis and Scientific Computing,
   TU Wien.

   -----------------
   ViennaCL - The Vienna Computing Library
   -----------------

   Project Head:    Karl Rupp                   rupp@iue.tuwien.ac.at

   (A list of authors and contributors can be found in the PDF manual)

License:         MIT (X11), see file LICENSE in the base directory
============================================================================= */

/** @file viennacl/linalg/ichol.hpp
  @brief Implementations of incomplete Cholesky factorization preconditioners with static nonzero pattern.
*/

#include <vector>
#include <cmath>
#include <iostream>
#include "viennacl/forwards.h"
#include "viennacl/tools/tools.hpp"
#include "viennacl/compressed_matrix.hpp"

#include "viennacl/linalg/single_threaded/common.hpp"

#include <map>

namespace viennacl
{
  namespace linalg
  {

    /** @brief A tag for incomplete Cholesky factorization with static pattern (ILU0)
    */
    class ichol0_tag {};

    
    /** @brief Implementation of a ILU-preconditioner with static pattern. Optimized version for CSR matrices.
      *
      *  Refer to Chih-Jen Lin and Jorge J. Moré, Incomplete Cholesky Factorizations with Limited Memory, SIAM J. Sci. Comput., 21(1), 24–45
      *  for one of many descriptions of incomplete Cholesky Factorizations
      *
      *  @param A       The input matrix in CSR format
      *  @param tag     An ichol0_tag in order to dispatch among several other preconditioners.
      */
    template<typename ScalarType>
    void precondition(viennacl::compressed_matrix<ScalarType> & A, ichol0_tag const & tag)
    {
      assert( (A.handle1().get_active_handle_id == viennacl::backend::MAIN_MEMORY) && bool("System matrix must reside in main memory for ILU0") );
      assert( (A.handle2().get_active_handle_id == viennacl::backend::MAIN_MEMORY) && bool("System matrix must reside in main memory for ILU0") );
      assert( (A.handle().get_active_handle_id == viennacl::backend::MAIN_MEMORY) && bool("System matrix must reside in main memory for ILU0") );
      
      ScalarType         * elements   = viennacl::linalg::single_threaded::detail::extract_raw_pointer<ScalarType>(A.handle());
      unsigned int const * row_buffer = viennacl::linalg::single_threaded::detail::extract_raw_pointer<unsigned int>(A.handle1());
      unsigned int const * col_buffer = viennacl::linalg::single_threaded::detail::extract_raw_pointer<unsigned int>(A.handle2());
      
      for (std::size_t i=1; i<A.size1(); ++i)
      {
        unsigned int row_i_begin = row_buffer[i];
        unsigned int row_i_end   = row_buffer[i+1];
        
        // get a_ii:
        ScalarType a_ii = 0;
        for (unsigned int buf_index_aii = row_i_begin; buf_index_aii < row_i_end; ++buf_index_aii)
        {
          if (col_buffer[buf_index_aii] == i)
          {
            a_ii = std::sqrt(elements[buf_index_aii]);
            elements[buf_index_aii] = a_ii;
            break;
          }
        }
          
        // Now scale column/row i, i.e. A(k, i) /= A(i, i)
        for (unsigned int buf_index_aii = row_i_begin; buf_index_aii < row_i_end; ++buf_index_aii)
        {
          if (col_buffer[buf_index_aii] > i)
            elements[buf_index_aii] /= a_ii;
        }
        
        // Update all columns/rows with higher index than i:
        for (std::size_t j = i+1; j < A.size1(); ++j)
        {
          unsigned int row_j_begin = row_buffer[j];
          unsigned int row_j_end   = row_buffer[j+1];

          // Step through the elements of row/column j and update them accordingly:
          for (unsigned int buf_index_j = row_j_begin; buf_index_j < row_j_end; ++buf_index_j)
          {
            unsigned int k = col_buffer[buf_index_j];
            if (k < j)
              continue;
            
            ScalarType a_ki = 0;
            ScalarType a_ji = 0;
          
            // find a_ki and a_ji and row/column i:
            for (unsigned int buf_index_i = row_i_begin; buf_index_i < row_i_end; ++buf_index_i)
            {
              if (col_buffer[buf_index_i] == k)
                a_ki = elements[buf_index_i];
              if (col_buffer[buf_index_i] == j)
                a_ji = elements[buf_index_i];
            }
            
            // Now compute A(k, j) -= A(k, i) * A(j, i)
            elements[buf_index_j] -= a_ki * a_ji;
          }
        }
      }
      
    }


    /** @brief Incomplete Cholesky preconditioner class with static pattern (ICHOL0), can be supplied to solve()-routines
    */
    template <typename MatrixType>
    class ichol0_precond
    {
        typedef typename MatrixType::value_type      ScalarType;

      public:
        ichol0_precond(MatrixType const & mat, ichol0_tag const & tag) : tag_(tag), LU()
        {
            //initialize preconditioner:
            //std::cout << "Start CPU precond" << std::endl;
            init(mat);          
            //std::cout << "End CPU precond" << std::endl;
        }

        template <typename VectorType>
        void apply(VectorType & vec) const
        {
          unsigned int const * row_buffer = viennacl::linalg::single_threaded::detail::extract_raw_pointer<unsigned int>(LU.handle1());
          unsigned int const * col_buffer = viennacl::linalg::single_threaded::detail::extract_raw_pointer<unsigned int>(LU.handle2());
          ScalarType   const * elements   = viennacl::linalg::single_threaded::detail::extract_raw_pointer<ScalarType>(LU.handle());
          
          //
          // L is stored in a column-oriented fashion, i.e. transposed to the row-oriented layout. Thus, the factorization A = L L^T holds L in the upper triangular part of A.
          //
          viennacl::linalg::single_threaded::detail::csr_trans_inplace_solve<ScalarType>(row_buffer, col_buffer, elements, vec, LU.size2(), lower_tag());
          viennacl::linalg::single_threaded::detail::csr_inplace_solve<ScalarType>(row_buffer, col_buffer, elements, vec, LU.size2(), upper_tag());
        }

      private:
        void init(MatrixType const & mat)
        {
          LU.handle1().switch_active_handle_id(viennacl::backend::MAIN_MEMORY);
          LU.handle2().switch_active_handle_id(viennacl::backend::MAIN_MEMORY);
          LU.handle().switch_active_handle_id(viennacl::backend::MAIN_MEMORY);
          
          viennacl::copy(mat, LU);
          viennacl::linalg::precondition(LU, tag_);
        }

        ichol0_tag const & tag_;
        
        viennacl::compressed_matrix<ScalarType> LU;
    };


    /** @brief ILU0 preconditioner class, can be supplied to solve()-routines.
      *
      *  Specialization for compressed_matrix
      */
    template <typename ScalarType, unsigned int MAT_ALIGNMENT>
    class ichol0_precond< compressed_matrix<ScalarType, MAT_ALIGNMENT> >
    {
        typedef compressed_matrix<ScalarType, MAT_ALIGNMENT>   MatrixType;

      public:
        ichol0_precond(MatrixType const & mat, ichol0_tag const & tag) : tag_(tag), LLT(mat.size1(), mat.size2())
        {
          //initialize preconditioner:
          //std::cout << "Start GPU precond" << std::endl;
          init(mat);          
          //std::cout << "End GPU precond" << std::endl;
        }

        void apply(vector<ScalarType> & vec) const
        {
          if (vec.handle().get_active_handle_id() != viennacl::backend::MAIN_MEMORY)
          {
            viennacl::backend::memory_types old_memory_location = vec.handle().get_active_handle_id();
            vec.handle().switch_active_handle_id(viennacl::backend::MAIN_MEMORY);
            viennacl::linalg::inplace_solve(trans(LLT), vec, unit_lower_tag());
            viennacl::linalg::inplace_solve(LLT, vec, upper_tag());
            vec.handle().switch_active_handle_id(old_memory_location);
          }
          else //apply ILU0 directly:
          {
            viennacl::linalg::inplace_solve(trans(LLT), vec, unit_lower_tag());
            viennacl::linalg::inplace_solve(LLT, vec, upper_tag());
          }
        }

      private:
        void init(MatrixType const & mat)
        {
          //std::cout << "GPU->CPU, nonzeros: " << gpu_matrix.nnz() << std::endl;
          viennacl::backend::integral_type_host_array<unsigned int> dummy(mat.handle1());
          
          LLT.handle1().switch_active_handle_id(viennacl::backend::MAIN_MEMORY);
          LLT.handle2().switch_active_handle_id(viennacl::backend::MAIN_MEMORY);
          LLT.handle().switch_active_handle_id(viennacl::backend::MAIN_MEMORY);
          
          if (dummy.element_size() != sizeof(unsigned int))  //Additional effort required: cl_uint on device is different from 'unsigned int' on host
          {
            // get data from input matrix
            viennacl::backend::integral_type_host_array<unsigned int> row_buffer(mat.handle1(), mat.size1() + 1);
            viennacl::backend::integral_type_host_array<unsigned int> col_buffer(mat.handle2(), mat.nnz());
            
            viennacl::backend::memory_read(mat.handle1(), 0, row_buffer.raw_size(), row_buffer.get());
            viennacl::backend::memory_read(mat.handle2(), 0, col_buffer.raw_size(), col_buffer.get());
            
            
            //conversion from cl_uint to host type 'unsigned int' required
            std::vector<unsigned int> row_buffer_host(row_buffer.size());
            for (std::size_t i=0; i<row_buffer_host.size(); ++i)
              row_buffer_host[i] = row_buffer[i];
            
            std::vector<unsigned int> col_buffer_host(col_buffer.size());
            for (std::size_t i=0; i<col_buffer_host.size(); ++i)
              col_buffer_host[i] = col_buffer[i];
            
            viennacl::backend::memory_create(LLT.handle1(), sizeof(unsigned int) * row_buffer_host.size(), &(row_buffer_host[0]));
            viennacl::backend::memory_create(LLT.handle2(), sizeof(unsigned int) * col_buffer_host.size(), &(col_buffer_host[0]));
          }
          else //direct copy to new data structure
          {
            viennacl::backend::memory_create(LLT.handle1(), sizeof(unsigned int) * (mat.size1() + 1));
            viennacl::backend::memory_create(LLT.handle2(), sizeof(unsigned int) * mat.nnz());
            
            viennacl::backend::memory_read(mat.handle1(), 0, LLT.handle1().raw_size(), LLT.handle1().ram_handle().get());
            viennacl::backend::memory_read(mat.handle2(), 0, LLT.handle2().raw_size(), LLT.handle2().ram_handle().get());
          }          
          
          viennacl::backend::memory_create(LLT.handle(), sizeof(ScalarType) * mat.nnz());
          viennacl::backend::memory_read(mat.handle(), 0,  sizeof(ScalarType) * mat.nnz(), LLT.handle().ram_handle().get());
          

          viennacl::linalg::precondition(LLT, tag_);
        }

        ichol0_tag const & tag_;
        viennacl::compressed_matrix<ScalarType> LLT;
    };

  }
}




#endif



