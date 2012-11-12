
#ifndef VIENNACL_LINALG_DETAIL_ILU0_HPP_
#define VIENNACL_LINALG_DETAIL_ILU0_HPP_

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

/** @file viennacl/linalg/detail/ilu/ilu0.hpp
  @brief Implementations of incomplete factorization preconditioners with static nonzero pattern. Contributed by Evan Bollig.

  ILU0 (Incomplete LU with zero fill-in) 
  - All preconditioner nonzeros exist at locations that were nonzero in the input matrix. 
  - The number of nonzeros in the output preconditioner are exactly the same number as the input matrix

 Evan Bollig 3/30/12
 
 Adapted from viennacl/linalg/detail/ilut.hpp

*/

#include <vector>
#include <cmath>
#include <iostream>
#include "viennacl/forwards.h"
#include "viennacl/tools/tools.hpp"
#include "viennacl/linalg/detail/ilu/common.hpp"
#include "viennacl/compressed_matrix.hpp"

#include "viennacl/linalg/single_threaded/common.hpp"

#include <map>

namespace viennacl
{
  namespace linalg
  {

    /** @brief A tag for incomplete LU factorization with threshold (ILUT)
    */
    class ilu0_tag {};

    
    /** @brief Implementation of a ILU-preconditioner with static pattern. Optimized version for CSR matrices.
      *
      * refer to the Algorithm in Saad's book (1996 edition)
      *
      *  @param input   The input matrix. Type requirements: const_iterator1 for iteration along rows, const_iterator2 for iteration along columns
      *  @param output  The output matrix. Type requirements: const_iterator1 for iteration along rows, const_iterator2 for iteration along columns and write access via operator()
      *  @param tag     An ilu0_tag in order to dispatch among several other preconditioners.
      */
    template<typename ScalarType>
    void precondition(viennacl::compressed_matrix<ScalarType> & A, ilu0_tag const & tag)
    {
      assert( (A.handle1().get_active_handle_id == viennacl::backend::MAIN_MEMORY) && bool("System matrix must reside in main memory for ILU0") );
      assert( (A.handle2().get_active_handle_id == viennacl::backend::MAIN_MEMORY) && bool("System matrix must reside in main memory for ILU0") );
      assert( (A.handle().get_active_handle_id == viennacl::backend::MAIN_MEMORY) && bool("System matrix must reside in main memory for ILU0") );
      
      ScalarType         * elements   = viennacl::linalg::single_threaded::detail::extract_raw_pointer<ScalarType>(A.handle());
      unsigned int const * row_buffer = viennacl::linalg::single_threaded::detail::extract_raw_pointer<unsigned int>(A.handle1());
      unsigned int const * col_buffer = viennacl::linalg::single_threaded::detail::extract_raw_pointer<unsigned int>(A.handle2());
      
      // Note: Line numbers in the following refer to the algorithm in Saad's book
      
      for (std::size_t i=1; i<A.size1(); ++i)  // Line 1
      {
        unsigned int row_i_begin = row_buffer[i];
        unsigned int row_i_end   = row_buffer[i+1];
        for (unsigned int buf_index_k = row_i_begin; buf_index_k < row_i_end; ++buf_index_k) //Note: We do not assume that the column indices within a row are sorted
        {
          unsigned int k = col_buffer[buf_index_k];
          if (k >= i)
            continue; //Note: We do not assume that the column indices within a row are sorted
            
          unsigned int row_k_begin = row_buffer[k];
          unsigned int row_k_end   = row_buffer[k+1];
            
          // get a_kk:
          ScalarType a_kk = 0;
          for (unsigned int buf_index_akk = row_k_begin; buf_index_akk < row_k_end; ++buf_index_akk)
          {
            if (col_buffer[buf_index_akk] == k)
            {
              a_kk = elements[buf_index_akk];
              break;
            }
          }
          
          ScalarType & a_ik = elements[buf_index_k];
          a_ik /= a_kk;                                 //Line 3
          
          for (unsigned int buf_index_j = row_i_begin; buf_index_j < row_i_end; ++buf_index_j) //Note: We do not assume that the column indices within a row are sorted
          {
            unsigned int j = col_buffer[buf_index_j];
            if (j <= k)
              continue;
            
            // determine a_kj:
            ScalarType a_kj = 0;
            for (unsigned int buf_index_akj = row_k_begin; buf_index_akj < row_k_end; ++buf_index_akj)
            {
              if (col_buffer[buf_index_akj] == j)
              {
                a_kk = elements[buf_index_akj];
                break;
              }
            }
            
            //a_ij -= a_ik * a_kj
            elements[buf_index_j] -= a_ik * a_kj;  //Line 5
          }
        }
      }
      
    }


    /** @brief ILU0 preconditioner class, can be supplied to solve()-routines
    */
    template <typename MatrixType>
    class ilu0_precond
    {
        typedef typename MatrixType::value_type      ScalarType;

      public:
        ilu0_precond(MatrixType const & mat, ilu0_tag const & tag) : tag_(tag), LU()
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
          
          viennacl::linalg::single_threaded::detail::csr_inplace_solve<ScalarType>(row_buffer, col_buffer, elements, vec, LU.size2(), unit_lower_tag());
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

        ilu0_tag const & tag_;
        
        viennacl::compressed_matrix<ScalarType> LU;
    };


    /** @brief ILU0 preconditioner class, can be supplied to solve()-routines.
      *
      *  Specialization for compressed_matrix
      */
    template <typename ScalarType, unsigned int MAT_ALIGNMENT>
    class ilu0_precond< compressed_matrix<ScalarType, MAT_ALIGNMENT> >
    {
        typedef compressed_matrix<ScalarType, MAT_ALIGNMENT>   MatrixType;

      public:
        ilu0_precond(MatrixType const & mat, ilu0_tag const & tag) : tag_(tag), LU(mat.size1(), mat.size2())
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
            viennacl::linalg::inplace_solve(LU, vec, unit_lower_tag());
            viennacl::linalg::inplace_solve(LU, vec, upper_tag());
            vec.handle().switch_active_handle_id(old_memory_location);
          }
          else //apply ILU0 directly:
          {
            viennacl::linalg::inplace_solve(LU, vec, unit_lower_tag());
            viennacl::linalg::inplace_solve(LU, vec, upper_tag());
          }
        }

      private:
        void init(MatrixType const & mat)
        {
          //std::cout << "GPU->CPU, nonzeros: " << gpu_matrix.nnz() << std::endl;
          viennacl::backend::integral_type_host_array<unsigned int> dummy(mat.handle1());
          
          LU.handle1().switch_active_handle_id(viennacl::backend::MAIN_MEMORY);
          LU.handle2().switch_active_handle_id(viennacl::backend::MAIN_MEMORY);
          LU.handle().switch_active_handle_id(viennacl::backend::MAIN_MEMORY);
          
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
            
            viennacl::backend::memory_create(LU.handle1(), sizeof(unsigned int) * row_buffer_host.size(), &(row_buffer_host[0]));
            viennacl::backend::memory_create(LU.handle2(), sizeof(unsigned int) * col_buffer_host.size(), &(col_buffer_host[0]));
          }
          else //direct copy to new data structure
          {
            viennacl::backend::memory_create(LU.handle1(), sizeof(unsigned int) * (mat.size1() + 1));
            viennacl::backend::memory_create(LU.handle2(), sizeof(unsigned int) * mat.nnz());
            
            viennacl::backend::memory_read(mat.handle1(), 0, LU.handle1().raw_size(), LU.handle1().ram_handle().get());
            viennacl::backend::memory_read(mat.handle2(), 0, LU.handle2().raw_size(), LU.handle2().ram_handle().get());
          }          
          
          viennacl::backend::memory_create(LU.handle(), sizeof(ScalarType) * mat.nnz());
          viennacl::backend::memory_read(mat.handle(), 0,  sizeof(ScalarType) * mat.nnz(), LU.handle().ram_handle().get());
          

          viennacl::linalg::precondition(LU, tag_);
        }

        ilu0_tag const & tag_;
        viennacl::compressed_matrix<ScalarType> LU;
    };

  }
}




#endif



