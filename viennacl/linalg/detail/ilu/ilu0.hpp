
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

#include "viennacl/linalg/single_threaded/common.hpp"

#include <map>

namespace viennacl
{
  namespace linalg
  {

    /** @brief A tag for incomplete LU factorization with threshold (ILUT)
    */
    class ilu0_tag
    {
      public:
        /** @brief The constructor.
          *
          * @param row_start     The starting row for the block to which we apply ILU
          * @param row_end       The end column of the block to which we apply ILU
          */
        ilu0_tag(unsigned int row_start = 0, unsigned int row_end = static_cast<unsigned int>(-1))
            : row_start_(row_start),  
              row_end_(row_end) {}
              
      public: 
        unsigned int row_start_, row_end_;
    };

    /** @brief Implementation of a ILU-preconditioner with static pattern. Generic version for UBLAS-compatible matrix interface.
      *
      * refer to the Algorithm in Saad's book (1996 edition)
      *
      *  @param input   The input matrix. Type requirements: const_iterator1 for iteration along rows, const_iterator2 for iteration along columns
      *  @param output  The output matrix. Type requirements: const_iterator1 for iteration along rows, const_iterator2 for iteration along columns and write access via operator()
      *  @param tag     An ilu0_tag in order to dispatch among several other preconditioners.
      */
    template<typename MatrixType, typename LUType>
    void precondition(MatrixType const & input, LUType & output, ilu0_tag const & tag)
    {
      typedef std::map<unsigned int, double>          SparseVector;
      typedef typename SparseVector::iterator         SparseVectorIterator;
      typedef typename MatrixType::const_iterator1    InputRowIterator;  //iterate along increasing row index
      typedef typename MatrixType::const_iterator2    InputColIterator;  //iterate along increasing column index
      typedef typename LUType::iterator1              OutputRowIterator;  //iterate along increasing row index
      typedef typename LUType::iterator2              OutputColIterator;  //iterate along increasing column index

      output.clear();
      assert(input.size1() == output.size1());
      assert(input.size2() == output.size2());
      output.resize(static_cast<unsigned int>(input.size1()), static_cast<unsigned int>(input.size2()), false);
      SparseVector w;


      std::map<double, unsigned int> temp_map;

      // For i = 2, ... , N, DO
      for (InputRowIterator row_iter = input.begin1(); row_iter != input.end1(); ++row_iter)
      {
        w.clear();
        for (InputColIterator col_iter = row_iter.begin(); col_iter != row_iter.end(); ++col_iter)
        {
          // Only work on the block described by (row_start:row_end, row_start:row_end)
          if ((static_cast<unsigned int>(row_iter.index1()) >= tag.row_start_) && (static_cast<unsigned int>(row_iter.index1()) < tag.row_end_))
          {
              if ((static_cast<unsigned int>(col_iter.index2()) >= tag.row_start_) && (static_cast<unsigned int>(col_iter.index2()) < tag.row_end_))
              {
                  w[static_cast<unsigned int>(col_iter.index2())] = *col_iter;
              }
          } 
          else 
          {
              // Put identity on the excluded diagonal
              w[static_cast<unsigned int>(row_iter.index1())] = 1.; 
          }
        }

        //line 3:
        OutputRowIterator row_iter_out = output.begin1();
        for (SparseVectorIterator k = w.begin(); k != w.end(); ++k)
        {
          unsigned int index_k = k->first;
          // Enforce i = 2 and 
          if (index_k >= static_cast<unsigned int>(row_iter.index1()))
              break;

          detail::ilu_inc_row_iterator_to_row_index(row_iter_out, index_k);

          //line 3: temp = a_ik = a_ik / a_kk
          double temp = k->second / output(index_k, index_k);
          if (output(index_k, index_k) == 0.0)
          {
              std::cerr << "ViennaCL: FATAL ERROR in ILU0(): Diagonal entry is zero in row " << index_k << "!" << std::endl;

          }

          for (OutputColIterator j = row_iter_out.begin(); j != row_iter_out.end(); ++j)
          {
              // Only fill if it a nonzero element of the input matrix
              if (input(row_iter.index1(), j.index2())) {
                  // Follow standard ILU algorithm (i.e., for j = k+1, ... , N)
                  if (j.index2() > index_k) 
                  {
                      // set a_ij
                      w[j.index2()] -= temp * *j;
                  }
              }
          }
          // Set a_ik
          w[index_k] = temp;
          
        } //for k

        // Write rows back to LU factor output
        unsigned int k_count = 0; 
        for (SparseVectorIterator k = w.begin(); k != w.end(); ++k )
        {
          output(static_cast<unsigned int>(row_iter.index1()), k->first) = static_cast<typename LUType::value_type>(w[k->first]);
          k_count ++; 
        }
      } //for i
    }

    
    
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
        ilu0_precond(MatrixType const & mat, ilu0_tag const & tag) : tag_(tag), LU(mat.size1())
        {
            //initialize preconditioner:
            //std::cout << "Start CPU precond" << std::endl;
            init(mat);          
            //std::cout << "End CPU precond" << std::endl;
        }

        template <typename VectorType>
            void apply(VectorType & vec) const
            {
                viennacl::tools::const_sparse_matrix_adapter<ScalarType> LU_const_adapter(LU);
                viennacl::linalg::detail::ilu_lu_substitute(LU_const_adapter, vec);
            }

      private:
        void init(MatrixType const & mat)
        {
            viennacl::tools::sparse_matrix_adapter<ScalarType>       LU_adapter(LU);
            viennacl::linalg::precondition(mat, LU_adapter, tag_);
        }

        ilu0_tag const & tag_;
        
        public: std::vector< std::map<unsigned int, ScalarType> > LU;
    };


    /** @brief ILUT preconditioner class, can be supplied to solve()-routines.
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
          // get data from input matrix
          viennacl::backend::integral_type_host_array<unsigned int> row_buffer(mat.handle1(), mat.size1() + 1);
          viennacl::backend::integral_type_host_array<unsigned int> col_buffer(mat.handle2(), mat.nnz());
          std::vector<ScalarType> elements(mat.nnz());
          
          //std::cout << "GPU->CPU, nonzeros: " << gpu_matrix.nnz() << std::endl;
          
          viennacl::backend::memory_read(mat.handle1(), 0, row_buffer.raw_size(), row_buffer.get());
          viennacl::backend::memory_read(mat.handle2(), 0, col_buffer.raw_size(), col_buffer.get());
          viennacl::backend::memory_read(mat.handle(), 0,  sizeof(ScalarType) * elements.size(), &(elements[0]));
          
          LU.handle1().switch_active_handle_id(viennacl::backend::MAIN_MEMORY);
          LU.handle2().switch_active_handle_id(viennacl::backend::MAIN_MEMORY);
          LU.handle().switch_active_handle_id(viennacl::backend::MAIN_MEMORY);
          
          if (row_buffer.element_size() != sizeof(unsigned int))
          {
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
          else
          {
            // fill LU with nonzero pattern:
            viennacl::backend::memory_create(LU.handle1(), viennacl::backend::integral_type_host_array<unsigned int>(LU.handle1()).element_size() * (mat.size1() + 1), row_buffer.get());
            viennacl::backend::memory_create(LU.handle2(), viennacl::backend::integral_type_host_array<unsigned int>(LU.handle2()).element_size() * mat.nnz(),         col_buffer.get());
          }          
          
          viennacl::backend::memory_create(LU.handle(), sizeof(ScalarType) * mat.nnz(), &(elements[0])); //initialize with all zeros
          

          viennacl::linalg::precondition(LU, tag_);
        }

         ilu0_tag const & tag_;
         viennacl::compressed_matrix<ScalarType> LU;
    };

  }
}




#endif



