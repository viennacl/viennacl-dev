#ifndef VIENNACL_LINALG_DETAIL_ILUT_HPP_
#define VIENNACL_LINALG_DETAIL_ILUT_HPP_

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

/** @file viennacl/linalg/detail/ilu/ilut.hpp
    @brief Implementations of an incomplete factorization preconditioner with threshold (ILUT)
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
    class ilut_tag
    {
      public:
        /** @brief The constructor.
        *
        * @param entries_per_row  Number of nonzero entries per row in L and U. Note that L and U are stored in a single matrix, thus there are 2*entries_per_row in total.
        * @param drop_tolerance   The drop tolerance for ILUT
        */
        ilut_tag(unsigned int entries_per_row = 20,
                 double drop_tolerance = 1e-4) : entries_per_row_(entries_per_row), drop_tolerance_(drop_tolerance) {}; 

        void set_drop_tolerance(double tol)
        {
          if (tol > 0)
            drop_tolerance_ = tol;
        }
        double get_drop_tolerance() const { return drop_tolerance_; }
        
        void set_entries_per_row(unsigned int e)
        {
          if (e > 0)
            entries_per_row_ = e;
        }

        unsigned int get_entries_per_row() const { return entries_per_row_; }

      private:
        unsigned int entries_per_row_;
        double drop_tolerance_;
    };
    
        
    /** @brief Implementation of a ILU-preconditioner with threshold
    *
    * refer to Algorithm 10.6 by Saad's book (1996 edition)
    *
    *  @param input   The input matrix. Type requirements: const_iterator1 for iteration along rows, const_iterator2 for iteration along columns
    *  @param output  The output matrix. Type requirements: const_iterator1 for iteration along rows, const_iterator2 for iteration along columns and write access via operator()
    *  @param tag     An ilut_tag in order to dispatch among several other preconditioners.
    */
    template<typename MatrixType, typename LUType>
    void precondition(MatrixType const & input, LUType & output, ilut_tag const & tag)
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
      
      for (InputRowIterator row_iter = input.begin1(); row_iter != input.end1(); ++row_iter)
      {
    /*    if (i%10 == 0)
      std::cout << i << std::endl;*/
        
        //line 2:
        w.clear();
        for (InputColIterator col_iter = row_iter.begin(); col_iter != row_iter.end(); ++col_iter)
          w[static_cast<unsigned int>(col_iter.index2())] = *col_iter;

        //line 3:
        OutputRowIterator row_iter_out = output.begin1();
        for (SparseVectorIterator w_k = w.begin(); w_k != w.end(); ++w_k)
        {
          unsigned int k = w_k->first;
          if (k >= static_cast<unsigned int>(row_iter.index1()))
            break;
          
          
          //while (row_iter_out.index1() < index_k)
          //  ++row_iter_out;
          //if (row_iter_out.index1() < index_k)
          //  row_iter_out += index_k - row_iter_out.index1();
          detail::ilu_inc_row_iterator_to_row_index(row_iter_out, k);
          
          //line 4:
          double a_kk = output(k, k);
          double temp = w_k->second / a_kk;
          if (a_kk == 0.0)
          {
            std::cerr << "ViennaCL: FATAL ERROR in ILUT(): Diagonal entry is zero in row " << k 
                      << " while processing line " << row_iter.index1() << "!" << std::endl;
          }
          
          //line 5: (dropping rule to w_k)
          if ( std::fabs(temp) > tag.get_drop_tolerance())
          {
            //line 7:
            for (OutputColIterator u_k = row_iter_out.begin(); u_k != row_iter_out.end(); ++u_k)
            {
              if (u_k.index2() >= k)
                w[u_k.index2()] -= temp * *u_k;
            }
          }
        } //for k
        
        //Line 10: Apply a dropping rule to w
        //Sort entries which are kept
        temp_map.clear();
        for (SparseVectorIterator w_k = w.begin(); w_k != w.end(); )
        {
          if ( (std::fabs(w_k->second) < tag.get_drop_tolerance()) 
               && (w_k->first != static_cast<unsigned int>(row_iter.index1())) //do not drop diagonal element!
             )
          { 
            long index = w_k->first;
            ++w_k;
            w.erase(index);
          }
          else
          {
            double temp = std::fabs(w_k->second);
            while (temp_map.find(temp) != temp_map.end())
              temp *= 1.00000001; //make entry slightly larger to maintain uniqueness of the entry
            temp_map[temp] = w_k->first;
            ++w_k;
          }
        }

        //Lines 10-12: write the largest p values to L and U
        unsigned int written_L = 0;
        unsigned int written_U = 0;
        for (typename std::map<double, unsigned int>::reverse_iterator iter = temp_map.rbegin(); iter != temp_map.rend(); ++iter)
        {
          if (iter->second > static_cast<unsigned int>(row_iter.index1())) //entry for U
          {
            if (written_U < tag.get_entries_per_row())
            {
              output(static_cast<unsigned int>(row_iter.index1()), iter->second) = static_cast<typename LUType::value_type>(w[iter->second]);
              ++written_U;
            }
          }
          else if (iter->second == static_cast<unsigned int>(row_iter.index1()))
          {
            output(iter->second, iter->second) = static_cast<typename LUType::value_type>(w[static_cast<unsigned int>(row_iter.index1())]);
          }
          else //entry for L
          {
            if (written_L < tag.get_entries_per_row())
            {
              output(static_cast<unsigned int>(row_iter.index1()), iter->second) = static_cast<typename LUType::value_type>(w[iter->second]);
              ++written_L;
            }
          }
        }
      } //for i
    }

    
    
    
    /** @brief Implementation of a ILU-preconditioner with threshold. Optimized implementation for compressed_matrix.
    *
    * refer to Algorithm 10.6 by Saad's book (1996 edition)
    *
    *  @param input   The input matrix. Type requirements: const_iterator1 for iteration along rows, const_iterator2 for iteration along columns
    *  @param output  The output matrix. Type requirements: const_iterator1 for iteration along rows, const_iterator2 for iteration along columns and write access via operator()
    *  @param tag     An ilut_tag in order to dispatch among several other preconditioners.
    */
    template<typename ScalarType, typename SizeType>
    void precondition(viennacl::compressed_matrix<ScalarType> const & A,
                      std::vector< std::map<SizeType, ScalarType> > & output,
                      ilut_tag const & tag)
    {
      typedef std::map<SizeType, ScalarType>          SparseVector;
      typedef typename SparseVector::iterator         SparseVectorIterator;
      typedef typename std::map<SizeType, ScalarType>::const_iterator   OutputRowConstIterator;
      typedef std::map<ScalarType, std::pair<SizeType, ScalarType> >  TemporarySortMap;

      assert( (A.handle1().get_active_handle_id == viennacl::backend::MAIN_MEMORY) && bool("System matrix must reside in main memory for ILU0") );
      assert( (A.handle2().get_active_handle_id == viennacl::backend::MAIN_MEMORY) && bool("System matrix must reside in main memory for ILU0") );
      assert( (A.handle().get_active_handle_id == viennacl::backend::MAIN_MEMORY) && bool("System matrix must reside in main memory for ILU0") );
      assert(input.size1() == input.size2() && bool("Input matrix must be square") );
      assert(input.size1() == output.size() && bool("Output matrix size mismatch") );
      
      ScalarType   const * elements   = viennacl::linalg::single_threaded::detail::extract_raw_pointer<ScalarType>(A.handle());
      unsigned int const * row_buffer = viennacl::linalg::single_threaded::detail::extract_raw_pointer<unsigned int>(A.handle1());
      unsigned int const * col_buffer = viennacl::linalg::single_threaded::detail::extract_raw_pointer<unsigned int>(A.handle2());
      
      SparseVector w;
      TemporarySortMap temp_map;
      
      for (SizeType i=0; i<A.size1(); ++i)  // Line 1
      {
    /*    if (i%10 == 0)
      std::cout << i << std::endl;*/
    
        //line 2: set up w
        SizeType row_i_begin = static_cast<SizeType>(row_buffer[i]);
        SizeType row_i_end   = static_cast<SizeType>(row_buffer[i+1]);
        ScalarType row_norm = 0;
        for (SizeType buf_index_i = row_i_begin; buf_index_i < row_i_end; ++buf_index_i) //Note: We do not assume that the column indices within a row are sorted
        {
          ScalarType entry = elements[buf_index_i];
          w[col_buffer[buf_index_i]] = entry;
          row_norm += entry * entry;
        }
        row_norm = std::sqrt(row_norm);
        ScalarType tau_i = tag.get_drop_tolerance() * row_norm;

        //line 3:
        for (SparseVectorIterator w_k = w.begin(); w_k != w.end(); ++w_k)
        {
          SizeType k = w_k->first;
          if (k >= i)
            break;
          
          //line 4:
          ScalarType a_kk = output[k][k];
          if (a_kk == 0)
          {
            std::cerr << "ViennaCL: FATAL ERROR in ILUT(): Diagonal entry is zero in row " << k 
                      << " while processing line " << i << "!" << std::endl;
            throw "ILUT zero diagonal!";
          }
          
          ScalarType w_k_entry = w_k->second / a_kk;
          w_k->second = w_k_entry;
          
          //line 5: (dropping rule to w_k)
          if ( std::fabs(w_k_entry) > tau_i)
          {
            //line 7:
            for (OutputRowConstIterator u_k = output[k].begin(); u_k != output[k].end(); ++u_k)
            {
              if (u_k->first > k)
                w[u_k->first] -= w_k_entry * u_k->second;
            }
          }
          //else
          //  w.erase(k);
          
        } //for w_k
        
        //Line 10: Apply a dropping rule to w
        //Sort entries which are kept
        temp_map.clear();
        for (SparseVectorIterator w_k = w.begin(); w_k != w.end(); ++w_k)
        {
          SizeType k = w_k->first;
          ScalarType w_k_entry = w_k->second;
          
          if ( (std::fabs(w_k_entry) > tau_i) || (k == i) )//do not drop diagonal element!
          { 
            ScalarType temp = std::fabs(w_k_entry);
            
            if (temp == 0) // this can only happen for diagonal entry
              throw "Triangular factor in ILUT singular!"; 
            
            while (temp_map.find(temp) != temp_map.end())
              temp *= static_cast<ScalarType>(1.000001); //make entry slightly larger to maintain uniqueness of the entry
            temp_map[temp] = std::make_pair(k, w_k_entry);
          }
        }

        //Lines 10-12: write the largest p values to L and U
        SizeType written_L = 0;
        SizeType written_U = 0;
        for (typename TemporarySortMap::reverse_iterator iter = temp_map.rbegin(); iter != temp_map.rend(); ++iter)
        {
          SizeType j = (iter->second).first;
          ScalarType w_j_entry = (iter->second).second;
          
          if (j < i) // Line 11: entry for L
          {
            if (written_L < tag.get_entries_per_row())
            {
              output[i][j] = w_j_entry;
              ++written_L;
            }
          }
          else if (j == i)  // Diagonal entry is always kept
          {
            output[i][j] = w_j_entry;
          }
          else //Line 12: entry for U
          {
            if (written_U < tag.get_entries_per_row())
            {
              output[i][j] = w_j_entry;
              ++written_U;
            }
          }
        }
        
        w.clear(); //Line 13
        
      } //for i
    }
    
    
    /** @brief ILUT preconditioner class, can be supplied to solve()-routines
    */
    template <typename MatrixType>
    class ilut_precond
    {
      typedef typename MatrixType::value_type      ScalarType;
      
      public:
        ilut_precond(MatrixType const & mat, ilut_tag const & tag) : tag_(tag), LU(mat.size1(), mat.size2())
        {
          //initialize preconditioner:
          //std::cout << "Start CPU precond" << std::endl;
          init(mat);          
          //std::cout << "End CPU precond" << std::endl;
        }
        
        template <typename VectorType>
        void apply(VectorType & vec) const
        {
          //viennacl::tools::const_sparse_matrix_adapter<ScalarType> LU_const_adapter(LU, LU.size(), LU.size());
          //viennacl::linalg::detail::ilu_lu_substitute(LU_const_adapter, vec);
          unsigned int const * row_buffer = viennacl::linalg::single_threaded::detail::extract_raw_pointer<unsigned int>(LU.handle1());
          unsigned int const * col_buffer = viennacl::linalg::single_threaded::detail::extract_raw_pointer<unsigned int>(LU.handle2());
          ScalarType   const * elements   = viennacl::linalg::single_threaded::detail::extract_raw_pointer<ScalarType>(LU.handle());
          
          viennacl::linalg::single_threaded::detail::csr_inplace_solve<ScalarType>(row_buffer, col_buffer, elements, vec, LU.size2(), unit_lower_tag());
          viennacl::linalg::single_threaded::detail::csr_inplace_solve<ScalarType>(row_buffer, col_buffer, elements, vec, LU.size2(), upper_tag());
        }
        
      private:
        void init(MatrixType const & mat)
        {
          std::vector< std::map<unsigned int, ScalarType> > LU_temp(mat.size1());
          viennacl::tools::sparse_matrix_adapter<ScalarType>       LU_temp_adapter(LU_temp, LU_temp.size(), LU_temp.size());
          
          viennacl::linalg::precondition(mat, LU_temp_adapter, tag_);
          
          LU.handle1().switch_active_handle_id(viennacl::backend::MAIN_MEMORY);
          LU.handle2().switch_active_handle_id(viennacl::backend::MAIN_MEMORY);
          LU.handle().switch_active_handle_id(viennacl::backend::MAIN_MEMORY);
          
          viennacl::copy(LU_temp, LU);
          std::cout << "ILUT nnz: " << LU.nnz() << std::endl;
        }
        
        ilut_tag const & tag_;
        viennacl::compressed_matrix<ScalarType> LU;
    };

    
    /** @brief ILUT preconditioner class, can be supplied to solve()-routines.
    *
    *  Specialization for compressed_matrix
    */
    template <typename ScalarType, unsigned int MAT_ALIGNMENT>
    class ilut_precond< compressed_matrix<ScalarType, MAT_ALIGNMENT> >
    {
      typedef compressed_matrix<ScalarType, MAT_ALIGNMENT>   MatrixType;
      
      public:
        ilut_precond(MatrixType const & mat, ilut_tag const & tag) : tag_(tag), LU(mat.size1(), mat.size2())
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
          else //apply ILUT directly:
          {
            viennacl::linalg::inplace_solve(LU, vec, unit_lower_tag());
            viennacl::linalg::inplace_solve(LU, vec, upper_tag());
          }
        }
        
      private:
        void init(MatrixType const & mat)
        {
          if (mat.handle().get_active_handle_id() == viennacl::backend::MAIN_MEMORY)
          {
            std::vector< std::map<unsigned int, ScalarType> > LU_temp(mat.size1());

            viennacl::linalg::precondition(mat, LU_temp, tag_);
            
            LU.handle1().switch_active_handle_id(viennacl::backend::MAIN_MEMORY);
            LU.handle2().switch_active_handle_id(viennacl::backend::MAIN_MEMORY);
            LU.handle().switch_active_handle_id(viennacl::backend::MAIN_MEMORY);
            
            viennacl::copy(LU_temp, LU);
            
            std::cout << "ILUT nnz: " << LU.nnz() << std::endl;
          }
          else
          {
            throw "not implemented!";
          }
        }
        
        ilut_tag const & tag_;
        viennacl::compressed_matrix<ScalarType> LU;
    };

  }
}




#endif



