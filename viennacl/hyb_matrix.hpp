#ifndef VIENNACL_HYB_MATRIX_HPP_
#define VIENNACL_HYB_MATRIX_HPP_

/* =========================================================================
   Copyright (c) 2010-2012, Institute for Microelectronics,
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

/** @file hyb_matrix.hpp
    @brief Implementation of the hyb_matrix class
    
    Contributed by Volodymyr Kysenko.
*/

#include "viennacl/forwards.h"
#include "viennacl/vector.hpp"

#include "viennacl/tools/tools.hpp"

#include "viennacl/linalg/sparse_matrix_operations.hpp"

namespace viennacl
{
    template<typename SCALARTYPE, unsigned int ALIGNMENT  /* see forwards.h for default argument */>
    class hyb_matrix
    {
      public:
        typedef viennacl::backend::mem_handle                                                              handle_type;
        typedef scalar<typename viennacl::tools::CHECK_SCALAR_TEMPLATE_ARGUMENT<SCALARTYPE>::ResultType>   value_type;
        
        hyb_matrix() : csr_threshold_(SCALARTYPE(0.8)), rows_(0), cols_(0) {}
        
        //hyb_matrix(std::size_t row_num, std::size_t col_num) : csr_threshold_(0.8), rows_(row_num), cols_(col_num)
        //{
        //  viennacl::linalg::kernels::hyb_matrix<SCALARTYPE, ALIGNMENT>::init();
        //}

        SCALARTYPE  csr_threshold()  const { return csr_threshold_; }
        void csr_threshold(SCALARTYPE thr) { csr_threshold_ = thr; }

        std::size_t internal_size1() const { return viennacl::tools::roundUpToNextMultiple<std::size_t>(rows_, ALIGNMENT); }
        std::size_t internal_size2() const { return viennacl::tools::roundUpToNextMultiple<std::size_t>(cols_, ALIGNMENT); }

        std::size_t size1() const { return rows_; }
        std::size_t size2() const { return cols_; }

        std::size_t internal_ellnnz() const {return viennacl::tools::roundUpToNextMultiple<std::size_t>(ellnnz_, ALIGNMENT); }
        std::size_t ell_nnz() const { return ellnnz_; }
        std::size_t csr_nnz() const { return csrnnz_; }

        const handle_type & handle() const { return ell_elements_; } 
        const handle_type & handle2() const { return ell_coords_; }
        const handle_type & handle3() const { return csr_rows_; } 
        const handle_type & handle4() const { return csr_cols_; } 
        const handle_type & handle5() const { return csr_elements_; }  
    
      public:    
      #if defined(_MSC_VER) && _MSC_VER < 1500          //Visual Studio 2005 needs special treatment
        template <typename CPU_MATRIX>
        friend void copy(const CPU_MATRIX & cpu_matrix, hyb_matrix & gpu_matrix );
      #else
        template <typename CPU_MATRIX, typename T, unsigned int ALIGN>
        friend void copy(const CPU_MATRIX & cpu_matrix, hyb_matrix<T, ALIGN> & gpu_matrix );
      #endif
        
      private:
        SCALARTYPE  csr_threshold_;
        std::size_t rows_;
        std::size_t cols_;
        std::size_t ellnnz_;
        std::size_t csrnnz_;

        handle_type ell_coords_; // ell coords
        handle_type ell_elements_; // ell elements
        
        handle_type csr_rows_;
        handle_type csr_cols_;
        handle_type csr_elements_;
    };

    template <typename CPU_MATRIX, typename SCALARTYPE, unsigned int ALIGNMENT>
    void copy(const CPU_MATRIX& cpu_matrix, hyb_matrix<SCALARTYPE, ALIGNMENT>& gpu_matrix )
    {
      if(cpu_matrix.size1() > 0 && cpu_matrix.size2() > 0)
      {
        //determine max capacity for row
        std::size_t max_entries_per_row = 0;
        std::vector<std::size_t> hist_entries(cpu_matrix.size1(), 0);

        for (typename CPU_MATRIX::const_iterator1 row_it = cpu_matrix.begin1(); row_it != cpu_matrix.end1(); ++row_it)
        {
            std::size_t num_entries = 0;
            for (typename CPU_MATRIX::const_iterator2 col_it = row_it.begin(); col_it != row_it.end(); ++col_it)
            {
                ++num_entries;
            }

            hist_entries[num_entries] += 1;
            max_entries_per_row = std::max(max_entries_per_row, num_entries);
        }
        
        std::size_t sum = 0;
        for(std::size_t ind = 0; ind <= max_entries_per_row; ind++)
        {
            sum += hist_entries[ind];
            
            if(sum >= gpu_matrix.csr_threshold() * cpu_matrix.size1())
            {
                max_entries_per_row = ind;
                break;
            }
            }

        //setup GPU matrix
        gpu_matrix.ellnnz_ = max_entries_per_row;
        gpu_matrix.rows_ = cpu_matrix.size1();
        gpu_matrix.cols_ = cpu_matrix.size2();

        std::size_t nnz = gpu_matrix.internal_size1() * gpu_matrix.internal_ellnnz();

        viennacl::backend::typesafe_host_array<unsigned int>  ell_coords(gpu_matrix.ell_coords_, nnz);
        viennacl::backend::typesafe_host_array<unsigned int>  csr_rows(gpu_matrix.csr_rows_, cpu_matrix.size1() + 1);
        std::vector<unsigned int> csr_cols;

        std::vector<SCALARTYPE> ell_elements(nnz);
        std::vector<SCALARTYPE> csr_elements;

        std::size_t csr_index = 0;

        for (typename CPU_MATRIX::const_iterator1 row_it = cpu_matrix.begin1(); row_it != cpu_matrix.end1(); ++row_it)
        {
          std::size_t data_index = 0;
  
          csr_rows.set(row_it.index1(), csr_index);
          
          for (typename CPU_MATRIX::const_iterator2 col_it = row_it.begin(); col_it != row_it.end(); ++col_it)
          {
            if(data_index < max_entries_per_row)
            {
                ell_coords.set(gpu_matrix.internal_size1() * data_index + col_it.index1(), col_it.index2());
                ell_elements[gpu_matrix.internal_size1() * data_index + col_it.index1()] = *col_it;
            }
            else
            {
                csr_cols.push_back(col_it.index2());
                csr_elements.push_back(*col_it);

                csr_index++;
            }

            data_index++;
          }

        }

        if(csr_cols.empty())
        {
          csr_cols.push_back(0);
          csr_elements.push_back(0);
        }

        csr_rows.set(csr_rows.size() - 1, csr_index);

        gpu_matrix.csrnnz_ = csr_cols.size();

        viennacl::backend::typesafe_host_array<unsigned int> csr_cols_for_gpu(gpu_matrix.csr_cols_, csr_cols.size());
        for (std::size_t i=0; i<csr_cols.size(); ++i)
          csr_cols_for_gpu.set(i, csr_cols[i]);
        
        viennacl::backend::memory_create(gpu_matrix.ell_coords_,   ell_coords.raw_size(), ell_coords.get());
        viennacl::backend::memory_create(gpu_matrix.ell_elements_, sizeof(SCALARTYPE) * ell_elements.size(), &(ell_elements[0]));
        
        viennacl::backend::memory_create(gpu_matrix.csr_rows_,     csr_rows.raw_size(),         csr_rows.get());
        viennacl::backend::memory_create(gpu_matrix.csr_cols_,     csr_cols_for_gpu.raw_size(), csr_cols_for_gpu.get());
        viennacl::backend::memory_create(gpu_matrix.csr_elements_, sizeof(SCALARTYPE) * csr_elements.size(), &(csr_elements[0]));
      }
    }

    template <typename CPU_MATRIX, typename SCALARTYPE, unsigned int ALIGNMENT>
    void copy(const hyb_matrix<SCALARTYPE, ALIGNMENT>& gpu_matrix, CPU_MATRIX& cpu_matrix)
    {
      if(gpu_matrix.size1() > 0 && gpu_matrix.size2() > 0)
      {
        cpu_matrix.resize(gpu_matrix.size1(), gpu_matrix.size2());

        std::vector<SCALARTYPE> ell_elements(gpu_matrix.internal_size1() * gpu_matrix.internal_ellnnz());
        viennacl::backend::typesafe_host_array<unsigned int> ell_coords(gpu_matrix.handle2(), gpu_matrix.internal_size1() * gpu_matrix.internal_ellnnz());

        std::vector<SCALARTYPE> csr_elements(gpu_matrix.csr_nnz());
        viennacl::backend::typesafe_host_array<unsigned int> csr_rows(gpu_matrix.handle3(), gpu_matrix.size1() + 1);
        viennacl::backend::typesafe_host_array<unsigned int> csr_cols(gpu_matrix.handle4(), gpu_matrix.csr_nnz());

        viennacl::backend::memory_read(gpu_matrix.handle(), 0, sizeof(SCALARTYPE) * ell_elements.size(), &(ell_elements[0]));
        viennacl::backend::memory_read(gpu_matrix.handle2(), 0, ell_coords.raw_size(), ell_coords.get());
        viennacl::backend::memory_read(gpu_matrix.handle3(), 0, csr_rows.raw_size(),   csr_rows.get());
        viennacl::backend::memory_read(gpu_matrix.handle4(), 0, csr_cols.raw_size(),   csr_cols.get());
        viennacl::backend::memory_read(gpu_matrix.handle5(), 0, sizeof(SCALARTYPE) * csr_elements.size(), &(csr_elements[0]));

        
        for(std::size_t row = 0; row < gpu_matrix.size1(); row++)
        {
          for(std::size_t ind = 0; ind < gpu_matrix.internal_ellnnz(); ind++)
          {
            std::size_t offset = gpu_matrix.internal_size1() * ind + row;
            
            if(ell_elements[offset] == static_cast<SCALARTYPE>(0.0))
            {
              continue;
            }

            if(ell_coords[offset] >= gpu_matrix.size2())
            {
              std::cerr << "ViennaCL encountered invalid data " << offset << " " << ind << " " << row << " " << ell_coords[offset] << " " << gpu_matrix.size2() << std::endl;
              return;
            }

            cpu_matrix(row, ell_coords[offset]) = ell_elements[offset];
          }

          for(std::size_t ind = csr_rows[row]; ind < csr_rows[row+1]; ind++)
          {
            if(csr_elements[ind] == static_cast<SCALARTYPE>(0.0))
            {
              continue;
            }

            if(csr_cols[ind] >= gpu_matrix.size2())
            {
              std::cerr << "ViennaCL encountered invalid data " << std::endl;
              return;
            }

            cpu_matrix(row, csr_cols[ind]) = csr_elements[ind];
          }
        }
      }
    }


}

#endif
