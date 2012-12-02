#ifndef VIENNACL_ELL_MATRIX_HPP_
#define VIENNACL_ELL_MATRIX_HPP_

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

/** @file ell_matrix.hpp
    @brief Implementation of the ell_matrix class
    
    Contributed by Volodymyr Kysenko.
*/


#include "viennacl/forwards.h"
#include "viennacl/vector.hpp"

#include "viennacl/tools/tools.hpp"

#include "viennacl/linalg/sparse_matrix_operations.hpp"

namespace viennacl
{
    template<typename SCALARTYPE, unsigned int ALIGNMENT /* see forwards.h for default argument */>
    class ell_matrix
    {
      public:
        typedef viennacl::backend::mem_handle                                                              handle_type;
        typedef scalar<typename viennacl::tools::CHECK_SCALAR_TEMPLATE_ARGUMENT<SCALARTYPE>::ResultType>   value_type;
        
        ell_matrix() : rows_(0), cols_(0), maxnnz_(0) {}
        
        //ell_matrix(std::size_t row_num, std::size_t col_num) 
        //{
        //  viennacl::linalg::kernels::ell_matrix<SCALARTYPE, ALIGNMENT>::init();
        //}
    
      public:
        std::size_t internal_size1() const { return viennacl::tools::roundUpToNextMultiple<std::size_t>(rows_, ALIGNMENT); }
        std::size_t internal_size2() const { return viennacl::tools::roundUpToNextMultiple<std::size_t>(cols_, ALIGNMENT); }

        std::size_t size1() const { return rows_; }
        std::size_t size2() const { return cols_; }
        
        std::size_t internal_maxnnz() const {return viennacl::tools::roundUpToNextMultiple<std::size_t>(maxnnz_, ALIGNMENT); }
        std::size_t maxnnz() const { return maxnnz_; }

        std::size_t nnz() const { return rows_ * maxnnz_; }
        std::size_t internal_nnz() const { return internal_size1() * internal_maxnnz(); }

              handle_type & handle()       { return elements_; } 
        const handle_type & handle() const { return elements_; } 
        
              handle_type & handle2()       { return coords_; }
        const handle_type & handle2() const { return coords_; }

      #if defined(_MSC_VER) && _MSC_VER < 1500          //Visual Studio 2005 needs special treatment
        template <typename CPU_MATRIX>
        friend void copy(const CPU_MATRIX & cpu_matrix, ell_matrix & gpu_matrix );
      #else
        template <typename CPU_MATRIX, typename T, unsigned int ALIGN>
        friend void copy(const CPU_MATRIX & cpu_matrix, ell_matrix<T, ALIGN> & gpu_matrix );
      #endif        
        
      private:
        std::size_t rows_;
        std::size_t cols_;
        std::size_t maxnnz_;

        handle_type coords_;
        handle_type elements_;        
    };

    template <typename CPU_MATRIX, typename SCALARTYPE, unsigned int ALIGNMENT>
    void copy(const CPU_MATRIX& cpu_matrix, ell_matrix<SCALARTYPE, ALIGNMENT>& gpu_matrix )
    {
      if(cpu_matrix.size1() > 0 && cpu_matrix.size2() > 0)
      {
        //determine max capacity for row
        std::size_t max_entries_per_row = 0;
        for (typename CPU_MATRIX::const_iterator1 row_it = cpu_matrix.begin1(); row_it != cpu_matrix.end1(); ++row_it)
        {
          std::size_t num_entries = 0;
          for (typename CPU_MATRIX::const_iterator2 col_it = row_it.begin(); col_it != row_it.end(); ++col_it)
          {
              ++num_entries;
          }

          max_entries_per_row = std::max(max_entries_per_row, num_entries);
        }

        //setup GPU matrix
        gpu_matrix.maxnnz_ = max_entries_per_row;
        gpu_matrix.rows_ = cpu_matrix.size1();
        gpu_matrix.cols_ = cpu_matrix.size2();

        std::size_t nnz = gpu_matrix.internal_nnz();

        viennacl::backend::typesafe_host_array<unsigned int> coords(gpu_matrix.handle2(), nnz);
        std::vector<SCALARTYPE> elements(nnz, 0);

        // std::cout << "ELL_MATRIX copy " << gpu_matrix.maxnnz_ << " " << gpu_matrix.rows_ << " " << gpu_matrix.cols_ << " " 
        //             << gpu_matrix.internal_maxnnz() << "\n";

        for (typename CPU_MATRIX::const_iterator1 row_it = cpu_matrix.begin1(); row_it != cpu_matrix.end1(); ++row_it)
        {
          std::size_t data_index = 0;
          
          for (typename CPU_MATRIX::const_iterator2 col_it = row_it.begin(); col_it != row_it.end(); ++col_it)
          {
            coords.set(gpu_matrix.internal_size1() * data_index + col_it.index1(), col_it.index2());
            elements[gpu_matrix.internal_size1() * data_index + col_it.index1()] = *col_it;
            //std::cout << *col_it << "\n";
              data_index++;
          }
        }

        viennacl::backend::memory_create(gpu_matrix.handle2(), coords.raw_size(), coords.get());
        viennacl::backend::memory_create(gpu_matrix.handle(), sizeof(SCALARTYPE) * elements.size(), &(elements[0]));
      }
    }

    template <typename CPU_MATRIX, typename SCALARTYPE, unsigned int ALIGNMENT>
    void copy(const ell_matrix<SCALARTYPE, ALIGNMENT>& gpu_matrix, CPU_MATRIX& cpu_matrix)
    {
      if(gpu_matrix.size1() > 0 && gpu_matrix.size2() > 0)
      {
        cpu_matrix.resize(gpu_matrix.size1(), gpu_matrix.size2());

        std::vector<SCALARTYPE> elements(gpu_matrix.internal_nnz());
        viennacl::backend::typesafe_host_array<unsigned int> coords(gpu_matrix.handle2(), gpu_matrix.internal_nnz());

        viennacl::backend::memory_read(gpu_matrix.handle(), 0, sizeof(SCALARTYPE) * elements.size(), &(elements[0]));
        viennacl::backend::memory_read(gpu_matrix.handle2(), 0, coords.raw_size(), coords.get());

        for(std::size_t row = 0; row < gpu_matrix.size1(); row++)
        {
          for(std::size_t ind = 0; ind < gpu_matrix.internal_maxnnz(); ind++)
          {
            std::size_t offset = gpu_matrix.internal_size1() * ind + row;
            
            if(elements[offset] == static_cast<SCALARTYPE>(0.0))
                continue;

            if(coords[offset] >= gpu_matrix.size2())
            {
                std::cerr << "ViennaCL encountered invalid data " << offset << " " << ind << " " << row << " " << coords[offset] << " " << gpu_matrix.size2() << std::endl;
                return;
            }

            cpu_matrix(row, coords[offset]) = elements[offset];
          }
        }
      }
    }

    
}

#endif


