#ifndef VIENNACL_COMPRESSED_MATRIX_HPP_
#define VIENNACL_COMPRESSED_MATRIX_HPP_

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

/** @file compressed_matrix.hpp
    @brief Implementation of the compressed_matrix class
*/

#include <vector>
#include <list>
#include <map>
#include "viennacl/forwards.h"
#include "viennacl/vector.hpp"

#include "viennacl/linalg/sparse_matrix_operations.hpp"

#include "viennacl/tools/tools.hpp"
#include "viennacl/tools/entry_proxy.hpp"

namespace viennacl
{
    

    //provide copy-operation:
    /** @brief Copies a sparse matrix from the host to the OpenCL device (either GPU or multi-core CPU)
    *
    * There are some type requirements on the CPU_MATRIX type (fulfilled by e.g. boost::numeric::ublas):
    * - .size1() returns the number of rows
    * - .size2() returns the number of columns
    * - const_iterator1    is a type definition for an iterator along increasing row indices
    * - const_iterator2    is a type definition for an iterator along increasing columns indices
    * - The const_iterator1 type provides an iterator of type const_iterator2 via members .begin() and .end() that iterates along column indices in the current row.
    * - The types const_iterator1 and const_iterator2 provide members functions .index1() and .index2() that return the current row and column indices respectively.
    * - Dereferenciation of an object of type const_iterator2 returns the entry.
    *
    * @param cpu_matrix   A sparse matrix on the host.
    * @param gpu_matrix   A compressed_matrix from ViennaCL
    */
    template <typename CPU_MATRIX, typename SCALARTYPE, unsigned int ALIGNMENT>
    void copy(const CPU_MATRIX & cpu_matrix,
              compressed_matrix<SCALARTYPE, ALIGNMENT> & gpu_matrix )
    {
      //std::cout << "copy for (" << cpu_matrix.size1() << ", " << cpu_matrix.size2() << ", " << cpu_matrix.nnz() << ")" << std::endl;
      
      if ( cpu_matrix.size1() > 0 && cpu_matrix.size2() > 0 )
      {
        //determine nonzeros:
        long num_entries = 0;
        for (typename CPU_MATRIX::const_iterator1 row_it = cpu_matrix.begin1();
              row_it != cpu_matrix.end1();
              ++row_it)
        {
          std::size_t entries_per_row = 0;
          for (typename CPU_MATRIX::const_iterator2 col_it = row_it.begin();
                col_it != row_it.end();
                ++col_it)
          {
            ++entries_per_row;
          }
          num_entries += viennacl::tools::roundUpToNextMultiple<std::size_t>(entries_per_row, ALIGNMENT);
        }
        
        if (num_entries == 0) //we copy an empty matrix
        {
          num_entries = 1;
        }
        
        //set up matrix entries:
        
        //std::vector<cl_uint> row_buffer(cpu_matrix.size1() + 1);
        //std::vector<cl_uint> col_buffer(num_entries);
        viennacl::backend::integral_type_host_array<unsigned int> row_buffer(gpu_matrix.handle1(), cpu_matrix.size1() + 1);
        viennacl::backend::integral_type_host_array<unsigned int> col_buffer(gpu_matrix.handle2(), num_entries);
        std::vector<SCALARTYPE> elements(num_entries);
        
        std::size_t row_index  = 0;
        std::size_t data_index = 0;
        
        for (typename CPU_MATRIX::const_iterator1 row_it = cpu_matrix.begin1();
              row_it != cpu_matrix.end1();
              ++row_it)
        {
          row_buffer.set(row_index, data_index);
          ++row_index;
          
          for (typename CPU_MATRIX::const_iterator2 col_it = row_it.begin();
                col_it != row_it.end();
                ++col_it)
          {
            col_buffer.set(data_index, col_it.index2());
            elements[data_index] = *col_it;
            ++data_index;
          }
          data_index = viennacl::tools::roundUpToNextMultiple<std::size_t>(data_index, ALIGNMENT); //take care of alignment
        }
        row_buffer.set(row_index, data_index);
        
        gpu_matrix.set(row_buffer.get(),
                       col_buffer.get(),
                       &elements[0], 
                       cpu_matrix.size1(),
                       cpu_matrix.size2(),
                       num_entries);
      }
    }
    
    
    //adapted for std::vector< std::map < > > argument:
    /** @brief Copies a sparse matrix in the std::vector< std::map < > > format to an OpenCL device.
    *
    * @param cpu_matrix   A sparse square matrix on the host using STL types
    * @param gpu_matrix   A compressed_matrix from ViennaCL
    */
    template <typename SCALARTYPE, unsigned int ALIGNMENT>
    void copy(const std::vector< std::map<unsigned int, SCALARTYPE> > & cpu_matrix,
                     compressed_matrix<SCALARTYPE, ALIGNMENT> & gpu_matrix )
    {
      copy(tools::const_sparse_matrix_adapter<SCALARTYPE>(cpu_matrix, cpu_matrix.size(), cpu_matrix.size()), gpu_matrix);
    }
    
    #ifdef VIENNACL_WITH_EIGEN
    template <typename SCALARTYPE, int flags, unsigned int ALIGNMENT>
    void copy(const Eigen::SparseMatrix<SCALARTYPE, flags> & eigen_matrix,
              compressed_matrix<SCALARTYPE, ALIGNMENT> & gpu_matrix)
    {
      std::vector< std::map<unsigned int, SCALARTYPE> >  stl_matrix(eigen_matrix.rows());
      
      for (int k=0; k < eigen_matrix.outerSize(); ++k)
        for (typename Eigen::SparseMatrix<SCALARTYPE, flags>::InnerIterator it(eigen_matrix, k); it; ++it)
          stl_matrix[it.row()][it.col()] = it.value();
        
      copy(tools::const_sparse_matrix_adapter<SCALARTYPE>(stl_matrix, eigen_matrix.rows(), eigen_matrix.cols()), gpu_matrix);
    }
    #endif
    
    
    #ifdef VIENNACL_WITH_MTL4
    template <typename SCALARTYPE, unsigned int ALIGNMENT>
    void copy(const mtl::compressed2D<SCALARTYPE> & cpu_matrix,
              compressed_matrix<SCALARTYPE, ALIGNMENT> & gpu_matrix)
    {
      typedef mtl::compressed2D<SCALARTYPE>  MatrixType;
      
      std::vector< std::map<unsigned int, SCALARTYPE> >  stl_matrix(cpu_matrix.num_rows());
      
      using mtl::traits::range_generator;
      using mtl::traits::range::min;

      // Choose between row and column traversal
      typedef typename min<range_generator<mtl::tag::row, MatrixType>,
                           range_generator<mtl::tag::col, MatrixType> >::type   range_type;
      range_type                                                      my_range;

      // Type of outer cursor
      typedef typename range_type::type                               c_type;
      // Type of inner cursor
      typedef typename mtl::traits::range_generator<mtl::tag::nz, c_type>::type ic_type;

      // Define the property maps
      typename mtl::traits::row<MatrixType>::type                              row(cpu_matrix); 
      typename mtl::traits::col<MatrixType>::type                              col(cpu_matrix);
      typename mtl::traits::const_value<MatrixType>::type                      value(cpu_matrix); 

      // Now iterate over the matrix    
      for (c_type cursor(my_range.begin(cpu_matrix)), cend(my_range.end(cpu_matrix)); cursor != cend; ++cursor)
        for (ic_type icursor(mtl::begin<mtl::tag::nz>(cursor)), icend(mtl::end<mtl::tag::nz>(cursor)); icursor != icend; ++icursor)
          stl_matrix[row(*icursor)][col(*icursor)] = value(*icursor);
      
      copy(tools::const_sparse_matrix_adapter<SCALARTYPE>(stl_matrix, cpu_matrix.num_rows(), cpu_matrix.num_cols()), gpu_matrix);
    }
    #endif
    
    
    
    
    
    
    
    //
    // gpu to cpu:
    //
    /** @brief Copies a sparse matrix from the OpenCL device (either GPU or multi-core CPU) to the host.
    *
    * There are two type requirements on the CPU_MATRIX type (fulfilled by e.g. boost::numeric::ublas):
    * - resize(rows, cols)  A resize function to bring the matrix into the correct size
    * - operator(i,j)       Write new entries via the parenthesis operator
    *
    * @param gpu_matrix   A compressed_matrix from ViennaCL
    * @param cpu_matrix   A sparse matrix on the host.
    */
    template <typename CPU_MATRIX, typename SCALARTYPE, unsigned int ALIGNMENT>
    void copy(const compressed_matrix<SCALARTYPE, ALIGNMENT> & gpu_matrix,
                     CPU_MATRIX & cpu_matrix )
    {
      if ( gpu_matrix.size1() > 0 && gpu_matrix.size2() > 0 )
      {
        cpu_matrix.resize(gpu_matrix.size1(), gpu_matrix.size2(), false);
        
        //get raw data from memory:
        viennacl::backend::integral_type_host_array<unsigned int> row_buffer(gpu_matrix.handle1(), cpu_matrix.size1() + 1);
        viennacl::backend::integral_type_host_array<unsigned int> col_buffer(gpu_matrix.handle2(), gpu_matrix.nnz());
        std::vector<SCALARTYPE> elements(gpu_matrix.nnz());
        
        //std::cout << "GPU->CPU, nonzeros: " << gpu_matrix.nnz() << std::endl;
        
        viennacl::backend::memory_read(gpu_matrix.handle1(), 0, row_buffer.raw_size(), row_buffer.get());
        viennacl::backend::memory_read(gpu_matrix.handle2(), 0, col_buffer.raw_size(), col_buffer.get());
        viennacl::backend::memory_read(gpu_matrix.handle(),  0, sizeof(SCALARTYPE)* gpu_matrix.nnz(), &(elements[0]));
        
        //fill the cpu_matrix:
        std::size_t data_index = 0;
        for (std::size_t row = 1; row <= gpu_matrix.size1(); ++row)
        {
          while (data_index < row_buffer[row])
          {
            if (col_buffer[data_index] >= gpu_matrix.size2())
            {
              std::cerr << "ViennaCL encountered invalid data at colbuffer[" << data_index << "]: " << col_buffer[data_index] << std::endl;
              return;
            }
            
            if (elements[data_index] != static_cast<SCALARTYPE>(0.0))
              cpu_matrix(row-1, col_buffer[data_index]) = elements[data_index];
            ++data_index;
          }
        }
      }
    }
    
    
    /** @brief Copies a sparse matrix from an OpenCL device to the host. The host type is the std::vector< std::map < > > format .
    *
    * @param gpu_matrix   A compressed_matrix from ViennaCL
    * @param cpu_matrix   A sparse matrix on the host.
    */
    template <typename SCALARTYPE, unsigned int ALIGNMENT>
    void copy(const compressed_matrix<SCALARTYPE, ALIGNMENT> & gpu_matrix,
              std::vector< std::map<unsigned int, SCALARTYPE> > & cpu_matrix)
    {
      tools::sparse_matrix_adapter<SCALARTYPE> temp(cpu_matrix, cpu_matrix.size(), cpu_matrix.size());
      copy(gpu_matrix, temp);
    }
    
    
    #ifdef VIENNACL_WITH_EIGEN
    template <typename SCALARTYPE, int flags, unsigned int ALIGNMENT>
    void copy(compressed_matrix<SCALARTYPE, ALIGNMENT> & gpu_matrix,
              Eigen::SparseMatrix<SCALARTYPE, flags> & eigen_matrix)
    {
      if ( gpu_matrix.size1() > 0 && gpu_matrix.size2() > 0 )
      {
        assert(static_cast<unsigned int>(eigen_matrix.rows()) >= gpu_matrix.size1()
               && static_cast<unsigned int>(eigen_matrix.cols()) >= gpu_matrix.size2()
               && bool("Provided Eigen compressed matrix is too small!"));
        
        //get raw data from memory:
        std::vector<cl_uint> row_buffer(gpu_matrix.size1() + 1);
        std::vector<cl_uint> col_buffer(gpu_matrix.nnz());
        std::vector<SCALARTYPE> elements(gpu_matrix.nnz());
        
        viennacl::backend::memory_read(gpu_matrix.handle1(), 0, sizeof(cl_uint)   *(gpu_matrix.size1() + 1), &(row_buffer[0]));
        viennacl::backend::memory_read(gpu_matrix.handle2(), 0, sizeof(cl_uint)   * gpu_matrix.nnz(),        &(col_buffer[0]));
        viennacl::backend::memory_read(gpu_matrix.handle(),  0, sizeof(SCALARTYPE)* gpu_matrix.nnz(),        &(elements[0]));
        
        eigen_matrix.setZero();
        eigen_matrix.startFill();
        std::size_t data_index = 0;
        for (std::size_t row = 1; row <= gpu_matrix.size1(); ++row)
        {
          while (data_index < row_buffer[row])
          {
            assert(col_buffer[data_index] < gpu_matrix.size2() && bool("ViennaCL encountered invalid data at col_buffer"));
            if (elements[data_index] != static_cast<SCALARTYPE>(0.0))
              eigen_matrix.fill(row-1, col_buffer[data_index]) = elements[data_index];
            ++data_index;
          }
        }
        eigen_matrix.endFill();
      }
    }
    #endif
    
    
    
    #ifdef VIENNACL_WITH_MTL4
    template <typename SCALARTYPE, unsigned int ALIGNMENT>
    void copy(compressed_matrix<SCALARTYPE, ALIGNMENT> & gpu_matrix,
              mtl::compressed2D<SCALARTYPE> & mtl4_matrix)
    {
      if ( gpu_matrix.size1() > 0 && gpu_matrix.size2() > 0 )
      {
        assert(mtl4_matrix.num_rows() >= gpu_matrix.size1()
               && mtl4_matrix.num_cols() >= gpu_matrix.size2()
               && bool("Provided MTL4 compressed matrix is too small!"));
        
        //get raw data from memory:
        std::vector<cl_uint> row_buffer(gpu_matrix.size1() + 1);
        std::vector<cl_uint> col_buffer(gpu_matrix.nnz());
        std::vector<SCALARTYPE> elements(gpu_matrix.nnz());
        
        viennacl::backend::memory_read(gpu_matrix.handle1(), 0, sizeof(cl_uint)   *(gpu_matrix.size1() + 1), &(row_buffer[0]));
        viennacl::backend::memory_read(gpu_matrix.handle2(), 0, sizeof(cl_uint)   * gpu_matrix.nnz(),        &(col_buffer[0]));
        viennacl::backend::memory_read(gpu_matrix.handle(),  0, sizeof(SCALARTYPE)* gpu_matrix.nnz(),        &(elements[0]));
        
        //set_to_zero(mtl4_matrix);  
        //mtl4_matrix.change_dim(gpu_matrix.size1(), gpu_matrix.size2());
        
        mtl::matrix::inserter< mtl::compressed2D<SCALARTYPE> >  ins(mtl4_matrix);
        std::size_t data_index = 0;
        for (std::size_t row = 1; row <= gpu_matrix.size1(); ++row)
        {
          while (data_index < row_buffer[row])
          {
            assert(col_buffer[data_index] < gpu_matrix.size2() && bool("ViennaCL encountered invalid data at col_buffer"));
            if (elements[data_index] != static_cast<SCALARTYPE>(0.0))
              ins(row-1, col_buffer[data_index]) << typename mtl::Collection< mtl::compressed2D<SCALARTYPE> >::value_type(elements[data_index]);
            ++data_index;
          }
        }
      }
    }
    #endif
    
    
    
    
    
    //////////////////////// compressed_matrix //////////////////////////
    /** @brief A sparse square matrix in compressed sparse rows format.
    *
    * @tparam SCALARTYPE    The floating point type (either float or double, checked at compile time)
    * @tparam ALIGNMENT     The internal memory size for the entries in each row is given by (size()/ALIGNMENT + 1) * ALIGNMENT. ALIGNMENT must be a power of two. Best values or usually 4, 8 or 16, higher values are usually a waste of memory.
    */
    template<class SCALARTYPE, unsigned int ALIGNMENT /* see VCLForwards.h */>
    class compressed_matrix
    {
      public:
        typedef viennacl::backend::mem_handle                                                              handle_type;
        typedef scalar<typename viennacl::tools::CHECK_SCALAR_TEMPLATE_ARGUMENT<SCALARTYPE>::ResultType>   value_type;
        
        /** @brief Default construction of a compressed matrix. No memory is allocated */
        compressed_matrix() : rows_(0), cols_(0), nonzeros_(0) {}
        
        /** @brief Construction of a compressed matrix with the supplied number of rows and columns. If the number of nonzeros is positive, memory is allocated
        *
        * @param rows     Number of rows
        * @param cols     Number of columns
        * @param nonzeros Optional number of nonzeros for memory preallocation
        */
        explicit compressed_matrix(std::size_t rows, std::size_t cols, std::size_t nonzeros = 0) : rows_(rows), cols_(cols), nonzeros_(nonzeros)
        {
          if (rows > 0)
          {
            viennacl::backend::memory_create(row_buffer_, viennacl::backend::integral_type_host_array<unsigned int>().element_size() * (rows + 1));
          }
          if (nonzeros > 0)
          {
            viennacl::backend::memory_create(col_buffer_, viennacl::backend::integral_type_host_array<unsigned int>().element_size() * nonzeros);
            viennacl::backend::memory_create(elements_, sizeof(SCALARTYPE) * nonzeros);
          }
        }
        
#ifdef VIENNACL_WITH_OPENCL
        explicit compressed_matrix(cl_mem mem_row_buffer, cl_mem mem_col_buffer, cl_mem mem_elements, 
                                  std::size_t rows, std::size_t cols, std::size_t nonzeros) : 
          rows_(rows), cols_(cols), nonzeros_(nonzeros)
        {
            row_buffer_.switch_active_handle_id(viennacl::backend::OPENCL_MEMORY);
            row_buffer_.opencl_handle() = mem_row_buffer;
            row_buffer_.opencl_handle().inc();             //prevents that the user-provided memory is deleted once the matrix object is destroyed.
            
            col_buffer_.switch_active_handle_id(viennacl::backend::OPENCL_MEMORY);
            col_buffer_.opencl_handle() = mem_col_buffer;
            col_buffer_.opencl_handle().inc();             //prevents that the user-provided memory is deleted once the matrix object is destroyed.
            
            elements_.switch_active_handle_id(viennacl::backend::OPENCL_MEMORY);
            elements_.opencl_handle() = mem_elements;
            elements_.opencl_handle().inc();               //prevents that the user-provided memory is deleted once the matrix object is destroyed.
        }
#endif        
        
        /** @brief Sets the row, column and value arrays of the compressed matrix
        *
        * @param row_jumper     Pointer to an array holding the indices of the first element of each row (starting with zero). E.g. row_jumper[10] returns the index of the first entry of the 11th row. The array length is 'cols + 1'
        * @param col_buffer     Pointer to an array holding the column index of each entry. The array length is 'nonzeros'
        * @param elements       Pointer to an array holding the entries of the sparse matrix. The array length is 'elements'
        * @param rows           Number of rows of the sparse matrix
        * @param cols           Number of columns of the sparse matrix
        * @param nonzeros       Number of nonzeros
        */
        void set(void * row_jumper, 
                 void * col_buffer,
                 SCALARTYPE * elements, 
                 std::size_t rows,
                 std::size_t cols,
                 std::size_t nonzeros)
        {
          assert( (rows > 0)     && bool("Error in compressed_matrix::set(): Number of rows must be larger than zero!"));
          assert( (cols > 0)     && bool("Error in compressed_matrix::set(): Number of columns must be larger than zero!"));
          assert( (nonzeros > 0) && bool("Error in compressed_matrix::set(): Number of nonzeros must be larger than zero!"));
          //std::cout << "Setting memory: " << cols + 1 << ", " << nonzeros << std::endl;
          
          //row_buffer_.switch_active_handle_id(viennacl::backend::OPENCL_MEMORY);
          viennacl::backend::memory_create(row_buffer_, viennacl::backend::integral_type_host_array<unsigned int>(row_buffer_).element_size() * (rows + 1), row_jumper);

          //col_buffer_.switch_active_handle_id(viennacl::backend::OPENCL_MEMORY);
          viennacl::backend::memory_create(col_buffer_, viennacl::backend::integral_type_host_array<unsigned int>(col_buffer_).element_size() * nonzeros, col_buffer);

          //elements_.switch_active_handle_id(viennacl::backend::OPENCL_MEMORY);
          viennacl::backend::memory_create(elements_, sizeof(SCALARTYPE) * nonzeros, elements);
          
          nonzeros_ = nonzeros;
          rows_ = rows;
          cols_ = cols;
        }
          
        /** @brief Allocate memory for the supplied number of nonzeros in the matrix. Old values are preserved. */
        void reserve(std::size_t new_nonzeros)
        {
          if (new_nonzeros > nonzeros_)
          {
            handle_type col_buffer_old;
            handle_type elements_old;
            viennacl::backend::memory_shallow_copy(col_buffer_, col_buffer_old);
            viennacl::backend::memory_shallow_copy(elements_, elements_old);
            
            viennacl::backend::integral_type_host_array<unsigned int> size_deducer(col_buffer_);
            viennacl::backend::memory_create(col_buffer_, size_deducer.element_size() * new_nonzeros);
            viennacl::backend::memory_create(elements_,   sizeof(SCALARTYPE) * new_nonzeros);
            
            viennacl::backend::memory_copy(col_buffer_old, col_buffer_, 0, 0, size_deducer.element_size() * nonzeros_);
            viennacl::backend::memory_copy(elements_old,   elements_,   0, 0, sizeof(SCALARTYPE)* nonzeros_);

            nonzeros_ = new_nonzeros;
          }
        }

        /** @brief Resize the matrix.
        *
        * @param new_size1    New number of rows
        * @param new_size2    New number of columns
        * @param preserve     If true, the old values are preserved. At present, old values are always discarded.
        */
        void resize(std::size_t new_size1, std::size_t new_size2, bool preserve = true)
        {
          assert(new_size1 > 0 && new_size2 > 0 && bool("Cannot resize to zero size!"));
          
          if (new_size1 != rows_ || new_size2 != cols_)
          {
            std::vector<std::map<unsigned int, SCALARTYPE> > stl_sparse_matrix;
            if (rows_ > 0)
              stl_sparse_matrix.resize(rows_);
            
            if (preserve && rows_ > 0)
              viennacl::copy(*this, stl_sparse_matrix);
              
            stl_sparse_matrix.resize(new_size1);
            
            //discard entries with column index larger than new_size2
            if (new_size2 < cols_ && rows_ > 0)
            {
              for (size_t i=0; i<stl_sparse_matrix.size(); ++i)
              {
                std::list<unsigned int> to_delete;
                for (typename std::map<unsigned int, SCALARTYPE>::iterator it = stl_sparse_matrix[i].begin();
                    it != stl_sparse_matrix[i].end();
                    ++it)
                {
                  if (it->first >= new_size2)
                    to_delete.push_back(it->first);
                }
                
                for (std::list<unsigned int>::iterator it = to_delete.begin(); it != to_delete.end(); ++it)
                  stl_sparse_matrix[i].erase(*it);
              }
            }
            
            copy(stl_sparse_matrix, *this);
            
            rows_ = new_size1;
            cols_ = new_size2;
          }
        }
        
        /** @brief Returns a reference to the (i,j)-th entry of the sparse matrix. If (i,j) does not exist (zero), it is inserted (slow!) */
        entry_proxy<SCALARTYPE> operator()(std::size_t i, std::size_t j)
        {
          assert( (i < rows_) && (j < cols_) && bool("compressed_matrix access out of bounds!"));
          
          std::size_t index = element_index(i, j);
          
          // check for element in sparsity pattern
          if (index < nonzeros_)
            return entry_proxy<SCALARTYPE>(index, elements_);

          // Element not found. Copying required. Very slow, but direct entry manipulation is painful anyway...
          std::vector< std::map<unsigned int, SCALARTYPE> > cpu_backup(rows_);
          viennacl::copy(*this, cpu_backup);
          cpu_backup[i][j] = 0.0;
          viennacl::copy(cpu_backup, *this);
          
          index = element_index(i, j);
          
          assert(index < nonzeros_);
          
          return entry_proxy<SCALARTYPE>(index, elements_);
        }

        /** @brief  Returns the number of rows */
        const std::size_t & size1() const { return rows_; }
        /** @brief  Returns the number of columns */
        const std::size_t & size2() const { return cols_; }
        /** @brief  Returns the number of nonzero entries */
        const std::size_t & nnz() const { return nonzeros_; }
        
        /** @brief  Returns the OpenCL handle to the row index array */
        const handle_type & handle1() const { return row_buffer_; }
        /** @brief  Returns the OpenCL handle to the column index array */
        const handle_type & handle2() const { return col_buffer_; }
        /** @brief  Returns the OpenCL handle to the matrix entry array */
        const handle_type & handle() const { return elements_; }
        
      private:
        
        std::size_t element_index(std::size_t i, std::size_t j)
        {
          //read row indices
          viennacl::backend::integral_type_host_array<unsigned int> row_indices(row_buffer_, 2);
          viennacl::backend::memory_read(row_buffer_, row_indices.element_size()*i, row_indices.element_size()*2, row_indices.get());

          //get column indices for row i:
          viennacl::backend::integral_type_host_array<unsigned int> col_indices(col_buffer_, row_indices[1] - row_indices[0]);
          viennacl::backend::memory_read(col_buffer_, col_indices.element_size()*row_indices[0], row_indices.element_size()*col_indices.size(), col_indices.get());

          //get entries for row i:
          std::vector<SCALARTYPE> row_entries(row_indices[1] - row_indices[0]);
          viennacl::backend::memory_read(elements_, sizeof(SCALARTYPE)*row_indices[0], sizeof(SCALARTYPE)*row_entries.size(), &(row_entries[0]));

          for (std::size_t k=0; k<col_indices.size(); ++k)
          {
            if (col_indices[k] == j)
              return row_indices[0] + k;
          }
          
          // if not found, return index past the end of the matrix (cf. matrix.end() in the spirit of the STL)
          return nonzeros_;
        }
        
        // /** @brief Copy constructor is by now not available. */
        //compressed_matrix(compressed_matrix const &);
        
        /** @brief Assignment is by now not available. */
        compressed_matrix & operator=(compressed_matrix const &);
        
        
        std::size_t rows_;
        std::size_t cols_;
        std::size_t nonzeros_;
        handle_type row_buffer_;
        handle_type col_buffer_;
        handle_type elements_;
    };

    
    

}

#endif
