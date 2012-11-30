#ifndef VIENNACL_MATRIX_HPP_
#define VIENNACL_MATRIX_HPP_

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

/** @file viennacl/matrix.hpp
    @brief Implementation of the dense matrix class
*/

#include "viennacl/forwards.h"
#include "viennacl/scalar.hpp"
#include "viennacl/vector.hpp"
#include "viennacl/linalg/matrix_operations.hpp"
#include "viennacl/tools/tools.hpp"
#include "viennacl/tools/matrix_size_deducer.hpp"
#include "viennacl/meta/result_of.hpp"
#include "viennacl/meta/enable_if.hpp"

namespace viennacl
{
  //
  // Initializer types
  //
  /** @brief Represents a vector consisting of 1 at a given index and zeros otherwise. To be used as an initializer for viennacl::vector, vector_range, or vector_slize only. */
  template <typename SCALARTYPE>
  class identity_matrix
  {
    public:
      typedef vcl_size_t         size_type;
      typedef SCALARTYPE const & const_reference;
      
      identity_matrix(size_type s) : size_(s), diag_(1), off_diag_(0) {}
      
      size_type size1() const { return size_; }
      size_type size2() const { return size_; }
      const_reference operator()(size_type i, size_type j) const { return (i == j) ? diag_ : off_diag_; }
      
    private:
      size_type size_;
      SCALARTYPE diag_;
      SCALARTYPE off_diag_;
  };

  
  /** @brief Represents a vector consisting of zeros only. To be used as an initializer for viennacl::vector, vector_range, or vector_slize only. */
  template <typename SCALARTYPE>
  class zero_matrix
  {
    public:
      typedef vcl_size_t         size_type;
      typedef SCALARTYPE const & const_reference;
      
      zero_matrix(size_type s1, size_type s2) : size1_(s1), size2_(s2), val_(0) {}
      
      size_type size1() const { return size1_; }
      size_type size2() const { return size2_; }
      const_reference operator()(size_type /*i*/, size_type /*j*/) const { return val_; }
      
    private:
      size_type size1_;
      size_type size2_;
      SCALARTYPE val_;
  };
  
  
  /** @brief Represents a vector consisting of scalars 's' only, i.e. v[i] = s for all i. To be used as an initializer for viennacl::vector, vector_range, or vector_slize only. */
  template <typename SCALARTYPE>
  class scalar_matrix
  {
    public:
      typedef vcl_size_t         size_type;
      typedef SCALARTYPE const & const_reference;
      
      scalar_matrix(size_type s1, size_type s2, const_reference val) : size1_(s1), size2_(s2), value_(val) {}
      
      size_type size1() const { return size1_; }
      size_type size2() const { return size2_; }
      const_reference operator()(size_type /*i*/, size_type /*j*/) const { return value_; }
      
    private:
      size_type size1_;
      size_type size2_;
      SCALARTYPE value_;
  };
  
  
  template <typename LHS, typename RHS, typename OP>
  class matrix_expression
  {
    public:
      typedef typename LHS::size_type       size_type;
      
      ///** @brief Extracts the vector type from the two operands.
      //*/
      typedef typename viennacl::tools::MATRIX_EXTRACTOR<LHS, RHS>::ResultType    matrix_type;
    
      matrix_expression(LHS & lhs, RHS & rhs) : lhs_(lhs), rhs_(rhs) {}
      
      /** @brief Get left hand side operand
      */
      LHS & lhs() const { return lhs_; }
      /** @brief Get right hand side operand
      */
      RHS & rhs() const { return rhs_; }
      
      /** @brief Returns the size of the result vector */
      std::size_t size1() const { return viennacl::tools::MATRIX_SIZE_DEDUCER<LHS, RHS, OP>::size1(lhs_, rhs_); }
      std::size_t size2() const { return viennacl::tools::MATRIX_SIZE_DEDUCER<LHS, RHS, OP>::size2(lhs_, rhs_); }
      
    private:
      /** @brief The left hand side operand */
      typename result_of::matrix_expression_internal_storage<LHS>::type lhs_;
      /** @brief The right hand side operand */
      typename result_of::matrix_expression_internal_storage<RHS>::type rhs_;
  };
  
  
  /** @brief A tag indicating iteration along increasing row index of a matrix */
  struct row_iteration {};
  
  /** @brief A tag indicating iteration along increasing columns index of a matrix */
  struct col_iteration {};

  //STL-like iterator. TODO: STL-compliance...
  template <typename ROWCOL, typename MATRIXTYPE>
  class matrix_iterator
  {
      typedef matrix_iterator<ROWCOL, MATRIXTYPE>    self_type;
    public:
      typedef typename MATRIXTYPE::value_type       value_type;
      
      matrix_iterator(MATRIXTYPE & mat, 
                      std::size_t start_row,
                      std::size_t start_col) : mat_(mat), row_(start_row), col_(start_col) {};
      
      value_type operator*(void) { return mat_(row_, col_); }
      self_type & operator++(void) { viennacl::tools::MATRIX_ITERATOR_INCREMENTER<ROWCOL, MATRIXTYPE>::apply(mat_, row_, col_); return *this; }
      self_type & operator++(int) { self_type tmp = *this; ++(*this); return tmp; }
      
      bool operator==(self_type const & other) { return (row_ == other.row_) && (col_ == other.col_); }
      bool operator!=(self_type const & other) { return !(*this == other); }
      
      vcl_size_t index1() { return row_; }
      vcl_size_t index2() { return col_; }
      
      MATRIXTYPE & operator()(void) const { return mat_; }
    
    private:
      MATRIXTYPE & mat_;
      vcl_size_t row_;
      vcl_size_t col_;
  };

  /** @brief A dense matrix class
  *
  * @tparam SCALARTYPE   The underlying scalar type (either float or double)
  * @tparam F            Storage layout: Either row_major or column_major (at present only row_major is supported)
  * @tparam ALIGNMENT   The internal memory size is given by (size()/ALIGNMENT + 1) * ALIGNMENT. ALIGNMENT must be a power of two. Best values or usually 4, 8 or 16, higher values are usually a waste of memory.
  */
  template <class SCALARTYPE, typename F, unsigned int ALIGNMENT>
  class matrix
  {
    typedef matrix<SCALARTYPE, F, ALIGNMENT>          self_type;
  public:
    
    typedef matrix_iterator<row_iteration, matrix<SCALARTYPE, F, ALIGNMENT> >   iterator1;
    typedef matrix_iterator<col_iteration, matrix<SCALARTYPE, F, ALIGNMENT> >   iterator2;
    typedef scalar<typename viennacl::tools::CHECK_SCALAR_TEMPLATE_ARGUMENT<SCALARTYPE>::ResultType>   value_type;
    typedef vcl_size_t                                                          size_type;
    typedef backend::mem_handle                                                 handle_type;
    typedef typename F::orientation_category                                    orientation_category;
    
    /** @brief The default constructor. Does not allocate any memory. */
    explicit matrix() : rows_(0), columns_(0) {};
    
    /** @brief Creates the matrix with the given dimensions
    *
    * @param rows     Number of rows
    * @param columns  Number of columns
    */
    explicit matrix(size_type rows, size_type columns) :
      rows_(rows), columns_(columns)
    {
      if (rows > 0 && columns > 0)
        viennacl::backend::memory_create(elements_, sizeof(SCALARTYPE)*internal_size());
    }

#ifdef VIENNACL_WITH_OPENCL
    explicit matrix(cl_mem mem, size_type rows, size_type columns) :
      rows_(rows), columns_(columns)
    {
      elements_.switch_active_handle_id(viennacl::OPENCL_MEMORY);
      elements_.opencl_handle() = mem;
      elements_.opencl_handle().inc();  //prevents that the user-provided memory is deleted once the vector object is destroyed.
    }
#endif

    template <typename LHS, typename RHS, typename OP>
    matrix(matrix_expression< LHS, RHS, OP> const & proxy) : rows_(proxy.size1()), columns_(proxy.size2())
    {
      elements_.switch_active_handle_id(viennacl::traits::active_handle_id(proxy));
      viennacl::backend::memory_create(elements_, sizeof(SCALARTYPE)*internal_size());
      
      *this = proxy;
    }
    
    /** @brief Creates the matrix from the supplied identity matrix. */
    matrix(identity_matrix<SCALARTYPE> const & m) : rows_(m.size1()), columns_(m.size2())
    {
      if (internal_size() > 0)
      {
        viennacl::backend::memory_create(elements_, sizeof(SCALARTYPE)*internal_size());
        clear();
        viennacl::linalg::matrix_diagonal_assign(*this, m(0,0));
      }
    }
    
    /** @brief Creates the matrix from the supplied zero matrix. */
    matrix(zero_matrix<SCALARTYPE> const & m) : rows_(m.size1()), columns_(m.size2())
    {
      if (internal_size() > 0)
      {
        viennacl::backend::memory_create(elements_, sizeof(SCALARTYPE)*internal_size());
        clear();
      }
    }

    /** @brief Creates the matrix from the supplied scalar matrix. */
    matrix(scalar_matrix<SCALARTYPE> const & m) : rows_(m.size1()), columns_(m.size2())
    {
      if (internal_size() > 0)
      {
        viennacl::backend::memory_create(elements_, sizeof(SCALARTYPE)*internal_size());
        viennacl::linalg::matrix_assign(*this, m(0,0));
      }
    }

    
    // matrix_range (implemented in matrix_proyx.hpp)
    matrix(matrix_range<self_type> const & proxy);
    matrix(matrix_range<const self_type> const & proxy);
    
    // matrix_slice (implemented in matrix_proxy.hpp)
    matrix(matrix_slice<self_type> const & proxy);
    matrix(matrix_slice<const self_type> const & proxy);


    //copy constructor:
    matrix(const self_type & other) : rows_(other.size1()), columns_(other.size2())
    {
      elements_.switch_active_handle_id(viennacl::traits::handle(other).get_active_handle_id());
      viennacl::backend::memory_create(elements_, sizeof(SCALARTYPE)*internal_size());
      viennacl::backend::memory_copy(other.handle(), elements_, 0, 0, sizeof(SCALARTYPE)*internal_size());
    }

    self_type & operator=(const self_type & other)  //enables implicit conversions
    {
      if (internal_size() == 0)
        resize(other.size1(), other.size2(), false);
      viennacl::backend::memory_copy(other.handle(), elements_, 0, 0, sizeof(SCALARTYPE)*internal_size());
      return *this;
    }
    
    
    // A = trans(B). Currently achieved in CPU memory
    self_type & operator=(const matrix_expression< const self_type,
                                                    const self_type,
                                                    op_trans> & proxy)
    {
      assert( (elements_ != proxy.lhs().handle()) && bool("Self-assignment of matrix transpose not implemented"));
      assert( ( (proxy.lhs().size1() == size2()) || (size2() == 0) ) && bool("Matrix dimensions do not match!"));
      assert( ( (proxy.lhs().size2() == size1()) || (size1() == 0) ) && bool("Matrix dimensions do not match!"));

      if (internal_size() == 0)
        resize(proxy.lhs().size2(), proxy.lhs().size1(), false);
      
      std::vector<SCALARTYPE> temp(proxy.lhs().internal_size());

      viennacl::backend::memory_read(proxy.lhs().handle(), 0, sizeof(SCALARTYPE)*proxy.lhs().internal_size(), &(temp[0]));

      // now transpose it
      std::vector<SCALARTYPE> temp_trans(internal_size());

      for (vcl_size_t i=0; i<proxy.lhs().size1(); ++i)
        for (vcl_size_t j=0; j<proxy.lhs().size2(); ++j)
          temp_trans[F::mem_index(j,i, internal_size1(), internal_size2())] 
            = temp[F::mem_index(i,j, proxy.lhs().internal_size1(), proxy.lhs().internal_size2())];

      // write back
      viennacl::backend::memory_write(elements_, 0, sizeof(SCALARTYPE)*internal_size(), &(temp_trans[0]));
        
      return *this;
    }
    
    template <typename M1>
    self_type & operator=(const matrix_expression< const M1, const M1, op_trans> & proxy)
    {
      self_type temp(proxy.lhs());
      *this = trans(temp);
      return *this;
    }
    
    

    /** @brief Implementation of the operation m1 = m2 @ alpha, where @ denotes either multiplication or division, and alpha is either a CPU or a GPU scalar
    *
    * @param proxy  An expression template proxy class.
    */
    template <typename M1, typename S1, typename OP>
    typename viennacl::enable_if< viennacl::is_any_dense_nonstructured_matrix<M1>::value && viennacl::is_any_scalar<S1>::value,
                                  self_type & >::type
    operator = (const matrix_expression< const M1, const S1, OP> & proxy)
    {
      assert(  (proxy.lhs().size1() == size1())
            && (proxy.lhs().size2() == size2())
            && bool("Incompatible matrix sizes!"));
      
      if (internal_size() == 0)
      {
        rows_ = proxy.lhs().size1();
        columns_ = proxy.lhs().size2();
        viennacl::backend::memory_create(elements_, sizeof(SCALARTYPE)*internal_size());
      } 

      if (internal_size() > 0)
        viennacl::linalg::am(*this,
                              proxy.lhs(), proxy.rhs(), 1, (viennacl::is_division<OP>::value ? true : false), false);
      return *this;
    }

    //m1 = m2 +- m3; 
    /** @brief Implementation of the operation m1 = m2 +- m3
    *
    * @param proxy  An expression template proxy class.
    */
    template <typename M1, typename M2, typename OP>
    typename viennacl::enable_if<    viennacl::is_any_dense_nonstructured_matrix<M1>::value
                                  && viennacl::is_any_dense_nonstructured_matrix<M2>::value
                                  && (viennacl::is_addition<OP>::value || viennacl::is_subtraction<OP>::value),
                                  self_type &>::type
    operator = (const matrix_expression< const M1,
                                          const M2,
                                          OP> & proxy)
    {
      assert(  (proxy.lhs().size1() == size1())
            && (proxy.lhs().size2() == size2())
            && bool("Incompatible matrix sizes!"));
      
      if (internal_size() == 0)
      {
        rows_ = proxy.lhs().size1();
        columns_ = proxy.lhs().size2();
        viennacl::backend::memory_create(elements_, sizeof(SCALARTYPE)*internal_size());
      } 

      if (internal_size() > 0)
        viennacl::linalg::ambm(*this, 
                                proxy.lhs(), SCALARTYPE(1.0), 1, false, false,
                                proxy.rhs(), SCALARTYPE(1.0), 1, false, (viennacl::is_subtraction<OP>::value ? true : false));
      return *this;
    }
    
    /** @brief Implementation of the operation m1 = m2 +- m3 @ beta, where @ is either product or division, and alpha, beta are either CPU or GPU scalars
    *
    * @param proxy  An expression template proxy class.
    */
    template <typename M1,
              typename M2, typename S2, typename OP2,
              typename OP>
    typename viennacl::enable_if<    viennacl::is_any_dense_nonstructured_matrix<M1>::value
                                  && viennacl::is_any_dense_nonstructured_matrix<M2>::value && viennacl::is_any_scalar<S2>::value && (viennacl::is_product<OP2>::value || viennacl::is_division<OP2>::value)
                                  && (viennacl::is_addition<OP>::value || viennacl::is_subtraction<OP>::value),
                                  self_type &>::type
    operator = (const matrix_expression< const M1,
                                          const matrix_expression<const M2, const S2, OP2>,
                                          OP> & proxy)
    {
      assert(  (proxy.lhs().size1() == size1())
            && (proxy.lhs().size2() == size2())
            && bool("Incompatible matrix sizes!"));
      
      if (internal_size() == 0)
      {
        rows_ = proxy.lhs().size1();
        columns_ = proxy.lhs().size2();
        viennacl::backend::memory_create(elements_, sizeof(SCALARTYPE)*internal_size());
      } 

      if (internal_size() > 0)
      {
        bool flip_sign_2 = (viennacl::is_subtraction<OP>::value ? true : false);
        if (viennacl::is_flip_sign_scalar<S2>::value)
          flip_sign_2 = !flip_sign_2;
        viennacl::linalg::ambm(*this, 
                                proxy.lhs(),         SCALARTYPE(1.0), 1, false                                             , false,
                                proxy.rhs().lhs(), proxy.rhs().rhs(), 1, (viennacl::is_division<OP2>::value ? true : false), flip_sign_2);
      }
      return *this;
    }

    /** @brief Implementation of the operation m1 = m2 @ alpha +- m3, where @ is either product or division, and alpha, beta are either CPU or GPU scalars
    *
    * @param proxy  An expression template proxy class.
    */
    template <typename M1, typename S1, typename OP1,
              typename M2,
              typename OP>
    typename viennacl::enable_if<    viennacl::is_any_dense_nonstructured_matrix<M1>::value && viennacl::is_any_scalar<S1>::value && (viennacl::is_product<OP1>::value || viennacl::is_division<OP1>::value)
                                  && viennacl::is_any_dense_nonstructured_matrix<M2>::value
                                  && (viennacl::is_addition<OP>::value || viennacl::is_subtraction<OP>::value),
                                  self_type &>::type
    operator = (const matrix_expression< const matrix_expression<const M1, const S1, OP1>,
                                          const M2,
                                          OP> & proxy)
    {
      assert(  (proxy.lhs().size1() == size1())
            && (proxy.lhs().size2() == size2())
            && bool("Incompatible matrix sizes!"));
      
      if (internal_size() == 0)
      {
        rows_ = proxy.lhs().size1();
        columns_ = proxy.lhs().size2();
        viennacl::backend::memory_create(elements_, sizeof(SCALARTYPE)*internal_size());
      } 

      if (internal_size() > 0)
        viennacl::linalg::ambm(*this, 
                              proxy.lhs().lhs(), proxy.lhs().rhs(), 1, (viennacl::is_division<OP1>::value ? true : false), (viennacl::is_flip_sign_scalar<S1>::value ? true : false),
                              proxy.rhs(),         SCALARTYPE(1.0), 1, false                                             , (viennacl::is_subtraction<OP>::value ? true : false));
      return *this;
    }
    
    /** @brief Implementation of the operation m1 = m2 @ alpha +- m3 @ beta, where @ is either product or division, and alpha, beta are either CPU or GPU scalars
    *
    * @param proxy  An expression template proxy class.
    */
    template <typename M1, typename S1, typename OP1,
              typename M2, typename S2, typename OP2,
              typename OP>
    typename viennacl::enable_if<    viennacl::is_any_dense_nonstructured_matrix<M1>::value && viennacl::is_any_scalar<S1>::value && (viennacl::is_product<OP1>::value || viennacl::is_division<OP1>::value)
                                  && viennacl::is_any_dense_nonstructured_matrix<M2>::value && viennacl::is_any_scalar<S2>::value && (viennacl::is_product<OP2>::value || viennacl::is_division<OP2>::value)
                                  && (viennacl::is_addition<OP>::value || viennacl::is_subtraction<OP>::value),
                                  self_type &>::type
    operator = (const matrix_expression< const matrix_expression<const M1, const S1, OP1>,
                                          const matrix_expression<const M2, const S2, OP2>,
                                          OP> & proxy)
    {
      assert(  (proxy.lhs().size1() == size1())
            && (proxy.lhs().size2() == size2())
            && bool("Incompatible matrix sizes!"));
      
      if (internal_size() == 0)
      {
        rows_ = proxy.lhs().size1();
        columns_ = proxy.lhs().size2();
        viennacl::backend::memory_create(elements_, sizeof(SCALARTYPE)*internal_size());
      } 

      if (internal_size() > 0)
      {
        bool flip_sign_2 = (viennacl::is_subtraction<OP>::value ? true : false);
        if (viennacl::is_flip_sign_scalar<S2>::value)
          flip_sign_2 = !flip_sign_2;
        viennacl::linalg::ambm(*this, 
                                proxy.lhs().lhs(), proxy.lhs().rhs(), 1, (viennacl::is_division<OP1>::value ? true : false), (viennacl::is_flip_sign_scalar<S1>::value ? true : false),
                                proxy.rhs().lhs(), proxy.rhs().rhs(), 1, (viennacl::is_division<OP2>::value ? true : false), flip_sign_2);
      }
      return *this;
    }
    
    
    /** @brief Assigns the supplied identity matrix to the matrix. */
    self_type & operator = (identity_matrix<SCALARTYPE> const & m)
    {
      assert( (m.size1() == rows_    || rows_    == 0) && bool("Size mismatch!") );
      assert( (m.size2() == columns_ || columns_ == 0) && bool("Size mismatch!") );

      if (internal_size() == 0)
      {
        rows_    = m.size1();
        columns_ = m.size2();
        if (internal_size() > 0)
          viennacl::backend::memory_create(elements_, sizeof(SCALARTYPE)*internal_size());  
      }
      
      if (internal_size() > 0)
      {
        clear();
        viennacl::linalg::matrix_diagonal_assign(*this, m(0,0));
      }
      
      return *this;
    }
    
    /** @brief Assigns the supplied zero matrix to the matrix. */
    self_type & operator = (zero_matrix<SCALARTYPE> const & m)
    {
      assert( (m.size1() == rows_    || rows_    == 0) && bool("Size mismatch!") );
      assert( (m.size2() == columns_ || columns_ == 0) && bool("Size mismatch!") );

      if (internal_size() == 0)
      {
        rows_    = m.size1();
        columns_ = m.size2();
        if (internal_size() > 0)
          viennacl::backend::memory_create(elements_, sizeof(SCALARTYPE)*internal_size());  
      }
      
      if (internal_size() > 0)
        clear();
      
      return *this;
    }

    /** @brief Assigns the supplied scalar vector to the matrix. */
    self_type & operator = (scalar_matrix<SCALARTYPE> const & m)
    {
      assert( (m.size1() == rows_    || rows_    == 0) && bool("Size mismatch!") );
      assert( (m.size2() == columns_ || columns_ == 0) && bool("Size mismatch!") );

      if (internal_size() == 0)
      {
        rows_    = m.size1();
        columns_ = m.size2();
        if (internal_size() > 0)
          viennacl::backend::memory_create(elements_, sizeof(SCALARTYPE)*internal_size());  
      }
      
      if (internal_size() > 0)
      {
        clear();
        viennacl::linalg::matrix_assign(*this, m(0,0));
      }
      
      return *this;
    }
    
    
    // assign matrix range or vector slice
    template <typename M1>
    typename viennacl::enable_if<viennacl::is_any_dense_nonstructured_matrix<M1>::value,
                                  self_type & >::type
    operator = (const M1 & m1)
    {
      assert(  (m1.size1() == size1())
            && (m1.size2() == size2())
            && bool("Incompatible matrix sizes!"));
      
      if (internal_size() > 0)
        viennacl::linalg::am(*this, 
                              m1, SCALARTYPE(1.0), 1, false, false);
      
      return *this;
    }


    /** @brief Resizes the matrix.
    *   Existing entries can be preserved, but 
    *
    * @param rows       New number of rows
    * @param columns    New number of columns
    * @param preserve   If true, existing values are preserved. 
    */
    void resize(size_type rows, size_type columns, bool preserve = true)
    {
      assert( (rows > 0 && columns > 0) && bool("Check failed in matrix::resize(): Number of rows and columns must be positive!"));

      if (preserve && internal_size() > 0)
      {
        //get old entries:
        std::vector< SCALARTYPE > old_entries(internal_size());
        viennacl::backend::memory_read(elements_, 0, sizeof(SCALARTYPE)*internal_size(), &(old_entries[0]));
        
        //set up entries of new matrix:
        std::vector< SCALARTYPE > new_entries(  viennacl::tools::roundUpToNextMultiple<vcl_size_t>(rows,    ALIGNMENT)
                                              * viennacl::tools::roundUpToNextMultiple<vcl_size_t>(columns, ALIGNMENT));
        for (size_type i=0; i<rows; ++i)
        {
          if (i >= rows_)
            continue;
            
          for (size_type j=0; j<columns; ++j)
          {
            if (j >= columns_)
              continue;
            new_entries[F::mem_index(i, j, viennacl::tools::roundUpToNextMultiple<vcl_size_t>(rows, ALIGNMENT), viennacl::tools::roundUpToNextMultiple<vcl_size_t>(columns, ALIGNMENT))] 
                = old_entries[F::mem_index(i, j, internal_size1(), internal_size2())];
          }
        }
        
        //copy new entries to GPU:
        rows_ = rows;
        columns_ = columns;
        viennacl::backend::memory_create(elements_, sizeof(SCALARTYPE)*new_entries.size(), &(new_entries[0]));
      }
      else //discard old entries:
      {
        rows_ = rows;
        columns_ = columns;
        
        std::vector< SCALARTYPE > new_entries(  viennacl::tools::roundUpToNextMultiple<vcl_size_t>(rows,    ALIGNMENT) 
                                              * viennacl::tools::roundUpToNextMultiple<vcl_size_t>(columns, ALIGNMENT));
        viennacl::backend::memory_create(elements_, sizeof(SCALARTYPE)*internal_size(), &(new_entries[0]));
      }
    }
    
    
    //read-write access to an element of the vector
    /** @brief Read-write access to a single element of the vector
    */
    entry_proxy<SCALARTYPE> operator()(size_type row_index, size_type col_index)
    {
      return entry_proxy<SCALARTYPE>(F::mem_index(row_index, col_index, internal_size1(), internal_size2()), elements_);
    }
    
    /** @brief Read access to a single element of the vector
    */
    scalar<SCALARTYPE> operator()(size_type row_index, size_type col_index) const
    {
      scalar<SCALARTYPE> tmp;
      
      viennacl::backend::memory_copy(elements_, tmp.handle(), 
                                      sizeof(SCALARTYPE) * F::mem_index(row_index, col_index, internal_size1(), internal_size2()), 0, //offsets
                                      sizeof(SCALARTYPE));
      return tmp;
    }
    
    //
    // Operator overloads for enabling implicit conversions:
    //
    self_type & operator += (const self_type & other) 
    {
      viennacl::linalg::ambm(*this,
                              *this, SCALARTYPE(1.0), 1, false, false,
                              other, SCALARTYPE(1.0), 1, false, false);
      return *this;
    }

    self_type & operator -= (const self_type & other) 
    {
      viennacl::linalg::ambm(*this,
                              *this, SCALARTYPE(1.0), 1, false, false,
                              other, SCALARTYPE(1.0), 1, false, true);
      return *this;
    }
    
    /** @brief Scales a matrix by a CPU scalar value
    */
    self_type & operator *= (SCALARTYPE val)
    {
      //viennacl::linalg::inplace_mult(*this, val);
      viennacl::linalg::am(*this,
                            *this, val, 1, false, false);
      return *this;
    }
    
    /** @brief Scales this matrix by a CPU scalar value
    */
    self_type & operator /= (SCALARTYPE val)
    {
      //viennacl::linalg::inplace_mult(*this, static_cast<SCALARTYPE>(1) / val);
      viennacl::linalg::am(*this,
                            *this, val, 1, true, false);
      return *this;
    }
    
    
    //
    // Matrix-Matrix products:
    //
    
    //this = A * B and related (with trans())
    template <typename M1, typename M2>
    typename viennacl::enable_if<   (viennacl::is_any_dense_nonstructured_matrix<M1>::value || viennacl::is_any_dense_nonstructured_transposed_matrix<M1>::value)
                                  && (viennacl::is_any_dense_nonstructured_matrix<M2>::value || viennacl::is_any_dense_nonstructured_transposed_matrix<M2>::value),
                                  self_type & >::type
    operator = (const matrix_expression< const M1, const M2, op_prod > & proxy) 
    {
      viennacl::linalg::prod_impl(proxy.lhs(), proxy.rhs(), *this, 1.0, 0.0);
      return *this;
    }


    /** @brief Returns the number of rows */
    const size_type & size1() const { return rows_;}
    /** @brief Returns the number of columns */
    const size_type & size2() const { return columns_; }
    
    /** @brief Resets all entries to zero */
    void clear()
    {
      std::vector<SCALARTYPE> temp(internal_size());
      viennacl::backend::memory_write(elements_, 0, sizeof(SCALARTYPE)*temp.size(), &(temp[0]));
    }
    
    
    //const unsigned int row_stride() const { return roundUpToNextMultiple<unsigned int>(columns(), ALIGNMENT); }
    /** @brief Returns the internal number of rows. Usually required for launching OpenCL kernels only */
    size_type internal_size1() const { return viennacl::tools::roundUpToNextMultiple<vcl_size_t>(rows_, ALIGNMENT); }
    /** @brief Returns the internal number of columns. Usually required for launching OpenCL kernels only */
    size_type internal_size2() const { return viennacl::tools::roundUpToNextMultiple<vcl_size_t>(columns_, ALIGNMENT); }
    /** @brief Returns the total amount of allocated memory in multiples of sizeof(SCALARTYPE) */
    size_type internal_size() const { return internal_size1() * internal_size2(); }
    
    /** @brief Returns the OpenCL handle, non-const-version */
          handle_type & handle()       { return elements_; }
    /** @brief Returns the OpenCL handle, const-version */
    const handle_type & handle() const { return elements_; }
    
    void switch_memory_domain(viennacl::memory_types new_domain)
    {
      viennacl::backend::switch_memory_domain<SCALARTYPE>(elements_, new_domain);
    }
    
    viennacl::memory_types memory_domain() const
    {
      return elements_.get_active_handle_id();
    }

    
  private:
    size_type rows_;
    size_type columns_;
    handle_type elements_;
  }; //matrix

  /** @brief Prints the matrix. Output is compatible to boost::numeric::ublas
  *
  * @param s            STL output stream
  * @param gpu_matrix   A dense ViennaCL matrix
  */
  template<class SCALARTYPE, typename F, unsigned int ALIGNMENT>
  std::ostream & operator<<(std::ostream & s, const matrix<SCALARTYPE, F, ALIGNMENT> & gpu_matrix)
  {
    typedef typename matrix<SCALARTYPE, F, ALIGNMENT>::size_type      size_type;
    
    std::vector<SCALARTYPE> tmp(gpu_matrix.internal_size());
    viennacl::backend::memory_read(gpu_matrix.handle(), 0, sizeof(SCALARTYPE) * gpu_matrix.internal_size(), &(tmp[0]));
    
    s << "[" << gpu_matrix.size1() << "," << gpu_matrix.size2() << "]";
    
    s << "(";
    for (size_type i = 0; i < gpu_matrix.size1(); ++i)
    {
      s << "(";
      for (size_type j = 0; j < gpu_matrix.size2(); ++j)
      {
        s << tmp[F::mem_index(i, j, gpu_matrix.internal_size1(), gpu_matrix.internal_size2())];
        if (j < gpu_matrix.size2() - 1)
          s << ",";
      }
      s << ")";
      if (i < gpu_matrix.size1() - 1)
        s << ",";
    }
    s << ")";
    return s;
  }

  /** @brief Prints the matrix. Output is compatible to boost::numeric::ublas
  *
  * @param s            STL output stream
  * @param expr         A matrix expression
  */
  template<typename LHS, typename RHS, typename OP>
  std::ostream & operator<<(std::ostream & s, const matrix_expression<LHS, RHS, OP> & expr)
  {
    typedef typename viennacl::tools::CPU_SCALAR_TYPE_DEDUCER< typename tools::CONST_REMOVER<LHS>::ResultType >::ResultType     ScalarType;

    matrix<ScalarType> temp = expr;
    s << temp;
    return s;
  }
  
  /** @brief Returns an expression template class representing a transposed matrix */
  template<typename M1>
  typename viennacl::enable_if<viennacl::is_any_dense_nonstructured_matrix<M1>::value,
                                matrix_expression< const M1, const M1, op_trans>
                              >::type
  trans(const M1 & mat)
  {
    return matrix_expression< const M1, const M1, op_trans>(mat, mat);
  }
  
  
  /////////////////////// transfer operations: //////////////////////////////////////

  //
  //cpu to gpu, generic type:
  //
  /** @brief Copies a dense matrix from the host (CPU) to the OpenCL device (GPU or multi-core CPU)
  *
  * @param cpu_matrix   A dense matrix on the host. Type requirements: .size1() returns number of rows, .size2() returns number of columns. Access to entries via operator()
  * @param gpu_matrix   A dense ViennaCL matrix
  */
  template <typename CPU_MATRIX, typename SCALARTYPE, typename F, unsigned int ALIGNMENT>
  void copy(const CPU_MATRIX & cpu_matrix,
            matrix<SCALARTYPE, F, ALIGNMENT> & gpu_matrix )
  {
    typedef typename matrix<SCALARTYPE, F, ALIGNMENT>::size_type      size_type;
    
    //std::cout << "Copying CPU_MATRIX!" << std::endl;
    //std::cout << "Size at begin: " << gpu_matrix.size1() << ", " << gpu_matrix.size2() << std::endl;
    if (gpu_matrix.size1() == 0 || gpu_matrix.size2() == 0)
    {
      gpu_matrix.resize(cpu_matrix.size1(),
                        cpu_matrix.size2(), false);
    }
    else
    {
      assert( (gpu_matrix.size1() == cpu_matrix.size1()) 
              && (gpu_matrix.size2() == cpu_matrix.size2())
              && bool("matrix size mismatch")
            );
    }

    std::vector<SCALARTYPE> data(gpu_matrix.internal_size());
    for (size_type i = 0; i < gpu_matrix.size1(); ++i)
    {
      for (size_type j = 0; j < gpu_matrix.size2(); ++j) 
        data[F::mem_index(i, j, gpu_matrix.internal_size1(), gpu_matrix.internal_size2())] = cpu_matrix(i,j);
    }
    
    viennacl::backend::memory_create(gpu_matrix.handle(), sizeof(SCALARTYPE) * data.size(), &(data[0]));
    //gpu_matrix.elements_ = viennacl::ocl::current_context().create_memory(CL_MEM_READ_WRITE, data);
    //std::cout << "Size at end: " << gpu_matrix.size1() << ", " << gpu_matrix.size2() << std::endl;
  }
  
  //
  //cpu to gpu, STL type:
  //
  /** @brief Copies a dense STL-type matrix from the host (CPU) to the OpenCL device (GPU or multi-core CPU)
  *
  * @param cpu_matrix   A dense matrix on the host of type std::vector< std::vector<> >. cpu_matrix[i][j] returns the element in the i-th row and j-th columns (both starting with zero)
  * @param gpu_matrix   A dense ViennaCL matrix
  */
  template <typename SCALARTYPE, typename A1, typename A2, typename F, unsigned int ALIGNMENT>
  void copy(const std::vector< std::vector<SCALARTYPE, A1>, A2> & cpu_matrix,
            matrix<SCALARTYPE, F, ALIGNMENT> & gpu_matrix )
  {
    typedef typename matrix<SCALARTYPE, F, ALIGNMENT>::size_type      size_type;
    
    if (gpu_matrix.size1() == 0 || gpu_matrix.size2() == 0)
    {
      gpu_matrix.resize(cpu_matrix.size(),
                        cpu_matrix[0].size(),
                        false);
    }
    else
    {
      assert( (gpu_matrix.size1() == cpu_matrix.size()) 
              && (gpu_matrix.size2() == cpu_matrix[0].size())
              && bool("matrix size mismatch")
            );
    }

    std::vector<SCALARTYPE> data(gpu_matrix.internal_size());
    for (size_type i = 0; i < gpu_matrix.size1(); ++i)
    {
      for (size_type j = 0; j < gpu_matrix.size2(); ++j) 
        data[F::mem_index(i, j, gpu_matrix.internal_size1(), gpu_matrix.internal_size2())] = cpu_matrix[i][j];
    }
    
    viennacl::backend::memory_create(gpu_matrix.handle(), sizeof(SCALARTYPE) * data.size(), &(data[0]));
    //gpu_matrix.elements_ = viennacl::ocl::current_context().create_memory(CL_MEM_READ_WRITE, data);
  }
  
  
  //
  //cpu to gpu, another STL type:
  //
  /** @brief Copies a dense matrix from the host (CPU) to the OpenCL device (GPU or multi-core CPU) without temporary. Matrix-Layout on CPU must be equal to the matrix-layout on the GPU.
  *
  * @param cpu_matrix_begin   Pointer to the first matrix entry. Cf. iterator concept in STL
  * @param cpu_matrix_end     Pointer past the last matrix entry. Cf. iterator concept in STL
  * @param gpu_matrix         A dense ViennaCL matrix
  */
  template <typename SCALARTYPE, typename F, unsigned int ALIGNMENT>
  void fast_copy(SCALARTYPE * cpu_matrix_begin,
                  SCALARTYPE * cpu_matrix_end,
                  matrix<SCALARTYPE, F, ALIGNMENT> & gpu_matrix)
  {
    viennacl::backend::memory_create(gpu_matrix.handle(), sizeof(SCALARTYPE) * (cpu_matrix_end - cpu_matrix_begin), cpu_matrix_begin);
    /*gpu_matrix.elements_ = viennacl::ocl::current_context().create_memory(CL_MEM_READ_WRITE,
                                                                          sizeof(SCALARTYPE) * (cpu_matrix_end - cpu_matrix_begin),
                                                                          cpu_matrix_begin);*/
  }
  
  
  #ifdef VIENNACL_WITH_EIGEN
  /** @brief Copies a dense Eigen matrix from the host (CPU) to the OpenCL device (GPU or multi-core CPU)
  *
  * @param cpu_matrix   A dense MTL matrix. cpu_matrix(i, j) returns the element in the i-th row and j-th columns (both starting with zero)
  * @param gpu_matrix   A dense ViennaCL matrix
  */
  template <typename F, unsigned int ALIGNMENT>
  void copy(const Eigen::MatrixXf & cpu_matrix,
            matrix<float, F, ALIGNMENT> & gpu_matrix)
  {
    typedef typename matrix<float, F, ALIGNMENT>::size_type      size_type;
    
    if (gpu_matrix.size1() == 0 || gpu_matrix.size2() == 0)
    {
      gpu_matrix.resize(cpu_matrix.rows(),
                        cpu_matrix.cols(),
                        false);
    }
    else
    {
      assert( (gpu_matrix.size1() == static_cast<std::size_t>(cpu_matrix.rows())) 
              && (gpu_matrix.size2() == static_cast<std::size_t>(cpu_matrix.cols()))
              && bool("matrix size mismatch")
            );
    }

    std::vector<float> data(gpu_matrix.internal_size());
    for (size_type i = 0; i < gpu_matrix.size1(); ++i)
    {
      for (size_type j = 0; j < gpu_matrix.size2(); ++j) 
        data[F::mem_index(i, j, gpu_matrix.internal_size1(), gpu_matrix.internal_size2())] = cpu_matrix(i,j);
    }
    
    viennacl::backend::memory_create(gpu_matrix.handle(), sizeof(float) * data.size(), &(data[0]));
    //gpu_matrix.elements_ = viennacl::ocl::current_context().create_memory(CL_MEM_READ_WRITE, data);
  }
  
  /** @brief Copies a dense Eigen matrix from the host (CPU) to the OpenCL device (GPU or multi-core CPU)
  *
  * @param cpu_matrix   A dense MTL matrix. cpu_matrix(i, j) returns the element in the i-th row and j-th columns (both starting with zero)
  * @param gpu_matrix   A dense ViennaCL matrix
  */
  template <typename F, unsigned int ALIGNMENT>
  void copy(const Eigen::MatrixXd & cpu_matrix,
            matrix<double, F, ALIGNMENT> & gpu_matrix)
  {
    typedef typename matrix<double, F, ALIGNMENT>::size_type      size_type;
    
    if (gpu_matrix.size1() == 0 || gpu_matrix.size2() == 0)
    {
      gpu_matrix.resize(cpu_matrix.rows(),
                        cpu_matrix.cols(),
                        false);
    }
    else
    {
      assert( (gpu_matrix.size1() == static_cast<std::size_t>(cpu_matrix.rows())) 
              && (gpu_matrix.size2() == static_cast<std::size_t>(cpu_matrix.cols()))
              && bool("matrix size mismatch")
            );
    }

    std::vector<double> data(gpu_matrix.internal_size());
    for (size_type i = 0; i < gpu_matrix.size1(); ++i)
    {
      for (size_type j = 0; j < gpu_matrix.size2(); ++j) 
        data[F::mem_index(i, j, gpu_matrix.internal_size1(), gpu_matrix.internal_size2())] = cpu_matrix(i,j);
    }
    
    viennacl::backend::memory_create(gpu_matrix.handle(), sizeof(double) * data.size(), &(data[0]));
    //gpu_matrix.elements_ = viennacl::ocl::current_context().create_memory(CL_MEM_READ_WRITE, data);
  }
  #endif
  
  #ifdef VIENNACL_WITH_MTL4
  /** @brief Copies a dense MTL matrix from the host (CPU) to the OpenCL device (GPU or multi-core CPU)
  *
  * @param cpu_matrix   A dense MTL matrix. cpu_matrix(i, j) returns the element in the i-th row and j-th columns (both starting with zero)
  * @param gpu_matrix   A dense ViennaCL matrix
  */
  template <typename SCALARTYPE, typename T, typename F, unsigned int ALIGNMENT>
  void copy(const mtl::dense2D<SCALARTYPE, T>& cpu_matrix,
            matrix<SCALARTYPE, F, ALIGNMENT> & gpu_matrix)
  {
    typedef typename matrix<SCALARTYPE, F, ALIGNMENT>::size_type      size_type;
    
    if (gpu_matrix.size1() == 0 || gpu_matrix.size2() == 0)
    {
      gpu_matrix.resize(cpu_matrix.num_rows(),
                        cpu_matrix.num_cols(),
                        false);
    }
    else
    {
      assert( (gpu_matrix.size1() == cpu_matrix.num_rows()) 
              && (gpu_matrix.size2() == cpu_matrix.num_cols())
              && bool("matrix size mismatch")
            );
    }

    std::vector<SCALARTYPE> data(gpu_matrix.internal_size());
    for (size_type i = 0; i < gpu_matrix.size1(); ++i)
    {
      for (size_type j = 0; j < gpu_matrix.size2(); ++j) 
        data[F::mem_index(i, j, gpu_matrix.internal_size1(), gpu_matrix.internal_size2())] = cpu_matrix[i][j];
    }
    
    viennacl::backend::memory_create(gpu_matrix.handle(), sizeof(SCALARTYPE) * data.size(), &(data[0]));
    //gpu_matrix.elements_ = viennacl::ocl::current_context().create_memory(CL_MEM_READ_WRITE, data);
  }
  #endif
  
  
  
  
  //
  //gpu to cpu, generic type
  //
  /** @brief Copies a dense matrix from the OpenCL device (GPU or multi-core CPU) to the host (CPU). 
  *
  * @param gpu_matrix   A dense ViennaCL matrix
  * @param cpu_matrix   A dense memory on the host. Must have at least as many rows and columns as the gpu_matrix! Type requirement: Access to entries via operator()
  */
  template <typename CPU_MATRIX, typename SCALARTYPE, typename F, unsigned int ALIGNMENT>
  void copy(const matrix<SCALARTYPE, F, ALIGNMENT> & gpu_matrix,
            CPU_MATRIX & cpu_matrix )
  {
    typedef typename matrix<float, F, ALIGNMENT>::size_type      size_type;
    
    if ( (gpu_matrix.size1() > 0) && (gpu_matrix.size2() > 0) )
    {
      std::vector<SCALARTYPE> temp_buffer(gpu_matrix.internal_size());
      viennacl::backend::memory_read(gpu_matrix.handle(), 0, sizeof(SCALARTYPE)*gpu_matrix.internal_size(), &(temp_buffer[0]));
      
      //now copy entries to cpu_matrix:
      for (size_type i = 0; i < gpu_matrix.size1(); ++i)
        for (size_type j = 0; j < gpu_matrix.size2(); ++j) 
          cpu_matrix(i,j) = temp_buffer[F::mem_index(i, j, gpu_matrix.internal_size1(), gpu_matrix.internal_size2())];
    }
  }

  //gpu to cpu, STL type
  /** @brief Copies a dense matrix from the OpenCL device (GPU or multi-core CPU) to the host (CPU). 
  *
  * @param gpu_matrix   A dense ViennaCL matrix
  * @param cpu_matrix   A dense memory on the host using STL types, typically std::vector< std::vector<> > Must have at least as many rows and columns as the gpu_matrix! Type requirement: Access to entries via operator()
  */
  template <typename SCALARTYPE, typename A1, typename A2, typename F, unsigned int ALIGNMENT>
  void copy(const matrix<SCALARTYPE, F, ALIGNMENT> & gpu_matrix,
            std::vector< std::vector<SCALARTYPE, A1>, A2> & cpu_matrix)
  {
    typedef typename matrix<float, F, ALIGNMENT>::size_type      size_type;
    
    if ( (gpu_matrix.size1() > 0) && (gpu_matrix.size2() > 0) 
        && (cpu_matrix.size() >= gpu_matrix.size1()) && (cpu_matrix[0].size() >= gpu_matrix.size2()))
    {
      std::vector<SCALARTYPE> temp_buffer(gpu_matrix.internal_size());
      viennacl::backend::memory_read(gpu_matrix.handle(), 0, sizeof(SCALARTYPE)*gpu_matrix.internal_size(), &(temp_buffer[0]));
      
      //now copy entries to cpu_matrix:
      for (size_type i = 0; i < gpu_matrix.size1(); ++i)
        for (size_type j = 0; j < gpu_matrix.size2(); ++j) 
          cpu_matrix[i][j] = temp_buffer[F::mem_index(i, j, gpu_matrix.internal_size1(), gpu_matrix.internal_size2())];
    }
  }

  //gpu to cpu, STL type
  /** @brief Copies a dense matrix from the OpenCL device (GPU or multi-core CPU) to the host (CPU). 
  *
  * @param gpu_matrix         A dense ViennaCL matrix
  * @param cpu_matrix_begin   Pointer to the output memory on the CPU. User must ensure that provided memory is large enough.
  */
  template <typename SCALARTYPE, typename F, unsigned int ALIGNMENT>
  void fast_copy(const matrix<SCALARTYPE, F, ALIGNMENT> & gpu_matrix,
                  SCALARTYPE * cpu_matrix_begin)
  {
    viennacl::backend::memory_read(gpu_matrix.handle(), 0, sizeof(SCALARTYPE)*gpu_matrix.internal_size(), cpu_matrix_begin);
  }



  /////////////////////// matrix operator overloads to follow ////////////////////////////////////////////


  // operator +
  /** @brief Generic 'catch-all' overload, which enforces a temporary if the expression tree gets too deep. */
  template <typename LHS1, typename RHS1, typename OP1,
            typename LHS2, typename RHS2, typename OP2>
  typename matrix_expression< LHS1, RHS1, OP1>::matrix_type
  operator + (matrix_expression< LHS1, RHS1, OP1> const & proxy1,
              matrix_expression< LHS2, RHS2, OP2> const & proxy2)
  {
    assert(    (viennacl::traits::size1(proxy1) == viennacl::traits::size1(proxy2))
            && (viennacl::traits::size2(proxy1) == viennacl::traits::size2(proxy1))
            && bool("Incompatible matrix sizes!"));
    typename matrix_expression< LHS1, RHS1, OP1>::matrix_type result(proxy1.size());
    result = proxy1;
    result += proxy2;
    return result;
  }
  
  /** @brief Operator overload for m1 + m2, where m1 and m2 are either dense matrices, matrix ranges, or matrix slices */
  template <typename M1, typename M2>
  typename viennacl::enable_if<    viennacl::is_any_dense_nonstructured_matrix<M1>::value
                                && viennacl::is_any_dense_nonstructured_matrix<M2>::value,
                                matrix_expression< const M1, const M2, op_add > 
                              >::type
  operator + (const M1 & m1, const M2 & m2) 
  {
    return matrix_expression< const M1, const M2, op_add > (m1, m2);
  }

  /** @brief Operator overload for the addition of a matrix expression m1 + m2 @ beta, where @ is either product or division, and beta is either a CPU or GPU scalar. */
  template <typename M1, typename M2, typename M3, typename OP>
  typename viennacl::enable_if<    viennacl::is_any_dense_nonstructured_matrix<M1>::value
                                && viennacl::is_any_dense_nonstructured_matrix<M2>::value
                                && viennacl::is_any_dense_nonstructured_matrix<M3>::value,
                                matrix_expression< const M1,
                                                    const matrix_expression<const M2, const M3, OP>,
                                                    op_add >
                              >::type
  operator + (const M1 & m1,
              const matrix_expression< const M2, const M3, OP> & proxy) 
  {
    return matrix_expression< const M1,
                              const matrix_expression<const M2, const M3, OP>,
                              op_add > (m1, proxy);
  }
  
  /** @brief Operator overload for the addition of a matrix expression m1 @ alpha + m2, where @ is either product or division, and beta is either a CPU or GPU scalar. */
  template <typename M1, typename M2, typename M3, typename OP>
  typename viennacl::enable_if<    viennacl::is_any_dense_nonstructured_matrix<M1>::value
                                && viennacl::is_any_dense_nonstructured_matrix<M2>::value
                                && viennacl::is_any_dense_nonstructured_matrix<M3>::value,
                                matrix_expression< const matrix_expression<const M1, const M2, OP>,
                                                    const M3,
                                                    op_add >
                              >::type
  operator + (const matrix_expression< const M1, const M2, OP> & proxy,
              const M3 & m3) 
  {
    return matrix_expression< const matrix_expression<const M1, const M2, OP>,
                              const M3,
                              op_add > (proxy, m3);
  }
  
  
  /** @brief Operator overload for the addition of a matrix expression m1 @ alpha + m2 @ beta, where @ denotes either product or division, and alpha, beta are either CPU or GPU scalars.
  *
  * @param lhs   Left hand side vector expression
  * @param rhs     Right hand side vector (also -range and -slice is allowed)
  */
  template <typename M1, typename S1, typename OP1,
            typename M2, typename S2, typename OP2>
  typename viennacl::enable_if<    viennacl::is_any_dense_nonstructured_matrix<M1>::value && viennacl::is_any_scalar<S1>::value
                                && viennacl::is_any_dense_nonstructured_matrix<M2>::value && viennacl::is_any_scalar<S2>::value,
                                matrix_expression<const matrix_expression<const M1, const S1, OP1>,
                                                  const matrix_expression<const M2, const S2, OP2>,
                                                  op_add>
                              >::type
  operator + (matrix_expression<const M1, const S1, OP1> const & lhs,
              matrix_expression<const M2, const S2, OP2> const & rhs)
  {
    return matrix_expression<const matrix_expression<const M1, const S1, OP1>,
                              const matrix_expression<const M2, const S2, OP2>,
                              op_add>(lhs, rhs);
  }

  
  
  
  // operator +=
  template <typename M1, typename M2>
  typename viennacl::enable_if<    viennacl::is_any_dense_nonstructured_matrix<M1>::value
                                && viennacl::is_any_dense_nonstructured_matrix<M2>::value,
                                M1 & 
                              >::type
  operator += (M1 & m1, const M2 & other) 
  {
    typedef typename viennacl::result_of::cpu_value_type<typename M1::value_type>::type   ScalarType;
    viennacl::linalg::ambm(m1,
                            m1,    ScalarType(1.0), 1, false, false,
                            other, ScalarType(1.0), 1, false, false);
    return m1;
  }

  /** @brief Inplace addition of a scaled matrix, i.e. m1 += m2 @ alpha, where @ is either product or division and alpha is either a CPU or a GPU scalar
  */
  template <typename M1, typename M2, typename S2, typename OP>
  typename viennacl::enable_if<    viennacl::is_any_dense_nonstructured_matrix<M1>::value
                                && viennacl::is_any_dense_nonstructured_matrix<M2>::value 
                                && viennacl::is_any_scalar<S2>::value,
                                M1 &>::type
  operator += (M1 & m1, 
                const matrix_expression< const M2, const S2, OP> & proxy)
  {
    typedef typename viennacl::result_of::cpu_value_type<M1>::type   cpu_value_type;
    
    assert(   (proxy.lhs().size1() == m1.size1())
            && (proxy.lhs().size2() == m1.size2())
            && bool("Incompatible matrix sizes!"));

    if (m1.size1() > 0 && m1.size2() > 0)
      viennacl::linalg::ambm(m1, 
                              m1,  cpu_value_type(1.0), 1, false,                                             false,
                              proxy.lhs(), proxy.rhs(), 1, (viennacl::is_division<OP>::value ? true : false), (viennacl::is_flip_sign_scalar<S2>::value ? true : false) );
    return m1;
  }

  /** @brief Implementation of the operation m1 += m2 +- m3
  *
  * @param m1     The result matrix where m2 +- m3 is added to
  * @param proxy  An expression template proxy class.
  */
  template <typename M1, typename M2, typename M3, typename OP>
  typename viennacl::enable_if<    viennacl::is_any_dense_nonstructured_matrix<M1>::value
                                && viennacl::is_any_dense_nonstructured_matrix<M2>::value
                                && viennacl::is_any_dense_nonstructured_matrix<M3>::value
                                && (viennacl::is_addition<OP>::value || viennacl::is_subtraction<OP>::value),
                                M1 &>::type
  operator += (M1 & m1, 
                const matrix_expression< const M2, const M3, OP> & proxy)
  {
    typedef typename viennacl::result_of::cpu_value_type<M1>::type   cpu_value_type;
    
    assert(   (proxy.lhs().size1() == m1.size1())
            && (proxy.lhs().size2() == m1.size2())
            && bool("Incompatible matrix sizes!"));

    if (m1.size1() > 0 && m1.size2() > 0)
      viennacl::linalg::ambm_m(m1, 
                                proxy.lhs(), cpu_value_type(1.0), 1, false, false,
                                proxy.rhs(), cpu_value_type(1.0), 1, false, (viennacl::is_subtraction<OP>::value ? true : false) );
    return m1;
  }
  
  /** @brief Implementation of the operation m1 += m2 +- m3 @ beta, where @ is either product or division, and alpha, beta are either CPU or GPU scalars
  *
  * @param m1     The result matrix where m2 +- m3 @ beta is added to
  * @param proxy  An expression template proxy class.
  */
  template <typename M1,
            typename M2,
            typename M3, typename S3, typename OP3,
            typename OP>
  typename viennacl::enable_if<    viennacl::is_any_dense_nonstructured_matrix<M1>::value
                                && viennacl::is_any_dense_nonstructured_matrix<M2>::value
                                && viennacl::is_any_dense_nonstructured_matrix<M3>::value && viennacl::is_any_scalar<S3>::value && (viennacl::is_product<OP3>::value || viennacl::is_division<OP3>::value)
                                && (viennacl::is_addition<OP>::value || viennacl::is_subtraction<OP>::value),
                                M1 &>::type
  operator += (M1 & m1,
                const matrix_expression< const M2,
                                        const matrix_expression<const M3, const S3, OP3>,
                                        OP> & proxy)
  {
    typedef typename viennacl::result_of::cpu_value_type<M1>::type   cpu_value_type;
    
    assert(   (proxy.lhs().size1() == m1.size1())
            && (proxy.lhs().size2() == m1.size2())
            && bool("Incompatible matrix sizes!"));

    if (m1.size1() > 0 && m1.size2() > 0)
    {
      bool flip_sign_3 = (viennacl::is_subtraction<OP>::value ? true : false);
      if (viennacl::is_flip_sign_scalar<S3>::value)
        flip_sign_3 = !flip_sign_3;
      viennacl::linalg::ambm_m(m1, 
                                proxy.lhs(),       cpu_value_type(1.0), 1, false                                             , false,
                                proxy.rhs().lhs(), proxy.rhs().rhs(),   1, (viennacl::is_division<OP3>::value ? true : false), flip_sign_3 );
    }
    return m1;
  }

  /** @brief Implementation of the operation m1 += m2 @ alpha +- m3, where @ is either product or division, and alpha, beta are either CPU or GPU scalars
  *
  * @param m1     The result matrix where m2 @ alpha +- m3 is added to
  * @param proxy  An expression template proxy class.
  */
  template <typename M1,
            typename M2, typename S2, typename OP2,
            typename M3,
            typename OP>
  typename viennacl::enable_if<    viennacl::is_any_dense_nonstructured_matrix<M1>::value
                                && viennacl::is_any_dense_nonstructured_matrix<M2>::value && viennacl::is_any_scalar<S2>::value && (viennacl::is_product<OP2>::value || viennacl::is_division<OP2>::value)
                                && viennacl::is_any_dense_nonstructured_matrix<M3>::value
                                && (viennacl::is_addition<OP>::value || viennacl::is_subtraction<OP>::value),
                                M1 &>::type
  operator += (M1 & m1,
                const matrix_expression< const matrix_expression<const M2, const S2, OP2>,
                                        const M3,
                                        OP> & proxy)
  {
    typedef typename viennacl::result_of::cpu_value_type<M1>::type   cpu_value_type;
    
    assert(   (proxy.size1() == m1.size1())
            && (proxy.size2() == m1.size2())
            && bool("Incompatible matrix sizes!"));

    if (m1.size1() > 0 && m1.size2() > 0)
      viennacl::linalg::ambm_m(m1, 
                                proxy.lhs().lhs(),   proxy.lhs().rhs(), 1, (viennacl::is_division<OP2>::value ? true : false), (viennacl::is_flip_sign_scalar<S2>::value ? true : false),
                                proxy.rhs(),       cpu_value_type(1.0), 1, false                                             , (viennacl::is_subtraction<OP>::value ? true : false) );
    return m1;
  }
  
  /** @brief Implementation of the operation m1 += m2 @ alpha +- m3 @ beta, where @ is either product or division, and alpha, beta are either CPU or GPU scalars
  *
  * @param m1     The result matrix where m2 @ alpha +- m3 @ beta is added to
  * @param proxy  An expression template proxy class.
  */
  template <typename M1,
            typename M2, typename S2, typename OP2,
            typename M3, typename S3, typename OP3,
            typename OP>
  typename viennacl::enable_if<    viennacl::is_any_dense_nonstructured_matrix<M1>::value
                                && viennacl::is_any_dense_nonstructured_matrix<M2>::value && viennacl::is_any_scalar<S2>::value && (viennacl::is_product<OP2>::value || viennacl::is_division<OP2>::value)
                                && viennacl::is_any_dense_nonstructured_matrix<M3>::value && viennacl::is_any_scalar<S3>::value && (viennacl::is_product<OP3>::value || viennacl::is_division<OP3>::value)
                                && (viennacl::is_addition<OP>::value || viennacl::is_subtraction<OP>::value),
                                M1 &>::type
  operator += (M1 & m1,
                const matrix_expression< const matrix_expression<const M2, const S2, OP2>,
                                        const matrix_expression<const M3, const S3, OP3>,
                                        OP> & proxy)
  {
    assert(   (proxy.size1() == m1.size1())
            && (proxy.size2() == m1.size2())
            && bool("Incompatible matrix sizes!"));

    if (m1.size1() > 0 && m1.size2() > 0)
    {
      bool flip_sign_3 = (viennacl::is_subtraction<OP>::value ? true : false);
      if (viennacl::is_flip_sign_scalar<S3>::value)
        flip_sign_3 = !flip_sign_3;
      viennacl::linalg::ambm_m(m1, 
                                proxy.lhs().lhs(), proxy.lhs().rhs(), 1, (viennacl::is_division<OP2>::value ? true : false), (viennacl::is_flip_sign_scalar<S2>::value ? true : false),
                                proxy.rhs().lhs(), proxy.rhs().rhs(), 1, (viennacl::is_division<OP3>::value ? true : false), flip_sign_3 );
    }
    return m1;
  }
  
  
  
  template <typename M1, typename V1, typename V2>
  typename viennacl::enable_if<    viennacl::is_any_dense_nonstructured_matrix<M1>::value
                                && viennacl::is_any_dense_nonstructured_vector<V1>::value
                                && viennacl::is_any_dense_nonstructured_vector<V2>::value,
                                M1 & 
                              >::type
  operator += (M1 & m1,
                const matrix_expression< const V1, const V2, op_prod > & proxy) 
  {
    typedef typename viennacl::result_of::cpu_value_type<typename M1::value_type>::type   ScalarType;
    viennacl::linalg::scaled_rank_1_update(m1,
                                            ScalarType(1.0), 1, false, false,
                                            proxy.lhs(),
                                            proxy.rhs());
    return m1;
  }

  template <typename M1, typename V1, typename V2, typename S1>
  typename viennacl::enable_if<    viennacl::is_any_dense_nonstructured_matrix<M1>::value
                                && viennacl::is_any_dense_nonstructured_vector<V1>::value
                                && viennacl::is_any_dense_nonstructured_vector<V2>::value
                                && viennacl::is_any_scalar<S1>::value,
                                M1 & 
                              >::type
  operator += (M1 & m1,
                const matrix_expression< const matrix_expression< const V1, const V2, op_prod >,
                                        const S1,
                                        op_prod > & proxy) 
  {
    viennacl::linalg::scaled_rank_1_update(m1,
                                            proxy.rhs(), 1, false, (viennacl::is_flip_sign_scalar<S1>::value ? true : false),
                                            proxy.lhs().lhs(),
                                            proxy.lhs().rhs());
    return m1;
  }
  
  //C += A * B and related (with trans())
  template <typename M1, typename M2, typename M3>
  typename viennacl::enable_if<   viennacl::is_any_dense_nonstructured_matrix<M1>::value
                                && (viennacl::is_any_dense_nonstructured_matrix<M2>::value || viennacl::is_any_dense_nonstructured_transposed_matrix<M2>::value)
                                && (viennacl::is_any_dense_nonstructured_matrix<M3>::value || viennacl::is_any_dense_nonstructured_transposed_matrix<M3>::value),
                                M1 & >::type
  operator += (M1 & m1, const matrix_expression< const M2, const M3, op_prod > & proxy) 
  {
    viennacl::linalg::prod_impl(proxy.lhs(), proxy.rhs(), m1, 1.0, 1.0);
    return m1;
  }

    
  
  
  // operator -
  /** @brief Generic 'catch-all' overload, which enforces a temporary if the expression tree gets too deep. */
  template <typename LHS1, typename RHS1, typename OP1,
            typename LHS2, typename RHS2, typename OP2>
  typename matrix_expression< LHS1, RHS1, OP1>::matrix_type
  operator - (matrix_expression< LHS1, RHS1, OP1> const & proxy1,
              matrix_expression< LHS2, RHS2, OP2> const & proxy2)
  {
    assert(    (viennacl::traits::size1(proxy1) == viennacl::traits::size1(proxy2))
            && (viennacl::traits::size2(proxy1) == viennacl::traits::size2(proxy1))
            && bool("Incompatible matrix sizes!"));
    typename matrix_expression< LHS1, RHS1, OP1>::matrix_type result(proxy1.size());
    result = proxy1;
    result -= proxy2;
    return result;
  }
  
  /** @brief Operator overload for m1 - m2, where m1 and m2 are either dense matrices, matrix ranges, or matrix slices */
  template <typename M1, typename M2>
  typename viennacl::enable_if<    viennacl::is_any_dense_nonstructured_matrix<M1>::value
                                && viennacl::is_any_dense_nonstructured_matrix<M2>::value,
                                matrix_expression< const M1, const M2, op_sub > 
                              >::type
  operator - (const M1 & m1, const M2 & m2) 
  {
    return matrix_expression< const M1, const M2, op_sub > (m1, m2);
  }
  
  /** @brief Operator overload for the addition of a matrix expression m1 - m2 @ beta, where @ is either product or division, and beta is either a CPU or GPU scalar. */
  template <typename M1, typename M2, typename M3, typename OP>
  typename viennacl::enable_if<    viennacl::is_any_dense_nonstructured_matrix<M1>::value
                                && viennacl::is_any_dense_nonstructured_matrix<M2>::value
                                && viennacl::is_any_dense_nonstructured_matrix<M3>::value,
                                matrix_expression< const M1,
                                                    const matrix_expression<const M2, const M3, OP>,
                                                    op_sub >
                              >::type
  operator - (const M1 & m1,
              const matrix_expression< const M2, const M3, OP> & proxy) 
  {
    return matrix_expression< const M1,
                              const matrix_expression<const M2, const M3, OP>,
                              op_sub > (m1, proxy);
  }
  
  /** @brief Operator overload for the addition of a matrix expression m1 @ alpha - m2, where @ is either product or division, and alpha is either a CPU or GPU scalar. */
  template <typename M1, typename M2, typename M3, typename OP>
  typename viennacl::enable_if<    viennacl::is_any_dense_nonstructured_matrix<M1>::value
                                && viennacl::is_any_dense_nonstructured_matrix<M2>::value
                                && viennacl::is_any_dense_nonstructured_matrix<M3>::value,
                                matrix_expression< const matrix_expression<const M1, const M2, OP>,
                                                    const M3,
                                                    op_sub >
                              >::type
  operator - (const matrix_expression< const M1, const M2, OP> & proxy,
              const M3 & m3) 
  {
    return matrix_expression< const matrix_expression<const M1, const M2, OP>,
                              const M3,
                              op_sub > (proxy, m3);
  }
  
  
  /** @brief Operator overload for the addition of a matrix expression m1 @ alpha + m2 @ beta, where @ denotes either product or division, and alpha, beta are either CPU or GPU scalars.
  *
  * @param lhs   Left hand side vector expression
  * @param rhs     Right hand side vector (also -range and -slice is allowed)
  */
  template <typename M1, typename S1, typename OP1,
            typename M2, typename S2, typename OP2>
  typename viennacl::enable_if<    viennacl::is_any_dense_nonstructured_matrix<M1>::value && viennacl::is_any_scalar<S1>::value
                                && viennacl::is_any_dense_nonstructured_matrix<M2>::value && viennacl::is_any_scalar<S2>::value,
                                matrix_expression<const matrix_expression<const M1, const S1, OP1>,
                                                  const matrix_expression<const M2, const S2, OP2>,
                                                  op_sub>
                              >::type
  operator - (matrix_expression<const M1, const S1, OP1> const & lhs,
              matrix_expression<const M2, const S2, OP2> const & rhs)
  {
    return matrix_expression<const matrix_expression<const M1, const S1, OP1>,
                              const matrix_expression<const M2, const S2, OP2>,
                              op_sub>(lhs, rhs);
  }

  
  
  
  // operator -=
  template <typename M1, typename M2>
  typename viennacl::enable_if<    viennacl::is_any_dense_nonstructured_matrix<M1>::value
                                && viennacl::is_any_dense_nonstructured_matrix<M2>::value,
                                M1 & 
                              >::type
  operator -= (M1 & m1, const M2 & other) 
  {
    typedef typename viennacl::result_of::cpu_value_type<typename M1::value_type>::type   ScalarType;
    viennacl::linalg::ambm(m1,
                            m1,    ScalarType(1.0), 1, false, false,
                            other, ScalarType(1.0), 1, false, true);
    return m1;
  }

  
  
  /** @brief Inplace subtraction of a scaled matrix, i.e. m1 -= m2 @ alpha, where @ is either product or division and alpha is either a CPU or a GPU scalar
  */
  template <typename M1, typename M2, typename S2, typename OP>
  typename viennacl::enable_if<    viennacl::is_any_dense_nonstructured_matrix<M1>::value
                                && viennacl::is_any_dense_nonstructured_matrix<M2>::value && viennacl::is_any_scalar<S2>::value,
                                M1 &>::type
  operator -= (M1 & m1, 
                const matrix_expression< const M2, const S2, OP> & proxy)
  {
    typedef typename viennacl::result_of::cpu_value_type<M1>::type   cpu_value_type;
    
    assert(    (viennacl::traits::size1(m1) == viennacl::traits::size1(proxy))
            && (viennacl::traits::size2(m1) == viennacl::traits::size2(proxy))
            && bool("Incompatible matrix sizes!"));

    if (m1.size1() > 0 && m1.size2() > 0)
      viennacl::linalg::ambm(m1, 
                              m1,  cpu_value_type(1.0), 1, false,                                             false,
                              proxy.lhs(), proxy.rhs(), 1, (viennacl::is_division<OP>::value ? true : false), (viennacl::is_flip_sign_scalar<S2>::value ? false : true));
    return m1;
  }
  
  /** @brief Implementation of the operation m1 -= m2 +- m3
  *
  * @param m1     The result matrix where m2 +- m3 is subtracted from
  * @param proxy  An expression template proxy class.
  */
  template <typename M1, typename M2, typename M3, typename OP>
  typename viennacl::enable_if<    viennacl::is_any_dense_nonstructured_matrix<M1>::value
                                && viennacl::is_any_dense_nonstructured_matrix<M2>::value
                                && (viennacl::is_addition<OP>::value || viennacl::is_subtraction<OP>::value),
                                M1 &>::type
  operator -= (M1 & m1, 
                const matrix_expression< const M2, const M3, OP> & proxy)
  {
    typedef typename viennacl::result_of::cpu_value_type<M1>::type   cpu_value_type;
    
    assert(    (viennacl::traits::size1(m1) == viennacl::traits::size1(proxy))
            && (viennacl::traits::size2(m1) == viennacl::traits::size2(proxy))
            && bool("Incompatible matrix sizes!"));

    if (m1.size1() > 0 && m1.size2() > 0)
      viennacl::linalg::ambm_m(m1, 
                                proxy.lhs(), cpu_value_type(1.0), 1, false, true,
                                proxy.rhs(), cpu_value_type(1.0), 1, false, (viennacl::is_subtraction<OP>::value ? false : true) );
    return m1;
  }
  
  /** @brief Implementation of the operation m1 -= m2 +- m3 @ beta, where @ is either product or division, and alpha, beta are either CPU or GPU scalars
  *
  * @param m1     The result matrix where m2 +- m3 @ beta is subtracted from
  * @param proxy  An expression template proxy class.
  */
  template <typename M1,
            typename M2,
            typename M3, typename S3, typename OP3,
            typename OP>
  typename viennacl::enable_if<    viennacl::is_any_dense_nonstructured_matrix<M1>::value
                                && viennacl::is_any_dense_nonstructured_matrix<M2>::value
                                && viennacl::is_any_dense_nonstructured_matrix<M3>::value && viennacl::is_any_scalar<S3>::value && (viennacl::is_product<OP3>::value || viennacl::is_division<OP3>::value)
                                && (viennacl::is_addition<OP>::value || viennacl::is_subtraction<OP>::value),
                                M1 &>::type
  operator -= (M1 & m1, 
                const matrix_expression< const M2,
                                        const matrix_expression<const M3, const S3, OP3>,
                                        OP> & proxy)
  {
    typedef typename viennacl::result_of::cpu_value_type<M1>::type   cpu_value_type;
    
    assert(    (viennacl::traits::size1(m1) == viennacl::traits::size1(proxy))
            && (viennacl::traits::size2(m1) == viennacl::traits::size2(proxy))
            && bool("Incompatible matrix sizes!"));

    if (m1.size1() > 0 && m1.size2() > 0)
    {
      bool flip_sign_3 = (viennacl::is_subtraction<OP>::value ? false : true);
      if (viennacl::is_flip_sign_scalar<S3>::value)
        flip_sign_3 = !flip_sign_3;
      viennacl::linalg::ambm_m(m1, 
                                proxy.lhs(),     cpu_value_type(1.0), 1, false                                             , true,
                                proxy.rhs().lhs(), proxy.rhs().rhs(), 1, (viennacl::is_division<OP3>::value ? true : false), flip_sign_3);
    }
    return m1;
  }

  /** @brief Implementation of the operation m1 -= m2 @ alpha +- m3, where @ is either product or division, and alpha, beta are either CPU or GPU scalars
  *
  * @param m1     The result matrix where m2 @ alpha +- m3 is subtracted from
  * @param proxy  An expression template proxy class.
  */
  template <typename M1,
            typename M2, typename S2, typename OP2,
            typename M3,
            typename OP>
  typename viennacl::enable_if<    viennacl::is_any_dense_nonstructured_matrix<M1>::value
                                && viennacl::is_any_dense_nonstructured_matrix<M2>::value && viennacl::is_any_scalar<S2>::value && (viennacl::is_product<OP2>::value || viennacl::is_division<OP2>::value)
                                && viennacl::is_any_dense_nonstructured_matrix<M3>::value
                                && (viennacl::is_addition<OP>::value || viennacl::is_subtraction<OP>::value),
                                M1 &>::type
  operator -= (M1 & m1, 
                const matrix_expression< const matrix_expression<const M2, const S2, OP2>,
                                        const M3,
                                        OP> & proxy)
  {
    typedef typename viennacl::result_of::cpu_value_type<M1>::type   cpu_value_type;
    
    assert(    (viennacl::traits::size1(m1) == viennacl::traits::size1(proxy))
            && (viennacl::traits::size2(m1) == viennacl::traits::size2(proxy))
            && bool("Incompatible matrix sizes!"));

    if (m1.size1() > 0 && m1.size2() > 0)
      viennacl::linalg::ambm_m(m1, 
                                proxy.lhs().lhs(), proxy.lhs().rhs(), 1, (viennacl::is_division<OP2>::value ? true : false), (viennacl::is_flip_sign_scalar<S2>::value ? false : true),
                                proxy.rhs(),     cpu_value_type(1.0), 1, false                                             , (viennacl::is_subtraction<OP>::value ? false : true) );
    return m1;
  }
  
  /** @brief Implementation of the operation m1 -= m2 @ alpha +- v3 @ beta, where @ is either product or division, and alpha, beta are either CPU or GPU scalars
  *
  * @param m1     The result matrix where m2 @ alpha +- m3 @ beta is subtracted from
  * @param proxy  An expression template proxy class.
  */
  template <typename M1,
            typename M2, typename S2, typename OP2,
            typename M3, typename S3, typename OP3,
            typename OP>
  typename viennacl::enable_if<    viennacl::is_any_dense_nonstructured_matrix<M1>::value
                                && viennacl::is_any_dense_nonstructured_matrix<M2>::value && viennacl::is_any_scalar<S2>::value && (viennacl::is_product<OP2>::value || viennacl::is_division<OP2>::value)
                                && viennacl::is_any_dense_nonstructured_matrix<M3>::value && viennacl::is_any_scalar<S3>::value && (viennacl::is_product<OP3>::value || viennacl::is_division<OP3>::value)
                                && (viennacl::is_addition<OP>::value || viennacl::is_subtraction<OP>::value),
                                M1 &>::type
  operator -= (M1 & m1, 
                const matrix_expression< const matrix_expression<const M2, const S2, OP2>,
                                        const matrix_expression<const M3, const S3, OP3>,
                                        OP> & proxy)
  {
    assert(    (viennacl::traits::size1(m1) == viennacl::traits::size1(proxy))
            && (viennacl::traits::size2(m1) == viennacl::traits::size2(proxy))
            && bool("Incompatible matrix sizes!"));

    if (m1.size1() > 0 && m1.size2() > 0)
    {
      bool flip_sign_3 = (viennacl::is_subtraction<OP>::value ? false : true);
      if (viennacl::is_flip_sign_scalar<S3>::value)
        flip_sign_3 = !flip_sign_3;
      viennacl::linalg::ambm_m(m1, 
                                proxy.lhs().lhs(), proxy.lhs().rhs(), 1, (viennacl::is_division<OP2>::value ? true : false), (viennacl::is_flip_sign_scalar<S2>::value ? false : true),
                                proxy.rhs().lhs(), proxy.rhs().rhs(), 1, (viennacl::is_division<OP3>::value ? true : false), flip_sign_3);
    }
    return m1;
  }
  

  
  template <typename M1, typename V1, typename V2>
  typename viennacl::enable_if<    viennacl::is_any_dense_nonstructured_matrix<M1>::value
                                && viennacl::is_any_dense_nonstructured_vector<V1>::value
                                && viennacl::is_any_dense_nonstructured_vector<V2>::value,
                                M1 & 
                              >::type
  operator -= (M1 & m1,
                const matrix_expression< const V1, const V2, op_prod > & proxy) 
  {
    typedef typename viennacl::result_of::cpu_value_type<typename M1::value_type>::type   ScalarType;
    viennacl::linalg::scaled_rank_1_update(m1,
                                            ScalarType(1.0), 1, false, true,
                                            proxy.lhs(),
                                            proxy.rhs());
    return m1;
  }

  template <typename M1, typename V1, typename V2, typename S1>
  typename viennacl::enable_if<    viennacl::is_any_dense_nonstructured_matrix<M1>::value
                                && viennacl::is_any_dense_nonstructured_vector<V1>::value
                                && viennacl::is_any_dense_nonstructured_vector<V2>::value
                                && viennacl::is_any_scalar<S1>::value,
                                M1 & 
                              >::type
  operator -= (M1 & m1,
                const matrix_expression< const matrix_expression< const V1, const V2, op_prod >,
                                        const S1,
                                        op_prod > & proxy) 
  {
    viennacl::linalg::scaled_rank_1_update(m1,
                                            proxy.rhs(), 1, false, (viennacl::is_flip_sign_scalar<S1>::value ? false : true),
                                            proxy.lhs(),
                                            proxy.rhs());
    return m1;
  }

  //C -= A * B and related (with trans())
  template <typename M1, typename M2, typename M3>
  typename viennacl::enable_if<    viennacl::is_any_dense_nonstructured_matrix<M1>::value
                                && (viennacl::is_any_dense_nonstructured_matrix<M2>::value || viennacl::is_any_dense_nonstructured_transposed_matrix<M2>::value)
                                && (viennacl::is_any_dense_nonstructured_matrix<M3>::value || viennacl::is_any_dense_nonstructured_transposed_matrix<M3>::value),
                                M1 & >::type
  operator -= (M1 & m1, const matrix_expression< const M2, const M3, op_prod > & proxy) 
  {
    viennacl::linalg::prod_impl(proxy.lhs(), proxy.rhs(), m1, -1.0, 1.0);
    return m1;
  }
  
  
  

  
  // operator *
  /** @brief Operator overload for the expression alpha * m1, where alpha is a host scalar (float or double) and m1 is a ViennaCL matrix.
  *
  * @param value   The host scalar (float or double)
  * @param m1      A ViennaCL matrix
  */
  template <typename S1, typename M1>
  typename viennacl::enable_if<    viennacl::is_any_scalar<S1>::value
                                && viennacl::is_any_dense_nonstructured_matrix<M1>::value,
                                matrix_expression< const M1, const S1, op_prod>
                              >::type 
  operator * (S1 const & value, M1 const & m1)
  {
    return matrix_expression< const M1, const S1, op_prod>(m1, value);
  }


  /** @brief Operator overload for the multiplication of a matrix expression with a scalar from the right, e.g. (beta * m1) * alpha. Here, beta * m1 is wrapped into a matrix_expression and then multiplied with alpha from the right.
  *
  * @param proxy   Left hand side matrix expression
  * @param val     Right hand side scalar
  */
  template <typename LHS, typename RHS, typename OP, typename S1>
  typename viennacl::enable_if< viennacl::is_any_scalar<S1>::value,
                                typename matrix_expression< LHS, RHS, OP>::matrix_type >::type
  operator * (matrix_expression< LHS, RHS, OP> const & proxy,
              S1 const & val)
  {
    typename matrix_expression< LHS, RHS, OP>::matrix_type result(proxy.size1(), proxy.size2());
    result = proxy;
    result *= val;
    return result;
  }


  /** @brief Operator overload for the multiplication of a matrix expression with a ViennaCL scalar from the left, e.g. alpha * (beta * m1). Here, beta * m1 is wrapped into a matrix_expression and then multiplied with alpha from the left.
  *
  * @param val     Right hand side scalar
  * @param proxy   Left hand side matrix expression
  */
  template <typename S1, typename LHS, typename RHS, typename OP>
  typename viennacl::enable_if< viennacl::is_any_scalar<S1>::value,
                                typename matrix_expression< LHS, RHS, OP>::matrix_type >::type
  operator * (S1 const & val,
              matrix_expression< LHS, RHS, OP> const & proxy)
  {
    typename matrix_expression< LHS, RHS, OP>::matrix_type result(proxy.size());
    result = proxy;
    result *= val;
    return result;
  }
  
  /** @brief Scales the matrix by a GPU scalar 'alpha' and returns an expression template
  */
  template <typename M1, typename S1>
  typename viennacl::enable_if<    viennacl::is_any_dense_nonstructured_matrix<M1>::value
                                && viennacl::is_any_scalar<S1>::value,
                                matrix_expression< const M1, const S1, op_prod> >::type
  operator * (M1 const & m1, S1 const & s1)
  {
    return matrix_expression< const M1, const S1, op_prod>(m1, s1);
  }
  
  
  // operator *=

  /** @brief Scales a matrix by a GPU scalar value
  */
  template <typename M1, typename S1>
  typename viennacl::enable_if<    viennacl::is_any_dense_nonstructured_matrix<M1>::value
                                && viennacl::is_scalar<S1>::value,
                                M1 & 
                              >::type
  operator *= (M1 & m1, S1 const & gpu_val)
  {
    //viennacl::linalg::inplace_mult(*this, gpu_val);
    viennacl::linalg::am(m1,
                          m1, gpu_val, 1, false, (viennacl::is_flip_sign_scalar<S1>::value ? true : false));
    return m1;
  }

  
  // operator /
  
  
  /** @brief Operator overload for the division of a matrix expression by a scalar from the right, e.g. (beta * m1) / alpha. Here, beta * m1 is wrapped into a matrix_expression and then divided by alpha.
  *
  * @param proxy   Left hand side matrix expression
  * @param val     Right hand side scalar
  */
  template <typename S1, typename LHS, typename RHS, typename OP>
  typename viennacl::enable_if< viennacl::is_any_scalar<S1>::value,
                                typename matrix_expression< LHS, RHS, OP>::matrix_type >::type
  operator / (matrix_expression< LHS, RHS, OP> const & proxy,
              S1 const & val)
  {
    typename matrix_expression< LHS, RHS, OP>::matrix_type result(proxy.size1(), proxy.size2());
    result = proxy;
    result /= val;
    return result;
  }


  /** @brief Returns an expression template for scaling the matrix by a GPU scalar 'alpha'
  */
  template <typename M1, typename S1>
  typename viennacl::enable_if<    viennacl::is_any_dense_nonstructured_matrix<M1>::value
                                && viennacl::is_any_scalar<S1>::value,
                                matrix_expression< const M1, const S1, op_div> >::type
  operator / (M1 const & m1, S1 const & s1)
  {
    return matrix_expression< const M1, const S1, op_div>(m1, s1);
  }
  
  
  // operator /=
  
  /** @brief Scales a matrix by a GPU scalar value
  */
  template <typename M1, typename S1>
  typename viennacl::enable_if<    viennacl::is_any_dense_nonstructured_matrix<M1>::value
                                && viennacl::is_scalar<S1>::value,
                                M1 & 
                              >::type
  operator /= (M1 & m1, S1 const & gpu_val)
  {
    //viennacl::linalg::inplace_divide(*this, gpu_val);
    viennacl::linalg::am(m1,
                          m1, gpu_val, 1, true, (viennacl::is_flip_sign_scalar<S1>::value ? true : false));
    return m1;
  }





  // outer_prod(v1, v2) * val;
  template<typename V1, typename V2, typename S1>
  typename viennacl::enable_if<    viennacl::is_any_dense_nonstructured_vector<V1>::value
                                && viennacl::is_any_dense_nonstructured_vector<V2>::value
                                && viennacl::is_scalar<S1>::value,
                                viennacl::matrix_expression< const viennacl::matrix_expression< const V1, const V2, op_prod>,
                                                              const S1,
                                                              op_prod>                                  
                              >::type
  operator*(const viennacl::matrix_expression< const V1, const V2, op_prod> & proxy,
            const S1 & val)
  {
    return viennacl::matrix_expression< const viennacl::matrix_expression< const V1, const V2, op_prod>,
                                        const S1,
                                        op_prod>(proxy, val);
  }

  template<typename V1, typename V2, typename S1>
  typename viennacl::enable_if<    viennacl::is_any_dense_nonstructured_vector<V1>::value
                                && viennacl::is_any_dense_nonstructured_vector<V2>::value
                                && viennacl::is_cpu_scalar<S1>::value,
                                viennacl::matrix_expression< const viennacl::matrix_expression< const V1, const V2, op_prod>,
                                                              const typename viennacl::result_of::cpu_value_type<V1>::type,
                                                              op_prod>                                  
                              >::type
  operator*(const viennacl::matrix_expression< const V1, const V2, op_prod> & proxy,
            const S1 & val)
  {
    typedef typename viennacl::result_of::cpu_value_type<V1>::type cpu_value_type;
    
    return viennacl::matrix_expression< const viennacl::matrix_expression< const V1, const V2, op_prod>,
                                        const cpu_value_type,
                                        op_prod>(proxy, cpu_value_type(val));
  }
  
  // val * outer_prod(v1, v2);
  template<typename V1, typename V2, typename S1>
  typename viennacl::enable_if<    viennacl::is_any_dense_nonstructured_vector<V1>::value
                                && viennacl::is_any_dense_nonstructured_vector<V2>::value
                                && viennacl::is_scalar<S1>::value,
                                viennacl::matrix_expression< const viennacl::matrix_expression< const V1, const V2, op_prod>,
                                                              const S1,
                                                              op_prod>                                  
                              >::type
  operator*(const S1 & val,
            const viennacl::matrix_expression< const V1, const V2, op_prod> & proxy)
  {
    return viennacl::matrix_expression< const viennacl::matrix_expression< const V1, const V2, op_prod>,
                                        const S1,
                                        op_prod>(proxy, val);
  }
  
  template<typename V1, typename V2, typename S1>
  typename viennacl::enable_if<    viennacl::is_any_dense_nonstructured_vector<V1>::value
                                && viennacl::is_any_dense_nonstructured_vector<V2>::value
                                && viennacl::is_cpu_scalar<S1>::value,
                                viennacl::matrix_expression< const viennacl::matrix_expression< const V1, const V2, op_prod>,
                                                              const typename viennacl::result_of::cpu_value_type<V1>::type,
                                                              op_prod>                                  
                              >::type
  operator*(const S1 & val,
            const viennacl::matrix_expression< const V1, const V2, op_prod> & proxy)
  {
    typedef typename viennacl::result_of::cpu_value_type<V1>::type cpu_value_type;
    
    return viennacl::matrix_expression< const viennacl::matrix_expression< const V1, const V2, op_prod>,
                                        const cpu_value_type,
                                        op_prod>(proxy, cpu_value_type(val));
  }
  
  

} //namespace viennacl

#endif
