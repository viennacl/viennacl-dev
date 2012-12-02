#ifndef VIENNACL_MATRIX_PROXY_HPP_
#define VIENNACL_MATRIX_PROXY_HPP_

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

/** @file matrix_proxy.hpp
    @brief Proxy classes for matrices.
*/

#include "viennacl/forwards.h"
#include "viennacl/range.hpp"
#include "viennacl/matrix.hpp"
#include "viennacl/linalg/matrix_operations.hpp"

namespace viennacl
{

  template <typename MatrixType>
  class matrix_range
  {
      typedef matrix_range<MatrixType>            self_type;
    
    public:
      typedef typename MatrixType::orientation_category       orientation_category;
      
      typedef typename MatrixType::value_type     value_type;
      typedef typename viennacl::result_of::cpu_value_type<value_type>::type    cpu_value_type;
      typedef range::size_type                    size_type;
      typedef range::difference_type              difference_type;
      typedef value_type                          reference;
      typedef const value_type &                  const_reference;
      
      matrix_range(MatrixType & A, 
                   range const & row_range,
                   range const & col_range) : A_(&A), row_range_(row_range), col_range_(col_range) {}
                   
      size_type start1() const { return row_range_.start(); }
      size_type size1() const { return row_range_.size(); }

      size_type start2() const { return col_range_.start(); }
      size_type size2() const { return col_range_.size(); }
      
      ////////// operator= //////////////////////////
      
      self_type & operator = (const self_type & other) 
      {
        viennacl::linalg::am(*this,
                             other, cpu_value_type(1.0), 1, false, false);
        return *this;
      }
      
      /** @brief Generic 'catch-all' overload for all operations which are not covered by the other specializations below */
      template <typename LHS, typename RHS, typename OP>
      self_type & operator=(const matrix_expression< LHS,
                                                     RHS,
                                                     OP > & proxy) 
      {
        MatrixType temp = proxy;
        *this = temp;
        return *this;
      }      
      
      template <typename M1>
      typename viennacl::enable_if<    viennacl::is_any_dense_nonstructured_matrix<M1>::value,
                                    self_type &>::type
      operator = (const M1 & other) 
      {
        viennacl::linalg::am(*this,
                             other, cpu_value_type(1.0), 1, false, false);
        return *this;
      }

      
      template <typename M1, typename M2>
      typename viennacl::enable_if<    viennacl::is_any_dense_nonstructured_matrix<M1>::value
                                    && viennacl::is_any_dense_nonstructured_matrix<M2>::value, 
                                    self_type &>::type
      operator = (const matrix_expression< const M1, const M2, op_prod > & proxy) 
      {
        viennacl::linalg::prod_impl(proxy.lhs(), proxy.rhs(), *this, 1.0, 0.0);
        return *this;
      }
      
      
      template <typename M1, typename M2, typename OP>
      typename viennacl::enable_if<    viennacl::is_any_dense_nonstructured_matrix<M1>::value
                                    && viennacl::is_any_dense_nonstructured_matrix<M2>::value
                                    && (viennacl::is_addition<OP>::value || viennacl::is_subtraction<OP>::value),
                                    self_type &>::type
      operator = (const matrix_expression< const M1, const M2, OP > & proxy) 
      {
        viennacl::linalg::ambm(*this,
                               proxy.lhs(), cpu_value_type(1.0), 1, false, false,
                               proxy.rhs(), cpu_value_type(1.0), 1, false, viennacl::is_subtraction<OP>::value ? true : false);
        return *this;
      }

      
      /** @brief Assignment of a scaled matrix (or -range or -slice), i.e. m1 = m2 @ alpha, where @ is either product or division and alpha is either a CPU or a GPU scalar
      */
      template <typename M1, typename S1, typename OP>
      typename viennacl::enable_if< viennacl::is_any_dense_nonstructured_matrix<M1>::value && viennacl::is_any_scalar<S1>::value,
                                    self_type &>::type
      operator = (const matrix_expression< const M1, const S1, OP> & proxy)
      {
        viennacl::linalg::am(*this, 
                             proxy.lhs(), proxy.rhs(), 1, (viennacl::is_division<OP>::value ? true : false), (viennacl::is_flip_sign_scalar<S1>::value ? true : false) );
        return *this;
      }

      /** @brief Assigns the supplied identity matrix to the matrix. */
      self_type & operator = (identity_matrix<cpu_value_type> const & m)
      {
        assert( (m.size1() == size1()) && bool("Size mismatch!") );
        assert( (m.size2() == size2()) && bool("Size mismatch!") );

        viennacl::linalg::matrix_assign(*this, m(1,0));          //set everything to zero
        viennacl::linalg::matrix_diagonal_assign(*this, m(0,0)); //set unit diagonal
        return *this;
      }
      
      /** @brief Assigns the supplied zero matrix to the matrix. */
      self_type & operator = (zero_matrix<cpu_value_type> const & m)
      {
        assert( (m.size1() == size1()) && bool("Size mismatch!") );
        assert( (m.size2() == size2()) && bool("Size mismatch!") );

        viennacl::linalg::matrix_assign(*this, m(0,0));
        return *this;
      }

      /** @brief Assigns the supplied scalar vector to the matrix. */
      self_type & operator = (scalar_matrix<cpu_value_type> const & m)
      {
        assert( (m.size1() == size1()) && bool("Size mismatch!") );
        assert( (m.size2() == size2()) && bool("Size mismatch!") );

        viennacl::linalg::matrix_assign(*this, m(0,0));
        return *this;
      }


      ////////// operator*= //////////////////////////

      self_type & operator *= (cpu_value_type val) // Enabling implicit conversions
      {
        viennacl::linalg::am(*this,
                             *this, val, 1, false, false);
        return *this;
      }
      
      ////////// operator/= //////////////////////////

      self_type & operator /= (cpu_value_type val) // Enabling implicit conversions
      {
        viennacl::linalg::am(*this,
                             *this, val, 1, true, false);
        return *this;
      }


      

      //const_reference operator()(size_type i, size_type j) const { return A_(start1() + i, start2() + i); }
      //reference operator()(size_type i, size_type j) { return A_(start1() + i, start2() + i); }

      MatrixType & get() { return *A_; }
      const MatrixType & get() const { return *A_; }

    private:
      MatrixType * A_;
      range row_range_;
      range col_range_;
  };


  // implement copy-CTOR for matrix:
  template <typename SCALARTYPE, typename F, unsigned int ALIGNMENT>
  viennacl::matrix<SCALARTYPE, F, ALIGNMENT>::matrix(matrix_range<viennacl::matrix<SCALARTYPE, F, ALIGNMENT> > const & proxy) : rows_(proxy.size1()), columns_(proxy.size2())
  {
    this->elements_.switch_active_handle_id(viennacl::traits::handle(proxy).get_active_handle_id());
    viennacl::backend::memory_create(elements_, sizeof(SCALARTYPE)*internal_size());
    *this = proxy;
  }

  template <typename SCALARTYPE, typename F, unsigned int ALIGNMENT>
  viennacl::matrix<SCALARTYPE, F, ALIGNMENT>::matrix(matrix_range<const viennacl::matrix<SCALARTYPE, F, ALIGNMENT> > const & proxy) : rows_(proxy.size1()), columns_(proxy.size2())
  {
    this->elements_.switch_active_handle_id(viennacl::traits::handle(proxy).get_active_handle_id());
    viennacl::backend::memory_create(elements_, sizeof(SCALARTYPE)*internal_size());
    *this = proxy;
  }
  
  
  
  /** @brief Returns an expression template class representing a transposed matrix */
  template <typename MatrixType>
  matrix_expression< const matrix_range<MatrixType>,
                     const matrix_range<MatrixType>,
                     op_trans> trans(const matrix_range<MatrixType> & mat)
  {
    return matrix_expression< const matrix_range<MatrixType>,
                              const matrix_range<MatrixType>,
                              op_trans>(mat, mat);
  }
  
  
  
  
  /////////////////////////////////////////////////////////////
  ///////////////////////// CPU to GPU ////////////////////////
  /////////////////////////////////////////////////////////////
  
  //row_major:
  template <typename CPU_MATRIX, typename SCALARTYPE>
  void copy(const CPU_MATRIX & cpu_matrix,
            matrix_range<matrix<SCALARTYPE, row_major, 1> > & gpu_matrix_range )
  {
    assert( (cpu_matrix.size1() == gpu_matrix_range.size1())
           && (cpu_matrix.size2() == gpu_matrix_range.size2())
           && bool("Matrix size mismatch!"));
    
    if ( gpu_matrix_range.start2() != 0 ||  gpu_matrix_range.size2() != gpu_matrix_range.get().size2())
    {
      std::vector<SCALARTYPE> entries(gpu_matrix_range.size2());
      
      //copy each stride separately:
      for (std::size_t i=0; i < gpu_matrix_range.size1(); ++i)
      {
        for (std::size_t j=0; j < gpu_matrix_range.size2(); ++j)
          entries[j] = cpu_matrix(i,j);
        
        std::size_t start_offset = (gpu_matrix_range.start1() + i) * gpu_matrix_range.get().internal_size2() + gpu_matrix_range.start2();
        std::size_t num_entries = gpu_matrix_range.size2();
        viennacl::backend::memory_write(gpu_matrix_range.get().handle(), sizeof(SCALARTYPE)*start_offset, sizeof(SCALARTYPE)*num_entries, &(entries[0]));
      //std::cout << "Strided copy worked!" << std::endl;
      }
    }
    else
    {
      //full block can be copied: 
      std::vector<SCALARTYPE> entries(gpu_matrix_range.size1()*gpu_matrix_range.size2());
      
      //copy each stride separately:
      for (std::size_t i=0; i < gpu_matrix_range.size1(); ++i)
        for (std::size_t j=0; j < gpu_matrix_range.size2(); ++j)
          entries[i*gpu_matrix_range.get().internal_size2() + j] = cpu_matrix(i,j);
      
      std::size_t start_offset = gpu_matrix_range.start1() * gpu_matrix_range.get().internal_size2();
      std::size_t num_entries = gpu_matrix_range.size1() * gpu_matrix_range.size2();
      viennacl::backend::memory_write(gpu_matrix_range.get().handle(), sizeof(SCALARTYPE)*start_offset, sizeof(SCALARTYPE)*num_entries, &(entries[0]));
      //std::cout << "Block copy worked!" << std::endl;
    }
  }
  
  //column_major:
  template <typename CPU_MATRIX, typename SCALARTYPE>
  void copy(const CPU_MATRIX & cpu_matrix,
            matrix_range<matrix<SCALARTYPE, column_major, 1> > & gpu_matrix_range )
  {
    assert( (cpu_matrix.size1() == gpu_matrix_range.size1())
           && (cpu_matrix.size2() == gpu_matrix_range.size2())
           && bool("Matrix size mismatch!"));
    
     if ( gpu_matrix_range.start1() != 0 ||  gpu_matrix_range.size1() != gpu_matrix_range.get().size1())
     {
       std::vector<SCALARTYPE> entries(gpu_matrix_range.size1());
       
       //copy each stride separately:
       for (std::size_t j=0; j < gpu_matrix_range.size2(); ++j)
       {
         for (std::size_t i=0; i < gpu_matrix_range.size1(); ++i)
           entries[i] = cpu_matrix(i,j);
         
         std::size_t start_offset = (gpu_matrix_range.start2() + j) * gpu_matrix_range.get().internal_size1() + gpu_matrix_range.start1();
         std::size_t num_entries = gpu_matrix_range.size1();
         viennacl::backend::memory_write(gpu_matrix_range.get().handle(), sizeof(SCALARTYPE)*start_offset, sizeof(SCALARTYPE)*num_entries, &(entries[0]));
        //std::cout << "Strided copy worked!" << std::endl;
       }
     }
     else
     {
       //full block can be copied: 
       std::vector<SCALARTYPE> entries(gpu_matrix_range.size1()*gpu_matrix_range.size2());
       
       //copy each stride separately:
       for (std::size_t i=0; i < gpu_matrix_range.size1(); ++i)
         for (std::size_t j=0; j < gpu_matrix_range.size2(); ++j)
           entries[i + j*gpu_matrix_range.get().internal_size1()] = cpu_matrix(i,j);
       
       std::size_t start_offset = gpu_matrix_range.start2() * gpu_matrix_range.get().internal_size1();
       std::size_t num_entries = gpu_matrix_range.size1() * gpu_matrix_range.size2();
       viennacl::backend::memory_write(gpu_matrix_range.get().handle(), sizeof(SCALARTYPE)*start_offset, sizeof(SCALARTYPE)*num_entries, &(entries[0]));
       //std::cout << "Block copy worked!" << std::endl;
     }
    
  }


  /////////////////////////////////////////////////////////////
  ///////////////////////// GPU to CPU ////////////////////////
  /////////////////////////////////////////////////////////////
  
  
  //row_major:
  template <typename CPU_MATRIX, typename SCALARTYPE>
  void copy(matrix_range<matrix<SCALARTYPE, row_major, 1> > const & gpu_matrix_range,
            CPU_MATRIX & cpu_matrix)
  {
    assert( (cpu_matrix.size1() == gpu_matrix_range.size1())
           && (cpu_matrix.size2() == gpu_matrix_range.size2())
           && bool("Matrix size mismatch!"));
    
     if ( gpu_matrix_range.start2() != 0 ||  gpu_matrix_range.size2() !=  gpu_matrix_range.get().size2())
     {
       std::vector<SCALARTYPE> entries(gpu_matrix_range.size2());
       
       //copy each stride separately:
       for (std::size_t i=0; i < gpu_matrix_range.size1(); ++i)
       {
         std::size_t start_offset = (gpu_matrix_range.start1() + i) * gpu_matrix_range.get().internal_size2() + gpu_matrix_range.start2();
         std::size_t num_entries = gpu_matrix_range.size2();
         viennacl::backend::memory_read(gpu_matrix_range.get().handle(), sizeof(SCALARTYPE)*start_offset, sizeof(SCALARTYPE)*num_entries, &(entries[0]));
        //std::cout << "Strided copy worked!" << std::endl;
        
        for (std::size_t j=0; j < gpu_matrix_range.size2(); ++j)
          cpu_matrix(i,j) = entries[j];
       }
     }
     else
     {
       //full block can be copied: 
       std::vector<SCALARTYPE> entries(gpu_matrix_range.size1()*gpu_matrix_range.size2());
       
       std::size_t start_offset = gpu_matrix_range.start1() * gpu_matrix_range.get().internal_size2();
       std::size_t num_entries = gpu_matrix_range.size1() * gpu_matrix_range.size2();
         viennacl::backend::memory_read(gpu_matrix_range.get().handle(), sizeof(SCALARTYPE)*start_offset, sizeof(SCALARTYPE)*num_entries, &(entries[0]));
       //std::cout << "Block copy worked!" << std::endl;

       for (std::size_t i=0; i < gpu_matrix_range.size1(); ++i)
         for (std::size_t j=0; j < gpu_matrix_range.size2(); ++j)
           cpu_matrix(i,j) = entries[i*gpu_matrix_range.get().internal_size2() + j];
    }
    
  }
  
  
  //column_major:
  template <typename CPU_MATRIX, typename SCALARTYPE>
  void copy(matrix_range<matrix<SCALARTYPE, column_major, 1> > const & gpu_matrix_range,
            CPU_MATRIX & cpu_matrix)
  {
    assert( (cpu_matrix.size1() == gpu_matrix_range.size1())
           && (cpu_matrix.size2() == gpu_matrix_range.size2())
           && bool("Matrix size mismatch!"));
    
     if ( gpu_matrix_range.start1() != 0 ||  gpu_matrix_range.size1() !=  gpu_matrix_range.get().size1())
     {
       std::vector<SCALARTYPE> entries(gpu_matrix_range.size1());
       
       //copy each stride separately:
       for (std::size_t j=0; j < gpu_matrix_range.size2(); ++j)
       {
         std::size_t start_offset = (gpu_matrix_range.start2() + j) * gpu_matrix_range.get().internal_size1() + gpu_matrix_range.start1();
         std::size_t num_entries = gpu_matrix_range.size1();
         viennacl::backend::memory_read(gpu_matrix_range.get().handle(), sizeof(SCALARTYPE)*start_offset, sizeof(SCALARTYPE)*num_entries, &(entries[0]));
        //std::cout << "Strided copy worked!" << std::endl;
        
        for (std::size_t i=0; i < gpu_matrix_range.size1(); ++i)
          cpu_matrix(i,j) = entries[i];
       }
     }
     else
     {
       //full block can be copied: 
       std::vector<SCALARTYPE> entries(gpu_matrix_range.size1()*gpu_matrix_range.size2());
       
       //copy each stride separately:
       std::size_t start_offset = gpu_matrix_range.start2() * gpu_matrix_range.get().internal_size1();
       std::size_t num_entries = gpu_matrix_range.size1() * gpu_matrix_range.size2();
       viennacl::backend::memory_read(gpu_matrix_range.get().handle(), sizeof(SCALARTYPE)*start_offset, sizeof(SCALARTYPE)*num_entries, &(entries[0]));
       //std::cout << "Block copy worked!" << std::endl;
       
       for (std::size_t i=0; i < gpu_matrix_range.size1(); ++i)
         for (std::size_t j=0; j < gpu_matrix_range.size2(); ++j)
           cpu_matrix(i,j) = entries[i + j*gpu_matrix_range.get().internal_size1()];
     }
    
  }


  template<typename MatrixType>
  std::ostream & operator<<(std::ostream & s, matrix_range<MatrixType> const & proxy)
  {
    MatrixType temp = proxy;
    s << temp;
    return s;
  }

  template<typename MatrixType>
  std::ostream & operator<<(std::ostream & s, matrix_range<const MatrixType> const & proxy)
  {
    MatrixType temp = proxy;
    s << temp;
    return s;
  }


  //
  // Convenience function
  //
  template <typename MatrixType>
  matrix_range<MatrixType> project(MatrixType & A, viennacl::range const & r1, viennacl::range const & r2)
  {
    assert(r1.size() <= A.size1() && r2.size() <= A.size2() && bool("Size of range invalid!"));
    
    return matrix_range<MatrixType>(A, r1, r2);
  }


  template <typename MatrixType>
  matrix_range<MatrixType> project(matrix_range<MatrixType> & A, viennacl::range const & r1, viennacl::range const & r2)
  {
    assert(r1.size() <= A.size1() && r2.size() <= A.size2() && bool("Size of range invalid!"));
    
    return matrix_range<MatrixType>(A.get(),
                                    viennacl::range(A.start1() + r1.start(), A.start1() + r1.start() + r1.size()),
                                    viennacl::range(A.start2() + r2.start(), A.start2() + r2.start() + r2.size())
                                   );
  }




//
//
//
/////////////////////////////// Slice /////////////////////////////////////////////
//
//
//









  template <typename MatrixType>
  class matrix_slice
  {
      typedef matrix_slice<MatrixType>            self_type;
    
    public:
      typedef typename MatrixType::orientation_category       orientation_category;
      
      typedef typename MatrixType::value_type     value_type;
      typedef typename viennacl::result_of::cpu_value_type<value_type>::type    cpu_value_type;
      typedef slice::size_type                    size_type;
      typedef slice::difference_type              difference_type;
      typedef value_type                          reference;
      typedef const value_type &                  const_reference;
      
      matrix_slice(MatrixType & A, 
                   slice const & row_slice,
                   slice const & col_slice) : A_(&A), row_slice_(row_slice), col_slice_(col_slice) {}
                   
      size_type start1() const { return row_slice_.start(); }
      size_type stride1() const { return row_slice_.stride(); }
      size_type size1() const { return row_slice_.size(); }

      size_type start2() const { return col_slice_.start(); }
      size_type stride2() const { return col_slice_.stride(); }
      size_type size2() const { return col_slice_.size(); }
      
      ////////// operator= //////////////////////////

      self_type & operator = (const self_type & other) 
      {
        viennacl::linalg::am(*this,
                             other, cpu_value_type(1.0), 1, false, false);
        return *this;
      }

      /** @brief Generic 'catch-all' overload for all operations which are not covered by the other specializations below */
      template <typename LHS, typename RHS, typename OP>
      self_type & operator=(const matrix_expression< LHS,
                                                     RHS,
                                                     OP > & proxy) 
      {
        MatrixType temp = proxy;
        *this = temp;
        return *this;
      }      
      
      template <typename M1>
      typename viennacl::enable_if<    viennacl::is_any_dense_nonstructured_matrix<M1>::value,
                                    self_type &>::type
      operator = (const M1 & other) 
      {
        viennacl::linalg::am(*this,
                             other, cpu_value_type(1.0), 1, false, false);
        return *this;
      }

      
      template <typename M1, typename M2>
      typename viennacl::enable_if<    viennacl::is_any_dense_nonstructured_matrix<M1>::value
                                    && viennacl::is_any_dense_nonstructured_matrix<M2>::value, 
                                    self_type &>::type
      operator = (const matrix_expression< const M1, const M2, op_prod > & proxy) 
      {
        viennacl::linalg::prod_impl(proxy.lhs(), proxy.rhs(), *this, 1.0, 0.0);
        return *this;
      }
      
      
      template <typename M1, typename M2, typename OP>
      typename viennacl::enable_if<    viennacl::is_any_dense_nonstructured_matrix<M1>::value
                                    && viennacl::is_any_dense_nonstructured_matrix<M2>::value
                                    && (viennacl::is_addition<OP>::value || viennacl::is_subtraction<OP>::value),
                                    self_type &>::type
      operator = (const matrix_expression< const M1, const M2, OP > & proxy) 
      {
        viennacl::linalg::ambm(*this,
                               proxy.lhs(), cpu_value_type(1.0), 1, false, false,
                               proxy.rhs(), cpu_value_type(1.0), 1, false, viennacl::is_subtraction<OP>::value ? true : false);
        return *this;
      }

      
      /** @brief Assignment of a scaled matrix (or -range or -slice), i.e. m1 = m2 @ alpha, where @ is either product or division and alpha is either a CPU or a GPU scalar
      */
      template <typename M1, typename S1, typename OP>
      typename viennacl::enable_if< viennacl::is_any_dense_nonstructured_matrix<M1>::value && viennacl::is_any_scalar<S1>::value,
                                    self_type &>::type
      operator = (const matrix_expression< const M1, const S1, OP> & proxy)
      {
        viennacl::linalg::am(*this, 
                             proxy.lhs(), proxy.rhs(), 1, (viennacl::is_division<OP>::value ? true : false), (viennacl::is_flip_sign_scalar<S1>::value ? true : false) );
        return *this;
      }

      /** @brief Assigns the supplied identity matrix to the matrix. */
      self_type & operator = (identity_matrix<cpu_value_type> const & m)
      {
        assert( (m.size1() == size1()) && bool("Size mismatch!") );
        assert( (m.size2() == size2()) && bool("Size mismatch!") );

        viennacl::linalg::matrix_assign(*this, m(1,0));          //set everything to zero
        viennacl::linalg::matrix_diagonal_assign(*this, m(0,0)); //set unit diagonal
        return *this;
      }
      
      /** @brief Assigns the supplied zero matrix to the matrix. */
      self_type & operator = (zero_matrix<cpu_value_type> const & m)
      {
        assert( (m.size1() == size1()) && bool("Size mismatch!") );
        assert( (m.size2() == size2()) && bool("Size mismatch!") );

        viennacl::linalg::matrix_assign(*this, m(0,0));
        return *this;
      }

      /** @brief Assigns the supplied scalar vector to the matrix. */
      self_type & operator = (scalar_matrix<cpu_value_type> const & m)
      {
        assert( (m.size1() == size1()) && bool("Size mismatch!") );
        assert( (m.size2() == size2()) && bool("Size mismatch!") );

        viennacl::linalg::matrix_assign(*this, m(0,0));
        return *this;
      }

      
      ////////// operator*= //////////////////////////
      
      self_type & operator *= (cpu_value_type val)
      {
        viennacl::linalg::am(*this,
                             *this, val, 1, false, false);
        return *this;
      }
      
      ////////// operator/= //////////////////////////

      self_type & operator /= (cpu_value_type val)
      {
        viennacl::linalg::am(*this,
                             *this, val, 1, true, false);
        return *this;
      }



      //const_reference operator()(size_type i, size_type j) const { return A_(start1() + i, start2() + i); }
      //reference operator()(size_type i, size_type j) { return A_(start1() + i, start2() + i); }

      MatrixType & get() { return *A_; }
      const MatrixType & get() const { return *A_; }

    private:
      MatrixType * A_;
      slice row_slice_;
      slice col_slice_;
  };

  template <typename SCALARTYPE, typename F, unsigned int ALIGNMENT>
  viennacl::matrix<SCALARTYPE, F, ALIGNMENT>::matrix(matrix_slice<viennacl::matrix<SCALARTYPE, F, ALIGNMENT> > const & proxy) : rows_(proxy.size1()), columns_(proxy.size2())
  {
    this->elements_.switch_active_handle_id(viennacl::traits::handle(proxy).get_active_handle_id());
    viennacl::backend::memory_create(elements_, sizeof(SCALARTYPE)*internal_size());
    *this = proxy;
  }

  template <typename SCALARTYPE, typename F, unsigned int ALIGNMENT>
  viennacl::matrix<SCALARTYPE, F, ALIGNMENT>::matrix(matrix_slice<const viennacl::matrix<SCALARTYPE, F, ALIGNMENT> > const & proxy) : rows_(proxy.size1()), columns_(proxy.size2())
  {
    this->elements_.switch_active_handle_id(viennacl::traits::handle(proxy).get_active_handle_id());
    viennacl::backend::memory_create(elements_, sizeof(SCALARTYPE)*internal_size());
    *this = proxy;
  }
  
  
  /** @brief Returns an expression template class representing a transposed matrix */
  template <typename MatrixType>
  matrix_expression< const matrix_slice<MatrixType>,
                     const matrix_slice<MatrixType>,
                     op_trans> trans(const matrix_slice<MatrixType> & mat)
  {
    return matrix_expression< const matrix_slice<MatrixType>,
                              const matrix_slice<MatrixType>,
                              op_trans>(mat, mat);
  }
  
  
  
  
  /////////////////////////////////////////////////////////////
  ///////////////////////// CPU to GPU ////////////////////////
  /////////////////////////////////////////////////////////////
  
  //row_major:
  template <typename CPU_MATRIX, typename SCALARTYPE>
  void copy(const CPU_MATRIX & cpu_matrix,
            matrix_slice<matrix<SCALARTYPE, row_major, 1> > & gpu_matrix_slice )
  {
    assert( (cpu_matrix.size1() == gpu_matrix_slice.size1())
           && (cpu_matrix.size2() == gpu_matrix_slice.size2())
           && bool("Matrix size mismatch!"));
    
     if ( (gpu_matrix_slice.size1() > 0) && (gpu_matrix_slice.size1() > 0) )
     {
       std::size_t num_entries = gpu_matrix_slice.size2() * gpu_matrix_slice.stride2(); //no. of entries per stride
       
       std::vector<SCALARTYPE> entries(num_entries);
       
       //copy each stride separately:
       for (std::size_t i=0; i < gpu_matrix_slice.size1(); ++i)
       {
         std::size_t start_offset = (gpu_matrix_slice.start1() + i * gpu_matrix_slice.stride1()) * gpu_matrix_slice.get().internal_size2() + gpu_matrix_slice.start2();
         viennacl::backend::memory_read(gpu_matrix_slice.get().handle(), sizeof(SCALARTYPE)*start_offset, sizeof(SCALARTYPE)*num_entries, &(entries[0]));
         
         for (std::size_t j=0; j < gpu_matrix_slice.size2(); ++j)
           entries[j * gpu_matrix_slice.stride2()] = cpu_matrix(i,j);
         
         viennacl::backend::memory_write(gpu_matrix_slice.get().handle(), sizeof(SCALARTYPE)*start_offset, sizeof(SCALARTYPE)*num_entries, &(entries[0]));
       }
     }
  }
  
  //column_major:
  template <typename CPU_MATRIX, typename SCALARTYPE>
  void copy(const CPU_MATRIX & cpu_matrix,
            matrix_slice<matrix<SCALARTYPE, column_major, 1> > & gpu_matrix_slice )
  {
    assert( (cpu_matrix.size1() == gpu_matrix_slice.size1())
           && (cpu_matrix.size2() == gpu_matrix_slice.size2())
           && bool("Matrix size mismatch!"));
           
    
    if ( (gpu_matrix_slice.size1() > 0) && (gpu_matrix_slice.size1() > 0) )
    {
      std::size_t num_entries = gpu_matrix_slice.size1() * gpu_matrix_slice.stride1(); //no. of entries per stride
      
      std::vector<SCALARTYPE> entries(num_entries);
      
      //copy each column stride separately:
      for (std::size_t j=0; j < gpu_matrix_slice.size2(); ++j)
      {
        std::size_t start_offset = gpu_matrix_slice.start1() + (gpu_matrix_slice.start2() + j * gpu_matrix_slice.stride2()) * gpu_matrix_slice.get().internal_size1();
        
        viennacl::backend::memory_read(gpu_matrix_slice.get().handle(), sizeof(SCALARTYPE)*start_offset, sizeof(SCALARTYPE)*num_entries, &(entries[0]));
        
        for (std::size_t i=0; i < gpu_matrix_slice.size1(); ++i)
          entries[i * gpu_matrix_slice.stride1()] = cpu_matrix(i,j);
        
        viennacl::backend::memory_write(gpu_matrix_slice.get().handle(), sizeof(SCALARTYPE)*start_offset, sizeof(SCALARTYPE)*num_entries, &(entries[0]));
      }
    }
    
  }


  /////////////////////////////////////////////////////////////
  ///////////////////////// GPU to CPU ////////////////////////
  /////////////////////////////////////////////////////////////
  
  
  //row_major:
  template <typename CPU_MATRIX, typename SCALARTYPE>
  void copy(matrix_slice<matrix<SCALARTYPE, row_major, 1> > const & gpu_matrix_slice,
            CPU_MATRIX & cpu_matrix)
  {
    assert( (cpu_matrix.size1() == gpu_matrix_slice.size1())
           && (cpu_matrix.size2() == gpu_matrix_slice.size2())
           && bool("Matrix size mismatch!"));
    
     if ( (gpu_matrix_slice.size1() > 0) && (gpu_matrix_slice.size1() > 0) )
     {
       std::size_t num_entries = gpu_matrix_slice.size2() * gpu_matrix_slice.stride2(); //no. of entries per stride
       
       std::vector<SCALARTYPE> entries(num_entries);
       
       //copy each stride separately:
       for (std::size_t i=0; i < gpu_matrix_slice.size1(); ++i)
       {
         std::size_t start_offset = (gpu_matrix_slice.start1() + i * gpu_matrix_slice.stride1()) * gpu_matrix_slice.get().internal_size2() + gpu_matrix_slice.start2();
         
         viennacl::backend::memory_read(gpu_matrix_slice.get().handle(), sizeof(SCALARTYPE)*start_offset, sizeof(SCALARTYPE)*num_entries, &(entries[0]));
         
         for (std::size_t j=0; j < gpu_matrix_slice.size2(); ++j)
           cpu_matrix(i,j) = entries[j * gpu_matrix_slice.stride2()];
       }
     }
    
  }
  
  
  //column_major:
  template <typename CPU_MATRIX, typename SCALARTYPE>
  void copy(matrix_slice<matrix<SCALARTYPE, column_major, 1> > const & gpu_matrix_slice,
            CPU_MATRIX & cpu_matrix)
  {
    assert( (cpu_matrix.size1() == gpu_matrix_slice.size1())
           && (cpu_matrix.size2() == gpu_matrix_slice.size2())
           && bool("Matrix size mismatch!"));
    
    if ( (gpu_matrix_slice.size1() > 0) && (gpu_matrix_slice.size1() > 0) )
    {
      std::size_t num_entries = gpu_matrix_slice.size1() * gpu_matrix_slice.stride1(); //no. of entries per stride
      
      std::vector<SCALARTYPE> entries(num_entries);
      
      //copy each column stride separately:
      for (std::size_t j=0; j < gpu_matrix_slice.size2(); ++j)
      {
        std::size_t start_offset = gpu_matrix_slice.start1() + (gpu_matrix_slice.start2() + j * gpu_matrix_slice.stride2()) * gpu_matrix_slice.get().internal_size1();
        
        viennacl::backend::memory_read(gpu_matrix_slice.get().handle(), sizeof(SCALARTYPE)*start_offset, sizeof(SCALARTYPE)*num_entries, &(entries[0]));
        
        for (std::size_t i=0; i < gpu_matrix_slice.size1(); ++i)
          cpu_matrix(i,j) = entries[i * gpu_matrix_slice.stride1()];
      }
    }
    
  }


  template<typename MatrixType>
  std::ostream & operator<<(std::ostream & s, matrix_slice<MatrixType> const & proxy)
  {
    MatrixType temp = proxy;
    s << temp;
    return s;
  }

  template<typename MatrixType>
  std::ostream & operator<<(std::ostream & s, matrix_slice<const MatrixType> const & proxy)
  {
    MatrixType temp = proxy;
    s << temp;
    return s;
  }


  //
  // Convenience function
  //
  template <typename MatrixType>
  matrix_slice<MatrixType> project(MatrixType & A, viennacl::slice const & r1, viennacl::slice const & r2)
  {
    assert(r1.size() <= A.size1() && r2.size() <= A.size2() && bool("Size of slice invalid!"));
    
    return matrix_slice<MatrixType>(A, r1, r2);
  }

  template <typename MatrixType>
  matrix_slice<MatrixType> project(matrix_range<MatrixType> & A, viennacl::slice const & r1, viennacl::slice const & r2)
  {
    assert(r1.size() <= A.size1() && r2.size() <= A.size2() && bool("Size of slice invalid!"));
    
    return matrix_slice<MatrixType>(A,
                                    viennacl::slice(A.start1() + r1.start(), r1.stride(), r1.size()),
                                    viennacl::slice(A.start2() + r2.start(), r2.stride(), r2.size())
                                   );
  }

  template <typename MatrixType>
  matrix_slice<MatrixType> project(matrix_slice<MatrixType> & A, viennacl::slice const & r1, viennacl::slice const & r2)
  {
    assert(r1.size() <= A.size1() && r2.size() <= A.size2() && bool("Size of slice invalid!"));
    
    return matrix_slice<MatrixType>(A,
                                    viennacl::slice(A.start1() + r1.start(), A.stride1() * r1.stride(), r1.size()),
                                    viennacl::slice(A.start2() + r2.start(), A.stride2() * r2.stride(), r2.size())
                                   );
  }

  // TODO: Allow mix of range/slice

}

#endif
