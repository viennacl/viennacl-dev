#ifndef VIENNACL_VECTOR_PROXY_HPP_
#define VIENNACL_VECTOR_PROXY_HPP_

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

/** @file vector_proxy.hpp
    @brief Proxy classes for vectors.
*/

#include "viennacl/forwards.h"
#include "viennacl/range.hpp"
#include "viennacl/slice.hpp"
#include "viennacl/vector.hpp"
#include "viennacl/tools/entry_proxy.hpp"

namespace viennacl
{

  template <typename VectorType>
  class vector_range
  {
      typedef vector_range<VectorType>             self_type;
    
    public:
      typedef typename VectorType::value_type      value_type;
      typedef range::size_type                     size_type;
      typedef range::difference_type               difference_type;
      typedef value_type                           reference;
      typedef const value_type &                   const_reference;
      typedef typename VectorType::const_iterator  const_iterator;
      typedef typename VectorType::iterator        iterator;
      

      typedef typename viennacl::result_of::cpu_value_type<value_type>::type    cpu_value_type;
      
      static const int alignment = VectorType::alignment;
      
      vector_range(VectorType & v, 
                   range const & entry_range) : v_(v), entry_range_(entry_range) {}
                   
      size_type start() const { return entry_range_.start(); }
      size_type size() const { return entry_range_.size(); }

      
      /** @brief Operator overload for v1 = A * v2, where v1 and v2 are vector ranges and A is a dense matrix.
      *
      * @param proxy An expression template proxy class
      */
      template <typename MatrixType>
      typename viennacl::enable_if< viennacl::is_matrix<MatrixType>::value, self_type &>::type
      operator=(const vector_expression< const MatrixType,
                                         const self_type,
                                         op_prod> & proxy);
      
      
      

      template <typename LHS, typename RHS, typename OP>
      self_type & operator=(const vector_expression< LHS,
                                                     RHS,
                                                     OP > & proxy) 
      {
        VectorType temp = proxy;
        *this = temp;
        return *this;
      }      


      /** @brief Convenience function, which allows to assign a vector range directly to another vector range of suitable size */
      self_type & operator=(const self_type & vec) 
      {
        viennacl::linalg::av(*this, 
                              vec,   cpu_value_type(1.0), 1, false, false);
        return *this;
      }      

      /** @brief Assignment of a vector (or -range or -slice) */
      template <typename V1>
      typename viennacl::enable_if< viennacl::is_vector<V1>::value, 
                                    self_type &>::type
      operator = (const V1 & vec)
      {
        viennacl::linalg::av(*this, 
                              vec,   cpu_value_type(1.0), 1, false, false);
        return *this;
      }
      
      /** @brief Assignment of a scaled vector (or -range or -slice), i.e. v1 -= v2 @ alpha, where @ is either product or division and alpha is either a CPU or a GPU scalar
      */
      template <typename V1, typename S1, typename OP>
      typename viennacl::enable_if< viennacl::is_vector<V1>::value && viennacl::is_any_scalar<S1>::value,
                                    self_type &>::type
      operator = (const vector_expression< const V1,
                                           const S1,
                                           OP> & proxy)
      {
        viennacl::linalg::av(*this, 
                             proxy.lhs(), proxy.rhs(), 1, (viennacl::is_division<OP>::value ? true : false), (viennacl::is_flip_sign_scalar<S1>::value ? true : false) );
        return *this;
      }
      
      ///////////// operator +=

      /** @brief Inplace addition of a vector (or -range or -slice) */
      template <typename V1>
      typename viennacl::enable_if< viennacl::is_vector<V1>::value, 
                                    self_type &>::type
      operator += (const V1 & vec)
      {
        viennacl::linalg::avbv(*this, 
                               *this, cpu_value_type(1.0), 1, false, false,
                               vec,   cpu_value_type(1.0), 1, false, false);
        return *this;
      }
      
      /** @brief Inplace addition of a scaled vector (or -range or -slice), i.e. v1 -= v2 @ alpha, where @ is either product or division and alpha is either a CPU or a GPU scalar
      */
      template <typename V1, typename S1, typename OP>
      typename viennacl::enable_if< viennacl::is_vector<V1>::value && viennacl::is_any_scalar<S1>::value,
                                    self_type &>::type
      operator += (const vector_expression< const V1,
                                            const S1,
                                            OP> & proxy)
      {
        //viennacl::linalg::inplace_mul_add(*this, proxy.lhs(), proxy.rhs());
        viennacl::linalg::avbv(*this, 
                               *this,   cpu_value_type(1.0), 1, false,                                         false,
                               proxy.lhs(), proxy.rhs(), 1, (viennacl::is_division<OP>::value ? true : false), (viennacl::is_flip_sign_scalar<S1>::value ? true : false) );
        return *this;
      }
      
      ///////////// operator -=

      /** @brief Inplace subtraction of a vector (or -range or -slice) */
      template <typename V1>
      typename viennacl::enable_if< viennacl::is_vector<V1>::value, 
                                    self_type &>::type
      operator -= (const V1 & vec)
      {
        //viennacl::linalg::inplace_sub(*this, vec);
        viennacl::linalg::avbv(*this, 
                               *this, cpu_value_type(1.0),  1, false, false,
                               vec,   cpu_value_type(-1.0), 1, false, false);
        return *this;
      }
      
      /** @brief Inplace subtraction of a scaled vector (or -range or -slice), i.e. v1 -= v2 @ alpha, where @ is either product or division and alpha is either a CPU or a GPU scalar
      */
      template <typename V1, typename S1, typename OP>
      typename viennacl::enable_if< viennacl::is_vector<V1>::value && viennacl::is_any_scalar<S1>::value,
                                    self_type &>::type
      operator -= (const vector_expression< const V1,
                                            const S1,
                                            OP> & proxy)
      {
        //viennacl::linalg::inplace_mul_add(*this, proxy.lhs(), proxy.rhs());
        viennacl::linalg::avbv(*this, 
                               *this,   cpu_value_type(1.0), 1, false,                                             false,
                               proxy.lhs(), proxy.rhs(), 1, (viennacl::is_division<OP>::value ? true : false), (viennacl::is_flip_sign_scalar<S1>::value ? false : true));
        return *this;
      }
      

      ///////////// operator *=
      /** @brief Scales this vector range by a CPU scalar value
      */
      self_type & operator *= (cpu_value_type val)
      {
        viennacl::linalg::av(*this,
                             *this, val, 1, false, false);
        return *this;
      }

      /** @brief Scales this vector range by a GPU scalar value
      */
      template <typename S1>
      typename viennacl::enable_if< viennacl::is_any_scalar<S1>::value,
                                    self_type & 
                                  >::type
      operator *= (S1 const & gpu_val)
      {
        viennacl::linalg::av(*this,
                             *this, gpu_val, 1, false, (viennacl::is_flip_sign_scalar<S1>::value ? true : false));
        return *this;
      }

      /** @brief Scales this vector range by a CPU scalar value
      */
      self_type & operator /= (cpu_value_type val)
      {
        viennacl::linalg::av(*this,
                             *this, val, 1, true, false);
        return *this;
      }
      
      /** @brief Scales this vector range by a CPU scalar value
      */
      template <typename S1>
      typename viennacl::enable_if< viennacl::is_any_scalar<S1>::value,
                                    self_type & 
                                  >::type
      operator /= (S1 const & gpu_val)
      {
        viennacl::linalg::av(*this,
                             *this, gpu_val, 1, true, (viennacl::is_flip_sign_scalar<S1>::value ? true : false));
        return *this;
      }
      
      
      ///////////// Direct manipulation via operator() and operator[]
      //read-write access to an element of the vector
      /** @brief Read-write access to a single element of the vector
      */
      entry_proxy<cpu_value_type> operator()(size_type index)
      {
        return entry_proxy<cpu_value_type>(index + start(), v_.get());
      }

      /** @brief Read-write access to a single element of the vector
      */
      entry_proxy<cpu_value_type> operator[](size_type index)
      {
        return entry_proxy<cpu_value_type>(index + start(), v_.get());
      }


      /** @brief Read access to a single element of the vector
      */
      scalar<cpu_value_type> operator()(size_type index) const
      {
        scalar<cpu_value_type> tmp;
        cl_int err;
        err = clEnqueueCopyBuffer(viennacl::ocl::get_queue().handle().get(), v_.get(), tmp.handle(), sizeof(cpu_value_type)*(index + start()), 0, sizeof(cpu_value_type), 0, NULL, NULL);
        VIENNACL_ERR_CHECK(err);
        return tmp;
      }
      
      /** @brief Read access to a single element of the vector
      */
      scalar<cpu_value_type> operator[](size_type index) const
      {
        return operator()(index);
      }
      
      ///////////// iterators:
      /** @brief Returns an iterator pointing to the beginning of the vector  (STL like)*/
      iterator begin()
      {
        return iterator(v_, 0, start());
      }

      /** @brief Returns an iterator pointing to the end of the vector (STL like)*/
      iterator end()
      {
        return iterator(v_, size(), start());
      }

      /** @brief Returns a const-iterator pointing to the beginning of the vector (STL like)*/
      const_iterator begin() const
      {
        return const_iterator(v_, start());
      }

      /** @brief Returns a const-iterator pointing to the end of the vector (STL like)*/
      const_iterator end() const
      {
        return const_iterator(v_, size(), start());
      }
      
      ///////////// Misc

      VectorType & get() { return v_; }
      const VectorType & get() const { return v_; }

    private:
      VectorType & v_;
      range entry_range_;
  };
  
  
  //implement copy-CTOR for vector from vector_range:
  
  template <typename SCALARTYPE, unsigned int ALIGNMENT>
  viennacl::vector<SCALARTYPE, ALIGNMENT>::vector(const vector_range< viennacl::vector<SCALARTYPE, ALIGNMENT> > & r) : size_(r.size())
  {
    assert(this->size() == r.size() && "Vector size mismatch!");
    
    if (this->size() > 0)
    {
      this->elements_ = viennacl::ocl::current_context().create_memory(CL_MEM_READ_WRITE, sizeof(SCALARTYPE)*internal_size());
      
      viennacl::linalg::av(*this, 
                           r, SCALARTYPE(1.0), 1, false, false);
    }
    
  }
  
  
  
  //implement operator= for vector:
  
  template <typename SCALARTYPE, unsigned int ALIGNMENT>
  viennacl::vector<SCALARTYPE, ALIGNMENT> & 
  viennacl::vector<SCALARTYPE, ALIGNMENT>::operator=(const vector_range< viennacl::vector<SCALARTYPE, ALIGNMENT> > & r) 
  {
    if (this->size() > 0)
      viennacl::linalg::av(*this, 
                           r, SCALARTYPE(1.0), 1, false, false);
    
    return *this;
  }
  
  
  

  
  template<typename VectorType>
  std::ostream & operator<<(std::ostream & s, vector_range<VectorType> const & proxy)
  {
    typedef typename VectorType::value_type   ScalarType;
    std::vector<ScalarType> temp(proxy.size());
    viennacl::copy(proxy, temp);
    
    //instead of printing 'temp' directly, let's reuse the existing functionality for viennacl::vector. It certainly adds overhead, but printing a vector is typically not about performance...
    VectorType temp2(temp.size());
    viennacl::copy(temp, temp2);
    s << temp2;
    return s;
  }
  
  
  
  
  /////////////////////////////////////////////////////////////
  ///////////////////////// CPU to GPU ////////////////////////
  /////////////////////////////////////////////////////////////
  
  template <typename VectorType, typename SCALARTYPE>
  void copy(const VectorType & cpu_vector,
            vector_range<vector<SCALARTYPE> > & gpu_vector_range )
  {
    assert(cpu_vector.end() - cpu_vector.begin() >= 0);
    
    if (cpu_vector.end() - cpu_vector.begin() > 0)
    {
      //we require that the size of the gpu_vector is larger or equal to the cpu-size
      std::vector<SCALARTYPE> temp_buffer(cpu_vector.end() - cpu_vector.begin());
      std::copy(cpu_vector.begin(), cpu_vector.end(), temp_buffer.begin());
      cl_int err = clEnqueueWriteBuffer(viennacl::ocl::get_queue().handle().get(),
                                        gpu_vector_range.get().handle().get(), CL_TRUE, sizeof(SCALARTYPE)*gpu_vector_range.start(),
                                        sizeof(SCALARTYPE)*temp_buffer.size(),
                                        &(temp_buffer[0]), 0, NULL, NULL);
      VIENNACL_ERR_CHECK(err);
    }
  }


  /** @brief Transfer from a cpu vector to a gpu vector. Convenience wrapper for viennacl::linalg::fast_copy(cpu_vec.begin(), cpu_vec.end(), gpu_vec.begin());
  *
  * @param cpu_vec    A cpu vector. Type requirements: Iterator can be obtained via member function .begin() and .end()
  * @param gpu_vec    The gpu vector.
  */
  template <typename CPUVECTOR, typename VectorType>
  void fast_copy(const CPUVECTOR & cpu_vec, vector_range<VectorType> & gpu_vec)
  {
    viennacl::fast_copy(cpu_vec.begin(), cpu_vec.end(), gpu_vec.begin());
  }

  /////////////////////////////////////////////////////////////
  ///////////////////////// GPU to CPU ////////////////////////
  /////////////////////////////////////////////////////////////
  

  template <typename SCALARTYPE, typename VectorType>
  void copy(vector_range<vector<SCALARTYPE> > const & gpu_vector_range,
            VectorType & cpu_vector)
  {
    assert(cpu_vector.end() - cpu_vector.begin() >= 0);
    
    if (cpu_vector.end() > cpu_vector.begin())
    {
      std::vector<SCALARTYPE> temp_buffer(cpu_vector.end() - cpu_vector.begin());
      cl_int err = clEnqueueReadBuffer(viennacl::ocl::get_queue().handle().get(),
                                        gpu_vector_range.get().handle().get(), CL_TRUE, sizeof(SCALARTYPE)*gpu_vector_range.start(), 
                                        sizeof(SCALARTYPE)*temp_buffer.size(),
                                        &(temp_buffer[0]), 0, NULL, NULL);
      VIENNACL_ERR_CHECK(err);
      viennacl::ocl::get_queue().finish();
      
      //now copy entries to cpu_vec:
      std::copy(temp_buffer.begin(), temp_buffer.end(), cpu_vector.begin());
    }
  }


  /** @brief Transfer from a GPU vector range to a CPU vector. Convenience wrapper for viennacl::linalg::fast_copy(gpu_vec.begin(), gpu_vec.end(), cpu_vec.begin());
  *
  * @param gpu_vec    A gpu vector range.
  * @param cpu_vec    The cpu vector. Type requirements: Output iterator can be obtained via member function .begin()
  */
  template <typename VectorType, typename CPUVECTOR>
  void fast_copy(vector_range< VectorType > const & gpu_vec,
                 CPUVECTOR & cpu_vec )
  {
    viennacl::fast_copy(gpu_vec.begin(), gpu_vec.end(), cpu_vec.begin());
  }



  //
  // Convenience function
  //
  template <typename VectorType>
  vector_range<VectorType> project(VectorType & vec, viennacl::range const & r1)
  {
    return vector_range<VectorType>(vec, r1);
  }


//
//
//
/////////////////////////////// Slice /////////////////////////////////////////////
//
//
//




  template <typename VectorType>
  class vector_slice
  {
      typedef vector_slice<VectorType>             self_type;
    
    public:
      typedef typename VectorType::value_type      value_type;
      typedef slice::size_type                     size_type;
      typedef slice::difference_type               difference_type;
      typedef value_type                           reference;
      typedef const value_type &                   const_reference;
      typedef typename VectorType::const_iterator  const_iterator;
      typedef typename VectorType::iterator        iterator;
      

      typedef typename viennacl::result_of::cpu_value_type<value_type>::type    cpu_value_type;
      
      static const int alignment = VectorType::alignment;
      
      vector_slice(VectorType & v, 
                   slice const & entry_slice) : v_(v), entry_slice_(entry_slice) {}
                   
      size_type start() const { return entry_slice_.start(); }
      size_type stride() const { return entry_slice_.stride(); }
      size_type size() const { return entry_slice_.size(); }

      
      /** @brief Operator overload for v1 = A * v2, where v1 and v2 are vector slices and A is a dense matrix.
      *
      * @param proxy An expression template proxy class
      */
      template <typename MatrixType>
      typename viennacl::enable_if< viennacl::is_matrix<MatrixType>::value, self_type &>::type
      operator=(const vector_expression< const MatrixType,
                                         const self_type,
                                         op_prod> & proxy);
      
      
      

      template <typename LHS, typename RHS, typename OP>
      self_type & operator=(const vector_expression< LHS,
                                                     RHS,
                                                     OP > & proxy) 
      {
        VectorType temp = proxy;
        *this = temp;
        return *this;
      }      


      /** @brief Convenience function, which allows to assign a vector range directly to another vector slice of suitable size */
      self_type & operator=(const self_type & vec) 
      {
        viennacl::linalg::av(*this, 
                              vec,   cpu_value_type(1.0), 1, false, false);
        return *this;
      }      

      /** @brief Assignment of a vector (or -range or -slice) */
      template <typename V1>
      typename viennacl::enable_if< viennacl::is_vector<V1>::value, 
                                    self_type &>::type
      operator = (const V1 & vec)
      {
        viennacl::linalg::av(*this, 
                              vec,   cpu_value_type(1.0), 1, false, false);
        return *this;
      }
      
      /** @brief Assignment of a scaled vector (or -range or -slice), i.e. v1 -= v2 @ alpha, where @ is either product or division and alpha is either a CPU or a GPU scalar
      */
      template <typename V1, typename S1, typename OP>
      typename viennacl::enable_if< viennacl::is_vector<V1>::value && viennacl::is_any_scalar<S1>::value,
                                    self_type &>::type
      operator = (const vector_expression< const V1,
                                           const S1,
                                           OP> & proxy)
      {
        viennacl::linalg::av(*this, 
                             proxy.lhs(), proxy.rhs(), 1, (viennacl::is_division<OP>::value ? true : false), (viennacl::is_flip_sign_scalar<S1>::value ? true : false) );
        return *this;
      }
      

      ///////////// operator +=

      /** @brief Inplace addition of a vector (or -range or -slice) */
      template <typename V1>
      typename viennacl::enable_if< viennacl::is_vector<V1>::value, 
                                    self_type &>::type
      operator += (const V1 & vec)
      {
        viennacl::linalg::avbv(*this, 
                               *this, cpu_value_type(1.0), 1, false, false,
                               vec,   cpu_value_type(1.0), 1, false, false);
        return *this;
      }
      
      /** @brief Inplace addition of a scaled vector (or -range or -slice), i.e. v1 -= v2 @ alpha, where @ is either product or division and alpha is either a CPU or a GPU scalar
      */
      template <typename V1, typename S1, typename OP>
      typename viennacl::enable_if< viennacl::is_vector<V1>::value && viennacl::is_any_scalar<S1>::value,
                                    self_type &>::type
      operator += (const vector_expression< const V1,
                                            const S1,
                                            OP> & proxy)
      {
        //viennacl::linalg::inplace_mul_add(*this, proxy.lhs(), proxy.rhs());
        viennacl::linalg::avbv(*this, 
                               *this,   cpu_value_type(1.0), 1, false,                                         false,
                               proxy.lhs(), proxy.rhs(), 1, (viennacl::is_division<OP>::value ? true : false), (viennacl::is_flip_sign_scalar<S1>::value ? true : false) );
        return *this;
      }
      
      ///////////// operator -=

      /** @brief Inplace subtraction of a vector (or -range or -slice) */
      template <typename V1>
      typename viennacl::enable_if< viennacl::is_vector<V1>::value, 
                                    self_type &>::type
      operator -= (const V1 & vec)
      {
        //viennacl::linalg::inplace_sub(*this, vec);
        viennacl::linalg::avbv(*this, 
                               *this, cpu_value_type(1.0),  1, false, false,
                               vec,   cpu_value_type(-1.0), 1, false, false);
        return *this;
      }
      
      /** @brief Inplace subtraction of a scaled vector (or -range or -slice), i.e. v1 -= v2 @ alpha, where @ is either product or division and alpha is either a CPU or a GPU scalar
      */
      template <typename V1, typename S1, typename OP>
      typename viennacl::enable_if< viennacl::is_vector<V1>::value && viennacl::is_any_scalar<S1>::value,
                                    self_type &>::type
      operator -= (const vector_expression< const V1,
                                            const S1,
                                            OP> & proxy)
      {
        //viennacl::linalg::inplace_mul_add(*this, proxy.lhs(), proxy.rhs());
        viennacl::linalg::avbv(*this, 
                               *this,   cpu_value_type(1.0), 1, false,                                             false,
                               proxy.lhs(), proxy.rhs(), 1, (viennacl::is_division<OP>::value ? true : false), (viennacl::is_flip_sign_scalar<S1>::value ? false : true));
        return *this;
      }
      

      ///////////// operator *=
      /** @brief Scales this vector range by a CPU scalar value
      */
      self_type & operator *= (cpu_value_type val)
      {
        viennacl::linalg::av(*this,
                             *this, val, 1, false, false);
        return *this;
      }

      /** @brief Scales this vector range by a GPU scalar value
      */
      template <typename S1>
      typename viennacl::enable_if< viennacl::is_any_scalar<S1>::value,
                                    self_type & 
                                  >::type
      operator *= (S1 const & gpu_val)
      {
        viennacl::linalg::av(*this,
                             *this, gpu_val, 1, false, (viennacl::is_flip_sign_scalar<S1>::value ? true : false));
        return *this;
      }

      /** @brief Scales this vector range by a CPU scalar value
      */
      self_type & operator /= (cpu_value_type val)
      {
        viennacl::linalg::av(*this,
                             *this, val, 1, true, false);
        return *this;
      }
      
      /** @brief Scales this vector range by a CPU scalar value
      */
      template <typename S1>
      typename viennacl::enable_if< viennacl::is_any_scalar<S1>::value,
                                    self_type & 
                                  >::type
      operator /= (S1 const & gpu_val)
      {
        viennacl::linalg::av(*this,
                             *this, gpu_val, 1, true, (viennacl::is_flip_sign_scalar<S1>::value ? true : false));
        return *this;
      }
      
      
      
      ///////////// Direct manipulation via operator() and operator[]
      //read-write access to an element of the vector
      /** @brief Read-write access to a single element of the vector
      */
      entry_proxy<cpu_value_type> operator()(size_type index)
      {
        return entry_proxy<cpu_value_type>(index * stride() + start(), v_.get());
      }

      /** @brief Read-write access to a single element of the vector
      */
      entry_proxy<cpu_value_type> operator[](size_type index)
      {
        return entry_proxy<cpu_value_type>(index * stride() + start(), v_.get());
      }


      /** @brief Read access to a single element of the vector
      */
      scalar<cpu_value_type> operator()(size_type index) const
      {
        scalar<cpu_value_type> tmp;
        cl_int err;
        err = clEnqueueCopyBuffer(viennacl::ocl::get_queue().handle().get(), v_.get(), tmp.handle(), sizeof(cpu_value_type)*(index * stride() + start()), 0, sizeof(cpu_value_type), 0, NULL, NULL);
        VIENNACL_ERR_CHECK(err);
        return tmp;
      }
      
      /** @brief Read access to a single element of the vector
      */
      scalar<cpu_value_type> operator[](size_type index) const
      {
        return operator()(index);
      }
      
      ///////////// iterators:
      /** @brief Returns an iterator pointing to the beginning of the vector  (STL like)*/
      iterator begin()
      {
        return iterator(v_, 0, start(), stride());
      }

      /** @brief Returns an iterator pointing to the end of the vector (STL like)*/
      iterator end()
      {
        return iterator(v_, size(), start(), stride());
      }

      /** @brief Returns a const-iterator pointing to the beginning of the vector (STL like)*/
      const_iterator begin() const
      {
        return const_iterator(v_, 0, start(), stride());
      }

      /** @brief Returns a const-iterator pointing to the end of the vector (STL like)*/
      const_iterator end() const
      {
        return const_iterator(v_, size(), start(), stride());
      }
      
      ///////////// Misc

      VectorType & get() { return v_; }
      const VectorType & get() const { return v_; }

    private:
      VectorType & v_;
      slice entry_slice_;
  };
  
  
  //implement copy-CTOR for vector from vector_slice:
  
  template <typename SCALARTYPE, unsigned int ALIGNMENT>
  viennacl::vector<SCALARTYPE, ALIGNMENT>::vector(const vector_slice< viennacl::vector<SCALARTYPE, ALIGNMENT> > & r) : size_(r.size())
  {
    if (this->size() > 0)
    {
      this->elements_ = viennacl::ocl::current_context().create_memory(CL_MEM_READ_WRITE, sizeof(SCALARTYPE)*internal_size());
      
      viennacl::linalg::av(*this, 
                           r, SCALARTYPE(1.0), 1, false, false);
    }
    
  }
  
  
  
  //implement operator= for vector:
  
  template <typename SCALARTYPE, unsigned int ALIGNMENT>
  viennacl::vector<SCALARTYPE, ALIGNMENT> & 
  viennacl::vector<SCALARTYPE, ALIGNMENT>::operator=(const vector_slice< viennacl::vector<SCALARTYPE, ALIGNMENT> > & r) 
  {
    assert(this->size() == r.size() && "Vector size mismatch!");
    
    if (this->size() > 0)
      viennacl::linalg::av(*this, 
                           r, SCALARTYPE(1.0), 1, false, false);
    
    return *this;
  }
  
  
  

  
  template<typename VectorType>
  std::ostream & operator<<(std::ostream & s, vector_slice<VectorType> const & proxy)
  {
    typedef typename VectorType::value_type   ScalarType;
    std::vector<ScalarType> temp(proxy.size());
    viennacl::copy(proxy, temp);
    
    //instead of printing 'temp' directly, let's reuse the existing functionality for viennacl::vector. It certainly adds overhead, but printing a vector is typically not about performance...
    VectorType temp2(temp.size());
    viennacl::copy(temp, temp2);
    s << temp2;
    return s;
  }
  
  
  
  
  /////////////////////////////////////////////////////////////
  ///////////////////////// CPU to GPU ////////////////////////
  /////////////////////////////////////////////////////////////
  
  template <typename VectorType, typename SCALARTYPE>
  void copy(const VectorType & cpu_vector,
            vector_slice<vector<SCALARTYPE> > & gpu_vector_slice )
  {
    assert(cpu_vector.end() - cpu_vector.begin() >= 0);
    
    if (cpu_vector.end() - cpu_vector.begin() > 0)
    {
      
      // OpenCL 1.0 version: (no use of clEnqueueWriteBufferRect())
      std::vector<SCALARTYPE> temp_buffer(gpu_vector_slice.stride() * gpu_vector_slice.size());
      
      cl_int err = clEnqueueReadBuffer(viennacl::ocl::get_queue().handle().get(),
                                        gpu_vector_slice.get().handle().get(), CL_TRUE, sizeof(SCALARTYPE)*gpu_vector_slice.start(), 
                                        sizeof(SCALARTYPE)*temp_buffer.size(),
                                        &(temp_buffer[0]), 0, NULL, NULL);
      
      VIENNACL_ERR_CHECK(err);

      for (std::size_t i=0; i<cpu_vector.size(); ++i)
      {
        temp_buffer[i * gpu_vector_slice.stride()] = cpu_vector[i];
      }
      
      err = clEnqueueWriteBuffer(viennacl::ocl::get_queue().handle().get(),
                                 gpu_vector_slice.get().handle().get(), CL_TRUE, sizeof(SCALARTYPE)*gpu_vector_slice.start(),
                                 sizeof(SCALARTYPE)*temp_buffer.size(),
                                 &(temp_buffer[0]), 0, NULL, NULL);
      
      VIENNACL_ERR_CHECK(err);
    }
  }



  /////////////////////////////////////////////////////////////
  ///////////////////////// GPU to CPU ////////////////////////
  /////////////////////////////////////////////////////////////
  

  template <typename VectorType, typename SCALARTYPE>
  void copy(vector_slice<vector<SCALARTYPE> > const & gpu_vector_slice,
            VectorType & cpu_vector)
  {
    assert(cpu_vector.end() - cpu_vector.begin() >= 0);
    
    if (cpu_vector.end() > cpu_vector.begin())
    {
      // OpenCL 1.0 version: (no use of clEnqueueWriteBufferRect())
      std::vector<SCALARTYPE> temp_buffer(gpu_vector_slice.stride() * gpu_vector_slice.size());
      
      cl_int err = clEnqueueReadBuffer(viennacl::ocl::get_queue().handle().get(),
                                        gpu_vector_slice.get().handle().get(), CL_TRUE, sizeof(SCALARTYPE)*gpu_vector_slice.start(), 
                                        sizeof(SCALARTYPE)*temp_buffer.size(),
                                        &(temp_buffer[0]), 0, NULL, NULL);
      
      VIENNACL_ERR_CHECK(err);

      for (std::size_t i=0; i<cpu_vector.size(); ++i)
      {
        cpu_vector[i] = temp_buffer[i * gpu_vector_slice.stride()];
      }
    }
  }





  //
  // Convenience function
  //
  template <typename VectorType>
  vector_slice<VectorType> project(VectorType & vec, viennacl::slice const & s1)
  {
    return vector_slice<VectorType>(vec, s1);
  }

}

#endif