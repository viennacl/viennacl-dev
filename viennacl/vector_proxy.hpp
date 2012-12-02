#ifndef VIENNACL_VECTOR_PROXY_HPP_
#define VIENNACL_VECTOR_PROXY_HPP_

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
      size_type internal_size() const { return entry_range_.size(); }

      
      /** @brief Operator overload for v1 = A * v2, where v1 and v2 are vector ranges and A is a dense matrix.
      *
      * @param proxy An expression template proxy class
      */
      template <typename M1, typename V1>
      typename viennacl::enable_if<    viennacl::is_any_dense_nonstructured_matrix<M1>::value
                                    && viennacl::is_any_dense_nonstructured_vector<V1>::value,
                                    self_type &>::type
      operator=(const vector_expression< const M1, const V1, op_prod> & proxy)
      {
        viennacl::linalg::prod_impl(proxy.lhs(), proxy.rhs(), *this);
        return *this;
      }
      

      /** @brief Generic overload for any assigned vector_expressions. Forward to vector<> */
      template <typename LHS, typename RHS, typename OP>
      self_type & operator=(const vector_expression< LHS, RHS, OP > & proxy) 
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
      typename viennacl::enable_if< viennacl::is_any_dense_nonstructured_vector<V1>::value, 
                                    self_type &>::type
      operator = (const V1 & vec)
      {
        viennacl::linalg::av(*this, 
                              vec,   cpu_value_type(1.0), 1, false, false);
        return *this;
      }
      
      /** @brief Assignment of a scaled vector (or -range or -slice), i.e. v1 = v2 @ alpha, where @ is either product or division and alpha is either a CPU or a GPU scalar
      */
      template <typename V1, typename S1, typename OP>
      typename viennacl::enable_if< viennacl::is_any_dense_nonstructured_vector<V1>::value && viennacl::is_any_scalar<S1>::value,
                                    self_type &>::type
      operator = (const vector_expression< const V1,
                                           const S1,
                                           OP> & proxy)
      {
        viennacl::linalg::av(*this, 
                             proxy.lhs(), proxy.rhs(), 1, (viennacl::is_division<OP>::value ? true : false), (viennacl::is_flip_sign_scalar<S1>::value ? true : false) );
        return *this;
      }
      
      /** @brief Creates the vector from the supplied unit vector. */
      self_type & operator = (unit_vector<cpu_value_type> const & v)
      {
        assert( (v.size() == size())
                && bool("Incompatible vector sizes!"));
        
        viennacl::linalg::vector_assign(*this, cpu_value_type(0));
        this->operator()(v.index()) = cpu_value_type(1);
        
        return *this;
      }
      
      /** @brief Creates the vector from the supplied zero vector. */
      self_type & operator = (zero_vector<cpu_value_type> const & v)
      {
        assert( (v.size() == size())
                && bool("Incompatible vector sizes!"));
        
        viennacl::linalg::vector_assign(*this, cpu_value_type(0));
        
        return *this;
      }

      /** @brief Creates the vector from the supplied scalar vector. */
      self_type & operator = (scalar_vector<cpu_value_type> const & v)
      {
        assert( (v.size() == size())
                && bool("Incompatible vector sizes!"));
        
        viennacl::linalg::vector_assign(*this, v[0]);
        
        return *this;
      }
      
      
      //
      ///////////// operators with implicit conversion
      //
      
      /** @brief Scales this vector range by a CPU scalar value
      */
      self_type & operator *= (cpu_value_type val)
      {
        viennacl::linalg::av(*this,
                             *this, val, 1, false, false);
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
      
      
      ///////////// Direct manipulation via operator() and operator[]
      //read-write access to an element of the vector
      /** @brief Read-write access to a single element of the vector
      */
      entry_proxy<cpu_value_type> operator()(size_type index)
      {
        return entry_proxy<cpu_value_type>(index + start(), v_.handle());
      }

      /** @brief Read-write access to a single element of the vector
      */
      entry_proxy<cpu_value_type> operator[](size_type index)
      {
        return entry_proxy<cpu_value_type>(index + start(), v_.handle());
      }


      /** @brief Read access to a single element of the vector
      */
      scalar<cpu_value_type> operator()(size_type index) const
      {
        scalar<cpu_value_type> tmp;
        viennacl::backend::memory_copy(viennacl::traits::handle(v_), viennacl::traits::handle(tmp),
                                       sizeof(cpu_value_type)*(index + start()), 0,
                                       sizeof(cpu_value_type));
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
        return const_iterator(v_, 0, start());
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
  
  
  // implement copy-CTOR for vector:
  template <typename SCALARTYPE, unsigned int ALIGNMENT>
  viennacl::vector<SCALARTYPE, ALIGNMENT>::vector(vector_range<viennacl::vector<SCALARTYPE, ALIGNMENT> > const & proxy) : size_(proxy.size())
  {
    if (proxy.size() > 0)
    {
      elements_.switch_active_handle_id(viennacl::traits::handle(proxy).get_active_handle_id());
      viennacl::backend::memory_create(elements_, sizeof(SCALARTYPE)*internal_size());
      pad();
      
      viennacl::linalg::av(*this, 
                            proxy, SCALARTYPE(1.0), 1, false, false);
    }
  }

  template <typename SCALARTYPE, unsigned int ALIGNMENT>
  viennacl::vector<SCALARTYPE, ALIGNMENT>::vector(vector_range<const viennacl::vector<SCALARTYPE, ALIGNMENT> > const & proxy) : size_(proxy.size())
  {
    if (proxy.size() > 0)
    {
      elements_.switch_active_handle_id(viennacl::traits::handle(proxy).get_active_handle_id());
      viennacl::backend::memory_create(elements_, sizeof(SCALARTYPE)*internal_size());
      pad();
      
      viennacl::linalg::av(*this, 
                            proxy, SCALARTYPE(1.0), 1, false, false);
    }
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
      viennacl::backend::memory_write(gpu_vector_range.get().handle(), sizeof(SCALARTYPE)*gpu_vector_range.start(), sizeof(SCALARTYPE)*temp_buffer.size(), &(temp_buffer[0]));
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
      viennacl::backend::memory_read(gpu_vector_range.get().handle(), sizeof(SCALARTYPE)*gpu_vector_range.start(), sizeof(SCALARTYPE)*temp_buffer.size(), &(temp_buffer[0]));
      
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

  template <typename VectorType>
  vector_range<VectorType> project(viennacl::vector_range<VectorType> & vec, viennacl::range const & r1)
  {
    assert(r1.size() <= vec.size() && bool("Size of range invalid!"));
    return vector_range<VectorType>(vec.get(), viennacl::range(vec.start() + r1.start(), vec.start() + r1.start() + r1.size()));
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
      size_type internal_size() const { return entry_slice_.size(); }

      
      /** @brief Operator overload for v1 = A * v2, where v1 and v2 are vector slices and A is a dense matrix.
      *
      * @param proxy An expression template proxy class
      */
      template <typename M1, typename V1>
      typename viennacl::enable_if<    viennacl::is_any_dense_nonstructured_matrix<M1>::value
                                    && viennacl::is_any_dense_nonstructured_vector<V1>::value,
                                    self_type &>::type
      operator=(const vector_expression< const M1,
                                         const V1,
                                         op_prod> & proxy)
      {
        viennacl::linalg::prod_impl(proxy.lhs(), proxy.rhs(), *this);
        return *this;
      }

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
      typename viennacl::enable_if< viennacl::is_any_dense_nonstructured_vector<V1>::value, 
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
      typename viennacl::enable_if< viennacl::is_any_dense_nonstructured_vector<V1>::value && viennacl::is_any_scalar<S1>::value,
                                    self_type &>::type
      operator = (const vector_expression< const V1,
                                           const S1,
                                           OP> & proxy)
      {
        viennacl::linalg::av(*this, 
                             proxy.lhs(), proxy.rhs(), 1, (viennacl::is_division<OP>::value ? true : false), (viennacl::is_flip_sign_scalar<S1>::value ? true : false) );
        return *this;
      }
      

      /** @brief Creates the vector from the supplied unit vector. */
      self_type & operator = (unit_vector<cpu_value_type> const & v)
      {
        assert( (v.size() == size())
                && bool("Incompatible vector sizes!"));
        
        viennacl::linalg::vector_assign(*this, cpu_value_type(0));
        this->operator()(v.index()) = cpu_value_type(1);
        return *this;
      }
      
      /** @brief Creates the vector from the supplied zero vector. */
      self_type & operator = (zero_vector<cpu_value_type> const & v)
      {
        assert( (v.size() == size())
                && bool("Incompatible vector sizes!"));
        
        viennacl::linalg::vector_assign(*this, cpu_value_type(0));
        return *this;
      }

      /** @brief Creates the vector from the supplied scalar vector. */
      self_type & operator = (scalar_vector<cpu_value_type> const & v)
      {
        assert( (v.size() == size())
                && bool("Incompatible vector sizes!"));
        
        viennacl::linalg::vector_assign(*this, v[0]);
        return *this;
      }
      

      ///////////// operator overloads with implicit conversion:
      /** @brief Scales this vector range by a CPU scalar value
      */
      self_type & operator *= (cpu_value_type val)
      {
        viennacl::linalg::av(*this,
                             *this, val, 1, false, false);
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
      
      
      ///////////// Direct manipulation via operator() and operator[]
      //read-write access to an element of the vector
      /** @brief Read-write access to a single element of the vector
      */
      entry_proxy<cpu_value_type> operator()(size_type index)
      {
        return entry_proxy<cpu_value_type>(index * stride() + start(), v_.handle());
      }

      /** @brief Read-write access to a single element of the vector
      */
      entry_proxy<cpu_value_type> operator[](size_type index)
      {
        return entry_proxy<cpu_value_type>(index * stride() + start(), v_.handle());
      }


      /** @brief Read access to a single element of the vector
      */
      scalar<cpu_value_type> operator()(size_type index) const
      {
        scalar<cpu_value_type> tmp = 1.0;
        viennacl::backend::memory_copy(viennacl::traits::handle(v_), viennacl::traits::handle(tmp),
                                       sizeof(cpu_value_type)*(index * stride() + start()), 0,
                                       sizeof(cpu_value_type));
        std::cout << tmp << std::endl;
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
  

  // implement copy-CTOR for vector:
  template <typename SCALARTYPE, unsigned int ALIGNMENT>
  viennacl::vector<SCALARTYPE, ALIGNMENT>::vector(vector_slice<viennacl::vector<SCALARTYPE, ALIGNMENT> > const & proxy) : size_(proxy.size())
  {
    if (proxy.size() > 0)
    {
      elements_.switch_active_handle_id(viennacl::traits::handle(proxy).get_active_handle_id());
      viennacl::backend::memory_create(elements_, sizeof(SCALARTYPE)*internal_size());
      pad();
      
      viennacl::linalg::av(*this, 
                            proxy, SCALARTYPE(1.0), 1, false, false);
    }
  }

  template <typename SCALARTYPE, unsigned int ALIGNMENT>
  viennacl::vector<SCALARTYPE, ALIGNMENT>::vector(vector_slice<const viennacl::vector<SCALARTYPE, ALIGNMENT> > const & proxy) : size_(proxy.size())
  {
    if (proxy.size() > 0)
    {
      elements_.switch_active_handle_id(viennacl::traits::handle(proxy).get_active_handle_id());
      viennacl::backend::memory_create(elements_, sizeof(SCALARTYPE)*internal_size());
      pad();
      
      viennacl::linalg::av(*this, 
                            proxy, SCALARTYPE(1.0), 1, false, false);
    }
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
    if (cpu_vector.size() > 0)
    {
      std::vector<SCALARTYPE> temp_buffer(gpu_vector_slice.stride() * gpu_vector_slice.size());
      
      viennacl::backend::memory_read(gpu_vector_slice.get().handle(), sizeof(SCALARTYPE)*gpu_vector_slice.start(), sizeof(SCALARTYPE)*temp_buffer.size(), &(temp_buffer[0]));

      for (std::size_t i=0; i<cpu_vector.size(); ++i)
        temp_buffer[i * gpu_vector_slice.stride()] = cpu_vector[i];
      
      viennacl::backend::memory_write(gpu_vector_slice.get().handle(), sizeof(SCALARTYPE)*gpu_vector_slice.start(), sizeof(SCALARTYPE)*temp_buffer.size(), &(temp_buffer[0]));
    }
  }



  /////////////////////////////////////////////////////////////
  ///////////////////////// GPU to CPU ////////////////////////
  /////////////////////////////////////////////////////////////
  

  template <typename VectorType, typename SCALARTYPE>
  void copy(vector_slice<vector<SCALARTYPE> > const & gpu_vector_slice,
            VectorType & cpu_vector)
  {
    assert(gpu_vector_slice.end() - gpu_vector_slice.begin() >= 0);
    
    if (gpu_vector_slice.end() - gpu_vector_slice.begin() > 0)
    {
      std::vector<SCALARTYPE> temp_buffer(gpu_vector_slice.stride() * gpu_vector_slice.size());
      viennacl::backend::memory_read(gpu_vector_slice.get().handle(), sizeof(SCALARTYPE)*gpu_vector_slice.start(), sizeof(SCALARTYPE)*temp_buffer.size(), &(temp_buffer[0]));

      for (std::size_t i=0; i<cpu_vector.size(); ++i)
        cpu_vector[i] = temp_buffer[i * gpu_vector_slice.stride()];
    }
  }





  //
  // Convenience functions
  //
  template <typename VectorType>
  vector_slice<VectorType> project(VectorType & vec, viennacl::slice const & s1)
  {
    assert(s1.size() <= vec.size() && bool("Size of slice larger than vector size!"));
    return vector_slice<VectorType>(vec, s1);
  }

  template <typename VectorType>
  vector_slice<VectorType> project(viennacl::vector_slice<VectorType> & vec, viennacl::slice const & s1)
  {
    assert(s1.size() <= vec.size() && bool("Size of slice larger than vector proxy!"));
    return vector_slice<VectorType>(vec.get(), viennacl::slice(vec.start() + s1.start(), vec.stride() * s1.stride(), s1.size()));
  }

  // interaction with range and vector_range:
  
  template <typename VectorType>
  vector_slice<VectorType> project(viennacl::vector_slice<VectorType> & vec, viennacl::range const & r1)
  {
    assert(r1.size() <= vec.size() && bool("Size of slice larger than vector proxy!"));
    return vector_slice<VectorType>(vec.get(), viennacl::slice(vec.start() + r1.start(), vec.stride(), r1.size()));
  }
  
  template <typename VectorType>
  vector_slice<VectorType> project(viennacl::vector_range<VectorType> & vec, viennacl::slice const & s1)
  {
    assert(s1.size() <= vec.size() && bool("Size of slice larger than vector proxy!"));
    return vector_slice<VectorType>(vec.get(), viennacl::range(vec.start() + s1.start(), s1.stride(), s1.size()));
  }
  
  
}

#endif
