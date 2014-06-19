#ifndef VIENNACL_VECTOR_DEF_HPP_
#define VIENNACL_VECTOR_DEF_HPP_

/* =========================================================================
   Copyright (c) 2010-2014, Institute for Microelectronics,
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

/** @file  viennacl/vector_def.hpp
    @brief The definition of the implicit_vector_base, vector_base class. Operators are declared, accessors are defined. Use this file to avoid circular dependencies
*/

#include "viennacl/forwards.h"
#include "viennacl/tools/entry_proxy.hpp"

namespace viennacl
{

  /** @brief Common base class for representing vectors where the entries are not all stored explicitly.
    *
    * Typical examples are zero_vector or scalar_vector.
    */
  template<typename SCALARTYPE>
  class implicit_vector_base
  {
    protected:
      implicit_vector_base(vcl_size_t s, vcl_size_t i, SCALARTYPE v, viennacl::context ctx) : size_(s), index_(std::make_pair(true,i)), value_(v), ctx_(ctx){ }
      implicit_vector_base(vcl_size_t s, SCALARTYPE v, viennacl::context ctx) : size_(s), index_(std::make_pair(false,0)), value_(v), ctx_(ctx){ }

    public:
      typedef SCALARTYPE const & const_reference;
      typedef SCALARTYPE cpu_value_type;

      viennacl::context context() const { return ctx_; }
      vcl_size_t size() const { return size_; }
      cpu_value_type  value() const { return value_; }
      vcl_size_t index() const { return index_.second; }
      bool has_index() const { return index_.first; }

      cpu_value_type operator()(vcl_size_t i) const {
        if(index_.first)
          return (i==index_.second)?value_:0;
        return value_;
      }

      cpu_value_type operator[](vcl_size_t i) const {
        if(index_.first)
          return (i==index_.second)?value_:0;
        return
            value_;
      }

    protected:
      vcl_size_t size_;
      std::pair<bool, vcl_size_t> index_;
      SCALARTYPE value_;
      viennacl::context ctx_;
  };

  /** @brief Represents a vector consisting of 1 at a given index and zeros otherwise.*/
  template <typename SCALARTYPE>
  struct unit_vector : public implicit_vector_base<SCALARTYPE>
  {
      unit_vector(vcl_size_t s, vcl_size_t ind, viennacl::context ctx = viennacl::context()) : implicit_vector_base<SCALARTYPE>(s, ind, 1, ctx)
      {
        assert( (ind < s) && bool("Provided index out of range!") );
      }
  };


  /** @brief Represents a vector consisting of scalars 's' only, i.e. v[i] = s for all i. To be used as an initializer for viennacl::vector, vector_range, or vector_slize only. */
  template <typename SCALARTYPE>
  struct scalar_vector : public implicit_vector_base<SCALARTYPE>
  {
    scalar_vector(vcl_size_t s, SCALARTYPE val, viennacl::context ctx = viennacl::context()) : implicit_vector_base<SCALARTYPE>(s, val, ctx) {}
  };

  template <typename SCALARTYPE>
  struct zero_vector : public scalar_vector<SCALARTYPE>
  {
    zero_vector(vcl_size_t s, viennacl::context ctx = viennacl::context()) : scalar_vector<SCALARTYPE>(s, 0, ctx){}
  };


  /** @brief Common base class for dense vectors, vector ranges, and vector slices.
    *
    * @tparam SCALARTYPE   The floating point type, either 'float' or 'double'
    */
  template<class SCALARTYPE, typename SizeType /* see forwards.h for default type */, typename DistanceType /* see forwards.h for default type */>
  class vector_base
  {
      typedef vector_base<SCALARTYPE, SizeType, DistanceType>         self_type;

    public:
      typedef scalar<SCALARTYPE>                                value_type;
      typedef SCALARTYPE                                        cpu_value_type;
      typedef viennacl::backend::mem_handle                     handle_type;
      typedef SizeType                                          size_type;
      typedef DistanceType                                      difference_type;
      typedef const_vector_iterator<SCALARTYPE, 1>              const_iterator;
      typedef vector_iterator<SCALARTYPE, 1>                    iterator;

      static const size_type alignment = 128;

      /** @brief Returns the length of the vector (cf. std::vector)  */
      size_type size() const { return size_; }
      /** @brief Returns the internal length of the vector, which is given by size() plus the extra memory due to padding the memory with zeros up to a multiple of 'ALIGNMENT' */
      size_type internal_size() const { return internal_size_; }
      /** @brief Returns the offset within the buffer  */
      size_type start() const { return start_; }
      /** @brief Returns the stride within the buffer (in multiples of sizeof(SCALARTYPE)) */
      size_type stride() const { return stride_; }
      /** @brief Returns true is the size is zero */
      bool empty() const { return size_ == 0; }
      /** @brief Returns the memory handle. */
      const handle_type & handle() const { return elements_; }
      /** @brief Returns the memory handle. */
      handle_type & handle() { return elements_; }
      viennacl::memory_types memory_domain() const { return elements_.get_active_handle_id();  }

      /** @brief Default constructor in order to be compatible with various containers.
      */
      explicit vector_base();

      /** @brief An explicit constructor for wrapping an existing vector into a vector_range or vector_slice.
       *
       * @param h          The existing memory handle from a vector/vector_range/vector_slice
       * @param vec_size   The length (i.e. size) of the buffer
       * @param vec_start  The offset from the beginning of the buffer identified by 'h'
       * @param vec_stride Increment between two elements in the original buffer (in multiples of SCALARTYPE)
      */
      explicit vector_base(viennacl::backend::mem_handle & h, size_type vec_size, size_type vec_start, difference_type vec_stride);

      /** @brief Creates a vector and allocates the necessary memory */
      explicit vector_base(size_type vec_size, viennacl::context ctx = viennacl::context());

      // CUDA or host memory:
      explicit vector_base(SCALARTYPE * ptr_to_mem, viennacl::memory_types mem_type, size_type vec_size, vcl_size_t start = 0, difference_type stride = 1);

#ifdef VIENNACL_WITH_OPENCL
      /** @brief Create a vector from existing OpenCL memory
      *
      * Note: The provided memory must take an eventual ALIGNMENT into account, i.e. existing_mem must be at least of size internal_size()!
      * This is trivially the case with the default alignment, but should be considered when using vector<> with an alignment parameter not equal to 1.
      *
      * @param existing_mem   An OpenCL handle representing the memory
      * @param vec_size       The size of the vector.
      */
      explicit vector_base(cl_mem existing_mem, size_type vec_size, size_type start = 0, difference_type stride = 1, viennacl::context ctx = viennacl::context());
#endif

      template <typename LHS, typename RHS, typename OP>
      explicit vector_base(vector_expression<const LHS, const RHS, OP> const & proxy);

      /** @brief Assignment operator. Other vector needs to be of the same size, or this vector is not yet initialized.
      */
      self_type & operator=(const self_type & vec);
      /** @brief Implementation of the operation v1 = v2 @ alpha, where @ denotes either multiplication or division, and alpha is either a CPU or a GPU scalar
      * @param proxy  An expression template proxy class.
      */
      template <typename LHS, typename RHS, typename OP>
      self_type & operator=(const vector_expression<const LHS, const RHS, OP> & proxy);
      // assign vector range or vector slice
      template <typename T>
      self_type &  operator = (const vector_base<T> & v1);
      /** @brief Creates the vector from the supplied unit vector. */
      self_type & operator = (unit_vector<SCALARTYPE> const & v);
      /** @brief Creates the vector from the supplied zero vector. */
      self_type & operator = (zero_vector<SCALARTYPE> const & v);
      /** @brief Creates the vector from the supplied scalar vector. */
      self_type & operator = (scalar_vector<SCALARTYPE> const & v);


      ///////////////////////////// Matrix Vector interaction start ///////////////////////////////////
      /** @brief Operator overload for v1 = A * v2, where v1, v2 are vectors and A is a dense matrix.
      * @param proxy An expression template proxy class
      */
      self_type & operator=(const viennacl::vector_expression< const matrix_base<SCALARTYPE>, const vector_base<SCALARTYPE>, viennacl::op_prod> & proxy);

      //transposed_matrix_proxy:
      /** @brief Operator overload for v1 = trans(A) * v2, where v1, v2 are vectors and A is a dense matrix.
      * @param proxy An expression template proxy class
      */
      self_type & operator=(const vector_expression< const matrix_expression< const matrix_base<SCALARTYPE>, const matrix_base<SCALARTYPE>, op_trans >,
                                                     const vector_base<SCALARTYPE>,
                                                     op_prod> & proxy);

      ///////////////////////////// Matrix Vector interaction end ///////////////////////////////////


      //read-write access to an element of the vector
      /** @brief Read-write access to a single element of the vector */
      entry_proxy<SCALARTYPE> operator()(size_type index);
      /** @brief Read-write access to a single element of the vector */
      entry_proxy<SCALARTYPE> operator[](size_type index);
      /** @brief Read access to a single element of the vector */
      const_entry_proxy<SCALARTYPE> operator()(size_type index) const;
      /** @brief Read access to a single element of the vector */
      const_entry_proxy<SCALARTYPE> operator[](size_type index) const;
      self_type & operator += (const self_type & vec);
      self_type & operator -= (const self_type & vec);
      template <typename LHS, typename RHS, typename OP>
      self_type & operator += (const vector_expression<const LHS, const RHS, OP> & proxy);
      template <typename LHS, typename RHS, typename OP>
      self_type & operator -= (const vector_expression<const LHS, const RHS, OP> & proxy);
      /** @brief Scales a vector (or proxy) by a CPU scalar value */
      self_type & operator *= (SCALARTYPE val);
      /** @brief Scales this vector by a CPU scalar value */
      self_type & operator /= (SCALARTYPE val);
      /** @brief Scales the vector by a CPU scalar 'alpha' and returns an expression template */
      vector_expression< const self_type, const SCALARTYPE, op_mult>
      operator * (SCALARTYPE value) const;
      /** @brief Scales the vector by a CPU scalar 'alpha' and returns an expression template */
      vector_expression< const self_type, const SCALARTYPE, op_div>
      operator / (SCALARTYPE value) const;
      /** @brief Sign flip for the vector. Emulated to be equivalent to -1.0 * vector */
      vector_expression<const self_type, const SCALARTYPE, op_mult> operator-() const;
      /** @brief Returns an iterator pointing to the beginning of the vector  (STL like)*/
      iterator begin();
      /** @brief Returns an iterator pointing to the end of the vector (STL like)*/
      iterator end();
      /** @brief Returns a const-iterator pointing to the beginning of the vector (STL like)*/
      const_iterator begin() const;
      /** @brief Returns a const-iterator pointing to the end of the vector (STL like)*/
      const_iterator end() const;
      /** @brief Swaps the entries of the two vectors */
      self_type & swap(self_type & other);

      /** @brief Resets all entries to zero. Does not change the size of the vector. */
      void clear();

    protected:

      void set_handle(viennacl::backend::mem_handle const & h) {  elements_ = h; }

      /** @brief Swaps the handles of two vectors by swapping the OpenCL handles only, no data copy */
      self_type & fast_swap(self_type & other);

      /** @brief Pads vectors with alignment > 1 with trailing zeros if the internal size is larger than the visible size */
      void pad();

      void switch_memory_context(viennacl::context new_ctx);

      //TODO: Think about implementing the following public member functions
      //void insert_element(unsigned int i, SCALARTYPE val){}
      //void erase_element(unsigned int i){}

      //enlarge or reduce allocated memory and set unused memory to zero
      /** @brief Resizes the allocated memory for the vector. Pads the memory to be a multiple of 'ALIGNMENT'
      *
      *  @param new_size  The new size of the vector
      *  @param preserve  If true, old entries of the vector are preserved, otherwise eventually discarded.
      */
      void resize(size_type new_size, bool preserve = true);

      /** @brief Resizes the allocated memory for the vector. Convenience function for setting an OpenCL context in case reallocation is needed
      *
      *  @param new_size  The new size of the vector
      *  @param ctx       The context within which the new memory should be allocated
      *  @param preserve  If true, old entries of the vector are preserved, otherwise eventually discarded.
      */
      void resize(size_type new_size, viennacl::context ctx, bool preserve = true);
    private:

      void resize_impl(size_type new_size, viennacl::context ctx, bool preserve = true);

      size_type       size_;
      size_type       start_;
      difference_type stride_;
      size_type       internal_size_;
      handle_type elements_;
  }; //vector_base

  /** \endcond */

} // namespace viennacl

#endif
