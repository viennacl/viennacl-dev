#ifndef VIENNACL_FORWARDS_H
#define VIENNACL_FORWARDS_H

/* =========================================================================
   Copyright (c) 2010-2013, Institute for Microelectronics,
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


/** @file viennacl/forwards.h
    @brief This file provides the forward declarations for the main types used within ViennaCL
*/

/**
 @mainpage Source Code Documentation for ViennaCL 1.5.0

 This is the source code documentation of ViennaCL. Detailed information about the functions in ViennaCL can be found here.

 For a general overview over the types and functionality provided by ViennaCL, please refer to the file doc/viennacl.pdf

*/


//compatibility defines:
#ifdef VIENNACL_HAVE_UBLAS
  #define VIENNACL_WITH_UBLAS
#endif

#ifdef VIENNACL_HAVE_EIGEN
  #define VIENNACL_WITH_EIGEN
#endif

#ifdef VIENNACL_HAVE_MTL4
  #define VIENNACL_WITH_MTL4
#endif

#include <cstddef>
#include <cassert>
#include "viennacl/meta/enable_if.hpp"

/** @brief Main namespace in ViennaCL. Holds all the basic types such as vector, matrix, etc. and defines operations upon them. */
namespace viennacl
{
  typedef std::size_t                                       vcl_size_t;
  typedef std::ptrdiff_t                                    vcl_ptrdiff_t;
#ifdef _MSC_VER
#ifdef  _WIN64
  typedef __int64  omp_index;
#else
  typedef _W64 int omp_index;
#endif
#else
  typedef std::size_t omp_index;
#endif


  /** @brief A tag class representing assignment */
  struct op_assign {};
  /** @brief A tag class representing inplace addition */
  struct op_inplace_add {};
  /** @brief A tag class representing inplace subtraction */
  struct op_inplace_sub {};

  /** @brief A tag class representing addition */
  struct op_add {};
  /** @brief A tag class representing subtraction */
  struct op_sub {};
  /** @brief A tag class representing multiplication by a scalar */
  struct op_mult {};
  /** @brief A tag class representing matrix-vector products and element-wise multiplications*/
  struct op_prod {};
  /** @brief A tag class representing matrix-matrix products */
  struct op_mat_mat_prod {};
  /** @brief A tag class representing division */
  struct op_div {};
  /** @brief A tag class representing the power function */
  struct op_pow {};

  /** @brief A tag class representing element-wise binary operations (like multiplication) on vectors or matrices */
  template <typename OP>
  struct op_element_binary {};

  /** @brief A tag class representing element-wise unary operations (like sin()) on vectors or matrices */
  template <typename OP>
  struct op_element_unary {};

  struct op_abs {};
  struct op_acos {};
  struct op_asin {};
  struct op_atan {};
  struct op_atan2 {};
  struct op_ceil {};
  struct op_cos {};
  struct op_cosh {};
  struct op_exp {};
  struct op_fabs {};
  struct op_fdim {};
  struct op_floor {};
  struct op_fmax {};
  struct op_fmin {};
  struct op_fmod {};
  struct op_log {};
  struct op_log10 {};
  struct op_sin {};
  struct op_sinh {};
  struct op_sqrt {};
  struct op_tan {};
  struct op_tanh {};

  /** @brief A tag class representing the (off-)diagonal of a matrix */
  struct op_matrix_diag {};

  /** @brief A tag class representing a matrix given by a vector placed on a certain (off-)diagonal */
  struct op_vector_diag {};

  /** @brief A tag class representing the extraction of a matrix row to a vector */
  struct op_row {};

  /** @brief A tag class representing the extraction of a matrix column to a vector */
  struct op_column {};

  /** @brief A tag class representing inner products of two vectors */
  struct op_inner_prod {};

  /** @brief A tag class representing the 1-norm of a vector */
  struct op_norm_1 {};

  /** @brief A tag class representing the 2-norm of a vector */
  struct op_norm_2 {};

  /** @brief A tag class representing the inf-norm of a vector */
  struct op_norm_inf {};

  /** @brief A tag class representing the Frobenius-norm of a matrix */
  struct op_norm_frobenius {};

  /** @brief A tag class representing transposed matrices */
  struct op_trans {};

  /** @brief A tag class representing sign flips (for scalars only. Vectors and matrices use the standard multiplication by the scalar -1.0) */
  struct op_flip_sign {};

  //forward declaration of basic types:
  template<class TYPE>
  class scalar;

  template <typename LHS, typename RHS, typename OP>
  class scalar_expression;

  template <typename SCALARTYPE>
  class entry_proxy;

  template <typename LHS, typename RHS, typename OP>
  class vector_expression;

  template<class SCALARTYPE, unsigned int ALIGNMENT>
  class vector_iterator;

  template<class SCALARTYPE, unsigned int ALIGNMENT>
  class const_vector_iterator;

  template<typename SCALARTYPE>
  class implicit_vector_base;

  template <typename SCALARTYPE>
  class zero_vector;

  template <typename SCALARTYPE>
  class unit_vector;

  template <typename SCALARTYPE>
  class one_vector;

  template <typename SCALARTYPE>
  class scalar_vector;

  template<class SCALARTYPE, typename SizeType = vcl_size_t, typename DistanceType = vcl_ptrdiff_t>
  class vector_base;

  template<class SCALARTYPE, unsigned int ALIGNMENT = 1>
  class vector;

  template <typename ScalarT>
  class vector_tuple;

  //the following forwards are needed for GMRES
  template <typename SCALARTYPE, unsigned int ALIGNMENT, typename CPU_ITERATOR>
  void copy(CPU_ITERATOR const & cpu_begin,
            CPU_ITERATOR const & cpu_end,
            vector_iterator<SCALARTYPE, ALIGNMENT> gpu_begin);

  template <typename SCALARTYPE, unsigned int ALIGNMENT_SRC, unsigned int ALIGNMENT_DEST>
  void copy(const_vector_iterator<SCALARTYPE, ALIGNMENT_SRC> const & gpu_src_begin,
            const_vector_iterator<SCALARTYPE, ALIGNMENT_SRC> const & gpu_src_end,
            vector_iterator<SCALARTYPE, ALIGNMENT_DEST> gpu_dest_begin);

  template <typename SCALARTYPE, unsigned int ALIGNMENT_SRC, unsigned int ALIGNMENT_DEST>
  void copy(const_vector_iterator<SCALARTYPE, ALIGNMENT_SRC> const & gpu_src_begin,
            const_vector_iterator<SCALARTYPE, ALIGNMENT_SRC> const & gpu_src_end,
            const_vector_iterator<SCALARTYPE, ALIGNMENT_DEST> gpu_dest_begin);

  template <typename SCALARTYPE, unsigned int ALIGNMENT, typename CPU_ITERATOR>
  void fast_copy(const const_vector_iterator<SCALARTYPE, ALIGNMENT> & gpu_begin,
                 const const_vector_iterator<SCALARTYPE, ALIGNMENT> & gpu_end,
                 CPU_ITERATOR cpu_begin );

  template <typename CPU_ITERATOR, typename SCALARTYPE, unsigned int ALIGNMENT>
  void fast_copy(CPU_ITERATOR const & cpu_begin,
                  CPU_ITERATOR const & cpu_end,
                  vector_iterator<SCALARTYPE, ALIGNMENT> gpu_begin);


  struct row_major_tag {};
  struct column_major_tag {};

  /** @brief A tag for row-major storage of a dense matrix. */
  struct row_major
  {
    typedef row_major_tag         orientation_category;

    /** @brief Returns the memory offset for entry (i,j) of a dense matrix.
    *
    * @param i   row index
    * @param j   column index
    * @param num_cols  number of entries per column (including alignment)
    */
    static vcl_size_t mem_index(vcl_size_t i, vcl_size_t j, vcl_size_t /* num_rows */, vcl_size_t num_cols)
    {
      return i * num_cols + j;
    }
  };

  struct column_major
  {
    typedef column_major_tag         orientation_category;

    /** @brief Returns the memory offset for entry (i,j) of a dense matrix.
    *
    * @param i   row index
    * @param j   column index
    * @param num_rows  number of entries per row (including alignment)
    */
    static vcl_size_t mem_index(vcl_size_t i, vcl_size_t j, vcl_size_t num_rows, vcl_size_t /* num_cols */)
    {
      return i + j * num_rows;
    }
  };

  struct row_iteration;
  struct col_iteration;

  template <typename LHS, typename RHS, typename OP>
  class matrix_expression;

  //
  // Matrix types:
  //

  template<class SCALARTYPE, typename F = row_major, typename SizeType = vcl_size_t, typename DistanceType = vcl_ptrdiff_t>
  class matrix_base;

  template <class SCALARTYPE, typename F = row_major, unsigned int ALIGNMENT = 1>
  class matrix;

  template<typename SCALARTYPE>
  class implicit_matrix_base;

  template <class SCALARTYPE>
  class identity_matrix;

  template <class SCALARTYPE>
  class zero_matrix;

  template <class SCALARTYPE>
  class scalar_matrix;

  template<class SCALARTYPE, unsigned int ALIGNMENT = 1>
  class compressed_matrix;

  template<class SCALARTYPE>
  class compressed_compressed_matrix;


  template<class SCALARTYPE, unsigned int ALIGNMENT = 128>
  class coordinate_matrix;

  template<class SCALARTYPE, unsigned int ALIGNMENT = 1>
  class ell_matrix;

  template<class SCALARTYPE, unsigned int ALIGNMENT = 1>
  class hyb_matrix;

  template<class SCALARTYPE, unsigned int ALIGNMENT = 1>
  class circulant_matrix;

  template<class SCALARTYPE, unsigned int ALIGNMENT = 1>
  class hankel_matrix;

  template<class SCALARTYPE, unsigned int ALIGNMENT = 1>
  class toeplitz_matrix;

  template<class SCALARTYPE, unsigned int ALIGNMENT = 1>
  class vandermonde_matrix;

  //
  // Proxies:
  //
  template <typename SizeType = std::size_t, typename DistanceType = std::ptrdiff_t>
  class basic_range;

  typedef basic_range<>  range;

  template <typename SizeType = std::size_t, typename DistanceType = std::ptrdiff_t>
  class basic_slice;

  typedef basic_slice<>  slice;

  template <typename VectorType>
  class vector_range;

  template <typename VectorType>
  class vector_slice;

  template <typename MatrixType>
  class matrix_range;

  template <typename MatrixType>
  class matrix_slice;


  template <typename T>
  struct is_cpu_scalar
  {
    enum { value = false };
  };

  template <typename T>
  struct is_scalar
  {
    enum { value = false };
  };

  template <typename T>
  struct is_flip_sign_scalar
  {
    enum { value = false };
  };

  template <typename T>
  struct is_any_scalar
  {
    enum { value = (is_scalar<T>::value || is_cpu_scalar<T>::value || is_flip_sign_scalar<T>::value )};
  };

  template <typename T>
  struct is_row_major
  {
    enum { value = false };
  };

  template <typename T>
  struct is_any_sparse_matrix
  {
    enum { value = false };
  };


  template <typename T>
  struct is_circulant_matrix
  {
    enum { value = false };
  };

  template <typename T>
  struct is_hankel_matrix
  {
    enum { value = false };
  };

  template <typename T>
  struct is_toeplitz_matrix
  {
    enum { value = false };
  };

  template <typename T>
  struct is_vandermonde_matrix
  {
    enum { value = false };
  };

  template <typename T>
  struct is_any_dense_structured_matrix
  {
    enum { value = viennacl::is_circulant_matrix<T>::value || viennacl::is_hankel_matrix<T>::value || viennacl::is_toeplitz_matrix<T>::value || viennacl::is_vandermonde_matrix<T>::value };
  };


  enum memory_types
  {
    MEMORY_NOT_INITIALIZED
    , MAIN_MEMORY
    , OPENCL_MEMORY
    , CUDA_MEMORY
  };

  /** @brief Exception class in case of memory errors */
  class memory_exception : public std::exception
  {
  public:
    memory_exception() : message_() {}
    memory_exception(std::string message) : message_("ViennaCL: Internal memory error: " + message) {}

    virtual const char* what() const throw() { return message_.c_str(); }

    virtual ~memory_exception() throw() {}
  private:
    std::string message_;
  };


  class context;

  namespace tools
  {
    //helper for matrix row/col iterators
    //must be specialized for every viennacl matrix type
    template <typename ROWCOL, typename MATRIXTYPE>
    struct MATRIX_ITERATOR_INCREMENTER
    {
      typedef typename MATRIXTYPE::ERROR_SPECIALIZATION_FOR_THIS_MATRIX_TYPE_MISSING          ErrorIndicator;

      static void apply(const MATRIXTYPE & /*mat*/, unsigned int & /*row*/, unsigned int & /*col*/) {}
    };
  }

  namespace linalg
  {
#if !defined(_MSC_VER) || defined(__CUDACC__)

    template<class SCALARTYPE, unsigned int ALIGNMENT>
    void convolve_i(viennacl::vector<SCALARTYPE, ALIGNMENT>& input1,
                    viennacl::vector<SCALARTYPE, ALIGNMENT>& input2,
                    viennacl::vector<SCALARTYPE, ALIGNMENT>& output);

    template <typename T>
    viennacl::vector_expression<const vector_base<T>, const vector_base<T>, op_element_binary<op_prod> >
    element_prod(vector_base<T> const & v1, vector_base<T> const & v2);

    template <typename T>
    viennacl::vector_expression<const vector_base<T>, const vector_base<T>, op_element_binary<op_div> >
    element_div(vector_base<T> const & v1, vector_base<T> const & v2);



    template <typename T>
    void inner_prod_impl(vector_base<T> const & vec1,
                         vector_base<T> const & vec2,
                         scalar<T> & result);

    template <typename LHS, typename RHS, typename OP, typename T>
    void inner_prod_impl(viennacl::vector_expression<LHS, RHS, OP> const & vec1,
                         vector_base<T> const & vec2,
                         scalar<T> & result);

    template <typename T, typename LHS, typename RHS, typename OP>
    void inner_prod_impl(vector_base<T> const & vec1,
                         viennacl::vector_expression<LHS, RHS, OP> const & vec2,
                         scalar<T> & result);

    template <typename LHS1, typename RHS1, typename OP1,
              typename LHS2, typename RHS2, typename OP2, typename T>
    void inner_prod_impl(viennacl::vector_expression<LHS1, RHS1, OP1> const & vec1,
                         viennacl::vector_expression<LHS2, RHS2, OP2> const & vec2,
                         scalar<T> & result);

    ///////////////////////////

    template <typename T>
    void inner_prod_cpu(vector_base<T> const & vec1,
                        vector_base<T> const & vec2,
                        T & result);

    template <typename LHS, typename RHS, typename OP, typename T>
    void inner_prod_cpu(viennacl::vector_expression<LHS, RHS, OP> const & vec1,
                        vector_base<T> const & vec2,
                        T & result);

    template <typename T, typename LHS, typename RHS, typename OP>
    void inner_prod_cpu(vector_base<T> const & vec1,
                        viennacl::vector_expression<LHS, RHS, OP> const & vec2,
                        T & result);

    template <typename LHS1, typename RHS1, typename OP1,
              typename LHS2, typename RHS2, typename OP2, typename S3>
    void inner_prod_cpu(viennacl::vector_expression<LHS1, RHS1, OP1> const & vec1,
                        viennacl::vector_expression<LHS2, RHS2, OP2> const & vec2,
                        S3 & result);



    //forward definition of norm_1_impl function
    template <typename T>
    void norm_1_impl(vector_base<T> const & vec, scalar<T> & result);

    //template <typename T, typename F>
    //void norm_1_impl(matrix_base<T, F> const & A, scalar<T> & result);

    template <typename LHS, typename RHS, typename OP, typename T>
    void norm_1_impl(viennacl::vector_expression<LHS, RHS, OP> const & vec,
                     scalar<T> & result);


    template <typename T>
    void norm_1_cpu(vector_base<T> const & vec,
                    T & result);

    //template <typename T, typename F>
    //void norm_1_cpu(matrix_base<T, F> const & vec,
    //                T & result);

    template <typename LHS, typename RHS, typename OP, typename S2>
    void norm_1_cpu(viennacl::vector_expression<LHS, RHS, OP> const & vec,
                    S2 & result);

    //forward definition of norm_2_impl function
    template <typename T>
    void norm_2_impl(vector_base<T> const & vec, scalar<T> & result);

    template <typename LHS, typename RHS, typename OP, typename T>
    void norm_2_impl(viennacl::vector_expression<LHS, RHS, OP> const & vec,
                     scalar<T> & result);

    template <typename T>
    void norm_2_cpu(vector_base<T> const & vec, T & result);

    template <typename LHS, typename RHS, typename OP, typename S2>
    void norm_2_cpu(viennacl::vector_expression<LHS, RHS, OP> const & vec,
                    S2 & result);


    //forward definition of norm_inf_impl function
    template <typename T>
    void norm_inf_impl(vector_base<T> const & vec, scalar<T> & result);

    //template <typename T, typename F>
    //void norm_inf_impl(matrix_base<T, F> const & vec, scalar<T> & result);

    template <typename LHS, typename RHS, typename OP, typename T>
    void norm_inf_impl(viennacl::vector_expression<LHS, RHS, OP> const & vec,
                      scalar<T> & result);


    template <typename T>
    void norm_inf_cpu(vector_base<T> const & vec, T & result);

    //template <typename T, typename F>
    //void norm_inf_cpu(matrix_base<T, F> const & vec, T & result);

    template <typename LHS, typename RHS, typename OP, typename S2>
    void norm_inf_cpu(viennacl::vector_expression<LHS, RHS, OP> const & vec,
                      S2 & result);

    template <typename T, typename F>
    void norm_frobenius_impl(matrix_base<T, F> const & vec, scalar<T> & result);

    template <typename T, typename F>
    void norm_frobenius_cpu(matrix_base<T, F> const & vec, T & result);


    template <typename T>
    std::size_t index_norm_inf(vector_base<T> const & vec);

    template <typename LHS, typename RHS, typename OP>
    std::size_t index_norm_inf(viennacl::vector_expression<LHS, RHS, OP> const & vec);

    //forward definition of prod_impl functions

    template <typename NumericT, typename F>
    void prod_impl(const matrix_base<NumericT, F> & mat,
                   const vector_base<NumericT> & vec,
                         vector_base<NumericT> & result);

    template <typename NumericT, typename F>
    void prod_impl(const matrix_expression< const matrix_base<NumericT, F>, const matrix_base<NumericT, F>, op_trans> & mat_trans,
                   const vector_base<NumericT> & vec,
                         vector_base<NumericT> & result);

    template<typename SparseMatrixType, class SCALARTYPE, unsigned int ALIGNMENT>
    typename viennacl::enable_if< viennacl::is_any_sparse_matrix<SparseMatrixType>::value,
                                  vector_expression<const SparseMatrixType,
                                                    const vector<SCALARTYPE, ALIGNMENT>,
                                                    op_prod >
                                 >::type
    prod_impl(const SparseMatrixType & mat,
              const vector<SCALARTYPE, ALIGNMENT> & vec);
#endif

    namespace detail
    {
      enum row_info_types
      {
        SPARSE_ROW_NORM_INF = 0,
        SPARSE_ROW_NORM_1,
        SPARSE_ROW_NORM_2,
        SPARSE_ROW_DIAGONAL
      };

    }


    /** @brief A tag class representing a lower triangular matrix */
    struct lower_tag
    {
      static const char * name() { return "lower"; }
    };      //lower triangular matrix
    /** @brief A tag class representing an upper triangular matrix */
    struct upper_tag
    {
      static const char * name() { return "upper"; }
    };      //upper triangular matrix
    /** @brief A tag class representing a lower triangular matrix with unit diagonal*/
    struct unit_lower_tag
    {
      static const char * name() { return "unit_lower"; }
    }; //unit lower triangular matrix
    /** @brief A tag class representing an upper triangular matrix with unit diagonal*/
    struct unit_upper_tag
    {
      static const char * name() { return "unit_upper"; }
    }; //unit upper triangular matrix

    //preconditioner tags
    class ilut_tag;

    /** @brief A tag class representing the use of no preconditioner */
    class no_precond
    {
      public:
        template <typename VectorType>
        void apply(VectorType &) const {}
    };


  } //namespace linalg

  //
  // More namespace comments to follow:
  //

  /** @brief Namespace providing routines for handling the different memory domains. */
  namespace backend
  {
    /** @brief Provides implementations for handling memory buffers in CPU RAM. */
    namespace cpu_ram
    {
      /** @brief Holds implementation details for handling memory buffers in CPU RAM. Not intended for direct use by library users. */
      namespace detail {}
    }

    /** @brief Provides implementations for handling CUDA memory buffers. */
    namespace cuda
    {
      /** @brief Holds implementation details for handling CUDA memory buffers. Not intended for direct use by library users. */
      namespace detail {}
    }

    /** @brief Implementation details for the generic memory backend interface. */
    namespace detail {}

    /** @brief Provides implementations for handling OpenCL memory buffers. */
    namespace opencl
    {
      /** @brief Holds implementation details for handling OpenCL memory buffers. Not intended for direct use by library users. */
      namespace detail {}
    }
  }


  /** @brief Holds implementation details for functionality in the main viennacl-namespace. Not intended for direct use by library users. */
  namespace detail
  {
    /** @brief Helper namespace for fast Fourier transforms. Not to be used directly by library users. */
    namespace fft
    {
      /** @brief Helper namespace for fast-Fourier transformation. Deprecated. */
      namespace FFT_DATA_ORDER {}
    }
  }


  /** @brief Provides an OpenCL kernel generator. */
  namespace generator
  {
    /** @brief Namespace holding unary math functions for use within the kernel generator. */
    namespace math {}

    /** @brief Contains all the meta-functions used within the OpenCL kernel generator. */
    namespace result_of {}

    /** @brief Contains helper routines for manipulating expression trees. */
    namespace tree_utils {}

    /** @brief Contains helper routines for manipulating typelists. */
    namespace typelist_utils {}
  }

  /** @brief Provides basic input-output functionality. */
  namespace io
  {
    /** @brief Implementation details for IO functionality. Usually not of interest for a library user. */
    namespace detail {}

    /** @brief Namespace holding the various XML tag definitions for the kernel parameter tuning facility. */
    namespace tag {}

    /** @brief Namespace holding the various XML strings for the kernel parameter tuning facility. */
    namespace val {}
  }

  /** @brief Provides all linear algebra operations which are not covered by operator overloads. */
  namespace linalg
  {
    /** @brief Holds all CUDA compute kernels used by ViennaCL. */
    namespace cuda
    {
      /** @brief Helper functions for the CUDA linear algebra backend. */
      namespace detail {}
    }

    /** @brief Namespace holding implementation details for linear algebra routines. Usually not of interest for a library user. */
    namespace detail
    {
      /** @brief Implementation namespace for algebraic multigrid preconditioner. */
      namespace amg {}

      /** @brief Implementation namespace for sparse approximate inverse preconditioner. */
      namespace spai {}
    }

    /** @brief Holds all compute kernels with conventional host-based execution (buffers in CPU RAM). */
    namespace host_based
    {
      /** @brief Helper functions for the host-based linear algebra backend. */
      namespace detail {}
    }

    /** @brief Namespace containing the OpenCL kernels. Deprecated, will be moved to viennacl::linalg::opencl in future releases. */
    namespace kernels {}

    /** @brief Holds all routines providing OpenCL linear algebra operations. */
    namespace opencl
    {
      /** @brief Helper functions for OpenCL-accelerated linear algebra operations. */
      namespace detail {}
    }
  }

  /** @brief OpenCL backend. Manages platforms, contexts, buffers, kernels, etc. */
  namespace ocl {}

  /** @brief Namespace containing many meta-functions. */
  namespace result_of {}

  /** @brief Namespace for various tools used within ViennaCL. */
  namespace tools
  {
    /** @brief Contains implementation details for the tools. Usually not of interest for the library user. */
    namespace detail {}
  }

  /** @brief Namespace providing traits-information as well as generic wrappers to common routines for vectors and matrices such as size() or clear() */
  namespace traits {}

} //namespace viennacl

#endif

/*@}*/
