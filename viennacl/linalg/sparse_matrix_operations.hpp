#ifndef VIENNACL_LINALG_SPARSE_MATRIX_OPERATIONS_HPP_
#define VIENNACL_LINALG_SPARSE_MATRIX_OPERATIONS_HPP_

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

/** @file viennacl/linalg/sparse_matrix_operations.hpp
    @brief Implementations of operations using sparse matrices
*/

#include "viennacl/forwards.h"
#include "viennacl/scalar.hpp"
#include "viennacl/vector.hpp"
#include "viennacl/matrix.hpp"
#include "viennacl/tools/tools.hpp"
#include "viennacl/linalg/host_based/sparse_matrix_operations.hpp"

#ifdef VIENNACL_WITH_OPENCL
  #include "viennacl/linalg/opencl/sparse_matrix_operations.hpp"
#endif

#ifdef VIENNACL_WITH_CUDA
  #include "viennacl/linalg/cuda/sparse_matrix_operations.hpp"
#endif

namespace viennacl
{
  namespace linalg
  {
    
    namespace detail
    {
      
      template<typename SparseMatrixType, typename SCALARTYPE, unsigned int VEC_ALIGNMENT>
      typename viennacl::enable_if< viennacl::is_any_sparse_matrix<SparseMatrixType>::value >::type
      row_info(SparseMatrixType const & mat,
               vector<SCALARTYPE, VEC_ALIGNMENT> & vec,
               row_info_types info_selector)
      {
        switch (viennacl::traits::handle(mat).get_active_handle_id())
        {
          case viennacl::MAIN_MEMORY:
            viennacl::linalg::host_based::detail::row_info(mat, vec, info_selector);
            break;
#ifdef VIENNACL_WITH_OPENCL
          case viennacl::OPENCL_MEMORY:
            viennacl::linalg::opencl::detail::row_info(mat, vec, info_selector);
            break;
#endif
#ifdef VIENNACL_WITH_CUDA
          case viennacl::CUDA_MEMORY:
            viennacl::linalg::cuda::detail::row_info(mat, vec, info_selector);
            break;
#endif
          default:
            throw "not implemented";
        }
      }
    
    }
    
    
    
    // A * x
    /** @brief Returns a proxy class that represents matrix-vector multiplication with any sparse matrix type
    *
    * This is used for the convenience expression result = prod(mat, vec);
    *
    * @param mat    The matrix
    * @param vec    The vector
    */
    template<typename SparseMatrixType, class SCALARTYPE, unsigned int ALIGNMENT>
    typename viennacl::enable_if< viennacl::is_any_sparse_matrix<SparseMatrixType>::value,
                                  vector_expression<const SparseMatrixType,
                                                    const vector<SCALARTYPE, ALIGNMENT>, 
                                                    op_prod >
                                 >::type
    prod_impl(const SparseMatrixType & mat, 
              const vector<SCALARTYPE, ALIGNMENT> & vec)
    {
      return vector_expression<const SparseMatrixType,
                               const vector<SCALARTYPE, ALIGNMENT>, 
                               op_prod >(mat, vec);
    }
    
    
    /** @brief Carries out matrix-vector multiplication involving a sparse matrix type
    *
    * Implementation of the convenience expression result = prod(mat, vec);
    *
    * @param mat    The matrix
    * @param vec    The vector
    * @param result The result vector
    */
    template<typename SparseMatrixType, class ScalarType, unsigned int ALIGNMENT>
    typename viennacl::enable_if< viennacl::is_any_sparse_matrix<SparseMatrixType>::value>::type
    prod_impl(const SparseMatrixType & mat, 
              const viennacl::vector<ScalarType, ALIGNMENT> & vec,
                    viennacl::vector<ScalarType, ALIGNMENT> & result)
    {
      assert( (mat.size1() == result.size()) && bool("Size check failed for compressed matrix-vector product: size1(mat) != size(result)"));
      assert( (mat.size2() == vec.size())    && bool("Size check failed for compressed matrix-vector product: size2(mat) != size(x)"));

      switch (viennacl::traits::handle(mat).get_active_handle_id())
      {
        case viennacl::MAIN_MEMORY:
          viennacl::linalg::host_based::prod_impl(mat, vec, result);
          break;
#ifdef VIENNACL_WITH_OPENCL
        case viennacl::OPENCL_MEMORY:
          viennacl::linalg::opencl::prod_impl(mat, vec, result);
          break;
#endif
#ifdef VIENNACL_WITH_CUDA
        case viennacl::CUDA_MEMORY:
          viennacl::linalg::cuda::prod_impl(mat, vec, result);
          break;
#endif
        default:
          throw "not implemented";
      }
    }
    
    
    /** @brief Carries out triangular inplace solves
    *
    * @param mat    The matrix
    * @param vec    The vector
    * @param tag    The solver tag (lower_tag, unit_lower_tag, unit_upper_tag, or upper_tag)
    */
    template<typename SparseMatrixType, class ScalarType, unsigned int ALIGNMENT, typename SOLVERTAG>
    typename viennacl::enable_if< viennacl::is_any_sparse_matrix<SparseMatrixType>::value>::type
    inplace_solve(const SparseMatrixType & mat, 
                  viennacl::vector<ScalarType, ALIGNMENT> & vec,
                  SOLVERTAG tag)
    {
      assert( (mat.size1() == mat.size2()) && bool("Size check failed for triangular solve on compressed matrix: size1(mat) != size2(mat)"));
      assert( (mat.size2() == vec.size())    && bool("Size check failed for compressed matrix-vector product: size2(mat) != size(x)"));

      switch (viennacl::traits::handle(mat).get_active_handle_id())
      {
        case viennacl::MAIN_MEMORY:
          viennacl::linalg::host_based::inplace_solve(mat, vec, tag);
          break;
#ifdef VIENNACL_WITH_OPENCL
        case viennacl::OPENCL_MEMORY:
          viennacl::linalg::opencl::inplace_solve(mat, vec, tag);
          break;
#endif
#ifdef VIENNACL_WITH_CUDA
        case viennacl::CUDA_MEMORY:
          viennacl::linalg::cuda::inplace_solve(mat, vec, tag);
          break;
#endif
        default:
          throw "not implemented";
      }
    }
    
    
    /** @brief Carries out transposed triangular inplace solves
    *
    * @param mat    The matrix
    * @param vec    The vector
    * @param tag    The solver tag (lower_tag, unit_lower_tag, unit_upper_tag, or upper_tag)
    */
    template<typename SparseMatrixType, class ScalarType, unsigned int ALIGNMENT, typename SOLVERTAG>
    typename viennacl::enable_if< viennacl::is_any_sparse_matrix<SparseMatrixType>::value>::type
    inplace_solve(const matrix_expression<const SparseMatrixType, const SparseMatrixType, op_trans> & mat, 
                  viennacl::vector<ScalarType, ALIGNMENT> & vec,
                  SOLVERTAG tag)
    {
      assert( (mat.size1() == mat.size2()) && bool("Size check failed for triangular solve on transposed compressed matrix: size1(mat) != size2(mat)"));
      assert( (mat.size1() == vec.size())    && bool("Size check failed for transposed compressed matrix triangular solve: size1(mat) != size(x)"));

      switch (viennacl::traits::handle(mat.lhs()).get_active_handle_id())
      {
        case viennacl::MAIN_MEMORY:
          viennacl::linalg::host_based::inplace_solve(mat, vec, tag);
          break;
#ifdef VIENNACL_WITH_OPENCL
        case viennacl::OPENCL_MEMORY:
          viennacl::linalg::opencl::inplace_solve(mat, vec, tag);
          break;
#endif
#ifdef VIENNACL_WITH_CUDA
        case viennacl::CUDA_MEMORY:
          viennacl::linalg::cuda::inplace_solve(mat, vec, tag);
          break;
#endif
        default:
          throw "not implemented";
      }
    }
    

    
    namespace detail
    {
      
      template<typename SparseMatrixType, class ScalarType, unsigned int ALIGNMENT, typename SOLVERTAG>
      typename viennacl::enable_if< viennacl::is_any_sparse_matrix<SparseMatrixType>::value>::type
      block_inplace_solve(const matrix_expression<const SparseMatrixType, const SparseMatrixType, op_trans> & mat, 
                          viennacl::backend::mem_handle const & block_index_array, std::size_t num_blocks,
                          viennacl::vector<ScalarType> const & mat_diagonal,
                          viennacl::vector<ScalarType, ALIGNMENT> & vec,
                          SOLVERTAG tag)
      {
        assert( (mat.size1() == mat.size2()) && bool("Size check failed for triangular solve on transposed compressed matrix: size1(mat) != size2(mat)"));
        assert( (mat.size1() == vec.size())  && bool("Size check failed for transposed compressed matrix triangular solve: size1(mat) != size(x)"));

        switch (viennacl::traits::handle(mat.lhs()).get_active_handle_id())
        {
          case viennacl::MAIN_MEMORY:
            viennacl::linalg::host_based::detail::block_inplace_solve(mat, block_index_array, num_blocks, mat_diagonal, vec, tag);
            break;
  #ifdef VIENNACL_WITH_OPENCL
          case viennacl::OPENCL_MEMORY:
            viennacl::linalg::opencl::detail::block_inplace_solve(mat, block_index_array, num_blocks, mat_diagonal, vec, tag);
            break;
  #endif
  #ifdef VIENNACL_WITH_CUDA
          case viennacl::CUDA_MEMORY:
            viennacl::linalg::cuda::detail::block_inplace_solve(mat, block_index_array, num_blocks, mat_diagonal, vec, tag);
            break;
  #endif
          default:
            throw "not implemented";
        }
      }
      
      
    }
    
    

  } //namespace linalg


  /** @brief Returns an expression template class representing a transposed matrix */
  template<typename M1>
  typename viennacl::enable_if<viennacl::is_any_sparse_matrix<M1>::value,
                                matrix_expression< const M1, const M1, op_trans>
                              >::type
  trans(const M1 & mat)
  {
    return matrix_expression< const M1, const M1, op_trans>(mat, mat);
  }


  /** @brief Implementation of the operation v1 = A * v2, where A is a sparse matrix
  *
  * @param proxy  An expression template proxy class.
  */
  template <typename SCALARTYPE, unsigned int ALIGNMENT>
  template <typename SparseMatrixType>
  typename viennacl::enable_if< viennacl::is_any_sparse_matrix<SparseMatrixType>::value,
                                viennacl::vector<SCALARTYPE, ALIGNMENT> & >::type
  viennacl::vector<SCALARTYPE, ALIGNMENT>::operator=(const viennacl::vector_expression< const SparseMatrixType,
                                                                                        const viennacl::vector<SCALARTYPE, ALIGNMENT>,
                                                                                        viennacl::op_prod> & proxy) 
  {
    // check for the special case x = A * x
    if (viennacl::traits::handle(proxy.rhs()) == viennacl::traits::handle(*this))
    {
      viennacl::vector<SCALARTYPE, ALIGNMENT> temp(proxy.lhs().size1());
      viennacl::linalg::prod_impl(proxy.lhs(), proxy.rhs(), temp);
      *this = temp;
      return *this;
    }

    viennacl::linalg::prod_impl(proxy.lhs(), proxy.rhs(), *this);
    return *this;
  }

  //v += A * x
  /** @brief Implementation of the operation v1 += A * v2, where A is a matrix
  *
  * @param result The result vector v1
  * @param proxy  An expression template proxy class.
  */
  template <typename SCALARTYPE, unsigned int ALIGNMENT, typename SparseMatrixType>
  typename viennacl::enable_if< viennacl::is_any_sparse_matrix<SparseMatrixType>::value,
                                viennacl::vector<SCALARTYPE, ALIGNMENT> & >::type
  operator+=(viennacl::vector<SCALARTYPE, ALIGNMENT> & result,
              const viennacl::vector_expression< const SparseMatrixType, const viennacl::vector<SCALARTYPE, ALIGNMENT>, viennacl::op_prod> & proxy) 
  {
    vector<SCALARTYPE, ALIGNMENT> temp(proxy.lhs().size1());
    viennacl::linalg::prod_impl(proxy.lhs(), proxy.rhs(), temp);
    result += temp;
    return result;
  }

  /** @brief Implementation of the operation v1 -= A * v2, where A is a matrix
  *
  * @param result The result vector v1
  * @param proxy  An expression template proxy class.
  */
  template <typename SCALARTYPE, unsigned int ALIGNMENT, typename SparseMatrixType>
  typename viennacl::enable_if< viennacl::is_any_sparse_matrix<SparseMatrixType>::value,
                                viennacl::vector<SCALARTYPE, ALIGNMENT> & >::type
  operator-=(viennacl::vector<SCALARTYPE, ALIGNMENT> & result,
              const viennacl::vector_expression< const SparseMatrixType, const viennacl::vector<SCALARTYPE, ALIGNMENT>, viennacl::op_prod> & proxy) 
  {
    vector<SCALARTYPE, ALIGNMENT> temp(proxy.lhs().size1());
    viennacl::linalg::prod_impl(proxy.lhs(), proxy.rhs(), temp);
    result -= temp;
    return result;
  }
  
  
  //free functions:
  /** @brief Implementation of the operation 'result = v1 + A * v2', where A is a matrix
  *
  * @param result The vector the result is written to.
  * @param proxy  An expression template proxy class holding v1, A, and v2.
  */
  template <typename SCALARTYPE, unsigned int ALIGNMENT, typename SparseMatrixType>
  typename viennacl::enable_if< viennacl::is_any_sparse_matrix<SparseMatrixType>::value,
                                viennacl::vector<SCALARTYPE, ALIGNMENT> >::type
  operator+(viennacl::vector<SCALARTYPE, ALIGNMENT> & result,
            const viennacl::vector_expression< const SparseMatrixType, const viennacl::vector<SCALARTYPE, ALIGNMENT>, viennacl::op_prod> & proxy) 
  {
    assert(proxy.lhs().size1() == result.size() && bool("Dimensions for addition of sparse matrix-vector product to vector don't match!"));
    vector<SCALARTYPE, ALIGNMENT> temp(proxy.lhs().size1());
    viennacl::linalg::prod_impl(proxy.lhs(), proxy.rhs(), temp);
    result += temp;
    return result;
  }

  /** @brief Implementation of the operation 'result = v1 - A * v2', where A is a matrix
  *
  * @param result The vector the result is written to.
  * @param proxy  An expression template proxy class.
  */
  template <typename SCALARTYPE, unsigned int ALIGNMENT, typename SparseMatrixType>
  typename viennacl::enable_if< viennacl::is_any_sparse_matrix<SparseMatrixType>::value,
                                viennacl::vector<SCALARTYPE, ALIGNMENT> >::type
  operator-(viennacl::vector<SCALARTYPE, ALIGNMENT> & result,
            const viennacl::vector_expression< const SparseMatrixType, const viennacl::vector<SCALARTYPE, ALIGNMENT>, viennacl::op_prod> & proxy) 
  {
    assert(proxy.lhs().size1() == result.size() && bool("Dimensions for addition of sparse matrix-vector product to vector don't match!"));
    vector<SCALARTYPE, ALIGNMENT> temp(proxy.lhs().size1());
    viennacl::linalg::prod_impl(proxy.lhs(), proxy.rhs(), temp);
    result += temp;
    return result;
  }

} //namespace viennacl


#endif
