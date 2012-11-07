#ifndef VIENNACL_LINALG_SPARSE_MATRIX_OPERATIONS_HPP_
#define VIENNACL_LINALG_SPARSE_MATRIX_OPERATIONS_HPP_

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

/** @file viennacl/linalg/sparse_matrix_operations.hpp
    @brief Implementations of operations using sparse matrices
*/

#include "viennacl/forwards.h"
#include "viennacl/scalar.hpp"
#include "viennacl/vector.hpp"
#include "viennacl/tools/tools.hpp"
#include "viennacl/linalg/single_threaded/sparse_matrix_operations.hpp"

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
    
    
    // A * x
    /** @brief Returns a proxy class that represents matrix-vector multiplication with any sparse matrix type
    *
    * This is used for the convenience expression result = prod(mat, vec);
    *
    * @param mat    The matrix
    * @param vec    The vector
    */
    template<typename SparseMatrixType, class SCALARTYPE, unsigned int ALIGNMENT>
    typename viennacl::enable_if< viennacl::is_sparse_matrix<SparseMatrixType>::value,
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
    typename viennacl::enable_if< viennacl::is_sparse_matrix<SparseMatrixType>::value>::type
    prod_impl(const SparseMatrixType & mat, 
              const viennacl::vector<ScalarType, ALIGNMENT> & vec,
                    viennacl::vector<ScalarType, ALIGNMENT> & result)
    {
      assert( (mat.size1() == result.size()) && bool("Size check failed for compressed matrix-vector product: size1(mat) != size(result)"));
      assert( (mat.size2() == vec.size())    && bool("Size check failed for compressed matrix-vector product: size2(mat) != size(x)"));

      switch (viennacl::traits::handle(mat).get_active_handle_id())
      {
        case viennacl::backend::MAIN_MEMORY:
          viennacl::linalg::single_threaded::prod_impl(mat, vec, result);
          break;
#ifdef VIENNACL_WITH_OPENCL
        case viennacl::backend::OPENCL_MEMORY:
          viennacl::linalg::opencl::prod_impl(mat, vec, result);
          break;
#endif
#ifdef VIENNACL_WITH_CUDA
        case viennacl::backend::CUDA_MEMORY:
          viennacl::linalg::cuda::prod_impl(mat, vec, result);
          break;
#endif
        default:
          throw "not implemented";
      }
    }

  } //namespace linalg



    /** @brief Implementation of the operation v1 = A * v2, where A is a sparse matrix
    *
    * @param proxy  An expression template proxy class.
    */
    template <typename SCALARTYPE, unsigned int ALIGNMENT>
    template <typename SparseMatrixType>
    typename viennacl::enable_if< viennacl::is_sparse_matrix<SparseMatrixType>::value,
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
    * @param proxy  An expression template proxy class.
    */
    template <typename SCALARTYPE, unsigned int ALIGNMENT, typename SparseMatrixType>
    typename viennacl::enable_if< viennacl::is_sparse_matrix<SparseMatrixType>::value,
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
    * @param proxy  An expression template proxy class.
    */
    template <typename SCALARTYPE, unsigned int ALIGNMENT, typename SparseMatrixType>
    typename viennacl::enable_if< viennacl::is_sparse_matrix<SparseMatrixType>::value,
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
    * @param proxy  An expression template proxy class.
    */
    template <typename SCALARTYPE, unsigned int ALIGNMENT, typename SparseMatrixType>
    typename viennacl::enable_if< viennacl::is_sparse_matrix<SparseMatrixType>::value,
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
    * @param proxy  An expression template proxy class.
    */
    template <typename SCALARTYPE, unsigned int ALIGNMENT, typename SparseMatrixType>
    typename viennacl::enable_if< viennacl::is_sparse_matrix<SparseMatrixType>::value,
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
