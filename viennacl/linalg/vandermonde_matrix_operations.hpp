#ifndef VIENNACL_LINALG_VANDERMONDE_MATRIX_OPERATIONS_HPP_
#define VIENNACL_LINALG_VANDERMONDE_MATRIX_OPERATIONS_HPP_

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

/** @file viennacl/linalg/vandermonde_matrix_operations.hpp
    @brief Implementations of operations using vandermonde_matrix. Experimental.
*/

#include "viennacl/forwards.h"
#include "viennacl/scalar.hpp"
#include "viennacl/vector.hpp"
#include "viennacl/tools/tools.hpp"
#include "viennacl/fft.hpp"
#include "viennacl/linalg/opencl/vandermonde_matrix_operations.hpp"

namespace viennacl
{
  namespace linalg
  {
    
    // A * x
    
    /** @brief Carries out matrix-vector multiplication with a vandermonde_matrix
    *
    * Implementation of the convenience expression result = prod(mat, vec);
    *
    * @param mat    The matrix
    * @param vec    The vector
    * @param result The result vector
    */
    template<class SCALARTYPE, unsigned int ALIGNMENT>
    void prod_impl(const viennacl::vandermonde_matrix<SCALARTYPE, ALIGNMENT> & mat, 
                   const viennacl::vector_base<SCALARTYPE> & vec,
                         viennacl::vector_base<SCALARTYPE> & result)
    {
      assert(mat.size1() == result.size());
      assert(mat.size2() == vec.size());
      
      switch (viennacl::traits::handle(mat).get_active_handle_id())
      {
        case viennacl::OPENCL_MEMORY:
          viennacl::linalg::opencl::prod_impl(mat, vec, result);
          break;
        default:
          throw "not implemented";
      }
    }

  } //namespace linalg



    /** @brief Implementation of the operation v1 = A * v2, where A is a matrix
    *
    * @param proxy  An expression template proxy class.
    */
    template <typename SCALARTYPE, unsigned int ALIGNMENT>
    template <unsigned int MAT_ALIGNMENT>
    viennacl::vector<SCALARTYPE, ALIGNMENT> & 
    viennacl::vector<SCALARTYPE, ALIGNMENT>::operator=(const viennacl::vector_expression< const vandermonde_matrix<SCALARTYPE, MAT_ALIGNMENT>,
                                                                                          const viennacl::vector_base<SCALARTYPE>,
                                                                                          viennacl::op_prod> & proxy) 
    {
      // check for the special case x = A * x
      if (viennacl::traits::handle(proxy.rhs()) == viennacl::traits::handle(*this))
      {
        viennacl::vector<SCALARTYPE, ALIGNMENT> result(proxy.rhs().size());
        viennacl::linalg::prod_impl(proxy.lhs(), proxy.rhs(), result);
        *this = result;
      }
      else
      {
        viennacl::linalg::prod_impl(proxy.lhs(), proxy.rhs(), *this);
      }
      return *this;
    }

    //v += A * x
    /** @brief Implementation of the operation v1 += A * v2, where A is a matrix
    *
    * @param proxy  An expression template proxy class.
    */
    template <typename SCALARTYPE, unsigned int ALIGNMENT>
    template <unsigned int MAT_ALIGNMENT>
    viennacl::vector<SCALARTYPE, ALIGNMENT> & 
    viennacl::vector<SCALARTYPE, ALIGNMENT>::operator+=(const vector_expression< const vandermonde_matrix<SCALARTYPE, MAT_ALIGNMENT>,
                                                                                 const vector_base<SCALARTYPE>,
                                                                                 op_prod> & proxy) 
    {
      vector<SCALARTYPE, ALIGNMENT> result(proxy.lhs().size1());
      viennacl::linalg::prod_impl(proxy.lhs(), proxy.rhs(), result);
      *this += result;
      return *this;
    }

    /** @brief Implementation of the operation v1 -= A * v2, where A is a matrix
    *
    * @param proxy  An expression template proxy class.
    */
    template <typename SCALARTYPE, unsigned int ALIGNMENT>
    template <unsigned int MAT_ALIGNMENT>
    viennacl::vector<SCALARTYPE, ALIGNMENT> & 
    viennacl::vector<SCALARTYPE, ALIGNMENT>::operator-=(const vector_expression< const vandermonde_matrix<SCALARTYPE, MAT_ALIGNMENT>,
                                                                                 const vector_base<SCALARTYPE>,
                                                                                 op_prod> & proxy) 
    {
      vector<SCALARTYPE, ALIGNMENT> result(proxy.get_lhs().size1());
      viennacl::linalg::prod_impl(proxy.lhs(), proxy.rhs(), result);
      *this -= result;
      return *this;
    }
    
    
    //free functions:
    /** @brief Implementation of the operation 'result = v1 + A * v2', where A is a matrix
    *
    * @param proxy  An expression template proxy class.
    */
    template <typename SCALARTYPE, unsigned int ALIGNMENT>
    template <unsigned int MAT_ALIGNMENT>
    viennacl::vector<SCALARTYPE, ALIGNMENT> 
    viennacl::vector<SCALARTYPE, ALIGNMENT>::operator+(const vector_expression< const vandermonde_matrix<SCALARTYPE, MAT_ALIGNMENT>,
                                                                                const vector_base<SCALARTYPE>,
                                                                                op_prod> & proxy) 
    {
      assert(proxy.get_lhs().size1() == base_type::size());
      vector<SCALARTYPE, ALIGNMENT> result(base_type::size());
      viennacl::linalg::prod_impl(proxy.lhs(), proxy.rhs(), result);
      result += *this;
      return result;
    }

    /** @brief Implementation of the operation 'result = v1 - A * v2', where A is a matrix
    *
    * @param proxy  An expression template proxy class.
    */
    template <typename SCALARTYPE, unsigned int ALIGNMENT>
    template <unsigned int MAT_ALIGNMENT>
    viennacl::vector<SCALARTYPE, ALIGNMENT> 
    viennacl::vector<SCALARTYPE, ALIGNMENT>::operator-(const vector_expression< const vandermonde_matrix<SCALARTYPE, MAT_ALIGNMENT>,
                                                                                const vector_base<SCALARTYPE>,
                                                                                op_prod> & proxy) 
    {
      assert(proxy.get_lhs().size1() == base_type::size());
      vector<SCALARTYPE, ALIGNMENT> result(base_type::size());
      viennacl::linalg::prod_impl(proxy.lhs(), proxy.rhs(), result);
      result = *this - result;
      return result;
    }

} //namespace viennacl


#endif
