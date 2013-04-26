#ifndef VIENNACL_TOOLS_MATRIX_SIZE_DEDUCER_HPP_
#define VIENNACL_TOOLS_MATRIX_SIZE_DEDUCER_HPP_

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

/** @file viennacl/tools/matrix_size_deducer.hpp
    @brief Helper implementations that deduce the dimensions of the supplied matrix-valued expressions.
*/

#include <string>
#include <fstream>
#include <sstream>
#include "viennacl/forwards.h"
#include "viennacl/tools/adapter.hpp"

#include <vector>
#include <map>

namespace viennacl
{
  namespace tools
  {

    /** @brief Deduces the size of the resulting vector represented by a vector_expression from the operands
    *
    * @tparam LHS   The left hand side operand
    * @tparam RHS   The right hand side operand
    * @tparam OP    The operation tag
    */
    template <typename LHS, typename RHS, typename OP>
    struct MATRIX_SIZE_DEDUCER
    {
      //Standard case: size1 from lhs, size2 from rhs (fits most cases)
      static std::size_t size1(LHS & lhs, RHS & /*rhs*/) { return lhs.size1(); }
      static std::size_t size2(LHS & /*lhs*/, RHS & rhs) { return rhs.size2(); }
    };

    /** \cond */
    //special case: outer vector product:
    template <typename ScalarType, unsigned int A1, unsigned int A2>
    struct MATRIX_SIZE_DEDUCER<const viennacl::vector<ScalarType, A1>,
                               const viennacl::vector<ScalarType, A2>,
                               viennacl::op_prod>
    {
      static std::size_t size1(viennacl::vector<ScalarType, A1> const & lhs,
                               viennacl::vector<ScalarType, A2> const & /*rhs*/) { return lhs.size(); }

      static std::size_t size2(viennacl::vector<ScalarType, A1> const & /*lhs*/,
                               viennacl::vector<ScalarType, A2> const & rhs) { return rhs.size(); }
    };

    //special case: multiplication with a GPU scalar
    template <typename ScalarType, typename F>
    struct MATRIX_SIZE_DEDUCER<const viennacl::matrix_base<ScalarType, F>,
                               const viennacl::scalar<ScalarType>,
                               viennacl::op_prod>
    {
      static std::size_t size1(viennacl::matrix_base<ScalarType, F> const & lhs,
                               viennacl::scalar<ScalarType> const & /*rhs*/) { return lhs.size1(); }

      static std::size_t size2(viennacl::matrix_base<ScalarType, F> const & lhs,
                               viennacl::scalar<ScalarType> const & /*rhs*/) { return lhs.size2(); }
    };
    
    //special case: multiplication with a CPU scalar
    template <typename ScalarType, typename F>
    struct MATRIX_SIZE_DEDUCER<const viennacl::matrix_base<ScalarType, F>,
                               const ScalarType,
                               viennacl::op_prod>
    {
      static std::size_t size1(viennacl::matrix_base<ScalarType, F> const & lhs,
                               ScalarType const & /*rhs*/) { return lhs.size1(); }

      static std::size_t size2(viennacl::matrix_base<ScalarType, F> const & lhs,
                               ScalarType const & /*rhs*/) { return lhs.size2(); }
    };
    



    
    //special case: division by with a GPU scalar
    template <typename ScalarType, typename F>
    struct MATRIX_SIZE_DEDUCER<const viennacl::matrix_base<ScalarType, F>,
                               const viennacl::scalar<ScalarType>,
                               viennacl::op_div>
    {
      static std::size_t size1(viennacl::matrix_base<ScalarType, F> const & lhs,
                               viennacl::scalar<ScalarType> const & /*rhs*/) { return lhs.size1(); }

      static std::size_t size2(viennacl::matrix_base<ScalarType, F> const & lhs,
                               viennacl::scalar<ScalarType> const & /*rhs*/) { return lhs.size2(); }
    };
    
    //special case: division by a CPU scalar
    template <typename ScalarType, typename F>
    struct MATRIX_SIZE_DEDUCER<const viennacl::matrix_base<ScalarType, F>,
                               const ScalarType,
                               viennacl::op_div>
    {
      static std::size_t size1(viennacl::matrix_base<ScalarType, F> const & lhs,
                               ScalarType const & /*rhs*/) { return lhs.size1(); }

      static std::size_t size2(viennacl::matrix_base<ScalarType, F> const & lhs,
                               ScalarType const & /*rhs*/) { return lhs.size2(); }
    };
    
    
    
    
    
    
    
    //special case: transposed matrix-vector product: Return the number of rows of the matrix
    template <typename MatrixType>
    struct MATRIX_SIZE_DEDUCER<MatrixType,
                               MatrixType,
                               viennacl::op_trans>
    {
      static std::size_t size1(const MatrixType & lhs,
                               const MatrixType & /*rhs*/) { return lhs.size2(); }
      static std::size_t size2(const MatrixType & lhs,
                               const MatrixType & /*rhs*/) { return lhs.size1(); }
    };

    // A^T * B
    template <typename ScalarType, typename T1, typename F2>
    struct MATRIX_SIZE_DEDUCER<const viennacl::matrix_expression<T1,
                                                                 T1, op_trans>,
                               const viennacl::matrix_base<ScalarType, F2>,
                               viennacl::op_prod>
    {
      static std::size_t size1(viennacl::matrix_expression<T1,
                                                           T1,
                                                           op_trans> const & lhs,
                               viennacl::matrix_base<ScalarType, F2> const & /*rhs*/) { return lhs.lhs().size2(); }
      static std::size_t size2(viennacl::matrix_expression<T1,
                                                           T1,
                                                           op_trans> const & /*lhs*/,
                               viennacl::matrix_base<ScalarType, F2> const & rhs) { return rhs.size2(); }
    };

    
    // A * B^T 
    
    template <typename ScalarType, typename F1, typename T2>
    struct MATRIX_SIZE_DEDUCER<const viennacl::matrix_base<ScalarType, F1>,
                               const viennacl::matrix_expression<T2,
                                                                 T2, op_trans>,
                               viennacl::op_prod>
    {
      static std::size_t size1(viennacl::matrix_base<ScalarType, F1> const & lhs,
                               viennacl::matrix_expression<T2,
                                                           T2,
                                                           op_trans> const & /*rhs*/) { return lhs.size1(); }
      static std::size_t size2(viennacl::matrix_base<ScalarType, F1> const & /*lhs*/,
                               viennacl::matrix_expression<T2,
                                                           T2,
                                                           op_trans> const & rhs) { return rhs.lhs().size1(); }
    };


    
    
    // A^T * B^T 
    
    template <typename T1, typename T2>
    struct MATRIX_SIZE_DEDUCER<const viennacl::matrix_expression<T1,
                                                                 T1, op_trans>,
                               const viennacl::matrix_expression<T2,
                                                                 T2, op_trans>,
                               viennacl::op_prod>
    {
      typedef viennacl::matrix_expression<T1, T1, op_trans>   LHSType;
      typedef viennacl::matrix_expression<T2, T2, op_trans>   RHSType;
      
      static std::size_t size1(LHSType const & lhs,
                               RHSType const & /*rhs*/) { return lhs.lhs().size2(); }
      static std::size_t size2(LHSType const & /*lhs*/,
                               RHSType const & rhs) { return rhs.lhs().size1(); }
    };
    /** \endcond */
  }
}

#endif

