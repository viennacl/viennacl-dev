#ifndef VIENNACL_GENERATOR_TRAITS_HPP
#define VIENNACL_GENERATOR_TRAITS_HPP

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


/** @file viennacl/generator/traits.hpp
    @brief Some traits for the generator.
*/

#include <sstream>
#include "symbolic_types.hpp"

namespace viennacl{

  namespace generator{

    namespace result_of{

      template<class T>
      struct is_transposed{ enum { value = 0 }; };

      template<class SUB>
      struct is_transposed<viennacl::generator::unary_matrix_expression<SUB, trans_type> >{ enum { value = 1 }; };

      template<class T>
      struct is_row_major{ enum { value = 0 }; };

      template<class SUB>
      struct is_row_major<viennacl::generator::unary_matrix_expression<SUB, trans_type> >{ enum { value = is_row_major<SUB>::value }; };

      template<class ScalarType, class ELEMENT_ACCESSOR, class ROW_INDEX, class COL_INDEX>
      struct is_row_major<viennacl::generator::symbolic_matrix<viennacl::matrix<ScalarType, viennacl::row_major>, ELEMENT_ACCESSOR, ROW_INDEX, COL_INDEX > >{ enum { value = 1 }; };

      //Binary Expressions

      template<class T>
      struct is_binary_matrix_expression { enum { value = 0 }; };

      template<class LHS, class OP, class RHS>
      struct is_binary_matrix_expression<viennacl::generator::binary_matrix_expression<LHS, OP, RHS> >{ enum{ value=1 }; };

      template<class T>
      struct is_binary_vector_expression { enum { value = 0 }; };

      template<class LHS, class OP, class RHS>
      struct is_binary_vector_expression<viennacl::generator::binary_vector_expression<LHS, OP, RHS> >{ enum{ value=1 }; };

      template<class T>
      struct is_binary_scalar_expression { enum { value = 0 }; };

      template<class LHS, class OP, class RHS>
      struct is_binary_scalar_expression<viennacl::generator::binary_scalar_expression<LHS, OP, RHS> >{ enum{ value=1 }; };

      template<class T>
      struct is_binary_expression { enum { value = is_binary_matrix_expression<T>::value
                                                  || is_binary_scalar_expression<T>::value
                                                  || is_binary_vector_expression<T>::value }; };
      //Unary Expressions

      template<class T>
      struct is_unary_matrix_expression { enum { value = 0 }; };

      template<class SUB, class OP>
      struct is_unary_matrix_expression<viennacl::generator::unary_matrix_expression<SUB, OP> >{ enum{ value=1 }; };

      template<class T>
      struct is_unary_vector_expression { enum { value = 0 }; };

      template<class SUB, class OP>
      struct is_unary_vector_expression<viennacl::generator::unary_vector_expression<SUB, OP> >{ enum{ value=1 }; };

      template<class T>
      struct is_unary_scalar_expression { enum { value = 0 }; };

      template<class SUB, class OP>
      struct is_unary_scalar_expression<viennacl::generator::unary_scalar_expression<SUB, OP> >{ enum{ value=1 }; };

      template<class T>
      struct is_unary_expression { enum { value = is_unary_matrix_expression<T>::value
                                                  || is_unary_scalar_expression<T>::value
                                                  || is_unary_vector_expression<T>::value }; };




      //Products and Reductions

      template<class T>
      struct is_scalar_reduction { enum { value = 0 }; };

      template<class LHS, class REDUCE_OP, class RHS>
      struct is_scalar_reduction<binary_scalar_expression<LHS,reduce_type<REDUCE_OP>,RHS> >{ enum { value = 1 }; };

      template<class T>
      struct is_vector_reduction { enum { value = 0 }; };

      template<class LHS, class REDUCE_OP, class RHS>
      struct is_vector_reduction<binary_vector_expression<LHS,reduce_type<REDUCE_OP>,RHS> >{ enum { value = 1 }; };


      template<class T>
      struct is_matrix_product { enum { value = 0 }; };

      template<class LHS, class REDUCE_OP, class RHS>
      struct is_matrix_product<binary_matrix_expression<LHS,reduce_type<REDUCE_OP>,RHS> >{ enum { value = 1 }; };

      //Count if

      template<template<class> class PRED, class TREE, class Enable=void>
      struct count_if{
          enum { value = PRED<TREE>::value };
      };

      template<template<class> class PRED, class TREE>
      struct count_if<PRED, TREE, typename viennacl::enable_if<is_binary_expression<TREE>::value>::type>{
          enum{ value = PRED<TREE>::value
                        + count_if<PRED, typename TREE::Lhs>::value
                        + count_if<PRED, typename TREE::Rhs>::value };
      };

      template<template<class> class PRED, class TREE>
      struct count_if<PRED, TREE, typename viennacl::enable_if<is_unary_expression<TREE>::value>::type>{
          enum{ value = PRED<TREE>::value
                        + count_if<PRED, typename TREE::Underlying>::value};
      };

      // Test expression nature
      template<class T>
      struct is_saxpy_vector_operation{
          enum { value = (is_binary_vector_expression<T>::value && count_if<is_vector_reduction,T>::value==0) //is vector expression
                        ||(is_binary_scalar_expression<T>::value && count_if<is_scalar_reduction,T>::value==0) };//is scalar expression
      };

      template<class T>
      struct is_saxpy_matrix_operation{
          enum { value = (is_binary_matrix_expression<T>::value && count_if<is_matrix_product,T>::value==0) };
      };

      template<class T>
      struct is_scalar_reduction_operation{
          enum { value = static_cast<bool>(count_if<is_scalar_reduction,T>::value) };
      };

      template<class T>
      struct is_vector_reduction_operation{
          enum { value = static_cast<bool>(count_if<is_vector_reduction,T>::value) };
      };

      template<class T>
      struct is_matrix_product_operation{
          enum { value = static_cast<bool>(count_if<is_matrix_product,T>::value) };
      };

    }

  }

}

#endif
