#ifndef VIENNACL_GENERATOR_OPERATORS_HPP
#define VIENNACL_GENERATOR_OPERATORS_HPP

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


/** @file viennacl/generator/operators.hpp
 *  @brief Definition of the operators between the symbolic types
 *
 *  Generator code contributed by Philippe Tillet
 */

#include "viennacl/generator/forwards.h"
#include "viennacl/generator/result_of.hpp"
#include "viennacl/generator/meta_tools/utils.hpp"

namespace viennacl
{
namespace generator
{

struct assign_type
{
  static const std::string expression_string() { return " = "; }
  static const std::string name() { return "eq"; }
};

struct add_type
{
  static const std::string expression_string() { return " + "; }
  static const std::string name() { return "p"; }
};

struct inplace_add_type
{
  static const std::string expression_string() { return " += "; }
  static const std::string name() { return "p_eq"; }
};

struct sub_type
{
  static const std::string expression_string() { return " - "; }
  static const std::string name() { return "m"; }
};

struct inplace_sub_type
{
  static const std::string expression_string() { return " -= "; }
  static const std::string name() { return "m_eq"; }
};

struct scal_mul_type
{
  static const std::string expression_string() { return " * "; }
  static const std::string name() { return "mu"; }
};

struct inplace_scal_mul_type
{
  static const std::string expression_string() { return " *= "; }
  static const std::string name() { return "mu_eq"; }
};


struct scal_div_type
{
  static const std::string expression_string() { return " / "; }
  static const std::string name() { return "d"; }
};

struct inplace_scal_div_type
{
  static const std::string expression_string() { return " /= "; }
  static const std::string name() { return "d_eq"; }
};

struct inner_prod_type
{
  static const std::string expression_string() { return "_i_"; }
  static const std::string name() { return "i"; }
};

struct prod_type
{
  static const std::string expression_string() { return "_p_"; }
  static const std::string name() { return "p"; }
};

struct elementwise_prod_type
{
    static const std::string expression_string() { return "*"; }
    static const std::string name() { return "ewp"; }
};

struct elementwise_div_type
{
    static const std::string expression_string() { return "/"; }
    static const std::string name() { return "ewd"; }
};


template<class T>
struct make_inplace
{
  typedef T Result;
};

template<>
struct make_inplace<add_type>
{
  typedef inplace_add_type Result;
};

template<>
struct make_inplace<sub_type>
{
  typedef inplace_sub_type Result;
};

template<>
struct make_inplace<scal_mul_type>
{
  typedef inplace_scal_mul_type Result;
};

template<>
struct make_inplace<scal_div_type>
{
  typedef inplace_scal_div_type Result;
};

template<class LHS, class RHS>
struct CHECK_ALIGNMENT_COMPATIBILITY
{
    typedef typename result_of::expression_type<LHS>::Result LHS_TYPE;
    typedef typename result_of::expression_type<RHS>::Result RHS_TYPE;
    VIENNACL_STATIC_ASSERT(LHS_TYPE::Alignment == RHS_TYPE::Alignment,AlignmentIncompatible);
};


/** @brief Scalar multiplication operator */
template<class LHS_TYPE, class RHS_TYPE>
typename enable_if_c<result_of::is_scalar_expression<LHS_TYPE>::value || result_of::is_scalar_expression<RHS_TYPE>::value,
                     compound_node<LHS_TYPE,scal_mul_type,RHS_TYPE> >::type
operator* ( LHS_TYPE const & lhs, RHS_TYPE const & rhs )
{
  return compound_node<LHS_TYPE, scal_mul_type,RHS_TYPE> ();
}

/** @brief Scalar division operator */
template<class LHS_TYPE, class RHS_TYPE>
typename enable_if_c< result_of::is_scalar_expression<RHS_TYPE>::value,
                      compound_node<LHS_TYPE,scal_div_type,RHS_TYPE> > ::type
operator/ ( LHS_TYPE const & lhs, RHS_TYPE const & rhs )
{
  return compound_node<LHS_TYPE,scal_div_type,RHS_TYPE> ();
}

/** @brief Addition operator on 2 elements of the same type */
template<class LHS_TYPE, class RHS_TYPE>
typename enable_if< result_of::is_same_expression_type<LHS_TYPE, RHS_TYPE>,
                    compound_node<LHS_TYPE, add_type, RHS_TYPE> >::type
operator+ ( LHS_TYPE const & lhs, RHS_TYPE const & rhs )
{
  CHECK_ALIGNMENT_COMPATIBILITY<LHS_TYPE,RHS_TYPE>();
  return compound_node<LHS_TYPE, add_type, RHS_TYPE>();
}

/** @brief Substraction operator on 2 elements of the same type */
template<class LHS_TYPE, class RHS_TYPE>
typename enable_if< result_of::is_same_expression_type<LHS_TYPE, RHS_TYPE>,
                    compound_node<LHS_TYPE, sub_type, RHS_TYPE> >::type
operator- ( LHS_TYPE const & lhs, RHS_TYPE const & rhs )
{
  CHECK_ALIGNMENT_COMPATIBILITY<LHS_TYPE,RHS_TYPE>();
  return compound_node<LHS_TYPE, sub_type, RHS_TYPE>();
}

/** @brief Helper for the inner_prod operator */
template<class LHS, class RHS>
struct make_inner_prod;

template<class LHS, class LHS_SIZE_DESCRIPTOR,
         class RHS, class RHS_SIZE_DESCRIPTOR>
struct make_inner_prod<result_of::vector_expression<LHS, LHS_SIZE_DESCRIPTOR>,
                       result_of::vector_expression<RHS, RHS_SIZE_DESCRIPTOR> >
{
  typedef compound_node<LHS,inner_prod_type,RHS> Result;
};


/** @brief Inner product operator */
template<class LHS, class RHS>
compound_node<LHS,inner_prod_type,RHS> inner_prod ( LHS vec_expr1,RHS vec_expr2 )
{
  CHECK_ALIGNMENT_COMPATIBILITY<LHS,RHS>();
  typedef typename result_of::expression_type<LHS>::Result LHS_TYPE;
  typedef typename result_of::expression_type<RHS>::Result RHS_TYPE;
  typename make_inner_prod<LHS_TYPE,RHS_TYPE>::Result result;
  return result;
}

/** @brief Product operator */
template<class LHS, class RHS>
compound_node<LHS,prod_type,RHS> prod ( LHS vec_expr1,RHS vec_expr2 )
{
  CHECK_ALIGNMENT_COMPATIBILITY<LHS,RHS>();
  return compound_node<LHS,prod_type,RHS>();
}

template<class LHS, class RHS>
compound_node<LHS,elementwise_prod_type,RHS> elementwise_prod(LHS expr1, RHS expr2){
    CHECK_ALIGNMENT_COMPATIBILITY<LHS,RHS>();
    return compound_node<LHS,elementwise_prod_type,RHS>();
}

template<class LHS, class RHS>
compound_node<LHS,elementwise_div_type,RHS> elementwise_div(LHS expr1, RHS expr2){
    CHECK_ALIGNMENT_COMPATIBILITY<LHS,RHS>();
    return compound_node<LHS,elementwise_div_type,RHS>();
}

/*
 * Traits
 */

namespace result_of{

template<class OP>
struct is_assignment{
  enum{ value = are_same_type<OP,assign_type>::value ||
        are_same_type<OP,inplace_add_type>::value ||
        are_same_type<OP,inplace_sub_type>::value ||
        are_same_type<OP,inplace_scal_mul_type>::value ||
        are_same_type<OP,inplace_scal_div_type>::value};
};


template<class T>
struct is_assignment_compound{
    enum { value = 0 };
};

template<class LHS, class OP, class RHS>
struct is_assignment_compound<compound_node<LHS,OP,RHS> >
{
    enum { value = is_assignment<OP>::value };
};

template<class OP>
struct is_arithmetic_operator{
  enum{ value = result_of::is_assignment<OP>::value ||
          are_same_type<OP,add_type>::value ||
          are_same_type<OP,sub_type>::value ||
          are_same_type<OP,scal_mul_type>::value ||
          are_same_type<OP,scal_div_type>::value};
};

template<class T>
struct is_arithmetic_compound{
  enum { value = 0 };
};

template<class LHS, class OP, class RHS>
struct is_arithmetic_compound<compound_node<LHS,OP,RHS> >
{
  enum { value = result_of::is_arithmetic_operator<OP>::value };
};

}

}

}

#endif
