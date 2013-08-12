#ifndef VIENNACL_SCHEDULER_EXECUTE_SCALAR_ASSIGN_HPP
#define VIENNACL_SCHEDULER_EXECUTE_SCALAR_ASSIGN_HPP

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


/** @file viennacl/scheduler/execute_vector assign.hpp
    @brief Deals with the execution of x = RHS; for a vector x and any compatible right hand side expression RHS.
*/

#include "viennacl/forwards.h"
#include "viennacl/scheduler/forwards.h"
#include "viennacl/linalg/vector_operations.hpp"

namespace viennacl
{
  namespace scheduler
  {
    /** @brief Deals with x = RHS where RHS is a vector expression */
    inline void execute_scalar_assign_composite(statement const & s, statement_node const & root_node)
    {
      statement_node const & leaf = s.array()[root_node.rhs.node_index];

      if (leaf.op.type  == OPERATION_BINARY_INNER_PROD_TYPE)
      {
        if (root_node.lhs.numeric_type == FLOAT_TYPE  && root_node.lhs.type_family == SCALAR_TYPE_FAMILY
            &&   leaf.lhs.numeric_type == FLOAT_TYPE  &&      leaf.lhs.type_family == VECTOR_TYPE_FAMILY
            &&   leaf.rhs.numeric_type == FLOAT_TYPE  &&      leaf.rhs.type_family == VECTOR_TYPE_FAMILY)

        {
          viennacl::scalar<float>            & s = *(root_node.lhs.scalar_float);
          viennacl::vector_base<float> const & y = *(leaf.lhs.vector_float);
          viennacl::vector_base<float> const & z = *(leaf.rhs.vector_float);
          viennacl::linalg::inner_prod_impl(y, z, s);
        }
        else if (root_node.lhs.numeric_type == DOUBLE_TYPE  && root_node.lhs.type_family == SCALAR_TYPE_FAMILY
                 &&   leaf.lhs.numeric_type == DOUBLE_TYPE  &&      leaf.lhs.type_family == VECTOR_TYPE_FAMILY
                 &&   leaf.rhs.numeric_type == DOUBLE_TYPE  &&      leaf.rhs.type_family == VECTOR_TYPE_FAMILY)
        {
          viennacl::scalar<double>            & s = *(root_node.lhs.scalar_double);
          viennacl::vector_base<double> const & y = *(leaf.lhs.vector_double);
          viennacl::vector_base<double> const & z = *(leaf.rhs.vector_double);
          viennacl::linalg::inner_prod_impl(y, z, s);
        }
        else
          throw statement_not_supported_exception("Cannot deal with inner product of the provided arguments");
      }
      else if (leaf.op.type  == OPERATION_UNARY_NORM_1_TYPE)
      {
        if (root_node.lhs.numeric_type == FLOAT_TYPE  && root_node.lhs.type_family == SCALAR_TYPE_FAMILY
            &&   leaf.lhs.numeric_type == FLOAT_TYPE  &&      leaf.lhs.type_family == VECTOR_TYPE_FAMILY)
        {
          viennacl::scalar<float>            & s = *(root_node.lhs.scalar_float);
          viennacl::vector_base<float> const & x = *(leaf.lhs.vector_float);
          viennacl::linalg::norm_1_impl(x, s);
        }
        else if (root_node.lhs.numeric_type == DOUBLE_TYPE  && root_node.lhs.type_family == SCALAR_TYPE_FAMILY
                 &&   leaf.lhs.numeric_type == DOUBLE_TYPE  &&      leaf.lhs.type_family == VECTOR_TYPE_FAMILY)
        {
          viennacl::scalar<double>            & s = *(root_node.lhs.scalar_double);
          viennacl::vector_base<double> const & x = *(leaf.lhs.vector_double);
          viennacl::linalg::norm_1_impl(x, s);
        }
        else
          throw statement_not_supported_exception("Cannot deal with norm_1 of the provided arguments");
      }
      else if (leaf.op.type  == OPERATION_UNARY_NORM_2_TYPE)
      {
        if (root_node.lhs.numeric_type == FLOAT_TYPE   && root_node.lhs.type_family == SCALAR_TYPE_FAMILY
            &&   leaf.lhs.numeric_type == FLOAT_TYPE   &&      leaf.lhs.type_family == VECTOR_TYPE_FAMILY)
        {
          viennacl::scalar<float>            & s = *(root_node.lhs.scalar_float);
          viennacl::vector_base<float> const & x = *(leaf.lhs.vector_float);
          viennacl::linalg::norm_2_impl(x, s);
        }
        else if (root_node.lhs.numeric_type == DOUBLE_TYPE && root_node.lhs.type_family == SCALAR_TYPE_FAMILY
                 &&   leaf.lhs.numeric_type == DOUBLE_TYPE &&      leaf.lhs.type_family == VECTOR_TYPE_FAMILY)
        {
          viennacl::scalar<double>            & s = *(root_node.lhs.scalar_double);
          viennacl::vector_base<double> const & x = *(leaf.lhs.vector_double);
          viennacl::linalg::norm_2_impl(x, s);
        }
        else
          throw statement_not_supported_exception("Cannot deal with norm_2 of the provided arguments");
      }
      else if (leaf.op.type  == OPERATION_UNARY_NORM_INF_TYPE)
      {
        if (root_node.lhs.numeric_type == FLOAT_TYPE  && root_node.lhs.type_family == SCALAR_TYPE_FAMILY
            &&   leaf.lhs.numeric_type == FLOAT_TYPE  &&      leaf.lhs.type_family == VECTOR_TYPE_FAMILY)
        {
          viennacl::scalar<float>            & s = *(root_node.lhs.scalar_float);
          viennacl::vector_base<float> const & x = *(leaf.lhs.vector_float);
          viennacl::linalg::norm_inf_impl(x, s);
        }
        else if (root_node.lhs.numeric_type == DOUBLE_TYPE && root_node.lhs.type_family == SCALAR_TYPE_FAMILY
                 &&   leaf.lhs.numeric_type == DOUBLE_TYPE &&      leaf.lhs.type_family == VECTOR_TYPE_FAMILY)
        {
          viennacl::scalar<double>            & s = *(root_node.lhs.scalar_double);
          viennacl::vector_base<double> const & x = *(leaf.lhs.vector_double);
          viennacl::linalg::norm_inf_impl(x, s);
        }
        else
          throw statement_not_supported_exception("Cannot deal with norm_inf of the provided arguments");
      }
      else
        throw statement_not_supported_exception("Unsupported operation for scalar.");
    }


  }

} //namespace viennacl

#endif

