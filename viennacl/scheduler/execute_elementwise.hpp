#ifndef VIENNACL_SCHEDULER_EXECUTE_ELEMENTWISE_HPP
#define VIENNACL_SCHEDULER_EXECUTE_ELEMENTWISE_HPP

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


/** @file viennacl/scheduler/execute_elementwise.hpp
    @brief Deals with the execution of unary and binary element-wise operations
*/

#include "viennacl/forwards.h"
#include "viennacl/scheduler/forwards.h"
#include "viennacl/scheduler/execute_util.hpp"
#include "viennacl/linalg/vector_operations.hpp"
#include "viennacl/linalg/matrix_operations.hpp"

namespace viennacl
{
  namespace scheduler
  {
    namespace detail
    {
      // result = element_op(x,y) for vectors or matrices x, y
      inline void element_op(lhs_rhs_element result,
                             lhs_rhs_element const & x,
                             lhs_rhs_element const & y,
                             operation_node_type  op_type)
      {
        assert(      x.numeric_type == y.numeric_type && bool("Numeric type not the same!"));
        assert( result.numeric_type == y.numeric_type && bool("Numeric type not the same!"));

        assert(      x.type_family == y.type_family && bool("Subtype not the same!"));
        assert( result.type_family == y.type_family && bool("Subtype not the same!"));

        switch (op_type)
        {

        case OPERATION_BINARY_ELEMENT_DIV_TYPE:
          if (x.subtype == DENSE_VECTOR_TYPE)
          {
            switch (x.numeric_type)
            {
              case FLOAT_TYPE:
                viennacl::linalg::element_op(*result.vector_float,
                                             vector_expression<const vector_base<float>,
                                                               const vector_base<float>,
                                                               op_element_binary<op_div> >(*x.vector_float, *y.vector_float));
                break;
              case DOUBLE_TYPE:
                viennacl::linalg::element_op(*result.vector_double,
                                             vector_expression<const vector_base<double>,
                                                               const vector_base<double>,
                                                               op_element_binary<op_div> >(*x.vector_double, *y.vector_double));
                break;
              default:
                throw statement_not_supported_exception("Invalid numeric type for binary elementwise division");
            }
          }
          else if (x.subtype == DENSE_ROW_MATRIX_TYPE)
          {
            switch (x.numeric_type)
            {
              case FLOAT_TYPE:
                viennacl::linalg::element_op(*result.matrix_row_float,
                                             matrix_expression< const matrix_base<float, row_major>,
                                                                const matrix_base<float, row_major>,
                                                                op_element_binary<op_div> >(*x.matrix_row_float, *y.matrix_row_float));
                break;
              case DOUBLE_TYPE:
                viennacl::linalg::element_op(*result.matrix_row_double,
                                             matrix_expression< const matrix_base<double, row_major>,
                                                                const matrix_base<double, row_major>,
                                                                op_element_binary<op_div> >(*x.matrix_row_double, *y.matrix_row_double));
                break;
              default:
                throw statement_not_supported_exception("Invalid numeric type for binary elementwise division");
            }
          }
          else if (x.subtype == DENSE_COL_MATRIX_TYPE)
          {
            switch (x.numeric_type)
            {
              case FLOAT_TYPE:
                viennacl::linalg::element_op(*result.matrix_col_float,
                                             matrix_expression< const matrix_base<float, column_major>,
                                                                const matrix_base<float, column_major>,
                                                                op_element_binary<op_div> >(*x.matrix_col_float, *y.matrix_col_float));
                break;
              case DOUBLE_TYPE:
                viennacl::linalg::element_op(*result.matrix_col_double,
                                             matrix_expression< const matrix_base<double, column_major>,
                                                                const matrix_base<double, column_major>,
                                                                op_element_binary<op_div> >(*x.matrix_col_double, *y.matrix_col_double));
                break;
              default:
                throw statement_not_supported_exception("Invalid numeric type for binary elementwise division");
            }
          }
          else
            throw statement_not_supported_exception("Invalid operand type for binary elementwise division");
          break;


        case OPERATION_BINARY_ELEMENT_PROD_TYPE:
          if (x.subtype == DENSE_VECTOR_TYPE)
          {
            switch (x.numeric_type)
            {
              case FLOAT_TYPE:
                viennacl::linalg::element_op(*result.vector_float,
                                             vector_expression<const vector_base<float>,
                                                               const vector_base<float>,
                                                               op_element_binary<op_prod> >(*x.vector_float, *y.vector_float));
                break;
              case DOUBLE_TYPE:
                viennacl::linalg::element_op(*result.vector_double,
                                             vector_expression<const vector_base<double>,
                                                               const vector_base<double>,
                                                               op_element_binary<op_prod> >(*x.vector_double, *y.vector_double));
                break;
              default:
                throw statement_not_supported_exception("Invalid numeric type for binary elementwise division");
            }
          }
          else if (x.subtype == DENSE_ROW_MATRIX_TYPE)
          {
            switch (x.numeric_type)
            {
              case FLOAT_TYPE:
                viennacl::linalg::element_op(*result.matrix_row_float,
                                             matrix_expression< const matrix_base<float, row_major>,
                                                                const matrix_base<float, row_major>,
                                                                op_element_binary<op_prod> >(*x.matrix_row_float, *y.matrix_row_float));
                break;
              case DOUBLE_TYPE:
                viennacl::linalg::element_op(*result.matrix_row_double,
                                             matrix_expression< const matrix_base<double, row_major>,
                                                                const matrix_base<double, row_major>,
                                                                op_element_binary<op_prod> >(*x.matrix_row_double, *y.matrix_row_double));
                break;
              default:
                throw statement_not_supported_exception("Invalid numeric type for binary elementwise division");
            }
          }
          else if (x.subtype == DENSE_COL_MATRIX_TYPE)
          {
            switch (x.numeric_type)
            {
              case FLOAT_TYPE:
                viennacl::linalg::element_op(*result.matrix_col_float,
                                             matrix_expression< const matrix_base<float, column_major>,
                                                                const matrix_base<float, column_major>,
                                                                op_element_binary<op_prod> >(*x.matrix_col_float, *y.matrix_col_float));
                break;
              case DOUBLE_TYPE:
                viennacl::linalg::element_op(*result.matrix_col_double,
                                             matrix_expression< const matrix_base<double, column_major>,
                                                                const matrix_base<double, column_major>,
                                                                op_element_binary<op_prod> >(*x.matrix_col_double, *y.matrix_col_double));
                break;
              default:
                throw statement_not_supported_exception("Invalid numeric type for binary elementwise division");
            }
          }
          else
            throw statement_not_supported_exception("Invalid operand type for binary elementwise division");
          break;
        default:
          throw statement_not_supported_exception("Invalid operation type for binary elementwise operations");
        }
      }
    }

    /** @brief Deals with x = RHS where RHS is a vector expression */
    inline void execute_element_composite(statement const & s, statement_node const & root_node)
    {
      statement_node const & leaf = s.array()[root_node.rhs.node_index];

      statement_node new_root_lhs;
      statement_node new_root_rhs;

      // check for temporary on lhs:
      if (leaf.lhs.type_family == COMPOSITE_OPERATION_FAMILY)
      {
        detail::new_element(new_root_lhs.lhs, root_node.lhs);

        new_root_lhs.op.type_family = OPERATION_BINARY_TYPE_FAMILY;
        new_root_lhs.op.type        = OPERATION_BINARY_ASSIGN_TYPE;

        new_root_lhs.rhs.type_family  = COMPOSITE_OPERATION_FAMILY;
        new_root_lhs.rhs.subtype      = INVALID_SUBTYPE;
        new_root_lhs.rhs.numeric_type = INVALID_NUMERIC_TYPE;
        new_root_lhs.rhs.node_index   = leaf.lhs.node_index;

        // work on subexpression:
        // TODO: Catch exception, free temporary, then rethrow
        detail::execute_composite(s, new_root_lhs);
      }

      // check for temporary on rhs:
      if (leaf.rhs.type_family == COMPOSITE_OPERATION_FAMILY)
      {
        detail::new_element(new_root_rhs.lhs, root_node.lhs);

        new_root_rhs.op.type_family = OPERATION_BINARY_TYPE_FAMILY;
        new_root_rhs.op.type        = OPERATION_BINARY_ASSIGN_TYPE;

        new_root_rhs.rhs.type_family  = COMPOSITE_OPERATION_FAMILY;
        new_root_rhs.rhs.subtype      = INVALID_SUBTYPE;
        new_root_rhs.rhs.numeric_type = INVALID_NUMERIC_TYPE;
        new_root_rhs.rhs.node_index   = leaf.rhs.node_index;

        // work on subexpression:
        // TODO: Catch exception, free temporary, then rethrow
        detail::execute_composite(s, new_root_rhs);
      }

      if (leaf.op.type == OPERATION_BINARY_ELEMENT_PROD_TYPE || leaf.op.type == OPERATION_BINARY_ELEMENT_DIV_TYPE)
      {
        lhs_rhs_element x = (leaf.lhs.type_family == COMPOSITE_OPERATION_FAMILY) ? new_root_lhs.lhs : leaf.lhs;
        lhs_rhs_element y = (leaf.rhs.type_family == COMPOSITE_OPERATION_FAMILY) ? new_root_rhs.lhs : leaf.rhs;

        // compute element-wise operation:
        detail::element_op(root_node.lhs, x, y, leaf.op.type);
      }
      else if (leaf.op.type_family  == OPERATION_UNARY_TYPE_FAMILY)
      {
        //lhs_rhs_element x = (leaf.lhs.type_family == COMPOSITE_OPERATION_FAMILY) ? new_root_lhs.lhs : leaf.lhs;

        // compute element-wise operation:
        //detail::element_op(new_root_rhs.lhs, x, y, leaf.op.type);

        throw statement_not_supported_exception("TODO");
      }
      else
        throw statement_not_supported_exception("Unsupported operation for scalar.");

      // clean up:
      if (leaf.lhs.type_family == COMPOSITE_OPERATION_FAMILY)
        detail::delete_element(new_root_lhs.lhs);
      if (leaf.rhs.type_family == COMPOSITE_OPERATION_FAMILY)
        detail::delete_element(new_root_rhs.lhs);

    }


  } // namespace scheduler

} // namespace viennacl

#endif

