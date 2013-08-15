#ifndef VIENNACL_SCHEDULER_EXECUTE_MATRIX_PROD_HPP
#define VIENNACL_SCHEDULER_EXECUTE_MATRIX_PROD_HPP

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


/** @file viennacl/scheduler/execute_matrix_prod.hpp
    @brief Deals with matrix-vector and matrix-matrix products.
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
      inline bool matrix_prod_temporary_required(statement const & s, lhs_rhs_element const & elem)
      {
        if (elem.type_family != COMPOSITE_OPERATION_FAMILY)
          return false;

        // check composite node for being a transposed matrix proxy:
        statement_node const & leaf = s.array()[elem.node_index];
        if (   leaf.op.type == OPERATION_UNARY_TRANS_TYPE && leaf.lhs.type_family == MATRIX_TYPE_FAMILY)
          return false;

        return true;
      }

      inline void matrix_matrix_prod(statement const & s,
                                     lhs_rhs_element result,
                                     lhs_rhs_element const & A,
                                     lhs_rhs_element const & B)
      {
        assert(      A.numeric_type == B.numeric_type && bool("Numeric type not the same!"));
        assert( result.numeric_type == B.numeric_type && bool("Numeric type not the same!"));

        assert( result.type_family == B.type_family && bool("Subtype not the same!"));

        // switch: numeric type
        if (A.numeric_type == FLOAT_TYPE)
        {
          // switch: trans for A, B
          throw statement_not_supported_exception("TODO");
        }
        else if (A.numeric_type == DOUBLE_TYPE)
        {
          throw statement_not_supported_exception("TODO");
        }
        else
          throw statement_not_supported_exception("Invalid numeric type in matrix-{matrix,vector} multiplication");

      }

      inline void matrix_vector_prod(statement const & s,
                                     lhs_rhs_element result,
                                     lhs_rhs_element const & A,
                                     lhs_rhs_element const & x)
      {
        assert( result.numeric_type == x.numeric_type && bool("Numeric type not the same!"));
        assert( result.type_family == x.type_family && bool("Subtype not the same!"));
        assert( result.subtype == DENSE_VECTOR_TYPE && bool("Result node for matrix-vector product not a vector type!"));

        // deal with transposed product first:
        // switch: trans for A
        if (A.type_family == COMPOSITE_OPERATION_FAMILY) // prod(trans(A), x)
        {
          statement_node const & leaf = s.array()[A.node_index];

          assert(leaf.lhs.type_family  == MATRIX_TYPE_FAMILY && bool("Logic error: Argument not a matrix transpose!"));
          assert(leaf.lhs.numeric_type == x.numeric_type && bool("Numeric type not the same!"));

          if (leaf.lhs.subtype == DENSE_ROW_MATRIX_TYPE)
          {
            switch (leaf.lhs.numeric_type)
            {
            case FLOAT_TYPE:
              viennacl::linalg::prod_impl(viennacl::matrix_expression< const matrix_base<float, row_major>,
                                                                       const matrix_base<float, row_major>,
                                                                       op_trans>(*leaf.lhs.matrix_row_float, *leaf.lhs.matrix_row_float),
                                          *x.vector_float,
                                          *result.vector_float); break;
            case DOUBLE_TYPE:
              viennacl::linalg::prod_impl(viennacl::matrix_expression< const matrix_base<double, row_major>,
                                                                       const matrix_base<double, row_major>,
                                                                       op_trans>(*leaf.lhs.matrix_row_double, *leaf.lhs.matrix_row_double),
                                          *x.vector_double,
                                          *result.vector_double); break;
            default:
              throw statement_not_supported_exception("Invalid numeric type in matrix-{matrix,vector} multiplication");
            }
          }
          else if (leaf.lhs.subtype == DENSE_COL_MATRIX_TYPE)
          {
            switch (leaf.lhs.numeric_type)
            {
            case FLOAT_TYPE:
              viennacl::linalg::prod_impl(viennacl::matrix_expression< const matrix_base<float, column_major>,
                                                                       const matrix_base<float, column_major>,
                                                                       op_trans>(*leaf.lhs.matrix_col_float, *leaf.lhs.matrix_col_float),
                                          *x.vector_float,
                                          *result.vector_float); break;
            case DOUBLE_TYPE:
              viennacl::linalg::prod_impl(viennacl::matrix_expression< const matrix_base<double, column_major>,
                                                                       const matrix_base<double, column_major>,
                                                                       op_trans>(*leaf.lhs.matrix_col_double, *leaf.lhs.matrix_col_double),
                                          *x.vector_double,
                                          *result.vector_double); break;
            default:
              throw statement_not_supported_exception("Invalid numeric type in matrix-{matrix,vector} multiplication");
            }
          }
          else
            throw statement_not_supported_exception("Invalid matrix type for transposed matrix-vector product");
        }
        else if (A.subtype == DENSE_ROW_MATRIX_TYPE)
        {
          switch (A.numeric_type)
          {
          case FLOAT_TYPE:
            viennacl::linalg::prod_impl(*A.matrix_row_float, *x.vector_float, *result.vector_float);
            break;
          case DOUBLE_TYPE:
            viennacl::linalg::prod_impl(*A.matrix_row_double, *x.vector_double, *result.vector_double);
            break;
          default:
            throw statement_not_supported_exception("Invalid numeric type in matrix-{matrix,vector} multiplication");
          }
        }
        else if (A.subtype == DENSE_COL_MATRIX_TYPE)
        {
          switch (A.numeric_type)
          {
          case FLOAT_TYPE:
            viennacl::linalg::prod_impl(*A.matrix_col_float, *x.vector_float, *result.vector_float);
            break;
          case DOUBLE_TYPE:
            viennacl::linalg::prod_impl(*A.matrix_col_double, *x.vector_double, *result.vector_double);
            break;
          default:
            throw statement_not_supported_exception("Invalid numeric type in matrix-{matrix,vector} multiplication");
          }
        }
        else
        {
          std::cout << "A.subtype: " << A.subtype << std::endl;
          throw statement_not_supported_exception("Invalid matrix type for matrix-vector product");
        }
      }

    } // namespace detail

    inline void execute_matrix_prod(statement const & s, statement_node const & root_node)
    {
      statement_node const & leaf = s.array()[root_node.rhs.node_index];

      // Part 1: Check whether temporaries are required //

      statement_node new_root_lhs;
      statement_node new_root_rhs;

      bool lhs_needs_temporary = detail::matrix_prod_temporary_required(s, leaf.lhs);
      bool rhs_needs_temporary = detail::matrix_prod_temporary_required(s, leaf.rhs);

      // check for temporary on lhs:
      if (lhs_needs_temporary)
      {
        std::cout << "Temporary for LHS!" << std::endl;
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
      if (rhs_needs_temporary)
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

      // Part 2: Run the actual computations //

      lhs_rhs_element x = lhs_needs_temporary ? new_root_lhs.lhs : leaf.lhs;
      lhs_rhs_element y = rhs_needs_temporary ? new_root_rhs.lhs : leaf.rhs;

      if (root_node.lhs.type_family == VECTOR_TYPE_FAMILY)
        detail::matrix_vector_prod(s, root_node.lhs, x, y);
      else
        detail::matrix_matrix_prod(s, root_node.lhs, x, y);

      // Part 3: Clean up //

      if (lhs_needs_temporary)
        detail::delete_element(new_root_lhs.lhs);

      if (rhs_needs_temporary)
        detail::delete_element(new_root_rhs.lhs);
    }

  } // namespace scheduler
} // namespace viennacl

#endif

