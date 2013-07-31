#ifndef VIENNACL_SCHEDULER_EXECUTE_MATRIX_HPP
#define VIENNACL_SCHEDULER_EXECUTE_MATRIX_HPP

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


/** @file viennacl/scheduler/execute_matrix.hpp
    @brief Deals with the execution of A = RHS; for a matrix A and any compatible right hand side expression RHS.
*/

#include "viennacl/forwards.h"
#include "viennacl/scheduler/forwards.h"
#include "viennacl/scheduler/execute_matrix_dispatcher.hpp"

namespace viennacl
{
  namespace scheduler
  {
    inline void execute_matrix(statement const & s, statement_node const & root_node);


    /** @brief Deals with x = RHS where RHS is a matrix expression */
    inline void execute_matrix_composite(statement const & s, statement_node const & root_node)
    {
      statement::container_type const & expr = s.array();

      statement_node const & leaf = expr[root_node.rhs.node_index];

      if (leaf.op.type  == OPERATION_BINARY_ADD_TYPE || leaf.op.type  == OPERATION_BINARY_SUB_TYPE) // x = (y) +- (z)  where y and z are either matrices or expressions
      {
        bool flip_sign_z = (leaf.op.type  == OPERATION_BINARY_SUB_TYPE);

        if ( leaf.lhs.type_family == MATRIX_ROW_TYPE_FAMILY || leaf.lhs.type_family == MATRIX_COL_TYPE_FAMILY )
        {
          lhs_rhs_element u = root_node.lhs;
          lhs_rhs_element v = leaf.lhs;
          lhs_rhs_element w = leaf.rhs;
          switch (root_node.op.type)
          {
            case OPERATION_BINARY_ASSIGN_TYPE:
              detail::ambm(u,
                           v, 1.0, 1, false, false,
                           w, 1.0, 1, false, flip_sign_z);
              break;
            case OPERATION_BINARY_INPLACE_ADD_TYPE:
              detail::ambm_m(u,
                             v, 1.0, 1, false, false,
                             w, 1.0, 1, false, flip_sign_z);
              break;
            case OPERATION_BINARY_INPLACE_SUB_TYPE:
              detail::ambm_m(u,
                             v, 1.0, 1, false, true,
                             w, 1.0, 1, false, !flip_sign_z);
              break;
            default:
              throw statement_not_supported_exception("Unsupported binary operator for matrix operation in root note (should be =, +=, or -=)");
          }
        }
        else if (  leaf.lhs.type_family == COMPOSITE_OPERATION_FAMILY
                && (leaf.rhs.type_family == MATRIX_COL_TYPE_FAMILY || leaf.rhs.type_family == MATRIX_ROW_TYPE_FAMILY) ) // x = (y) + z, y being a subtree itself, z being a matrix
        {
          statement_node const & y = expr[leaf.lhs.node_index];

          if (y.op.type_family == OPERATION_BINARY_TYPE_FAMILY)
          {
            // y might be  'A * alpha' or 'A / alpha' with matrix A
            if (   (y.op.type == OPERATION_BINARY_MULT_TYPE     || y.op.type == OPERATION_BINARY_DIV_TYPE)
                && (y.lhs.type_family == MATRIX_ROW_TYPE_FAMILY || y.lhs.type_family == MATRIX_COL_TYPE_FAMILY)
                && (y.rhs.type_family == SCALAR_TYPE_FAMILY     || y.rhs.type_family == HOST_SCALAR_TYPE_FAMILY))
            {
              lhs_rhs_element u = root_node.lhs;
              lhs_rhs_element v = y.lhs;
              lhs_rhs_element w = leaf.rhs;
              lhs_rhs_element alpha = y.rhs;

              bool is_division = (y.op.type == OPERATION_BINARY_DIV_TYPE);
              switch (root_node.op.type)
              {
                case OPERATION_BINARY_ASSIGN_TYPE:
                  detail::ambm(u,
                               v, alpha, 1, is_division, false,
                               w,   1.0, 1, false,       flip_sign_z);
                  break;
                case OPERATION_BINARY_INPLACE_ADD_TYPE:
                  detail::ambm_m(u,
                                 v, alpha, 1, is_division, false,
                                 w,   1.0, 1, false,       flip_sign_z);
                  break;
                case OPERATION_BINARY_INPLACE_SUB_TYPE:
                  detail::ambm_m(u,
                                 v, alpha, 1, is_division, true,
                                 w,   1.0, 1, false,       !flip_sign_z);
                  break;
                default:
                  throw statement_not_supported_exception("Unsupported binary operator for matrix operation in root note (should be =, +=, or -=)");
              }
            }
            else // no built-in kernel, we use a temporary.
            {
              statement_node new_root_y;

              new_root_y.lhs.type_family = root_node.lhs.type_family;
              new_root_y.lhs.type        = root_node.lhs.type;
              detail::new_matrix(new_root_y.lhs, (root_node.lhs.matrix_row_float)->size1(),  (root_node.lhs.matrix_row_float)->size2());

              new_root_y.op.type_family = OPERATION_BINARY_TYPE_FAMILY;
              new_root_y.op.type        = OPERATION_BINARY_ASSIGN_TYPE;

              new_root_y.rhs.type_family = COMPOSITE_OPERATION_FAMILY;
              new_root_y.rhs.type        = COMPOSITE_OPERATION_TYPE;
              new_root_y.rhs.node_index  = leaf.lhs.node_index;

              // work on subexpression:
              // TODO: Catch exception, free temporary, then rethrow
              execute_matrix(s, new_root_y);

              // now add:
              lhs_rhs_element u = root_node.lhs;
              lhs_rhs_element v = new_root_y.lhs;
              lhs_rhs_element w = leaf.rhs;
              switch (root_node.op.type)
              {
                case OPERATION_BINARY_ASSIGN_TYPE:
                  detail::ambm(u,
                               v, 1.0, 1, false, false,
                               w, 1.0, 1, false, flip_sign_z);
                  break;
                case OPERATION_BINARY_INPLACE_ADD_TYPE:
                  detail::ambm_m(u,
                                 v, 1.0, 1, false, false,
                                 w, 1.0, 1, false, flip_sign_z);
                  break;
                case OPERATION_BINARY_INPLACE_SUB_TYPE:
                  detail::ambm_m(u,
                                 v, 1.0, 1, false, true,
                                 w, 1.0, 1, false, !flip_sign_z);
                  break;
                default:
                  throw statement_not_supported_exception("Unsupported binary operator for matrix operation in root note (should be =, +=, or -=)");
              }

              detail::delete_matrix(new_root_y.lhs);
            }
          }
          else
            throw statement_not_supported_exception("Cannot deal with unary operations on matrices");

        }
        else if (  (leaf.lhs.type_family == MATRIX_COL_TYPE_FAMILY || leaf.lhs.type_family == MATRIX_ROW_TYPE_FAMILY)
                && leaf.rhs.type_family == COMPOSITE_OPERATION_FAMILY) // x = y + (z), y being a matrix, z being a subtree itself
        {
          statement_node const & z = expr[leaf.rhs.node_index];

          if (z.op.type_family == OPERATION_BINARY_TYPE_FAMILY)
          {
            // z might be  'A * alpha' or 'A / alpha' with matrix A
            if (   (z.op.type == OPERATION_BINARY_MULT_TYPE     || z.op.type == OPERATION_BINARY_DIV_TYPE)
                && (z.lhs.type_family == MATRIX_COL_TYPE_FAMILY || z.lhs.type_family == MATRIX_ROW_TYPE_FAMILY)
                && (z.rhs.type_family == SCALAR_TYPE_FAMILY     || z.rhs.type_family == HOST_SCALAR_TYPE_FAMILY))
            {
              lhs_rhs_element u = root_node.lhs;
              lhs_rhs_element v = leaf.rhs;
              lhs_rhs_element w = z.lhs;
              lhs_rhs_element beta = z.rhs;

              bool is_division = (z.op.type == OPERATION_BINARY_DIV_TYPE);
              switch (root_node.op.type)
              {
                case OPERATION_BINARY_ASSIGN_TYPE:
                  detail::ambm(u,
                               v,  1.0, 1, false, false,
                               w, beta, 1, is_division, flip_sign_z);
                  break;
                case OPERATION_BINARY_INPLACE_ADD_TYPE:
                  detail::ambm_m(u,
                                 v,  1.0, 1, false, false,
                                 w, beta, 1, is_division, flip_sign_z);
                  break;
                case OPERATION_BINARY_INPLACE_SUB_TYPE:
                  detail::ambm_m(u,
                                 v,  1.0, 1, false, true,
                                 w, beta, 1, is_division, !flip_sign_z);
                  break;
                default:
                  throw statement_not_supported_exception("Unsupported binary operator for matrix operation in root note (should be =, +=, or -=)");
              }
            }
            else // no built-in kernel, we use a temporary.
            {
              statement_node new_root_z;

              new_root_z.lhs.type_family = root_node.lhs.type_family;
              new_root_z.lhs.type        = root_node.lhs.type;
              detail::new_matrix(new_root_z.lhs, (root_node.lhs.matrix_col_float)->size1(), (root_node.lhs.matrix_col_float)->size2());

              new_root_z.op.type_family = OPERATION_BINARY_TYPE_FAMILY;
              new_root_z.op.type        = OPERATION_BINARY_ASSIGN_TYPE;

              new_root_z.rhs.type_family = COMPOSITE_OPERATION_FAMILY;
              new_root_z.rhs.type        = COMPOSITE_OPERATION_TYPE;
              new_root_z.rhs.node_index  = leaf.rhs.node_index;

              // work on subexpression:
              // TODO: Catch exception, free temporary, then rethrow
              execute_matrix(s, new_root_z);

              // now add:
              lhs_rhs_element u = root_node.lhs;
              lhs_rhs_element v = leaf.lhs;
              lhs_rhs_element w = new_root_z.lhs;
              switch (root_node.op.type)
              {
                case OPERATION_BINARY_ASSIGN_TYPE:
                  detail::ambm(u,
                               v, 1.0, 1, false, false,
                               w, 1.0, 1, false, flip_sign_z);
                  break;
                case OPERATION_BINARY_INPLACE_ADD_TYPE:
                  detail::ambm_m(u,
                                 v, 1.0, 1, false, false,
                                 w, 1.0, 1, false, flip_sign_z);
                  break;
                case OPERATION_BINARY_INPLACE_SUB_TYPE:
                  detail::ambm_m(u,
                                 v, 1.0, 1, false, true,
                                 w, 1.0, 1, false, !flip_sign_z);
                  break;
                default:
                  throw statement_not_supported_exception("Unsupported binary operator for matrix operation in root note (should be =, +=, or -=)");
              }

              detail::delete_matrix(new_root_z.lhs);
            }
          }
          else
            throw statement_not_supported_exception("Cannot deal with unary operations on matrices");

        }
        else if (  leaf.lhs.type_family == COMPOSITE_OPERATION_FAMILY
                && leaf.rhs.type_family == COMPOSITE_OPERATION_FAMILY) // x = (y) + (z), y and z being subtrees
        {
          statement_node const & y = expr[leaf.lhs.node_index];
          statement_node const & z = expr[leaf.rhs.node_index];

          if (   y.op.type_family == OPERATION_BINARY_TYPE_FAMILY
              && z.op.type_family == OPERATION_BINARY_TYPE_FAMILY)
          {
            // z might be  'A * alpha' or 'A / alpha' with matrix A
            if (   (y.op.type == OPERATION_BINARY_MULT_TYPE     || y.op.type == OPERATION_BINARY_DIV_TYPE)
                && (y.lhs.type_family == MATRIX_COL_TYPE_FAMILY || y.lhs.type_family == MATRIX_ROW_TYPE_FAMILY)
                && (y.rhs.type_family == SCALAR_TYPE_FAMILY     || y.rhs.type_family == HOST_SCALAR_TYPE_FAMILY)
                && (z.op.type == OPERATION_BINARY_MULT_TYPE     || z.op.type == OPERATION_BINARY_DIV_TYPE)
                && (z.lhs.type_family == MATRIX_COL_TYPE_FAMILY || z.lhs.type_family == MATRIX_ROW_TYPE_FAMILY)
                && (z.rhs.type_family == SCALAR_TYPE_FAMILY     || z.rhs.type_family == HOST_SCALAR_TYPE_FAMILY))
            {
              lhs_rhs_element u = root_node.lhs;
              lhs_rhs_element v = y.lhs;
              lhs_rhs_element w = z.lhs;
              lhs_rhs_element alpha = y.rhs;
              lhs_rhs_element beta  = z.rhs;

              bool is_division_y = (y.op.type == OPERATION_BINARY_DIV_TYPE);
              bool is_division_z = (z.op.type == OPERATION_BINARY_DIV_TYPE);
              switch (root_node.op.type)
              {
                case OPERATION_BINARY_ASSIGN_TYPE:
                  detail::ambm(u,
                               v, alpha, 1, is_division_y, false,
                               w,  beta, 1, is_division_z, flip_sign_z);
                  break;
                case OPERATION_BINARY_INPLACE_ADD_TYPE:
                  detail::ambm_m(u,
                                 v, alpha, 1, is_division_y, false,
                                 w,  beta, 1, is_division_z, flip_sign_z);
                  break;
                case OPERATION_BINARY_INPLACE_SUB_TYPE:
                  detail::ambm_m(u,
                                 v, alpha, 1, is_division_y, true,
                                 w,  beta, 1, is_division_z, !flip_sign_z);
                  break;
                default:
                  throw statement_not_supported_exception("Unsupported binary operator for matrix operation in root note (should be =, +=, or -=)");
              }
            }
            else // no built-in kernel, we use a temporary.
            {
              statement_node new_root_y;

              new_root_y.lhs.type_family = root_node.lhs.type_family;
              new_root_y.lhs.type        = root_node.lhs.type;
              detail::new_matrix(new_root_y.lhs, (root_node.lhs.matrix_col_float)->size1(), (root_node.lhs.matrix_col_float)->size2());

              new_root_y.op.type_family = OPERATION_BINARY_TYPE_FAMILY;
              new_root_y.op.type        = OPERATION_BINARY_ASSIGN_TYPE;

              new_root_y.rhs.type_family = COMPOSITE_OPERATION_FAMILY;
              new_root_y.rhs.type        = COMPOSITE_OPERATION_TYPE;
              new_root_y.rhs.node_index  = leaf.lhs.node_index;

              // work on subexpression:
              // TODO: Catch exception, free temporary, then rethrow
              execute_matrix(s, new_root_y);

              statement_node new_root_z;

              new_root_z.lhs.type_family = root_node.lhs.type_family;
              new_root_z.lhs.type        = root_node.lhs.type;
              detail::new_matrix(new_root_z.lhs, (root_node.lhs.matrix_col_float)->size1(), (root_node.lhs.matrix_col_float)->size2());

              new_root_z.op.type_family = OPERATION_BINARY_TYPE_FAMILY;
              new_root_z.op.type        = OPERATION_BINARY_ASSIGN_TYPE;

              new_root_z.rhs.type_family = COMPOSITE_OPERATION_FAMILY;
              new_root_z.rhs.type        = COMPOSITE_OPERATION_TYPE;
              new_root_z.rhs.node_index  = leaf.rhs.node_index;

              // work on subexpression:
              // TODO: Catch exception, free temporaries, then rethrow
              execute_matrix(s, new_root_z);

              // now add:
              lhs_rhs_element u = root_node.lhs;
              lhs_rhs_element v = new_root_y.lhs;
              lhs_rhs_element w = new_root_z.lhs;

              switch (root_node.op.type)
              {
                case OPERATION_BINARY_ASSIGN_TYPE:
                  detail::ambm(u,
                               v, 1.0, 1, false, false,
                               w, 1.0, 1, false, flip_sign_z);
                  break;
                case OPERATION_BINARY_INPLACE_ADD_TYPE:
                  detail::ambm_m(u,
                                 v, 1.0, 1, false, false,
                                 w, 1.0, 1, false, flip_sign_z);
                  break;
                case OPERATION_BINARY_INPLACE_SUB_TYPE:
                  detail::ambm_m(u,
                                 v, 1.0, 1, false, true,
                                 w, 1.0, 1, false, !flip_sign_z);
                  break;
                default:
                  throw statement_not_supported_exception("Unsupported binary operator for matrices operation in root note (should be =, +=, or -=)");
              }

              detail::delete_matrix(new_root_y.lhs);
              detail::delete_matrix(new_root_z.lhs);
            }
          }
          else
            throw statement_not_supported_exception("Cannot deal with unary operations on matrices");
        }
        else
          throw statement_not_supported_exception("Cannot deal with addition of matrices");
      }
      else if (leaf.op.type  == OPERATION_BINARY_MULT_TYPE || leaf.op.type  == OPERATION_BINARY_DIV_TYPE) // x = y * / alpha;
      {
        if (   (leaf.lhs.type_family == MATRIX_COL_TYPE_FAMILY || leaf.lhs.type_family == MATRIX_ROW_TYPE_FAMILY)
            && (leaf.rhs.type_family == SCALAR_TYPE_FAMILY || leaf.rhs.type_family == HOST_SCALAR_TYPE_FAMILY))
        {
          lhs_rhs_element u = root_node.lhs;
          lhs_rhs_element v = leaf.lhs;
          lhs_rhs_element alpha = leaf.rhs;

          bool is_division = (leaf.op.type  == OPERATION_BINARY_DIV_TYPE);
          switch (root_node.op.type)
          {
            case OPERATION_BINARY_ASSIGN_TYPE:
              detail::am(u,
                         v, alpha, 1, is_division, false);
              break;
            case OPERATION_BINARY_INPLACE_ADD_TYPE:
              detail::ambm(u,
                           u,   1.0, 1, false,       false,
                           v, alpha, 1, is_division, false);
              break;
            case OPERATION_BINARY_INPLACE_SUB_TYPE:
              detail::ambm(u,
                           u,   1.0, 1, false,       false,
                           v, alpha, 1, is_division, true);
              break;
            default:
              throw statement_not_supported_exception("Unsupported binary operator for matrix operation in root note (should be =, +=, or -=)");
          }

        }
        else
          throw statement_not_supported_exception("Unsupported binary operator for OPERATION_BINARY_MULT_TYPE || OPERATION_BINARY_DIV_TYPE on leaf node.");
      }
      else
        throw statement_not_supported_exception("Unsupported binary operator for matrix operations");
    }

    /** @brief Deals with A = B  for a matrix B */
    inline void execute_matrix_matrix(statement const & /*s*/, statement_node const & root_node)
    {
      lhs_rhs_element u = root_node.lhs;
      lhs_rhs_element v = root_node.rhs;
      switch (root_node.op.type)
      {
        case OPERATION_BINARY_ASSIGN_TYPE:
          detail::am(u,
                     v, 1.0, 1, false, false);
          break;
        case OPERATION_BINARY_INPLACE_ADD_TYPE:
          detail::ambm(u,
                       u, 1.0, 1, false, false,
                       v, 1.0, 1, false, false);
          break;
        case OPERATION_BINARY_INPLACE_SUB_TYPE:
          detail::ambm(u,
                       u, 1.0, 1, false, false,
                       v, 1.0, 1, false, true);
          break;
        default:
          throw statement_not_supported_exception("Unsupported binary operator for matrix operation in root note (should be =, +=, or -=)");
      }
    }

    /** @brief Generic dispatcher */
    inline void execute_matrix(statement const & s, statement_node const & root_node)
    {
      switch (root_node.rhs.type_family)
      {
        case COMPOSITE_OPERATION_FAMILY:
          execute_matrix_composite(s, root_node);
          break;
        case MATRIX_COL_TYPE_FAMILY:
        case MATRIX_ROW_TYPE_FAMILY:
          execute_matrix_matrix(s, root_node);
          break;
        default:
          throw statement_not_supported_exception("Invalid rvalue encountered in matrix assignment");
      }
    }


  }

} //namespace viennacl

#endif

