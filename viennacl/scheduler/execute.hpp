#ifndef VIENNACL_SCHEDULER_EXECUTE_HPP
#define VIENNACL_SCHEDULER_EXECUTE_HPP

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


/** @file viennacl/scheduler/execute.hpp
    @brief Provides the datastructures for dealing with a single statement such as 'x = y + z;'
*/

#include "viennacl/forwards.h"
#include "viennacl/scheduler/forwards.h"

#include "viennacl/scheduler/execute_scalar_assign.hpp"
#include "viennacl/scheduler/execute_generic_dispatcher.hpp"

namespace viennacl
{
  namespace scheduler
  {

    namespace detail
    {
      /** @brief Deals with x = RHS where RHS is an expression and x is either a scalar, a vector, or a matrix */
      inline void execute_composite(statement const & s, statement_node const & root_node)
      {
        statement::container_type const & expr = s.array();

        statement_node const & leaf = expr[root_node.rhs.node_index];

        if (leaf.op.type  == OPERATION_BINARY_ADD_TYPE || leaf.op.type  == OPERATION_BINARY_SUB_TYPE) // x = (y) +- (z)  where y and z are either data objects or expressions
        {
          bool flip_sign_z = (leaf.op.type  == OPERATION_BINARY_SUB_TYPE);

          if (   leaf.lhs.type_family != COMPOSITE_OPERATION_FAMILY
              && leaf.lhs.type_family != COMPOSITE_OPERATION_FAMILY)
          {
            lhs_rhs_element u = root_node.lhs;
            lhs_rhs_element v = leaf.lhs;
            lhs_rhs_element w = leaf.rhs;
            switch (root_node.op.type)
            {
              case OPERATION_BINARY_ASSIGN_TYPE:
                detail::axbx(u,
                             v, 1.0, 1, false, false,
                             w, 1.0, 1, false, flip_sign_z);
                break;
              case OPERATION_BINARY_INPLACE_ADD_TYPE:
                detail::axbx_x(u,
                               v, 1.0, 1, false, false,
                               w, 1.0, 1, false, flip_sign_z);
                break;
              case OPERATION_BINARY_INPLACE_SUB_TYPE:
                detail::axbx_x(u,
                               v, 1.0, 1, false, true,
                               w, 1.0, 1, false, !flip_sign_z);
                break;
              default:
                throw statement_not_supported_exception("Unsupported binary operator for operation in root note (should be =, +=, or -=)");
            }
          }
          else if (  leaf.lhs.type_family == COMPOSITE_OPERATION_FAMILY
                  && leaf.rhs.type_family != COMPOSITE_OPERATION_FAMILY) // x = (y) + z, y being a subtree itself, z being a scalar, vector, or matrix
          {
            statement_node const & y = expr[leaf.lhs.node_index];

            if (y.op.type_family == OPERATION_BINARY_TYPE_FAMILY)
            {
              // y might be  'v * alpha' or 'v / alpha' with {scalar|vector|matrix} v
              if (   (y.op.type == OPERATION_BINARY_MULT_TYPE || y.op.type == OPERATION_BINARY_DIV_TYPE)
                  &&  y.lhs.type_family != COMPOSITE_OPERATION_FAMILY
                  &&  y.rhs.type_family == SCALAR_TYPE_FAMILY)
              {
                lhs_rhs_element u = root_node.lhs;
                lhs_rhs_element v = y.lhs;
                lhs_rhs_element w = leaf.rhs;
                lhs_rhs_element alpha = y.rhs;

                bool is_division = (y.op.type == OPERATION_BINARY_DIV_TYPE);
                switch (root_node.op.type)
                {
                  case OPERATION_BINARY_ASSIGN_TYPE:
                    detail::axbx(u,
                                 v, alpha, 1, is_division, false,
                                 w,   1.0, 1, false,       flip_sign_z);
                    break;
                  case OPERATION_BINARY_INPLACE_ADD_TYPE:
                    detail::axbx_x(u,
                                   v, alpha, 1, is_division, false,
                                   w,   1.0, 1, false,       flip_sign_z);
                    break;
                  case OPERATION_BINARY_INPLACE_SUB_TYPE:
                    detail::axbx_x(u,
                                   v, alpha, 1, is_division, true,
                                   w,   1.0, 1, false,       !flip_sign_z);
                    break;
                  default:
                    throw statement_not_supported_exception("Unsupported binary operator for vector operation in root note (should be =, +=, or -=)");
                }
              }
              else // no built-in kernel, we use a temporary.
              {
                statement_node new_root_y;

                new_root_y.lhs.type_family  = root_node.lhs.type_family;
                new_root_y.lhs.subtype      = root_node.lhs.subtype;
                new_root_y.lhs.numeric_type = root_node.lhs.numeric_type;
                detail::new_element(new_root_y.lhs, root_node.lhs);

                new_root_y.op.type_family = OPERATION_BINARY_TYPE_FAMILY;
                new_root_y.op.type        = OPERATION_BINARY_ASSIGN_TYPE;

                new_root_y.rhs.type_family  = COMPOSITE_OPERATION_FAMILY;
                new_root_y.rhs.subtype      = INVALID_SUBTYPE;
                new_root_y.rhs.numeric_type = INVALID_NUMERIC_TYPE;
                new_root_y.rhs.node_index   = leaf.lhs.node_index;

                // work on subexpression:
                // TODO: Catch exception, free temporary, then rethrow
                execute_composite(s, new_root_y);

                // now add:
                lhs_rhs_element u = root_node.lhs;
                lhs_rhs_element v = new_root_y.lhs;
                lhs_rhs_element w = leaf.rhs;
                switch (root_node.op.type)
                {
                  case OPERATION_BINARY_ASSIGN_TYPE:
                    detail::axbx(u,
                                 v, 1.0, 1, false, false,
                                 w, 1.0, 1, false, flip_sign_z);
                    break;
                  case OPERATION_BINARY_INPLACE_ADD_TYPE:
                    detail::axbx_x(u,
                                   v, 1.0, 1, false, false,
                                   w, 1.0, 1, false, flip_sign_z);
                    break;
                  case OPERATION_BINARY_INPLACE_SUB_TYPE:
                    detail::axbx_x(u,
                                   v, 1.0, 1, false, true,
                                   w, 1.0, 1, false, !flip_sign_z);
                    break;
                  default:
                    throw statement_not_supported_exception("Unsupported binary operator for vector operation in root note (should be =, +=, or -=)");
                }

                detail::delete_element(new_root_y.lhs);
              }
            }
            else
              throw statement_not_supported_exception("Cannot deal with unary operations on vectors");

          }
          else if (  leaf.lhs.type_family != COMPOSITE_OPERATION_FAMILY
                  && leaf.rhs.type_family == COMPOSITE_OPERATION_FAMILY) // x = y + (z), y being vector, z being a subtree itself
          {
            statement_node const & z = expr[leaf.rhs.node_index];

            if (z.op.type_family == OPERATION_BINARY_TYPE_FAMILY)
            {
              // z might be  'v * alpha' or 'v / alpha' with vector v
              if (   (z.op.type == OPERATION_BINARY_MULT_TYPE || z.op.type == OPERATION_BINARY_DIV_TYPE)
                  &&  z.lhs.type_family != COMPOSITE_OPERATION_FAMILY
                  &&  z.rhs.type_family == SCALAR_TYPE_FAMILY)
              {
                lhs_rhs_element u = root_node.lhs;
                lhs_rhs_element v = leaf.rhs;
                lhs_rhs_element w = z.lhs;
                lhs_rhs_element beta = z.rhs;

                bool is_division = (z.op.type == OPERATION_BINARY_DIV_TYPE);
                switch (root_node.op.type)
                {
                  case OPERATION_BINARY_ASSIGN_TYPE:
                    detail::axbx(u,
                                 v,  1.0, 1, false, false,
                                 w, beta, 1, is_division, flip_sign_z);
                    break;
                  case OPERATION_BINARY_INPLACE_ADD_TYPE:
                    detail::axbx_x(u,
                                   v,  1.0, 1, false, false,
                                   w, beta, 1, is_division, flip_sign_z);
                    break;
                  case OPERATION_BINARY_INPLACE_SUB_TYPE:
                    detail::axbx_x(u,
                                   v,  1.0, 1, false, true,
                                   w, beta, 1, is_division, !flip_sign_z);
                    break;
                  default:
                    throw statement_not_supported_exception("Unsupported binary operator for vector operation in root note (should be =, +=, or -=)");
                }
              }
              else // no built-in kernel, we use a temporary.
              {
                statement_node new_root_z;

                new_root_z.lhs.type_family  = root_node.lhs.type_family;
                new_root_z.lhs.subtype      = root_node.lhs.subtype;
                new_root_z.lhs.numeric_type = root_node.lhs.numeric_type;
                detail::new_element(new_root_z.lhs, root_node.lhs);

                new_root_z.op.type_family = OPERATION_BINARY_TYPE_FAMILY;
                new_root_z.op.type        = OPERATION_BINARY_ASSIGN_TYPE;

                new_root_z.rhs.type_family  = COMPOSITE_OPERATION_FAMILY;
                new_root_z.rhs.subtype      = INVALID_SUBTYPE;
                new_root_z.rhs.numeric_type = INVALID_NUMERIC_TYPE;
                new_root_z.rhs.node_index   = leaf.rhs.node_index;

                // work on subexpression:
                // TODO: Catch exception, free temporary, then rethrow
                execute_composite(s, new_root_z);

                // now add:
                lhs_rhs_element u = root_node.lhs;
                lhs_rhs_element v = leaf.lhs;
                lhs_rhs_element w = new_root_z.lhs;
                switch (root_node.op.type)
                {
                  case OPERATION_BINARY_ASSIGN_TYPE:
                    detail::axbx(u,
                                 v, 1.0, 1, false, false,
                                 w, 1.0, 1, false, flip_sign_z);
                    break;
                  case OPERATION_BINARY_INPLACE_ADD_TYPE:
                    detail::axbx_x(u,
                                   v, 1.0, 1, false, false,
                                   w, 1.0, 1, false, flip_sign_z);
                    break;
                  case OPERATION_BINARY_INPLACE_SUB_TYPE:
                    detail::axbx_x(u,
                                   v, 1.0, 1, false, true,
                                   w, 1.0, 1, false, !flip_sign_z);
                    break;
                  default:
                    throw statement_not_supported_exception("Unsupported binary operator for vector operation in root note (should be =, +=, or -=)");
                }

                detail::delete_element(new_root_z.lhs);
              }
            }
            else
              throw statement_not_supported_exception("Cannot deal with unary operations on vectors");

          }
          else if (  leaf.lhs.type_family == COMPOSITE_OPERATION_FAMILY
                  && leaf.rhs.type_family == COMPOSITE_OPERATION_FAMILY) // x = (y) + (z), y and z being subtrees
          {
            statement_node const & y = expr[leaf.lhs.node_index];
            statement_node const & z = expr[leaf.rhs.node_index];

            if (   y.op.type_family == OPERATION_BINARY_TYPE_FAMILY
                && z.op.type_family == OPERATION_BINARY_TYPE_FAMILY)
            {
              // z might be  'v * alpha' or 'v / alpha' with vector v
              if (   (y.op.type == OPERATION_BINARY_MULT_TYPE || y.op.type == OPERATION_BINARY_DIV_TYPE)
                  &&  y.lhs.type_family != COMPOSITE_OPERATION_FAMILY
                  &&  y.rhs.type_family == SCALAR_TYPE_FAMILY
                  && (z.op.type == OPERATION_BINARY_MULT_TYPE || z.op.type == OPERATION_BINARY_DIV_TYPE)
                  &&  z.lhs.type_family != COMPOSITE_OPERATION_FAMILY
                  &&  z.rhs.type_family == SCALAR_TYPE_FAMILY)
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
                    detail::axbx(u,
                                 v, alpha, 1, is_division_y, false,
                                 w,  beta, 1, is_division_z, flip_sign_z);
                    break;
                  case OPERATION_BINARY_INPLACE_ADD_TYPE:
                    detail::axbx_x(u,
                                   v, alpha, 1, is_division_y, false,
                                   w,  beta, 1, is_division_z, flip_sign_z);
                    break;
                  case OPERATION_BINARY_INPLACE_SUB_TYPE:
                    detail::axbx_x(u,
                                   v, alpha, 1, is_division_y, true,
                                   w,  beta, 1, is_division_z, !flip_sign_z);
                    break;
                  default:
                    throw statement_not_supported_exception("Unsupported binary operator for vector operation in root note (should be =, +=, or -=)");
                }
              }
              else // no built-in kernel, we use a temporary.
              {
                statement_node new_root_y;

                new_root_y.lhs.type_family  = root_node.lhs.type_family;
                new_root_y.lhs.subtype      = root_node.lhs.subtype;
                new_root_y.lhs.numeric_type = root_node.lhs.numeric_type;
                detail::new_element(new_root_y.lhs, root_node.lhs);

                new_root_y.op.type_family = OPERATION_BINARY_TYPE_FAMILY;
                new_root_y.op.type   = OPERATION_BINARY_ASSIGN_TYPE;

                new_root_y.rhs.type_family  = COMPOSITE_OPERATION_FAMILY;
                new_root_y.rhs.subtype      = INVALID_SUBTYPE;
                new_root_y.rhs.numeric_type = INVALID_NUMERIC_TYPE;
                new_root_y.rhs.node_index   = leaf.lhs.node_index;

                // work on subexpression:
                // TODO: Catch exception, free temporary, then rethrow
                execute_composite(s, new_root_y);

                statement_node new_root_z;

                new_root_z.lhs.type_family  = root_node.lhs.type_family;
                new_root_z.lhs.subtype      = root_node.lhs.subtype;
                new_root_z.lhs.numeric_type = root_node.lhs.numeric_type;
                detail::new_element(new_root_z.lhs, root_node.lhs);

                new_root_z.op.type_family = OPERATION_BINARY_TYPE_FAMILY;
                new_root_z.op.type        = OPERATION_BINARY_ASSIGN_TYPE;

                new_root_z.rhs.type_family  = COMPOSITE_OPERATION_FAMILY;
                new_root_z.rhs.subtype      = INVALID_SUBTYPE;
                new_root_z.rhs.numeric_type = INVALID_NUMERIC_TYPE;
                new_root_z.rhs.node_index   = leaf.rhs.node_index;

                // work on subexpression:
                // TODO: Catch exception, free temporaries, then rethrow
                execute_composite(s, new_root_z);

                // now add:
                lhs_rhs_element u = root_node.lhs;
                lhs_rhs_element v = new_root_y.lhs;
                lhs_rhs_element w = new_root_z.lhs;

                switch (root_node.op.type)
                {
                  case OPERATION_BINARY_ASSIGN_TYPE:
                    detail::axbx(u,
                                 v, 1.0, 1, false, false,
                                 w, 1.0, 1, false, flip_sign_z);
                    break;
                  case OPERATION_BINARY_INPLACE_ADD_TYPE:
                    detail::axbx_x(u,
                                   v, 1.0, 1, false, false,
                                   w, 1.0, 1, false, flip_sign_z);
                    break;
                  case OPERATION_BINARY_INPLACE_SUB_TYPE:
                    detail::axbx_x(u,
                                   v, 1.0, 1, false, true,
                                   w, 1.0, 1, false, !flip_sign_z);
                    break;
                  default:
                    throw statement_not_supported_exception("Unsupported binary operator for vector operation in root note (should be =, +=, or -=)");
                }

                detail::delete_element(new_root_y.lhs);
                detail::delete_element(new_root_z.lhs);
              }
            }
            else
              throw statement_not_supported_exception("Cannot deal with unary operations on vectors");
          }
          else
            throw statement_not_supported_exception("Cannot deal with addition of vectors");
        }
        else if (leaf.op.type == OPERATION_BINARY_MULT_TYPE || leaf.op.type == OPERATION_BINARY_DIV_TYPE) // x = y * / alpha;
        {
          if (   leaf.lhs.type_family != COMPOSITE_OPERATION_FAMILY
              && leaf.rhs.type_family == SCALAR_TYPE_FAMILY)
          {
            lhs_rhs_element u = root_node.lhs;
            lhs_rhs_element v = leaf.lhs;
            lhs_rhs_element alpha = leaf.rhs;

            bool is_division = (leaf.op.type  == OPERATION_BINARY_DIV_TYPE);
            switch (root_node.op.type)
            {
              case OPERATION_BINARY_ASSIGN_TYPE:
                detail::ax(u,
                           v, alpha, 1, is_division, false);
                break;
              case OPERATION_BINARY_INPLACE_ADD_TYPE:
                detail::axbx(u,
                             u,   1.0, 1, false,       false,
                             v, alpha, 1, is_division, false);
                break;
              case OPERATION_BINARY_INPLACE_SUB_TYPE:
                detail::axbx(u,
                             u,   1.0, 1, false,       false,
                             v, alpha, 1, is_division, true);
                break;
              default:
                throw statement_not_supported_exception("Unsupported binary operator for vector operation in root note (should be =, +=, or -=)");
            }

          }
          else
            throw statement_not_supported_exception("Unsupported binary operator for OPERATION_BINARY_MULT_TYPE || OPERATION_BINARY_DIV_TYPE on leaf node.");
        }
        else if (   leaf.op.type == OPERATION_BINARY_INNER_PROD_TYPE
                 || leaf.op.type == OPERATION_UNARY_NORM_1_TYPE
                 || leaf.op.type == OPERATION_UNARY_NORM_2_TYPE
                 || leaf.op.type == OPERATION_UNARY_NORM_INF_TYPE)
        {
          execute_scalar_assign_composite(s, root_node);
        }
        else
          throw statement_not_supported_exception("Unsupported binary operator for vector operations");
      }


      /** @brief Deals with x = y  for a scalar/vector/matrix x, y */
      inline void execute_single(statement const &, statement_node const & root_node)
      {
        lhs_rhs_element u = root_node.lhs;
        lhs_rhs_element v = root_node.rhs;
        switch (root_node.op.type)
        {
          case OPERATION_BINARY_ASSIGN_TYPE:
            detail::ax(u,
                       v, 1.0, 1, false, false);
            break;
          case OPERATION_BINARY_INPLACE_ADD_TYPE:
            detail::axbx(u,
                         u, 1.0, 1, false, false,
                         v, 1.0, 1, false, false);
            break;
          case OPERATION_BINARY_INPLACE_SUB_TYPE:
            detail::axbx(u,
                         u, 1.0, 1, false, false,
                         v, 1.0, 1, false, true);
            break;
          default:
            throw statement_not_supported_exception("Unsupported binary operator for operation in root note (should be =, +=, or -=)");
        }

      }


      inline void execute_impl(statement const & s, statement_node const & root_node)
      {
        typedef statement::container_type   StatementContainer;

        if (   root_node.lhs.type_family != SCALAR_TYPE_FAMILY
            && root_node.lhs.type_family != VECTOR_TYPE_FAMILY
            && root_node.lhs.type_family != MATRIX_TYPE_FAMILY)
          throw statement_not_supported_exception("Unsupported lvalue encountered in head node.");

        switch (root_node.rhs.type_family)
        {
          case COMPOSITE_OPERATION_FAMILY:
            execute_composite(s, root_node);
            break;
          case SCALAR_TYPE_FAMILY:
          case VECTOR_TYPE_FAMILY:
          case MATRIX_TYPE_FAMILY:
            execute_single(s, root_node);
            break;
          default:
            throw statement_not_supported_exception("Invalid rvalue encountered in vector assignment");
        }

      }
    }

    inline void execute(statement const & s)
    {
      // simply start execution from the root node:
      detail::execute_impl(s, s.array()[s.root()]);
    }


  }

} //namespace viennacl

#endif

