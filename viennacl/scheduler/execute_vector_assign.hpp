#ifndef VIENNACL_SCHEDULER_EXECUTE_VECTOR_ASSIGN_HPP
#define VIENNACL_SCHEDULER_EXECUTE_VECTOR_ASSIGN_HPP

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
#include "viennacl/scheduler/execute_vector_dispatcher.hpp"

namespace viennacl
{
  namespace scheduler
  {
    namespace detail
    {
      inline void new_vector_on_lhs(statement_node & node, std::size_t size)
      {
        switch (node.lhs_type)
        {
          case VECTOR_FLOAT_TYPE:
            node.lhs.vector_float = new viennacl::vector<float>(size);
            return;
          case VECTOR_DOUBLE_TYPE:
            node.lhs.vector_double = new viennacl::vector<double>(size);
            return;
          default:
            throw statement_not_supported_exception("Invalid vector type for vector construction");
        }
      }

      inline void delete_vector_on_lhs(statement_node & node)
      {
        switch (node.lhs_type)
        {
          case VECTOR_FLOAT_TYPE:
            delete node.lhs.vector_float;
            return;
          case VECTOR_DOUBLE_TYPE:
            delete node.lhs.vector_double;
            return;
          default:
            throw statement_not_supported_exception("Invalid vector type for vector construction");
        }
      }
    }


    // forward declaration
    inline void execute_vector_assign(statement const & s, statement_node const & root_node);


    /** @brief Deals with x = RHS where RHS is a vector expression */
    inline void execute_vector_assign_composite(statement const & s, statement_node const & root_node)
    {
      statement::container_type const & expr = s.array();

      statement_node const & leaf = expr[root_node.rhs.node_index];

      if (leaf.op_type  == OPERATION_BINARY_ADD_TYPE || leaf.op_type  == OPERATION_BINARY_SUB_TYPE) // x = (y) +- (z)  where y and z are either vectors or expressions
      {
        bool flip_sign_z = (leaf.op_type  == OPERATION_BINARY_SUB_TYPE);

        if (   leaf.lhs_type_family == VECTOR_TYPE_FAMILY
            && leaf.lhs_type_family == VECTOR_TYPE_FAMILY)
        {
          lhs_rhs_element_2 u; u.type_family = VECTOR_TYPE_FAMILY; u.type = root_node.lhs_type; u.data = root_node.lhs;
          lhs_rhs_element_2 v; v.type_family = VECTOR_TYPE_FAMILY; v.type = leaf.lhs_type;      v.data = leaf.lhs;
          lhs_rhs_element_2 w; w.type_family = VECTOR_TYPE_FAMILY; w.type = leaf.rhs_type;      w.data = leaf.rhs;
          detail::avbv(u,
                       v, 1.0, 1, false, false,
                       w, 1.0, 1, false, flip_sign_z);
        }
        else if (  leaf.lhs_type_family == COMPOSITE_OPERATION_FAMILY
                && leaf.rhs_type_family == VECTOR_TYPE_FAMILY) // x = (y) + z, y being a subtree itself, z being a vector
        {
          statement_node const & y = expr[leaf.lhs.node_index];

          if (y.op_family == OPERATION_BINARY_TYPE_FAMILY)
          {
            // y might be  'v * alpha' or 'v / alpha' with vector v
            if (   (y.op_type == OPERATION_BINARY_MULT_TYPE || y.op_type == OPERATION_BINARY_DIV_TYPE)
                &&  y.lhs_type_family == VECTOR_TYPE_FAMILY
                && (y.rhs_type_family == SCALAR_TYPE_FAMILY || y.rhs_type_family == HOST_SCALAR_TYPE_FAMILY))
            {
              lhs_rhs_element_2 u;         u.type_family = VECTOR_TYPE_FAMILY;    u.type = root_node.lhs_type; u.data = root_node.lhs;
              lhs_rhs_element_2 v;         v.type_family = VECTOR_TYPE_FAMILY;    v.type = y.lhs_type;         v.data = y.lhs;
              lhs_rhs_element_2 w;         w.type_family = VECTOR_TYPE_FAMILY;    w.type = leaf.rhs_type;      w.data = leaf.rhs;
              lhs_rhs_element_2 alpha; alpha.type_family = y.rhs_type_family; alpha.type = y.rhs_type;     alpha.data = y.rhs;

              bool is_division = (y.op_type == OPERATION_BINARY_DIV_TYPE);
              detail::avbv(u,
                           v, alpha, 1, is_division, false,
                           w,   1.0, 1, false,       flip_sign_z);
            }
            else // no built-in kernel, we use a temporary.
            {
              statement_node new_root_y;

              new_root_y.lhs_type_family = root_node.lhs_type_family;
              new_root_y.lhs_type        = root_node.lhs_type;
              detail::new_vector_on_lhs(new_root_y, (root_node.lhs.vector_float)->size());

              new_root_y.op_family = OPERATION_BINARY_TYPE_FAMILY;
              new_root_y.op_type   = OPERATION_BINARY_ASSIGN_TYPE;

              new_root_y.rhs_type_family = COMPOSITE_OPERATION_FAMILY;
              new_root_y.rhs_type        = COMPOSITE_OPERATION_TYPE;
              new_root_y.rhs.node_index  = leaf.lhs.node_index;

              // work on subexpression:
              // TODO: Catch exception, free temporary, then rethrow
              execute_vector_assign(s, new_root_y);

              // now add:
              lhs_rhs_element_2 u; u.type_family = VECTOR_TYPE_FAMILY; u.type = root_node.lhs_type;  u.data = root_node.lhs;
              lhs_rhs_element_2 v; v.type_family = VECTOR_TYPE_FAMILY; v.type = new_root_y.lhs_type; v.data = new_root_y.lhs;
              lhs_rhs_element_2 w; w.type_family = VECTOR_TYPE_FAMILY; w.type = leaf.rhs_type;       w.data = leaf.rhs;
              detail::avbv(u,
                           v, 1.0, 1, false, false,
                           w, 1.0, 1, false, flip_sign_z);

              detail::delete_vector_on_lhs(new_root_y);
            }
          }
          else
            throw statement_not_supported_exception("Cannot deal with unary operations on vectors");

        }
        else if (  leaf.lhs_type_family == VECTOR_TYPE_FAMILY
                && leaf.rhs_type_family == COMPOSITE_OPERATION_FAMILY) // x = y + (z), y being vector, z being a subtree itself
        {
          statement_node const & z = expr[leaf.rhs.node_index];

          if (z.op_family == OPERATION_BINARY_TYPE_FAMILY)
          {
            // z might be  'v * alpha' or 'v / alpha' with vector v
            if (   (z.op_type == OPERATION_BINARY_MULT_TYPE || z.op_type == OPERATION_BINARY_DIV_TYPE)
                &&  z.lhs_type_family == VECTOR_TYPE_FAMILY
                && (z.rhs_type_family == SCALAR_TYPE_FAMILY || z.rhs_type_family == HOST_SCALAR_TYPE_FAMILY))
            {
              lhs_rhs_element_2 u;         u.type_family = VECTOR_TYPE_FAMILY;    u.type = root_node.lhs_type; u.data = root_node.lhs;
              lhs_rhs_element_2 v;         v.type_family = VECTOR_TYPE_FAMILY;    v.type = leaf.rhs_type;      v.data = leaf.rhs;
              lhs_rhs_element_2 w;         w.type_family = VECTOR_TYPE_FAMILY;    w.type = z.lhs_type;         w.data = z.lhs;
              lhs_rhs_element_2 beta;   beta.type_family = z.rhs_type_family;  beta.type = z.rhs_type;      beta.data = z.rhs;

              bool is_division = (z.op_type == OPERATION_BINARY_DIV_TYPE);
              detail::avbv(u,
                           v,  1.0, 1, false, false,
                           w, beta, 1, is_division, flip_sign_z);
            }
            else // no built-in kernel, we use a temporary.
            {
              statement_node new_root_z;

              new_root_z.lhs_type_family = root_node.lhs_type_family;
              new_root_z.lhs_type        = root_node.lhs_type;
              detail::new_vector_on_lhs(new_root_z, (root_node.lhs.vector_float)->size());

              new_root_z.op_family = OPERATION_BINARY_TYPE_FAMILY;
              new_root_z.op_type   = OPERATION_BINARY_ASSIGN_TYPE;

              new_root_z.rhs_type_family = COMPOSITE_OPERATION_FAMILY;
              new_root_z.rhs_type        = COMPOSITE_OPERATION_TYPE;
              new_root_z.rhs.node_index  = leaf.rhs.node_index;

              // work on subexpression:
              // TODO: Catch exception, free temporary, then rethrow
              execute_vector_assign(s, new_root_z);

              // now add:
              lhs_rhs_element_2 u; u.type_family = VECTOR_TYPE_FAMILY; u.type = root_node.lhs_type;  u.data = root_node.lhs;
              lhs_rhs_element_2 v; v.type_family = VECTOR_TYPE_FAMILY; v.type = leaf.rhs_type;       v.data = leaf.lhs;
              lhs_rhs_element_2 w; w.type_family = VECTOR_TYPE_FAMILY; w.type = new_root_z.lhs_type; w.data = new_root_z.lhs;
              detail::avbv(u,
                           v, 1.0, 1, false, false,
                           w, 1.0, 1, false, flip_sign_z);

              detail::delete_vector_on_lhs(new_root_z);
            }
          }
          else
            throw statement_not_supported_exception("Cannot deal with unary operations on vectors");

        }
        else if (  leaf.lhs_type_family == COMPOSITE_OPERATION_FAMILY
                && leaf.rhs_type_family == COMPOSITE_OPERATION_FAMILY) // x = (y) + (z), y and z being subtrees
        {
          statement_node const & y = expr[leaf.lhs.node_index];
          statement_node const & z = expr[leaf.rhs.node_index];

          if (   y.op_family == OPERATION_BINARY_TYPE_FAMILY
              && z.op_family == OPERATION_BINARY_TYPE_FAMILY)
          {
            // z might be  'v * alpha' or 'v / alpha' with vector v
            if (   (y.op_type == OPERATION_BINARY_MULT_TYPE || y.op_type == OPERATION_BINARY_DIV_TYPE)
                &&  y.lhs_type_family == VECTOR_TYPE_FAMILY
                && (y.rhs_type_family == SCALAR_TYPE_FAMILY || y.rhs_type_family == HOST_SCALAR_TYPE_FAMILY)
                && (z.op_type == OPERATION_BINARY_MULT_TYPE || z.op_type == OPERATION_BINARY_DIV_TYPE)
                &&  z.lhs_type_family == VECTOR_TYPE_FAMILY
                && (z.rhs_type_family == SCALAR_TYPE_FAMILY || z.rhs_type_family == HOST_SCALAR_TYPE_FAMILY))
            {
              lhs_rhs_element_2 u;         u.type_family = VECTOR_TYPE_FAMILY;    u.type = root_node.lhs_type; u.data = root_node.lhs;
              lhs_rhs_element_2 v;         v.type_family = VECTOR_TYPE_FAMILY;    v.type = y.lhs_type;         v.data = y.lhs;
              lhs_rhs_element_2 w;         w.type_family = VECTOR_TYPE_FAMILY;    w.type = z.rhs_type;         w.data = z.rhs;
              lhs_rhs_element_2 alpha; alpha.type_family = y.rhs_type_family; alpha.type = y.rhs_type;     alpha.data = y.rhs;
              lhs_rhs_element_2 beta;   beta.type_family = z.rhs_type_family;  beta.type = z.rhs_type;      beta.data = z.rhs;

              bool is_division_y = (y.op_type == OPERATION_BINARY_DIV_TYPE);
              bool is_division_z = (z.op_type == OPERATION_BINARY_DIV_TYPE);
              detail::avbv(u,
                           v, alpha, 1, is_division_y, false,
                           w,  beta, 1, is_division_z, flip_sign_z);
            }
            else // no built-in kernel, we use a temporary.
            {
              statement_node new_root_y;

              new_root_y.lhs_type_family = root_node.lhs_type_family;
              new_root_y.lhs_type        = root_node.lhs_type;
              detail::new_vector_on_lhs(new_root_y, (root_node.lhs.vector_float)->size());

              new_root_y.op_family = OPERATION_BINARY_TYPE_FAMILY;
              new_root_y.op_type   = OPERATION_BINARY_ASSIGN_TYPE;

              new_root_y.rhs_type_family = COMPOSITE_OPERATION_FAMILY;
              new_root_y.rhs_type        = COMPOSITE_OPERATION_TYPE;
              new_root_y.rhs.node_index  = leaf.lhs.node_index;

              // work on subexpression:
              // TODO: Catch exception, free temporary, then rethrow
              execute_vector_assign(s, new_root_y);

              statement_node new_root_z;

              new_root_z.lhs_type_family = root_node.lhs_type_family;
              new_root_z.lhs_type        = root_node.lhs_type;
              detail::new_vector_on_lhs(new_root_z, (root_node.lhs.vector_float)->size());

              new_root_z.op_family = OPERATION_BINARY_TYPE_FAMILY;
              new_root_z.op_type   = OPERATION_BINARY_ASSIGN_TYPE;

              new_root_z.rhs_type_family = COMPOSITE_OPERATION_FAMILY;
              new_root_z.rhs_type        = COMPOSITE_OPERATION_TYPE;
              new_root_z.rhs.node_index  = leaf.rhs.node_index;

              // work on subexpression:
              // TODO: Catch exception, free temporaries, then rethrow
              execute_vector_assign(s, new_root_z);

              // now add:
              lhs_rhs_element_2 u; u.type_family = VECTOR_TYPE_FAMILY; u.type = root_node.lhs_type;  u.data = root_node.lhs;
              lhs_rhs_element_2 v; v.type_family = VECTOR_TYPE_FAMILY; v.type = new_root_y.rhs_type; v.data = new_root_y.lhs;
              lhs_rhs_element_2 w; w.type_family = VECTOR_TYPE_FAMILY; w.type = new_root_z.lhs_type; w.data = new_root_z.lhs;
              detail::avbv(u,
                           v, 1.0, 1, false, false,
                           w, 1.0, 1, false, flip_sign_z);

              detail::delete_vector_on_lhs(new_root_y);
              detail::delete_vector_on_lhs(new_root_z);
            }
          }
          else
            throw statement_not_supported_exception("Cannot deal with unary operations on vectors");
        }
        else
          throw statement_not_supported_exception("Cannot deal with addition of vectors");
      }
      else
        throw statement_not_supported_exception("Unsupported binary operator for vector operations");
    }

    /** @brief Deals with x = y  for a vector y */
    inline void execute_vector_assign_vector(statement const & s, statement_node const & root_node)
    {
      lhs_rhs_element_2 u; u.type_family = VECTOR_TYPE_FAMILY; u.type = root_node.lhs_type; u.data = root_node.lhs;
      lhs_rhs_element_2 v; v.type_family = VECTOR_TYPE_FAMILY; v.type = root_node.rhs_type; v.data = root_node.rhs;
      detail::av(u,
                 v, 1.0, 1, false, false);
    }

    /** @brief Generic dispatcher */
    inline void execute_vector_assign(statement const & s, statement_node const & root_node)
    {
      switch (root_node.rhs_type_family)
      {
        case COMPOSITE_OPERATION_FAMILY:
          execute_vector_assign_composite(s, root_node);
          break;
        case VECTOR_TYPE_FAMILY:
          execute_vector_assign_vector(s, root_node);
          break;
        default:
          throw statement_not_supported_exception("Invalid rvalue encountered in vector assignment");
      }
    }


  }

} //namespace viennacl

#endif

