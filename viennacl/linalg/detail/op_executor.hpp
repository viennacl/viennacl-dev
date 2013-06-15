#ifndef VIENNACL_LINALG_DETAIL_OP_EXECUTOR_HPP
#define VIENNACL_LINALG_DETAIL_OP_EXECUTOR_HPP

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

#include "viennacl/forwards.h"

namespace viennacl
{
  namespace linalg
  {
    namespace detail
    {
      template <typename T, typename B>
      bool op_aliasing(vector_base<T> const & lhs, B const & b)
      {
        return false;
      }

      template <typename T>
      bool op_aliasing(vector_base<T> const & lhs, vector_base<T> const & b)
      {
        return lhs.handle() == b.handle();
      }

      template <typename T, typename LHS, typename RHS, typename OP>
      bool op_aliasing(vector_base<T> const & lhs, vector_expression<const LHS, const RHS, OP> const & rhs)
      {
        return op_aliasing(lhs, rhs.lhs()) || op_aliasing(lhs, rhs.rhs());
      }



      /** @brief Worker class for decomposing expression templates.
        *
        * @tparam A    Type to which is assigned to
        * @tparam OP   One out of {op_assign, op_inplace_add, op_inplace_sub}
        @ @tparam T    Right hand side of the assignment
      */
      template <typename A, typename OP, typename T>
      struct op_executor {};


      // generic x += vec_expr1 + vec_expr2:
      template <typename A, typename LHS, typename RHS>
      struct op_executor<A, op_inplace_add, vector_expression<const LHS, const RHS, op_add> >
      {
          static void apply(A & lhs, vector_expression<const LHS, const RHS, op_add> const & proxy)
          {
            bool op_aliasing_lhs = op_aliasing(lhs, proxy.lhs());
            bool op_aliasing_rhs = op_aliasing(lhs, proxy.rhs());

            if (op_aliasing_lhs || op_aliasing_rhs)
            {
              A temp(proxy.lhs());
              op_executor<A, op_inplace_add, RHS>::apply(temp, proxy.rhs());
              lhs += temp;
            }
            else
            {
              op_executor<A, op_inplace_add, LHS>::apply(lhs, proxy.lhs());
              op_executor<A, op_inplace_add, RHS>::apply(lhs, proxy.rhs());
            }
          }
      };

      // generic x -= vec_expr1 + vec_expr2:
      template <typename A, typename LHS, typename RHS>
      struct op_executor<A, op_inplace_sub, vector_expression<const LHS, const RHS, op_add> >
      {
          static void apply(A & lhs, vector_expression<const LHS, const RHS, op_add> const & proxy)
          {
            bool op_aliasing_lhs = op_aliasing(lhs, proxy.lhs());
            bool op_aliasing_rhs = op_aliasing(lhs, proxy.rhs());

            if (op_aliasing_lhs || op_aliasing_rhs)
            {
              A temp(proxy.lhs());
              op_executor<A, op_inplace_add, RHS>::apply(temp, proxy.rhs());
              lhs -= temp;
            }
            else
            {
              op_executor<A, op_inplace_sub, LHS>::apply(lhs, proxy.lhs());
              op_executor<A, op_inplace_sub, RHS>::apply(lhs, proxy.rhs());
            }
          }
      };

      ////////////////////////////////////////////////

      // generic x = vec_expr1 - vec_expr2:
      template <typename A, typename LHS, typename RHS>
      struct op_executor<A, op_assign, vector_expression<const LHS, const RHS, op_sub> >
      {
          static void apply(A & lhs, vector_expression<const LHS, const RHS, op_sub> const & proxy)
          {
            bool op_aliasing_lhs = op_aliasing(lhs, proxy.lhs());
            bool op_aliasing_rhs = op_aliasing(lhs, proxy.rhs());

            if (op_aliasing_lhs || op_aliasing_rhs)
            {
              A temp(proxy.lhs());
              op_executor<A, op_inplace_sub, RHS>::apply(temp, proxy.rhs());
              lhs = temp;
            }
            else
            {
              op_executor<A, op_assign, LHS>::apply(lhs, proxy.lhs());
              op_executor<A, op_inplace_sub, RHS>::apply(lhs, proxy.rhs());
            }
          }
      };

      // generic x += vec_expr1 - vec_expr2:
      template <typename A, typename LHS, typename RHS>
      struct op_executor<A, op_inplace_add, vector_expression<const LHS, const RHS, op_sub> >
      {
          static void apply(A & lhs, vector_expression<const LHS, const RHS, op_sub> const & proxy)
          {
            bool op_aliasing_lhs = op_aliasing(lhs, proxy.lhs());
            bool op_aliasing_rhs = op_aliasing(lhs, proxy.rhs());

            if (op_aliasing_lhs || op_aliasing_rhs)
            {
              A temp(proxy.lhs());
              op_executor<A, op_inplace_sub, RHS>::apply(temp, proxy.rhs());
              lhs += temp;
            }
            else
            {
              op_executor<A, op_inplace_add, LHS>::apply(lhs, proxy.lhs());
              op_executor<A, op_inplace_sub, RHS>::apply(lhs, proxy.rhs());
            }
          }
      };

      // generic x -= vec_expr1 - vec_expr2:
      template <typename A, typename LHS, typename RHS>
      struct op_executor<A, op_inplace_sub, vector_expression<const LHS, const RHS, op_sub> >
      {
          static void apply(A & lhs, vector_expression<const LHS, const RHS, op_sub> const & proxy)
          {
            bool op_aliasing_lhs = op_aliasing(lhs, proxy.lhs());
            bool op_aliasing_rhs = op_aliasing(lhs, proxy.rhs());

            if (op_aliasing_lhs || op_aliasing_rhs)
            {
              A temp(proxy.lhs());
              op_executor<A, op_inplace_sub, RHS>::apply(temp, proxy.rhs());
              lhs -= temp;
            }
            else
            {
              op_executor<A, op_inplace_sub, LHS>::apply(lhs, proxy.lhs());
              op_executor<A, op_inplace_add, RHS>::apply(lhs, proxy.rhs());
            }
          }
      };

    }
  }
}

#endif // VIENNACL_LINALG_DETAIL_OP_EXECUTOR_HPP
