#ifndef VIENNACL_SCHEDULER_EXECUTE_MATRIX_ROW_ASSIGN_HPP
#define VIENNACL_SCHEDULER_EXECUTE_MATRIX_ROW_ASSIGN_HPP

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


/** @file viennacl/scheduler/execute_matrix_row_assign.hpp
    @brief Deals with the execution of A = RHS; for a matrix A and any compatible right hand side expression RHS.
*/

#include "viennacl/forwards.h"
#include "viennacl/scheduler/forwards.h"
#include "viennacl/linalg/matrix_operations.hpp"

namespace viennacl
{
  namespace scheduler
  {
    /** @brief Deals with x = RHS where RHS is a vector expression */
    inline void execute_matrix_row_assign_composite(statement const & s, statement_node const & root_node)
    {
      statement::container_type const & expr = s.array();

      if (expr[1].op_type  == OPERATION_BINARY_ADD_TYPE)
      {
        if (expr[0].lhs_type == MATRIX_ROW_FLOAT_TYPE
            && expr[1].lhs_type == MATRIX_ROW_FLOAT_TYPE
            && expr[1].rhs_type == MATRIX_ROW_FLOAT_TYPE)
        {
          viennacl::matrix_base<float, viennacl::row_major>       & A = *(expr[0].lhs.matrix_row_float);
          viennacl::matrix_base<float, viennacl::row_major> const & B = *(expr[1].lhs.matrix_row_float);
          viennacl::matrix_base<float, viennacl::row_major> const & C = *(expr[1].rhs.matrix_row_float);
          viennacl::linalg::ambm(A,
                                 B, 1.0, 1, false, false,
                                 C, 1.0, 1, false, false);
        }
        else if (expr[0].lhs_type == MATRIX_ROW_DOUBLE_TYPE
                 && expr[1].lhs_type == MATRIX_ROW_DOUBLE_TYPE
                 && expr[1].rhs_type == MATRIX_ROW_DOUBLE_TYPE)
        {
          viennacl::matrix_base<double, viennacl::row_major>       & A = *(expr[0].lhs.matrix_row_double);
          viennacl::matrix_base<double, viennacl::row_major> const & B = *(expr[1].lhs.matrix_row_double);
          viennacl::matrix_base<double, viennacl::row_major> const & C = *(expr[1].rhs.matrix_row_double);
          viennacl::linalg::ambm(A,
                                 B, 1.0, 1, false, false,
                                 C, 1.0, 1, false, false);
        }
        else
          throw statement_not_supported_exception("Cannot deal with addition of row-major matrix");
      }
      else if (expr[1].op_type  == OPERATION_BINARY_SUB_TYPE)
      {
        if (expr[0].lhs_type == MATRIX_ROW_FLOAT_TYPE
            && expr[1].lhs_type == MATRIX_ROW_FLOAT_TYPE
            && expr[1].rhs_type == MATRIX_ROW_FLOAT_TYPE)
        {
          viennacl::matrix_base<float, viennacl::row_major>       & A = *(expr[0].lhs.matrix_row_float);
          viennacl::matrix_base<float, viennacl::row_major> const & B = *(expr[1].lhs.matrix_row_float);
          viennacl::matrix_base<float, viennacl::row_major> const & C = *(expr[1].rhs.matrix_row_float);
          viennacl::linalg::ambm(A,
                                 B,  1.0, 1, false, false,
                                 C, -1.0, 1, false, false);
        }
        else if (expr[0].lhs_type == MATRIX_ROW_DOUBLE_TYPE
                 && expr[1].lhs_type == MATRIX_ROW_DOUBLE_TYPE
                 && expr[1].rhs_type == MATRIX_ROW_DOUBLE_TYPE)
        {
          viennacl::matrix_base<double, viennacl::row_major>       & A = *(expr[0].lhs.matrix_row_double);
          viennacl::matrix_base<double, viennacl::row_major> const & B = *(expr[1].lhs.matrix_row_double);
          viennacl::matrix_base<double, viennacl::row_major> const & C = *(expr[1].rhs.matrix_row_double);
          viennacl::linalg::ambm(A,
                                 B,  1.0, 1, false, false,
                                 C, -1.0, 1, false, false);
        }
        else
          throw statement_not_supported_exception("Cannot deal with subtraction of row-major matrix");
      }
      else
        throw statement_not_supported_exception("Unsupported binary operator for row-major matrix operations");
    }

    /** @brief Deals with A = B  for a matrix B */
    inline void execute_matrix_row_assign_matrix(statement const & s, statement_node const & root_node)
    {
      typedef statement::container_type   StatementContainer;

      StatementContainer const & expr = s.array();

      if (expr[0].lhs_type == MATRIX_ROW_FLOAT_TYPE && expr[0].rhs_type == MATRIX_ROW_FLOAT_TYPE)
      {
        viennacl::matrix_base<float, viennacl::row_major>       & A = *(expr[0].lhs.matrix_row_float);
        viennacl::matrix_base<float, viennacl::row_major> const & B = *(expr[0].rhs.matrix_row_float);
        viennacl::linalg::am(A,
                             B, 1.0, 1, false, false);
      }
      else if (expr[0].lhs_type == MATRIX_ROW_DOUBLE_TYPE && expr[0].rhs_type == MATRIX_ROW_DOUBLE_TYPE)
      {
        viennacl::matrix_base<double, viennacl::row_major>       & A = *(expr[0].lhs.matrix_row_double);
        viennacl::matrix_base<double, viennacl::row_major> const & B = *(expr[0].rhs.matrix_row_double);
        viennacl::linalg::am(A,
                             B, 1.0, 1, false, false);
      }
      else
        throw statement_not_supported_exception("Unsupported assignment to row-major matrix");
    }

    /** @brief Generic dispatcher */
    inline void execute_matrix_row_assign(statement const & s, statement_node const & root_node)
    {
      typedef statement::container_type   StatementContainer;

      StatementContainer const & expr = s.array();

      switch (expr[0].rhs_type_family)
      {
        case COMPOSITE_OPERATION_FAMILY:
          execute_matrix_row_assign_composite(s, root_node);
          break;
        case MATRIX_ROW_TYPE_FAMILY:
          execute_matrix_row_assign_matrix(s, root_node);
          break;
        default:
          throw statement_not_supported_exception("Invalid rvalue encountered in row-major matrix assignment");
      }
    }


  }

} //namespace viennacl

#endif

