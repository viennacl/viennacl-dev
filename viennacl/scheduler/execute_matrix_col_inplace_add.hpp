#ifndef VIENNACL_SCHEDULER_EXECUTE_MATRIX_COL_INPLACE_ADD_HPP
#define VIENNACL_SCHEDULER_EXECUTE_MATRIX_COL_INPLACE_ADD_HPP

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


/** @file viennacl/scheduler/execute_matrix_col_inplace_add.hpp
    @brief Dealing with inplace-add statements for column-major matrices such as 'A += B + C;'
*/

#include "viennacl/forwards.h"
#include "viennacl/scheduler/forwards.h"
#include "viennacl/linalg/matrix_operations.hpp"

namespace viennacl
{
  namespace scheduler
  {

    /** @brief Deals with x = RHS where RHS is a vector expression */
    inline void execute_matrix_col_inplace_add_composite(statement const & s, statement_node const & root_node)
    {
      throw statement_not_supported_exception("Composite inplace-additions for column-major matrices not supported yet");
    }

    /** @brief Deals with A += B  for a matrix B */
    inline void execute_matrix_col_inplace_add_matrix(statement const & s, statement_node const & root_node)
    {
      typedef statement::container_type   StatementContainer;

      StatementContainer const & expr = s.array();

      if (expr[0].lhs_type == MATRIX_COL_FLOAT_TYPE && expr[0].rhs_type == MATRIX_COL_FLOAT_TYPE)
      {
        viennacl::matrix_base<float, viennacl::column_major>       & A = *(expr[0].lhs.matrix_col_float);
        viennacl::matrix_base<float, viennacl::column_major> const & B = *(expr[0].rhs.matrix_col_float);
        viennacl::linalg::ambm(A,
                               A,  1.0, 1, false, false,
                               B,  1.0, 1, false, false);
      }
      else if (expr[0].lhs_type == MATRIX_COL_DOUBLE_TYPE && expr[0].rhs_type == MATRIX_COL_DOUBLE_TYPE)
      {
        viennacl::matrix_base<double, viennacl::column_major>       & A = *(expr[0].lhs.matrix_col_double);
        viennacl::matrix_base<double, viennacl::column_major> const & B = *(expr[0].rhs.matrix_col_double);
        viennacl::linalg::ambm(A,
                               A,  1.0, 1, false, false,
                               B,  1.0, 1, false, false);
      }
      else
        throw statement_not_supported_exception("Unsupported rvalue for inplace-add to column-major matrix");
    }

    /** @brief Generic dispatcher */
    inline void execute_matrix_col_inplace_add(statement const & s, statement_node const & root_node)
    {
      typedef statement::container_type   StatementContainer;

      StatementContainer const & expr = s.array();

      switch (expr[0].rhs_type_family)
      {
        case COMPOSITE_OPERATION_FAMILY:
          execute_matrix_col_inplace_add_composite(s, root_node);
          break;
        case MATRIX_COL_TYPE_FAMILY:
          execute_matrix_col_inplace_add_matrix(s, root_node);
          break;
        default:
          throw statement_not_supported_exception("Invalid rvalue encountered in column-major matrix inplace-add");
      }
    }

  }

} //namespace viennacl

#endif

