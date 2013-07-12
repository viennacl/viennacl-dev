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
    void execute_matrix_col_inplace_add_composite(statement const & s)
    {
      throw "TODO";
    }

    /** @brief Deals with A += B  for a matrix B */
    void execute_matrix_col_inplace_add_matrix(statement const & s)
    {
      typedef typename statement::container_type   StatementContainer;

      StatementContainer const & expr = s.array();

      if (expr[0].lhs_type_ == MATRIX_COL_FLOAT_TYPE && expr[0].rhs_type_ == MATRIX_COL_FLOAT_TYPE)
      {
        viennacl::matrix_base<float, viennacl::column_major>       & A = *(expr[0].lhs_.matrix_col_float_);
        viennacl::matrix_base<float, viennacl::column_major> const & B = *(expr[0].rhs_.matrix_col_float_);
        viennacl::linalg::ambm(A,
                               A,  1.0, 1, false, false,
                               B,  1.0, 1, false, false);
      }
      else if (expr[0].lhs_type_ == MATRIX_COL_DOUBLE_TYPE && expr[0].rhs_type_ == MATRIX_COL_DOUBLE_TYPE)
      {
        viennacl::matrix_base<double, viennacl::column_major>       & A = *(expr[0].lhs_.matrix_col_double_);
        viennacl::matrix_base<double, viennacl::column_major> const & B = *(expr[0].rhs_.matrix_col_double_);
        viennacl::linalg::ambm(A,
                               A,  1.0, 1, false, false,
                               B,  1.0, 1, false, false);
      }
      else
        throw "not yet supported!";
    }

    /** @brief Generic dispatcher */
    void execute_matrix_col_inplace_add(statement const & s)
    {
      typedef typename statement::container_type   StatementContainer;

      StatementContainer const & expr = s.array();

      switch (expr[0].rhs_type_family_)
      {
        case COMPOSITE_OPERATION_FAMILY:
          execute_matrix_col_inplace_add_composite(s);
          break;
        case MATRIX_COL_TYPE_FAMILY:
          execute_matrix_col_inplace_add_matrix(s);
          break;
        default:
          throw "invalid rvalue in vector assignment";
      }
    }

  }

} //namespace viennacl

#endif

