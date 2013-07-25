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
#include "viennacl/linalg/vector_operations.hpp"

#include "viennacl/scheduler/execute_scalar_assign.hpp"

#include "viennacl/scheduler/execute_vector_assign.hpp"
#include "viennacl/scheduler/execute_vector_inplace_add.hpp"
#include "viennacl/scheduler/execute_vector_inplace_sub.hpp"

#include "viennacl/scheduler/execute_matrix_col_assign.hpp"
#include "viennacl/scheduler/execute_matrix_col_inplace_add.hpp"
#include "viennacl/scheduler/execute_matrix_col_inplace_sub.hpp"

#include "viennacl/scheduler/execute_matrix_row_assign.hpp"
#include "viennacl/scheduler/execute_matrix_row_inplace_add.hpp"
#include "viennacl/scheduler/execute_matrix_row_inplace_sub.hpp"

namespace viennacl
{
  namespace scheduler
  {

    inline void execute(statement const & s)
    {
      typedef statement::container_type   StatementContainer;

      StatementContainer const & expr = s.array();

      switch (expr[0].lhs_type_family)
      {
        case SCALAR_TYPE_FAMILY:
          switch (expr[0].op_type)
          {
            case OPERATION_BINARY_ASSIGN_TYPE:
              execute_scalar_assign(s); break;
            default:
              throw statement_not_supported_exception("Scalar operation does not use '=' in head node.");
          }
          break;

        case VECTOR_TYPE_FAMILY:
          switch (expr[0].op_type)
          {
            case OPERATION_BINARY_ASSIGN_TYPE:
              execute_vector_assign(s); break;
            case OPERATION_BINARY_INPLACE_ADD_TYPE:
              execute_vector_inplace_add(s); break;
            case OPERATION_BINARY_INPLACE_SUB_TYPE:
              execute_vector_inplace_sub(s); break;
            default:
              throw statement_not_supported_exception("Vector operation does not use '=', '+=' or '-=' in head node.");
          }
          break;

        case MATRIX_COL_TYPE_FAMILY:
          switch (expr[0].op_type)
          {
            case OPERATION_BINARY_ASSIGN_TYPE:
              execute_matrix_col_assign(s); break;
            case OPERATION_BINARY_INPLACE_ADD_TYPE:
              execute_matrix_col_inplace_add(s); break;
            case OPERATION_BINARY_INPLACE_SUB_TYPE:
              execute_matrix_col_inplace_sub(s); break;
            default:
              throw statement_not_supported_exception("Column-major matrix operation does not use '=', '+=' or '-=' in head node.");
          }
          break;

        case MATRIX_ROW_TYPE_FAMILY:
          switch (expr[0].op_type)
          {
            case OPERATION_BINARY_ASSIGN_TYPE:
              execute_matrix_row_assign(s); break;
            case OPERATION_BINARY_INPLACE_ADD_TYPE:
              execute_matrix_row_inplace_add(s); break;
            case OPERATION_BINARY_INPLACE_SUB_TYPE:
              execute_matrix_row_inplace_sub(s); break;
            default:
              throw statement_not_supported_exception("Row-major matrix operation does not use '=', '+=' or '-=' in head node.");
          }
          break;

        default:
          throw statement_not_supported_exception("Unsupported lvalue encountered in head node.");
      }
    }


  }

} //namespace viennacl

#endif

