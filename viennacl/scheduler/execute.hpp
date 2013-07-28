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
#include "viennacl/scheduler/execute_matrix.hpp"

namespace viennacl
{
  namespace scheduler
  {

    namespace detail
    {
      inline void execute_impl(statement const & s, statement_node const & root_node)
      {
        typedef statement::container_type   StatementContainer;

        switch (root_node.lhs.type_family)
        {
          case SCALAR_TYPE_FAMILY:
            switch (root_node.op.type)
            {
              case OPERATION_BINARY_ASSIGN_TYPE:
                execute_scalar_assign(s, root_node); break;
              default:
                throw statement_not_supported_exception("Scalar operation does not use '=' in head node.");
            }
            break;

          case VECTOR_TYPE_FAMILY:
            execute_vector(s, root_node); break;

          case MATRIX_COL_TYPE_FAMILY:
          case MATRIX_ROW_TYPE_FAMILY:
            execute_matrix(s, root_node); break;

          default:
            throw statement_not_supported_exception("Unsupported lvalue encountered in head node.");
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

