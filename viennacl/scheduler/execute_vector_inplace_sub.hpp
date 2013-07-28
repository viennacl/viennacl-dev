#ifndef VIENNACL_SCHEDULER_EXECUTE_VECTOR_INPLACE_SUB_HPP
#define VIENNACL_SCHEDULER_EXECUTE_VECTOR_INPLACE_SUB_HPP

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

namespace viennacl
{
  namespace scheduler
  {

    /** @brief Deals with x = RHS where RHS is a vector expression */
    inline void execute_vector_inplace_sub_composite(statement const & s, statement_node const & root_node)
    {
      throw statement_not_supported_exception("Composite inplace-subtractions for vectors not supported yet");
    }

    /** @brief Deals with x = y  for a vector y */
    inline void execute_vector_inplace_sub_vector(statement const & s, statement_node const & root_node)
    {
      lhs_rhs_element_2 u; u.type_family = VECTOR_TYPE_FAMILY; u.type = root_node.lhs_type; u.data = root_node.lhs;
      lhs_rhs_element_2 v; v.type_family = VECTOR_TYPE_FAMILY; v.type = root_node.rhs_type; v.data = root_node.rhs;
      detail::avbv(u,
                   u,  1.0, 1, false, false,
                   v, -1.0, 1, false, false);
    }

    /** @brief Generic dispatcher */
    inline void execute_vector_inplace_sub(statement const & s, statement_node const & root_node)
    {
      switch (root_node.rhs_type_family)
      {
        case COMPOSITE_OPERATION_FAMILY:
          execute_vector_inplace_sub_composite(s, root_node);
          break;
        case VECTOR_TYPE_FAMILY:
          execute_vector_inplace_sub_vector(s, root_node);
          break;
        default:
          throw statement_not_supported_exception("Invalid rvalue encountered in vector inplace-sub");
      }
    }

  }

} //namespace viennacl

#endif

