#ifndef VIENNACL_SCHEDULER_EXECUTE_VECTOR_INPLACE_ADD_HPP
#define VIENNACL_SCHEDULER_EXECUTE_VECTOR_INPLACE_ADD_HPP

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
    void execute_vector_inplace_add_composite(statement const & s)
    {
      throw "TODO";
    }

    /** @brief Deals with x = y  for a vector y */
    void execute_vector_inplace_add_vector(statement const & s)
    {
      typedef typename statement::container_type   StatementContainer;

      StatementContainer const & expr = s.array();

      if (expr[0].lhs_type_ == VECTOR_FLOAT_TYPE && expr[0].rhs_type_ == VECTOR_FLOAT_TYPE)
      {
        viennacl::vector_base<float>       & x = *(expr[0].lhs_.vector_float_);
        viennacl::vector_base<float> const & y = *(expr[0].rhs_.vector_float_);
        viennacl::linalg::avbv(x,
                               x,  1.0, 1, false, false,
                               y,  1.0, 1, false, false);
      }
      else if (expr[0].lhs_type_ == VECTOR_DOUBLE_TYPE && expr[0].rhs_type_ == VECTOR_DOUBLE_TYPE)
      {
        viennacl::vector_base<double>       & x = *(expr[0].lhs_.vector_double_);
        viennacl::vector_base<double> const & y = *(expr[0].rhs_.vector_double_);
        viennacl::linalg::avbv(x,
                               x,  1.0, 1, false, false,
                               y,  1.0, 1, false, false);
      }
      else
        throw "not yet supported!";
    }

    /** @brief Generic dispatcher */
    void execute_vector_inplace_add(statement const & s)
    {
      typedef typename statement::container_type   StatementContainer;

      StatementContainer const & expr = s.array();

      switch (expr[0].rhs_type_family_)
      {
        case COMPOSITE_OPERATION_FAMILY:
          execute_vector_inplace_add_composite(s);
          break;
        case VECTOR_TYPE_FAMILY:
          execute_vector_inplace_add_vector(s);
          break;
        default:
          throw "invalid rvalue in vector assignment";
      }
    }

  }

} //namespace viennacl

#endif

