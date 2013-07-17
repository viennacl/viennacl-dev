#ifndef VIENNACL_SCHEDULER_EXECUTE_SCALAR_ASSIGN_HPP
#define VIENNACL_SCHEDULER_EXECUTE_SCALAR_ASSIGN_HPP

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
#include "viennacl/linalg/vector_operations.hpp"

namespace viennacl
{
  namespace scheduler
  {
    /** @brief Deals with x = RHS where RHS is a vector expression */
    void execute_scalar_assign_composite(statement const & s)
    {
      typename statement::container_type const & expr = s.array();

      if (expr[1].op_type_  == OPERATION_BINARY_INNER_PROD_TYPE)
      {
        if (expr[0].lhs_type_ == SCALAR_FLOAT_TYPE
            && expr[1].lhs_type_ == VECTOR_FLOAT_TYPE
            && expr[1].rhs_type_ == VECTOR_FLOAT_TYPE)
        {
          viennacl::scalar<float>            & s = *(expr[0].lhs_.scalar_float_);
          viennacl::vector_base<float> const & y = *(expr[1].lhs_.vector_float_);
          viennacl::vector_base<float> const & z = *(expr[1].rhs_.vector_float_);
          viennacl::linalg::inner_prod_impl(y, z, s);
        }
        else if (expr[0].lhs_type_ == SCALAR_DOUBLE_TYPE
                 && expr[1].lhs_type_ == VECTOR_DOUBLE_TYPE
                 && expr[1].rhs_type_ == VECTOR_DOUBLE_TYPE)
        {
          viennacl::scalar<double>            & s = *(expr[0].lhs_.scalar_double_);
          viennacl::vector_base<double> const & y = *(expr[1].lhs_.vector_double_);
          viennacl::vector_base<double> const & z = *(expr[1].rhs_.vector_double_);
          viennacl::linalg::inner_prod_impl(y, z, s);
        }
        else
          throw "TODO";
      }
      else if (expr[1].op_type_  == OPERATION_UNARY_NORM_1_TYPE)
      {
        if (expr[0].lhs_type_ == SCALAR_FLOAT_TYPE
            && expr[1].lhs_type_ == VECTOR_FLOAT_TYPE)
        {
          viennacl::scalar<float>            & s = *(expr[0].lhs_.scalar_float_);
          viennacl::vector_base<float> const & x = *(expr[1].lhs_.vector_float_);
          viennacl::linalg::norm_1_impl(x, s);
        }
        else if (expr[0].lhs_type_ == SCALAR_DOUBLE_TYPE
                 && expr[1].lhs_type_ == VECTOR_DOUBLE_TYPE
                 && expr[1].rhs_type_ == VECTOR_DOUBLE_TYPE)
        {
          viennacl::scalar<double>            & s = *(expr[0].lhs_.scalar_double_);
          viennacl::vector_base<double> const & x = *(expr[1].lhs_.vector_double_);
          viennacl::linalg::norm_1_impl(x, s);
        }
        else
          throw "TODO";
      }
      else if (expr[1].op_type_  == OPERATION_UNARY_NORM_2_TYPE)
      {
        if (expr[0].lhs_type_ == SCALAR_FLOAT_TYPE
            && expr[1].lhs_type_ == VECTOR_FLOAT_TYPE)
        {
          viennacl::scalar<float>            & s = *(expr[0].lhs_.scalar_float_);
          viennacl::vector_base<float> const & x = *(expr[1].lhs_.vector_float_);
          viennacl::linalg::norm_2_impl(x, s);
        }
        else if (expr[0].lhs_type_ == SCALAR_DOUBLE_TYPE
                 && expr[1].lhs_type_ == VECTOR_DOUBLE_TYPE
                 && expr[1].rhs_type_ == VECTOR_DOUBLE_TYPE)
        {
          viennacl::scalar<double>            & s = *(expr[0].lhs_.scalar_double_);
          viennacl::vector_base<double> const & x = *(expr[1].lhs_.vector_double_);
          viennacl::linalg::norm_2_impl(x, s);
        }
        else
          throw "TODO";
      }
      else if (expr[1].op_type_  == OPERATION_UNARY_NORM_INF_TYPE)
      {
        if (expr[0].lhs_type_ == SCALAR_FLOAT_TYPE
            && expr[1].lhs_type_ == VECTOR_FLOAT_TYPE)
        {
          viennacl::scalar<float>            & s = *(expr[0].lhs_.scalar_float_);
          viennacl::vector_base<float> const & x = *(expr[1].lhs_.vector_float_);
          viennacl::linalg::norm_inf_impl(x, s);
        }
        else if (expr[0].lhs_type_ == SCALAR_DOUBLE_TYPE
                 && expr[1].lhs_type_ == VECTOR_DOUBLE_TYPE
                 && expr[1].rhs_type_ == VECTOR_DOUBLE_TYPE)
        {
          viennacl::scalar<double>            & s = *(expr[0].lhs_.scalar_double_);
          viennacl::vector_base<double> const & x = *(expr[1].lhs_.vector_double_);
          viennacl::linalg::norm_inf_impl(x, s);
        }
        else
          throw "TODO";
      }
      else
        throw "TODO";
    }

    /** @brief Generic dispatcher */
    void execute_scalar_assign(statement const & s)
    {
      typedef typename statement::container_type   StatementContainer;

      StatementContainer const & expr = s.array();

      switch (expr[0].rhs_type_family_)
      {
        case COMPOSITE_OPERATION_FAMILY:
          execute_scalar_assign_composite(s);
          break;
        default:
          throw "invalid rvalue in vector assignment";
      }
    }


  }

} //namespace viennacl

#endif

