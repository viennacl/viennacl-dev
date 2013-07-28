#ifndef VIENNACL_SCHEDULER_EXECUTE_UTIL_HPP
#define VIENNACL_SCHEDULER_EXECUTE_UTIL_HPP

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


/** @file viennacl/scheduler/execute_util.hpp
    @brief Provides various utilities for implementing the execution of statements
*/

#include <assert.h>

#include "viennacl/forwards.h"
#include "viennacl/scheduler/forwards.h"

namespace viennacl
{
  namespace scheduler
  {
    namespace detail
    {
      // helper routines for extracting the scalar type
      inline float convert_to_float(float f) { return f; }
      inline float convert_to_float(lhs_rhs_element const & el)
      {
        if (el.type == HOST_SCALAR_FLOAT_TYPE)
          return el.host_float;
        if (el.type == SCALAR_FLOAT_TYPE)
          return *el.scalar_float;

        throw statement_not_supported_exception("Cannot convert to float");
      }

      // helper routines for extracting the scalar type
      inline double convert_to_double(double d) { return d; }
      inline double convert_to_double(lhs_rhs_element const & el)
      {
        if (el.type == HOST_SCALAR_DOUBLE_TYPE)
          return el.host_double;
        if (el.type == SCALAR_DOUBLE_TYPE)
          return *el.scalar_double;

        throw statement_not_supported_exception("Cannot convert to double");
      }

      /////////////////// Create/Destory temporary vector ///////////////////////

      inline void new_vector(lhs_rhs_element & elem, std::size_t size)
      {
        switch (elem.type)
        {
          case VECTOR_FLOAT_TYPE:
            elem.vector_float = new viennacl::vector<float>(size);
            return;
          case VECTOR_DOUBLE_TYPE:
            elem.vector_double = new viennacl::vector<double>(size);
            return;
          default:
            throw statement_not_supported_exception("Invalid vector type for vector construction");
        }
      }

      inline void delete_vector(lhs_rhs_element & elem)
      {
        switch (elem.type)
        {
          case VECTOR_FLOAT_TYPE:
            delete elem.vector_float;
            return;
          case VECTOR_DOUBLE_TYPE:
            delete elem.vector_double;
            return;
          default:
            throw statement_not_supported_exception("Invalid vector type for vector destruction");
        }
      }

      /////////////////// Create/Destory temporary matrix ///////////////////////

      inline void new_matrix(lhs_rhs_element & elem, std::size_t size1, std::size_t size2)
      {
        switch (elem.type)
        {
          case MATRIX_ROW_FLOAT_TYPE:
            elem.matrix_row_float = new viennacl::matrix<float, viennacl::row_major>(size1, size2);
            return;
          case MATRIX_COL_FLOAT_TYPE:
            elem.matrix_col_float = new viennacl::matrix<float, viennacl::column_major>(size1, size2);
            return;

          case MATRIX_ROW_DOUBLE_TYPE:
            elem.matrix_row_double = new viennacl::matrix<double, viennacl::row_major>(size1, size2);
            return;
          case MATRIX_COL_DOUBLE_TYPE:
            elem.matrix_col_double = new viennacl::matrix<double, viennacl::column_major>(size1, size2);
            return;

          default:
            throw statement_not_supported_exception("Invalid vector type for vector construction");
        }
      }

      inline void delete_matrix(lhs_rhs_element & elem)
      {
        switch (elem.type)
        {
          case MATRIX_ROW_FLOAT_TYPE:
            delete elem.matrix_row_float;
            return;
          case MATRIX_COL_FLOAT_TYPE:
            delete elem.matrix_col_float;
            return;
          case MATRIX_ROW_DOUBLE_TYPE:
            delete elem.matrix_row_double;
            return;
          case MATRIX_COL_DOUBLE_TYPE:
            delete elem.matrix_col_double;
            return;
          default:
            throw statement_not_supported_exception("Invalid vector type for vector destruction");
        }
      }

    } // namespace detail


  } // namespace scheduler
} // namespace viennacl

#endif

