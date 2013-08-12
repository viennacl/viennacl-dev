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
        if (el.type_family == SCALAR_TYPE_FAMILY && el.subtype == HOST_SCALAR_TYPE && el.numeric_type == FLOAT_TYPE)
          return el.host_float;
        if (el.type_family == SCALAR_TYPE_FAMILY && el.subtype == DEVICE_SCALAR_TYPE && el.numeric_type == FLOAT_TYPE)
          return *el.scalar_float;

        throw statement_not_supported_exception("Cannot convert to float");
      }

      // helper routines for extracting the scalar type
      inline double convert_to_double(double d) { return d; }
      inline double convert_to_double(lhs_rhs_element const & el)
      {
        if (el.type_family == SCALAR_TYPE_FAMILY && el.subtype == HOST_SCALAR_TYPE && el.numeric_type == DOUBLE_TYPE)
          return el.host_double;
        if (el.type_family == SCALAR_TYPE_FAMILY && el.subtype == DEVICE_SCALAR_TYPE && el.numeric_type == DOUBLE_TYPE)
          return *el.scalar_double;

        throw statement_not_supported_exception("Cannot convert to double");
      }

      /////////////////// Create/Destory temporary vector ///////////////////////

      inline void new_element(lhs_rhs_element & elem, lhs_rhs_element const & lhs_element)
      {
        if (elem.type_family == SCALAR_TYPE_FAMILY)
        {
          assert(elem.subtype == DEVICE_SCALAR_TYPE && bool("Expected a device scalar in root node"));

          switch (elem.numeric_type)
          {
            case FLOAT_TYPE:
              elem.scalar_float = new viennacl::scalar<float>();
              return;
            case DOUBLE_TYPE:
              elem.scalar_double = new viennacl::scalar<double>();
              return;
            default:
              throw statement_not_supported_exception("Invalid vector type for vector construction");
          }
        }
        else if (elem.type_family == VECTOR_TYPE_FAMILY)
        {
          assert(elem.subtype == DENSE_VECTOR_TYPE && bool("Expected a dense vector in root node"));

          switch (elem.numeric_type)
          {
            case FLOAT_TYPE:
              elem.vector_float = new viennacl::vector<float>((lhs_element.vector_float)->size());
              return;
            case DOUBLE_TYPE:
              elem.vector_double = new viennacl::vector<double>((lhs_element.vector_float)->size());
              return;
            default:
              throw statement_not_supported_exception("Invalid vector type for vector construction");
          }
        }
        else if (elem.type_family == MATRIX_TYPE_FAMILY)
        {
          assert( (elem.subtype == DENSE_COL_MATRIX_TYPE || elem.subtype == DENSE_ROW_MATRIX_TYPE)
                 && bool("Expected a dense matrix in root node"));

          if (elem.subtype == DENSE_COL_MATRIX_TYPE)
          {
            switch (elem.numeric_type)
            {
              case FLOAT_TYPE:
                elem.matrix_col_float = new viennacl::matrix<float, viennacl::column_major>((lhs_element.matrix_col_float)->size1(), (lhs_element.matrix_col_float)->size2());
                return;
              case DOUBLE_TYPE:
                elem.matrix_col_double = new viennacl::matrix<double, viennacl::column_major>((lhs_element.matrix_col_double)->size1(), (lhs_element.matrix_col_double)->size2());
                return;
              default:
                throw statement_not_supported_exception("Invalid vector type for vector construction");
            }
          }
          else if (elem.subtype == DENSE_ROW_MATRIX_TYPE)
          {
            switch (elem.numeric_type)
            {
              case FLOAT_TYPE:
                elem.matrix_row_float = new viennacl::matrix<float, viennacl::row_major>((lhs_element.matrix_row_float)->size1(), (lhs_element.matrix_row_float)->size2());
                return;
              case DOUBLE_TYPE:
                elem.matrix_row_double = new viennacl::matrix<double, viennacl::row_major>((lhs_element.matrix_row_double)->size1(), (lhs_element.matrix_row_double)->size2());
                return;
              default:
                throw statement_not_supported_exception("Invalid vector type for vector construction");
            }
          }
          else
            throw statement_not_supported_exception("Expected a dense matrix in root node when creating a temporary");
        }
        else
          throw statement_not_supported_exception("Unknown type familty when creating new temporary object");
      }

      inline void delete_element(lhs_rhs_element & elem)
      {
        if (elem.type_family == SCALAR_TYPE_FAMILY)
        {
          switch (elem.numeric_type)
          {
            case FLOAT_TYPE:
              delete elem.scalar_float;
              return;
            case DOUBLE_TYPE:
              delete elem.scalar_double;
              return;
            default:
              throw statement_not_supported_exception("Invalid vector type for vector destruction");
          }
        }
        else if (elem.type_family == VECTOR_TYPE_FAMILY)
        {
          switch (elem.numeric_type)
          {
            case FLOAT_TYPE:
              delete elem.vector_float;
              return;
            case DOUBLE_TYPE:
              delete elem.vector_double;
              return;
            default:
              throw statement_not_supported_exception("Invalid vector type for vector destruction");
          }
        }
        else if (elem.type_family == MATRIX_TYPE_FAMILY)
        {
          if (elem.subtype == DENSE_COL_MATRIX_TYPE)
          {
            switch (elem.numeric_type)
            {
              case FLOAT_TYPE:
                delete elem.matrix_col_float;
                return;
              case DOUBLE_TYPE:
                delete elem.matrix_col_double;
                return;
              default:
                throw statement_not_supported_exception("Invalid vector type for vector destruction");
            }
          }
          else if (elem.subtype == DENSE_ROW_MATRIX_TYPE)
          {
            switch (elem.numeric_type)
            {
              case FLOAT_TYPE:
                delete elem.matrix_row_float;
                return;
              case DOUBLE_TYPE:
                delete elem.matrix_row_double;
                return;
              default:
                throw statement_not_supported_exception("Invalid vector type for vector destruction");
            }
          }
          else
            throw statement_not_supported_exception("Expected a dense matrix in root node when deleting temporary");
        }
        else
          throw statement_not_supported_exception("Unknown type familty when deleting temporary object");
      }

    } // namespace detail


  } // namespace scheduler
} // namespace viennacl

#endif

