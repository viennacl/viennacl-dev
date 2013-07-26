#ifndef VIENNACL_SCHEDULER_IO_HPP
#define VIENNACL_SCHEDULER_IO_HPP

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


/** @file viennacl/scheduler/io.hpp
    @brief Some helper routines for reading/writing/printing scheduler expressions
*/

#include <iostream>
#include <sstream>

#include "viennacl/forwards.h"
#include "viennacl/scheduler/forwards.h"


namespace viennacl
{
  namespace scheduler
  {

    namespace detail
    {
#define VIENNACL_TRANSLATE_OP_TO_STRING(NAME)   case NAME: return #NAME;

      /** @brief Helper routine for converting the operation enums to string */
      std::string to_string(viennacl::scheduler::operation_node_type_family family,
                            viennacl::scheduler::operation_node_type type)
      {
        if (family == OPERATION_UNARY_TYPE_FAMILY)
        {
          switch (type)
          {
            VIENNACL_TRANSLATE_OP_TO_STRING(OPERATION_UNARY_ABS_TYPE)
            VIENNACL_TRANSLATE_OP_TO_STRING(OPERATION_UNARY_ACOS_TYPE)
            VIENNACL_TRANSLATE_OP_TO_STRING(OPERATION_UNARY_ASIN_TYPE)
            VIENNACL_TRANSLATE_OP_TO_STRING(OPERATION_UNARY_ATAN_TYPE)
            VIENNACL_TRANSLATE_OP_TO_STRING(OPERATION_UNARY_CEIL_TYPE)
            VIENNACL_TRANSLATE_OP_TO_STRING(OPERATION_UNARY_COS_TYPE)
            VIENNACL_TRANSLATE_OP_TO_STRING(OPERATION_UNARY_COSH_TYPE)
            VIENNACL_TRANSLATE_OP_TO_STRING(OPERATION_UNARY_EXP_TYPE)
            VIENNACL_TRANSLATE_OP_TO_STRING(OPERATION_UNARY_FABS_TYPE)
            VIENNACL_TRANSLATE_OP_TO_STRING(OPERATION_UNARY_FLOOR_TYPE)
            VIENNACL_TRANSLATE_OP_TO_STRING(OPERATION_UNARY_LOG_TYPE)
            VIENNACL_TRANSLATE_OP_TO_STRING(OPERATION_UNARY_LOG10_TYPE)
            VIENNACL_TRANSLATE_OP_TO_STRING(OPERATION_UNARY_SIN_TYPE)
            VIENNACL_TRANSLATE_OP_TO_STRING(OPERATION_UNARY_SINH_TYPE)
            VIENNACL_TRANSLATE_OP_TO_STRING(OPERATION_UNARY_SQRT_TYPE)
            VIENNACL_TRANSLATE_OP_TO_STRING(OPERATION_UNARY_TAN_TYPE)
            VIENNACL_TRANSLATE_OP_TO_STRING(OPERATION_UNARY_TANH_TYPE)
            VIENNACL_TRANSLATE_OP_TO_STRING(OPERATION_UNARY_NORM_1_TYPE)
            VIENNACL_TRANSLATE_OP_TO_STRING(OPERATION_UNARY_NORM_2_TYPE)
            VIENNACL_TRANSLATE_OP_TO_STRING(OPERATION_UNARY_NORM_INF_TYPE)

            default: throw statement_not_supported_exception("Cannot convert unary operation to string");
          }
        }
        else if (family == OPERATION_BINARY_TYPE_FAMILY)
        {
          switch (type)
          {
            VIENNACL_TRANSLATE_OP_TO_STRING(OPERATION_BINARY_ASSIGN_TYPE)
            VIENNACL_TRANSLATE_OP_TO_STRING(OPERATION_BINARY_INPLACE_ADD_TYPE)
            VIENNACL_TRANSLATE_OP_TO_STRING(OPERATION_BINARY_INPLACE_SUB_TYPE)
            VIENNACL_TRANSLATE_OP_TO_STRING(OPERATION_BINARY_ADD_TYPE)
            VIENNACL_TRANSLATE_OP_TO_STRING(OPERATION_BINARY_SUB_TYPE)
            VIENNACL_TRANSLATE_OP_TO_STRING(OPERATION_BINARY_MAT_VEC_PROD_TYPE)
            VIENNACL_TRANSLATE_OP_TO_STRING(OPERATION_BINARY_MAT_MAT_PROD_TYPE)
            VIENNACL_TRANSLATE_OP_TO_STRING(OPERATION_BINARY_MULT_TYPE)
            VIENNACL_TRANSLATE_OP_TO_STRING(OPERATION_BINARY_ELEMENT_MULT_TYPE)
            VIENNACL_TRANSLATE_OP_TO_STRING(OPERATION_BINARY_ELEMENT_DIV_TYPE)
            VIENNACL_TRANSLATE_OP_TO_STRING(OPERATION_BINARY_INNER_PROD_TYPE)

            default: throw statement_not_supported_exception("Cannot convert unary operation to string");
          }
        }
        else
          throw statement_not_supported_exception("Unknown operation family when converting to string");
      }

#undef VIENNACL_TRANSLATE_OP_TO_STRING

#define VIENNACL_TRANSLATE_ELEMENT_TO_STRING(NAME, ELEMENT)   case NAME: ss << "(" << element.ELEMENT << ")"; return #NAME + ss.str();

      /** @brief Helper routine converting the enum and union values inside a statement node to a string */
      std::string to_string(viennacl::scheduler::statement_node_type_family family,
                            viennacl::scheduler::statement_node_type        type,
                            viennacl::scheduler::lhs_rhs_element            element)
      {
        std::stringstream ss;

        if (family == COMPOSITE_OPERATION_FAMILY)
        {
          switch (type)
          {
            VIENNACL_TRANSLATE_ELEMENT_TO_STRING(COMPOSITE_OPERATION_TYPE, node_index)

            default: throw statement_not_supported_exception("Cannot convert composite operation type to string");
          }
        }
        else if (family == HOST_SCALAR_TYPE_FAMILY)
        {
          switch (type)
          {
            VIENNACL_TRANSLATE_ELEMENT_TO_STRING(HOST_SCALAR_CHAR_TYPE,   host_char)
            VIENNACL_TRANSLATE_ELEMENT_TO_STRING(HOST_SCALAR_UCHAR_TYPE,  host_uchar)
            VIENNACL_TRANSLATE_ELEMENT_TO_STRING(HOST_SCALAR_SHORT_TYPE,  host_short)
            VIENNACL_TRANSLATE_ELEMENT_TO_STRING(HOST_SCALAR_USHORT_TYPE, host_ushort)
            VIENNACL_TRANSLATE_ELEMENT_TO_STRING(HOST_SCALAR_INT_TYPE,    host_int)
            VIENNACL_TRANSLATE_ELEMENT_TO_STRING(HOST_SCALAR_UINT_TYPE,   host_uint)
            VIENNACL_TRANSLATE_ELEMENT_TO_STRING(HOST_SCALAR_LONG_TYPE,   host_long)
            VIENNACL_TRANSLATE_ELEMENT_TO_STRING(HOST_SCALAR_ULONG_TYPE,  host_ulong)
            VIENNACL_TRANSLATE_ELEMENT_TO_STRING(HOST_SCALAR_FLOAT_TYPE,  host_float)
            VIENNACL_TRANSLATE_ELEMENT_TO_STRING(HOST_SCALAR_DOUBLE_TYPE, host_double)

            default: throw statement_not_supported_exception("Cannot convert host scalar type to string");
          }
        }
        else if (family == SCALAR_TYPE_FAMILY)
        {
          switch (type)
          {
            //VIENNACL_TRANSLATE_ELEMENT_TO_STRING(SCALAR_CHAR_TYPE,   scalar_char)
            //VIENNACL_TRANSLATE_ELEMENT_TO_STRING(SCALAR_UCHAR_TYPE,  scalar_uchar)
            //VIENNACL_TRANSLATE_ELEMENT_TO_STRING(SCALAR_SHORT_TYPE,  scalar_short)
            //VIENNACL_TRANSLATE_ELEMENT_TO_STRING(SCALAR_USHORT_TYPE, scalar_ushort)
            //VIENNACL_TRANSLATE_ELEMENT_TO_STRING(SCALAR_INT_TYPE,    scalar_int)
            //VIENNACL_TRANSLATE_ELEMENT_TO_STRING(SCALAR_UINT_TYPE,   scalar_uint)
            //VIENNACL_TRANSLATE_ELEMENT_TO_STRING(SCALAR_LONG_TYPE,   scalar_long)
            //VIENNACL_TRANSLATE_ELEMENT_TO_STRING(SCALAR_ULONG_TYPE,  scalar_ulong)
            //VIENNACL_TRANSLATE_ELEMENT_TO_STRING(SCALAR_HALF_TYPE,   scalar_half)
            VIENNACL_TRANSLATE_ELEMENT_TO_STRING(SCALAR_FLOAT_TYPE,  scalar_float)
            VIENNACL_TRANSLATE_ELEMENT_TO_STRING(SCALAR_DOUBLE_TYPE, scalar_double)

            default: throw statement_not_supported_exception("Cannot convert scalar type to string");
          }
        }
        else if (family == VECTOR_TYPE_FAMILY)
        {
          switch (type)
          {
            //VIENNACL_TRANSLATE_ELEMENT_TO_STRING(VECTOR_CHAR_TYPE,   vector_char)
            //VIENNACL_TRANSLATE_ELEMENT_TO_STRING(VECTOR_UCHAR_TYPE,  vector_uchar)
            //VIENNACL_TRANSLATE_ELEMENT_TO_STRING(VECTOR_SHORT_TYPE,  vector_short)
            //VIENNACL_TRANSLATE_ELEMENT_TO_STRING(VECTOR_USHORT_TYPE, vector_ushort)
            //VIENNACL_TRANSLATE_ELEMENT_TO_STRING(VECTOR_INT_TYPE,    vector_int)
            //VIENNACL_TRANSLATE_ELEMENT_TO_STRING(VECTOR_UINT_TYPE,   vector_uint)
            //VIENNACL_TRANSLATE_ELEMENT_TO_STRING(VECTOR_LONG_TYPE,   vector_long)
            //VIENNACL_TRANSLATE_ELEMENT_TO_STRING(VECTOR_ULONG_TYPE,  vector_ulong)
            //VIENNACL_TRANSLATE_ELEMENT_TO_STRING(VECTOR_HALF_TYPE,   vector_half)
            VIENNACL_TRANSLATE_ELEMENT_TO_STRING(VECTOR_FLOAT_TYPE,  vector_float)
            VIENNACL_TRANSLATE_ELEMENT_TO_STRING(VECTOR_DOUBLE_TYPE, vector_double)

            default: throw statement_not_supported_exception("Cannot convert vector type to string");
          }
        }
        else if (family == MATRIX_ROW_TYPE_FAMILY)
        {
          switch (type)
          {
            //VIENNACL_TRANSLATE_ELEMENT_TO_STRING(MATRIX_ROW_CHAR_TYPE,   matrix_row_char)
            //VIENNACL_TRANSLATE_ELEMENT_TO_STRING(MATRIX_ROW_UCHAR_TYPE,  matrix_row_uchar)
            //VIENNACL_TRANSLATE_ELEMENT_TO_STRING(MATRIX_ROW_SHORT_TYPE,  matrix_row_short)
            //VIENNACL_TRANSLATE_ELEMENT_TO_STRING(MATRIX_ROW_USHORT_TYPE, matrix_row_ushort)
            //VIENNACL_TRANSLATE_ELEMENT_TO_STRING(MATRIX_ROW_INT_TYPE,    matrix_row_int)
            //VIENNACL_TRANSLATE_ELEMENT_TO_STRING(MATRIX_ROW_UINT_TYPE,   matrix_row_uint)
            //VIENNACL_TRANSLATE_ELEMENT_TO_STRING(MATRIX_ROW_LONG_TYPE,   matrix_row_long)
            //VIENNACL_TRANSLATE_ELEMENT_TO_STRING(MATRIX_ROW_ULONG_TYPE,  matrix_row_ulong)
            //VIENNACL_TRANSLATE_ELEMENT_TO_STRING(MATRIX_ROW_HALF_TYPE,   matrix_row_half)
            VIENNACL_TRANSLATE_ELEMENT_TO_STRING(MATRIX_ROW_FLOAT_TYPE,  matrix_row_float)
            VIENNACL_TRANSLATE_ELEMENT_TO_STRING(MATRIX_ROW_DOUBLE_TYPE, matrix_row_double)

            default: throw statement_not_supported_exception("Cannot convert row-major matrix type to string");
          }
        }
        else if (family == MATRIX_COL_TYPE_FAMILY)
        {
          switch (type)
          {
            //VIENNACL_TRANSLATE_ELEMENT_TO_STRING(MATRIX_COL_CHAR_TYPE,   matrix_col_char)
            //VIENNACL_TRANSLATE_ELEMENT_TO_STRING(MATRIX_COL_UCHAR_TYPE,  matrix_col_uchar)
            //VIENNACL_TRANSLATE_ELEMENT_TO_STRING(MATRIX_COL_SHORT_TYPE,  matrix_col_short)
            //VIENNACL_TRANSLATE_ELEMENT_TO_STRING(MATRIX_COL_USHORT_TYPE, matrix_col_ushort)
            //VIENNACL_TRANSLATE_ELEMENT_TO_STRING(MATRIX_COL_INT_TYPE,    matrix_col_int)
            //VIENNACL_TRANSLATE_ELEMENT_TO_STRING(MATRIX_COL_UINT_TYPE,   matrix_col_uint)
            //VIENNACL_TRANSLATE_ELEMENT_TO_STRING(MATRIX_COL_LONG_TYPE,   matrix_col_long)
            //VIENNACL_TRANSLATE_ELEMENT_TO_STRING(MATRIX_COL_ULONG_TYPE,  matrix_col_ulong)
            //VIENNACL_TRANSLATE_ELEMENT_TO_STRING(MATRIX_COL_HALF_TYPE,   matrix_col_half)
            VIENNACL_TRANSLATE_ELEMENT_TO_STRING(MATRIX_COL_FLOAT_TYPE,  matrix_col_float)
            VIENNACL_TRANSLATE_ELEMENT_TO_STRING(MATRIX_COL_DOUBLE_TYPE, matrix_col_double)

            default: throw statement_not_supported_exception("Cannot convert column-major matrix type to string");
          }
        }
        else
          throw statement_not_supported_exception("Unknown operation family when converting to string");
      }

#undef VIENNACL_TRANSLATE_ELEMENT_TO_STRING

    } // namespace detail


    /** @brief Print a single statement_node. Non-recursive */
    std::ostream & operator<<(std::ostream & os, viennacl::scheduler::statement_node const & s_node)
    {
      os << "LHS: " << detail::to_string(s_node.lhs_type_family, s_node.lhs_type, s_node.lhs) << ", "
         << "OP: "  << detail::to_string(s_node.op_family,       s_node.op_type) << ", "
         << "RHS: " << detail::to_string(s_node.rhs_type_family, s_node.rhs_type, s_node.rhs);

      return os;
    }





    namespace detail
    {
      /** @brief Recursive worker routine for printing a whole statement */
      void print_node(std::ostream & os, viennacl::scheduler::statement const & s, std::size_t node_index, std::size_t indent = 0)
      {
        typedef viennacl::scheduler::statement::container_type   StatementNodeContainer;
        typedef viennacl::scheduler::statement::value_type       StatementNode;

        StatementNodeContainer const & nodes = s.array();
        StatementNode const & current_node = nodes[node_index];

        for (std::size_t i=0; i<indent; ++i)
          os << " ";

        os << "Node " << node_index << ": " << current_node << std::endl;

        if (current_node.lhs_type_family == COMPOSITE_OPERATION_FAMILY)
          print_node(os, s, current_node.lhs.node_index, indent+1);

        if (current_node.rhs_type_family == COMPOSITE_OPERATION_FAMILY)
          print_node(os, s, current_node.rhs.node_index, indent+1);
      }
    }

    /** @brief Writes a string identifying the scheduler statement to an output stream.
      *
      * Typically used for debugging
      * @param os    The output stream
      * @param s     The statement object
      */
    std::ostream & operator<<(std::ostream & os, viennacl::scheduler::statement const & s)
    {
      detail::print_node(os, s, s.root());
      return os;
    }
  }

} //namespace viennacl

#endif

