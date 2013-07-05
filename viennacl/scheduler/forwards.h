#ifndef VIENNACL_SCHEDULER_STATEMENT_HPP
#define VIENNACL_SCHEDULER_STATEMENT_HPP

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


/** @file viennacl/scheduler/statement.hpp
    @brief Provides the datastructures for dealing with a single statement such as 'x = y + z;'
*/

#include "viennacl/forwards.h"

namespace viennacl
{
  namespace scheduler
  {

    /** @brief Optimization enum for grouping operations into unary or binary operations. Just for optimization of lookups. */
    enum operation_node_type_family
    {
      // unary or binary expression
      OPERATION_UNARY_TYPE_FAMILY,
      OPERATION_BINARY_TYPE_FAMILY
    };

    /** @brief Enumeration for identifying the possible operations */
    enum operation_node_type
    {
      // unary expression
      OPERATION_UNARY_ABS_TYPE,
      OPERATION_UNARY_ACOS_TYPE,
      OPERATION_UNARY_ASIN_TYPE,
      OPERATION_UNARY_ATAN_TYPE,
      OPERATION_UNARY_CEIL_TYPE,
      OPERATION_UNARY_COS_TYPE,
      OPERATION_UNARY_COSH_TYPE,
      OPERATION_UNARY_EXP_TYPE,
      OPERATION_UNARY_FABS_TYPE,
      OPERATION_UNARY_FLOOR_TYPE,
      OPERATION_UNARY_LOG_TYPE,
      OPERATION_UNARY_LOG10_TYPE,
      OPERATION_UNARY_SIN_TYPE,
      OPERATION_UNARY_SINH_TYPE,
      OPERATION_UNARY_SQRT_TYPE,
      OPERATION_UNARY_TAN_TYPE,
      OPERATION_UNARY_TANH_TYPE,

      // binary expression
      OPERATION_BINARY_ASSIGN_TYPE,
      OPERATION_BINARY_INPLACE_ADD_TYPE,
      OPERATION_BINARY_INPLACE_SUB_TYPE,
      OPERATION_BINARY_ADD_TYPE,
      OPERATION_BINARY_SUB_TYPE,
      OPERATION_BINARY_PROD_TYPE,
      OPERATION_BINARY_MULT_TYPE,    // scalar times vector/matrix
      OPERATION_BINARY_ELEMENT_MULT_TYPE,
      OPERATION_BINARY_ELEMENT_DIV_TYPE
    };



    namespace result_of
    {
      template <typename T>
      struct op_type_info
      {
        typedef typename T::ERROR_UNKNOWN_OP_TYPE   error_type;
      };

      // unary operations
      template <> struct op_type_info<op_element_unary<op_abs>   > { enum { id = OPERATION_UNARY_ABS_TYPE,   family = OPERATION_UNARY_TYPE_FAMILY }; };
      template <> struct op_type_info<op_element_unary<op_acos>  > { enum { id = OPERATION_UNARY_ACOS_TYPE,  family = OPERATION_UNARY_TYPE_FAMILY }; };
      template <> struct op_type_info<op_element_unary<op_asin>  > { enum { id = OPERATION_UNARY_ASIN_TYPE,  family = OPERATION_UNARY_TYPE_FAMILY }; };
      template <> struct op_type_info<op_element_unary<op_atan>  > { enum { id = OPERATION_UNARY_ATAN_TYPE,  family = OPERATION_UNARY_TYPE_FAMILY }; };
      template <> struct op_type_info<op_element_unary<op_ceil>  > { enum { id = OPERATION_UNARY_CEIL_TYPE,  family = OPERATION_UNARY_TYPE_FAMILY }; };
      template <> struct op_type_info<op_element_unary<op_cos>   > { enum { id = OPERATION_UNARY_COS_TYPE,   family = OPERATION_UNARY_TYPE_FAMILY }; };
      template <> struct op_type_info<op_element_unary<op_cosh>  > { enum { id = OPERATION_UNARY_COSH_TYPE,  family = OPERATION_UNARY_TYPE_FAMILY }; };
      template <> struct op_type_info<op_element_unary<op_exp>   > { enum { id = OPERATION_UNARY_EXP_TYPE,   family = OPERATION_UNARY_TYPE_FAMILY }; };
      template <> struct op_type_info<op_element_unary<op_fabs>  > { enum { id = OPERATION_UNARY_FABS_TYPE,  family = OPERATION_UNARY_TYPE_FAMILY }; };
      template <> struct op_type_info<op_element_unary<op_floor> > { enum { id = OPERATION_UNARY_FLOOR_TYPE, family = OPERATION_UNARY_TYPE_FAMILY }; };
      template <> struct op_type_info<op_element_unary<op_log>   > { enum { id = OPERATION_UNARY_LOG_TYPE,   family = OPERATION_UNARY_TYPE_FAMILY }; };
      template <> struct op_type_info<op_element_unary<op_log10> > { enum { id = OPERATION_UNARY_LOG10_TYPE, family = OPERATION_UNARY_TYPE_FAMILY }; };
      template <> struct op_type_info<op_element_unary<op_sin>   > { enum { id = OPERATION_UNARY_SIN_TYPE,   family = OPERATION_UNARY_TYPE_FAMILY }; };
      template <> struct op_type_info<op_element_unary<op_sinh>  > { enum { id = OPERATION_UNARY_SINH_TYPE,  family = OPERATION_UNARY_TYPE_FAMILY }; };
      template <> struct op_type_info<op_element_unary<op_sqrt>  > { enum { id = OPERATION_UNARY_SQRT_TYPE,  family = OPERATION_UNARY_TYPE_FAMILY }; };
      template <> struct op_type_info<op_element_unary<op_tan>   > { enum { id = OPERATION_UNARY_TAN_TYPE,   family = OPERATION_UNARY_TYPE_FAMILY }; };
      template <> struct op_type_info<op_element_unary<op_tanh>  > { enum { id = OPERATION_UNARY_TANH_TYPE,  family = OPERATION_UNARY_TYPE_FAMILY }; };

      // binary operations
      template <> struct op_type_info<op_assign>                   { enum { id = OPERATION_BINARY_ASSIGN_TYPE,       family = OPERATION_BINARY_TYPE_FAMILY }; };
      template <> struct op_type_info<op_inplace_add>              { enum { id = OPERATION_BINARY_INPLACE_ADD_TYPE,  family = OPERATION_BINARY_TYPE_FAMILY }; };
      template <> struct op_type_info<op_inplace_sub>              { enum { id = OPERATION_BINARY_INPLACE_SUB_TYPE,  family = OPERATION_BINARY_TYPE_FAMILY }; };
      template <> struct op_type_info<op_add>                      { enum { id = OPERATION_BINARY_ADD_TYPE,          family = OPERATION_BINARY_TYPE_FAMILY }; };
      template <> struct op_type_info<op_sub>                      { enum { id = OPERATION_BINARY_SUB_TYPE,          family = OPERATION_BINARY_TYPE_FAMILY }; };
      template <> struct op_type_info<op_prod>                     { enum { id = OPERATION_BINARY_PROD_TYPE,         family = OPERATION_BINARY_TYPE_FAMILY }; };
      template <> struct op_type_info<op_mult>                     { enum { id = OPERATION_BINARY_MULT_TYPE,         family = OPERATION_BINARY_TYPE_FAMILY }; };
      template <> struct op_type_info<op_element_binary<op_mult> > { enum { id = OPERATION_BINARY_ELEMENT_MULT_TYPE, family = OPERATION_BINARY_TYPE_FAMILY }; };
      template <> struct op_type_info<op_element_binary<op_div>  > { enum { id = OPERATION_BINARY_ELEMENT_DIV_TYPE,  family = OPERATION_BINARY_TYPE_FAMILY }; };

    } // namespace result_of





    /** @brief Groups the type of a node in the statement tree. Used for faster dispatching */
    enum statement_node_type_family
    {
      // LHS or RHS are again an expression:
      COMPOSITE_OPERATION_FAMILY,

      // host scalars:
      HOST_SCALAR_TYPE_FAMILY,

      // device scalars:
      SCALAR_TYPE_FAMILY,

      // vector:
      VECTOR_TYPE_FAMILY,

      // matrices:
      MATRIX_ROW_TYPE_FAMILY,
      MATRIX_COL_TYPE_FAMILY
    };

    /** @brief Encodes the type of a node in the statement tree. */
    enum statement_node_type
    {
      COMPOSITE_OPERATION_TYPE,

      // host scalars:
      HOST_SCALAR_CHAR_TYPE,
      HOST_SCALAR_UCHAR_TYPE,
      HOST_SCALAR_SHORT_TYPE,
      HOST_SCALAR_USHORT_TYPE,
      HOST_SCALAR_INT_TYPE,
      HOST_SCALAR_UINT_TYPE,
      HOST_SCALAR_LONG_TYPE,
      HOST_SCALAR_ULONG_TYPE,
      HOST_SCALAR_HALF_TYPE,
      HOST_SCALAR_FLOAT_TYPE,
      HOST_SCALAR_DOUBLE_TYPE,

      // device scalars:
      SCALAR_CHAR_TYPE,
      SCALAR_UCHAR_TYPE,
      SCALAR_SHORT_TYPE,
      SCALAR_USHORT_TYPE,
      SCALAR_INT_TYPE,
      SCALAR_UINT_TYPE,
      SCALAR_LONG_TYPE,
      SCALAR_ULONG_TYPE,
      SCALAR_HALF_TYPE,
      SCALAR_FLOAT_TYPE,
      SCALAR_DOUBLE_TYPE,

      // vector:
      VECTOR_CHAR_TYPE,
      VECTOR_UCHAR_TYPE,
      VECTOR_SHORT_TYPE,
      VECTOR_USHORT_TYPE,
      VECTOR_INT_TYPE,
      VECTOR_UINT_TYPE,
      VECTOR_LONG_TYPE,
      VECTOR_ULONG_TYPE,
      VECTOR_HALF_TYPE,
      VECTOR_FLOAT_TYPE,
      VECTOR_DOUBLE_TYPE,

      // matrix, row major:
      MATRIX_ROW_CHAR_TYPE,
      MATRIX_ROW_UCHAR_TYPE,
      MATRIX_ROW_SHORT_TYPE,
      MATRIX_ROW_USHORT_TYPE,
      MATRIX_ROW_INT_TYPE,
      MATRIX_ROW_UINT_TYPE,
      MATRIX_ROW_LONG_TYPE,
      MATRIX_ROW_ULONG_TYPE,
      MATRIX_ROW_HALF_TYPE,
      MATRIX_ROW_FLOAT_TYPE,
      MATRIX_ROW_DOUBLE_TYPE,

      // matrix, row major:
      MATRIX_COL_CHAR_TYPE,
      MATRIX_COL_UCHAR_TYPE,
      MATRIX_COL_SHORT_TYPE,
      MATRIX_COL_USHORT_TYPE,
      MATRIX_COL_INT_TYPE,
      MATRIX_COL_UINT_TYPE,
      MATRIX_COL_LONG_TYPE,
      MATRIX_COL_ULONG_TYPE,
      MATRIX_COL_HALF_TYPE,
      MATRIX_COL_FLOAT_TYPE,
      MATRIX_COL_DOUBLE_TYPE
    };

    namespace result_of
    {
      template <typename T>
      struct vector_type_for_scalar {};

#define VIENNACL_GENERATE_VECTOR_TYPE_MAPPING(TYPE, ENUMVALUE) \
      template <> struct vector_type_for_scalar<TYPE> { enum { value = ENUMVALUE }; };

      VIENNACL_GENERATE_VECTOR_TYPE_MAPPING(char,           VECTOR_CHAR_TYPE);
      VIENNACL_GENERATE_VECTOR_TYPE_MAPPING(unsigned char,  VECTOR_UCHAR_TYPE);
      VIENNACL_GENERATE_VECTOR_TYPE_MAPPING(short,          VECTOR_SHORT_TYPE);
      VIENNACL_GENERATE_VECTOR_TYPE_MAPPING(unsigned short, VECTOR_USHORT_TYPE);
      VIENNACL_GENERATE_VECTOR_TYPE_MAPPING(int,            VECTOR_INT_TYPE);
      VIENNACL_GENERATE_VECTOR_TYPE_MAPPING(unsigned int,   VECTOR_UINT_TYPE);
      VIENNACL_GENERATE_VECTOR_TYPE_MAPPING(long,           VECTOR_LONG_TYPE);
      VIENNACL_GENERATE_VECTOR_TYPE_MAPPING(unsigned long,  VECTOR_ULONG_TYPE);
      VIENNACL_GENERATE_VECTOR_TYPE_MAPPING(float,          VECTOR_FLOAT_TYPE);
      VIENNACL_GENERATE_VECTOR_TYPE_MAPPING(double,         VECTOR_DOUBLE_TYPE);

#undef VIENNACL_GENERATE_VECTOR_TYPE_MAPPING
    }



    /** @brief A union representing the 'data' for the LHS operand of the respective node.
      *
      * If it represents a compound expression, the union holds the array index within the respective statement array.
      * If it represents a object with data (vector, matrix, etc.) it holds the respective pointer (scalar, vector, matrix) or value (host scalar)
      */
    typedef union lhs_rhs_union_t
    {
      /////// Case 1: Node is another compound expression:
      std::size_t        node_index_;

      /////// Case 2: Node is a leaf, hence carries an operand:

      // host scalars:
      char               host_char_;
      unsigned char      host_uchar_;
      short              host_short_;
      unsigned short     host_ushort_;
      int                host_int_;
      unsigned int       host_uint_;
      long               host_long_;
      unsigned long      host_ulong_;
      float              host_float_;
      double             host_double_;

      // Note: ViennaCL types have potentially expensive copy-CTORs, hence using pointers:

      // scalars:
      //viennacl::scalar<char>             *scalar_char_;
      //viennacl::scalar<unsigned char>    *scalar_uchar_;
      //viennacl::scalar<short>            *scalar_short_;
      //viennacl::scalar<unsigned short>   *scalar_ushort_;
      //viennacl::scalar<int>              *scalar_int_;
      //viennacl::scalar<unsigned int>     *scalar_uint_;
      //viennacl::scalar<long>             *scalar_long_;
      //viennacl::scalar<unsigned long>    *scalar_ulong_;
      viennacl::scalar<float>            *scalar_float_;
      viennacl::scalar<double>           *scalar_double_;

      // vectors:
      //viennacl::vector_base<char>             *vector_char_;
      //viennacl::vector_base<unsigned char>    *vector_uchar_;
      //viennacl::vector_base<short>            *vector_short_;
      //viennacl::vector_base<unsigned short>   *vector_ushort_;
      //viennacl::vector_base<int>              *vector_int_;
      //viennacl::vector_base<unsigned int>     *vector_uint_;
      //viennacl::vector_base<long>             *vector_long_;
      //viennacl::vector_base<unsigned long>    *vector_ulong_;
      viennacl::vector_base<float>            *vector_float_;
      viennacl::vector_base<double>           *vector_double_;

      // row-major matrices:
      //viennacl::matrix_base<char>             *matrix_row_char_;
      //viennacl::matrix_base<unsigned char>    *matrix_row_uchar_;
      //viennacl::matrix_base<short>            *matrix_row_short_;
      //viennacl::matrix_base<unsigned short>   *matrix_row_ushort_;
      //viennacl::matrix_base<int>              *matrix_row_int_;
      //viennacl::matrix_base<unsigned int>     *matrix_row_uint_;
      //viennacl::matrix_base<long>             *matrix_row_long_;
      //viennacl::matrix_base<unsigned long>    *matrix_row_ulong_;
      viennacl::matrix_base<float>            *matrix_row_float_;
      viennacl::matrix_base<double>           *matrix_row_double_;

      // column-major matrices:
      //viennacl::matrix_base<char,           viennacl::column_major>    *matrix_col_char_;
      //viennacl::matrix_base<unsigned char,  viennacl::column_major>    *matrix_col_uchar_;
      //viennacl::matrix_base<short,          viennacl::column_major>    *matrix_col_short_;
      //viennacl::matrix_base<unsigned short, viennacl::column_major>    *matrix_col_ushort_;
      //viennacl::matrix_base<int,            viennacl::column_major>    *matrix_col_int_;
      //viennacl::matrix_base<unsigned int,   viennacl::column_major>    *matrix_col_uint_;
      //viennacl::matrix_base<long,           viennacl::column_major>    *matrix_col_long_;
      //viennacl::matrix_base<unsigned long,  viennacl::column_major>    *matrix_col_ulong_;
      viennacl::matrix_base<float,          viennacl::column_major>    *matrix_col_float_;
      viennacl::matrix_base<double,         viennacl::column_major>    *matrix_col_double_;

    } lhs_rhs_element;


    /** @brief Main datastructure for an node in the statement tree */
    struct statement_node
    {
      statement_node_type_family   lhs_type_family_;
      statement_node_type          lhs_type_;
      lhs_rhs_element              lhs_;

      statement_node_type_family   rhs_type_family_;
      statement_node_type          rhs_type_;
      lhs_rhs_element              rhs_;

      operation_node_type_family   op_family_;
      operation_node_type          op_type_;
      // note: since operation tags are state-less, no 'op_' object is needed here.
    };

    namespace result_of{

      template<class T> struct num_nodes { enum { value = 0 }; };
      template<class LHS, class OP, class RHS> struct num_nodes< vector_expression<LHS, OP, RHS> > { enum { value = 1 + num_nodes<LHS>::value + num_nodes<RHS>::value }; };
      template<class LHS, class OP, class RHS> struct num_nodes< matrix_expression<LHS, OP, RHS> > { enum { value = 1 + num_nodes<LHS>::value + num_nodes<RHS>::value }; };
      template<class LHS, class OP, class RHS> struct num_nodes< scalar_expression<LHS, OP, RHS> > { enum { value = 1 + num_nodes<LHS>::value + num_nodes<RHS>::value }; };

    }

    class statement
    {
      public:
        typedef statement_node              value_type;
        typedef std::vector<value_type>     container_type;

        template <typename LHS, typename OP, typename RHS>
        statement(LHS & lhs, OP const & op, RHS const & rhs) : array_(1 + result_of::num_nodes<RHS>::value)
        {
          // set OP:
          array_[0].op_family_ = operation_node_type_family(result_of::op_type_info<OP>::family);
          array_[0].op_type_   = operation_node_type(result_of::op_type_info<OP>::id);

          // set LHS:
          add_lhs(0, 1, lhs);

          // set RHS:
          add_rhs(0, 1, rhs);
        }

        container_type const & array() const { return array_; }

      private:

        // TODO: add integer vector overloads here
        void assign_element(lhs_rhs_element & element_, viennacl::vector_base<float>  const & t) { element_.vector_float_  = const_cast<viennacl::vector_base<float> *>(&t); }
        void assign_element(lhs_rhs_element & element_, viennacl::vector_base<double> const & t) { element_.vector_double_ = const_cast<viennacl::vector_base<double> *>(&t); }

        template <typename T>
        std::size_t add_element(std::size_t next_free,
                                statement_node_type_family & type_family_,
                                statement_node_type        & type_,
                                lhs_rhs_element            & element_,
                                viennacl::vector_base<T> const & t)
        {
          type_family_           = VECTOR_TYPE_FAMILY;
          type_                  = statement_node_type(result_of::vector_type_for_scalar<T>::value);
          assign_element(element_, t);
          return next_free;
        }

        template <typename LHS, typename RHS, typename OP>
        std::size_t add_element(std::size_t next_free,
                                statement_node_type_family & type_family_,
                                statement_node_type        & type_,
                                lhs_rhs_element            & element_,
                                viennacl::vector_expression<LHS, RHS, OP> const & t)
        {
          type_family_           = COMPOSITE_OPERATION_FAMILY;
          type_                  = COMPOSITE_OPERATION_TYPE;
          element_.node_index_   = next_free;
          return add_node(next_free, next_free + 1, t);
        }

        template <typename T>
        std::size_t add_lhs(std::size_t current_index, std::size_t next_free, T const & t)
        {
          return add_element(next_free,
                             array_[current_index].lhs_type_family_,
                             array_[current_index].lhs_type_,
                             array_[current_index].lhs_,
                             t);
        }

        template <typename T>
        std::size_t add_rhs(std::size_t current_index, std::size_t next_free, T const & t)
        {
          return add_element(next_free,
                             array_[current_index].rhs_type_family_,
                             array_[current_index].rhs_type_,
                             array_[current_index].rhs_,
                             t);
        }

        template <typename LHS, typename OP, typename RHS>
        std::size_t add_node(std::size_t current_index, std::size_t next_free, viennacl::vector_expression<LHS, RHS, OP> const & proxy)
        {
          // set OP:
          array_[current_index].op_family_ = operation_node_type_family(result_of::op_type_info<OP>::family);
          array_[current_index].op_type_   = operation_node_type(result_of::op_type_info<OP>::id);

          // set LHS and RHS:
          return add_rhs(current_index, add_lhs(current_index, next_free, proxy.lhs()), proxy.rhs());
        }

        container_type   array_;
    };

  } // namespace scheduler

} // namespace viennacl

#endif

