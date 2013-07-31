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

    /** @brief Exception for the case the scheduler is unable to deal with the operation */
    class statement_not_supported_exception : public std::exception
    {
    public:
      statement_not_supported_exception() : message_() {}
      statement_not_supported_exception(std::string message) : message_("ViennaCL: Internal error: The scheduler encountered a problem with the operation provided: " + message) {}

      virtual const char* what() const throw() { return message_.c_str(); }

      virtual ~statement_not_supported_exception() throw() {}
    private:
      std::string message_;
    };


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
      OPERATION_UNARY_TRANS_TYPE,
      OPERATION_UNARY_NORM_1_TYPE,
      OPERATION_UNARY_NORM_2_TYPE,
      OPERATION_UNARY_NORM_INF_TYPE,

      // binary expression
      OPERATION_BINARY_ACCESS_TYPE,
      OPERATION_BINARY_ASSIGN_TYPE,
      OPERATION_BINARY_INPLACE_ADD_TYPE,
      OPERATION_BINARY_INPLACE_SUB_TYPE,
      OPERATION_BINARY_ADD_TYPE,
      OPERATION_BINARY_SUB_TYPE,
      OPERATION_BINARY_MAT_VEC_PROD_TYPE,
      OPERATION_BINARY_MAT_MAT_PROD_TYPE,
      OPERATION_BINARY_MULT_TYPE,    // scalar times vector/matrix
      OPERATION_BINARY_DIV_TYPE,     // vector/matrix divided by scalar
      OPERATION_BINARY_ELEMENT_MULT_TYPE,
      OPERATION_BINARY_ELEMENT_DIV_TYPE,
      OPERATION_BINARY_INNER_PROD_TYPE
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
      template <> struct op_type_info<op_norm_1                  > { enum { id = OPERATION_UNARY_NORM_1_TYPE,   family = OPERATION_UNARY_TYPE_FAMILY }; };
      template <> struct op_type_info<op_norm_2                  > { enum { id = OPERATION_UNARY_NORM_2_TYPE,   family = OPERATION_UNARY_TYPE_FAMILY }; };
      template <> struct op_type_info<op_norm_inf                > { enum { id = OPERATION_UNARY_NORM_INF_TYPE, family = OPERATION_UNARY_TYPE_FAMILY }; };
      template <> struct op_type_info<op_trans                   > { enum { id = OPERATION_UNARY_TRANS_TYPE, family = OPERATION_UNARY_TYPE_FAMILY }; };

      // binary operations
      template <> struct op_type_info<op_assign>                   { enum { id = OPERATION_BINARY_ASSIGN_TYPE,       family = OPERATION_BINARY_TYPE_FAMILY }; };
      template <> struct op_type_info<op_inplace_add>              { enum { id = OPERATION_BINARY_INPLACE_ADD_TYPE,  family = OPERATION_BINARY_TYPE_FAMILY }; };
      template <> struct op_type_info<op_inplace_sub>              { enum { id = OPERATION_BINARY_INPLACE_SUB_TYPE,  family = OPERATION_BINARY_TYPE_FAMILY }; };
      template <> struct op_type_info<op_add>                      { enum { id = OPERATION_BINARY_ADD_TYPE,          family = OPERATION_BINARY_TYPE_FAMILY }; };
      template <> struct op_type_info<op_sub>                      { enum { id = OPERATION_BINARY_SUB_TYPE,          family = OPERATION_BINARY_TYPE_FAMILY }; };
      template <> struct op_type_info<op_prod>                     { enum { id = OPERATION_BINARY_MAT_VEC_PROD_TYPE, family = OPERATION_BINARY_TYPE_FAMILY }; };
      template <> struct op_type_info<op_mat_mat_prod>             { enum { id = OPERATION_BINARY_MAT_MAT_PROD_TYPE, family = OPERATION_BINARY_TYPE_FAMILY }; };
      template <> struct op_type_info<op_mult>                     { enum { id = OPERATION_BINARY_MULT_TYPE,         family = OPERATION_BINARY_TYPE_FAMILY }; };
      template <> struct op_type_info<op_div>                      { enum { id = OPERATION_BINARY_DIV_TYPE,          family = OPERATION_BINARY_TYPE_FAMILY }; };
      template <> struct op_type_info<op_element_binary<op_mult> > { enum { id = OPERATION_BINARY_ELEMENT_MULT_TYPE, family = OPERATION_BINARY_TYPE_FAMILY }; };
      template <> struct op_type_info<op_element_binary<op_div>  > { enum { id = OPERATION_BINARY_ELEMENT_DIV_TYPE,  family = OPERATION_BINARY_TYPE_FAMILY }; };
      template <> struct op_type_info<op_inner_prod>               { enum { id = OPERATION_BINARY_INNER_PROD_TYPE,   family = OPERATION_BINARY_TYPE_FAMILY }; };

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

      // symbolic vector:
      SYMBOLIC_VECTOR_TYPE_FAMILY,

      // matrices:
      MATRIX_ROW_TYPE_FAMILY,
      MATRIX_COL_TYPE_FAMILY,

      // symbolic matrix:
      SYMBOLIC_MATRIX_TYPE_FAMILY
    };

    /** @brief Encodes the type of a node in the statement tree. */
    enum statement_node_type
    {
      COMPOSITE_OPERATION_TYPE,

      CHAR_TYPE,
      UCHAR_TYPE,
      SHORT_TYPE,
      USHORT_TYPE,
      INT_TYPE,
      UINT_TYPE,
      LONG_TYPE,
      ULONG_TYPE,
      HALF_TYPE,
      FLOAT_TYPE,
      DOUBLE_TYPE

    };

    namespace result_of
    {
      ///////////// scalar type ID deduction /////////////

      template <typename T>
      struct scalar_type {};

      template <> struct scalar_type<char>           { enum { value = CHAR_TYPE   }; };
      template <> struct scalar_type<unsigned char>  { enum { value = UCHAR_TYPE  }; };
      template <> struct scalar_type<short>          { enum { value = SHORT_TYPE  }; };
      template <> struct scalar_type<unsigned short> { enum { value = USHORT_TYPE }; };
      template <> struct scalar_type<int>            { enum { value = INT_TYPE    }; };
      template <> struct scalar_type<unsigned int>   { enum { value = UINT_TYPE   }; };
      template <> struct scalar_type<long>           { enum { value = LONG_TYPE   }; };
      template <> struct scalar_type<unsigned long>  { enum { value = ULONG_TYPE  }; };
      template <> struct scalar_type<float>          { enum { value = FLOAT_TYPE  }; };
      template <> struct scalar_type<double>         { enum { value = DOUBLE_TYPE }; };

      ///////////// vector type ID deduction /////////////

      template <typename T>
      struct vector_type_for_scalar {};

#define VIENNACL_GENERATE_VECTOR_TYPE_MAPPING(TYPE, ENUMVALUE) \
      template <> struct vector_type_for_scalar<TYPE> { enum { value = ENUMVALUE }; };

      VIENNACL_GENERATE_VECTOR_TYPE_MAPPING(char,           CHAR_TYPE)
      VIENNACL_GENERATE_VECTOR_TYPE_MAPPING(unsigned char,  UCHAR_TYPE)
      VIENNACL_GENERATE_VECTOR_TYPE_MAPPING(short,          SHORT_TYPE)
      VIENNACL_GENERATE_VECTOR_TYPE_MAPPING(unsigned short, USHORT_TYPE)
      VIENNACL_GENERATE_VECTOR_TYPE_MAPPING(int,            INT_TYPE)
      VIENNACL_GENERATE_VECTOR_TYPE_MAPPING(unsigned int,   UINT_TYPE)
      VIENNACL_GENERATE_VECTOR_TYPE_MAPPING(long,           LONG_TYPE)
      VIENNACL_GENERATE_VECTOR_TYPE_MAPPING(unsigned long,  ULONG_TYPE)
      VIENNACL_GENERATE_VECTOR_TYPE_MAPPING(float,          FLOAT_TYPE)
      VIENNACL_GENERATE_VECTOR_TYPE_MAPPING(double,         DOUBLE_TYPE)

#undef VIENNACL_GENERATE_VECTOR_TYPE_MAPPING

      ///////////// symbolic vector ID deduction /////////

      template <typename T>
      struct symbolic_vector_type_for_scalar {};
#define VIENNACL_GENERATE_SYMBOLIC_VECTOR_TYPE_MAPPING(TYPE, ENUMVALUE) \
      template <> struct symbolic_vector_type_for_scalar<TYPE> { enum { value = ENUMVALUE }; }

      VIENNACL_GENERATE_SYMBOLIC_VECTOR_TYPE_MAPPING(char,           CHAR_TYPE);
      VIENNACL_GENERATE_SYMBOLIC_VECTOR_TYPE_MAPPING(unsigned char,  UCHAR_TYPE);
      VIENNACL_GENERATE_SYMBOLIC_VECTOR_TYPE_MAPPING(short,          SHORT_TYPE);
      VIENNACL_GENERATE_SYMBOLIC_VECTOR_TYPE_MAPPING(unsigned short, USHORT_TYPE);
      VIENNACL_GENERATE_SYMBOLIC_VECTOR_TYPE_MAPPING(int,            INT_TYPE);
      VIENNACL_GENERATE_SYMBOLIC_VECTOR_TYPE_MAPPING(unsigned int,   UINT_TYPE);
      VIENNACL_GENERATE_SYMBOLIC_VECTOR_TYPE_MAPPING(long,           LONG_TYPE);
      VIENNACL_GENERATE_SYMBOLIC_VECTOR_TYPE_MAPPING(unsigned long,  ULONG_TYPE);
      VIENNACL_GENERATE_SYMBOLIC_VECTOR_TYPE_MAPPING(float,          FLOAT_TYPE);
      VIENNACL_GENERATE_SYMBOLIC_VECTOR_TYPE_MAPPING(double,         DOUBLE_TYPE);

#undef VIENNACL_GENERATE_SYMBOLIC_VECTOR_TYPE_MAPPING

      ///////////// matrix type ID deduction /////////////

      template <typename T, typename F>
      struct matrix_type_for_scalar_and_layout {};

#define VIENNACL_GENERATE_MATRIX_TYPE_MAPPING(TYPE, LAYOUT, ENUMVALUE) \
      template <> struct matrix_type_for_scalar_and_layout<TYPE, LAYOUT> { enum { value = ENUMVALUE }; };

      VIENNACL_GENERATE_MATRIX_TYPE_MAPPING(char,           viennacl::column_major, CHAR_TYPE)
      VIENNACL_GENERATE_MATRIX_TYPE_MAPPING(unsigned char,  viennacl::column_major, UCHAR_TYPE)
      VIENNACL_GENERATE_MATRIX_TYPE_MAPPING(short,          viennacl::column_major, SHORT_TYPE)
      VIENNACL_GENERATE_MATRIX_TYPE_MAPPING(unsigned short, viennacl::column_major, USHORT_TYPE)
      VIENNACL_GENERATE_MATRIX_TYPE_MAPPING(int,            viennacl::column_major, INT_TYPE)
      VIENNACL_GENERATE_MATRIX_TYPE_MAPPING(unsigned int,   viennacl::column_major, UINT_TYPE)
      VIENNACL_GENERATE_MATRIX_TYPE_MAPPING(long,           viennacl::column_major, LONG_TYPE)
      VIENNACL_GENERATE_MATRIX_TYPE_MAPPING(unsigned long,  viennacl::column_major, ULONG_TYPE)
      VIENNACL_GENERATE_MATRIX_TYPE_MAPPING(float,          viennacl::column_major, FLOAT_TYPE)
      VIENNACL_GENERATE_MATRIX_TYPE_MAPPING(double,         viennacl::column_major, DOUBLE_TYPE)

      VIENNACL_GENERATE_MATRIX_TYPE_MAPPING(char,           viennacl::row_major, CHAR_TYPE)
      VIENNACL_GENERATE_MATRIX_TYPE_MAPPING(unsigned char,  viennacl::row_major, UCHAR_TYPE)
      VIENNACL_GENERATE_MATRIX_TYPE_MAPPING(short,          viennacl::row_major, SHORT_TYPE)
      VIENNACL_GENERATE_MATRIX_TYPE_MAPPING(unsigned short, viennacl::row_major, USHORT_TYPE)
      VIENNACL_GENERATE_MATRIX_TYPE_MAPPING(int,            viennacl::row_major, INT_TYPE)
      VIENNACL_GENERATE_MATRIX_TYPE_MAPPING(unsigned int,   viennacl::row_major, UINT_TYPE)
      VIENNACL_GENERATE_MATRIX_TYPE_MAPPING(long,           viennacl::row_major, LONG_TYPE)
      VIENNACL_GENERATE_MATRIX_TYPE_MAPPING(unsigned long,  viennacl::row_major, ULONG_TYPE)
      VIENNACL_GENERATE_MATRIX_TYPE_MAPPING(float,          viennacl::row_major, FLOAT_TYPE)
      VIENNACL_GENERATE_MATRIX_TYPE_MAPPING(double,         viennacl::row_major, DOUBLE_TYPE)

#undef VIENNACL_GENERATE_VECTOR_TYPE_MAPPING

      ///////// symbolic matrix ID deduction ///////

      template <typename T>
      struct symbolic_matrix_type_for_scalar{};

#define VIENNACL_GENERATE_SYMBOLIC_MATRIX_TYPE_MAPPING(TYPE, ENUMVALUE) \
      template <> struct symbolic_matrix_type_for_scalar<TYPE> { enum { value = ENUMVALUE }; }

      VIENNACL_GENERATE_SYMBOLIC_MATRIX_TYPE_MAPPING(char,           CHAR_TYPE);
      VIENNACL_GENERATE_SYMBOLIC_MATRIX_TYPE_MAPPING(unsigned char,  UCHAR_TYPE);
      VIENNACL_GENERATE_SYMBOLIC_MATRIX_TYPE_MAPPING(short,          SHORT_TYPE);
      VIENNACL_GENERATE_SYMBOLIC_MATRIX_TYPE_MAPPING(unsigned short, USHORT_TYPE);
      VIENNACL_GENERATE_SYMBOLIC_MATRIX_TYPE_MAPPING(int,            INT_TYPE);
      VIENNACL_GENERATE_SYMBOLIC_MATRIX_TYPE_MAPPING(unsigned int,   UINT_TYPE);
      VIENNACL_GENERATE_SYMBOLIC_MATRIX_TYPE_MAPPING(long,           LONG_TYPE);
      VIENNACL_GENERATE_SYMBOLIC_MATRIX_TYPE_MAPPING(unsigned long,  ULONG_TYPE);
      VIENNACL_GENERATE_SYMBOLIC_MATRIX_TYPE_MAPPING(float,          FLOAT_TYPE);
      VIENNACL_GENERATE_SYMBOLIC_MATRIX_TYPE_MAPPING(double,         DOUBLE_TYPE);
#undef VIENNACL_GENERATE_SYMBOLIC_MATRIX_TYPE_MAPPING



      template <typename F>
      struct matrix_family {};

      template <> struct matrix_family<viennacl::row_major   > { enum { value = MATRIX_ROW_TYPE_FAMILY }; };
      template <> struct matrix_family<viennacl::column_major> { enum { value = MATRIX_COL_TYPE_FAMILY }; };



    }



    /** @brief A class representing the 'data' for the LHS or RHS operand of the respective node.
      *
      * If it represents a compound expression, the union holds the array index within the respective statement array.
      * If it represents a object with data (vector, matrix, etc.) it holds the respective pointer (scalar, vector, matrix) or value (host scalar)
      *
      * The member 'type_family' is an optimization for quickly retrieving the 'type', which denotes the currently 'active' member in the union
      */
    struct lhs_rhs_element
    {
      statement_node_type_family   type_family;
      statement_node_type          type;

      union
      {
        /////// Case 1: Node is another compound expression:
        std::size_t        node_index;

        /////// Case 2: Node is a leaf, hence carries an operand:

        // host scalars:
        char               host_char;
        unsigned char      host_uchar;
        short              host_short;
        unsigned short     host_ushort;
        int                host_int;
        unsigned int       host_uint;
        long               host_long;
        unsigned long      host_ulong;
        float              host_float;
        double             host_double;

        // Note: ViennaCL types have potentially expensive copy-CTORs, hence using pointers:

        // scalars:
        //viennacl::scalar<char>             *scalar_char;
        //viennacl::scalar<unsigned char>    *scalar_uchar;
        //viennacl::scalar<short>            *scalar_short;
        //viennacl::scalar<unsigned short>   *scalar_ushort;
        //viennacl::scalar<int>              *scalar_int;
        //viennacl::scalar<unsigned int>     *scalar_uint;
        //viennacl::scalar<long>             *scalar_long;
        //viennacl::scalar<unsigned long>    *scalar_ulong;
        viennacl::scalar<float>            *scalar_float;
        viennacl::scalar<double>           *scalar_double;

        // vectors:
        //viennacl::vector_base<char>             *vector_char;
        //viennacl::vector_base<unsigned char>    *vector_uchar;
        //viennacl::vector_base<short>            *vector_short;
        //viennacl::vector_base<unsigned short>   *vector_ushort;
        //viennacl::vector_base<int>              *vector_int;
        //viennacl::vector_base<unsigned int>     *vector_uint;
        //viennacl::vector_base<long>             *vector_long;
        //viennacl::vector_base<unsigned long>    *vector_ulong;
        viennacl::vector_base<float>            *vector_float;
        viennacl::vector_base<double>           *vector_double;

        // symbolic vectors:
        //viennacl::symbolic_vector_base<char>             *symbolic_vector_char;
         //viennacl::symbolic_vector_base<unsigned char>    *symbolic_vector_uchar;
         //viennacl::symbolic_vector_base<short>            *symbolic_vector_short;
         //viennacl::symbolic_vector_base<unsigned short>   *symbolic_vector_ushort;
         //viennacl::symbolic_vector_base<int>              *symbolic_vector_int;
         //viennacl::symbolic_vector_base<unsigned int>     *symbolic_vector_uint;
         //viennacl::symbolic_vector_base<long>             *symbolic_vector_long;
         //viennacl::symbolic_vector_base<unsigned long>    *symbolic_vector_ulong;
         viennacl::symbolic_vector_base<float>            *symbolic_vector_float;
         viennacl::symbolic_vector_base<double>           *symbolic_vector_double;

        // row-major matrices:
        //viennacl::matrix_base<char>             *matrix_row_char;
        //viennacl::matrix_base<unsigned char>    *matrix_row_uchar;
        //viennacl::matrix_base<short>            *matrix_row_short;
        //viennacl::matrix_base<unsigned short>   *matrix_row_ushort;
        //viennacl::matrix_base<int>              *matrix_row_int;
        //viennacl::matrix_base<unsigned int>     *matrix_row_uint;
        //viennacl::matrix_base<long>             *matrix_row_long;
        //viennacl::matrix_base<unsigned long>    *matrix_row_ulong;
        viennacl::matrix_base<float>            *matrix_row_float;
        viennacl::matrix_base<double>           *matrix_row_double;

        // column-major matrices:
        //viennacl::matrix_base<char,           viennacl::column_major>    *matrix_col_char;
        //viennacl::matrix_base<unsigned char,  viennacl::column_major>    *matrix_col_uchar;
        //viennacl::matrix_base<short,          viennacl::column_major>    *matrix_col_short;
        //viennacl::matrix_base<unsigned short, viennacl::column_major>    *matrix_col_ushort;
        //viennacl::matrix_base<int,            viennacl::column_major>    *matrix_col_int;
        //viennacl::matrix_base<unsigned int,   viennacl::column_major>    *matrix_col_uint;
        //viennacl::matrix_base<long,           viennacl::column_major>    *matrix_col_long;
        //viennacl::matrix_base<unsigned long,  viennacl::column_major>    *matrix_col_ulong;
        viennacl::matrix_base<float,          viennacl::column_major>    *matrix_col_float;
        viennacl::matrix_base<double,         viennacl::column_major>    *matrix_col_double;

        //viennacl::symbolic_matrix_base<char>             *symbolic_matrix_char;
        //viennacl::symbolic_matrix_base<unsigned char>    *symbolic_matrix_uchar;
        //viennacl::symbolic_matrix_base<short>            *symbolic_matrix_short;
        //viennacl::symbolic_matrix_base<unsigned short>   *symbolic_matrix_ushort;
        //viennacl::symbolic_matrix_base<int>              *symbolic_matrix_int;
        //viennacl::symbolic_matrix_base<unsigned int>     *symbolic_matrix_uint;
        //viennacl::symbolic_matrix_base<long>             *symbolic_matrix_long;
        //viennacl::symbolic_matrix_base<unsigned long>    *symbolic_matrix_ulong;
        viennacl::symbolic_matrix_base<float>            *symbolic_matrix_float;
        viennacl::symbolic_matrix_base<double>           *symbolic_matrix_double;

      };
    };


    struct op_element
    {
      operation_node_type_family   type_family;
      operation_node_type          type;
    };

    /** @brief Main datastructure for an node in the statement tree */
    struct statement_node
    {
      lhs_rhs_element    lhs;
      op_element         op;
      lhs_rhs_element    rhs;
    };

    namespace result_of
    {

      template <class T> struct num_nodes { enum { value = 0 }; };
      template <class LHS, class OP, class RHS> struct num_nodes<       vector_expression<LHS, RHS, OP> > { enum { value = 1 + num_nodes<LHS>::value + num_nodes<RHS>::value }; };
      template <class LHS, class OP, class RHS> struct num_nodes< const vector_expression<LHS, RHS, OP> > { enum { value = 1 + num_nodes<LHS>::value + num_nodes<RHS>::value }; };
      template <class LHS, class OP, class RHS> struct num_nodes<       matrix_expression<LHS, RHS, OP> > { enum { value = 1 + num_nodes<LHS>::value + num_nodes<RHS>::value }; };
      template <class LHS, class OP, class RHS> struct num_nodes< const matrix_expression<LHS, RHS, OP> > { enum { value = 1 + num_nodes<LHS>::value + num_nodes<RHS>::value }; };
      template <class LHS, class OP, class RHS> struct num_nodes<       scalar_expression<LHS, RHS, OP> > { enum { value = 1 + num_nodes<LHS>::value + num_nodes<RHS>::value }; };
      template <class LHS, class OP, class RHS> struct num_nodes< const scalar_expression<LHS, RHS, OP> > { enum { value = 1 + num_nodes<LHS>::value + num_nodes<RHS>::value }; };

    }

    class statement
    {
      public:
        typedef statement_node              value_type;
        typedef viennacl::vcl_size_t        size_type;
        typedef std::vector<value_type>     container_type;

        statement(container_type const & custom_array) : array_(custom_array) {}

        template <typename LHS, typename OP, typename RHS>
        statement(LHS & lhs, OP const &, RHS const & rhs) : array_(1 + result_of::num_nodes<RHS>::value)
        {
          // set OP:
          array_[0].op.type_family = operation_node_type_family(result_of::op_type_info<OP>::family);
          array_[0].op.type        = operation_node_type(result_of::op_type_info<OP>::id);

          // set LHS:
          add_lhs(0, 1, lhs);

          // set RHS:
          add_rhs(0, 1, rhs);
        }

        container_type const & array() const { return array_; }

        size_type root() const { return 0; }

      private:

        ///////////// Scalar node helper ////////////////

        // TODO: add integer vector overloads here
        void assign_element(lhs_rhs_element & elem, viennacl::scalar<float>  const & t) { elem.scalar_float  = const_cast<viennacl::scalar<float> *>(&t); }
        void assign_element(lhs_rhs_element & elem, viennacl::scalar<double> const & t) { elem.scalar_double = const_cast<viennacl::scalar<double> *>(&t); }

        ///////////// Vector node helper ////////////////
        // TODO: add integer vector overloads here
        void assign_element(lhs_rhs_element & elem, viennacl::vector_base<float>  const & t) { elem.vector_float  = const_cast<viennacl::vector_base<float> *>(&t); }
        void assign_element(lhs_rhs_element & elem, viennacl::vector_base<double> const & t) { elem.vector_double = const_cast<viennacl::vector_base<double> *>(&t); }

        ///////////// Matrix node helper ////////////////
        // TODO: add integer matrix overloads here
        void assign_element(lhs_rhs_element & elem, viennacl::matrix_base<float,  viennacl::column_major> const & t) { elem.matrix_col_float  = const_cast<viennacl::matrix_base<float,  viennacl::column_major> *>(&t); }
        void assign_element(lhs_rhs_element & elem, viennacl::matrix_base<float,  viennacl::row_major>    const & t) { elem.matrix_row_float  = const_cast<viennacl::matrix_base<float,  viennacl::row_major>    *>(&t); }
        void assign_element(lhs_rhs_element & elem, viennacl::matrix_base<double, viennacl::column_major> const & t) { elem.matrix_col_double = const_cast<viennacl::matrix_base<double, viennacl::column_major> *>(&t); }
        void assign_element(lhs_rhs_element & elem, viennacl::matrix_base<double, viennacl::row_major>    const & t) { elem.matrix_row_double = const_cast<viennacl::matrix_base<double, viennacl::row_major>    *>(&t); }

        //////////// Tree leaves (terminals) ////////////////////

        std::size_t add_element(std::size_t       next_free,
                                lhs_rhs_element & elem,
                                float const &     t)
        {
          elem.type_family = HOST_SCALAR_TYPE_FAMILY;
          elem.type        = FLOAT_TYPE;
          elem.host_float  = t;
          return next_free;
        }

        std::size_t add_element(std::size_t       next_free,
                                lhs_rhs_element & elem,
                                double const &    t)
        {
          elem.type_family = HOST_SCALAR_TYPE_FAMILY;
          elem.type        = DOUBLE_TYPE;
          elem.host_double = t;
          return next_free;
        }

        template <typename T>
        std::size_t add_element(std::size_t next_free,
                                lhs_rhs_element            & elem,
                                viennacl::scalar<T> const & t)
        {
          elem.type_family = SCALAR_TYPE_FAMILY;
          elem.type        = statement_node_type(result_of::scalar_type<T>::value);
          assign_element(elem, t);
          return next_free;
        }


        template <typename T>
        std::size_t add_element(std::size_t next_free,
                                lhs_rhs_element            & elem,
                                viennacl::vector_base<T> const & t)
        {
          elem.type_family           = VECTOR_TYPE_FAMILY;
          elem.type                  = statement_node_type(result_of::vector_type_for_scalar<T>::value);
          assign_element(elem, t);
          return next_free;
        }

        template <typename T, typename F>
        std::size_t add_element(std::size_t next_free,
                                lhs_rhs_element            & elem,
                                viennacl::matrix_base<T, F> const & t)
        {
          elem.type_family           = statement_node_type_family(result_of::matrix_family<F>::value);
          elem.type                  = statement_node_type(result_of::matrix_type_for_scalar_and_layout<T, F>::value);
          assign_element(elem, t);
          return next_free;
        }


        //////////// Tree nodes (non-terminals) ////////////////////

        template <typename LHS, typename RHS, typename OP>
        std::size_t add_element(std::size_t       next_free,
                                lhs_rhs_element & elem,
                                viennacl::scalar_expression<LHS, RHS, OP> const & t)
        {
          elem.type_family  = COMPOSITE_OPERATION_FAMILY;
          elem.type         = COMPOSITE_OPERATION_TYPE;
          elem.node_index   = next_free;
          return add_node(next_free, next_free + 1, t);
        }

        template <typename LHS, typename RHS, typename OP>
        std::size_t add_element(std::size_t       next_free,
                                lhs_rhs_element & elem,
                                viennacl::vector_expression<LHS, RHS, OP> const & t)
        {
          elem.type_family  = COMPOSITE_OPERATION_FAMILY;
          elem.type         = COMPOSITE_OPERATION_TYPE;
          elem.node_index   = next_free;
          return add_node(next_free, next_free + 1, t);
        }

        template <typename LHS, typename RHS, typename OP>
        std::size_t add_element(std::size_t next_free,
                                lhs_rhs_element & elem,
                                viennacl::matrix_expression<LHS, RHS, OP> const & t)
        {
          elem.type_family   = COMPOSITE_OPERATION_FAMILY;
          elem.type          = COMPOSITE_OPERATION_TYPE;
          elem.node_index    = next_free;
          return add_node(next_free, next_free + 1, t);
        }


        //////////// Helper routines ////////////////////


        template <typename T>
        std::size_t add_lhs(std::size_t current_index, std::size_t next_free, T const & t)
        {
          return add_element(next_free, array_[current_index].lhs, t);
        }

        template <typename T>
        std::size_t add_rhs(std::size_t current_index, std::size_t next_free, T const & t)
        {
          return add_element(next_free, array_[current_index].rhs, t);
        }

        //////////// Internal interfaces ////////////////////

        template <template <typename, typename, typename> class ExpressionT, typename LHS, typename RHS, typename OP>
        std::size_t add_node(std::size_t current_index, std::size_t next_free, ExpressionT<LHS, RHS, OP> const & proxy)
        {
          // set OP:
          array_[current_index].op.type_family = operation_node_type_family(result_of::op_type_info<OP>::family);
          array_[current_index].op.type        = operation_node_type(result_of::op_type_info<OP>::id);

          // set LHS and RHS:
          return add_rhs(current_index, add_lhs(current_index, next_free, proxy.lhs()), proxy.rhs());
        }

        container_type   array_;
    };

  } // namespace scheduler

} // namespace viennacl

#endif

