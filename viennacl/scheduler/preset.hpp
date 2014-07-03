#ifndef VIENNACL_SCHEDULER_PRESET_HPP_
#define VIENNACL_SCHEDULER_PRESET_HPP_

#include "viennacl/device_specific/forwards.h"
#include "viennacl/meta/predicate.hpp"
#include "viennacl/scheduler/forwards.h"

namespace viennacl{

  namespace scheduler{

    namespace preset{

      template<typename T, typename ScalarType1, typename ScalarType2>
      scheduler::statement avbv(scheduler::operation_node_type ASSIGN_OP, T const * x, T const * y, ScalarType1 const * a, bool flip_a, bool reciprocal_a,
                              T const * z, ScalarType2 const * b, bool flip_b, bool reciprocal_b)
      {
        statement::container_type array(6);
        vcl_size_t dummy = 0;
        //0
        statement::add_element(dummy, array[0].lhs, *x);
        array[0].op.type_family = OPERATION_BINARY_TYPE_FAMILY;
        array[0].op.type = ASSIGN_OP;
        array[0].rhs.type_family = COMPOSITE_OPERATION_FAMILY;
        if(z)
          array[0].rhs.node_index = 1;
        else
          array[0].rhs.node_index = flip_a?2:3;

        //1
        array[1].lhs.type_family = COMPOSITE_OPERATION_FAMILY;
        array[1].lhs.node_index = flip_a?2:3;
        array[1].op.type_family = OPERATION_BINARY_TYPE_FAMILY;
        array[1].op.type = OPERATION_BINARY_ADD_TYPE;
        array[1].rhs.type_family = COMPOSITE_OPERATION_FAMILY;
        array[1].rhs.node_index = flip_b?4:5;

        //2
        array[2].lhs.type_family = COMPOSITE_OPERATION_FAMILY;
        array[2].lhs.node_index = 3;
        array[2].op.type_family = OPERATION_UNARY_TYPE_FAMILY;
        array[2].op.type = OPERATION_UNARY_MINUS_TYPE;

        //3
        statement::add_element(dummy, array[3].lhs, *y);
        array[3].op.type_family = OPERATION_BINARY_TYPE_FAMILY;
        array[3].op.type = reciprocal_a?OPERATION_BINARY_DIV_TYPE:OPERATION_BINARY_MULT_TYPE;
        statement::add_element(dummy, array[3].rhs, *a);


        //4
        array[4].lhs.type_family = COMPOSITE_OPERATION_FAMILY;
        array[4].lhs.node_index = 5;
        array[4].op.type_family = OPERATION_UNARY_TYPE_FAMILY;
        array[4].op.type = OPERATION_UNARY_MINUS_TYPE;

        //5
        if(z)
        {
          statement::add_element(dummy, array[5].lhs, *z);
          array[5].op.type_family = OPERATION_BINARY_TYPE_FAMILY;
          array[5].op.type = reciprocal_b?OPERATION_BINARY_DIV_TYPE:OPERATION_BINARY_MULT_TYPE;
          statement::add_element(dummy, array[5].rhs, *b);
        }

        return statement(array);
      }

      template<typename T, typename ScalarType1>
      scheduler::statement av(scheduler::operation_node_type ASSIGN_OP, T const * x, T const * y, ScalarType1 const * a, bool flip_a, bool reciprocal_a)
      {
        return scheduler::preset::avbv(ASSIGN_OP, x, y, a, flip_a, reciprocal_a, (T const*)NULL, (ScalarType1 const*)NULL, false, false);
      }


      template<typename T>
      device_specific::statements_container plane_rotation(vector_base<T> const * x, vector_base<T> const * y, T const * a, T const * b)
      {
        return device_specific::statements_container(avbv(OPERATION_BINARY_ASSIGN_TYPE, x, x, a, false, false, y, b, false, false),
                                                     avbv(OPERATION_BINARY_ASSIGN_TYPE, y, y, a, false, false, x, b, true, false),
                                                     device_specific::statements_container::INDEPENDENT);
      }

      template<typename T>
      device_specific::statements_container swap(T const * x, T const * y)
      {
        vcl_size_t dummy = 0;
        statement::container_type array0(1);
        statement::container_type array1(1);

        statement::add_element(dummy, array0[0].lhs, *x);
        array0[0].op.type_family = OPERATION_BINARY_TYPE_FAMILY;
        array0[0].op.type = OPERATION_BINARY_ASSIGN_TYPE;
        statement::add_element(dummy, array0[0].rhs, *y);

        statement::add_element(dummy, array1[0].lhs, *y);
        array1[0].op.type_family = OPERATION_BINARY_TYPE_FAMILY;
        array1[0].op.type = OPERATION_BINARY_ASSIGN_TYPE;
        statement::add_element(dummy, array1[0].rhs, *x);

        return device_specific::statements_container(scheduler::statement(array0), scheduler::statement(array1), device_specific::statements_container::INDEPENDENT);
      }

      template<typename T>
      scheduler::statement assign_cpu(vector_base<T> const * x, implicit_vector_base<T> const * y)
      {
        vcl_size_t dummy = 0;
        statement::container_type array(1);
        statement::add_element(dummy, array[0].lhs, *x);
        array[0].op.type_family = OPERATION_BINARY_TYPE_FAMILY;
        array[0].op.type = OPERATION_BINARY_ASSIGN_TYPE;
        statement::add_element(dummy, array[0].rhs, *y);
        return scheduler::statement(array);
      }

      template<typename T>
      scheduler::statement assign_cpu(matrix_base<T> const * x, implicit_matrix_base<T> const * y)
      {
        vcl_size_t dummy = 0;
        statement::container_type array(1);
        statement::add_element(dummy, array[0].lhs, *x);
        array[0].op.type_family = OPERATION_BINARY_TYPE_FAMILY;
        array[0].op.type = OPERATION_BINARY_ASSIGN_TYPE;
        statement::add_element(dummy, array[0].rhs, *y);
        return scheduler::statement(array);
      }

      template<typename T>
      scheduler::statement diagonal_assign_cpu(matrix_base<T> const * x, implicit_vector_base<T> const * y)
      {
        vcl_size_t dummy = 0;
        statement::container_type array(2);
        array[0].lhs.type_family = COMPOSITE_OPERATION_FAMILY;
        array[0].lhs.node_index = 1;
        array[0].op.type_family = OPERATION_BINARY_TYPE_FAMILY;
        array[0].op.type = OPERATION_BINARY_ASSIGN_TYPE;
        statement::add_element(dummy, array[0].rhs, *y);

        statement::add_element(dummy, array[1].lhs, *x);
        array[1].op.type_family = OPERATION_BINARY_TYPE_FAMILY;
        array[1].op.type = OPERATION_BINARY_MATRIX_DIAG_TYPE;
        statement::add_element(dummy, array[1].rhs, 0);

        return scheduler::statement(array);
      }

      template<typename ScalarType, typename T>
      scheduler::statement reduction_inner_prod(ScalarType const * s, vector_base<T> const * x, vector_base<T> const * y,
                                      scheduler::operation_node_type ROP, bool use_sqrt, bool x_abs)
      {
        vcl_size_t dummy = 0;
        statement::container_type array(5);

        statement::add_element(dummy, array[0].lhs, *s);
        array[0].op.type_family = OPERATION_BINARY_TYPE_FAMILY;
        array[0].op.type = OPERATION_BINARY_ASSIGN_TYPE;
        array[0].rhs.type_family = COMPOSITE_OPERATION_FAMILY;
        array[0].rhs.node_index = use_sqrt?1:2;

        array[1].lhs.type_family = COMPOSITE_OPERATION_FAMILY;
        array[1].lhs.node_index = 2;
        array[1].op.type_family = OPERATION_UNARY_TYPE_FAMILY;
        array[1].op.type = OPERATION_UNARY_SQRT_TYPE;


        if(x_abs)
        {
          array[2].lhs.type_family = COMPOSITE_OPERATION_FAMILY;
          array[2].lhs.node_index = 3;
        }
        else
        {
          statement::add_element(dummy, array[2].lhs, *x);
        }
        if(y)
        {
          array[2].op.type_family = OPERATION_BINARY_TYPE_FAMILY;
          array[2].op.type = OPERATION_BINARY_INNER_PROD_TYPE;
          statement::add_element(dummy, array[2].rhs, *y);
        }
        else
        {
          array[2].op.type_family = OPERATION_VECTOR_REDUCTION_TYPE_FAMILY;
          array[2].op.type = ROP;
        }

        if(is_floating_point<T>::value)
        {
          statement::add_element(dummy, array[3].lhs, *x);
          array[3].op.type_family = OPERATION_UNARY_TYPE_FAMILY;
          array[3].op.type = OPERATION_UNARY_FABS_TYPE;
        }
        else{
          array[3].lhs.type_family = COMPOSITE_OPERATION_FAMILY;
          array[3].lhs.node_index = 4;
          array[3].op.type_family = OPERATION_UNARY_TYPE_FAMILY;
          array[3].op.type = scheduler::operation_node_type(scheduler::result_of::op_type_info<op_element_cast<T> >::id);

          statement::add_element(dummy, array[4].lhs, *x);
          array[4].op.type_family = OPERATION_UNARY_TYPE_FAMILY;
          array[4].op.type = OPERATION_UNARY_ABS_TYPE;
        }


        return scheduler::statement(array);
      }

      template<class ScalarType, class T>
      statement inner_prod(ScalarType const * s, vector_base<T> const * x, vector_base<T> const * y)
      {
        return preset::reduction_inner_prod(s,x,y, OPERATION_INVALID_TYPE, false, false);
      }

      template<class T>
      statement norm_1(scalar<T> const * s, vector_base<T> const * x)
      {
        return preset::reduction_inner_prod(s,x, (vector_base<T>*)NULL, OPERATION_BINARY_ADD_TYPE, false, true);
      }

      template<class T>
      statement norm_2(scalar<T> const * s, vector_base<T> const * x)
      {
        return  preset::reduction_inner_prod(s, x, x, OPERATION_INVALID_TYPE, true, false);
      }

      template<class T>
      statement norm_inf(scalar<T> const * s, vector_base<T> const * x)
      {
        return preset::reduction_inner_prod(s, x, (vector_base<T>*)NULL, is_floating_point<T>::value?OPERATION_BINARY_ELEMENT_FMAX_TYPE:
                                                                                                     OPERATION_BINARY_ELEMENT_MAX_TYPE, false, true);
      }

      template<class T>
      statement index_norm_inf(scalar<T> const * s, vector_base<T> const * x)
      {
        return preset::reduction_inner_prod(s, x, (vector_base<T>*)NULL, is_floating_point<T>::value?OPERATION_BINARY_ELEMENT_ARGFMAX_TYPE:
                                                                                                     OPERATION_BINARY_ELEMENT_ARGMAX_TYPE, false, true);
      }

      template<class T>
      statement sum(scalar<T> const * s, vector_base<T> const * x)
      {
        return preset::reduction_inner_prod(s, x, (vector_base<T>*)NULL, OPERATION_BINARY_ADD_TYPE, false, false);
      }

      template<typename T>
      statement binary_element_op(T const * x, T const * y, T const * z, scheduler::operation_node_type TYPE)
      {
        vcl_size_t dummy = 0;
        statement::container_type array(2);

        statement::add_element(dummy, array[0].lhs, *x);
        array[0].op.type_family = OPERATION_BINARY_TYPE_FAMILY;
        array[0].op.type = OPERATION_BINARY_ASSIGN_TYPE;
        array[0].rhs.type_family = COMPOSITE_OPERATION_FAMILY;
        array[0].rhs.node_index = 1;

        statement::add_element(dummy, array[1].lhs, *y);
        array[1].op.type_family = z?OPERATION_BINARY_TYPE_FAMILY:OPERATION_UNARY_TYPE_FAMILY;
        array[1].op.type = TYPE;
        if(z)
          statement::add_element(dummy, array[1].rhs, *z);

        return statement(array);
      }

      template<typename T>
      statement unary_element_op(T const * x, T const * y, scheduler::operation_node_type TYPE)
      {
        return binary_element_op(x, y, static_cast<T const *>(NULL), TYPE);
      }

      template<typename T>
      statement matrix_row_column_diag(viennacl::vector_base<T> const * x, viennacl::matrix_base<T> const * A, int id, int op)
      {
        vcl_size_t dummy = 0;
        statement::container_type array(2);

        if(op==3)
          statement::add_element(dummy, array[0].lhs, *A);
        else
          statement::add_element(dummy, array[0].lhs, *x);
        array[0].op.type_family = OPERATION_BINARY_TYPE_FAMILY;
        array[0].op.type = OPERATION_BINARY_ASSIGN_TYPE;
        array[0].rhs.type_family = COMPOSITE_OPERATION_FAMILY;
        array[0].rhs.node_index = 1;

        if(op==3)
          statement::add_element(dummy, array[1].lhs, *x);
        else
          statement::add_element(dummy, array[1].lhs, *A);
        array[1].op.type_family = OPERATION_BINARY_TYPE_FAMILY;
        if(op==0)
         array[1].op.type = OPERATION_BINARY_MATRIX_ROW_TYPE;
        else if(op==1)
          array[1].op.type = OPERATION_BINARY_MATRIX_COLUMN_TYPE;
        else if(op==2)
          array[1].op.type = OPERATION_BINARY_MATRIX_DIAG_TYPE;
        else
          array[1].op.type = OPERATION_BINARY_VECTOR_DIAG_TYPE;
        statement::add_element(dummy, array[1].rhs, id);

        return statement(array);
      }

      template<typename T>
      statement matrix_row(viennacl::vector_base<T> const * x, viennacl::matrix_base<T> const * A, int id)
      {
        return matrix_row_column_diag(x, A, id, 0);
      }

      template<typename T>
      statement matrix_column(viennacl::vector_base<T> const * x, viennacl::matrix_base<T> const * A, int id)
      {
        return matrix_row_column_diag(x, A, id, 1);
      }


      template<typename T>
      statement matrix_diag_to_vector(viennacl::vector_base<T> const * x, viennacl::matrix_base<T> const * A, int id)
      {
        return matrix_row_column_diag(x, A, id, 2);
      }

      template<typename T>
      statement matrix_diag_from_vector(viennacl::vector_base<T> const * x, viennacl::matrix_base<T> const * A, int id)
      {
        return matrix_row_column_diag(x, A, id, 3);
      }

    }
  }

}

#endif
