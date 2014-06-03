#ifndef VIENNACL_SCHEDULER_PRESET_HPP_
#define VIENNACL_SCHEDULER_PRESET_HPP_

#include "viennacl/device_specific/forwards.h"
#include "viennacl/scheduler/forwards.h"

namespace viennacl{

  namespace scheduler{

    namespace preset{

      template<typename T, typename ScalarType1, typename ScalarType2>
      scheduler::statement avbv(scheduler::operation_node_type ASSIGN_OP, vector_base<T> const * x, vector_base<T> const * y, ScalarType1 const * a, bool flip_a, bool reciprocal_a,
                              vector_base<T> const * z, ScalarType2 const * b, bool flip_b, bool reciprocal_b)
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
        if(z){
          statement::add_element(dummy, array[5].lhs, *z);
          array[5].op.type_family = OPERATION_BINARY_TYPE_FAMILY;
          array[5].op.type = reciprocal_b?OPERATION_BINARY_DIV_TYPE:OPERATION_BINARY_MULT_TYPE;
          statement::add_element(dummy, array[5].rhs, *b);
        }

        return statement(array);
      }

      template<typename T>
      device_specific::statements_container plane_rotation(vector_base<T> const * x, vector_base<T> const * y, T const * a, T const * b)
      {
        return device_specific::statements_container(avbv(OPERATION_BINARY_ASSIGN_TYPE, x, x, a, false, false, y, b, false, false),
                                                     avbv(OPERATION_BINARY_ASSIGN_TYPE, y, y, a, false, false, x, b, true, false),
                                                     device_specific::statements_container::INDEPENDENT);
      }

      template<typename T>
      device_specific::statements_container swap(vector_base<T> const * x, vector_base<T> const * y)
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
      scheduler::statement reduction(scalar<T> const * s, vector_base<T> const * x, vector_base<T> const * y,
                                      scheduler::operation_node_type_family LHS_FAMILY, scheduler::operation_node_type LHS_TYPE)
      {
        vcl_size_t dummy = 0;
        statement::container_type array(2);

        statement::add_element(dummy, array[0].lhs, *s);
        array[0].op.type_family = OPERATION_BINARY_TYPE_FAMILY;
        array[0].op.type = OPERATION_BINARY_ASSIGN_TYPE;
        array[0].rhs.type_family = COMPOSITE_OPERATION_FAMILY;
        array[0].rhs.node_index = 1;

        statement::add_element(dummy, array[1].lhs, *x);
        array[1].op.type_family = OPERATION_BINARY_TYPE_FAMILY;
        array[1].op.type = OPERATION_BINARY_INNER_PROD_TYPE;
        statement::add_element(dummy, array[1].rhs, *y);

        return scheduler::statement(array);
      }


    }
  }

}

#endif
