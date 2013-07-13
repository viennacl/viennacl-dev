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

/*
*
*   Tutorial: Show how to use the scheduler
*
*/


// include necessary system headers
#include <iostream>

//include basic scalar and vector types of ViennaCL
#include "viennacl/scalar.hpp"
#include "viennacl/vector.hpp"
#include "viennacl/matrix.hpp"

#include "viennacl/scheduler/execute.hpp"

int main()
{
  typedef float       ScalarType;

  viennacl::vector<ScalarType> vcl_vec1(10);
  viennacl::vector<ScalarType> vcl_vec2(10);
  viennacl::vector<ScalarType> vcl_vec3(10);

  //
  // Let us fill the CPU vectors with random values:
  // (random<> is a helper function from Random.hpp)
  //

  for (unsigned int i = 0; i < 10; ++i)
  {
    vcl_vec1[i] = ScalarType(i);
    vcl_vec2[i] = ScalarType(10 - i);
  }

  //
  // Build expression graph for the operation vcl_vec3 = vcl_vec1 + vcl_vec2
  //
  // This requires the following expression graph:
  //
  //             ( = )
  //            /     \
  //    vcl_vec3      ( + )
  //                 /     \
  //           vcl_vec1    vcl_vec2
  //
  // One expression node consists of two leaves and the operation connecting the two.
  // Here we thus need two nodes: One for {vcl_vec3, = , link}, where 'link' points to the second node
  // {vcl_vec1, +, vcl_vec2}.
  //
  // The following is the lowest level on which one could build the expression tree.
  // Even for a C API one would introduce some additional convenience layer such as add_vector_float_to_lhs(...); etc.
  //
  typedef viennacl::scheduler::statement::container_type   NodeContainerType;   // this is just std::vector<viennacl::scheduler::statement_node>
  NodeContainerType expression_nodes(2);                                        //container with two nodes

  ////// First node //////

  // specify LHS of first node, i.e. vcl_vec3:
  expression_nodes[0].lhs_type_family_   = viennacl::scheduler::VECTOR_TYPE_FAMILY;   // family of vectors
  expression_nodes[0].lhs_type_          = viennacl::scheduler::VECTOR_FLOAT_TYPE;    // vector consisting of floats
  expression_nodes[0].lhs_.vector_float_ = &vcl_vec3;                                 // provide pointer to vcl_vec3;

  // specify assignment operation for this node:
  expression_nodes[0].op_family_         = viennacl::scheduler::OPERATION_BINARY_TYPE_FAMILY; // this is a binary operation, so both LHS and RHS operands are important
  expression_nodes[0].op_type_           = viennacl::scheduler::OPERATION_BINARY_ASSIGN_TYPE; // assignment operation: '='

  // specify RHS: Just refer to the second node:
  expression_nodes[0].rhs_type_family_   = viennacl::scheduler::COMPOSITE_OPERATION_FAMILY; // this links to another node
  expression_nodes[0].rhs_type_          = viennacl::scheduler::COMPOSITE_OPERATION_TYPE;   // this links to another node
  expression_nodes[0].rhs_.node_index_   = 1;                                               // index of the other node

  ////// Second node //////

  // LHS
  expression_nodes[1].lhs_type_family_   = viennacl::scheduler::VECTOR_TYPE_FAMILY;   // family of vectors
  expression_nodes[1].lhs_type_          = viennacl::scheduler::VECTOR_FLOAT_TYPE;    // vector consisting of floats
  expression_nodes[1].lhs_.vector_float_ = &vcl_vec1;                                 // provide pointer to vcl_vec1

  // OP
  expression_nodes[1].op_family_         = viennacl::scheduler::OPERATION_BINARY_TYPE_FAMILY; // this is a binary operation, so both LHS and RHS operands are important
  expression_nodes[1].op_type_           = viennacl::scheduler::OPERATION_BINARY_ADD_TYPE;    // assignment operation: '='

  // RHS
  expression_nodes[1].rhs_type_family_   = viennacl::scheduler::VECTOR_TYPE_FAMILY;  // family of vectors
  expression_nodes[1].rhs_type_          = viennacl::scheduler::VECTOR_FLOAT_TYPE;   // vector consisting of floats
  expression_nodes[1].rhs_.vector_float_ = &vcl_vec2;                                // provide pointer to vcl_vec2


  // create the full statement (aka. single line of code such as vcl_vec3 = vcl_vec1 + vcl_vec2):
  viennacl::scheduler::statement vec_addition(expression_nodes);

  // run it
  viennacl::scheduler::execute(vec_addition);

  // print vectors
  std::cout << "vcl_vec1: " << vcl_vec1 << std::endl;
  std::cout << "vcl_vec2: " << vcl_vec2 << std::endl;
  std::cout << "vcl_vec3: " << vcl_vec3 << std::endl;


  std::cout << "!!!! TUTORIAL COMPLETED SUCCESSFULLY !!!!" << std::endl;

  return EXIT_SUCCESS;
}

