/* =========================================================================
   Copyright (c) 2010-2015, Institute for Microelectronics,
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

/** \example scheduler.cpp
*
*   This tutorial show how to use the low-level scheduler to generate efficient custom OpenCL kernels at run time.
*   The purpose of the scheduler is to provide a low-level interface for interfacing ViennaCL from languages other than C++,
*   yet providing the user the ability to specify complex operations.
*   Typical consumers are scripting languages such as Python (e.g. PyViennaCL), but the facility should be used in the future to also fuse compute kernels on the fly.
*
*   \warning The scheduler is experimental and only intended for expert users.
*
*   We start this tutorial with including the necessary headers:
**/

// include necessary system headers
#include <iostream>

// include basic scalar and vector types of ViennaCL
#include "viennacl/scalar.hpp"
#include "viennacl/vector.hpp"
#include "viennacl/matrix.hpp"

#include "viennacl/scheduler/execute.hpp"
#include "viennacl/scheduler/io.hpp"


/**
*  This tutorial sets up three vectors and finally assigns the sum of two to the third.
*  Although this can be achieved with only a few lines of code using the standard ViennaCL C++ API,
*  we go through the low-level interface for demonstration purposes.
**/
int main()
{
  typedef float       ScalarType;  // do not change without adjusting the code for the low-level interface below

  /**
  * Create three vectors, initialize two of them with ascending/descending integers:
  **/
  viennacl::vector<ScalarType> vcl_vec1(10);
  viennacl::vector<ScalarType> vcl_vec2(10);
  viennacl::vector<ScalarType> vcl_vec3(10);

  for (unsigned int i = 0; i < 10; ++i)
  {
    vcl_vec1[i] = ScalarType(i);
    vcl_vec2[i] = ScalarType(10 - i);
  }

  /**
  * Build expression graph for the operation vcl_vec3 = vcl_vec1 + vcl_vec2
  *
  * This requires the following expression graph:
  * \code
  *             ( = )
  *            /      |
  *    vcl_vec3      ( + )
  *                 /     |
  *           vcl_vec1    vcl_vec2
  * \endcode
  * One expression node consists of two leaves and the operation connecting the two.
  * Here we thus need two nodes: One for {vcl_vec3, = , link}, where 'link' points to the second node
  * {vcl_vec1, +, vcl_vec2}.
  *
  * The following is the lowest level on which one could build the expression tree.
  * Even for a C API one would introduce some additional convenience layer such as add_vector_float_to_lhs(...); etc.
  **/
  typedef viennacl::scheduler::statement::container_type   NodeContainerType;   // this is just std::vector<viennacl::scheduler::statement_node>
  NodeContainerType expression_nodes(2);                                        //container with two nodes

  /**
  * <h2>First Node (Assignment)</h2>
  **/

  // specify LHS of first node, i.e. vcl_vec3:
  expression_nodes[0].lhs.type_family  = viennacl::scheduler::VECTOR_TYPE_FAMILY;   // family of vectors
  expression_nodes[0].lhs.subtype      = viennacl::scheduler::DENSE_VECTOR_TYPE;    // a dense vector
  expression_nodes[0].lhs.numeric_type = viennacl::scheduler::FLOAT_TYPE;           // vector consisting of floats
  expression_nodes[0].lhs.vector_float = &vcl_vec3;                                 // provide pointer to vcl_vec3;

  // specify assignment operation for this node:
  expression_nodes[0].op.type_family   = viennacl::scheduler::OPERATION_BINARY_TYPE_FAMILY; // this is a binary operation, so both LHS and RHS operands are important
  expression_nodes[0].op.type          = viennacl::scheduler::OPERATION_BINARY_ASSIGN_TYPE; // assignment operation: '='

  // specify RHS: Just refer to the second node:
  expression_nodes[0].rhs.type_family  = viennacl::scheduler::COMPOSITE_OPERATION_FAMILY; // this links to another node (no need to set .subtype and .numeric_type)
  expression_nodes[0].rhs.node_index   = 1;                                               // index of the other node

  /**
  * <h2>Second Node (Addition)</h2>
  **/

  // LHS
  expression_nodes[1].lhs.type_family  = viennacl::scheduler::VECTOR_TYPE_FAMILY;   // family of vectors
  expression_nodes[1].lhs.subtype      = viennacl::scheduler::DENSE_VECTOR_TYPE;    // a dense vector
  expression_nodes[1].lhs.numeric_type = viennacl::scheduler::FLOAT_TYPE;           // vector consisting of floats
  expression_nodes[1].lhs.vector_float = &vcl_vec1;                                 // provide pointer to vcl_vec1

  // OP
  expression_nodes[1].op.type_family   = viennacl::scheduler::OPERATION_BINARY_TYPE_FAMILY; // this is a binary operation, so both LHS and RHS operands are important
  expression_nodes[1].op.type          = viennacl::scheduler::OPERATION_BINARY_ADD_TYPE;    // addition operation: '+'

  // RHS
  expression_nodes[1].rhs.type_family  = viennacl::scheduler::VECTOR_TYPE_FAMILY;  // family of vectors
  expression_nodes[1].rhs.subtype      = viennacl::scheduler::DENSE_VECTOR_TYPE;   // a dense vector
  expression_nodes[1].rhs.numeric_type = viennacl::scheduler::FLOAT_TYPE;          // vector consisting of floats
  expression_nodes[1].rhs.vector_float = &vcl_vec2;                                // provide pointer to vcl_vec2


  /**
  *  Create the full statement (aka. single line of code such as vcl_vec3 = vcl_vec1 + vcl_vec2):
  **/
  viennacl::scheduler::statement vec_addition(expression_nodes);

  /**
  *  Print the expression. Resembles the tree outlined in comments above.
  **/
  std::cout << vec_addition << std::endl;

  /**
  *  Execute the operation
  **/
  viennacl::scheduler::execute(vec_addition);

  /**
  *  Print vectors in order to check the result:
  **/
  std::cout << "vcl_vec1: " << vcl_vec1 << std::endl;
  std::cout << "vcl_vec2: " << vcl_vec2 << std::endl;
  std::cout << "vcl_vec3: " << vcl_vec3 << std::endl;

  /**
  *   That's it! Print success message and exit.
  **/
  std::cout << "!!!! TUTORIAL COMPLETED SUCCESSFULLY !!!!" << std::endl;

  return EXIT_SUCCESS;
}

