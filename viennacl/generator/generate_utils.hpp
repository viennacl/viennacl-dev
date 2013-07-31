#ifndef VIENNACL_GENERATOR_GENERATE_UTILS_HPP
#define VIENNACL_GENERATOR_GENERATE_UTILS_HPP

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


/** @file viennacl/generator/generate_utils.hpp
    @brief Internal upper layer to the scheduler
*/

#include <set>

#include "CL/cl.h"

#include "viennacl/forwards.h"
#include "viennacl/scheduler/forwards.h"

#include "viennacl/generator/utils.hpp"
#include "viennacl/generator/forwards.h"

namespace viennacl{

  namespace generator{

    namespace detail{

      static std::string generate_value_kernel_argument(std::string const & scalartype, std::string const & name){
        return scalartype + ' ' + name + ",";
      }

      static std::string generate_pointer_kernel_argument(std::string const & address_space, std::string const & scalartype, std::string const & name){
        return address_space +  " " + scalartype + "* " + name + ",";
      }

      static const char * generate(operation_node_type type){
        // unary expression
        switch(type){
          case OPERATION_UNARY_ABS_TYPE : return "abs";
          case OPERATION_UNARY_TRANS_TYPE : return "trans";
          case OPERATION_BINARY_ASSIGN_TYPE : return "=";
          case OPERATION_BINARY_ADD_TYPE : return "+";
          case OPERATION_BINARY_INNER_PROD_TYPE : return "iprod";
          case OPERATION_BINARY_MAT_MAT_PROD_TYPE : return "prod";
          case OPERATION_BINARY_MAT_VEC_PROD_TYPE : return "prod";
          case OPERATION_BINARY_ACCESS_TYPE : return "[]";
          default : throw "not implemented";
        }
      }

      class traversal_functor{
        public:
          void call_on_op(operation_node_type_family, operation_node_type) const { }
          void call_before_expansion() const { }
          void call_after_expansion() const { }
      };

      class expression_generation_traversal : public traversal_functor{
        private:
          std::pair<std::string, std::string> index_string_;
          int vector_element_;
          std::string & str_;
          mapping_type const & mapping_;
        public:
          expression_generation_traversal(std::pair<std::string, std::string> const & index, int vector_element, std::string & str, mapping_type const & mapping) : index_string_(index), vector_element_(vector_element), str_(str), mapping_(mapping){ }
          void call_on_leaf(index_info const & key, statement_node const &,  statement::container_type const *) const { str_ += generate(index_string_, vector_element_, *mapping_.at(key)); }
          void call_on_op(operation_node_type_family, operation_node_type type) const {
            if(type!=scheduler::OPERATION_UNARY_TRANS_TYPE)
              str_ += detail::generate(type);
          }
          void call_before_expansion() const { str_ += '('; }
          void call_after_expansion() const { str_ += ')';  }
      };

      class fetch_traversal : public traversal_functor{
        private:
          std::set<std::string> & fetched_;
          std::pair<std::string, std::string> index_string_;
          unsigned int vectorization_;
          utils::kernel_generation_stream & stream_;
          mapping_type const & mapping_;
        public:
          fetch_traversal(std::set<std::string> & fetched, std::pair<std::string, std::string> const & index, unsigned int vectorization, utils::kernel_generation_stream & stream, mapping_type const & mapping) : fetched_(fetched), index_string_(index), vectorization_(vectorization), stream_(stream), mapping_(mapping){ }
          void call_on_leaf(index_info const & key, statement_node const &,  statement::container_type const *) const {
            fetch(index_string_, vectorization_, fetched_, stream_, *mapping_.at(key));
          }
      };

      static index_info get_new_key(statement_node_type_family type_family, std::size_t current_index, std::size_t next_index, node_type node_tag){
        if(type_family==COMPOSITE_OPERATION_FAMILY)
          return std::make_pair(next_index, PARENT_TYPE);
        else
          return std::make_pair(current_index, node_tag);
      }


      template<class TraversalFunctor>
      static void traverse(statement::container_type const & array, TraversalFunctor const & fun, bool deep_traversal, index_info const & key){
        std::size_t index = key.first;
        std::size_t node_tag = key.second;
        statement::value_type const & element = array[index];
        operation_node_type op_type = element.op.type;
        operation_node_type_family op_family = element.op.type_family;
        if(node_tag == PARENT_TYPE){
          if(op_family==OPERATION_UNARY_TYPE_FAMILY){
            fun.call_on_op(op_family, op_type);
            fun.call_before_expansion();
            traverse(array, fun, deep_traversal, get_new_key(element.lhs.type_family, index, element.lhs.node_index, LHS_NODE_TYPE));
            fun.call_after_expansion();
          }
          if(op_family==OPERATION_BINARY_TYPE_FAMILY){
            if(op_type==OPERATION_BINARY_ACCESS_TYPE){
              fun.call_on_leaf(key, element, &array);
              if(deep_traversal)
                traverse(array, fun, deep_traversal, get_new_key(element.rhs.type_family, index, element.rhs.node_index, RHS_NODE_TYPE));
            }
            else{
              bool is_binary_leaf = (op_type==OPERATION_BINARY_INNER_PROD_TYPE)
                                  ||(op_type==OPERATION_BINARY_MAT_VEC_PROD_TYPE)
                                  ||(op_type==OPERATION_BINARY_MAT_MAT_PROD_TYPE);
              bool recurse = !is_binary_leaf || (is_binary_leaf && deep_traversal);
              if(is_binary_leaf)
                fun.call_on_leaf(key, element, &array);
              if(recurse){
                fun.call_before_expansion();
                traverse(array, fun, deep_traversal, get_new_key(element.lhs.type_family, index, element.lhs.node_index, LHS_NODE_TYPE));
                fun.call_on_op(op_family, op_type);
                traverse(array, fun, deep_traversal, get_new_key(element.rhs.type_family, index, element.rhs.node_index, RHS_NODE_TYPE));
                fun.call_after_expansion();
              }
            }
          }
        }
        else
          fun.call_on_leaf(key, element, &array);
      }

    }

  }

}
#endif
