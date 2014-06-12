#ifndef VIENNACL_DEVICE_SPECIFIC_TREE_PARSING_STATEMENT_REPRESENTATION_HPP
#define VIENNACL_DEVICE_SPECIFIC_TREE_PARSING_STATEMENT_REPRESENTATION_HPP

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


/** @file viennacl/generator/statement_representation_functor.hpp
    @brief Functor to generate the string id of a statement
*/

#include <set>
#include <cstring>

#include "viennacl/forwards.h"

#include "viennacl/scheduler/forwards.h"

#include "viennacl/device_specific/forwards.h"
#include "viennacl/device_specific/tree_parsing/traverse.hpp"
#include "viennacl/device_specific/utils.hpp"
#include "viennacl/device_specific/mapped_objects.hpp"


namespace viennacl{

  namespace device_specific{

    namespace tree_parsing{

      class statement_representation_functor : public traversal_functor{
        private:
          static void append_id(char * & ptr, unsigned int val){
            if(val==0)
              *ptr++='0';
            else
              while(val>0)
              {
                  *ptr++='0' + (val % 10);
                  val /= 10;
              }
          }

        public:
          typedef void result_type;

          statement_representation_functor(symbolic_binder & binder, char *& ptr) : binder_(binder), ptr_(ptr){ }

          template<class ScalarType>
          inline result_type operator()(ScalarType const & /*scal*/) const {
            *ptr_++='h'; //host
            *ptr_++='s'; //scalar
            *ptr_++=utils::first_letter_of_type<ScalarType>::value();
          }

          /** @brief Scalar mapping */
          template<class ScalarType>
          inline result_type operator()(scalar<ScalarType> const & scal) const {
            *ptr_++='s'; //scalar
            *ptr_++=utils::first_letter_of_type<ScalarType>::value();
            append_id(ptr_, binder_.get(&traits::handle(scal)));
          }

          /** @brief Vector mapping */
          template<class ScalarType>
          inline result_type operator()(vector_base<ScalarType> const & vec) const {
            *ptr_++='v'; //vector
            *ptr_++=utils::first_letter_of_type<ScalarType>::value();
            append_id(ptr_, binder_.get(&traits::handle(vec)));
          }

          /** @brief Implicit vector mapping */
          template<class ScalarType>
          inline result_type operator()(implicit_vector_base<ScalarType> const & vec) const {
            *ptr_++='i'; //implicit
            *ptr_++='v'; //vector
            if(vec.has_index())
              *ptr_++='i'; //index
            *ptr_++=utils::first_letter_of_type<ScalarType>::value();
          }

          /** @brief Matrix mapping */
          template<class ScalarType>
          inline result_type operator()(matrix_base<ScalarType> const & mat) const {
            *ptr_++='m'; //Matrix
            *ptr_++=utils::first_letter_of_type<ScalarType>::value();
            append_id(ptr_, binder_.get(&traits::handle(mat)));
          }

          /** @brief Implicit matrix mapping */
          template<class ScalarType>
          inline result_type operator()(implicit_matrix_base<ScalarType> const & mat) const {
            *ptr_++='i'; //implicit
            *ptr_++='m'; //matrix
            *ptr_++=utils::first_letter_of_type<ScalarType>::value();
          }

          static inline void append(char*& p, const char * str){
            std::size_t n = std::strlen(str);
            std::memcpy(p, str, n);
            p+=n;
          }

          inline void operator()(scheduler::statement const & statement, unsigned int root_idx, node_type node_type) const {
            scheduler::statement_node const & root_node = statement.array()[root_idx];
            if(node_type==LHS_NODE_TYPE && root_node.lhs.type_family != scheduler::COMPOSITE_OPERATION_FAMILY)
              utils::call_on_element(root_node.lhs, *this);
            else if(root_node.op.type_family==scheduler::OPERATION_BINARY_TYPE_FAMILY && node_type==RHS_NODE_TYPE && root_node.rhs.type_family != scheduler::COMPOSITE_OPERATION_FAMILY)
              utils::call_on_element(root_node.rhs, *this);
            else if(node_type==PARENT_NODE_TYPE){
              if(root_node.op.type_family==scheduler::OPERATION_VECTOR_REDUCTION_TYPE_FAMILY)
                append(ptr_,"vecred");
              if(root_node.op.type_family==scheduler::OPERATION_ROWS_REDUCTION_TYPE_FAMILY)
                append(ptr_,"rowred");
              if(root_node.op.type_family==scheduler::OPERATION_COLUMNS_REDUCTION_TYPE_FAMILY)
                append(ptr_,"colred");
              append(ptr_,evaluate_str(root_node.op.type));
            }
          }

        private:
          symbolic_binder & binder_;
          char *& ptr_;
      };

      inline std::string statements_representation(statements_container const & statements, binding_policy_t binding_policy)
      {
          std::vector<char> program_name_vector(256);
          char* program_name = program_name_vector.data();
          if(statements.order()==statements_container::INDEPENDENT)
            *program_name++='i';
          else
            *program_name++='s';
          tools::shared_ptr<symbolic_binder> binder = make_binder(binding_policy);
          for(statements_container::data_type::const_iterator it = statements.data().begin() ; it != statements.data().end() ; ++it)
              tree_parsing::traverse(*it, it->root(), tree_parsing::statement_representation_functor(*binder, program_name),true);
          *program_name='\0';
          return std::string(program_name_vector.data());
      }

    }

  }

}
#endif
