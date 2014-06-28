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

#include "viennacl/tools/shared_ptr.hpp"

#include "viennacl/ocl/kernel.hpp"

#include "viennacl/device_specific/tree_parsing/traverse.hpp"
#include "viennacl/device_specific/utils.hpp"
#include "viennacl/device_specific/mapped_objects.hpp"

namespace viennacl{

  namespace device_specific{

    namespace tree_parsing{

      class statement_representation_functor : public traversal_functor{
        private:
          unsigned int get_id(void * handle) const{
            unsigned int i = 0;
            for( ; i < 64 ; ++i){
              void* current = memory_[i];
              if(current==NULL)
                break;
              if(current==handle)
                return i;
            }
            memory_[i] = handle;
            return i;
          }

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

          statement_representation_functor(void* (&memory)[64], unsigned int & current_arg, char *& ptr) : memory_(memory), current_arg_(current_arg), ptr_(ptr){ }

          template<class ScalarType>
          result_type operator()(ScalarType const & /*scal*/) const {
            *ptr_++='h'; //host
            *ptr_++='s'; //scalar
            *ptr_++=utils::first_letter_of_type<ScalarType>::value();
          }

          /** @brief Scalar mapping */
          template<class ScalarType>
          result_type operator()(scalar<ScalarType> const & scal) const {
            *ptr_++='s'; //scalar
            *ptr_++=utils::first_letter_of_type<ScalarType>::value();
            append_id(ptr_, get_id((void*)&scal));
          }

          /** @brief Vector mapping */
          template<class ScalarType>
          result_type operator()(vector_base<ScalarType> const & vec) const {
            *ptr_++='v'; //vector
            if(viennacl::traits::start(vec)>0)
              *ptr_++='r';
            if(vec.stride()>1)
              *ptr_++='s';
            *ptr_++=utils::first_letter_of_type<ScalarType>::value();
            append_id(ptr_, get_id((void*)&vec));
          }

          /** @brief Implicit vector mapping */
          template<class ScalarType>
          result_type operator()(implicit_vector_base<ScalarType> const & /*vec*/) const {
            *ptr_++='i'; //implicit
            *ptr_++='v'; //vector
            *ptr_++=utils::first_letter_of_type<ScalarType>::value();
          }

          /** @brief Matrix mapping */
          template<class ScalarType>
          result_type operator()(matrix_base<ScalarType> const & mat) const {
            *ptr_++='m'; //Matrix
            *ptr_++=utils::first_letter_of_type<ScalarType>::value();
            if(mat.row_major())
              *ptr_++='r';
            else
              *ptr_++='c';
            append_id(ptr_, get_id((void*)&mat));
          }

          /** @brief Implicit matrix mapping */
          template<class ScalarType>
          result_type operator()(implicit_matrix_base<ScalarType> const & /*mat*/) const {
            *ptr_++='i'; //implicit
            *ptr_++='m'; //matrix
            *ptr_++=utils::first_letter_of_type<ScalarType>::value();
          }

          static void append(char*& p, const char * str){
            std::size_t n = std::strlen(str);
            std::memcpy(p, str, n);
            p+=n;
          }

          void operator()(scheduler::statement const *, scheduler::statement_node const * root_node, node_type node_type) const {
            if(node_type==LHS_NODE_TYPE && root_node->lhs.type_family != scheduler::COMPOSITE_OPERATION_FAMILY)
              utils::call_on_element(root_node->lhs, *this);
            else if(root_node->op.type_family==scheduler::OPERATION_BINARY_TYPE_FAMILY && node_type==RHS_NODE_TYPE && root_node->rhs.type_family != scheduler::COMPOSITE_OPERATION_FAMILY)
              utils::call_on_element(root_node->rhs, *this);
            else if(node_type==PARENT_NODE_TYPE){
              if(root_node->op.type_family==scheduler::OPERATION_VECTOR_REDUCTION_TYPE_FAMILY)
                append(ptr_,"vecred");
              if(root_node->op.type_family==scheduler::OPERATION_ROWS_REDUCTION_TYPE_FAMILY)
                append(ptr_,"rowred");
              if(root_node->op.type_family==scheduler::OPERATION_COLUMNS_REDUCTION_TYPE_FAMILY)
                append(ptr_,"colred");
              append(ptr_,generate(root_node->op.type));
            }
          }

        private:
          void* (&memory_)[64];
          unsigned int & current_arg_;
          char *& ptr_;
      };

    }

  }

}
#endif
