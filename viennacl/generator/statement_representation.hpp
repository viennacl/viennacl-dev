#ifndef VIENNACL_GENERATOR_STATEMENT_REPRESENTATION_HPP
#define VIENNACL_GENERATOR_STATEMENT_REPRESENTATION_HPP

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


/** @file viennacl/generator/statement_representation.hpp
    @brief Functor to enqueue the leaves of an expression tree
*/

#include <set>
#include <cstring>

#include "viennacl/forwards.h"
#include "viennacl/scheduler/forwards.h"
#include "viennacl/generator/forwards.h"

#include "viennacl/tools/shared_ptr.hpp"

#include "viennacl/ocl/kernel.hpp"

#include "viennacl/generator/generate_utils.hpp"
#include "viennacl/generator/utils.hpp"
#include "viennacl/generator/mapped_types.hpp"

namespace viennacl{

  namespace generator{

    namespace detail{

      class representation_functor{
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

          representation_functor(void* (&memory)[64], unsigned int & current_arg, char *& ptr) : memory_(memory), current_arg_(current_arg), ptr_(ptr){ }

          template<class ScalarType>
          result_type operator()(ScalarType const & scal) const {
            *ptr_++='h'; //host
            *ptr_++='s'; //scalar
            *ptr_++=utils::first_letter_of_type<ScalarType>::value();
            append_id(ptr_, get_id((void*)&scal));
          }

          //Scalar mapping
          template<class ScalarType>
          result_type operator()(scalar<ScalarType> const & scal) const {
            *ptr_++='s'; //scalar
            *ptr_++=utils::first_letter_of_type<ScalarType>::value();
            append_id(ptr_, get_id((void*)&scal));
          }

          //Vector mapping
          template<class ScalarType>
          result_type operator()(vector_base<ScalarType> const & vec) const {
            *ptr_++='v'; //vector
            if(vec.start()>0)
              *ptr_++='r';
            if(vec.stride()>1)
              *ptr_++='s';
            *ptr_++=utils::first_letter_of_type<ScalarType>::value();
            append_id(ptr_, get_id((void*)&vec));
          }

          //Symbolic vector mapping
          template<class ScalarType>
          result_type operator()(symbolic_vector_base<ScalarType> const & vec) const {
            *ptr_++='s'; //symbolic
            *ptr_++='v'; //vector
            *ptr_++=utils::first_letter_of_type<ScalarType>::value();
          }

          //Matrix mapping
          template<class ScalarType, class Layout>
          result_type operator()(matrix_base<ScalarType, Layout> const & mat) const {
            *ptr_++='m'; //vector
            if(mat.start1()>0)
              *ptr_++='r';
            if(mat.stride1()>1)
              *ptr_++='s';
            if(mat.start2()>0)
              *ptr_++='r';
            if(mat.stride2()>1)
              *ptr_++='s';
            *ptr_++=utils::first_letter_of_type<ScalarType>::value();
            *ptr_++=utils::first_letter_of_type<Layout>::value();
            append_id(ptr_, get_id((void*)&mat));
          }

          //Symbolic matrix mapping
          template<class ScalarType>
          result_type operator()(symbolic_matrix_base<ScalarType> const & mat) const {
            *ptr_++='s'; //symbolic
            *ptr_++='m'; //matrix
            *ptr_++=utils::first_letter_of_type<ScalarType>::value();
          }

        private:
          void* (&memory_)[64];
          unsigned int & current_arg_;
          char *& ptr_;
      };

      static void statement_representation(scheduler::statement const & statement, void* (&memory)[64], unsigned int & current_arg, char*& ptr){
          scheduler::statement::container_type expr = statement.array();
          for(std::size_t i = 0 ; i < expr.size() ; ++i){
            scheduler::statement_node node = expr[i];
            if(node.lhs.type_family!=COMPOSITE_OPERATION_FAMILY){
              utils::call_on_element(node.lhs.type_family, node.lhs.type, node.lhs, representation_functor(memory, current_arg, ptr));
            }

            {
              const char * op_expr = detail::generate(node.op.type);
              std::size_t n = std::strlen(op_expr);
              std::memcpy(ptr, op_expr, n);
              ptr+=n;
            }

            if(node.rhs.type_family!=COMPOSITE_OPERATION_FAMILY){
              utils::call_on_element(node.rhs.type_family, node.rhs.type, node.rhs, representation_functor(memory, current_arg, ptr));
            }

            *ptr++='_';
          }
      }

    }

  }

}
#endif
