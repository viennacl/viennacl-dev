#ifndef VIENNACL_GENERATOR_MAP_GENERATE_PROTOTYPE_HPP
#define VIENNACL_GENERATOR_MAP_GENERATE_PROTOTYPE_HPP

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


/** @file viennacl/generator/map_generate_prototype.hpp
    @brief Functor to map a statement and generate the prototype
*/

#include <set>

#include "viennacl/forwards.h"
#include "viennacl/scheduler/forwards.h"
#include "viennacl/generator/forwards.h"

#include "viennacl/tools/shared_ptr.hpp"

#include "viennacl/generator/generate_utils.hpp"
#include "viennacl/generator/utils.hpp"
#include "viennacl/generator/mapped_types.hpp"

namespace viennacl{

  namespace generator{

    namespace detail{

      class map_functor{
          std::string create_name(unsigned int & current_arg, std::map<void *, std::size_t> & memory, void * handle) const{
            if(handle==NULL)
              return "arg" + utils::to_string(current_arg_++);
            if(memory.insert(std::make_pair(handle, current_arg)).second)
              return "arg" + utils::to_string(current_arg_++);
            else
              return "arg" + utils::to_string(memory.at(handle));
          }

        public:
          typedef mapped_container * result_type;

          map_functor(std::map<void *, std::size_t> & memory, unsigned int & current_arg) : memory_(memory), current_arg_(current_arg){ }

          //Binary leaf
          template<class T>
          result_type binary_leaf(unsigned int i, statement_node const & node,  statement::container_type const * array, mapping_type const * mapping){
            T * p = new T("float");

            p->lhs_.array = array;
            p->lhs_.index = get_new_key(node.lhs.type_family, i, node.lhs.node_index, LHS_NODE_TYPE);
            p->lhs_.mapping = mapping;

            p->rhs_.array = array;
            p->rhs_.index = get_new_key(node.rhs.type_family, i, node.rhs.node_index, RHS_NODE_TYPE);
            p->rhs_.mapping = mapping;

            return p;
          }

          template<class ScalarType>
          result_type operator()(ScalarType const & scal) const {
            mapped_host_scalar * p = new mapped_host_scalar(utils::type_to_string<ScalarType>::value());
            p->name_ = create_name(current_arg_, memory_, (void*)&scal);
            return p;
          }

          //Scalar mapping
          template<class ScalarType>
          result_type operator()(scalar<ScalarType> const & scal) const {
            mapped_scalar * p = new mapped_scalar(utils::type_to_string<ScalarType>::value());
            p->name_ = create_name(current_arg_, memory_, (void*)&scal);
            return p;
          }

          //Vector mapping
          template<class ScalarType>
          result_type operator()(vector_base<ScalarType> const & vec) const {
            mapped_vector * p = new mapped_vector(utils::type_to_string<ScalarType>::value());
            p->name_ = create_name(current_arg_, memory_, (void*)&vec);
            if(vec.start() > 0)
              p->start_name_ = p->name_ +"_start";
            if(vec.stride() > 1)
              p->stride_name_ = p->name_ + "_stride";
            return p;
          }

          //Symbolic vector mapping
          template<class ScalarType>
          result_type operator()(symbolic_vector_base<ScalarType> const & vec) const {
            mapped_symbolic_vector * p = new mapped_symbolic_vector(utils::type_to_string<ScalarType>::value());

            if(vec.is_value_static()==false)
              p->value_name_ = create_name(current_arg_, memory_, NULL);
            if(vec.has_index())
              p->value_name_ = create_name(current_arg_, memory_, NULL);
            return p;
          }

          //Matrix mapping
          template<class ScalarType, class Layout>
          result_type operator()(matrix_base<ScalarType, Layout> const & mat) const {
            mapped_matrix * p = new mapped_matrix(utils::type_to_string<ScalarType>::value());
            p->name_ = create_name(current_arg_, memory_, (void*)&mat);
            p->is_row_major_ = static_cast<bool>(utils::is_same_type<Layout, viennacl::row_major>::value);
            if(mat.start1() > 0)
              p->start1_name_ = p->name_ +"_start1";
            if(mat.stride1() > 1)
              p->stride1_name_ = p->name_ + "_stride1";
            if(mat.start2() > 0)
              p->start2_name_ = p->name_ +"_start2";
            if(mat.stride2() > 1)
              p->stride2_name_ = p->name_ + "_stride2";
            return p;
          }

          //Symbolic matrix mapping
          template<class ScalarType>
          result_type operator()(symbolic_matrix_base<ScalarType> const & vec) const {
            mapped_symbolic_matrix * p = new mapped_symbolic_matrix(utils::type_to_string<ScalarType>::value());
            return p;
          }

        private:
          std::map<void *, std::size_t> & memory_;
          unsigned int & current_arg_;
      };

      static void map_statement(scheduler::statement const & statement, std::map<void *, std::size_t> & memory, unsigned int & current_arg, mapping_type & mapping){
          scheduler::statement::container_type const & expr = statement.array();
          for(std::size_t i = 0 ; i < expr.size() ; ++i){
            scheduler::statement_node node = expr[i];

            if(node.lhs.type_family!=COMPOSITE_OPERATION_FAMILY)
              mapping.insert(std::make_pair(std::make_pair(i, LHS_NODE_TYPE), utils::call_on_element(node.lhs.type_family, node.lhs.type, node.lhs, map_functor(memory, current_arg))));

            if(node.op.type==OPERATION_BINARY_INNER_PROD_TYPE){
              mapping.insert(std::make_pair(std::make_pair(i, PARENT_TYPE), map_functor(memory, current_arg).binary_leaf<mapped_scalar_reduction>(i, node, &expr, &mapping)));
            }
            else if(node.op.type==OPERATION_BINARY_MAT_VEC_PROD_TYPE){
              mapping.insert(std::make_pair(std::make_pair(i, PARENT_TYPE), map_functor(memory, current_arg).binary_leaf<mapped_vector_reduction>(i, node, &expr, &mapping)));
            }
            else if(node.op.type==OPERATION_BINARY_MAT_MAT_PROD_TYPE){
              mapping.insert(std::make_pair(std::make_pair(i, PARENT_TYPE), map_functor(memory, current_arg).binary_leaf<mapped_matrix_product>(i, node, &expr, &mapping)));
            }

            if(node.rhs.type_family!=COMPOSITE_OPERATION_FAMILY)
              mapping.insert(std::make_pair(std::make_pair(i, RHS_NODE_TYPE), utils::call_on_element(node.rhs.type_family, node.rhs.type, node.rhs, map_functor(memory, current_arg))));
          }
      }

      template<class InputIterator>
      static void map_all_statements(InputIterator begin, InputIterator end, std::vector<detail::mapping_type> & mapping){
        std::map<void *, std::size_t> memory;
        unsigned int current_arg = 0;
        std::size_t i = 0;
        while(begin!=end){
          map_statement(*begin, memory, current_arg, mapping[i++]);
          ++begin;
        }
      }

    }

  }

}
#endif
