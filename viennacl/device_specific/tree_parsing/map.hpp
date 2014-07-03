#ifndef VIENNACL_DEVICE_SPECIFIC_TREE_PARSING_MAP_HPP
#define VIENNACL_DEVICE_SPECIFIC_TREE_PARSING_MAP_HPP

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


/** @file viennacl/generator/map_functor.hpp
    @brief Functor to map the statements to the types defined in mapped_objects.hpp
*/

#include <set>

#include "viennacl/forwards.h"

#include "viennacl/scheduler/forwards.h"
#include "viennacl/scheduler/io.hpp"
#include "viennacl/device_specific/forwards.h"

#include "viennacl/tools/shared_ptr.hpp"

#include "viennacl/device_specific/forwards.h"
#include "viennacl/device_specific/tree_parsing/traverse.hpp"
#include "viennacl/device_specific/utils.hpp"
#include "viennacl/device_specific/mapped_objects.hpp"

#include "viennacl/traits/row_major.hpp"

namespace viennacl{

  namespace device_specific{

    namespace tree_parsing{

      /** @brief Functor to map the statements to the types defined in mapped_objects.hpp */
      class map_functor : public traversal_functor{

          scheduler::statement_node_numeric_type numeric_type(scheduler::statement const * statement, vcl_size_t root_idx) const
          {
              scheduler::statement_node const * root_node = &statement->array()[root_idx];
              while(root_node->lhs.numeric_type==scheduler::INVALID_NUMERIC_TYPE)
                  root_node = &statement->array()[root_node->lhs.node_index];
              return root_node->lhs.numeric_type;
          }

        public:
          typedef tools::shared_ptr<mapped_object> result_type;

          map_functor(symbolic_binder & binder, mapping_type & mapping) : binder_(binder), mapping_(mapping){ }

          /** @brief Binary leaf */
          template<class T>
          result_type binary_leaf(scheduler::statement const * statement, vcl_size_t root_idx, mapping_type const * mapping) const
          {
            return result_type(new T(utils::numeric_type_to_string(numeric_type(statement,root_idx)), binder_.get(NULL), mapped_object::node_info(mapping, statement, root_idx)));
          }

          template<class ScalarType>
          result_type operator()(ScalarType const & /*sca*l*/) const
          {
            return result_type(new mapped_host_scalar(utils::type_to_string<ScalarType>::value(), binder_.get(NULL)));
          }

          /** @brief Scalar mapping */
          template<class ScalarType>
          result_type operator()(scalar<ScalarType> const & scal) const
          {
            return result_type(new mapped_scalar(utils::type_to_string<ScalarType>::value(), binder_.get(&viennacl::traits::handle(scal))));
          }

          /** @brief Vector mapping */
          template<class ScalarType>
          result_type operator()(vector_base<ScalarType> const & vec) const
          {
            return result_type(new mapped_vector(utils::type_to_string<ScalarType>::value(), binder_.get(&viennacl::traits::handle(vec))));
          }

          /** @brief Implicit vector mapping */
          template<class ScalarType>
          result_type operator()(implicit_vector_base<ScalarType> const & /*vec*/) const
          {
            return result_type(new mapped_implicit_vector(utils::type_to_string<ScalarType>::value(), binder_.get(NULL)));
          }

          /** @brief Matrix mapping */
          template<class ScalarType>
          result_type operator()(matrix_base<ScalarType> const & mat) const
          {
            return result_type(new mapped_matrix(utils::type_to_string<ScalarType>::value(), binder_.get(&viennacl::traits::handle(mat)),
                                                  viennacl::traits::row_major(mat)));
          }

          /** @brief Implicit matrix mapping */
          template<class ScalarType>
          result_type operator()(implicit_matrix_base<ScalarType> const & /*mat*/) const
          {
            return result_type(new mapped_implicit_matrix(utils::type_to_string<ScalarType>::value(), binder_.get(NULL)));
          }

          /** @brief Traversal functor */
          void operator()(scheduler::statement const & statement, vcl_size_t root_idx, node_type node_type) const {
            mapping_type::key_type key(root_idx, node_type);
            scheduler::statement_node const & root_node = statement.array()[root_idx];

            if(node_type == LHS_NODE_TYPE && root_node.lhs.type_family != scheduler::COMPOSITE_OPERATION_FAMILY)
                 mapping_.insert(mapping_type::value_type(key, utils::call_on_element(root_node.lhs, *this)));
            else if(node_type == RHS_NODE_TYPE && root_node.rhs.type_family != scheduler::COMPOSITE_OPERATION_FAMILY)
                 mapping_.insert(mapping_type::value_type(key,  utils::call_on_element(root_node.rhs, *this)));
            else if( node_type== PARENT_NODE_TYPE)
            {
                if(root_node.op.type==scheduler::OPERATION_BINARY_VECTOR_DIAG_TYPE)
                  mapping_.insert(mapping_type::value_type(key, binary_leaf<mapped_vector_diag>(&statement, root_idx, &mapping_)));
                else if(root_node.op.type==scheduler::OPERATION_BINARY_MATRIX_DIAG_TYPE)
                  mapping_.insert(mapping_type::value_type(key, binary_leaf<mapped_matrix_diag>(&statement, root_idx, &mapping_)));
                else if(root_node.op.type==scheduler::OPERATION_BINARY_MATRIX_ROW_TYPE)
                  mapping_.insert(mapping_type::value_type(key, binary_leaf<mapped_matrix_row>(&statement, root_idx, &mapping_)));
                else if(root_node.op.type==scheduler::OPERATION_BINARY_MATRIX_COLUMN_TYPE)
                  mapping_.insert(mapping_type::value_type(key, binary_leaf<mapped_matrix_column>(&statement, root_idx, &mapping_)));
                else if(is_scalar_reduction(root_node))
                  mapping_.insert(mapping_type::value_type(key, binary_leaf<mapped_scalar_reduction>(&statement, root_idx, &mapping_)));
                else if(is_vector_reduction(root_node))
                  mapping_.insert(mapping_type::value_type(key, binary_leaf<mapped_vector_reduction>(&statement, root_idx, &mapping_)));
                else if(root_node.op.type == scheduler::OPERATION_BINARY_MAT_MAT_PROD_TYPE)
                  mapping_.insert(mapping_type::value_type(key, binary_leaf<mapped_matrix_product>(&statement, root_idx, &mapping_)));
                else if(root_node.op.type == scheduler::OPERATION_UNARY_TRANS_TYPE)
                  mapping_.insert(mapping_type::value_type(key, binary_leaf<mapped_trans>(&statement, root_idx, &mapping_)));
           }
          }

        private:
          symbolic_binder & binder_;
          mapping_type & mapping_;
      };

    }

  }

}
#endif
