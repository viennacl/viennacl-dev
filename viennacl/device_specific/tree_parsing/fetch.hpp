#ifndef VIENNACL_DEVICE_SPECIFIC_TREE_PARSING_FETCH_HPP
#define VIENNACL_DEVICE_SPECIFIC_TREE_PARSING_FETCH_HPP

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


/** @file viennacl/generator/helpers.hpp
    @brief several code generation helpers
*/

#include <set>

#include "CL/cl.h"

#include "viennacl/forwards.h"
#include "viennacl/scheduler/forwards.h"

#include "viennacl/device_specific/utils.hpp"
#include "viennacl/device_specific/forwards.h"
#include "viennacl/device_specific/tree_parsing/traverse.hpp"

namespace viennacl{

  namespace device_specific{

    namespace tree_parsing{

      /** @brief functor for fetching the elements of a statement */
      class fetch_traversal : public traversal_functor{
        private:
          std::set<std::string> & fetched_;
          std::pair<std::string, std::string> index_string_;
          utils::kernel_generation_stream & stream_;
          mapping_type const & mapping_;
        public:
          fetch_traversal(std::set<std::string> & fetched, std::pair<std::string, std::string> const & index, utils::kernel_generation_stream & stream, mapping_type const & mapping) : fetched_(fetched), index_string_(index), stream_(stream), mapping_(mapping){ }

          void operator()(scheduler::statement const & statement, unsigned int root_idx, node_type node_type) const {
            scheduler::statement_node const & root_node = statement.array()[root_idx];
            if( (node_type==LHS_NODE_TYPE && root_node.lhs.type_family!=scheduler::COMPOSITE_OPERATION_FAMILY)
              ||(node_type==RHS_NODE_TYPE && root_node.rhs.type_family!=scheduler::COMPOSITE_OPERATION_FAMILY) )
              fetch(index_string_, fetched_, stream_, *mapping_.at(std::make_pair(root_idx, node_type)));
          }
      };

      /** @brief functor for fetching the LHS of a statement's node
      *
      *   Forwards to fetch_traversal functor if the LHS is not a leaf
      */
      inline void fetch_all_lhs(std::set<std::string> & fetched
                                , scheduler::statement const & statement
                                , unsigned int root_idx
                                , std::pair<std::string, std::string> const & index
                                , utils::kernel_generation_stream & stream
                                , mapping_type const & mapping){
        scheduler::statement_node const & root_node = statement.array()[root_idx];
        if(root_node.lhs.type_family==scheduler::COMPOSITE_OPERATION_FAMILY)
          traverse(statement, root_node.lhs.node_index, fetch_traversal(fetched, index, stream, mapping));
        else
          fetch(index, fetched, stream, *mapping.at(std::make_pair(root_idx,LHS_NODE_TYPE)));

      }

      /** @brief functor for fetching the RHS of a statement's node
      *
      *   Forwards to fetch_traversal functor if the RHS is not a leaf
      */
      inline void fetch_all_rhs(std::set<std::string> & fetched
                                , scheduler::statement const & statement
                                , unsigned int root_idx
                                , std::pair<std::string, std::string> const & index
                                , utils::kernel_generation_stream & stream
                                , mapping_type const & mapping){
        scheduler::statement_node const & root_node = statement.array()[root_idx];
        if(root_node.rhs.type_family==scheduler::COMPOSITE_OPERATION_FAMILY)
          traverse(statement, root_node.rhs.node_index, fetch_traversal(fetched, index, stream, mapping));
        else
          fetch(index, fetched, stream, *mapping.at(std::make_pair(root_idx,RHS_NODE_TYPE)));

      }

    }
  }
}
#endif
