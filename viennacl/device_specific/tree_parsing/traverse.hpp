#ifndef VIENNACL_DEVICE_SPECIFIC_TREE_PARSING_TRAVERSE_HPP
#define VIENNACL_DEVICE_SPECIFIC_TREE_PARSING_TRAVERSE_HPP

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

namespace viennacl{

  namespace device_specific{

    namespace tree_parsing{

      /** @brief base functor class for traversing a statement */
      class traversal_functor{
        public:
          void call_before_expansion(scheduler::statement const &, unsigned int) const { }
          void call_after_expansion(scheduler::statement const &, unsigned int) const { }
      };

      /** @brief Recursively execute a functor on a statement */
      template<class Fun>
      inline void traverse(scheduler::statement const & statement, unsigned int root_idx, Fun const & fun, bool inspect){
        scheduler::statement_node const & root_node = statement.array()[root_idx];
        bool recurse = utils::node_leaf(root_node.op)?inspect:true;

        fun.call_before_expansion(statement, root_idx);

        //Lhs:
        if(recurse)
        {
          if(root_node.lhs.type_family==scheduler::COMPOSITE_OPERATION_FAMILY)
            traverse(statement, root_node.lhs.node_index, fun, inspect);
          if(root_node.lhs.type_family != scheduler::INVALID_TYPE_FAMILY)
            fun(statement, root_idx, LHS_NODE_TYPE);
        }

        //Self:
        fun(statement, root_idx, PARENT_NODE_TYPE);

        //Rhs:
        if(recurse && root_node.op.type_family!=scheduler::OPERATION_UNARY_TYPE_FAMILY)
        {
          if(root_node.rhs.type_family==scheduler::COMPOSITE_OPERATION_FAMILY)
            traverse(statement, root_node.rhs.node_index, fun, inspect);
          if(root_node.rhs.type_family != scheduler::INVALID_TYPE_FAMILY)
            fun(statement, root_idx, RHS_NODE_TYPE);
        }

        fun.call_after_expansion(statement, root_idx);


      }

    }
  }
}
#endif
