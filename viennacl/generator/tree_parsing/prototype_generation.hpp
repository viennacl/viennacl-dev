#ifndef VIENNACL_GENERATOR_TREE_PARSING_PROTOTYPE_GENERATION_HPP
#define VIENNACL_GENERATOR_TREE_PARSING_PROTOTYPE_GENERATION_HPP

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


/** @file viennacl/generator/set_arguments_functor.hpp
    @brief Functor to set the arguments of a statement into a kernel
*/

#include <set>

#include "viennacl/vector.hpp"

#include "viennacl/forwards.h"
#include "viennacl/scheduler/forwards.h"
#include "viennacl/generator/forwards.h"

#include "viennacl/generator/tree_parsing/traverse.hpp"



namespace viennacl{

  namespace generator{

    namespace tree_parsing{

      /** @brief functor for generating the prototype of a statement */
      class prototype_generation_traversal : public traversal_functor{
        private:
          std::string & str_;
          std::set<std::string> & already_generated_;
          mapping_type const & mapping_;
        public:
          prototype_generation_traversal(std::set<std::string> & already_generated, std::string & str, mapping_type const & mapping) : str_(str), already_generated_(already_generated), mapping_(mapping){ }

          void operator()(scheduler::statement const *, scheduler::statement_node const * root_node, node_type node_type) const {
              if( (node_type==LHS_NODE_TYPE && root_node->lhs.type_family!=scheduler::COMPOSITE_OPERATION_FAMILY)
                ||(node_type==RHS_NODE_TYPE && root_node->rhs.type_family!=scheduler::COMPOSITE_OPERATION_FAMILY) )
                  append_kernel_arguments(already_generated_, str_, *mapping_.at(std::make_pair(root_node,node_type)));
          }
      };

    }

  }

}
#endif
