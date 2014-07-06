#ifndef VIENNACL_DEVICE_SPECIFIC_TREE_PARSING_FILTER_HPP
#define VIENNACL_DEVICE_SPECIFIC_TREE_PARSING_FILTER_HPP

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

#include "viennacl/device_specific/forwards.h"
#include "viennacl/device_specific/tree_parsing/traverse.hpp"

namespace viennacl{

  namespace device_specific{

    namespace tree_parsing{

      class filter : public traversal_functor{
        public:
          typedef bool (*pred_t)(scheduler::statement_node const & node);

          filter(pred_t pred, std::vector<vcl_size_t> & out) : pred_(pred), out_(out){ }

          void operator()(scheduler::statement const & statement, vcl_size_t root_idx, leaf_t) const
          {
             scheduler::statement_node const * root_node = &statement.array()[root_idx];
             if(pred_(*root_node))
               out_.push_back(root_idx);
          }
      private:
          pred_t pred_;
          std::vector<vcl_size_t> & out_;
      };


    }
  }
}
#endif
