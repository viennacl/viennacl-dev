#ifndef VIENNACL_DEVICE_SPECIFIC_TREE_PARSING_READ_WRITE_HPP
#define VIENNACL_DEVICE_SPECIFIC_TREE_PARSING_READ_WRITE_HPP

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


#include <set>

#include "CL/cl.h"

#include "viennacl/forwards.h"
#include "viennacl/scheduler/forwards.h"

#include "viennacl/device_specific/utils.hpp"
#include "viennacl/device_specific/forwards.h"
#include "viennacl/device_specific/mapped_objects.hpp"
#include "viennacl/device_specific/tree_parsing/traverse.hpp"

namespace viennacl{

  namespace device_specific{

    namespace tree_parsing{

      class read_write_traversal : public traversal_functor{
        public:
          enum mode_t { FETCH, WRITE_BACK };

          read_write_traversal(mode_t mode, unsigned int simd_width, std::string suffix, std::set<std::string> & cache,
                               index_tuple const & index, utils::kernel_generation_stream & stream, mapping_type const & mapping)
            : mode_(mode), simd_width_(simd_width), suffix_(suffix), cache_(cache), index_(index), stream_(stream), mapping_(mapping){ }

          void operator()(scheduler::statement const & /*statement*/, unsigned int root_idx, node_type leaf) const {
             mapping_type::const_iterator it = mapping_.find(std::make_pair(root_idx, leaf));
             if(it!=mapping_.end())
             {
               if(mode_==FETCH)
               {
                 if(fetchable * p = dynamic_cast<fetchable *>(it->second.get()))
                   p->fetch(simd_width_, suffix_, index_, cache_, stream_);
               }

               if(mode_==WRITE_BACK)
               {
                 if(writable * p = dynamic_cast<writable *>(it->second.get()))
                   p->write_back(simd_width_, suffix_, index_, cache_, stream_);
               }
             }
          }
      private:
        mode_t mode_;
        unsigned int simd_width_;
        std::string suffix_;
        std::set<std::string> & cache_;
        index_tuple index_;
        utils::kernel_generation_stream & stream_;
        mapping_type const & mapping_;
      };


      inline void read_write(read_write_traversal::mode_t mode, unsigned int simd_width, std::string const & suffix,
                                  std::set<std::string> & cache, scheduler::statement const & statement,unsigned int root_idx
                                  ,index_tuple const & index,utils::kernel_generation_stream & stream, mapping_type const & mapping, node_type leaf)
      {
        read_write_traversal traversal_functor(mode, simd_width, suffix, cache, index, stream, mapping);
        scheduler::statement_node const & root_node = statement.array()[root_idx];


        if(leaf==RHS_NODE_TYPE)
        {
          if(root_node.rhs.type_family==scheduler::COMPOSITE_OPERATION_FAMILY)
            traverse(statement, root_node.rhs.node_index, traversal_functor, false);
          else
            traversal_functor(statement, root_idx, leaf);
        }
        else if(leaf==LHS_NODE_TYPE)
        {
          if(root_node.lhs.type_family==scheduler::COMPOSITE_OPERATION_FAMILY)
            traverse(statement, root_node.lhs.node_index, traversal_functor, false);
          else
            traversal_functor(statement, root_idx, leaf);
        }
        else
          traverse(statement, root_idx, traversal_functor, false);
      }

    }
  }
}
#endif
