#ifndef VIENNACL_GENERATOR_FORWARDS_H
#define VIENNACL_GENERATOR_FORWARDS_H

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


/** @file viennacl/generator/forwards.h
    @brief Forwards declaration
*/

#include <map>
#include <list>
#include "viennacl/tools/shared_ptr.hpp"
#include "viennacl/scheduler/forwards.h"

namespace viennacl{

  namespace generator{

    namespace utils{
      class kernel_generation_stream;
    }

    namespace detail{

      using namespace viennacl::scheduler;

      enum node_type{
        LHS_NODE_TYPE,
        PARENT_NODE_TYPE,
        RHS_NODE_TYPE
      };

      class mapped_container;

      typedef std::pair<scheduler::statement_node const *, node_type> key_type;
      typedef tools::shared_ptr<detail::mapped_container> container_ptr_type;
      typedef std::map<key_type, container_ptr_type> mapping_type;

      template<class Fun>
      static void traverse(scheduler::statement const & statement, scheduler::statement_node const & root_node, Fun const & fun, bool recurse_binary_leaf = true);
      static std::string generate(std::pair<std::string, std::string> const & index, int vector_index, mapped_container const & s);
      static std::string & append_kernel_arguments(std::set<std::string> & already_generated, std::string & str, unsigned int vector_size, mapped_container const & s);
      static void fetch(std::pair<std::string, std::string> const & index, unsigned int vectorization, std::set<std::string> & fetched, utils::kernel_generation_stream & stream, mapped_container & s);
      static const char * generate(scheduler::operation_node_type arg);
      static void generate_all_rhs(scheduler::statement const & statement
                                , scheduler::statement_node const & root_node
                                , std::pair<std::string, std::string> const & index
                                , int vector_element
                                , std::string & str
                                , detail::mapping_type const & mapping);

    }

  }

}
#endif
