#ifndef VIENNACL_DEVICE_SPECIFIC_TEMPLATES_VECTOR_AXPY_HPP
#define VIENNACL_DEVICE_SPECIFIC_TEMPLATES_VECTOR_AXPY_HPP

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


/** @file viennacl/generator/vector_axpy.hpp
 *
 * Kernel template for the vector axpy-like operations
*/

#include <vector>

#include "viennacl/scheduler/forwards.h"

#include "viennacl/device_specific/mapped_objects.hpp"
#include "viennacl/device_specific/tree_parsing/fetch.hpp"
#include "viennacl/device_specific/tree_parsing/elementwise_expression.hpp"
#include "viennacl/device_specific/forwards.h"
#include "viennacl/device_specific/utils.hpp"

#include "viennacl/device_specific/templates/template_base.hpp"

#include "viennacl/tools/tools.hpp"

namespace viennacl{

  namespace device_specific{

    class vector_axpy_template : public template_base{

    public:
      vector_axpy_template(const char * scalartype, unsigned int simd_width,
                   unsigned int group_size, unsigned int num_groups,
                   unsigned int decomposition) : template_base(scalartype, simd_width, group_size, 1, 1), num_groups_(num_groups), decomposition_(decomposition){ }

      void configure_range_enqueue_arguments(unsigned int kernel_id, statements_container const statements, viennacl::ocl::kernel & k, unsigned int & n_arg)  const{
        configure_local_sizes(k, kernel_id);

        k.global_work_size(0,local_size_0_*num_groups_);
        k.global_work_size(1,1);

        scheduler::statement_node const & root = statements.front().first.array()[statements.front().second];
        viennacl::vcl_size_t N = utils::call_on_vector(root.lhs, utils::internal_size_fun());
        k.arg(n_arg++, cl_uint(N/simd_width_));
      }

      void add_kernel_arguments(std::string & arguments_string) const{
        arguments_string += generate_value_kernel_argument("unsigned int", "N");
      }

    private:

      void core(unsigned int /*kernel_id*/, utils::kernel_generation_stream& stream, statements_container const statements, std::vector<mapping_type> const & mapping) const {
        stream << "for(unsigned int i = get_global_id(0) ; i < N ; i += get_global_size(0))" << std::endl;
        stream << "{" << std::endl;
        stream.inc_tab();

        //Fetches entries to registers
        std::set<std::string>  fetched;
        for(std::vector<mapping_type>::const_iterator it = mapping.begin() ; it != mapping.end() ; ++it)
          for(mapping_type::const_reverse_iterator iit = it->rbegin() ; iit != it->rend() ; ++iit)
            //Useless to fetch cpu scalars into registers
            if(mapped_handle * p = dynamic_cast<mapped_handle *>(iit->second.get()))
              p->fetch( std::make_pair("i","0"), fetched, stream);

        //Generates all the expression, in order
        unsigned int i = 0;
        for(statements_container::const_iterator it = statements.begin() ; it != statements.end() ; ++it){
          std::string str;
          tree_parsing::traverse(it->first, it->second, tree_parsing::expression_generation_traversal(std::make_pair("i","0"), -1, str, mapping[i++]));
          stream << str << ";" << std::endl;
        }

        //Writes back
        for(statements_container::const_iterator it = statements.begin() ; it != statements.end() ; ++it)
          //Gets the mapped object at the LHS of each expression
          if(mapped_handle * p = dynamic_cast<mapped_handle *>(mapping.at(std::distance(statements.begin(),it)).at(std::make_pair(it->second, tree_parsing::LHS_NODE_TYPE)).get()))
            p->write_back( std::make_pair("i", "0"), fetched, stream);

        stream.dec_tab();
        stream << "}" << std::endl;
      }

    private:
      unsigned int num_groups_;
      unsigned int decomposition_;
    };

  }

}

#endif
