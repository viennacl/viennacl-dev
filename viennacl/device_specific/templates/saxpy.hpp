#ifndef VIENNACL_DEVICE_SPECIFIC_TEMPLATES_SAXPY_HPP
#define VIENNACL_DEVICE_SPECIFIC_TEMPLATES_SAXPY_HPP

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


/** @file viennacl/generator/saxpy.hpp
 *
 * Kernel template for the saxpy-like operation
*/

#include <vector>

#include "viennacl/scheduler/forwards.h"

#include "viennacl/device_specific/mapped_objects.hpp"
#include "viennacl/device_specific/tree_parsing/fetch.hpp"
#include "viennacl/device_specific/tree_parsing/elementwise_expression.hpp"
#include "viennacl/device_specific/utils.hpp"

#include "viennacl/device_specific/templates/template_base.hpp"

#include "viennacl/tools/tools.hpp"

namespace viennacl{

  namespace device_specific{

    class vector_saxpy : public profile_base{

    public:
      vector_saxpy(const char * scalartype, unsigned int simd_width,
                   unsigned int group_size, unsigned int num_groups,
                   unsigned int decomposition) : profile_base(scalartype, simd_width, group_size, 1, 1), num_groups_(num_groups), decomposition_(decomposition){ }

      unsigned int num_groups() const{ return num_groups_; }
      unsigned int decomposition() const{ return decomposition_; }

      void configure_range_enqueue_arguments(unsigned int kernel_id, viennacl::ocl::kernel & k, unsigned int & n_arg)  const{
        configure_local_sizes(k, kernel_id);

        k.global_work_size(0,local_size_1_*num_groups_);
        k.global_work_size(1,1);

        scheduler::statement_node const & first_node = statements_->front().second;
        viennacl::vcl_size_t N = utils::call_on_vector(first_node.lhs, utils::internal_size_fun());
        k.arg(n_arg++, cl_uint(N/simd_width_));
      }

      void add_kernel_arguments(std::string & arguments_string) const{
        arguments_string += generate_value_kernel_argument("unsigned int", "N");
      }

    private:

      void core(unsigned int /*kernel_id*/, utils::kernel_generation_stream& stream, std::vector<mapping_type> const & mapping) const {
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
        std::size_t i = 0;
        for(std::list< std::pair<scheduler::statement, scheduler::statement_node> >::const_iterator it = statements_->begin() ; it != statements_->end() ; ++it){
          std::string str;
          tree_parsing::traverse(it->first, it->second, tree_parsing::expression_generation_traversal(std::make_pair("i","0"), -1, str, mapping[i++]));
          stream << str << ";" << std::endl;
        }

        //Writes back
        for(std::list< std::pair<scheduler::statement, scheduler::statement_node> >::const_iterator it = statements_->begin() ; it != statements_->end() ; ++it)
          //Gets the mapped object at the LHS of each expression
          if(mapped_handle * p = dynamic_cast<mapped_handle *>(mapping.at(std::distance(statements_->begin(),it)).at(std::make_pair(&it->second, tree_parsing::LHS_NODE_TYPE)).get()))
            p->write_back( std::make_pair("i", "0"), fetched, stream);

        stream.dec_tab();
        stream << "}" << std::endl;
      }

    private:
      unsigned int num_groups_;
      unsigned int decomposition_;
    };



    class matrix_saxpy : public profile_base{

      bool invalid_impl(viennacl::ocl::device const & /*dev*/, size_t /*scalartype_size*/) const{ return false; }
      bool is_slow_impl(viennacl::ocl::device const &) const { return false; }

    public:
      matrix_saxpy(const char * scalartype, unsigned int simd_width,
                   unsigned int gs1, unsigned int gs2,
                   unsigned int ng1, unsigned int ng2, unsigned int d) : profile_base(scalartype, simd_width, gs1, gs2, 1), num_groups_row_(ng1), num_groups_col_(ng2), decomposition_(d){ }

      void configure_range_enqueue_arguments(unsigned int kernel_id, viennacl::ocl::kernel & k, unsigned int & n_arg)  const
      {
        configure_local_sizes(k, kernel_id);

        k.global_work_size(0,local_size_1_*num_groups_row_);
        k.global_work_size(1,local_size_1_*num_groups_col_);

        scheduler::statement_node const & first_node = statements_->front().second;
        k.arg(n_arg++, cl_uint(utils::call_on_matrix(first_node.lhs, utils::internal_size1_fun())));
        k.arg(n_arg++, cl_uint(utils::call_on_matrix(first_node.lhs, utils::internal_size2_fun())));
      }

      virtual void add_kernel_arguments(std::string & arguments_string) const{
        arguments_string += generate_value_kernel_argument("unsigned int", "M");
        arguments_string += generate_value_kernel_argument("unsigned int", "N");
      }

    private:
      void core(unsigned int /*kernel_id*/, utils::kernel_generation_stream& stream, std::vector<mapping_type> const & mapping) const {
        stream << "for(unsigned int i = get_global_id(0) ; i < M ; i += get_global_size(0))" << std::endl;
        stream << "{" << std::endl;
        stream.inc_tab();
        stream << "for(unsigned int j = get_global_id(1) ; j < N ; j += get_global_size(1))" << std::endl;
        stream << "{" << std::endl;
        stream.inc_tab();

        //Fetches entries to registers
        std::set<std::string>  fetched;
        for(std::vector<mapping_type>::const_iterator it = mapping.begin() ; it != mapping.end() ; ++it)
          for(mapping_type::const_reverse_iterator it2 = it->rbegin() ; it2 != it->rend() ; ++it2)
            if(mapped_matrix * p = dynamic_cast<mapped_matrix *>(it2->second.get()))
              p->fetch(std::make_pair("i", "j"), fetched, stream);


        std::size_t i = 0;
        for(std::list< std::pair<scheduler::statement, scheduler::statement_node> >::const_iterator it = statements_->begin() ; it != statements_->end() ; ++it){
          std::string str;
          tree_parsing::traverse(it->first, it->second, tree_parsing::expression_generation_traversal(std::make_pair("i", "j"), -1, str, mapping[i++]));
          stream << str << ";" << std::endl;
        }

        //Writes back
        for(std::list< std::pair<scheduler::statement, scheduler::statement_node> >::const_iterator it = statements_->begin() ; it != statements_->end() ; ++it){
          if(mapped_handle * p = dynamic_cast<mapped_handle *>(mapping.at(std::distance(statements_->begin(),it)).at(std::make_pair(&it->second,tree_parsing::LHS_NODE_TYPE)).get()))
            p->write_back(std::make_pair("i", "j"), fetched, stream);
        }

        stream.dec_tab();
        stream << "}" << std::endl;
        stream.dec_tab();
        stream << "}" << std::endl;
      }

    private:
      unsigned int num_groups_row_;
      unsigned int num_groups_col_;
      unsigned int decomposition_;
    };
  }

}

#endif
