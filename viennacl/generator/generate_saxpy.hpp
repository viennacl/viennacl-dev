#ifndef VIENNACL_GENERATOR_GENERATE_SAXPY_HPP
#define VIENNACL_GENERATOR_GENERATE_SAXPY_HPP

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


/** @file viennacl/generator/templates/saxpy.hpp
 *
 * Kernel template for the SAXPY operation
*/

#include <vector>

#include "viennacl/scheduler/forwards.h"

#include "viennacl/generator/mapped_types.hpp"
#include "viennacl/generator/generate_utils.hpp"
#include "viennacl/generator/utils.hpp"

#include "viennacl/generator/generate_template_base.hpp"

#include "viennacl/tools/tools.hpp"

namespace viennacl{

  namespace generator{

    class vector_saxpy : public profile_base{
      public:
        vector_saxpy(unsigned int v, std::size_t gs, std::size_t ng, bool d) : profile_base(v, 1), group_size_(gs), num_groups_(ng), global_decomposition_(d){ }

        void set_local_sizes(std::size_t & size1, std::size_t & size2, std::size_t/* kernel_id*/) const{
          size1 = group_size_;
          size2 = 1;
        }

        void configure_range_enqueue_arguments(std::size_t kernel_id, statements_type  const & statements, viennacl::ocl::kernel & k, unsigned int & n_arg)  const{
          configure_local_sizes(k, kernel_id);

          k.global_work_size(0,group_size_*num_groups_);
          k.global_work_size(1,1);

          scheduler::statement_node const & first_node = statements.front().second;
          viennacl::vcl_size_t N = utils::call_on_vector(first_node.lhs, utils::size_fun());
          k.arg(n_arg++, cl_uint(N/vectorization_));
        }
        void kernel_arguments(statements_type  const & /*statements*/, std::string & arguments_string) const{
          arguments_string += detail::generate_value_kernel_argument("unsigned int", "N");
        }

        virtual void print(std::ostream & s) const{
          s << "Vector Saxpy : { vector_type, group_size, num_groups, global_decomposition } = {"
            << vectorization_
            << ", " << group_size_
            << ", " << num_groups_
            << ", " << global_decomposition_
            << "}";
        }

      private:
        void core(std::size_t kernel_id, utils::kernel_generation_stream& stream, statements_type const & statements, std::vector<detail::mapping_type> const & mapping) const {
          stream << "for(unsigned int i = get_global_id(0) ; i < N ; i += get_global_size(0))" << std::endl;
          stream << "{" << std::endl;
          stream.inc_tab();

          //Fetches entries to registers
          std::set<std::string>  fetched;
          for(std::vector<detail::mapping_type>::const_iterator it = mapping.begin() ; it != mapping.end() ; ++it)
            for(detail::mapping_type::const_reverse_iterator iit = it->rbegin() ; iit != it->rend() ; ++iit)
              if(detail::mapped_handle * p = dynamic_cast<detail::mapped_handle *>(iit->second.get()))
                p->fetch( std::make_pair("i","0"), vectorization_, fetched, stream);

          std::size_t i = 0;
          for(statements_type::const_iterator it = statements.begin() ; it != statements.end() ; ++it){
            std::string str;
            detail::traverse(it->first, it->second, detail::expression_generation_traversal(std::make_pair("i","0"), -1, str, mapping[i++]));
            stream << str << ";" << std::endl;
          }

          //Writes back
          for(statements_type::const_iterator it = statements.begin() ; it != statements.end() ; ++it)
            if(detail::mapped_handle * p = dynamic_cast<detail::mapped_handle *>(mapping.at(std::distance(statements.begin(),it)).at(std::make_pair(&it->second, detail::LHS_NODE_TYPE)).get()))
              p->write_back( std::make_pair("i", "0"), fetched, stream);

          stream.dec_tab();
          stream << "}" << std::endl;
        }

      private:
        std::size_t group_size_;
        std::size_t num_groups_;
        bool global_decomposition_;

    };



    class matrix_saxpy : public profile_base{

        virtual void print(std::ostream & s) const{
          s << "Matrix Saxpy : { vector_type, group_size_row, group_size_col, num_groups_row, num_group_col, global_decomposition } = {"
            << vectorization_
            << ", " << group_size_row_
            << ", " << group_size_col_
            << ", " << num_groups_row_
            << ", " << num_groups_col_
            << ", " << global_decomposition_
            << "}";
        }

      public:
        matrix_saxpy(unsigned int v, std::size_t gs1, std::size_t gs2, std::size_t ng1, std::size_t ng2, bool d) : profile_base(v, 1), group_size_row_(gs1), group_size_col_(gs2), num_groups_row_(ng1), num_groups_col_(ng2), global_decomposition_(d){ }

        void set_local_sizes(std::size_t & s1, std::size_t & s2, std::size_t /*kernel_id*/) const{
          s1 = group_size_row_;
          s2 = group_size_col_;
        }

        void configure_range_enqueue_arguments(std::size_t kernel_id, statements_type  const & statements, viennacl::ocl::kernel & k, unsigned int & n_arg)  const{
          configure_local_sizes(k, kernel_id);

          k.global_work_size(0,group_size_row_*num_groups_row_);
          k.global_work_size(1,group_size_col_*num_groups_col_);

          scheduler::statement_node const & first_node = statements.front().second;
          k.arg(n_arg++, cl_uint(utils::call_on_matrix(first_node.lhs, utils::size1_fun())));
          k.arg(n_arg++, cl_uint(utils::call_on_matrix(first_node.lhs, utils::size2_fun())));
        }

        void kernel_arguments(statements_type  const & /*statements*/, std::string & arguments_string) const{
          arguments_string += detail::generate_value_kernel_argument("unsigned int", "M");
          arguments_string += detail::generate_value_kernel_argument("unsigned int", "N");
        }

      private:
        void core(std::size_t kernel_id, utils::kernel_generation_stream& stream, statements_type const & statements, std::vector<detail::mapping_type> const & mapping) const {

          for(std::vector<detail::mapping_type>::const_iterator it = mapping.begin() ; it != mapping.end() ; ++it){
            for(detail::mapping_type::const_iterator iit = it->begin() ; iit != it->end() ; ++iit){
              if(detail::mapped_matrix * p = dynamic_cast<detail::mapped_matrix*>(iit->second.get()))
                p->bind_sizes("M","N");
            }
          }

          stream << "for(unsigned int i = get_global_id(0) ; i < M ; i += get_global_size(0))" << std::endl;
          stream << "{" << std::endl;
          stream.inc_tab();
          stream << "for(unsigned int j = get_global_id(1) ; j < N ; j += get_global_size(1))" << std::endl;
          stream << "{" << std::endl;
          stream.inc_tab();

          //Fetches entries to registers
          std::set<std::string>  fetched;
          for(std::vector<detail::mapping_type>::const_iterator it = mapping.begin() ; it != mapping.end() ; ++it)
            for(detail::mapping_type::const_reverse_iterator it2 = it->rbegin() ; it2 != it->rend() ; ++it2)
              if(detail::mapped_matrix * p = dynamic_cast<detail::mapped_matrix *>(it2->second.get()))
                p->fetch(std::make_pair("i", "j"), vectorization_, fetched, stream);


          std::size_t i = 0;
          for(statements_type::const_iterator it = statements.begin() ; it != statements.end() ; ++it){
            std::string str;
            detail::traverse(it->first, it->second, detail::expression_generation_traversal(std::make_pair("i", "j"), -1, str, mapping[i++]));
            stream << str << ";" << std::endl;
          }

          //Writes back
          for(statements_type::const_iterator it = statements.begin() ; it != statements.end() ; ++it){
            if(detail::mapped_handle * p = dynamic_cast<detail::mapped_handle *>(mapping.at(std::distance(statements.begin(),it)).at(std::make_pair(&it->second,detail::LHS_NODE_TYPE)).get()))
              p->write_back(std::make_pair("i", "j"), fetched, stream);
          }

          stream.dec_tab();
          stream << "}" << std::endl;
          stream.dec_tab();
          stream << "}" << std::endl;
        }

      private:
        std::size_t group_size_row_;
        std::size_t group_size_col_;

        std::size_t num_groups_row_;
        std::size_t num_groups_col_;

        bool global_decomposition_;
    };
  }

}

#endif
