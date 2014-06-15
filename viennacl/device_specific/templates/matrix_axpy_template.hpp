#ifndef VIENNACL_DEVICE_SPECIFIC_TEMPLATES_MATRIX_AXPY_HPP
#define VIENNACL_DEVICE_SPECIFIC_TEMPLATES_MATRIX_AXPY_HPP

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


/** @file viennacl/generator/matrix_axpy.hpp
 *
 * Kernel template for the matrix axpy-like operations
*/

#include <vector>

#include "viennacl/scheduler/forwards.h"

#include "viennacl/device_specific/mapped_objects.hpp"
#include "viennacl/device_specific/tree_parsing/read_write.hpp"
#include "viennacl/device_specific/tree_parsing/evaluate_expression.hpp"
#include "viennacl/device_specific/utils.hpp"

#include "viennacl/device_specific/templates/template_base.hpp"

#include "viennacl/tools/tools.hpp"

namespace viennacl{

  namespace device_specific{

    class matrix_axpy_template : public template_base
    {

    public:
      class parameters : public template_base::parameters
      {
      public:
        parameters(const char * scalartype, unsigned int simd_width,
                   unsigned int local_size_0, unsigned int local_size_1,
                   unsigned int num_groups_0, unsigned int num_groups_1,
                   unsigned int decomposition) : template_base::parameters(scalartype, simd_width, local_size_0, local_size_1, 1), num_groups_0_(num_groups_0), num_groups_1_(num_groups_1), decomposition_(decomposition){ }

        unsigned int num_groups_0() const { return num_groups_0_; }
        unsigned int num_groups_1() const { return num_groups_1_; }
        unsigned int decomposition() const { return decomposition_; }

      private:
        unsigned int num_groups_0_;
        unsigned int num_groups_1_;
        unsigned int decomposition_;
      };

    private:
      void core(unsigned int /*kernel_id*/, utils::kernel_generation_stream& stream, statements_container const & statements, std::vector<mapping_type> const & mapping) const
      {
        stream << "for(unsigned int i = get_global_id(0) ; i < M ; i += get_global_size(0))" << std::endl;
        stream << "{" << std::endl;
        stream.inc_tab();
        stream << "for(unsigned int j = get_global_id(1) ; j < N ; j += get_global_size(1))" << std::endl;
        stream << "{" << std::endl;
        stream.inc_tab();

        //Fetches entries to registers
        std::set<std::string>  cache;
        for(statements_container::data_type::const_iterator it = statements.data().begin() ; it != statements.data().end() ; ++it)
          tree_parsing::read_write(tree_parsing::read_write_traversal::FETCH, "reg", cache,*it, it->root(), index_tuple("i","M","j","N"), stream,mapping[std::distance(statements.data().begin(),it)], tree_parsing::PARENT_NODE_TYPE);

        unsigned int i = 0;
        for(statements_container::data_type::const_iterator it = statements.data().begin() ; it != statements.data().end() ; ++it){
          std::string str;
          tree_parsing::traverse(*it, it->root(), tree_parsing::evaluate_expression_traversal(index_tuple("i","M","j","N"), -1, str, mapping[i++]), false);
          stream << str << ";" << std::endl;
        }

        //Write back
        for(statements_container::data_type::const_iterator it = statements.data().begin() ; it != statements.data().end() ; ++it)
          tree_parsing::read_write(tree_parsing::read_write_traversal::WRITE_BACK, "reg", cache,*it, it->root(), index_tuple("i","M","j","N"), stream,mapping[std::distance(statements.data().begin(),it)], tree_parsing::LHS_NODE_TYPE);


        stream.dec_tab();
        stream << "}" << std::endl;
        stream.dec_tab();
        stream << "}" << std::endl;
      }

      void add_kernel_arguments(statements_container const & statements, std::string & arguments_string) const
      {
        arguments_string += generate_value_kernel_argument("unsigned int", "M");
        arguments_string += generate_value_kernel_argument("unsigned int", "N");
      }

      void configure_impl(unsigned int /*kernel_id*/, statements_container const & statements, viennacl::ocl::kernel & k, unsigned int & n_arg)  const
      {
        k.global_work_size(0,parameters_.local_size_0()*parameters_.num_groups_0());
        k.global_work_size(1,parameters_.local_size_1()*parameters_.num_groups_1());

        scheduler::statement_node const & root = statements.data().front().array()[statements.data().front().root()];
        if(up_to_internal_size_)
        {
          k.arg(n_arg++, cl_uint(utils::call_on_matrix(root.lhs, utils::internal_size1_fun())));
          k.arg(n_arg++, cl_uint(utils::call_on_matrix(root.lhs, utils::internal_size2_fun())));
        }
        else
        {
          k.arg(n_arg++, cl_uint(utils::call_on_matrix(root.lhs, utils::size1_fun())));
          k.arg(n_arg++, cl_uint(utils::call_on_matrix(root.lhs, utils::size2_fun())));
        }
      }

    public:
      matrix_axpy_template(matrix_axpy_template::parameters const & parameters, binding_policy_t binding_policy = BIND_ALL_UNIQUE) : template_base(parameters, binding_policy), parameters_(parameters){ }

      void enqueue(std::string const & program_name, statements_container const & statements, bool up_to_internal_size = false)
      {
        up_to_internal_size_ = up_to_internal_size;
        template_base::enqueue(program_name, statements);
      }

    private:
      matrix_axpy_template::parameters const & parameters_;
      bool up_to_internal_size_;
    };

  }

}

#endif
