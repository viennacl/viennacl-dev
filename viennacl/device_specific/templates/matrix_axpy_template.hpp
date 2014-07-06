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
        parameters(const char * _scalartype, unsigned int _simd_width,
                   unsigned int _local_size_0, unsigned int _local_size_1,
                   unsigned int _num_groups_0, unsigned int _num_groups_1,
                   unsigned int _decomposition) : template_base::parameters(_scalartype, _simd_width, _local_size_0, _local_size_1, 1), num_groups_0(_num_groups_0), num_groups_1(_num_groups_1), decomposition(_decomposition){ }

        unsigned int num_groups_0;
        unsigned int num_groups_1;
        unsigned int decomposition;
      };

    private:
      void core(unsigned int /*kernel_id*/, utils::kernel_generation_stream& stream, statements_container const & statements, std::vector<mapping_type> const & mappings) const
      {
        statements_container::data_type::const_iterator sit;
        std::vector<mapping_type>::const_iterator mit;

        stream << "for(unsigned int i = get_global_id(0) ; i < M ; i += get_global_size(0))" << std::endl;
        stream << "{" << std::endl;
        stream.inc_tab();
        stream << "for(unsigned int j = get_global_id(1) ; j < N ; j += get_global_size(1))" << std::endl;
        stream << "{" << std::endl;
        stream.inc_tab();

        index_tuple idx("i","M","j","N");

        //Fetches entries to registers
        std::set<std::string>  cache;
        for(mit = mappings.begin(), sit = statements.data().begin() ; sit != statements.data().end() ; ++sit, ++mit)
          tree_parsing::read_write(tree_parsing::read_write_traversal::FETCH, "reg", cache,*sit, sit->root(), idx, stream, *mit, tree_parsing::PARENT_NODE_TYPE);

        unsigned int i = 0;
        for(mit = mappings.begin(), sit = statements.data().begin() ; sit != statements.data().end() ; ++sit, ++mit){
          std::string str;
          tree_parsing::traverse(*sit, sit->root(), tree_parsing::evaluate_expression_traversal(idx, 0, str, mappings[i++]), false);
          stream << str << ";" << std::endl;
        }

        //Write back
        for(mit = mappings.begin(), sit = statements.data().begin() ; sit != statements.data().end() ; ++sit, ++mit)
          tree_parsing::read_write(tree_parsing::read_write_traversal::WRITE_BACK, "reg", cache,*sit, sit->root(), idx, stream, *mit, tree_parsing::LHS_NODE_TYPE);


        stream.dec_tab();
        stream << "}" << std::endl;
        stream.dec_tab();
        stream << "}" << std::endl;
      }

      void add_kernel_arguments(statements_container const & /*statements*/, std::string & arguments_string) const
      {
        arguments_string += generate_value_kernel_argument("unsigned int", "M");
        arguments_string += generate_value_kernel_argument("unsigned int", "N");
      }

      void configure_impl(vcl_size_t /*kernel_id*/, viennacl::ocl::context & /*context*/, statements_container const & statements, viennacl::ocl::kernel & k, unsigned int & n_arg)  const
      {
        k.global_work_size(0,parameters_.local_size_0*parameters_.num_groups_0);
        k.global_work_size(1,parameters_.local_size_1*parameters_.num_groups_1);

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
      matrix_axpy_template(matrix_axpy_template::parameters const & parameters, binding_policy_t binding_policy = BIND_ALL_UNIQUE) : template_base(parameters, binding_policy), up_to_internal_size_(false), parameters_(parameters){ }

      void up_to_internal_size(bool v) { up_to_internal_size_ = v; }
    private:
      bool up_to_internal_size_;
      matrix_axpy_template::parameters const & parameters_;
    };

  }

}

#endif
