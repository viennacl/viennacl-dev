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
#include "viennacl/device_specific/tree_parsing.hpp"
#include "viennacl/device_specific/utils.hpp"

#include "viennacl/device_specific/templates/template_base.hpp"

#include "viennacl/tools/tools.hpp"

namespace viennacl{

  namespace device_specific{

    class matrix_axpy_template : public template_base
    {

    public:
      class parameters_type : public template_base::parameters_type
      {
      public:
        parameters_type(unsigned int _simd_width,
                   unsigned int _local_size_0, unsigned int _local_size_1,
                   unsigned int _num_groups_0, unsigned int _num_groups_1,
                   fetching_policy_type _fetching_policy) : template_base::parameters_type(_simd_width, _local_size_0, _local_size_1, 1), num_groups_0(_num_groups_0), num_groups_1(_num_groups_1), fetching_policy(_fetching_policy){ }

        unsigned int num_groups_0;
        unsigned int num_groups_1;
        fetching_policy_type fetching_policy;
      };

    private:
      virtual int check_invalid_impl(viennacl::ocl::device const & /*dev*/) const
      {
          if(p_.simd_width>1)
            return TEMPLATE_INVALID_SIMD_WIDTH;
          return TEMPLATE_VALID;
      }

      void core(unsigned int /*kernel_id*/, utils::kernel_generation_stream& stream, statements_container const & statements, std::vector<mapping_type> const & mappings) const
      {
//        statements_container::data_type::const_iterator sit;
//        std::vector<mapping_type>::const_iterator mit;

//        std::string init0, upper_bound0, inc0, init1, upper_bound1, inc1;
//        fetching_loop_info(p_.fetching_policy, "M", 0, stream, init0, upper_bound0, inc0);
//        fetching_loop_info(p_.fetching_policy, "N", 1, stream, init1, upper_bound1, inc1);



//        stream << "for(unsigned int i = " << init0 << "; i < " << upper_bound0 << " ; i += " << inc0 << ")" << std::endl;
//        stream << "{" << std::endl;
//        stream.inc_tab();
//        stream << "for(unsigned int j = " << init1 << "; j < " << upper_bound1 << " ; j += " << inc1 << ")" << std::endl;
//        stream << "{" << std::endl;
//        stream.inc_tab();

//        index_tuple idx("i","M","j","N");

//        //Fetches entries to registers
//        std::set<std::string>  cache;
//        for(mit = mappings.begin(), sit = statements.data().begin() ; sit != statements.data().end() ; ++sit, ++mit)
//          tree_parsing::read_write(tree_parsing::read_write_traversal::FETCH, p_.simd_width, "reg", cache,*sit, sit->root(), idx, stream, *mit, PARENT_NODE_TYPE);

//        unsigned int i = 0;
//        for(mit = mappings.begin(), sit = statements.data().begin() ; sit != statements.data().end() ; ++sit, ++mit){
//          std::string str;
//          tree_parsing::traverse(*sit, sit->root(), tree_parsing::evaluate_expression_traversal(idx, 0, str, mappings[i++]), false);
//          stream << str << ";" << std::endl;
//        }

//        //Write back
//        for(mit = mappings.begin(), sit = statements.data().begin() ; sit != statements.data().end() ; ++sit, ++mit)
//          tree_parsing::read_write(tree_parsing::read_write_traversal::WRITE_BACK, p_.simd_width, "reg", cache,*sit, sit->root(), idx, stream, *mit, LHS_NODE_TYPE);


//        stream.dec_tab();
//        stream << "}" << std::endl;
//        stream.dec_tab();
//        stream << "}" << std::endl;
      }

      void add_kernel_arguments(statements_container const & /*statements*/, std::string & arguments_string) const
      {
        arguments_string += generate_value_kernel_argument("unsigned int", "M");
        arguments_string += generate_value_kernel_argument("unsigned int", "N");
      }

      void configure_impl(vcl_size_t /*kernel_id*/, viennacl::ocl::context & /*context*/, statements_container const & statements, viennacl::ocl::kernel & k, unsigned int & n_arg)  const
      {
        k.global_work_size(0,p_.local_size_0*p_.num_groups_0);
        k.global_work_size(1,p_.local_size_1*p_.num_groups_1);

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
      matrix_axpy_template(matrix_axpy_template::parameters_type const & parameters, std::string const & kernel_prefix, binding_policy_t binding_policy = BIND_ALL_UNIQUE) : template_base(p_, kernel_prefix, binding_policy), up_to_internal_size_(false), p_(parameters){ }

      void up_to_internal_size(bool v) { up_to_internal_size_ = v; }
      matrix_axpy_template::parameters_type const & parameters() const { return p_; }

    private:
      bool up_to_internal_size_;
      matrix_axpy_template::parameters_type p_;
    };

  }

}

#endif
