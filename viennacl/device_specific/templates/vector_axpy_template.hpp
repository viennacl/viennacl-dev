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
#include <cmath>

#include "viennacl/scheduler/forwards.h"

#include "viennacl/device_specific/mapped_objects.hpp"
#include "viennacl/device_specific/tree_parsing.hpp"
#include "viennacl/device_specific/forwards.h"
#include "viennacl/device_specific/utils.hpp"

#include "viennacl/device_specific/templates/template_base.hpp"

#include "viennacl/tools/tools.hpp"

namespace viennacl
{

  namespace device_specific
  {

    class vector_axpy_template : public template_base
    {

    public:
      class parameters_type : public template_base::parameters_type
      {
      public:
        parameters_type(unsigned int _simd_width,
                   unsigned int _group_size, unsigned int _num_groups,
                   unsigned int _decomposition) : template_base::parameters_type(_simd_width, _group_size, 1, 1), num_groups(_num_groups), decomposition(_decomposition){ }


        unsigned int num_groups;
        unsigned int decomposition;
      };

    private:
      void core(unsigned int /*kernel_id*/, utils::kernel_generation_stream& stream, statements_container const & statements, std::vector<mapping_type> const & mapping) const
      {
        statements_container::data_type::const_iterator sit;
        std::vector<mapping_type>::const_iterator mit;

        stream << "for(unsigned int i = get_global_id(0) ; i < N ; i += get_global_size(0))" << std::endl;
        stream << "{" << std::endl;
        stream.inc_tab();

        //Registers already allocated
        std::set<std::string>  cache;

        //Fetch
        std::string rhs_suffix = "reg";
        std::string lhs_suffix = statements.order()==statements_container::INDEPENDENT?"tmp":rhs_suffix;

        for(mit = mapping.begin(), sit = statements.data().begin() ; sit != statements.data().end() ; ++sit, ++mit)
        {
          tree_parsing::read_write(tree_parsing::read_write_traversal::FETCH, lhs_suffix, cache, *sit, sit->root(), index_tuple("i", "N"), stream, *mit, LHS_NODE_TYPE);
          tree_parsing::read_write(tree_parsing::read_write_traversal::FETCH, rhs_suffix, cache, *sit, sit->root(), index_tuple("i", "N"), stream, *mit, RHS_NODE_TYPE);
        }

        //Generates all the expression, in order
        for(mit = mapping.begin(), sit = statements.data().begin() ; sit != statements.data().end() ; ++sit, ++mit)
          stream << tree_parsing::evaluate_expression(*sit, sit->root(), index_tuple("i", "N"), 0, *mit, PARENT_NODE_TYPE) << ";" << std::endl;

        //Write back
        for(mit = mapping.begin(), sit = statements.data().begin() ; sit != statements.data().end() ; ++sit, ++mit)
        {
          tree_parsing::read_write(tree_parsing::read_write_traversal::WRITE_BACK, lhs_suffix, cache,*sit, sit->root(), index_tuple("i", "N"), stream, *mit, LHS_NODE_TYPE);
        }

        stream.dec_tab();
        stream << "}" << std::endl;
      }

      void add_kernel_arguments(statements_container const & /*statements*/, std::string & arguments_string) const
      {
        arguments_string += generate_value_kernel_argument("unsigned int", "N");
      }


      vcl_size_t get_vector_size(scheduler::statement const & statement) const
      {
        scheduler::statement_node const & root = statement.array()[statement.root()];
        if(root.lhs.type_family==scheduler::COMPOSITE_OPERATION_FAMILY)
        {
          scheduler::statement_node lhs = statement.array()[root.lhs.node_index];
          if(lhs.op.type==scheduler::OPERATION_BINARY_MATRIX_DIAG_TYPE)
          {
            vcl_size_t size1 = up_to_internal_size_?utils::call_on_matrix(lhs.lhs, utils::internal_size1_fun()): utils::call_on_matrix(lhs.lhs, utils::size1_fun());
            vcl_size_t size2 = up_to_internal_size_?utils::call_on_matrix(lhs.lhs, utils::internal_size2_fun()): utils::call_on_matrix(lhs.lhs, utils::size2_fun());
            return std::min<vcl_size_t>(size1, size2);
          }
          throw generator_not_supported_exception("Vector AXPY : Unimplemented LHS size deduction");
        }
        return up_to_internal_size_?utils::call_on_vector(root.lhs, utils::internal_size_fun()): utils::call_on_vector(root.lhs, utils::size_fun());
      }

      void configure_impl(vcl_size_t /*kernel_id*/, viennacl::ocl::context & /*context*/, statements_container const & statements, viennacl::ocl::kernel & k, unsigned int & n_arg)  const
      {
        k.global_work_size(0,p_.local_size_0*p_.num_groups);
        k.global_work_size(1,1);
        cl_uint size = static_cast<cl_uint>(get_vector_size(statements.data().front()));
        k.arg(n_arg++, size/p_.simd_width);
      }

    public:
      vector_axpy_template(vector_axpy_template::parameters_type const & parameters, std::string const & kernel_prefix, binding_policy_t binding_policy = BIND_ALL_UNIQUE) : template_base(p_, kernel_prefix, binding_policy), up_to_internal_size_(false), p_(parameters){ }

      void up_to_internal_size(bool v) { up_to_internal_size_ = v; }
      vector_axpy_template::parameters_type const & parameters() const { return p_; }

    private:
      bool up_to_internal_size_;
      vector_axpy_template::parameters_type p_;
    };

  }

}

#endif
