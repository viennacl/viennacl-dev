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
#include "viennacl/device_specific/templates/utils.hpp"

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
                   fetching_policy_type _fetching_policy) : template_base::parameters_type(_simd_width, _group_size, 1, 1), num_groups(_num_groups), fetching_policy(_fetching_policy){ }



        unsigned int num_groups;
        fetching_policy_type fetching_policy;
      };

    private:
      virtual int check_invalid_impl(viennacl::ocl::device const & /*dev*/) const
      {
          if(p_.fetching_policy==FETCH_FROM_LOCAL)
            return TEMPLATE_INVALID_FETCHING_POLICY_TYPE;
          return TEMPLATE_VALID;
      }

      static bool has_strided_access(statements_container const & statements)
      {
        for(statements_container::data_type::const_iterator it = statements.data().begin() ; it != statements.data().end() ; ++it)
        {
          //checks for vectors
          std::vector<scheduler::lhs_rhs_element> vectors;
          tree_parsing::traverse(*it, it->root(), tree_parsing::filter_elements(scheduler::DENSE_VECTOR_TYPE, vectors), false);
          for(std::vector<scheduler::lhs_rhs_element>::iterator itt = vectors.begin() ; itt != vectors.end() ; ++itt)
            if(utils::call_on_vector(*itt, utils::stride_fun())>1)
              return true;
        }
        return false;
      }

      struct loop_body
      {
        loop_body(statements_container const & _statements, std::vector<mapping_type> const & _mappings) : statements(_statements), mappings(_mappings) { }

        void operator()(utils::kernel_generation_stream & stream, unsigned int simd_width) const
        {
          std::string pw = tools::to_string(simd_width);
          std::set<std::string> already_fetched;
          statements_container::data_type::const_iterator sit;
          std::vector<mapping_type>::const_iterator mit;

          for(mit = mappings.begin(), sit = statements.data().begin() ; sit != statements.data().end() ; ++mit, ++sit)
            if(simd_width==1)
              tree_parsing::process(sit->root(),PARENT_NODE_TYPE, *sit, "vector", "#scalartype #namereg = vload(i*#stride, #name);", stream, *mit, &already_fetched);
            else
              tree_parsing::process(sit->root(),PARENT_NODE_TYPE, *sit, "vector", "#scalartype" + pw + " #namereg = vload" + pw + "(i, #name);", stream, *mit, &already_fetched);

          //Generates all the expression, in order
          for(mit = mappings.begin(), sit = statements.data().begin() ; sit != statements.data().end() ; ++sit, ++mit)
          {
            std::map<std::string, std::string> accessors;
            accessors["vector"] = "#namereg";
            accessors["scalar"] = "#namereg";
            stream << tree_parsing::evaluate_expression(*sit, sit->root(), accessors, *mit, PARENT_NODE_TYPE) << ";" << std::endl;
          }

          for(mit = mappings.begin(), sit = statements.data().begin() ; sit != statements.data().end() ; ++mit, ++sit)
            if(simd_width==1)
              tree_parsing::process(sit->root(),LHS_NODE_TYPE, *sit, "vector","vstore(#namereg, i*#stride, #name);", stream, *mit, NULL);
            else
              tree_parsing::process(sit->root(),LHS_NODE_TYPE, *sit, "vector","vstore" + pw +"(#namereg, i, #name);", stream, *mit, NULL);

        }

      private:
        statements_container const & statements;
        std::vector<mapping_type> const & mappings;
      };

      void generate_impl(utils::kernel_generation_stream& stream, std::string const & kernel_prefix, statements_container const & statements, std::vector<mapping_type> const & mappings, bool fallback) const
      {
        statements_container::data_type::const_iterator sit;
        std::vector<mapping_type>::const_iterator mit;

        stream << " __attribute__((reqd_work_group_size(" << p_.local_size_0 << ",1,1)))" << std::endl;
        generate_prototype(stream, kernel_prefix, "unsigned int N,", mappings, statements);
        stream << "{" << std::endl;
        stream.inc_tab();

        //Fetch the pointed scalars
        std::set<std::string> already_fetched;
        for(mit = mappings.begin(), sit = statements.data().begin() ; sit != statements.data().end() ; ++mit, ++sit)
          tree_parsing::process(sit->root(),PARENT_NODE_TYPE, *sit, "scalar", "#scalartype #namereg = *#name;", stream, *mit, &already_fetched);

        //Increment vector offsets
        for(mit = mappings.begin(), sit = statements.data().begin() ; sit != statements.data().end() ; ++mit, ++sit)
          tree_parsing::process(sit->root(),PARENT_NODE_TYPE, *sit, "vector", "#name += #start;", stream, *mit, NULL);


        element_wise_loop_1D(stream, loop_body(statements, mappings), p_.fetching_policy, fallback?1:p_.simd_width, "i", "N");

        stream.dec_tab();
        stream << "}" << std::endl;
      }

      bool requires_fallback(statements_container const & statements) const
      {
        return has_strided_access(statements) && p_.simd_width > 1;
      }

    public:
      vector_axpy_template(vector_axpy_template::parameters_type const & parameters, binding_policy_t binding_policy = BIND_ALL_UNIQUE) : template_base(p_, binding_policy), up_to_internal_size_(false), p_(parameters){ }

      void up_to_internal_size(bool v) { up_to_internal_size_ = v; }
      vector_axpy_template::parameters_type const & parameters() const { return p_; }

      void enqueue(std::string const & kernel_prefix, lazy_program_compiler & program_fallback, lazy_program_compiler & program_optimized,  statements_container const & statements)
      {
        viennacl::ocl::program & program = requires_fallback(statements)?program_fallback.program():program_optimized.program();
        viennacl::ocl::kernel & kernel = program.get_kernel(kernel_prefix);
        kernel.local_work_size(0, p_.local_size_0);
        kernel.global_work_size(0, p_.local_size_0*p_.num_groups);
        unsigned int current_arg = 0;
        scheduler::statement const & statement = statements.data().front();
        cl_uint size = static_cast<cl_uint>(vector_size(lhs_most(statement.array(), statement.root()), up_to_internal_size_));
        kernel.arg(current_arg++, size);
        set_arguments(statements, kernel, current_arg);
        viennacl::ocl::enqueue(kernel);
      }

    private:
      bool up_to_internal_size_;
      vector_axpy_template::parameters_type p_;
    };

  }

}

#endif
