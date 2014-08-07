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
          if(optimized_parameters_.fetching_policy==FETCH_FROM_LOCAL)
            return TEMPLATE_INVALID_FETCHING_POLICY_TYPE;
          return TEMPLATE_VALID;
      }

      void loop_body(utils::kernel_generation_stream & stream, unsigned int simd_width, statements_container const & statements, std::vector<mapping_type> const & mappings) const
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

      void generate_impl(utils::kernel_generation_stream& stream, statements_container const & statements, std::vector<mapping_type> const & mappings, bool fallback) const
      {
        statements_container::data_type::const_iterator sit;
        std::vector<mapping_type>::const_iterator mit;
        std::string pw = tools::to_string(optimized_parameters_.simd_width);

        stream << " __attribute__((reqd_work_group_size(" << optimized_parameters_.local_size_0 << ",1,1)))" << std::endl;
        generate_prototype(stream, kernel_prefix_, "unsigned int N,", mappings, statements);
        stream << "{" << std::endl;
        stream.inc_tab();

        stream << "//Fetch the pointed scalars" << std::endl;
        std::set<std::string> already_fetched;
        for(mit = mappings.begin(), sit = statements.data().begin() ; sit != statements.data().end() ; ++mit, ++sit)
          tree_parsing::process(sit->root(),PARENT_NODE_TYPE, *sit, "scalar", "#scalartype #namereg = *#name;", stream, *mit, &already_fetched);

        stream << "//Increment vector offsets" << std::endl;
        for(mit = mappings.begin(), sit = statements.data().begin() ; sit != statements.data().end() ; ++mit, ++sit)
          tree_parsing::process(sit->root(),PARENT_NODE_TYPE, *sit, "vector", "#name += #start;", stream, *mit, NULL);

        std::string init, upper_bound, inc;
        fetching_loop_info(optimized_parameters_.fetching_policy, "N/"+pw, 0, stream, init, upper_bound, inc);

        stream << "for(unsigned int i = " << init << "; i < " << upper_bound << " ; i += " << inc << ")" << std::endl;
        stream << "{" << std::endl;
        stream.inc_tab();
        loop_body(stream, optimized_parameters_.simd_width, statements, mappings);
        stream.dec_tab();
        stream << "}" << std::endl;

        if(optimized_parameters_.simd_width>1)
        {
          stream << "for(unsigned int i = N/" << pw << "*" << pw << " + get_global_id(0) ; i < N; i += get_global_size(0))" << std::endl;
          stream << "{" << std::endl;
          stream.inc_tab();
          loop_body(stream, 1, statements, mappings);
          stream.dec_tab();
          stream << "}" << std::endl;
        }



        stream.dec_tab();
        stream << "}" << std::endl;
      }

    public:
      vector_axpy_template(vector_axpy_template::parameters_type const & parameters, std::string const & kernel_prefix, binding_policy_t binding_policy = BIND_ALL_UNIQUE) : template_base(optimized_parameters_, kernel_prefix, binding_policy), up_to_internal_size_(false), optimized_parameters_(parameters){ }

      void up_to_internal_size(bool v) { up_to_internal_size_ = v; }
      vector_axpy_template::parameters_type const & parameters() const { return optimized_parameters_; }

      virtual bool requires_fallback(statements_container const & statements) const
      {
        return false;
      }

      void enqueue(viennacl::ocl::program & program, statements_container const & statements)
      {
        viennacl::ocl::kernel & kernel = program.get_kernel(kernel_prefix_);
        kernel.local_work_size(0, optimized_parameters_.local_size_0);
        kernel.global_work_size(0, optimized_parameters_.local_size_0*optimized_parameters_.num_groups);
        unsigned int current_arg = 0;
        scheduler::statement const & statement = statements.data().front();
        cl_uint size = static_cast<cl_uint>(vector_size(lhs_most(statement.array(), statement.root()), up_to_internal_size_));
        kernel.arg(current_arg++, size);
        set_arguments(statements, kernel, current_arg);
        viennacl::ocl::enqueue(kernel);
      }

      void enqueue_fallback(viennacl::ocl::program & program_optimized, viennacl::ocl::program & program_fallback, statements_container const & statements)
      {

      }

    private:
      bool up_to_internal_size_;
      vector_axpy_template::parameters_type optimized_parameters_;
    };

  }

}

#endif
