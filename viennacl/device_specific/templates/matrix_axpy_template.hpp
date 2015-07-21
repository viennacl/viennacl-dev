#ifndef VIENNACL_DEVICE_SPECIFIC_TEMPLATES_MATRIX_AXPY_HPP
#define VIENNACL_DEVICE_SPECIFIC_TEMPLATES_MATRIX_AXPY_HPP

/* =========================================================================
   Copyright (c) 2010-2015, Institute for Microelectronics,
                            Institute for Analysis and Scientific Computing,
                            TU Wien.
   Portions of this software are copyright by UChicago Argonne, LLC.

                            -----------------
                  ViennaCL - The Vienna Computing Library
                            -----------------

   Project Head:    Karl Rupp                   rupp@iue.tuwien.ac.at

   (A list of authors and contributors can be found in the manual)

   License:         MIT (X11), see file LICENSE in the base directory
============================================================================= */


/** @file viennacl/device_specific/templates/matrix_axpy_template.hpp
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

namespace viennacl
{
namespace device_specific
{

class matrix_axpy_parameters_type : public template_base::parameters_type
{
public:
  matrix_axpy_parameters_type(unsigned int _simd_width,
                              unsigned int _local_size_0, unsigned int _local_size_1,
                              unsigned int _num_groups_0, unsigned int _num_groups_1,
                              fetching_policy_type _fetching_policy) : template_base::parameters_type(_simd_width, _local_size_0, _local_size_1, 1), num_groups_0(_num_groups_0), num_groups_1(_num_groups_1), fetching_policy(_fetching_policy){ }

  unsigned int num_groups_0;
  unsigned int num_groups_1;
  fetching_policy_type fetching_policy;
};

class matrix_axpy_template : public template_base_impl<matrix_axpy_template, matrix_axpy_parameters_type>
{
private:
  int check_invalid_impl(viennacl::ocl::device const & /*dev*/) const
  {
    if (p_.simd_width>1)
      return TEMPLATE_INVALID_SIMD_WIDTH;
    return TEMPLATE_VALID;
  }

  std::string generate_impl(std::string const & kernel_prefix, statements_container const & statements, std::vector<mapping_type> const & mappings, unsigned int simd_width) const
  {
    std::string process_str;
    utils::kernel_generation_stream stream;

    std::string init0, upper_bound0, inc0, init1, upper_bound1, inc1;

    stream << " __attribute__((reqd_work_group_size(" << p_.local_size_0 << "," << p_.local_size_1 << ",1)))" << std::endl;
    generate_prototype(stream, kernel_prefix, "unsigned int M, unsigned int N,", mappings, statements);
    stream << "{" << std::endl;
    stream.inc_tab();

    tree_parsing::process(stream, PARENT_NODE_TYPE, "scalar", "#scalartype #namereg = *#pointer;", statements, mappings);
    tree_parsing::process(stream, PARENT_NODE_TYPE, "matrix", "#pointer += $OFFSET{#start1, #start2};", statements, mappings);
    tree_parsing::process(stream, PARENT_NODE_TYPE, "vector", "#pointer += #start;", statements, mappings);

    fetching_loop_info(p_.fetching_policy, "M", stream, init0, upper_bound0, inc0, "get_global_id(0)", "get_global_size(0)");
    stream << "for(unsigned int i = " << init0 << "; i < " << upper_bound0 << "; i += " << inc0 << ")" << std::endl;
    stream << "{" << std::endl;
    stream.inc_tab();
    fetching_loop_info(p_.fetching_policy, "N", stream, init1, upper_bound1, inc1, "get_global_id(1)", "get_global_size(1)");
    stream << "for(unsigned int j = " << init1 << "; j < " << upper_bound1 << "; j += " << inc1 << ")" << std::endl;
    stream << "{" << std::endl;
    stream.inc_tab();

    process_str = utils::append_width("#scalartype",simd_width) + " #namereg = " + vload(simd_width, "$OFFSET{i*#stride1,j*#stride2}", "#pointer")+ ";";
    tree_parsing::process(stream, PARENT_NODE_TYPE, "matrix", process_str, statements, mappings);
    tree_parsing::process(stream, PARENT_NODE_TYPE, "vector_diag", "#scalartype #namereg = ((i + ((#diag_offset<0)?#diag_offset:0))!=(j-((#diag_offset>0)?#diag_offset:0)))?0:#pointer[min(i*#stride, j*#stride)];", statements, mappings);


    std::map<std::string, std::string> accessors;
    accessors["matrix"] = "#namereg";
    accessors["vector_diag"] = "#namereg";
    accessors["scalar"] = "#namereg";
    tree_parsing::evaluate(stream, PARENT_NODE_TYPE, accessors, statements, mappings);

    process_str = vstore(simd_width, "#namereg", "$OFFSET{i*#stride1,j*#stride2}", "#pointer")+";";
    tree_parsing::process(stream, LHS_NODE_TYPE, "matrix", process_str, statements, mappings);

    stream.dec_tab();
    stream << "}" << std::endl;
    stream.dec_tab();
    stream << "}" << std::endl;

    stream.dec_tab();
    stream << "}" << std::endl;

    return stream.str();
  }

  std::vector<std::string> generate_impl(std::string const & kernel_prefix, statements_container const & statements, std::vector<mapping_type> const & mappings) const
  {
    std::vector<std::string> res;
    res.push_back(generate_impl(kernel_prefix, statements, mappings, 1));
    return res;
  }

public:
  matrix_axpy_template(parameters_type const & parameters, binding_policy_t binding_policy = BIND_ALL_UNIQUE) : template_base_impl<matrix_axpy_template, matrix_axpy_parameters_type>(parameters, binding_policy), up_to_internal_size_(false){ }

  void up_to_internal_size(bool v)
  {
    up_to_internal_size_ = v;
  }

  void enqueue(std::string const & kernel_prefix, std::vector<lazy_program_compiler> & programs, statements_container const & statements)
  {
    viennacl::ocl::kernel & kernel = programs[0].program().get_kernel(kernel_prefix);

    kernel.local_work_size(0, p_.local_size_0);
    kernel.local_work_size(1, p_.local_size_1);
    kernel.global_work_size(0,p_.local_size_0*p_.num_groups_0);
    kernel.global_work_size(1,p_.local_size_1*p_.num_groups_1);

    scheduler::statement_node const & root = statements.data().front().array()[statements.data().front().root()];
    unsigned int current_arg = 0;
    if (up_to_internal_size_)
    {
      kernel.arg(current_arg++, cl_uint(utils::call_on_matrix(root.lhs, utils::internal_size1_fun())));
      kernel.arg(current_arg++, cl_uint(utils::call_on_matrix(root.lhs, utils::internal_size2_fun())));
    }
    else
    {
      kernel.arg(current_arg++, cl_uint(utils::call_on_matrix(root.lhs, utils::size1_fun())));
      kernel.arg(current_arg++, cl_uint(utils::call_on_matrix(root.lhs, utils::size2_fun())));
    }

    set_arguments(statements, kernel, current_arg);

    viennacl::ocl::enqueue(kernel);
  }


private:
  bool up_to_internal_size_;
};

}
}

#endif
