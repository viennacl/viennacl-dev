#ifndef VIENNACL_DEVICE_SPECIFIC_TEMPLATES_REDUCTION_TEMPLATE_HPP
#define VIENNACL_DEVICE_SPECIFIC_TEMPLATES_REDUCTION_TEMPLATE_HPP

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


/** @file viennacl/generator/scalar_reduction.hpp
 *
 * Kernel template for the scalar reduction operation
*/

#include <vector>

#include "viennacl/backend/opencl.hpp"

#include "viennacl/scheduler/forwards.h"
#include "viennacl/device_specific/tree_parsing.hpp"
#include "viennacl/device_specific/utils.hpp"

#include "viennacl/device_specific/templates/template_base.hpp"
#include "viennacl/device_specific/templates/utils.hpp"

#include "viennacl/tools/tools.hpp"

namespace viennacl
{

  namespace device_specific
  {

    class reduction_template : public template_base
    {

    public:
      struct parameters_type : public template_base::parameters_type
      {
        parameters_type(unsigned int _simd_width,
                   unsigned int _group_size, unsigned int _num_groups,
                   fetching_policy_type _fetching_policy) : template_base::parameters_type(_simd_width, _group_size, 1, 2), num_groups(_num_groups), fetching_policy(_fetching_policy){ }

        unsigned int num_groups;
        fetching_policy_type fetching_policy;
      };

    private:

      inline void reduce_1d_local_memory(utils::kernel_generation_stream & stream, unsigned int size, std::vector<mapped_scalar_reduction*> exprs,
                                         std::string const & buf_str, std::string const & buf_value_str,
                                         std::vector<scheduler::op_element> const & rops) const
      {
        stream << "#pragma unroll" << std::endl;
        stream << "for(unsigned int stride = " << size/2 << "; stride >0 ; stride /=2)" << std::endl;
        stream << "{" << std::endl;
        stream.inc_tab();
        stream << "barrier(CLK_LOCAL_MEM_FENCE); " << std::endl;
        stream << "if(lid <  stride)" << std::endl;
        stream << "{" << std::endl;
        stream.inc_tab();

        for(unsigned int k = 0 ; k < exprs.size() ; k++)
            if(utils::is_index_reduction(exprs[k]->statement().array()[exprs[k]->root_idx()].op))
                compute_index_reduction(stream, exprs[k]->process(buf_str+"[lid]"), exprs[k]->process(buf_str+"[lid+stride]")
                                    , exprs[k]->process(buf_value_str+"[lid]"), exprs[k]->process(buf_value_str+"[lid+stride]"),
                                    rops[k]);
            else
                compute_reduction(stream, exprs[k]->process(buf_str+"[lid]"), exprs[k]->process(buf_str+"[lid+stride]"), rops[k]);
        stream.dec_tab();
        stream << "}" << std::endl;
        stream.dec_tab();
        stream << "}" << std::endl;
      }

      virtual int check_invalid_impl(viennacl::ocl::device const & /*dev*/) const
      {
          if(p_.fetching_policy==FETCH_FROM_LOCAL)
            return TEMPLATE_INVALID_FETCHING_POLICY_TYPE;
          return TEMPLATE_VALID;
      }

      unsigned int n_lmem_elements() const
      {
        return p_.local_size_0;
      }

      static bool is_reduction(scheduler::statement_node const & node)
      {
        return node.op.type_family==scheduler::OPERATION_VECTOR_REDUCTION_TYPE_FAMILY
            || node.op.type==scheduler::OPERATION_BINARY_INNER_PROD_TYPE;
      }

      void configure_impl(vcl_size_t kernel_id, viennacl::ocl::context & context, statements_container const & statements, viennacl::ocl::kernel & kernel, unsigned int & n_arg)  const
      {
        scheduler::statement_node_numeric_type numeric_type = lhs_most(statements.data().front()).numeric_type;

        //configure ND range
        if(kernel_id==0)
        {
          kernel.global_work_size(0,p_.local_size_0*p_.num_groups);
          kernel.global_work_size(1,1);
        }
        else
        {
          kernel.global_work_size(0,p_.local_size_0);
          kernel.global_work_size(1,1);
        }

        //set arguments
        vcl_size_t size = get_vector_size(statements.data().front());
        kernel.arg(n_arg++, cl_uint(size)/p_.simd_width);

        std::vector<scheduler::statement_node const *> reductions;
        for(statements_container::data_type::const_iterator it = statements.data().begin() ; it != statements.data().end() ; ++it)
        {
          std::vector<vcl_size_t> reductions_idx;
          tree_parsing::traverse(*it, it->root(), tree_parsing::filter(&is_reduction, reductions_idx), false);
          for(std::vector<vcl_size_t>::iterator itt = reductions_idx.begin() ; itt != reductions_idx.end() ; ++itt)
            reductions.push_back(&it->array()[*itt]);
        }

        unsigned int i = 0;
        unsigned int j = 0;
        for(std::vector<scheduler::statement_node const *>::const_iterator it = reductions.begin() ; it != reductions.end() ; ++it)
        {
          if(tmp_.size() <= i)
            tmp_.push_back(context.create_memory(CL_MEM_READ_WRITE, p_.num_groups*utils::size_of(numeric_type)));
          kernel.arg(n_arg++, tmp_[i]);
          i++;

          if(utils::is_index_reduction((*it)->op))
          {
            if(tmpidx_.size() <= j)
              tmpidx_.push_back(context.create_memory(CL_MEM_READ_WRITE, p_.num_groups*4));
            kernel.arg(n_arg++, tmpidx_[j]);
            j++;
          }
        }
      }

      void add_kernel_arguments(statements_container const & statements, std::string & arguments_string) const{
        arguments_string += generate_value_kernel_argument("unsigned int", "N");

        scheduler::statement_node_numeric_type numeric_type = lhs_most(statements.data().front()).numeric_type;
        std::string numeric_type_string = utils::numeric_type_to_string(numeric_type);
        std::vector<scheduler::statement_node const *> reductions;
        for(statements_container::data_type::const_iterator it = statements.data().begin() ; it != statements.data().end() ; ++it)
        {
          std::vector<vcl_size_t> reductions_idx;
          tree_parsing::traverse(*it, it->root(), tree_parsing::filter(&is_reduction, reductions_idx), false);
          for(std::vector<vcl_size_t>::iterator itt = reductions_idx.begin() ; itt != reductions_idx.end() ; ++itt)
            reductions.push_back(&it->array()[*itt]);
        }

        for(std::vector<scheduler::statement_node const *>::iterator it = reductions.begin() ; it != reductions.end() ; ++it)
        {
          std::string strk = tools::to_string(std::distance(reductions.begin(), it));
          if(utils::is_index_reduction((*it)->op))
          {
            arguments_string += generate_pointer_kernel_argument("__global", numeric_type_string,  "temp" + strk + "_value");
            arguments_string += generate_pointer_kernel_argument("__global", "unsigned int",  "temp" + strk);
          }
          else
          {
            arguments_string += generate_pointer_kernel_argument("__global", numeric_type_string,  "temp" + strk);
          }
        }
      }

      void core_0(utils::kernel_generation_stream& stream, std::vector<mapped_scalar_reduction*> exprs, statements_container const & statements, std::vector<mapping_type> const & mappings) const
      {
        scheduler::statement_node_numeric_type numeric_type = lhs_most(statements.data().front()).numeric_type;
        statements_container::data_type::const_iterator sit;
        std::vector<mapping_type>::const_iterator mit;
        std::set<std::string> already_fetched;
        std::size_t N = exprs.size();
        std::vector<scheduler::op_element> rops(N);

        for(mit = mappings.begin(), sit = statements.data().begin() ; sit != statements.data().end() ; ++mit, ++sit)
          tree_parsing::process(sit->root(),PARENT_NODE_TYPE, *sit, "scalar", "#scalartype #namereg = *#name;", stream, *mit, &already_fetched);

        for(unsigned int k = 0 ; k < N ; ++k)
        {
          scheduler::op_element root_op = exprs[k]->statement().array()[exprs[k]->root_idx()].op;
          rops[k].type_family = scheduler::OPERATION_BINARY_TYPE_FAMILY;
          if(root_op.type==scheduler::OPERATION_BINARY_INNER_PROD_TYPE)
            rops[k].type        = scheduler::OPERATION_BINARY_ADD_TYPE;
          else
            rops[k].type        = root_op.type;
        }

        stream << "unsigned int lid = get_local_id(0);" << std::endl;

        for(unsigned int k = 0 ; k < N ; ++k)
        {
          if(utils::is_index_reduction(exprs[k]->statement().array()[exprs[k]->root_idx()].op))
          {
            stream << exprs[k]->process("__local #scalartype #name_buf_value[" + tools::to_string(p_.local_size_0) + "];") << std::endl;
            stream << exprs[k]->process("#scalartype #name_acc_value = " + neutral_element(rops[k]) + ";") << std::endl;
            stream << exprs[k]->process("__local unsigned int #name_buf[" + tools::to_string(p_.local_size_0) + "];") << std::endl;
            stream << exprs[k]->process("unsigned int #name_acc = 0;") << std::endl;
          }
          else
          {
            stream << exprs[k]->process("__local #scalartype #name_buf[" + tools::to_string(p_.local_size_0) + "];") << std::endl;
            stream << exprs[k]->process("#scalartype #name_acc = " + neutral_element(rops[k]) + ";") << std::endl;
          }
        }

        std::string init, upper_bound, inc;
        fetching_loop_info(p_.fetching_policy, "N", 0, stream, init, upper_bound, inc);
        stream << "for(unsigned int i = " << init << "; i < " << upper_bound << " ; i += " << inc << ")" << std::endl;
        stream << "{" << std::endl;
        stream.inc_tab();

        //Fetch vector entry
        for(std::vector<mapped_scalar_reduction*>::iterator it = exprs.begin() ; it != exprs.end() ; ++it)
          tree_parsing::process((*it)->root_idx(), PARENT_NODE_TYPE, (*it)->statement(), "vector", "#scalartype #namereg = #name[#start + i*#stride];", stream, (*it)->mapping(), &already_fetched);

        //Update accumulators
        for(unsigned int k = 0 ; k < N ; ++k)
        {
          for(unsigned int a = 0 ; a < p_.simd_width ; ++a)
          {
            std::map<std::string, std::string> accessors;
            std::string str = "#namereg";
            if(p_.simd_width > 1)
              str += ".s" + tools::to_string(a);
            accessors["vector"] = "#namereg";
            accessors["scalar"] = "#namereg";
            std::string value = tree_parsing::evaluate_expression(exprs[k]->statement(), exprs[k]->root_idx(), accessors, exprs[k]->mapping(), LHS_NODE_TYPE);
            if(exprs[k]->statement().array()[exprs[k]->root_idx()].op.type==scheduler::OPERATION_BINARY_INNER_PROD_TYPE)
              value+= "*" + tree_parsing::evaluate_expression(exprs[k]->statement(), exprs[k]->root_idx(), accessors, exprs[k]->mapping(), RHS_NODE_TYPE);

            if(utils::is_index_reduction(exprs[k]->statement().array()[exprs[k]->root_idx()].op))
              compute_index_reduction(stream, exprs[k]->process("#name_acc"), "i", exprs[k]->process("#name_acc_value"), value,rops[k]);
            else
              compute_reduction(stream, exprs[k]->process("#name_acc"), value,rops[k]);
          }
        }

        stream.dec_tab();
        stream << "}" << std::endl;

        //Fills local memory
        for(unsigned int k = 0 ; k < N ; ++k)
        {
          if(utils::is_index_reduction(exprs[k]->statement().array()[exprs[k]->root_idx()].op))
            stream << exprs[k]->process("#name_buf_value[lid] = #name_acc_value;") << std::endl;
          stream << exprs[k]->process("#name_buf[lid] = #name_acc;") << std::endl;
        }

        //Reduce local memory
        reduce_1d_local_memory(stream, p_.local_size_0, exprs, "#name_buf", "#name_buf_value", rops);

        //Write to temporary buffers
        stream << "if(lid==0)" << std::endl;
        stream << "{" << std::endl;
        stream.inc_tab();
        for(unsigned int k = 0 ; k < N ; ++k)
        {
          std::string strk = tools::to_string(k);
          if(utils::is_index_reduction(exprs[k]->statement().array()[exprs[k]->root_idx()].op))
            stream << exprs[k]->process("temp"+strk+"_value[get_group_id(0)] = #name_buf_value[0];") << std::endl;
          stream << exprs[k]->process("temp"+strk+"[get_group_id(0)] = #name_buf[0];") << std::endl;
        }
        stream.dec_tab();
        stream << "}" << std::endl;
      }


      void core_1(utils::kernel_generation_stream& stream, std::vector<mapped_scalar_reduction*> exprs, statements_container const & statements, std::vector<mapping_type> const & mapping) const
      {
        statements_container::data_type::const_iterator sit;
        std::vector<mapping_type>::const_iterator mit;

        std::size_t N = exprs.size();
        std::vector<scheduler::op_element> rops(N);

        for(unsigned int k = 0 ; k < N ; ++k)
        {
          scheduler::op_element root_op = exprs[k]->statement().array()[exprs[k]->root_idx()].op;
          rops[k].type_family = scheduler::OPERATION_BINARY_TYPE_FAMILY;
          if(root_op.type==scheduler::OPERATION_BINARY_INNER_PROD_TYPE)
            rops[k].type        = scheduler::OPERATION_BINARY_ADD_TYPE;
          else
            rops[k].type        = root_op.type;
        }

        stream << "unsigned int lid = get_local_id(0);" << std::endl;

        for(unsigned int k = 0 ; k < N ; ++k)
        {
          if(utils::is_index_reduction(exprs[k]->statement().array()[exprs[k]->root_idx()].op))
          {
            stream << exprs[k]->process("__local unsigned int #name_buf[" + tools::to_string(p_.local_size_0) + "];");
            stream << exprs[k]->process("unsigned int #name_acc = 0;") << std::endl;
            stream << exprs[k]->process("__local #scalartype #name_buf_value[" + tools::to_string(p_.local_size_0) + "];") << std::endl;
            stream << exprs[k]->process("#scalartype #name_acc_value = " + neutral_element(rops[k]) + ";");
          }
          else
          {
            stream << exprs[k]->process("__local #scalartype #name_buf[" + tools::to_string(p_.local_size_0) + "];") << std::endl;
            stream << exprs[k]->process("#scalartype #name_acc = " + neutral_element(rops[k]) + ";");
          }
        }

        stream << "for(unsigned int i = lid ; i < " << p_.num_groups << " ; i += get_local_size(0))" << std::endl;
        stream << "{" << std::endl;
        stream.inc_tab();
        for(unsigned int k = 0 ; k < N ; ++k)
          if(utils::is_index_reduction(exprs[k]->statement().array()[exprs[k]->root_idx()].op))
            compute_index_reduction(stream, exprs[k]->process("#name_acc"), "temp"+tools::to_string(k)+"[i]", exprs[k]->process("#name_acc_value"), "temp"+tools::to_string(k)+"_value[i]",rops[k]);
          else
            compute_reduction(stream, exprs[k]->process("#name_acc"), "temp"+tools::to_string(k)+"[i]", rops[k]);

        stream.dec_tab();
        stream << "}" << std::endl;

        for(unsigned int k = 0 ; k < N ; ++k)
        {
          if(utils::is_index_reduction(exprs[k]->statement().array()[exprs[k]->root_idx()].op))
            stream << exprs[k]->process("#name_buf_value[lid] = #name_acc_value;") << std::endl;
          stream << exprs[k]->process("#name_buf[lid] = #name_acc;") << std::endl;
        }


        //Reduce and write final result
         reduce_1d_local_memory(stream, p_.local_size_0, exprs, "#name_buf", "#name_buf_value", rops);

        stream << "if(lid==0)" << std::endl;
        stream << "{" << std::endl;
        stream.inc_tab();
        //Generates all the expression, in order
        for(mit = mapping.begin(), sit = statements.data().begin() ; sit != statements.data().end() ; ++sit, ++mit)
        {
          std::map<std::string, std::string> accessors;
          accessors["scalar_reduction"] = "#name_buf[0]";
          accessors["scalar"] = "*#name";
          accessors["vectors"] = "#name[#start]";
          stream << tree_parsing::evaluate_expression(*sit, sit->root(), accessors, *mit, PARENT_NODE_TYPE) << ";" << std::endl;
        }
        stream.dec_tab();
        stream << "}" << std::endl;
      }

      vcl_size_t get_vector_size(viennacl::scheduler::statement const & s) const
      {
        scheduler::statement::container_type exprs = s.array();
        for(scheduler::statement::container_type::iterator it = exprs.begin() ; it != exprs.end() ; ++it)
        {
          if(is_scalar_reduction(*it))
          {
            scheduler::statement_node const * current_node = &(*it);
            //The LHS of the prod is a vector
            while(current_node->lhs.type_family!=scheduler::VECTOR_TYPE_FAMILY)
              current_node = &exprs[current_node->lhs.node_index];
            return utils::call_on_vector(current_node->lhs, utils::internal_size_fun());
          }
        }
        throw generator_not_supported_exception("unexpected expression tree");
      }

      void core(unsigned int kernel_id, utils::kernel_generation_stream& stream, statements_container const & statements, std::vector<mapping_type> const & mapping) const
      {
        std::vector<mapped_scalar_reduction*> exprs;
        for(std::vector<mapping_type>::const_iterator it = mapping.begin() ; it != mapping.end() ; ++it)
          for(mapping_type::const_iterator iit = it->begin() ; iit != it->end() ; ++iit)
            if(mapped_scalar_reduction * p = dynamic_cast<mapped_scalar_reduction*>(iit->second.get()))
              exprs.push_back(p);

        if(kernel_id==0)
          core_0(stream,exprs,statements,mapping);
        else
          core_1(stream,exprs,statements,mapping);
      }


    public:
      reduction_template(reduction_template::parameters_type const & parameters, std::string const & kernel_prefix, binding_policy_t binding_policy = BIND_ALL_UNIQUE) : template_base(p_, kernel_prefix, binding_policy), p_(parameters){ }
      reduction_template::parameters_type const & parameters() const { return p_; }

    private:
      reduction_template::parameters_type p_;
      mutable std::vector< viennacl::ocl::handle<cl_mem> > tmp_;
      mutable std::vector< viennacl::ocl::handle<cl_mem> > tmpidx_;
    };

  }

}

#endif
