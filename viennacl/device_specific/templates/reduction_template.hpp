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
          if(optimized_parameters_.fetching_policy==FETCH_FROM_LOCAL)
            return TEMPLATE_INVALID_FETCHING_POLICY_TYPE;
          return TEMPLATE_VALID;
      }

      unsigned int n_lmem_elements() const  { return optimized_parameters_.local_size_0; }

      static bool is_reduction(scheduler::statement_node const & node) { return utils::is_reduction(node.op); }

      void generate_impl(utils::kernel_generation_stream& stream, statements_container const & statements, std::vector<mapping_type> const & mappings, bool fallback) const
      {
        std::vector<mapped_scalar_reduction*> exprs;
        statements_container::data_type::const_iterator sit;
        std::vector<mapping_type>::const_iterator mit;
        std::set<std::string> already_fetched;

        for(std::vector<mapping_type>::const_iterator it = mappings.begin() ; it != mappings.end() ; ++it)
          for(mapping_type::const_iterator iit = it->begin() ; iit != it->end() ; ++iit)
            if(mapped_scalar_reduction * p = dynamic_cast<mapped_scalar_reduction*>(iit->second.get()))
              exprs.push_back(p);


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

        std::string arguments = generate_value_kernel_argument("unsigned int", "N");
        for(unsigned int k = 0 ; k < N ; ++k)
        {
          std::string numeric_type = utils::numeric_type_to_string(lhs_most(exprs[k]->statement().array(),
                                                                            exprs[k]->statement().root()).lhs.numeric_type);
          if(utils::is_index_reduction(exprs[k]->statement().array()[exprs[k]->root_idx()].op))
          {
            arguments += generate_pointer_kernel_argument("__global", "unsigned int",  exprs[k]->process("#name_temp"));
            arguments += generate_pointer_kernel_argument("__global", numeric_type,  exprs[k]->process("#name_temp_value"));
          }
          else
           arguments += generate_pointer_kernel_argument("__global", numeric_type,  exprs[k]->process("#name_temp"));
        }


        /* ------------------------
         * First Kernel
         * -----------------------*/
        stream << " __attribute__((reqd_work_group_size(" << optimized_parameters_.local_size_0 << ",1,1)))" << std::endl;
        generate_prototype(stream, kernel_prefix_ + "_0", arguments, mappings, statements);
        stream << "{" << std::endl;
        stream.inc_tab();
        for(mit = mappings.begin(), sit = statements.data().begin() ; sit != statements.data().end() ; ++mit, ++sit)
          tree_parsing::process(sit->root(),PARENT_NODE_TYPE, *sit, "scalar", "#scalartype #namereg = *#name;", stream, *mit, &already_fetched);

        stream << "unsigned int lid = get_local_id(0);" << std::endl;

        for(unsigned int k = 0 ; k < N ; ++k)
        {
          if(utils::is_index_reduction(exprs[k]->statement().array()[exprs[k]->root_idx()].op))
          {
            stream << exprs[k]->process("__local #scalartype #name_buf_value[" + tools::to_string(optimized_parameters_.local_size_0) + "];") << std::endl;
            stream << exprs[k]->process("#scalartype #name_acc_value = " + neutral_element(rops[k]) + ";") << std::endl;
            stream << exprs[k]->process("__local unsigned int #name_buf[" + tools::to_string(optimized_parameters_.local_size_0) + "];") << std::endl;
            stream << exprs[k]->process("unsigned int #name_acc = 0;") << std::endl;
          }
          else
          {
            stream << exprs[k]->process("__local #scalartype #name_buf[" + tools::to_string(optimized_parameters_.local_size_0) + "];") << std::endl;
            stream << exprs[k]->process("#scalartype #name_acc = " + neutral_element(rops[k]) + ";") << std::endl;
          }
        }

        std::string init, upper_bound, inc;
        fetching_loop_info(optimized_parameters_.fetching_policy, "N", 0, stream, init, upper_bound, inc);
        stream << "for(unsigned int i = " << init << "; i < " << upper_bound << " ; i += " << inc << ")" << std::endl;
        stream << "{" << std::endl;
        stream.inc_tab();

        //Fetch vector entry
        for(std::vector<mapped_scalar_reduction*>::iterator it = exprs.begin() ; it != exprs.end() ; ++it)
          tree_parsing::process((*it)->root_idx(), PARENT_NODE_TYPE, (*it)->statement(), "vector", "#scalartype #namereg = #name[#start + i*#stride];", stream, (*it)->mapping(), &already_fetched);

        //Update accumulators
        for(unsigned int k = 0 ; k < N ; ++k)
        {
          for(unsigned int a = 0 ; a < optimized_parameters_.simd_width ; ++a)
          {
            std::map<std::string, std::string> accessors;
            std::string str = "#namereg";
            if(optimized_parameters_.simd_width > 1)
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
        reduce_1d_local_memory(stream, optimized_parameters_.local_size_0, exprs, "#name_buf", "#name_buf_value", rops);

        //Write to temporary buffers
        stream << "if(lid==0)" << std::endl;
        stream << "{" << std::endl;
        stream.inc_tab();
        for(unsigned int k = 0 ; k < N ; ++k)
        {
          if(utils::is_index_reduction(exprs[k]->statement().array()[exprs[k]->root_idx()].op))
            stream << exprs[k]->process("#name_temp_value[get_group_id(0)] = #name_buf_value[0];") << std::endl;
          stream << exprs[k]->process("#name_temp[get_group_id(0)] = #name_buf[0];") << std::endl;
        }
        stream.dec_tab();
        stream << "}" << std::endl;

        stream.dec_tab();
        stream << "}" << std::endl;

        /* ------------------------
         * Second kernel
         * -----------------------*/
        stream << " __attribute__((reqd_work_group_size(" << optimized_parameters_.local_size_0 << ",1,1)))" << std::endl;
        generate_prototype(stream, kernel_prefix_ + "_1", arguments, mappings, statements);
        stream << "{" << std::endl;
        stream.inc_tab();

        stream << "unsigned int lid = get_local_id(0);" << std::endl;

        for(unsigned int k = 0 ; k < N ; ++k)
        {
          if(utils::is_index_reduction(exprs[k]->statement().array()[exprs[k]->root_idx()].op))
          {
            stream << exprs[k]->process("__local unsigned int #name_buf[" + tools::to_string(optimized_parameters_.local_size_0) + "];");
            stream << exprs[k]->process("unsigned int #name_acc = 0;") << std::endl;
            stream << exprs[k]->process("__local #scalartype #name_buf_value[" + tools::to_string(optimized_parameters_.local_size_0) + "];") << std::endl;
            stream << exprs[k]->process("#scalartype #name_acc_value = " + neutral_element(rops[k]) + ";");
          }
          else
          {
            stream << exprs[k]->process("__local #scalartype #name_buf[" + tools::to_string(optimized_parameters_.local_size_0) + "];") << std::endl;
            stream << exprs[k]->process("#scalartype #name_acc = " + neutral_element(rops[k]) + ";");
          }
        }

        stream << "for(unsigned int i = lid ; i < " << optimized_parameters_.num_groups << " ; i += get_local_size(0))" << std::endl;
        stream << "{" << std::endl;
        stream.inc_tab();
        for(unsigned int k = 0 ; k < N ; ++k)
          if(utils::is_index_reduction(exprs[k]->statement().array()[exprs[k]->root_idx()].op))
            compute_index_reduction(stream, exprs[k]->process("#name_acc"), exprs[k]->process("#name_temp[i]"),
                                            exprs[k]->process("#name_acc_value"),exprs[k]->process("#name_temp_value[i]"),rops[k]);
          else
            compute_reduction(stream, exprs[k]->process("#name_acc"), exprs[k]->process("#name_temp[i]"), rops[k]);

        stream.dec_tab();
        stream << "}" << std::endl;

        for(unsigned int k = 0 ; k < N ; ++k)
        {
          if(utils::is_index_reduction(exprs[k]->statement().array()[exprs[k]->root_idx()].op))
            stream << exprs[k]->process("#name_buf_value[lid] = #name_acc_value;") << std::endl;
          stream << exprs[k]->process("#name_buf[lid] = #name_acc;") << std::endl;
        }


        //Reduce and write final result
         reduce_1d_local_memory(stream, optimized_parameters_.local_size_0, exprs, "#name_buf", "#name_buf_value", rops);

        stream << "if(lid==0)" << std::endl;
        stream << "{" << std::endl;
        stream.inc_tab();
        //Generates all the expression, in order
        for(mit = mappings.begin(), sit = statements.data().begin() ; sit != statements.data().end() ; ++sit, ++mit)
        {
          std::map<std::string, std::string> accessors;
          accessors["scalar_reduction"] = "#name_buf[0]";
          accessors["scalar"] = "*#name";
          accessors["vectors"] = "#name[#start]";
          stream << tree_parsing::evaluate_expression(*sit, sit->root(), accessors, *mit, PARENT_NODE_TYPE) << ";" << std::endl;
        }
        stream.dec_tab();
        stream << "}" << std::endl;

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

    public:
      reduction_template(reduction_template::parameters_type const & parameters, std::string const & kernel_prefix, binding_policy_t binding_policy = BIND_ALL_UNIQUE) : template_base(optimized_parameters_, kernel_prefix, binding_policy), optimized_parameters_(parameters){ }
      reduction_template::parameters_type const & parameters() const { return optimized_parameters_; }

      void enqueue(viennacl::ocl::program & program, statements_container const & statements)
      {
        std::vector<scheduler::statement_node const *> reductions;
        cl_uint size;
        for(statements_container::data_type::const_iterator it = statements.data().begin() ; it != statements.data().end() ; ++it)
        {
          std::vector<size_t> reductions_idx;
          tree_parsing::traverse(*it, it->root(), tree_parsing::filter(&is_reduction, reductions_idx), false);
          size = static_cast<cl_uint>(vector_size(lhs_most(it->array(), reductions_idx[0]), false));
          for(std::vector<size_t>::iterator itt = reductions_idx.begin() ; itt != reductions_idx.end() ; ++itt)
            reductions.push_back(&it->array()[*itt]);
        }

        scheduler::statement const & statement = statements.data().front();
        unsigned int scalartype_size = utils::size_of(lhs_most(statement.array(), statement.root()).lhs.numeric_type);


        viennacl::ocl::kernel* kernels[2] = {&program.get_kernel(kernel_prefix_+"_0"), &program.get_kernel(kernel_prefix_+"_1")};

        kernels[0]->local_work_size(0, optimized_parameters_.local_size_0);
        kernels[0]->global_work_size(0,optimized_parameters_.local_size_0*optimized_parameters_.num_groups);

        kernels[1]->local_work_size(0, optimized_parameters_.local_size_0);
        kernels[1]->global_work_size(0,optimized_parameters_.local_size_0);

        for(unsigned int k = 0 ; k < 2 ; k++)
        {
            unsigned int n_arg = 0;
            kernels[k]->arg(n_arg++, size/optimized_parameters_.simd_width);
            unsigned int i = 0;
            unsigned int j = 0;
            for(std::vector<scheduler::statement_node const *>::const_iterator it = reductions.begin() ; it != reductions.end() ; ++it)
            {
              if(tmp_.size() <= i)
                tmp_.push_back(kernels[k]->context().create_memory(CL_MEM_READ_WRITE, optimized_parameters_.num_groups*scalartype_size));
              kernels[k]->arg(n_arg++, tmp_[i]);

              if(utils::is_index_reduction((*it)->op))
              {
                if(tmpidx_.size() <= j)
                  tmpidx_.push_back(kernels[k]->context().create_memory(CL_MEM_READ_WRITE, optimized_parameters_.num_groups*4));
                kernels[k]->arg(n_arg++, tmpidx_[j]);
              }
            }
            set_arguments(statements, *kernels[k], n_arg);
        }

        for(unsigned int k = 0 ; k < 2 ; k++)
            viennacl::ocl::enqueue(*kernels[k]);
      }

      void enqueue_fallback(viennacl::ocl::program & program_optimized, viennacl::ocl::program & program_fallback, statements_container const & statements)
      {

      }

    private:
      reduction_template::parameters_type optimized_parameters_;
      std::vector< viennacl::ocl::handle<cl_mem> > tmp_;
      std::vector< viennacl::ocl::handle<cl_mem> > tmpidx_;
    };

  }

}

#endif
