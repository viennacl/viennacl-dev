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
#include "viennacl/device_specific/templates/reduction_utils.hpp"

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
          arguments_string += generate_pointer_kernel_argument("__global", numeric_type_string,  "temp" + tools::to_string(std::distance(reductions.begin(), it)));
          if(utils::is_index_reduction((*it)->op))
            arguments_string += generate_pointer_kernel_argument("__global", "unsigned int",  "temp" + tools::to_string(std::distance(reductions.begin(), it)) + "idx");
        }
      }

      void core_0(utils::kernel_generation_stream& stream, std::vector<mapped_scalar_reduction*> exprs, statements_container const & statements, std::vector<mapping_type> const & /*mapping*/) const
      {
        scheduler::statement_node_numeric_type numeric_type = lhs_most(statements.data().front()).numeric_type;
        std::string numeric_type_string = utils::numeric_type_to_string(numeric_type);

        std::size_t N = exprs.size();

        std::vector<scheduler::op_element> rops(N);
        std::vector<std::string> accs(N);
        std::vector<std::string> accsidx(N);
        std::vector<std::string> local_buffers_names(N);

        for(unsigned int k = 0 ; k < N ; ++k){
          scheduler::op_element root_op = exprs[k]->statement().array()[exprs[k]->root_idx()].op;
          rops[k].type_family = scheduler::OPERATION_BINARY_TYPE_FAMILY;
          if(root_op.type==scheduler::OPERATION_BINARY_INNER_PROD_TYPE){
            rops[k].type        = scheduler::OPERATION_BINARY_ADD_TYPE;
          }
          else{
            rops[k].type        = root_op.type;
          }
          accs[k] = "acc"+tools::to_string(k);
          accsidx[k] = accs[k] + "idx";
          local_buffers_names[k] = "buf"+tools::to_string(k);
        }

        stream << "unsigned int lid = get_local_id(0);" << std::endl;

        for(unsigned int k = 0 ; k < N ; ++k){
          stream << numeric_type_string << " " << accs[k] << " = " << neutral_element(rops[k]) << ";" << std::endl;
          if(utils::is_index_reduction(exprs[k]->statement().array()[exprs[k]->root_idx()].op))
            stream << "unsigned int " << accsidx[k] << " = " << 0 << ";" << std::endl;
        }

        std::string init, upper_bound, inc;
        fetching_loop_info(p_.fetching_policy, "N", 0, stream, init, upper_bound, inc);

        stream << "for(unsigned int i = " << init << "; i < " << upper_bound << " ; i += " << inc << "){" << std::endl;
        stream.inc_tab();
        {
          //Fetch vector entry
          std::set<std::string>  cache;
          for(std::vector<mapped_scalar_reduction*>::iterator it = exprs.begin() ; it != exprs.end() ; ++it)
          {
            tree_parsing::read_write(tree_parsing::read_write_traversal::FETCH, "reg", cache, (*it)->statement(), (*it)->root_idx(), index_tuple("i", "N"),stream,(*it)->mapping(), PARENT_NODE_TYPE);
          }
          //Update accs;
          for(unsigned int k = 0 ; k < exprs.size() ; ++k)
          {
            viennacl::scheduler::statement const & statement = exprs[k]->statement();
            vcl_size_t root_idx = exprs[k]->root_idx();
            mapping_type const & mapping = exprs[k]->mapping();
            index_tuple idx("i","N");
            if(p_.simd_width > 1){
              for(unsigned int a = 0 ; a < p_.simd_width ; ++a){
                std::string value = tree_parsing::evaluate_expression(statement,root_idx,idx,a,mapping,LHS_NODE_TYPE);
                if(statement.array()[root_idx].op.type==scheduler::OPERATION_BINARY_INNER_PROD_TYPE){
                  value += "*";
                  value += tree_parsing::evaluate_expression(statement,root_idx,idx,a,mapping,RHS_NODE_TYPE);
                }
                compute_reduction(stream,accsidx[k],"i",accs[k],value,rops[k]);
              }
            }
            else{
              std::string value = tree_parsing::evaluate_expression(statement,root_idx,idx,0,mapping,LHS_NODE_TYPE);
              if(statement.array()[root_idx].op.type==scheduler::OPERATION_BINARY_INNER_PROD_TYPE){
                value += "*";
                value += tree_parsing::evaluate_expression(statement,root_idx,idx,0,mapping,RHS_NODE_TYPE);
              }
              compute_reduction(stream,accsidx[k],"i",accs[k],value,rops[k]);
            }
          }
        }
        stream.dec_tab();
        stream << "}" << std::endl;


        //Declare and fill local memory
        for(unsigned int k = 0 ; k < N ; ++k){
          stream << "__local " << numeric_type_string << " " << local_buffers_names[k] << "[" << p_.local_size_0 << "];" << std::endl;
          if(utils::is_index_reduction(exprs[k]->statement().array()[exprs[k]->root_idx()].op))
            stream << "__local " << "unsigned int" << " " << local_buffers_names[k] << "idx[" << p_.local_size_0 << "];" << std::endl;
        }


        for(unsigned int k = 0 ; k < N ; ++k){
          stream << local_buffers_names[k] << "[lid] = " << accs[k] << ";" << std::endl;
          if(utils::is_index_reduction(exprs[k]->statement().array()[exprs[k]->root_idx()].op))
            stream << local_buffers_names[k] << "idx[lid] = " << accsidx[k] << ";" << std::endl;
        }

        //Reduce and write to temporary buffers
        reduce_1d_local_memory(stream, p_.local_size_0,local_buffers_names,rops);

        stream << "if(lid==0){" << std::endl;
        stream.inc_tab();
        for(unsigned int k = 0 ; k < N ; ++k)
        {
          stream << "temp"<< k << "[get_group_id(0)] = buf" << k << "[0];" << std::endl;
          if(utils::is_index_reduction(exprs[k]->statement().array()[exprs[k]->root_idx()].op))
            stream << "temp"<< k << "idx[get_group_id(0)] = buf" << k << "idx[0];" << std::endl;
        }
        stream.dec_tab();
        stream << "}" << std::endl;
      }


      void core_1(utils::kernel_generation_stream& stream, std::vector<mapped_scalar_reduction*> exprs, statements_container const & statements, std::vector<mapping_type> const & mapping) const
      {
        std::size_t N = exprs.size();
        scheduler::statement_node_numeric_type numeric_type = lhs_most(statements.data().front()).numeric_type;
        std::string numeric_type_string = utils::numeric_type_to_string(numeric_type);

        std::vector<scheduler::op_element> rops(N);
        std::vector<std::string> accs(N);
        std::vector<std::string> accsidx(N);
        std::vector<std::string> local_buffers_names(N);
        for(unsigned int k = 0 ; k < N ; ++k){
          scheduler::op_element root_op = exprs[k]->statement().array()[exprs[k]->root_idx()].op;
          rops[k].type_family = scheduler::OPERATION_BINARY_TYPE_FAMILY;
          if(root_op.type==scheduler::OPERATION_BINARY_INNER_PROD_TYPE){
            rops[k].type        = scheduler::OPERATION_BINARY_ADD_TYPE;
          }
          else{
            rops[k].type        = root_op.type;
          }
          accs[k] = "acc"+tools::to_string(k);
          accsidx[k] = accs[k] + "idx";
          local_buffers_names[k] = "buf"+tools::to_string(k);
        }

        stream << "unsigned int lid = get_local_id(0);" << std::endl;

        for(unsigned int k = 0 ; k < exprs.size() ; ++k){
          stream << "__local " << numeric_type_string << " " << local_buffers_names[k] << "[" << p_.local_size_0 << "];" << std::endl;
          if(utils::is_index_reduction(exprs[k]->statement().array()[exprs[k]->root_idx()].op))
            stream << "__local " << "unsigned int" << " " << local_buffers_names[k] << "idx[" << p_.local_size_0 << "];" << std::endl;
        }

        for(unsigned int k = 0 ; k < local_buffers_names.size() ; ++k){
          stream << numeric_type_string << " " << accs[k] << " = " << neutral_element(rops[k]) << ";" << std::endl;
          if(utils::is_index_reduction(exprs[k]->statement().array()[exprs[k]->root_idx()].op))
            stream << "unsigned int" << " " << accsidx[k] << " = " << 0 << ";" << std::endl;
        }

        stream << "for(unsigned int i = lid ; i < " << p_.num_groups << " ; i += get_local_size(0)){" << std::endl;
        stream.inc_tab();
        for(unsigned int k = 0 ; k < N ; ++k)
          compute_reduction(stream,accsidx[k],"temp"+tools::to_string(k)+"idx[i]",accs[k],"temp"+tools::to_string(k)+"[i]",rops[k]);
        stream.dec_tab();
        stream << "}" << std::endl;

        for(unsigned int k = 0 ; k < local_buffers_names.size() ; ++k)
        {
          stream << local_buffers_names[k] << "[lid] = " << accs[k] << ";" << std::endl;
          if(utils::is_index_reduction(exprs[k]->statement().array()[exprs[k]->root_idx()].op))
            stream << local_buffers_names[k] << "idx[lid] = " << accsidx[k] << ";" << std::endl;
        }


        //Reduce and write final result
        reduce_1d_local_memory(stream, p_.local_size_0,local_buffers_names,rops);
        for(unsigned int k = 0 ; k < N ; ++k)
        {
          std::string suffix = "";
          if(utils::is_index_reduction(exprs[k]->statement().array()[exprs[k]->root_idx()].op))
            suffix = "idx";
          exprs[k]->access_name(local_buffers_names[k]+suffix+"[0]");
        }

        stream << "if(lid==0){" << std::endl;
        stream.inc_tab();
        unsigned int i = 0;
        for(statements_container::data_type::const_iterator it = statements.data().begin() ; it != statements.data().end() ; ++it)
          stream << tree_parsing::evaluate_expression(*it, it->root(), index_tuple("0", "N"), 0, mapping[i++], PARENT_NODE_TYPE) << ";" << std::endl;

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
