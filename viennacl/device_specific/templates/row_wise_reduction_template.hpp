#ifndef VIENNACL_DEVICE_SPECIFIC_TEMPLATES_ROW_WISE_REDUCTION_HPP
#define VIENNACL_DEVICE_SPECIFIC_TEMPLATES_ROW_WISE_REDUCTION_HPP

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


/** @file viennacl/generator/row_wise_reduction.hpp
 *
 * Kernel template for the vector reduction operation
*/

#include <vector>

#include "viennacl/scheduler/forwards.h"

#include "viennacl/device_specific/mapped_objects.hpp"
#include "viennacl/device_specific/tree_parsing/read_write.hpp"
#include "viennacl/device_specific/tree_parsing/evaluate_expression.hpp"
#include "viennacl/device_specific/utils.hpp"

#include "viennacl/device_specific/templates/template_base.hpp"
#include "viennacl/device_specific/templates/reduction_utils.hpp"

#include "viennacl/tools/tools.hpp"

#include "viennacl/scheduler/io.hpp"

namespace viennacl{

  namespace device_specific{

    class row_wise_reduction_template : public template_base{
    public:
      /** @brief The user constructor */
      row_wise_reduction_template(const char * scalartype, char A_trans, unsigned int simd_width,
                                  unsigned int local_size_0, unsigned int local_size_1, unsigned int num_groups_0) : template_base(scalartype, simd_width, local_size_0, local_size_1, 1), A_trans_(A_trans),  num_groups_0_(num_groups_0){ }

      void configure_range_enqueue_arguments(unsigned int kernel_id, statements_container const & statements, viennacl::ocl::kernel & k, unsigned int & n_arg)  const{
        configure_local_sizes(kernel, kernel_id);
        kernel.global_work_size(0,local_size_0_*num_groups_0_);
        kernel.global_work_size(1,local_size_1_);

        for(statements_container::data_type::const_iterator it = statements.data().begin() ; it != statements.data().end() ; ++it){
          scheduler::statement::container_type exprs = it->first.array();
          for(scheduler::statement::container_type::iterator iit = exprs.begin() ; iit != exprs.end() ; ++iit){
            if(is_vector_reduction(*iit)){
              scheduler::statement_node const * current_node = &(*iit);
              //The LHS of the prod is a matrix
              if(current_node->lhs.type_family==scheduler::MATRIX_TYPE_FAMILY)
              {
                kernel.arg(n_arg++, cl_uint(utils::call_on_matrix(current_node->lhs, utils::internal_size1_fun())));
                kernel.arg(n_arg++, cl_uint(utils::call_on_matrix(current_node->lhs, utils::internal_size2_fun())));
                return;
              }
              else
              {
                //The LHS of the prod is a matrix expression
                current_node = &exprs[current_node->lhs.node_index];
                if(current_node->lhs.type_family==scheduler::MATRIX_TYPE_FAMILY)
                {
                  kernel.arg(n_arg++, cl_uint(utils::call_on_matrix(current_node->lhs, utils::internal_size1_fun())));
                  kernel.arg(n_arg++, cl_uint(utils::call_on_matrix(current_node->lhs, utils::internal_size2_fun())));
                  return;
                }
                else if(current_node->rhs.type_family==scheduler::MATRIX_TYPE_FAMILY)
                {
                  kernel.arg(n_arg++, cl_uint(utils::call_on_matrix(current_node->lhs, utils::internal_size1_fun())));
                  kernel.arg(n_arg++, cl_uint(utils::call_on_matrix(current_node->lhs, utils::internal_size2_fun())));
                  return;
                }
                else
                  throw generator_not_supported_exception("Unexpected expression tree");
              }
              return;
            }
          }
        }

      }

      void add_kernel_arguments(std::string & arguments_string) const{
        arguments_string += generate_value_kernel_argument("unsigned int", "M");
        arguments_string += generate_value_kernel_argument("unsigned int", "N");
      }

      private:
        unsigned int lmem_used(unsigned int scalartype_size) const {
          return local_size_0_*(local_size_1_+1)*scalartype_size;
        }

        void core(unsigned int /*kernel_id*/, utils::kernel_generation_stream& stream, statements_container const & statements, std::vector<mapping_type> const & mapping) const {
          std::vector<mapped_vector_reduction*> exprs;
          for(std::vector<mapping_type>::const_iterator it = mapping.begin() ; it != mapping.end() ; ++it){
            for(mapping_type::const_iterator iit = it->begin() ; iit != it->end() ; ++iit){
              if(mapped_vector_reduction * p = dynamic_cast<mapped_vector_reduction*>(iit->second.get()))
                exprs.push_back(p);
            }
          }

          unsigned int N = exprs.size();

          std::vector<scheduler::op_element> rops(N);
          std::vector<std::string> accs(N);
          std::vector<std::string> local_buffers_names(N);
          for(unsigned int k = 0 ; k < N ; ++k){
            scheduler::op_element root_op = exprs[k]->root_node().op;
            rops[k].type_family = scheduler::OPERATION_BINARY_TYPE_FAMILY;
            if(root_op.type==scheduler::OPERATION_BINARY_MAT_VEC_PROD_TYPE){
              rops[k].type        = scheduler::OPERATION_BINARY_ADD_TYPE;
            }
            else{
              rops[k].type        = root_op.type;
            }
            accs[k] = "sum"+tools::to_string(k);
            local_buffers_names[k] = "buf"+tools::to_string(k);
          }



          unsigned int lsize1 = local_size_0_;
          unsigned int lsize2 = local_size_1_+1;

          std::string size1 = "M", size2 = "N";


          for(std::vector<mapped_vector_reduction*>::iterator it = exprs.begin() ; it != exprs.end() ; ++it){
            stream << "__local " <<  (*it)->scalartype() << " buf" << std::distance(exprs.begin(), it) << '[' << lsize1*lsize2 << "];" << std::endl;
          }

          stream << "unsigned int lid0 = get_local_id(0);" << std::endl;
          stream << "unsigned int lid1 = get_local_id(1);" << std::endl;


          stream << "for(unsigned int r = get_global_id(0) ; r < " << size1 << " ; r += get_global_size(0)){" << std::endl;
          stream.inc_tab();
          {
            for(unsigned int k = 0 ; k < exprs.size() ; ++k)
              stream << exprs[k]->scalartype() << " " << accs[k] << " = " << neutral_element(rops[k]) << ";" << std::endl;

            stream << "for( unsigned int c = get_local_id(1) ; c < " << size2 << " ; c += get_local_size(1)){" << std::endl;
            stream.inc_tab();
            {
              std::set<std::string>  cache;
              unsigned int N = exprs.size();

              for(unsigned int k = 0 ; k < N ; ++k)
              {
                viennacl::scheduler::statement const & statement = exprs[k]->statement();
                viennacl::scheduler::statement_node const & root_node = exprs[k]->root_node();
                if(A_trans_=='T')
                  tree_parsing::read_write(mapped_handle::fetch, "reg", cache,statement,root_node, std::make_pair("c", "r"),stream,exprs[k]->mapping(), tree_parsing::LHS_NODE_TYPE);
                else
                  tree_parsing::read_write(mapped_handle::fetch, "reg", cache,statement,root_node, std::make_pair("r", "c"),stream,exprs[k]->mapping(), tree_parsing::LHS_NODE_TYPE);

                if(root_node.op.type==scheduler::OPERATION_BINARY_MAT_VEC_PROD_TYPE)
                  tree_parsing::read_write(mapped_handle::fetch, "reg", cache,statement,root_node, std::make_pair("c", "0"),stream,exprs[k]->mapping(), tree_parsing::RHS_NODE_TYPE);
              }


              //Update sums;
              for(unsigned int k = 0 ; k < N ; ++k)
              {
                viennacl::scheduler::statement const & statement = exprs[k]->statement();
                unsigned int root_idx = exprs[k]->root_idx();
                std::string str;
                tree_parsing::generate_all_lhs(statement,root_idx,std::make_pair("",""),-1,str,exprs[k]->mapping());
                if(root_node.op.type==scheduler::OPERATION_BINARY_MAT_VEC_PROD_TYPE){
                  str += "*";
                  tree_parsing::generate_all_rhs(statement,root_node,std::make_pair("",""),-1,str,exprs[k]->mapping());
                }
                compute_reduction(stream,accs[k],str,rops[k]);
              }
            }
            stream.dec_tab();
            stream << "}" << std::endl;


            for(unsigned int k = 0 ; k < exprs.size() ; ++k){
              stream << "buf" << k << "[lid0*" << lsize2 << "+ lid1] = " << accs[k] << ";" << std::endl;
            }

            for(unsigned int stride = local_size_1_/2 ; stride>0 ; stride /=2){
              stream << "barrier(CLK_LOCAL_MEM_FENCE); " << std::endl;
              stream <<  "if(lid1 < " << stride << ")" ;
              stream << "{" << std::endl;
              stream.inc_tab();

              for(unsigned int k = 0 ; k < N ; ++k)
                compute_reduction(stream
                                      ,local_buffers_names[k] + "[lid0*" + tools::to_string(lsize2) + "+ lid1]"
                                      ,local_buffers_names[k] + "[lid0*" + tools::to_string(lsize2) + "+ lid1 + " + tools::to_string(stride) + "]"
                                      ,rops[k]);

              stream.dec_tab();
              stream << "}" << std::endl;
            }


            stream <<  "if(lid1 == 0)" ;
            stream << "{" << std::endl;
            stream.inc_tab();
            for(unsigned int k = 0 ; k < N ; ++k)
              exprs[k]->access_name(local_buffers_names[k] + "[lid0*"+tools::to_string(lsize2)+"]");

            unsigned int i = 0;
            for(statements_container::data_type::const_iterator it = statements.data().begin() ; it != statements.data().end() ; ++it){
              std::string str;
              tree_parsing::traverse(*it, it->root(), tree_parsing::evaluate_expression_traversal(std::make_pair("r","0"), -1, str, mapping[i++]), false);
              stream << str << ";" << std::endl;
            }
            stream.dec_tab();
            stream << "}" << std::endl;

          }
          stream.dec_tab();
          stream << "}" << std::endl;

        }

      private:
        const char A_trans_;
        unsigned int num_groups_0_;
    };

  }
}

#endif
