#ifndef VIENNACL_GENERATOR_TEMPLATES_SCALAR_REDUCTION_HPP
#define VIENNACL_GENERATOR_TEMPLATES_SCALAR_REDUCTION_HPP

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
#include "viennacl/generator/tree_parsing/fetch.hpp"
#include "viennacl/generator/tree_parsing/elementwise_expression.hpp"
#include "viennacl/generator/utils.hpp"

#include "viennacl/generator/templates/template_base.hpp"
#include "viennacl/generator/templates/reduction_utils.hpp"

#include "viennacl/tools/tools.hpp"

namespace viennacl{

  namespace generator{

    class scalar_reduction : public profile_base{
      private:
        typedef std::vector<std::pair<const char *, viennacl::ocl::handle<cl_mem> > > temporaries_type;

        static void fill_scalartypes(statements_type statements, std::vector<const char *> & res){
          res.reserve(statements.size());
          for(statements_type::const_iterator it = statements.begin() ; it != statements.end() ; ++it){
            if (it->second.lhs.type_family == scheduler::SCALAR_TYPE_FAMILY)
            {
              switch(it->second.lhs.numeric_type){
                case scheduler::FLOAT_TYPE:
                  res.push_back("float");
                  break;
                case scheduler::DOUBLE_TYPE:
                  res.push_back("double");
                  break;
                default:
                  res.push_back("");
                  break;
              }
            }
            else
            {
              res.push_back("");
            }
          }
        }

      public:

        std::size_t lmem_used(std::size_t scalartype_size) const {
          return local_size_1_*scalartype_size;
        }

        void init_temporaries(statements_type const & statements) const {
          if(temporaries_.empty()){
            //set temporary buffer argument
            for(statements_type::const_iterator it = statements.begin() ; it != statements.end() ; ++it){
              scheduler::statement::container_type const & array = it->first.array();
              std::size_t size_of_scalartype;
              const char * scalartype_name;
              switch(array[0].lhs.numeric_type){
                case scheduler::FLOAT_TYPE: scalartype_name = "float"; size_of_scalartype = sizeof(float); break;
                case scheduler::DOUBLE_TYPE: scalartype_name = "double"; size_of_scalartype = sizeof(double); break;
                default : throw generator_not_supported_exception("Unsupported scalartype");
              }
              for(scheduler::statement::container_type::const_iterator iit = array.begin() ; iit != array.end() ; ++iit){
                if(is_scalar_reduction(*iit)){
                  temporaries_.push_back(std::make_pair(scalartype_name, viennacl::ocl::current_context().create_memory(CL_MEM_READ_WRITE, num_groups_*size_of_scalartype)));
                }
              }
            }
          }
        }

        void set_size_argument(viennacl::scheduler::statement const & s, viennacl::scheduler::statement_node const & /*root_node*/, unsigned int & n_arg, viennacl::ocl::kernel & k) const {
          scheduler::statement::container_type exprs = s.array();
          for(scheduler::statement::container_type::iterator it = exprs.begin() ; it != exprs.end() ; ++it){
            if(is_scalar_reduction(*it)){
              //set size argument
              scheduler::statement_node const * current_node = &(*it);

              std::size_t vector_size;
              //The LHS of the prod is a vector
              if(current_node->lhs.type_family==scheduler::VECTOR_TYPE_FAMILY)
              {
                vector_size = utils::call_on_vector(current_node->lhs, utils::internal_size_fun());
              }
              else{
                //The LHS of the prod is a vector expression
                current_node = &exprs[current_node->lhs.node_index];
                if(current_node->lhs.type_family==scheduler::VECTOR_TYPE_FAMILY)
                {
                  vector_size = cl_uint(utils::call_on_vector(current_node->lhs, utils::internal_size_fun()));
                }
                else if(current_node->rhs.type_family==scheduler::VECTOR_TYPE_FAMILY)
                {
                  vector_size = cl_uint(utils::call_on_vector(current_node->lhs, utils::internal_size_fun()));
                }
                else{
                  assert(false && bool("unexpected expression tree"));
                }
              }
              k.arg(n_arg++, cl_uint(vector_size/simd_width_));
            }
          }
        }

      public:
        /** @brief The user constructor */
        scalar_reduction(unsigned int vectorization, unsigned int local_size, unsigned int num_groups, unsigned int decomposition) : profile_base(vectorization, local_size, 1, 2), num_groups_(num_groups), decomposition_(decomposition){ }


        static std::string csv_format() {
          return "Vec,LSize,NumGroups,GlobalDecomposition";
        }

        std::string csv_representation() const{
          std::ostringstream oss;
          oss << simd_width_
                 << "," << local_size_1_
                 << "," << num_groups_
                 << "," << decomposition_;
          return oss.str();
        }

        unsigned int num_groups() const { return num_groups_; }


        unsigned int decomposition() const { return decomposition_; }


        void configure_range_enqueue_arguments(std::size_t kernel_id, statements_type  const & statements, viennacl::ocl::kernel & k, unsigned int & n_arg)  const{

          //create temporaries
          init_temporaries(statements);

          //configure ND range
          if(kernel_id==0){
            configure_local_sizes(k, 0);

            std::size_t gsize = local_size_1_*num_groups_;
            k.global_work_size(0,gsize);
            k.global_work_size(1,1);
          }
          else{
            configure_local_sizes(k, 1);

            k.global_work_size(0,local_size_1_);
            k.global_work_size(1,1);
          }

          //set arguments
          set_size_argument(statements.front().first, statements.front().second, n_arg, k);
          for(temporaries_type::iterator it = temporaries_.begin() ; it != temporaries_.end() ; ++it){
            k.arg(n_arg++, it->second);
          }
        }

        void add_kernel_arguments(statements_type  const & statements, std::string & arguments_string) const{
          init_temporaries(statements);
          arguments_string += generate_value_kernel_argument("unsigned int", "N");
          for(temporaries_type::iterator it = temporaries_.begin() ; it != temporaries_.end() ; ++it){
            arguments_string += generate_pointer_kernel_argument("__global", it->first, "temp" + utils::to_string(std::distance(temporaries_.begin(), it)));
          }
        }

      private:

        void core_0(utils::kernel_generation_stream& stream, std::vector<mapped_scalar_reduction*> exprs, std::vector<const char *> const & scalartypes, statements_type const & /*statements*/, std::vector<mapping_type> const & /*mapping*/) const {
          std::size_t N = exprs.size();

          std::vector<scheduler::op_element> rops(N);
          std::vector<std::string> accs(N);
          std::vector<std::string> local_buffers_names(N);
          for(std::size_t k = 0 ; k < N ; ++k){
            scheduler::op_element root_op = exprs[k]->root_node().op;
            rops[k].type_family = scheduler::OPERATION_BINARY_TYPE_FAMILY;
            if(root_op.type==scheduler::OPERATION_BINARY_INNER_PROD_TYPE){
              rops[k].type_subfamily = scheduler::OPERATION_ELEMENTWISE_OPERATOR_TYPE_SUBFAMILY;
              rops[k].type        = scheduler::OPERATION_BINARY_ADD_TYPE;
            }
            else{
              rops[k].type_subfamily = get_subfamily(root_op.type);
              rops[k].type        = root_op.type;
            }
            accs[k] = "sum"+utils::to_string(k);
            local_buffers_names[k] = "buf"+utils::to_string(k);
          }

          stream << "unsigned int lid = get_local_id(0);" << std::endl;

          for(std::size_t k = 0 ; k < N ; ++k)
            stream << scalartypes[k] << " " << accs[k] << " = " << neutral_element(rops[k]) << ";" << std::endl;

          std::string init;
          std::string upper_bound;
          std::string inc;
          if(decomposition_){
            init = "get_global_id(0)";
            upper_bound = "N";
            inc = "get_global_size(0)";
          }
          else{
            stream << "unsigned int chunk_size = (N + get_num_groups(0)-1)/get_num_groups(0);" << std::endl;
            stream << "unsigned int chunk_start = get_group_id(0)*chunk_size;" << std::endl;
            stream << "unsigned int chunk_end = min(chunk_start+chunk_size, N);" << std::endl;
            init = "chunk_start + get_local_id(0)";
            upper_bound = "chunk_end";
            inc = "get_local_size(0)";
          }

          stream << "for(unsigned int i = " << init << "; i < " << upper_bound << " ; i += " << inc << "){" << std::endl;
          stream.inc_tab();
          {
            //Fetch vector entry
            std::set<std::string>  fetched;
            for(std::vector<mapped_scalar_reduction*>::iterator it = exprs.begin() ; it != exprs.end() ; ++it)
            {
              viennacl::scheduler::statement const & statement = (*it)->statement();
              viennacl::scheduler::statement_node const & root_node = (*it)->root_node();
              tree_parsing::fetch_all_lhs(fetched,statement,root_node, std::make_pair("i", "0"),stream,(*it)->mapping());
              tree_parsing::fetch_all_rhs(fetched,statement,root_node, std::make_pair("i", "0"),stream,(*it)->mapping());
            }
            //Update sums;
            for(std::size_t k = 0 ; k < exprs.size() ; ++k)
            {
              viennacl::scheduler::statement const & statement = exprs[k]->statement();
              viennacl::scheduler::statement_node const & root_node = exprs[k]->root_node();
              mapping_type const & mapping = exprs[k]->mapping();
              if(simd_width_ > 1){
                for(unsigned int a = 0 ; a < simd_width_ ; ++a){
                  std::string str;
                  tree_parsing::generate_all_lhs(statement,root_node,std::make_pair("i","0"),a,str,mapping);
                  if(root_node.op.type==scheduler::OPERATION_BINARY_INNER_PROD_TYPE){
                      str += "*";
                      tree_parsing::generate_all_rhs(statement,root_node,std::make_pair("i","0"),a,str,mapping);
                  }
                  compute_reduction(stream,accs[k],str,rops[k]);
                }
              }
              else{
                std::string str;
                tree_parsing::generate_all_lhs(statement,root_node,std::make_pair("i","0"),-1,str,mapping);
                if(root_node.op.type==scheduler::OPERATION_BINARY_INNER_PROD_TYPE){
                    str += "*";
                    tree_parsing::generate_all_rhs(statement,root_node,std::make_pair("i","0"),-1,str,mapping);
                }
                compute_reduction(stream,accs[k],str,rops[k]);
              }
            }
          }
          stream.dec_tab();
          stream << "}" << std::endl;


          //Declare and fill local memory
          for(std::size_t k = 0 ; k < N ; ++k)
            stream << "__local " << scalartypes[k] << " " << local_buffers_names[k] << "[" << local_size_1_ << "];" << std::endl;

          for(std::size_t k = 0 ; k < N ; ++k)
            stream << local_buffers_names[k] << "[lid] = " << accs[k] << ";" << std::endl;

          //Reduce and write to temporary buffers
          reduce_1d_local_memory(stream, local_size_1_,local_buffers_names,rops);

          stream << "if(lid==0){" << std::endl;
          stream.inc_tab();
          for(std::size_t k = 0 ; k < N ; ++k)
            stream << "temp"<< k << "[get_group_id(0)] = buf" << k << "[0];" << std::endl;
          stream.dec_tab();
          stream << "}" << std::endl;
        }


        void core_1(utils::kernel_generation_stream& stream, std::vector<mapped_scalar_reduction*> exprs, std::vector<const char *> scalartypes, statements_type const & statements, std::vector<mapping_type> const & mapping) const {
          std::size_t N = exprs.size();
          std::vector<scheduler::op_element> rops(N);
          std::vector<std::string> accs(N);
          std::vector<std::string> local_buffers_names(N);
          for(std::size_t k = 0 ; k < N ; ++k){
            scheduler::op_element root_op = exprs[k]->root_node().op;
            rops[k].type_family = scheduler::OPERATION_BINARY_TYPE_FAMILY;
            if(root_op.type==scheduler::OPERATION_BINARY_INNER_PROD_TYPE){
              rops[k].type_subfamily = scheduler::OPERATION_ELEMENTWISE_OPERATOR_TYPE_SUBFAMILY;
              rops[k].type        = scheduler::OPERATION_BINARY_ADD_TYPE;
            }
            else{
              rops[k].type_subfamily = get_subfamily(root_op.type);
              rops[k].type        = root_op.type;
            }
            accs[k] = "sum"+utils::to_string(k);
            local_buffers_names[k] = "buf"+utils::to_string(k);
          }



          stream << "unsigned int lid = get_local_id(0);" << std::endl;

          for(std::size_t k = 0 ; k < exprs.size() ; ++k)
            stream << "__local " << scalartypes[k] << " " << local_buffers_names[k] << "[" << local_size_1_ << "];" << std::endl;

          for(std::size_t k = 0 ; k < local_buffers_names.size() ; ++k)
            stream << scalartypes[0] << " " << accs[k] << " = " << neutral_element(rops[k]) << ";" << std::endl;

          stream << "for(unsigned int i = lid ; i < " << num_groups_ << " ; i += get_local_size(0)){" << std::endl;
          stream.inc_tab();
          for(std::size_t k = 0 ; k < N ; ++k)
            compute_reduction(stream,accs[k],"temp"+utils::to_string(k)+"[i]",rops[k]);
          stream.dec_tab();
          stream << "}" << std::endl;

          for(std::size_t k = 0 ; k < local_buffers_names.size() ; ++k)
             stream << local_buffers_names[k] << "[lid] = " << accs[k] << ";" << std::endl;


          //Reduce and write final result
          reduce_1d_local_memory(stream, local_size_1_,local_buffers_names,rops);
          for(std::size_t k = 0 ; k < N ; ++k)
            exprs[k]->access_name(local_buffers_names[k]+"[0]");

          stream << "if(lid==0){" << std::endl;
          stream.inc_tab();
          std::size_t i = 0;
          for(statements_type::const_iterator it = statements.begin() ; it != statements.end() ; ++it){
            std::string str;
            tree_parsing::traverse(it->first, it->second, tree_parsing::expression_generation_traversal(std::make_pair("0", "0"), -1, str, mapping[i++]), false);
            stream << str << ";" << std::endl;
          }
          stream.dec_tab();
          stream << "}" << std::endl;
        }

        void core(std::size_t kernel_id, utils::kernel_generation_stream& stream, expression_descriptor /*descriptor*/, statements_type const & statements, std::vector<mapping_type> const & mapping) const {
          std::vector<mapped_scalar_reduction*> exprs;
          for(std::vector<mapping_type>::const_iterator it = mapping.begin() ; it != mapping.end() ; ++it)
            for(mapping_type::const_iterator iit = it->begin() ; iit != it->end() ; ++iit)
              if(mapped_scalar_reduction * p = dynamic_cast<mapped_scalar_reduction*>(iit->second.get()))
                exprs.push_back(p);

          std::vector<const char *> scalartypes;
          fill_scalartypes(statements, scalartypes);

          if(kernel_id==0){
            core_0(stream,exprs,scalartypes,statements,mapping);
          }
          else{
            core_1(stream,exprs,scalartypes,statements,mapping);
          }
        }

      private:
        unsigned int num_groups_;
        unsigned int decomposition_;
        mutable temporaries_type temporaries_;
    };


  }

}

#endif
