#ifndef VIENNACL_GENERATOR_GENERATE_SCALAR_REDUCTION_HPP
#define VIENNACL_GENERATOR_GENERATE_SCALAR_REDUCTION_HPP

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


/** @file viennacl/generator/templates/scalar_reduction.hpp
 *
 * Kernel template for the scalar reduction operation
*/

#include <vector>

#include "viennacl/backend/opencl.hpp"

#include "viennacl/scheduler/forwards.h"

#include "viennacl/generator/generate_utils.hpp"
#include "viennacl/generator/utils.hpp"

#include "viennacl/generator/generate_template_base.hpp"

#include "viennacl/tools/tools.hpp"

namespace viennacl{

  namespace generator{


    class scalar_reduction : public template_base{
      private:
        typedef template_base base_type;

        static void fill_scalartypes(statements_type statements, std::vector<const char *> & res){
          res.reserve(statements.size());
          for(statements_type::const_iterator it = statements.begin() ; it != statements.end() ; ++it){
	    if (it->array()[0].lhs.type_family == scheduler::SCALAR_TYPE_FAMILY)
	    {
	      switch(it->array()[0].lhs.type){
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
        typedef base_type::statements_type statements_type;

        class profile : public template_base::profile{
            friend class scalar_reduction;

            typedef std::vector<std::pair<const char *, viennacl::ocl::handle<cl_mem> > > temporaries_type;

            std::size_t lmem_used(std::size_t scalartype_size) const {
              return group_size_*scalartype_size;
            }

            void init_temporaries(statements_type const & statements) const {
              if(temporaries_.empty()){
                temporaries_.reserve(statements.size());
                //set temporary buffer argument
                for(statements_type::const_iterator it = statements.begin() ; it != statements.end() ; ++it){
                  scheduler::statement::container_type const & exprs = it->array();
                  std::size_t size_of_scalartype;
                  const char * scalartype_name;
		  if (exprs[0].lhs.type_family != scheduler::SCALAR_TYPE_FAMILY) throw "not implemented";
                  switch(exprs[0].lhs.type){
                    case scheduler::FLOAT_TYPE: scalartype_name = "float"; size_of_scalartype = sizeof(float); break;
                    case scheduler::DOUBLE_TYPE: scalartype_name = "double"; size_of_scalartype = sizeof(double); break;
                    default: throw "not implemented"; break;
                  }
                  for(scheduler::statement::container_type::const_iterator iit = exprs.begin() ; iit != exprs.end() ; ++iit){
                    if(iit->op.type==scheduler::OPERATION_BINARY_INNER_PROD_TYPE){
                      temporaries_.push_back(std::make_pair(scalartype_name, viennacl::ocl::current_context().create_memory(CL_MEM_READ_WRITE, num_groups_*size_of_scalartype)));
                    }
                  }
                }
              }
            }

            void set_size_argument(viennacl::scheduler::statement const & s, unsigned int & n_arg, viennacl::ocl::kernel & k) const {
              scheduler::statement::container_type exprs = s.array();
              for(scheduler::statement::container_type::iterator it = exprs.begin() ; it != exprs.end() ; ++it){
                if(it->op.type==scheduler::OPERATION_BINARY_INNER_PROD_TYPE){
                  //set size argument
                  scheduler::statement_node const * current_node = &(*it);

                  std::size_t vector_size;
                  //The LHS of the prod is a vector
                  if(current_node->lhs.type_family==scheduler::VECTOR_TYPE_FAMILY)
                  {
                    vector_size = utils::call_on_vector(current_node->lhs.type, current_node->lhs, utils::size_fun());
                  }
                  else{
                    //The LHS of the prod is a vector expression
                    current_node = &exprs[current_node->lhs.node_index];
                    if(current_node->lhs.type_family==scheduler::VECTOR_TYPE_FAMILY)
                    {
                      vector_size = cl_uint(utils::call_on_vector(current_node->lhs.type, current_node->lhs, utils::size_fun()));
                    }
                    else if(current_node->rhs.type_family==scheduler::VECTOR_TYPE_FAMILY)
                    {
                      vector_size = cl_uint(utils::call_on_vector(current_node->lhs.type, current_node->lhs, utils::size_fun()));
                    }
                    else{
                      assert(false && bool("unexpected expression tree"));
                    }
                  }
                  k.arg(n_arg++, cl_uint(vector_size/vectorization_));
                }
              }
            }

          public:
            /** @brief The user constructor */
            profile(unsigned int vectorization, unsigned int group_size, unsigned int num_groups, bool global_decomposition) : template_base::profile(vectorization, 2), group_size_(group_size), num_groups_(num_groups), global_decomposition_(global_decomposition){ }

            void set_local_sizes(std::size_t& s1, std::size_t& s2, std::size_t /*kernel_id*/) const{
              s1 = group_size_;
              s2 = 1;
            }


            void configure_range_enqueue_arguments(std::size_t kernel_id, statements_type  const & statements, viennacl::ocl::kernel & k, unsigned int & n_arg)  const{

              //create temporaries
              init_temporaries(statements);

              //configure ND range
              if(kernel_id==0){
                configure_local_sizes(k, 0);

                std::size_t gsize = group_size_*num_groups_;
                k.global_work_size(0,gsize);
                k.global_work_size(1,1);
              }
              else{
                configure_local_sizes(k, 1);

                k.global_work_size(0,group_size_);
                k.global_work_size(1,1);
              }

              //set arguments
              set_size_argument(statements.front(), n_arg, k);
              for(temporaries_type::iterator it = temporaries_.begin() ; it != temporaries_.end() ; ++it){
                k.arg(n_arg++, it->second);
              }
            }

            void kernel_arguments(statements_type  const & statements, std::string & arguments_string) const{
              init_temporaries(statements);
              arguments_string += detail::generate_value_kernel_argument("unsigned int", "N");
              for(temporaries_type::iterator it = temporaries_.begin() ; it != temporaries_.end() ; ++it){
                arguments_string += detail::generate_pointer_kernel_argument("__global", it->first, "temp" + utils::to_string(std::distance(temporaries_.begin(), it)));
              }
            }

            unsigned int num_groups() const { return num_groups_; }

            unsigned int group_size() const { return group_size_; }
          private:
            unsigned int group_size_;
            unsigned int num_groups_;
            bool global_decomposition_;
            mutable temporaries_type temporaries_;
        };



      private:

        void core_0(utils::kernel_generation_stream& stream, std::vector<detail::mapped_scalar_reduction*> exprs, std::vector<const char *> const & scalartypes) const {

          stream << "unsigned int lid = get_local_id(0);" << std::endl;

          for(std::size_t k = 0 ; k < exprs.size() ; ++k)
            stream << scalartypes[k] << " sum" << k << " = 0;" << std::endl;

          if(profile_.global_decomposition_){
            stream << "for(unsigned int i = get_global_id(0) ; i < N ; i += get_global_size(0)){" << std::endl;
          }
          else{
            stream << "unsigned int chunk_size = (N + get_num_groups(0)-1)/get_num_groups(0);" << std::endl;
            stream << "unsigned int chunk_start = get_group_id(0)*chunk_size;" << std::endl;
            stream << "unsigned int chunk_end = min(chunk_start+chunk_size, N);" << std::endl;
            stream << "for(unsigned int i = chunk_start + get_local_id(0) ; i < chunk_end ; i += get_local_size(0)){" << std::endl;
          }
          stream.inc_tab();

          //Fetch vector entry
          std::set<std::string>  fetched;

          for(std::size_t k = 0 ; k < exprs.size() ; ++k){
            detail::traverse(*exprs[k]->lhs().array, detail::fetch_traversal(fetched, std::make_pair("i", "0"), profile_.vectorization_, stream, *exprs[k]->lhs().mapping), false, exprs[k]->lhs().index);
            detail::traverse(*exprs[k]->rhs().array, detail::fetch_traversal(fetched, std::make_pair("i", "0"), profile_.vectorization_, stream, *exprs[k]->rhs().mapping), false, exprs[k]->rhs().index);
          }


          //Update sums;
          for(std::size_t k = 0 ; k < exprs.size() ; ++k){
            if(profile_.vectorization_ > 1){
              for(unsigned int a = 0 ; a < profile_.vectorization_ ; ++a){
                std::string expr_str;
                detail::traverse(*exprs[k]->lhs().array, detail::expression_generation_traversal(std::make_pair("i", "0"), a, expr_str, *exprs[k]->lhs().mapping), false, exprs[k]->lhs().index);
                expr_str += "*";
                detail::traverse(*exprs[k]->rhs().array, detail::expression_generation_traversal(std::make_pair("i", "0"), a, expr_str, *exprs[k]->rhs().mapping), false, exprs[k]->rhs().index);
                stream << " sum" << k << " += "  << expr_str << ";" << std::endl;
              }
            }
            else{
              std::string expr_str;
              detail::traverse(*exprs[k]->lhs().array, detail::expression_generation_traversal(std::make_pair("i", "0"), -1, expr_str, *exprs[k]->lhs().mapping), false, exprs[k]->lhs().index);
              expr_str += "*";
              detail::traverse(*exprs[k]->rhs().array, detail::expression_generation_traversal(std::make_pair("i", "0"), -1, expr_str, *exprs[k]->rhs().mapping), false, exprs[k]->rhs().index);
              stream << " sum" << k << " += "  << expr_str << ";" << std::endl;
            }
          }

          stream.dec_tab();
          stream << "}" << std::endl;
          //Declare and fill local memory
          for(std::size_t k = 0 ; k < exprs.size() ; ++k)
            stream << "__local " << scalartypes[k] << " buf" << k << "[" << profile_.group_size_ << "];" << std::endl;

          for(std::size_t k = 0 ; k < exprs.size() ; ++k)
            stream << "buf" << k << "[lid] = sum" << k << ";" << std::endl;

          //Reduce local memory
          for(unsigned int stride = profile_.group_size_/2 ; stride>1 ; stride /=2){
            stream << "barrier(CLK_LOCAL_MEM_FENCE); " << std::endl;
            stream << "if(lid < " << stride << "){" << std::endl;
            stream.inc_tab();
            for(std::size_t k = 0 ; k < exprs.size() ; ++k){
              stream << "buf" << k << "[lid] += buf" << k << "[lid + " << stride << "];" << std::endl;
            }
            stream.dec_tab();
            stream << "}" << std::endl;
          }

          //Last reduction and fetches to temporary buffer
          stream << "barrier(CLK_LOCAL_MEM_FENCE); " << std::endl;
          stream << "if(lid==0){" << std::endl;
          stream.inc_tab();
          for(std::size_t k = 0 ; k < exprs.size() ; ++k)
            stream << "buf" << k << "[0] += buf" << k << "[1];" << std::endl;

          for(std::size_t k = 0 ; k < exprs.size() ; ++k)
            stream << "temp"<< k << "[get_group_id(0)] = buf" << k << "[0];" << std::endl;

          stream.dec_tab();
          stream << "}" << std::endl;
        }


        void core_1(utils::kernel_generation_stream& stream, std::vector<detail::mapped_scalar_reduction*> exprs, std::vector<const char *> scalartypes) const {
          stream << "unsigned int lid = get_local_id(0);" << std::endl;

          for(std::size_t k = 0 ; k < exprs.size() ; ++k)
            stream << "__local " << scalartypes[k] << " buf" << k << "[" << profile_.group_size_ << "];" << std::endl;

          for(std::size_t k = 0 ; k < exprs.size() ; ++k)
            stream << scalartypes[0] << " sum" << k << " = 0;" << std::endl;

          stream << "for(unsigned int i = lid ; i < " << profile_.num_groups_ << " ; i += get_local_size(0)){" << std::endl;
          stream.inc_tab();
          for(std::size_t k = 0 ; k < exprs.size() ; ++k)
            stream << "sum" << k << " += temp" << k << "[i];" << std::endl;
          stream.dec_tab();
          stream << "}" << std::endl;

          for(std::size_t k = 0 ; k < exprs.size() ; ++k)
            stream << "buf" << k << "[lid] = sum" << k << ";" << std::endl;

          //Reduce local memory
          for(unsigned int stride = profile_.group_size_/2 ; stride>1 ; stride /=2){
            stream << "barrier(CLK_LOCAL_MEM_FENCE); " << std::endl;
            stream << "if(lid < " << stride << "){" << std::endl;
            stream.inc_tab();
            for(std::size_t k = 0 ; k < exprs.size() ; ++k){
              stream << "buf" << k << "[lid] += buf" << k << "[lid + " << stride << "];" << std::endl;
            }
            stream.dec_tab();
            stream << "}" << std::endl;
          }

          stream << "barrier(CLK_LOCAL_MEM_FENCE); " << std::endl;
          stream << "if(lid==0){" << std::endl;
          stream.inc_tab();
          for(std::size_t k = 0 ; k < exprs.size() ; ++k){
            stream << "buf" << k << "[0] += buf" << k << "[1];" << std::endl;
            exprs[k]->access_name("buf"+utils::to_string(k)+"[0]");
          }

          std::size_t i = 0;
          for(statements_type::const_iterator it = statements_.begin() ; it != statements_.end() ; ++it){
            std::string str;
            detail::traverse(it->array(), detail::expression_generation_traversal(std::make_pair("0", "0"), -1, str, mapping_[i++]), false);
            stream << str << ";" << std::endl;
          }

          stream.dec_tab();
          stream << "}" << std::endl;
        }

      public:
        scalar_reduction(template_base::statements_type const & s, profile const & p) : template_base(s, profile_), profile_(p){

        }

        void core(std::size_t kernel_id, utils::kernel_generation_stream& stream) const {

          std::vector<detail::mapped_scalar_reduction*> exprs;
          for(std::vector<detail::mapping_type>::iterator it = mapping_.begin() ; it != mapping_.end() ; ++it)
            for(detail::mapping_type::iterator iit = it->begin() ; iit != it->end() ; ++iit)
              if(detail::mapped_scalar_reduction * p = dynamic_cast<detail::mapped_scalar_reduction*>(iit->second.get()))
                exprs.push_back(p);

          std::vector<const char *> scalartypes;
          fill_scalartypes(statements_, scalartypes);

          if(kernel_id==0){
            core_0(stream,exprs,scalartypes);
          }
          else{
            core_1(stream,exprs,scalartypes);
          }
        }

      private:
        profile profile_;
    };

  }

}

#endif
