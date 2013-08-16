#ifndef VIENNACL_GENERATOR_GENERATE_HPP
#define VIENNACL_GENERATOR_GENERATE_HPP

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


/** @file viennacl/generator/generate.hpp
    @brief User interface
*/

#include <cstring>
#include <vector>
#include <typeinfo>

#include "viennacl/scheduler/forwards.h"
#include "viennacl/generator/forwards.h"

#include "viennacl/generator/profiles.hpp"
#include "viennacl/generator/enqueue_statement.hpp"
#include "viennacl/generator/statement_representation.hpp"

#include "viennacl/generator/map_generate_prototype.hpp"

namespace viennacl{

  namespace generator{

    using namespace scheduler;

    class code_generator{
      public:
        typedef std::pair<expression_type, std::size_t> forced_profile_key_type;
      private:
        typedef std::pair<expression_descriptor, generator::profile_base::statements_type> representation_node_type;
        typedef std::vector<representation_node_type> statements_type;
        typedef std::map<forced_profile_key_type, tools::shared_ptr<profile_base> > forced_profiles_type;


        static bool is_flow_transposed(scheduler::statement const & statement, scheduler::statement_node const & root_node){
          scheduler::statement::container_type const & expr = statement.array();
          if(root_node.op.type==scheduler::OPERATION_UNARY_TRANS_TYPE)
            return root_node.lhs.subtype==scheduler::DENSE_ROW_MATRIX_TYPE;
          else{
            bool res = root_node.lhs.subtype==scheduler::DENSE_COL_MATRIX_TYPE || root_node.rhs.subtype==scheduler::DENSE_COL_MATRIX_TYPE;
            if(root_node.lhs.type_family==scheduler::COMPOSITE_OPERATION_FAMILY)
              res = res || is_lhs_flow_transposed(statement, expr[root_node.lhs.node_index]);
            if(root_node.rhs.type_family==scheduler::COMPOSITE_OPERATION_FAMILY)
              res = res || is_lhs_flow_transposed(statement, expr[root_node.rhs.node_index]);
            return res;
          }
        }

        static bool is_lhs_flow_transposed(scheduler::statement const & statement, scheduler::statement_node const & root_node){
          scheduler::statement::container_type const & expr = statement.array();
          if(root_node.lhs.type_family==COMPOSITE_OPERATION_FAMILY)
            return is_flow_transposed(statement, expr[root_node.lhs.node_index]);
          else
            return root_node.lhs.subtype==scheduler::DENSE_COL_MATRIX_TYPE;
        }

        static bool is_rhs_flow_transposed(scheduler::statement const & statement, scheduler::statement_node const & root_node){
          scheduler::statement::container_type const & expr = statement.array();
          if(root_node.rhs.type_family==COMPOSITE_OPERATION_FAMILY)
            return is_flow_transposed(statement, expr[root_node.rhs.node_index]);
          else
            return root_node.rhs.subtype==scheduler::DENSE_COL_MATRIX_TYPE;
        }

        static void fill_expression_descriptor_scalar(scheduler::statement const & statement, scheduler::statement_node const & root_node, expression_descriptor & descriptor){
          scheduler::statement::container_type const & expr = statement.array();
          bool is_invalid = (root_node.op.type == OPERATION_BINARY_MAT_VEC_PROD_TYPE)
                          || (descriptor.type_family==SCALAR_REDUCE_FAMILY && root_node.op.type == OPERATION_BINARY_INNER_PROD_TYPE);
          if(is_invalid){
            descriptor.type_family =INVALID_EXPRESSION_FAMILY;
            descriptor.type = INVALID_EXPRESSION_TYPE;
          }
          else if(root_node.op.type==OPERATION_BINARY_INNER_PROD_TYPE){
            descriptor.type_family = SCALAR_REDUCE_FAMILY;
            descriptor.type = SCALAR_REDUCE_TYPE;
          }
          if(descriptor.type_family!=INVALID_EXPRESSION_FAMILY && root_node.lhs.type_family==scheduler::COMPOSITE_OPERATION_FAMILY)
            fill_expression_descriptor_scalar(statement, expr[root_node.lhs.node_index],descriptor);
          if(descriptor.type_family!=INVALID_EXPRESSION_FAMILY && root_node.rhs.type_family==scheduler::COMPOSITE_OPERATION_FAMILY)
            fill_expression_descriptor_scalar(statement, expr[root_node.rhs.node_index],descriptor);
        }

        static void fill_expression_descriptor_vector(scheduler::statement const & statement, scheduler::statement_node const & root_node, expression_descriptor & descriptor){
          scheduler::statement::container_type const & expr = statement.array();
          bool is_invalid =  (root_node.op.type == OPERATION_BINARY_INNER_PROD_TYPE)
                          || (root_node.op.type == OPERATION_BINARY_MAT_MAT_PROD_TYPE)
                          || (descriptor.type_family==VECTOR_REDUCE_FAMILY && root_node.op.type == OPERATION_BINARY_MAT_VEC_PROD_TYPE);
          if(is_invalid){
            descriptor.type_family=INVALID_EXPRESSION_FAMILY;
            descriptor.type=INVALID_EXPRESSION_TYPE;
          }
          else if(root_node.op.type==OPERATION_BINARY_MAT_VEC_PROD_TYPE){
            descriptor.type_family=VECTOR_REDUCE_FAMILY;
            if(is_lhs_flow_transposed(statement,root_node))
              descriptor.type=VECTOR_REDUCE_Tx_TYPE;
            else
              descriptor.type=VECTOR_REDUCE_Ax_TYPE;
          }
          if(descriptor.type_family!=INVALID_EXPRESSION_FAMILY && root_node.lhs.type_family==scheduler::COMPOSITE_OPERATION_FAMILY)
            fill_expression_descriptor_vector(statement, expr[root_node.lhs.node_index],descriptor);
          if(descriptor.type_family!=INVALID_EXPRESSION_FAMILY && root_node.rhs.type_family==scheduler::COMPOSITE_OPERATION_FAMILY)
            fill_expression_descriptor_vector(statement, expr[root_node.rhs.node_index],descriptor);
        }

        static void fill_expression_descriptor_matrix(scheduler::statement const & statement, scheduler::statement_node const & root_node, expression_descriptor & descriptor){
          scheduler::statement::container_type const & expr = statement.array();
          bool is_invalid =  (root_node.op.type == OPERATION_BINARY_INNER_PROD_TYPE)
                          || (root_node.op.type == OPERATION_BINARY_MAT_VEC_PROD_TYPE)
                          || (descriptor.type_family==MATRIX_PRODUCT_FAMILY && root_node.op.type == OPERATION_BINARY_MAT_MAT_PROD_TYPE);
          if(is_invalid){
            descriptor.type_family=INVALID_EXPRESSION_FAMILY;
            descriptor.type=INVALID_EXPRESSION_TYPE;
          }
          else if(root_node.op.type==OPERATION_BINARY_MAT_MAT_PROD_TYPE){
            descriptor.type_family=MATRIX_PRODUCT_FAMILY;
            bool lhs_trans = is_lhs_flow_transposed(statement,root_node);
            bool rhs_trans = is_rhs_flow_transposed(statement,root_node);
            if(!lhs_trans && !rhs_trans)
              descriptor.type=MATRIX_PRODUCT_AA_TYPE;
            else if(lhs_trans && !rhs_trans)
              descriptor.type=MATRIX_PRODUCT_TA_TYPE;
            else if(!lhs_trans && rhs_trans)
              descriptor.type=MATRIX_PRODUCT_AT_TYPE;
            else if(lhs_trans && rhs_trans)
              descriptor.type=MATRIX_PRODUCT_TT_TYPE;

          }
          if(descriptor.type_family!=INVALID_EXPRESSION_FAMILY && root_node.lhs.type_family==scheduler::COMPOSITE_OPERATION_FAMILY)
            fill_expression_descriptor_matrix(statement, expr[root_node.lhs.node_index],descriptor);
          if(descriptor.type_family!=INVALID_EXPRESSION_FAMILY && root_node.rhs.type_family==scheduler::COMPOSITE_OPERATION_FAMILY)
            fill_expression_descriptor_matrix(statement, expr[root_node.rhs.node_index],descriptor);
        }

        void fill_descriptor(scheduler::statement const & statement, scheduler::statement_node const & root_node, expression_descriptor & descriptor){
          scheduler::statement_node_type_family lhs_family = root_node.lhs.type_family;
          descriptor.scalartype_size = utils::call_on_element(root_node.lhs, utils::scalartype_size_fun());
          if(lhs_family==VECTOR_TYPE_FAMILY){
            descriptor.type_family = VECTOR_SAXPY_FAMILY;
            descriptor.type = VECTOR_SAXPY_TYPE;
            fill_expression_descriptor_vector(statement,root_node,descriptor);
          }
          else if(lhs_family==MATRIX_TYPE_FAMILY){
            descriptor.type_family = MATRIX_SAXPY_FAMILY;
            descriptor.type = MATRIX_SAXPY_TYPE;
            fill_expression_descriptor_matrix(statement,root_node,descriptor);
          }
          else if(lhs_family==SCALAR_TYPE_FAMILY){
            descriptor.type_family = SCALAR_SAXPY_FAMILY;
            descriptor.type = SCALAR_SAXPY_TYPE;
            fill_expression_descriptor_scalar(statement,root_node,descriptor);
          }
        }

        template<class StatementsType>
        void enqueue_expression(profile_base const & profile, unsigned int device_offset, StatementsType const & statements, unsigned int & kernel_id, viennacl::ocl::program & p, std::list<viennacl::ocl::kernel *> & kernels) const {
          for(std::size_t i = 0 ; i < profile.num_kernels() ; ++i){
            //add kernel name
            char str[32];
            std::sprintf(str,"kernel_%d_%d",device_offset,kernel_id);
            viennacl::ocl::kernel & kernel = p.get_kernel(str);
            kernels.push_back(&kernel);
            unsigned int current_arg = 0;
            //Configure ND Range and enqueue arguments
            profile.configure_range_enqueue_arguments(i, statements, kernel, current_arg);
            std::set<void *> memory;
            for(typename StatementsType::const_iterator it = statements.begin() ; it != statements.end() ; ++it){
              detail::traverse(it->first, it->second, detail::enqueue_functor(memory,current_arg,kernel));
            }
            ++kernel_id;
          }
        }

        profile_base const & get_profile(viennacl::ocl::device const & device, expression_descriptor const & descriptor) const {
          forced_profiles_type::const_iterator it = forced_profiles_.find(std::make_pair(descriptor.type, descriptor.scalartype_size));
          if(it != forced_profiles_.end())
            return *it->second;
          return *profiles::get(device,descriptor);
        }

      public:
        code_generator(viennacl::ocl::context const & ctx = viennacl::ocl::current_context()) : ctx_(ctx){
          statements_.reserve(16);
        }

        template<class T>
        void force_profile(forced_profile_key_type key, T const & t){
          forced_profiles_.insert(std::make_pair(key, new T(t)));
        }

        bool add(scheduler::statement const & statement, scheduler::statement_node const & root_node) {
          expression_descriptor descriptor;
          fill_descriptor(statement, root_node, descriptor);
          if(descriptor.type_family==INVALID_EXPRESSION_FAMILY)
            return false;
          if(statements_.empty())
            statements_.push_back(std::make_pair(descriptor,profile_base::statements_type(1,std::make_pair(statement, root_node))));
          else
            if(statements_.back().first == descriptor)
              statements_.back().second.push_back(std::make_pair(statement, root_node));
            else
              statements_.push_back(std::make_pair(descriptor,profile_base::statements_type(1,std::make_pair(statement, root_node))));
          return true;
        }

        void configure_program(viennacl::ocl::program & p, std::list<viennacl::ocl::kernel *> & kernels) const {
          unsigned int kernel_id = 0;
          std::vector<viennacl::ocl::device>::const_iterator found = std::find(ctx_.devices().begin(),ctx_.devices().end(),ctx_.current_device());
          for(statements_type::const_iterator it = statements_.begin() ; it != statements_.end() ; ++it)
            enqueue_expression(get_profile(ctx_.current_device(), it->first), std::distance(ctx_.devices().begin(), found), it->second, kernel_id, p, kernels);
        }

        void make_program_name(char * program_name) const {
          unsigned int current_arg = 0;
          void* memory[64] = {NULL};
          for(statements_type::const_iterator it = statements_.begin() ; it != statements_.end() ; ++it){
            for(profile_base::statements_type::const_iterator iit = it->second.begin() ; iit != it->second.end() ; ++iit){
              detail::traverse(iit->first, iit->second, detail::representation_functor(memory, current_arg, program_name));
            }
          }
          *program_name='\0';
        }

        std::string make_program_string() const {
          utils::kernel_generation_stream stream;

          //Headers generation
          stream << "#if defined(cl_khr_fp64)\n";
          stream <<  "#  pragma OPENCL EXTENSION cl_khr_fp64: enable\n";
          stream <<  "#elif defined(cl_amd_fp64)\n";
          stream <<  "#  pragma OPENCL EXTENSION cl_amd_fp64: enable\n";
          stream <<  "#endif\n";
          stream << std::endl;

          std::size_t device_offset =0;
          for(std::vector<viennacl::ocl::device>::const_iterator it = ctx_.devices().begin() ; it != ctx_.devices().end() ; ++it)
            for(statements_type::const_iterator iit = statements_.begin() ; iit != statements_.end() ; ++iit)
              get_profile(*it,iit->first)(stream,device_offset++,iit->second);

          return stream.str();
        }

      private:
        statements_type statements_;
        viennacl::ocl::context const & ctx_;
        forced_profiles_type forced_profiles_;
    };

    static viennacl::ocl::program & get_configured_program(viennacl::generator::code_generator const & generator, std::list<viennacl::ocl::kernel*> & kernels, bool force_recompilation = false){
      char* program_name = (char*)malloc(256*sizeof(char));
      generator.make_program_name(program_name);
      if(force_recompilation)
        viennacl::ocl::current_context().delete_program(program_name);
      if(!viennacl::ocl::current_context().has_program(program_name)){
        std::string source_code = generator.make_program_string();
    #ifdef VIENNACL_DEBUG_BUILD
        std::cout << "Building " << program_name << "..." << std::endl;
        std::cout << source_code << std::endl;
    #endif
        viennacl::ocl::current_context().add_program(source_code, program_name);
      }
      viennacl::ocl::program & p = viennacl::ocl::current_context().get_program(program_name);
      generator.configure_program(p, kernels);
      return p;
    }

    inline void enqueue(viennacl::generator::code_generator const & generator, bool force_recompilation = false){
      std::list<viennacl::ocl::kernel*> kernels;
      get_configured_program(generator, kernels, force_recompilation);
      for(std::list<viennacl::ocl::kernel*>::iterator it = kernels.begin() ; it != kernels.end() ; ++it){
        viennacl::ocl::enqueue(**it, (*it)->context().get_queue());
      }
    }

    inline void generate_enqueue_statement(viennacl::scheduler::statement const & s, scheduler::statement_node const & root_node){
      generator::code_generator gen;
      gen.add(s,root_node);
      viennacl::generator::enqueue(gen);
    }

    inline void generate_enqueue_statement(viennacl::scheduler::statement const & s){
      generate_enqueue_statement(s, s.array()[0]);
    }

  }
}
#endif
