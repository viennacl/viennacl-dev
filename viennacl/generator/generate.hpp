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
    @brief the user interface for the code generator
*/

#include <cstring>
#include <vector>
#include <typeinfo>

#include "viennacl/scheduler/forwards.h"
#include "viennacl/generator/forwards.h"

#include "viennacl/generator/autotuning/profiles.hpp"

#include "viennacl/generator/tree_parsing/statement_representation.hpp"
#include "viennacl/generator/tree_parsing/set_arguments.hpp"
#include "viennacl/generator/tree_parsing/map.hpp"

#include "viennacl/tools/tools.hpp"

namespace viennacl{

  namespace generator{


    /** @brief Class for handling code generation
     *
     *  It is meant to be only used along with the scheduler.*/
    class code_generator{
      public:
        /** @brief typedef of the key used in the forced profiles. Contains the expression type and the size of the scalartype */
        typedef std::pair<expression_type, std::size_t> forced_profile_key_type;
      private:
        typedef std::list< std::pair<scheduler::statement, scheduler::statement_node> > statements_type;
        typedef std::map<forced_profile_key_type, tools::shared_ptr<profile_base> > forced_profiles_type;

        /** @brief Fills the expression descriptor for an operation of the type scalar = RHS */
        static void fill_descriptor_impl(scheduler::statement const & statement, scheduler::statement_node const & root_node, expression_descriptor & descriptor){
          scheduler::statement::container_type const & expr = statement.array();
          if(descriptor.type_family == SCALAR_SAXPY_FAMILY && is_scalar_reduction(root_node)){
              descriptor.type_family = SCALAR_REDUCE_FAMILY;
              descriptor.type = SCALAR_REDUCE_TYPE;
          }
          else if(descriptor.type_family == VECTOR_SAXPY_FAMILY && is_vector_reduction(root_node)){
            if(  (root_node.op.type_family==scheduler::OPERATION_COLUMNS_REDUCTION_TYPE_FAMILY  && !utils::interpret_as_transposed(expr,root_node.lhs))
               ||(root_node.op.type_family==scheduler::OPERATION_ROWS_REDUCTION_TYPE_FAMILY && utils::interpret_as_transposed(expr,root_node.lhs))
               ||(root_node.op.type==scheduler::OPERATION_BINARY_MAT_VEC_PROD_TYPE && utils::interpret_as_transposed(statement.array(),root_node.lhs)))
              descriptor.type=VECTOR_REDUCE_Tx_TYPE;
            else
              descriptor.type=VECTOR_REDUCE_Nx_TYPE;
          }
          else if(descriptor.type_family == MATRIX_SAXPY_FAMILY && root_node.op.type == scheduler::OPERATION_BINARY_MAT_MAT_PROD_TYPE){
              descriptor.type_family=MATRIX_PRODUCT_FAMILY;
              bool lhs_trans = utils::interpret_as_transposed(expr,root_node.lhs);
              bool rhs_trans = utils::interpret_as_transposed(expr,root_node.rhs);
              if(!lhs_trans && !rhs_trans)
                descriptor.type=MATRIX_PRODUCT_NN_TYPE;
              else if(lhs_trans && !rhs_trans)
                descriptor.type=MATRIX_PRODUCT_TN_TYPE;
              else if(!lhs_trans && rhs_trans)
                descriptor.type=MATRIX_PRODUCT_NT_TYPE;
              else if(lhs_trans && rhs_trans)
                descriptor.type=MATRIX_PRODUCT_TT_TYPE;
          }
          else if(root_node.op.type_subfamily==scheduler::OPERATION_FUNCTION_TYPE_SUBFAMILY){
              descriptor.type_family=INVALID_EXPRESSION_FAMILY;
              descriptor.type=INVALID_EXPRESSION_TYPE;
          }

          if(descriptor.type_family!=INVALID_EXPRESSION_FAMILY && root_node.lhs.type_family==scheduler::COMPOSITE_OPERATION_FAMILY)
            fill_descriptor_impl(statement, expr[root_node.lhs.node_index],descriptor);
          if(descriptor.type_family!=INVALID_EXPRESSION_FAMILY && root_node.rhs.type_family==scheduler::COMPOSITE_OPERATION_FAMILY)
            fill_descriptor_impl(statement, expr[root_node.rhs.node_index],descriptor);
        }


        /** @brief Fills the expression descriptor for a statement */
        expression_descriptor fill_descriptor(scheduler::statement const & statement, scheduler::statement_node const & root_node){
          expression_descriptor res;
          scheduler::statement_node_type_family lhs_family = root_node.lhs.type_family;
          res.scalartype_size = utils::call_on_element(root_node.lhs, utils::scalartype_size_fun());
          if(lhs_family==scheduler::VECTOR_TYPE_FAMILY){
            res.type_family = VECTOR_SAXPY_FAMILY;
            res.type = VECTOR_SAXPY_TYPE;
            fill_descriptor_impl(statement,root_node,res);
          }
          else if(lhs_family==scheduler::MATRIX_TYPE_FAMILY){
            res.type_family = MATRIX_SAXPY_FAMILY;
            res.type = MATRIX_SAXPY_TYPE;
            fill_descriptor_impl(statement,root_node,res);
          }
          else if(lhs_family==scheduler::SCALAR_TYPE_FAMILY){
            res.type_family = SCALAR_SAXPY_FAMILY;
            res.type = SCALAR_SAXPY_TYPE;
            fill_descriptor_impl(statement,root_node,res);
          }
          return res;
        }

        /** @brief Sets the kernel arguments and enqueue the kernels associated with a list of statements.
        *
        *   The kernels are named 'kernel_'index of device in context'_'index of kernel in program'
        */
        template<class StatementsType>
        void set_expression_arguments(profile_base const * profile, unsigned int device_offset, StatementsType const & statements, unsigned int & kernel_id, viennacl::ocl::program & p, std::list<viennacl::ocl::kernel *> & kernels) const
        {
          for(std::size_t i = 0 ; i < profile->num_kernels() ; ++i){
            //add kernel name
            char str[32];
            std::sprintf(str,"kernel_%d_%d",device_offset,kernel_id);
            viennacl::ocl::kernel & kernel = p.get_kernel(str);
            kernels.push_back(&kernel);
            unsigned int current_arg = 0;
            //Configure ND Range and enqueue arguments
            profile->configure_range_enqueue_arguments(i, kernel, current_arg);
            std::set<void *> memory;
            for(typename StatementsType::const_iterator it = statements.begin() ; it != statements.end() ; ++it){
              tree_parsing::traverse(it->first, it->second, tree_parsing::set_arguments_functor(memory,current_arg,kernel));
            }
            ++kernel_id;
          }
        }

        /** @brief Gets the profile associated with a device and an expression descriptor */
        profile_base * get_profile(viennacl::ocl::device const & device, expression_descriptor const & descriptor)
        {
          forced_profiles_type::const_iterator it = forced_profiles_.find(std::make_pair(descriptor.type, descriptor.scalartype_size));
          if(it != forced_profiles_.end())
            return &(*it->second);
          return profiles::get(device,descriptor);
        }

      public:

        /** @brief The constructor */
        code_generator(viennacl::ocl::context const & ctx = viennacl::ocl::current_context()) : ctx_(ctx){
            descriptor_.type_family=INVALID_EXPRESSION_FAMILY;
            descriptor_.type=INVALID_EXPRESSION_TYPE;
        }

        /** @brief Force the generator to use a specific profile for an operation */
        template<class T>
        void force_profile(forced_profile_key_type key, T const & t)
        {
          forced_profiles_.insert(std::pair<forced_profile_key_type, tools::shared_ptr<profile_base> >(key, tools::shared_ptr<profile_base>(new T(t))));
        }

        /** @brief Add a statement and the root node to the expression list
        *   @return Whether or not the operation could be handled by the generator
        */
        bool add(scheduler::statement const & statement, scheduler::statement_node const & root_node)
        {
          expression_descriptor descriptor = fill_descriptor(statement, root_node);

          //Abort if the statement passed is not handled
          if(descriptor.type_family==INVALID_EXPRESSION_FAMILY)
            return false;

          //Init if not already done
          if(descriptor_.type_family==INVALID_EXPRESSION_FAMILY){
              descriptor_ = descriptor;
              for(std::vector<viennacl::ocl::device>::const_iterator it = ctx_.devices().begin() ; it != ctx_.devices().end() ; ++it)
                  profiles_.insert(std::make_pair(it->id(), get_profile(*it, descriptor_)));
              for(std::map<cl_device_id, profile_base*>::iterator it = profiles_.begin() ; it != profiles_.end() ; ++it)
                  it->second->bind_statements(&statements_);
          }
          //Abort if the current statement is incompatible with the current one
          else if(descriptor.type!=descriptor_.type)
              return false;

          statements_.push_back(std::make_pair(statement, root_node));
          return true;
        }

        /** @brief Set the arguments for a program previously generated by the generator and fills the kernels */
        void configure_program(viennacl::ocl::program & p, std::list<viennacl::ocl::kernel *> & kernels)
        {
          unsigned int kernel_id = 0;
          std::vector<viennacl::ocl::device>::const_iterator found = std::find(ctx_.devices().begin(),ctx_.devices().end(),ctx_.current_device());
          set_expression_arguments(profiles_.at(ctx_.current_device().id()), std::distance(ctx_.devices().begin(), found), statements_, kernel_id, p, kernels);
        }

        /** @brief Creates an identifier string for the set of expressions in the object */
        void make_program_name(char * program_name) const
        {
          unsigned int current_arg = 0;
          void* memory[64] = {NULL};
          for(std::list< std::pair<scheduler::statement, scheduler::statement_node> >::const_iterator it = statements_.begin() ; it != statements_.end() ; ++it)
              tree_parsing::traverse(it->first, it->second, tree_parsing::statement_representation_functor(memory, current_arg, program_name),true);
          *program_name='\0';
        }

        /** @brief Creates the OpenCL program string from the set of expressions in the object */
        std::string make_opencl_program_string()
        {
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
            (*profiles_.at(it->id()))(stream,device_offset++);

          return stream.str();
        }

        /** @brief Creates the CUDA device code from the set of expressions in the object
        *
        *   Performs just a direct translation...
        */
        std::string make_cuda_program_string()
        {
          //Creates OpenCL string with #ifdef and attributes
          utils::kernel_generation_stream stream;
          std::size_t device_offset =0;
          for(std::vector<viennacl::ocl::device>::const_iterator it = ctx_.devices().begin() ; it != ctx_.devices().end() ; ++it)
            (*profiles_.at(it->id()))(stream,device_offset++);
          std::string res = stream.str();

          viennacl::tools::find_and_replace(res,"__attribute__","//__attribute__");

          //Pointer
          viennacl::tools::find_and_replace(res, "__global float*", "float*");
          viennacl::tools::find_and_replace(res, "__local float*", "float*");

          viennacl::tools::find_and_replace(res, "__global double*", "double*");
          viennacl::tools::find_and_replace(res, "__local double*", "double*");

          //Qualifiers
          viennacl::tools::find_and_replace(res,"__global","__device__");
          viennacl::tools::find_and_replace(res,"__kernel","__global__");
          viennacl::tools::find_and_replace(res,"__constant","__constant__");
          viennacl::tools::find_and_replace(res,"__local","__shared__");

          //Indexing
          viennacl::tools::find_and_replace(res,"get_num_groups(0)","gridDim.x");
          viennacl::tools::find_and_replace(res,"get_num_groups(1)","gridDim.y");

          viennacl::tools::find_and_replace(res,"get_local_size(0)","blockDim.x");
          viennacl::tools::find_and_replace(res,"get_local_size(1)","blockDim.y");

          viennacl::tools::find_and_replace(res,"get_group_id(0)","blockIdx.x");
          viennacl::tools::find_and_replace(res,"get_group_id(1)","blockIdx.y");

          viennacl::tools::find_and_replace(res,"get_local_id(0)","threadIdx.x");
          viennacl::tools::find_and_replace(res,"get_local_id(1)","threadIdx.y");

          viennacl::tools::find_and_replace(res,"get_global_id(0)","(blockIdx.x*blockDim.x + threadIdx.x)");
          viennacl::tools::find_and_replace(res,"get_global_id(1)","(blockIdx.y*blockDim.y + threadIdx.y)");

          //Synchronization
          viennacl::tools::find_and_replace(res,"barrier(CLK_LOCAL_MEM_FENCE)","__syncthreads()");
          viennacl::tools::find_and_replace(res,"barrier(CLK_GLOBAL_MEM_FENCE)","__syncthreads()");


          return res;
        }

      private:
        statements_type statements_;
        std::map<cl_device_id, profile_base *> profiles_;
        viennacl::ocl::context const & ctx_;
        expression_descriptor descriptor_;
        forced_profiles_type forced_profiles_;
    };

    /** @brief Creates the program associated with a generator object and fills the kernels. Checks the context for the program and possibly (re)compile it.
    *
    *   @param generator the generator to work on
    *   @param kernels this list will be filled with the kernels associated with the generator
    *   @param force_recompilation if true, the program will be recompiled
    */
    inline viennacl::ocl::program & get_configured_program(viennacl::generator::code_generator & generator, std::list<viennacl::ocl::kernel*> & kernels, bool force_recompilation = false){
      std::vector<char> program_name_vector(256);
      char* program_name = program_name_vector.data();
      generator.make_program_name(program_name);
      if(force_recompilation)
        viennacl::ocl::current_context().delete_program(program_name);
      if(!viennacl::ocl::current_context().has_program(program_name)){
        std::string source_code = generator.make_opencl_program_string();
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

    /** @brief Set the arguments and enqueue a generator object */
    inline void enqueue(viennacl::generator::code_generator & generator, bool force_recompilation = false){
      std::list<viennacl::ocl::kernel*> kernels;
      get_configured_program(generator, kernels, force_recompilation);
      for(std::list<viennacl::ocl::kernel*>::iterator it = kernels.begin() ; it != kernels.end() ; ++it){
        viennacl::ocl::enqueue(**it, (*it)->context().get_queue());
      }
    }

    /** @brief Convenience function to get the OpenCL program string for a single statement */
    inline std::string get_opencl_program_string(viennacl::scheduler::statement const & s){
      generator::code_generator gen;
      gen.add(s,s.array()[0]);
      return gen.make_opencl_program_string();
    }

    /** @brief Convenience function to get the CUDA device code for a single statement */
    inline std::string get_cuda_device_code(viennacl::scheduler::statement const & s){
      generator::code_generator gen;
      gen.add(s, s.array()[0]);
      return gen.make_cuda_program_string();
    }

    /** @brief Generate and enqueue a statement+root_node into the current queue */
    inline void generate_enqueue_statement(viennacl::scheduler::statement const & s, scheduler::statement_node const & root_node){
      generator::code_generator gen;
      gen.add(s,root_node);
      viennacl::generator::enqueue(gen);
    }

    /** @brief Generate and enqueue a statement into the current queue, assumes the root_node is the first node of the statement */
    inline void generate_enqueue_statement(viennacl::scheduler::statement const & s){
      generate_enqueue_statement(s, s.array()[0]);
    }

  }
}
#endif
