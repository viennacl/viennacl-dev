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

#include "viennacl/generator/enqueue_statement.hpp"
#include "viennacl/generator/statement_representation.hpp"

#include "viennacl/generator/map_generate_prototype.hpp"

#include "viennacl/generator/generate_saxpy.hpp"
#include "viennacl/generator/generate_scalar_reduction.hpp"
#include "viennacl/generator/generate_vector_reduction.hpp"
#include "viennacl/generator/generate_matrix_product.hpp"

namespace viennacl{

  namespace generator{

    using namespace scheduler;

    enum expression_type_family{
      SCALAR_SAXPY_FAMILY,
      VECTOR_SAXPY_FAMILY,
      MATRIX_SAXPY_FAMILY,
      SCALAR_REDUCE_FAMILY,
      VECTOR_REDUCE_FAMILY,
      MATRIX_PRODUCT_FAMILY,
      INVALID_EXPRESSION_FAMILY
    };

    enum expression_type{
      SCALAR_SAXPY_TYPE,
      VECTOR_SAXPY_TYPE,
      MATRIX_SAXPY_TYPE,
      SCALAR_REDUCE_TYPE,
      VECTOR_REDUCE_Ax_TYPE,
      VECTOR_REDUCE_Tx_TYPE,
      MATRIX_PRODUCT_AA_TYPE,
      MATRIX_PRODUCT_TA_TYPE,
      MATRIX_PRODUCT_AT_TYPE,
      MATRIX_PRODUCT_TT_TYPE,
      INVALID_EXPRESSION_TYPE
    };

    struct expression_descriptor{
        expression_type_family type_family;
        expression_type type;
        bool operator==(expression_descriptor const & other) const{
          return type_family==other.type_family
               &&type==other.type;
        }
    };

    class code_generator{
      private:
        typedef std::pair<expression_descriptor, generator::template_base::statements_type> representation_node_type;
        typedef std::vector<representation_node_type> statements_type;

        static bool is_transposed(scheduler::statement const & statement, scheduler::statement_node const & root_node){
          scheduler::statement::container_type const & expr = statement.array();
          if(root_node.op.type==scheduler::OPERATION_UNARY_TRANS_TYPE)
            return true;
          else{
            bool res = false;
            if(root_node.lhs.type_family==scheduler::COMPOSITE_OPERATION_FAMILY)
              res = res || is_lhs_transposed(statement, expr[root_node.lhs.node_index]);
            if(root_node.rhs.type_family==scheduler::COMPOSITE_OPERATION_FAMILY)
              res = res || is_lhs_transposed(statement, expr[root_node.rhs.node_index]);
            return res;
          }
        }

        static bool is_lhs_transposed(scheduler::statement const & statement, scheduler::statement_node const & root_node){
          scheduler::statement::container_type const & expr = statement.array();
          if(root_node.lhs.type_family==COMPOSITE_OPERATION_FAMILY)
            return is_transposed(statement, expr[root_node.lhs.node_index]);
          else
            return false;
        }

        static bool is_rhs_transposed(scheduler::statement const & statement, scheduler::statement_node const & root_node){
          scheduler::statement::container_type const & expr = statement.array();
          if(root_node.rhs.type_family==COMPOSITE_OPERATION_FAMILY)
            return is_transposed(statement, expr[root_node.rhs.node_index]);
          else
            return false;
        }

        static void fill_expression_descriptor_scalar(scheduler::statement const & statement, scheduler::statement_node const & root_node, expression_descriptor & descriptor){
          scheduler::statement::container_type const & expr = statement.array();
          bool is_invalid = (root_node.op.type == OPERATION_BINARY_MAT_VEC_PROD_TYPE)
                          || (descriptor.type_family==SCALAR_REDUCE_FAMILY && root_node.op.type == OPERATION_BINARY_INNER_PROD_TYPE);
          if(is_invalid){
            descriptor.type_family =INVALID_EXPRESSION_FAMILY;
            descriptor.type == INVALID_EXPRESSION_TYPE;
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
            if(is_lhs_transposed(statement,root_node))
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
            bool lhs_trans = is_lhs_transposed(statement,root_node);
            bool rhs_trans = is_rhs_transposed(statement,root_node);
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
          if(lhs_family==VECTOR_TYPE_FAMILY){
            descriptor.type_family = VECTOR_SAXPY_FAMILY;
            descriptor.type = VECTOR_SAXPY_TYPE;
            fill_expression_descriptor_vector(statement,root_node,descriptor);
          }
          else if(lhs_family==MATRIX_ROW_TYPE_FAMILY || lhs_family==MATRIX_COL_TYPE_FAMILY){
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
        void enqueue_expression(template_base::profile const & profile, StatementsType const & statements, unsigned int & kernel_id, viennacl::ocl::program & p, std::list<viennacl::ocl::kernel *> & kernels) const {
          for(std::size_t i = 0 ; i < profile.num_kernels() ; ++i){
            //add kernel name
            char str[10];
            std::sprintf(str,"kernel_%d",kernel_id);
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

      public:
        code_generator() : vector_saxpy_profile_(1,128,128,true)
                          , matrix_saxpy_profile_(1,16,16,16,16,true)
                          , scalar_reduction_profile_(4, 128, 512, 0)
                          , vector_reduction_profile_(1, 1, 256, 32)
                          , matrix_product_profile_(2,16,32,64,8,4,2,1,0,1)
                           {
          statements_.reserve(16);
        }

        bool add(scheduler::statement const & statement, scheduler::statement_node const & root_node) {
          expression_descriptor descriptor;
          fill_descriptor(statement, root_node, descriptor);

          if(descriptor.type_family==INVALID_EXPRESSION_FAMILY)
            return false;

          if(statements_.empty())
            statements_.push_back(std::make_pair(descriptor,template_base::statements_type(1,std::make_pair(statement, root_node))));
          else
            if(statements_.back().first == descriptor)
              statements_.back().second.push_back(std::make_pair(statement, root_node));
            else
              statements_.push_back(std::make_pair(descriptor,template_base::statements_type(1,std::make_pair(statement, root_node))));
          return true;
        }

        template_base::profile const & get_profile(expression_descriptor descriptor) const {
          expression_type_family family = descriptor.type_family;
          expression_type type = descriptor.type;
          switch(family){
            case VECTOR_SAXPY_FAMILY: return vector_saxpy_profile_;
            case MATRIX_SAXPY_FAMILY: return matrix_saxpy_profile_;
            case SCALAR_REDUCE_FAMILY: return scalar_reduction_profile_;
            case VECTOR_REDUCE_FAMILY:
              switch(type){
                case VECTOR_REDUCE_Ax_TYPE: return vector_reduction_profile_;
                case VECTOR_REDUCE_Tx_TYPE: return vector_reduction_profile_;
                default: throw "vector reduction profile type not recognized";
              }
              break;
            case MATRIX_PRODUCT_FAMILY:
              switch(type){
                case MATRIX_PRODUCT_AA_TYPE: return matrix_product_profile_;
                case MATRIX_PRODUCT_TA_TYPE: return matrix_product_profile_;
                case MATRIX_PRODUCT_AT_TYPE: return matrix_product_profile_;
                case MATRIX_PRODUCT_TT_TYPE: return matrix_product_profile_;
                default: throw "matrix reduction profile type not recognized";
              }
              break;
            default: throw "profile type family not recognized";
          }
        }

        void configure_program(viennacl::ocl::program & p, std::list<viennacl::ocl::kernel *> & kernels) const {
          unsigned int kernel_id = 0;
          for(statements_type::const_iterator it = statements_.begin() ; it != statements_.end() ; ++it)
            enqueue_expression(get_profile(it->first), it->second, kernel_id, p, kernels);
        }

        void make_program_name(char * program_name) const {
          unsigned int current_arg = 0;
          void* memory[64] = {NULL};
          for(statements_type::const_iterator it = statements_.begin() ; it != statements_.end() ; ++it){
            for(template_base::statements_type::const_iterator iit = it->second.begin() ; iit != it->second.end() ; ++iit){
              detail::traverse(iit->first, iit->second, detail::representation_functor(memory, current_arg, program_name));
            }
          }
          *program_name='\0';
        }


        void force_profile(vector_saxpy::profile const & profile){
          vector_saxpy_profile_ = profile;
        }

        void force_profile(matrix_saxpy::profile const & profile){
          matrix_saxpy_profile_ = profile;
        }

        void force_profile(scalar_reduction::profile const & profile){
          scalar_reduction_profile_ = profile;
        }

        void force_profile(vector_reduction::profile const & profile){
          vector_reduction_profile_ = profile;
        }

        void force_profile(matrix_product::profile const & profile){
          matrix_product_profile_ = profile;
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

          for(statements_type::const_iterator it = statements_.begin() ; it != statements_.end() ; ++it){
            expression_type_family family = it->first.type_family;
            expression_type type = it->first.type;
            representation_node_type::second_type const & s = it->second;
            switch(family){
              case VECTOR_SAXPY_FAMILY: vector_saxpy(s,vector_saxpy_profile_)(stream); break;
              case MATRIX_SAXPY_FAMILY: matrix_saxpy(s,matrix_saxpy_profile_)(stream); break;
              case SCALAR_REDUCE_FAMILY: scalar_reduction(s,scalar_reduction_profile_)(stream); break;
              case VECTOR_REDUCE_FAMILY:
                switch(type){
                  case VECTOR_REDUCE_Ax_TYPE: vector_reduction(s,vector_reduction_profile_)(stream); break;
                  case VECTOR_REDUCE_Tx_TYPE: vector_reduction(s,vector_reduction_profile_)(stream); break;
                  default: throw "vector reduction profile type not recognized";
                }
                break;
              case MATRIX_PRODUCT_FAMILY:
                switch(type){
                  case MATRIX_PRODUCT_AA_TYPE: matrix_product(s,matrix_product_profile_)(stream); break;
                  case MATRIX_PRODUCT_TA_TYPE: matrix_product(s,matrix_product_profile_)(stream); break;
                  case MATRIX_PRODUCT_AT_TYPE: matrix_product(s,matrix_product_profile_)(stream); break;
                  case MATRIX_PRODUCT_TT_TYPE: matrix_product(s,matrix_product_profile_)(stream); break;
                  default: throw "matrix reduction profile type not recognized";
                }
                break;
              default: throw "profile type family not recognized";
            }
          }
          return stream.str();
        }

      private:
        statements_type statements_;

        vector_saxpy::profile vector_saxpy_profile_;
        matrix_saxpy::profile matrix_saxpy_profile_;
        scalar_reduction::profile scalar_reduction_profile_;
        vector_reduction::profile vector_reduction_profile_;
        matrix_product::profile matrix_product_profile_;
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

    static void enqueue(viennacl::generator::code_generator const & generator, bool force_recompilation = false){
      std::list<viennacl::ocl::kernel*> kernels;
      get_configured_program(generator, kernels, force_recompilation);
      for(std::list<viennacl::ocl::kernel*>::iterator it = kernels.begin() ; it != kernels.end() ; ++it){
        viennacl::ocl::enqueue(**it, (*it)->context().get_queue());
      }
    }

    static void generate_enqueue_statement(viennacl::scheduler::statement const & s, scheduler::statement_node const & root_node){
      generator::code_generator gen;
      gen.add(s,root_node);
      viennacl::generator::enqueue(gen);
    }

  }
}
#endif
