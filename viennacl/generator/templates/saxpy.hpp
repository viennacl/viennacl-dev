#ifndef VIENNACL_GENERATOR_CODE_GENERATION_TEMPLATES_SAXPY_HPP
#define VIENNACL_GENERATOR_CODE_GENERATION_TEMPLATES_SAXPY_HPP

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


/** @file viennacl/generator/templates/saxpy.hpp
 *
 * Kernel template for the SAXPY operation
*/

#include "viennacl/tools/tools.hpp"

#include "viennacl/generator/templates/base_classes.hpp"
#include "viennacl/generator/symbolic_types.hpp"

namespace viennacl{

  namespace generator{

    namespace code_generation{

      namespace saxpy{

        class profile : public optimization_profile{
          public:
            profile(){
              loop_unroll_ = 1;
              group_size0_ = 128;
            }

            profile(unsigned int vectorization, unsigned int loop_unroll, size_t group_size0) : optimization_profile(vectorization){
              loop_unroll_ = loop_unroll;
              group_size0_ = group_size0;
            }

            unsigned int loop_unroll() const{
              return loop_unroll_;
            }

            std::pair<size_t,size_t> local_work_size() const{
              return std::make_pair(group_size0_,1);
            }

            void config_nd_range(viennacl::ocl::kernel & k, symbolic_expression_tree_base* p){
              k.local_work_size(0,group_size0_);
              if(symbolic_vector_base* vec = dynamic_cast<symbolic_vector_base*>(p)){
                k.global_work_size(0,viennacl::tools::roundUpToNextMultiple<cl_uint>(vec->real_size()/(vectorization_*loop_unroll_),group_size0_)); //Note: now using for-loop for good performance on CPU
              }
              else if(symbolic_matrix_base * mat = dynamic_cast<symbolic_matrix_base*>(p)){
                k.global_work_size(0,viennacl::tools::roundUpToNextMultiple<cl_uint>(mat->real_size1() * mat->real_size2()/(vectorization_*loop_unroll_),group_size0_));
              }
            }


            std::string repr() const{
              std::ostringstream oss;
              oss << "V" << vectorization_
                  <<  "U" << loop_unroll_
                   << "GROUP" << group_size0_;
              return oss.str();
            }

            bool is_invalid(viennacl::ocl::device const & dev, size_t scalartype_size){
              return optimization_profile::is_invalid(dev,0);
            }

          private:
            unsigned int loop_unroll_;
            unsigned int group_size0_;
        };

        class generator : public code_generation::generator{
          public:
            generator(std::list<symbolic_binary_vector_expression_base* > const & vector_expressions
                      ,std::list<symbolic_binary_scalar_expression_base *> const & scalar_expressions
                      ,std::list<symbolic_binary_matrix_expression_base * > const & matrix_expressions
                      ,profile * kernel_config): vector_expressions_(vector_expressions), matrix_expressions_(matrix_expressions), scalar_expressions_(scalar_expressions), profile_(kernel_config)
            {
            }


            void operator()(utils::kernel_generation_stream& kss){

              unsigned int n_unroll = profile_->loop_unroll();
              symbolic_vector_base * first_vector =  NULL;
              symbolic_matrix_base * first_matrix = NULL;
              if(vector_expressions_.size()) first_vector = static_cast<symbolic_vector_base*>(&(*vector_expressions_.begin())->lhs());
              if(matrix_expressions_.size()) first_matrix = static_cast<symbolic_matrix_base*>(&(*matrix_expressions_.begin())->lhs());
              if(first_vector){
                kss << "int i = get_global_id(0)" ; if(n_unroll>1) kss << "*" << n_unroll; kss << ";" << std::endl;
                //            kss << "if(i < " << first_vector->size() << "){" << std::endl;
                kss.inc_tab();


                //Set access indices
                for(std::list<symbolic_binary_vector_expression_base*>::iterator it=vector_expressions_.begin() ; it!=vector_expressions_.end();++it){
                  for(unsigned int j=0 ; j < n_unroll ; ++j){
                    if(j==0) (*it)->access_index(j,"i","0");
                    else (*it)->access_index(j,"i + " + utils::to_string(j),"0");
                    (*it)->fetch(j,kss);
                  }
                }

                //Compute expressions
                for(std::list<symbolic_binary_vector_expression_base*>::iterator it=vector_expressions_.begin() ; it!=vector_expressions_.end();++it){
                  for(unsigned int j=0 ; j < n_unroll ; ++j){
                    kss << (*it)->generate(j) << ";" << std::endl;
                  }
                }

                for(std::list<symbolic_binary_vector_expression_base*>::iterator it=vector_expressions_.begin() ; it!=vector_expressions_.end();++it){
                  for(unsigned int j=0 ; j < n_unroll ; ++j){
                    (*it)->write_back(j,kss);
                  }
                }

                kss.dec_tab();
                //            kss << "}" << std::endl;
              }
              if(first_matrix){
                if(first_matrix->is_rowmajor()){
                  kss << "unsigned int r = get_global_id(0)/" << first_matrix->internal_size2() << ";" << std::endl;
                  kss << "unsigned int c = get_global_id(0)%" << first_matrix->internal_size2() << ";" << std::endl;
                }
                else{
                  kss << "unsigned int r = get_global_id(0)%" << first_matrix->internal_size1() << ";" << std::endl;
                  kss << "unsigned int c = get_global_id(0)/" << first_matrix->internal_size1() << ";" << std::endl;
                }
                kss << "if(r < " << first_matrix->internal_size1() << "){" << std::endl;
                kss.inc_tab();

                //Set access indices
                for(std::list<symbolic_binary_matrix_expression_base*>::iterator it=matrix_expressions_.begin() ; it!=matrix_expressions_.end();++it){
                  (*it)->access_index(0,"r","c");
                  (*it)->fetch(0,kss);
                }

                //Compute expressions
                for(std::list<symbolic_binary_matrix_expression_base*>::iterator it = matrix_expressions_.begin(); it!=matrix_expressions_.end(); ++it)
                  kss << (*it)->generate(0) << ";" << std::endl;

                for(std::list<symbolic_binary_matrix_expression_base*>::iterator it=matrix_expressions_.begin() ; it!=matrix_expressions_.end();++it)
                  (*it)->write_back(0,kss);


                kss.dec_tab();
                kss << "}" << std::endl;
              }
              for(unsigned int i=0 ; i < n_unroll ; ++i){
                for(std::list<symbolic_binary_vector_expression_base*>::iterator it = vector_expressions_.begin(); it != vector_expressions_.end() ; ++it)
                  (*it)->clear_private_value(i);
                for(std::list<symbolic_binary_matrix_expression_base*>::iterator it = matrix_expressions_.begin(); it != matrix_expressions_.end() ; ++it)
                  (*it)->clear_private_value(i);
                for(std::list<symbolic_binary_scalar_expression_base*>::iterator it = scalar_expressions_.begin() ; it != scalar_expressions_.end(); ++it)
                  (*it)->clear_private_value(i);
              }
            }

          private:
            std::list<symbolic_binary_vector_expression_base* >  vector_expressions_;
            std::list<symbolic_binary_matrix_expression_base* >  matrix_expressions_;
            std::list<symbolic_binary_scalar_expression_base* >  scalar_expressions_;
            profile * profile_;
        };

      }

    }

  }

}

#endif
