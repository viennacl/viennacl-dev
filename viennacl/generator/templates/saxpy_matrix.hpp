#ifndef VIENNACL_GENERATOR_CODE_GENERATION_TEMPLATES_SAXPY_MATRIX_HPP
#define VIENNACL_GENERATOR_CODE_GENERATION_TEMPLATES_SAXPY_MATRIX_HPP

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

      namespace saxpy_matrix{

        /** @brief profile template for the SAXPY kernel
        *
        *   No persistent threads (yet ?).
        */
        class profile : public optimization_profile{
          public:

            /** @brief The default constructor : Unroll factor : 1, Group size : 128. */
            profile(){
              group_size_ = 128;
            }

            /** @brief The user constructor */
            profile(unsigned int vectorization, size_t group_size0) : optimization_profile(vectorization){
              group_size_ = group_size0;
            }

            /** @brief Return the group sizes used by this kernel */
            std::pair<size_t,size_t> local_work_size() const{
              return std::make_pair(group_size_,1);
            }

            /** @brief Configure the NDRange of a given kernel for this profile */
            void config_nd_range(viennacl::ocl::kernel & k, symbolic_expression_tree_base* p){
              symbolic_matrix_base * mat = dynamic_cast<symbolic_matrix_base*>(p);
              k.local_work_size(0,group_size_);
              k.global_work_size(0,viennacl::tools::roundUpToNextMultiple<cl_uint>(mat->real_size1()*mat->real_size2()/vectorization_,group_size_));
            }

            /** @brief Returns the representation string of this profile */
            std::string repr() const{
              std::ostringstream oss;
              oss << "V" << vectorization_  << "GROUP" << group_size_;
              return oss.str();
            }

            /** @brief returns whether or not the profile leads to undefined behavior on particular device
             *  @param dev the given device*/
            bool is_invalid(viennacl::ocl::device const & dev, size_t scalartype_size){
              return optimization_profile::is_invalid(dev,0);
            }

          private:
            unsigned int group_size_;
        };

        class generator : public code_generation::generator{
          public:
            generator(std::list<symbolic_binary_matrix_expression_base * > const & matrix_expressions
                      ,profile * kernel_config): matrix_expressions_(matrix_expressions), profile_(kernel_config)
            {
            }


            void operator()(utils::kernel_generation_stream& kss){
              symbolic_matrix_base * first_matrix = static_cast<symbolic_matrix_base*>(&(*matrix_expressions_.begin())->lhs());
              if(first_matrix->is_rowmajor()){
                kss << "unsigned int r = get_global_id(0)/" << first_matrix->internal_size2() << ";" << std::endl;
                kss << "unsigned int c = get_global_id(0)%" << first_matrix->internal_size2() << ";" << std::endl;
              }
              else{
                kss << "unsigned int r = get_global_id(0)%" << first_matrix->internal_size1() << ";" << std::endl;
                kss << "unsigned int c = get_global_id(0)/" << first_matrix->internal_size1() << ";" << std::endl;
              }
              kss << "if(r < " << first_matrix->internal_size1() << " && c < " << first_matrix->internal_size2() << "){" << std::endl;
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
              kss << "}";
              kss.dec_tab();

              for(std::list<symbolic_binary_matrix_expression_base*>::iterator it = matrix_expressions_.begin(); it != matrix_expressions_.end() ; ++it)
                (*it)->clear_private_value(0);
            }

          private:
            std::list<symbolic_binary_matrix_expression_base* >  matrix_expressions_;
            profile * profile_;
        };

      }

    }

  }

}

#endif
