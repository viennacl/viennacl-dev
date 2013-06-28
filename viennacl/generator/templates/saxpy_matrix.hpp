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

#include "viennacl/generator/templates/generator_base.hpp"
#include "viennacl/generator/templates/profile_base.hpp"

#include "viennacl/generator/symbolic_types.hpp"

namespace viennacl{

  namespace generator{

    namespace code_generation{


        /** @brief profile template for the SAXPY kernel
        *
        *   No persistent threads (yet ?).
        */
        class saxpy_matrix_profile : public profile_base{
          public:

            /** @brief The user constructor */
            saxpy_matrix_profile(unsigned int vectorization, size_t group_size0) : profile_base(vectorization){
              group_size_ = group_size0;
            }

            /** @brief Return the group sizes used by this kernel */
            std::pair<size_t,size_t> local_work_size() const{
              return std::make_pair(group_size_,1);
            }

            /** @brief Configure the NDRange of a given kernel for this profile */
            void config_nd_range(viennacl::ocl::kernel & k, symbolic_expression_tree_base* p) const {
              symbolic_matrix_base * mat = dynamic_cast<symbolic_matrix_base*>(p);
              k.local_work_size(0,group_size_);
              k.global_work_size(0,viennacl::tools::roundUpToNextMultiple<cl_uint>(mat->real_size1()*mat->real_size2()/vectorization_,group_size_));
            }

            /** @brief returns whether or not the profile leads to undefined behavior on particular device
             *  @param dev the given device*/
            bool is_invalid(viennacl::ocl::device const & dev, size_t scalartype_size) const {
              return profile_base::invalid_base(dev,0);
            }

          private:
            unsigned int group_size_;
        };

        class saxpy_matrix_generator : public generator_base{
          private:
            void generate_body_impl(unsigned int i, utils::kernel_generation_stream& kss){
              symbolic_matrix_base * first_matrix = static_cast<symbolic_matrix_base*>(&(*expressions_.begin())->lhs());
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
              for(std::list<tools::shared_ptr<symbolic_binary_expression_tree_infos_base> >::iterator it=expressions_.begin() ; it!=expressions_.end();++it){
                (*it)->access_index(0,"r","c");
                (*it)->fetch(0,kss);
              }
              //Compute expressions
              for(std::list<tools::shared_ptr<symbolic_binary_expression_tree_infos_base> >::iterator it = expressions_.begin(); it!=expressions_.end(); ++it)
                kss << (*it)->generate(0) << ";" << std::endl;
              for(std::list<tools::shared_ptr<symbolic_binary_expression_tree_infos_base> >::iterator it=expressions_.begin() ; it!=expressions_.end();++it)
                (*it)->write_back(0,kss);
              kss << "}";
              kss.dec_tab();

              for(std::list<tools::shared_ptr<symbolic_binary_expression_tree_infos_base> >::iterator it = expressions_.begin(); it != expressions_.end() ; ++it)
                (*it)->clear_private_value(0);
            }

        };

      }


  }

}

#endif
