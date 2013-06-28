#ifndef VIENNACL_GENERATOR_CODE_GENERATION_TEMPLATES_SAXPY_VECTOR_HPP
#define VIENNACL_GENERATOR_CODE_GENERATION_TEMPLATES_SAXPY_VECTOR_HPP

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
        *   Possibility of loop unrolling.
        *   No persistent threads (yet ?).
        */
        class saxpy_vector_profile : public profile_base{
          public:

            /** @brief The user constructor */
            saxpy_vector_profile(unsigned int vectorization, unsigned int loop_unroll, size_t group_size0) : profile_base(vectorization){
              loop_unroll_ = loop_unroll;
              group_size_ = group_size0;
            }

            /** @brief Returns the unrolling factor */
            unsigned int loop_unroll() const{
              return loop_unroll_;
            }

            /** @brief Return the group sizes used by this kernel */
            std::pair<size_t,size_t> local_work_size() const{
              return std::make_pair(group_size_,1);
            }

            /** @brief Configure the NDRange of a given kernel for this profile */
            void config_nd_range(viennacl::ocl::kernel & k, symbolic_expression_tree_base* p) const {
              symbolic_vector_base * vec = dynamic_cast<symbolic_vector_base*>(p);
              k.local_work_size(0,group_size_);
              k.global_work_size(0,viennacl::tools::roundUpToNextMultiple<cl_uint>(vec->real_size()/(vectorization_*loop_unroll_),group_size_)); //Note: now using for-loop for good performance on CPU
            }

            /** @brief returns whether or not the profile leads to undefined behavior on particular device
             *  @param dev the given device*/
            bool is_invalid(viennacl::ocl::device const & dev, size_t scalartype_size) const {
              return profile_base::invalid_base(dev,0);
            }

          private:
            unsigned int loop_unroll_;
            unsigned int group_size_;
        };

        class saxpy_vector_generator : public generator_base{
          private:
            void generate_body_impl(unsigned int i, utils::kernel_generation_stream& kss){
              saxpy_vector_profile const * casted_prof = static_cast<saxpy_vector_profile const *>(prof_);

              symbolic_vector_base * first_vector = static_cast<symbolic_vector_base*>(&(*expressions_.begin())->lhs());
              unsigned int n_unroll = casted_prof->loop_unroll();
              kss << "int i = get_global_id(0)" ; if(n_unroll>1) kss << "*" << n_unroll; kss << ";" << std::endl;
              kss << "if(i<" << first_vector->size() << "){" << std::endl;
              kss.inc_tab();
              //Set access indices
              for(std::list<tools::shared_ptr<symbolic_binary_expression_tree_infos_base> >::iterator it=expressions_.begin() ; it!=expressions_.end();++it)
                for(unsigned int j=0 ; j < n_unroll ; ++j){
                  if(j==0) (*it)->access_index(j,"i","0");
                  else (*it)->access_index(j,"i + " + utils::to_string(j),"0");
                  (*it)->fetch(j,kss);
                }
              //Compute expressions
              for(std::list<tools::shared_ptr<symbolic_binary_expression_tree_infos_base> >::iterator it=expressions_.begin() ; it!=expressions_.end();++it)
                for(unsigned int j=0 ; j < n_unroll ; ++j)
                  kss << (*it)->generate(j) << ";" << std::endl;
              for(std::list<tools::shared_ptr<symbolic_binary_expression_tree_infos_base> >::iterator it=expressions_.begin() ; it!=expressions_.end();++it)
                for(unsigned int j=0 ; j < n_unroll ; ++j)
                  (*it)->write_back(j,kss);
              kss << "}" << std::endl;
              kss.dec_tab();

              for(unsigned int i=0 ; i < n_unroll ; ++i)
                for(std::list<tools::shared_ptr<symbolic_binary_expression_tree_infos_base> >::iterator it = expressions_.begin(); it != expressions_.end() ; ++it)
                  (*it)->clear_private_value(i);

            }


        };


    }

  }

}

#endif
