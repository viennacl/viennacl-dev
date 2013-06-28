#ifndef VIENNACL_GENERATOR_CODE_GENERATION_TEMPLATES_SCALAR_REDUCTION_HPP
#define VIENNACL_GENERATOR_CODE_GENERATION_TEMPLATES_SCALAR_REDUCTION_HPP

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

#include "viennacl/generator/symbolic_types.hpp"


#include "viennacl/generator/templates/generator_base.hpp"
#include "viennacl/generator/templates/profile_base.hpp"

#include "viennacl/generator/utils.hpp"

namespace viennacl{

  namespace generator{

    namespace code_generation{


      /** @brief profile template for the scalar reduction kernel
      *
      *   Use of persistent threads determined by GROUP_SIZE and NUM_GROUPS
      */

        class scalar_reduction_profile : public profile_base{
          public:
            /** @brief The user constructor */
            scalar_reduction_profile(unsigned int vectorization, unsigned int group_size, unsigned int num_groups) : profile_base(vectorization){
              group_size_ = group_size;
              num_groups_ = num_groups;

              current_group_size_ = group_size_;
              current_num_groups_ = num_groups_;
            }

            /** @brief Return the group sizes used by this kernel */
            std::pair<size_t,size_t> local_work_size() const{  return std::make_pair(group_size_,1); }

            unsigned int group_size() const {
              return current_group_size_;
            }

            unsigned int num_groups() const {
              return num_groups_;
            }

            void set_state(unsigned int i) const {
              if(i==0){
                current_group_size_ = group_size_;
                current_num_groups_ = num_groups_;
              }
              else{
                current_group_size_ = num_groups_;
                current_num_groups_ = 1;
              }
            }

            /** @brief Configure the NDRange of a given kernel for this profile */
            void config_nd_range(viennacl::ocl::kernel & k, symbolic_expression_tree_base* p) const {
              k.local_work_size(0,current_group_size_);
              k.global_work_size(0,current_group_size_*current_num_groups_);
            }

            /** @brief returns whether or not the profile leads to undefined behavior on particular device
             *  @param dev the given device*/
            bool is_invalid(viennacl::ocl::device const & dev, size_t scalartype_size) const {
              return profile_base::invalid_base(dev,current_group_size_*scalartype_size);
            }

          private:
            unsigned int group_size_;
            unsigned int num_groups_;

            mutable unsigned int current_group_size_;
            mutable unsigned int current_num_groups_;
        };

        class scalar_reduction_generator: public generator_base{
          private:
            void compute_reductions_samesize(utils::kernel_generation_stream& kss, std::map<binary_operator const *, symbolic_local_memory<1> > const & lmems){
              unsigned int size = lmems.begin()->second.size();
              for(unsigned int stride = size/2 ; stride>0 ; stride /=2){
                kss << "barrier(CLK_LOCAL_MEM_FENCE); ";
                for(std::map<binary_operator const *, symbolic_local_memory<1> >::const_iterator it = lmems.begin(); it != lmems.end() ; ++it){
                  kss <<  it->second.access("lid") <<  " = " << it->first->generate(it->second.access("lid"), "((lid < " + utils::to_string(stride) + ")?" + it->second.access("lid+" + utils::to_string(stride)) + " : 0)" ) << ";" << std::endl;
                }
              }
            }

            void generate_body_impl(unsigned int kernel_id, utils::kernel_generation_stream& kss){
              scalar_reduction_profile const * casted_prof = static_cast<scalar_reduction_profile const *>(prof_);
              std::list<symbolic_scalar_reduction_base*>  inner_prods;
              std::list<symbolic_vector_base *>  vectors;
              std::list<symbolic_pointer_scalar_base *> gpu_scalars;
              for(std::list<tools::shared_ptr<symbolic_binary_expression_tree_infos_base> >::const_iterator it=expressions_.begin() ; it!=expressions_.end() ; ++it){
                extract_as(*it,vectors,utils::is_type<symbolic_vector_base>());
                extract_as(*it,gpu_scalars,utils::is_type<symbolic_pointer_scalar_base>());
                extract_as(*it,inner_prods,utils::is_type<symbolic_scalar_reduction_base>());
              }

              kss << "unsigned int lid = get_local_id(0);" << std::endl;
              unsigned int alignment = casted_prof->vectorization();
              if(kernel_id==0){
                for(std::list<symbolic_scalar_reduction_base*>::iterator it = inner_prods.begin() ; it!=inner_prods.end() ; ++it){
                  std::string sum_name = (*it)->sum_name();
                  kss << (*it)->scalartype() << " " << sum_name << " = 0;" << std::endl;
                }
                std::string size = (*vectors.begin())->size();
                kss << "unsigned int chunk_size = (" << size << "+get_num_groups(0)-1)/get_num_groups(0);" << std::endl;
                kss << "unsigned int chunk_start = get_group_id(0)*chunk_size;" << std::endl;
                kss << "unsigned int chunk_end = min(chunk_start+chunk_size, " << size << ");" << std::endl;
                kss << "for(unsigned int i = chunk_start + get_local_id(0) ; i < chunk_end ; i += get_local_size(0)){" << std::endl;
                kss.inc_tab();

                //Set access index
                for(std::list<symbolic_scalar_reduction_base*>::iterator it = inner_prods.begin() ; it!=inner_prods.end() ; ++it){
                  (*it)->access_index(0,"i","0");
                  (*it)->fetch(0,kss);
                }
                for(std::list<symbolic_scalar_reduction_base*>::iterator it=inner_prods.begin() ; it!=inner_prods.end();++it){
                  std::string sum_name = (*it)->sum_name();
                  for(unsigned int a=0; a<alignment;++a){
                    kss << sum_name << " = " << (*it)->op_reduce().generate(sum_name, (*it)->symbolic_binary_expression_tree_infos_base::generate(0,a)) << ";" << std::endl;
                  }
                }
                kss.dec_tab();
                kss << "}" << std::endl;
                std::map<binary_operator const *, symbolic_local_memory<1> > local_mems;
                for( std::list<symbolic_scalar_reduction_base *>::const_iterator it = inner_prods.begin(); it != inner_prods.end() ; ++it){
                  std::string sum_name = (*it)->sum_name();
                  symbolic_local_memory<1> lmem = (*it)->make_local_memory(casted_prof->group_size());
                  local_mems.insert(std::make_pair(&(*it)->op_reduce(),lmem));
                  kss << lmem.declare() << ";" << std::endl;
                  kss << lmem.access("lid") << " = " << sum_name << ";" << std::endl;
                }
                compute_reductions_samesize(kss,local_mems);
                for(std::list<symbolic_scalar_reduction_base *>::iterator it=inner_prods.begin() ; it!=inner_prods.end();++it){
                  kss << "if(lid==0) " << (*it)->name() << "[get_group_id(0)]" << "=" << (*it)->name()+"_local" << "[0]" << ";" << std::endl;
                }
              }
              else{
                std::map<binary_operator const *, symbolic_local_memory<1> > local_mems;
                for( std::list<symbolic_scalar_reduction_base *>::const_iterator it = inner_prods.begin(); it != inner_prods.end() ; ++it){
                  symbolic_local_memory<1> lmem = (*it)->make_local_memory(casted_prof->group_size());
                  local_mems.insert(std::make_pair(&(*it)->op_reduce(),lmem));
                  kss << lmem.declare() << ";" << std::endl;
                  kss << lmem.access("lid") << " = " << (*it)->access("lid") << ";" << std::endl;
                  (*it)->access_name(lmem.access("0"));
                }
                compute_reductions_samesize(kss,local_mems);
                for(std::list<tools::shared_ptr<symbolic_binary_expression_tree_infos_base> >::iterator it = expressions_.begin() ; it!=expressions_.end() ; ++it){
                  kss << "barrier(CLK_LOCAL_MEM_FENCE); if(lid==0) " << (*it)->generate(0) << ";" << std::endl;
                }
              }
            }
        };

      }


  }

}

#endif
