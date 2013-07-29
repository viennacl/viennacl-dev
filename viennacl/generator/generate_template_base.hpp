#ifndef VIENNACL_GENERATOR_GENERATE_TEMPLATE_BASE_BASE
#define VIENNACL_GENERATOR_GENERATE_TEMPLATE_BASE_BASE

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


/** @file viennacl/generator/templates/tgenerate_template_base.hpp
 *
 * Base classes for the templates
*/

#include <list>
#include <set>

#include "viennacl/ocl/device.hpp"
#include "viennacl/ocl/infos.hpp"

#include "viennacl/scheduler/forwards.h"

#include "viennacl/generator/generate_utils.hpp"

namespace viennacl{

  namespace generator{


    /** @brief Base class for an optimization profile */
    class template_base{
      public:
        typedef std::vector<scheduler::statement> statements_type;

        class profile{
          protected:
            virtual bool invalid_impl(viennacl::ocl::device const & dev, size_t scalartype_size) const{ return false; }

            virtual std::size_t lmem_used(std::size_t scalartype_size) const { return 0; }

            void configure_local_sizes(viennacl::ocl::kernel & k, std::size_t kernel_id) const {
              std::size_t lsize1, lsize2;
              set_local_sizes(lsize1,lsize2, kernel_id);
              k.local_work_size(0,lsize1);
              k.local_work_size(1,lsize2);
            }


          public:
            profile(unsigned int vectorization, std::size_t num_kernels) : vectorization_(vectorization), num_kernels_(num_kernels){ }

            virtual void configure_range_enqueue_arguments(std::size_t kernel_id, statements_type  const & statements, viennacl::ocl::kernel & k, unsigned int & n_arg) const = 0;

            virtual void kernel_arguments(statements_type  const & statements, std::string & arguments_string) const = 0;
            virtual void set_local_sizes(std::size_t & x, std::size_t & y, std::size_t kernel_id) const = 0;

            unsigned int vectorization() const { return vectorization_; }

            /** @brief returns whether or not the profile leads to undefined behavior on particular device
         *  @param dev the given device*/
            bool is_invalid(viennacl::ocl::device const & dev, size_t scalartype_size) const{
              //Query profile informations
              std::size_t size1, size2;
              set_local_sizes(size1, size2, 0);

              //Query device informations
              size_t lmem_available = dev.local_mem_size();
              size_t max_workgroup_size = dev.max_work_group_size();
              std::vector<size_t> max_work_item_sizes = dev.max_work_item_sizes();

              bool invalid_work_group_sizes = size1*size2 > max_workgroup_size; // uses too much resources
              invalid_work_group_sizes = invalid_work_group_sizes || size1 > max_work_item_sizes[0];
              if(max_work_item_sizes.size()>1) invalid_work_group_sizes = invalid_work_group_sizes || size2 > max_work_item_sizes[1];

              return  invalid_work_group_sizes || lmem_used(scalartype_size)>lmem_available || invalid_impl(dev, scalartype_size);
            }

            std::size_t num_kernels() const{
              return num_kernels_;
            }

          protected:
            unsigned int vectorization_;
            std::size_t num_kernels_;
        };

      protected:
        virtual void core(std::size_t kernel_id, utils::kernel_generation_stream& stream) const = 0;
        template_base(statements_type const & s, profile const & p) : statements_(s), mapping_(s.size()), profile_(p) { }

        std::string init_get_prototype() const {
          std::string prototype;
          std::set<std::string> already_generated;
          profile_.kernel_arguments(statements_, prototype);
          detail::map_all_statements(statements_.begin(), statements_.end(), mapping_);
          for(std::size_t i = 0 ; i < mapping_.size() ; ++i){
            for(detail::mapping_type::iterator it = mapping_[i].begin() ; it != mapping_[i].end() ; ++it){
              it->second->append_kernel_arguments(already_generated, prototype, profile_.vectorization());
            }
          }
          prototype.erase(prototype.size()-1); //Last comma pruned
          return prototype;
        }

      public:

        virtual void operator()(utils::kernel_generation_stream & stream) const {
          std::string prototype = init_get_prototype();
          for(std::size_t n = 0 ; n < profile_.num_kernels() ; ++n){
            stream << "__kernel void " << "kernel_" << n << "(" << std::endl;
            stream << prototype << std::endl;
            stream << ")" << std::endl;

            //core:
            stream << "{" << std::endl;
            stream.inc_tab();
            core(n, stream);
            stream.dec_tab();
            stream << "}" << std::endl;
          }
        }

      protected:
        statements_type const & statements_;
        mutable std::vector<detail::mapping_type> mapping_;
        profile const & profile_;
    };


  }

}

#endif
