#ifndef VIENNACL_GENERATOR_CODE_GENERATION_OPTIMIZATION_PROFILE
#define VIENNACL_GENERATOR_CODE_GENERATION_OPTIMIZATION_PROFILE

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


/** @file viennacl/generator/templates/base_classes.hpp
 *
 * Base classes for the kernel templates
*/

#include <list>
#include <set>

#include "viennacl/ocl/device.hpp"
#include "viennacl/ocl/infos.hpp"
#include "viennacl/generator/utils.hpp"

namespace viennacl{

  namespace generator{

    namespace code_generation{

      /** @brief Base class for an optimization profile */
      class optimization_profile{
        protected:
          typedef unsigned int size_type;
        protected:
          bool is_invalid(viennacl::ocl::device const & dev, size_t lmem_used){
            //Query profile informations
            std::pair<size_t, size_t> workgroup_size = local_work_size();

            //Query device informations
            size_t lmem_available = viennacl::ocl::info<CL_DEVICE_LOCAL_MEM_SIZE>(dev.id());
            size_t max_workgroup_size = viennacl::ocl::info<CL_DEVICE_MAX_WORK_GROUP_SIZE>(dev.id());
            std::vector<size_t> max_work_item_sizes = viennacl::ocl::info<CL_DEVICE_MAX_WORK_ITEM_SIZES>(dev.id());

            bool invalid_work_group_sizes = workgroup_size.first*workgroup_size.second > max_workgroup_size; // uses too much resources
            invalid_work_group_sizes = invalid_work_group_sizes || workgroup_size.first > max_work_item_sizes[0];
            if(max_work_item_sizes.size()>1) invalid_work_group_sizes = invalid_work_group_sizes || workgroup_size.second > max_work_item_sizes[1];

            return  invalid_work_group_sizes
                || lmem_used>lmem_available;
          }
        public:
          optimization_profile() : vectorization_(1){ }
          optimization_profile(unsigned int vectorization) : vectorization_(vectorization){ }
          virtual std::string repr() const = 0;
          virtual void config_nd_range(viennacl::ocl::kernel & k, symbolic_expression_tree_base * p) = 0;
          unsigned int vectorization() const{ return vectorization_; }
          virtual std::pair<size_t,size_t> local_work_size() const = 0;
          virtual ~optimization_profile(){ }
        protected:
          unsigned int vectorization_;
      };

      /** @brief Base class for a generator
       *
       *  Fills a given kernel generation stream
       */
      class generator{
        public:
          virtual void operator()(utils::kernel_generation_stream& kss) = 0;
          virtual ~generator(){ }
      };


      /** @brief Prints a given profile */
      inline std::ostream& operator<<(std::ostream& os, optimization_profile const & prof){
        os << prof.repr();
        return os;
      }

    }

  }

}

#endif
