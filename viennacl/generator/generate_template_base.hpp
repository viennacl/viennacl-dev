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

#include "viennacl/ocl/kernel.hpp"
#include "viennacl/ocl/device.hpp"
#include "viennacl/ocl/device_utils.hpp"
#include "viennacl/ocl/infos.hpp"

#include "viennacl/scheduler/forwards.h"

#include "viennacl/generator/generate_utils.hpp"
#include "viennacl/generator/map_generate_prototype.hpp"

namespace viennacl{

  namespace generator{


    /** @brief Base class for an optimization profile */
    class profile_base{
      public:
        typedef std::list< std::pair<scheduler::statement, scheduler::statement_node> > statements_type;

      protected:
        friend std::ostream & operator<<(std::ostream &, profile_base const &);

        virtual bool invalid_impl(viennacl::ocl::device const & /*dev*/, size_t /*scalartype_size*/) const { return false; }
        virtual bool is_slow_impl(viennacl::ocl::device const &) const { return false; }

        virtual std::size_t lmem_used(std::size_t /*scalartype_size*/) const { return 0; }

        void configure_local_sizes(viennacl::ocl::kernel & k, std::size_t /*kernel_id*/) const {
          k.local_work_size(0,local_size_1_);
          k.local_work_size(1,local_size_2_);
        }

        virtual void print(std::ostream & s) const{
          s << csv_representation();
        }

        virtual void core(std::size_t kernel_id, utils::kernel_generation_stream& stream, statements_type const & statements, std::vector<detail::mapping_type> const & mapping) const = 0;

      public:
        profile_base(unsigned int vectorization, std::size_t local_size_1, std::size_t local_size_2, std::size_t num_kernels) : vectorization_(vectorization), local_size_1_(local_size_1), local_size_2_(local_size_2), num_kernels_(num_kernels){ }

        virtual ~profile_base(){ }

        virtual void configure_range_enqueue_arguments(std::size_t kernel_id, statements_type  const & statements, viennacl::ocl::kernel & k, unsigned int & n_arg) const = 0;

        virtual void kernel_arguments(statements_type  const & statements, std::string & arguments_string) const = 0;

        unsigned int vectorization() const { return vectorization_; }

        virtual std::string csv_representation() const = 0;

        /** @brief returns whether or not the profile is likely to be slow on a particular device
         *  @param dev the given device*/
        bool is_slow(viennacl::ocl::device const & dev) const{
          bool res = false;
          if(dev.type()==CL_DEVICE_TYPE_GPU){
            std::size_t warp_size = 32;
            if(dev.vendor_id()==4098)
              warp_size = 64;
            res = static_cast<bool>(((local_size_1_*local_size_2_)%warp_size)>0);
          }
          return res || is_slow_impl(dev);
        }

        /** @brief returns whether or not the profile leads to undefined behavior on particular device
         *  @param dev the given device*/
        bool is_invalid(viennacl::ocl::device const & dev, size_t scalartype_size) const{
          //Query device informations
          size_t lmem_available = static_cast<size_t>(dev.local_mem_size());
          size_t max_workgroup_size = dev.max_work_group_size();

          std::vector<size_t> max_work_item_sizes = dev.max_work_item_sizes();
          bool invalid_work_group_sizes = local_size_1_*local_size_2_ > max_workgroup_size
              || local_size_1_ > max_work_item_sizes[0]
              || local_size_2_ > max_work_item_sizes[1]; // uses too much resources

          return  invalid_work_group_sizes
              || lmem_used(scalartype_size)>lmem_available
              || invalid_impl(dev, scalartype_size);
        }

        std::size_t num_kernels() const{ return num_kernels_; }

        virtual void operator()(utils::kernel_generation_stream & stream, std::size_t device_offset, statements_type const & statements) const {
          std::vector<detail::mapping_type> mapping(statements.size());

          ///Get Prototype, initialize mapping
          std::string prototype;
          std::set<std::string> already_generated;
          kernel_arguments(statements, prototype);

          {
            std::map<void *, std::size_t> memory;
            unsigned int current_arg = 0;
            std::size_t i = 0;
            for(statements_type::const_iterator it = statements.begin() ; it != statements.end() ; ++it)
              detail::traverse(it->first, it->second, detail::map_functor(memory,current_arg,mapping[i++]));
          }

          for(statements_type::const_iterator it = statements.begin() ; it != statements.end() ; ++it){
            detail::traverse(it->first, it->second, detail::prototype_generation_traversal(already_generated, prototype, vectorization(), mapping[std::distance(statements.begin(), it)]));
          }

          prototype.erase(prototype.size()-1); //Last comma pruned

          //Generate
          for(std::size_t n = 0 ; n < num_kernels() ; ++n){
            //stream << "__attribute__((vec_type_hint()))" << std::endl;
            stream << " __attribute__((reqd_work_group_size(" << local_size_1_ << "," << local_size_2_ << "," << 1 << ")))" << std::endl;
            stream << "__kernel " << "void " << "kernel_" << device_offset << "_" << n << "(" << std::endl;
            stream << prototype << std::endl;
            stream << ")" << std::endl;

            //core:
            stream << "{" << std::endl;
            stream.inc_tab();
            core(n, stream, statements, mapping);
            stream.dec_tab();
            stream << "}" << std::endl;
          }
        }

      protected:
        unsigned int vectorization_;
        std::size_t local_size_1_;
        std::size_t local_size_2_;
        std::size_t num_kernels_;
    };


    inline std::ostream & operator<<(std::ostream & os, profile_base const & profile){
      profile.print(os);
      return os;
    }

  }

}

#endif
