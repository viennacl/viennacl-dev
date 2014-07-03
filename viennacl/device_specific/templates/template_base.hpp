#ifndef VIENNACL_DEVICE_SPECIFIC_TEMPLATES_TEMPLATE_BASE_
#define VIENNACL_DEVICE_SPECIFIC_TEMPLATES_TEMPLATE_BASE_

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


/** @file viennacl/generator/profile_base.hpp
 *
 * Base classes for the profiles
*/

#include <list>
#include <set>

#include "viennacl/ocl/kernel.hpp"
#include "viennacl/ocl/device.hpp"
#include "viennacl/ocl/device_utils.hpp"
#include "viennacl/ocl/infos.hpp"

#include "viennacl/scheduler/forwards.h"

#include "viennacl/device_specific/tree_parsing/traverse.hpp"
#include "viennacl/device_specific/tree_parsing/map.hpp"
#include "viennacl/device_specific/tree_parsing/prototype_generation.hpp"
#include "viennacl/device_specific/tree_parsing/set_arguments.hpp"
#include "viennacl/device_specific/tree_parsing/statement_representation.hpp"

namespace viennacl
{

  namespace device_specific
  {

    class template_base
    {
    public:
      class parameters{
      private:
        virtual bool invalid_impl(viennacl::ocl::device const & /*dev*/, size_t /*scalartype_size*/) const { return false; }
        virtual unsigned int lmem_used(unsigned int /*scalartype_size*/) const { return 0; }
      public:
        parameters(const char * scalartype, unsigned int simd_width, unsigned int local_size_1, unsigned int local_size_2, unsigned int num_kernels) :
          scalartype_(scalartype), simd_width_(simd_width), local_size_0_(local_size_1), local_size_1_(local_size_2), num_kernels_(num_kernels){ }

        unsigned int num_kernels() const  { return num_kernels_; }
        std::string const & scalartype() const { return scalartype_; }
        unsigned int local_size_0() const { return local_size_0_; }
        unsigned int local_size_1() const { return local_size_1_; }
        unsigned int simd_width() const { return simd_width_; }

        /** @brief returns whether or not the profile has undefined behavior on particular device */
        bool is_invalid() const
        {
          bool invalid = false;
          viennacl::ocl::device const & dev = viennacl::ocl::current_device();

          //Query device informations
          size_t lmem_available = static_cast<size_t>(dev.local_mem_size());
          unsigned int scalartype_size = utils::scalartype_size(scalartype_);
          invalid |= (lmem_used(scalartype_size)>lmem_available);

          //Invalid work group size
          size_t max_workgroup_size = dev.max_work_group_size();
          std::vector<size_t> max_work_item_sizes = dev.max_work_item_sizes();
          invalid |= local_size_0_*local_size_1_ > max_workgroup_size
              || local_size_0_ > max_work_item_sizes[0]
              || local_size_1_ > max_work_item_sizes[1]; // uses too much resources


          //Not warp multiple
          if(dev.type()==CL_DEVICE_TYPE_GPU){
            unsigned int warp_size = 32;
            if(dev.vendor_id()==4098)
              warp_size = 64;
            invalid |= (((local_size_0_*local_size_1_)%warp_size)>0);
          }

          //Invalid SIMD Width
          invalid |= (simd_width_!=1 && simd_width_!=2 &&
                      simd_width_!=4 && simd_width_!=8 &&
                      simd_width_!=16);

          return  invalid || invalid_impl(dev, scalartype_size);
        }
      protected:
        std::string scalartype_;

        unsigned int simd_width_;
        unsigned int local_size_0_;
        unsigned int local_size_1_;
        unsigned int num_kernels_;
      };

    private:

      /** @brief Generates the body of the associated kernel function */
      virtual void core(unsigned int kernel_id, utils::kernel_generation_stream& stream, statements_container const & statements, std::vector<mapping_type> const & mapping) const = 0;

      /** @brief generates the arguments that are global to the kernel (different from the object-specific arguments) */
      virtual void add_kernel_arguments(statements_container const & statements, std::string & arguments_string) const = 0;

      virtual void configure_impl(vcl_size_t kernel_id, viennacl::ocl::context & context, statements_container const & statements, viennacl::ocl::kernel & kernel, unsigned int & n_arg)  const = 0;

    public:
      /** @brief The constructor */
      template_base(template_base::parameters const & parameters, binding_policy_t binding_policy) : parameters_(parameters), binding_policy_(binding_policy){ }

      /** @brief Generates the code associated with this profile onto the provided stream */
      std::string generate(statements_container const & statements, std::string kernel_prefix = "")
      {
        statements_container::data_type::const_iterator sit;
        std::vector<mapping_type>::iterator mit;

        if(kernel_prefix.empty())
          kernel_prefix = tree_parsing::statements_representation(statements, binding_policy_);

        utils::kernel_generation_stream stream;

        //Create mapping
        std::vector<mapping_type> mapping(statements.data().size());
        tools::shared_ptr<symbolic_binder> binder = make_binder(binding_policy_);
        for(mit = mapping.begin(), sit = statements.data().begin() ; sit != statements.data().end() ; ++sit, ++mit)
          tree_parsing::traverse(*sit, sit->root(), tree_parsing::map_functor(*binder,*mit), true);

        //Generate Prototype
        std::string prototype;
        std::set<std::string> already_generated;
        add_kernel_arguments(statements, prototype);
        for(mit = mapping.begin(), sit = statements.data().begin() ; sit != statements.data().end() ; ++sit, ++mit)
          tree_parsing::traverse(*sit, sit->root(), tree_parsing::prototype_generation_traversal(parameters_.simd_width(), already_generated, prototype, *mit), true);
        prototype.erase(prototype.size()-1); //Last comma pruned

        for(unsigned int i = 0 ; i < parameters_.num_kernels() ; ++i)
        {
          stream << " __attribute__((reqd_work_group_size(" << parameters_.local_size_0() << "," << parameters_.local_size_1() << "," << 1 << ")))" << std::endl;
          stream << "__kernel " << "void " << kernel_prefix << i << "(" << prototype << ")" << std::endl;
          stream << "{" << std::endl;
          stream.inc_tab();
          core(i, stream, statements, mapping);
          stream.dec_tab();
          stream << "}" << std::endl;
        }

        return stream.str();
      }

      void enqueue(viennacl::ocl::program & program, statements_container const & statements, std::string kernel_prefix = "")
      {
        std::vector<viennacl::ocl::kernel*> ::iterator kit;
        vcl_size_t current_idx;

        if(kernel_prefix.empty())
          kernel_prefix = tree_parsing::statements_representation(statements, binding_policy_);

        //Get the kernels
        std::vector<viennacl::ocl::kernel*> kernels(parameters_.num_kernels());
        for(current_idx=0, kit = kernels.begin() ; kit != kernels.end() ; ++kit, ++current_idx)
           *kit = &program.get_kernel(kernel_prefix+tools::to_string(current_idx));

        //Configure
        for(current_idx=0, kit = kernels.begin() ; kit != kernels.end() ; ++kit, ++current_idx)
        {
          unsigned int current_arg = 0;
          tools::shared_ptr<symbolic_binder> binder = make_binder(binding_policy_);
          (*kit)->local_work_size(0,parameters_.local_size_0());
          (*kit)->local_work_size(1,parameters_.local_size_1());
          configure_impl(current_idx, const_cast<viennacl::ocl::context &>(*program.p_context()), statements, **kit, current_arg);
          for(statements_container::data_type::const_iterator itt = statements.data().begin() ; itt != statements.data().end() ; ++itt)
            tree_parsing::traverse(*itt, itt->root(), tree_parsing::set_arguments_functor(*binder,current_arg,**kit), true);
        }

        //Enqueue
        for(std::vector<viennacl::ocl::kernel*>::iterator it = kernels.begin() ; it != kernels.end() ; ++it)
          viennacl::ocl::enqueue(**it);
      }


    protected:
      template_base::parameters const & parameters_;
      binding_policy_t binding_policy_;
    };

  }

}

#endif
