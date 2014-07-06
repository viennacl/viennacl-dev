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

#include "viennacl/device_specific/forwards.h"
#include "viennacl/device_specific/tree_parsing/traverse.hpp"
#include "viennacl/device_specific/tree_parsing/map.hpp"
#include "viennacl/device_specific/tree_parsing/set_arguments.hpp"
#include "viennacl/device_specific/tree_parsing/statement_representation.hpp"

namespace viennacl
{

  namespace device_specific
  {

    class template_base
    {
    protected:

      /** @brief functor for generating the prototype of a statement */
      class prototype_generation_traversal : public tree_parsing::traversal_functor
      {
        private:
          std::set<std::string> & already_generated_;
          std::string & str_;
          mapping_type const & mapping_;
        public:
          prototype_generation_traversal(std::set<std::string> & already_generated, std::string & str, mapping_type const & mapping) : already_generated_(already_generated), str_(str),  mapping_(mapping){ }

          void operator()(scheduler::statement const & statement, vcl_size_t root_idx, tree_parsing::leaf_t leaf) const
          {
              scheduler::statement_node const & root_node = statement.array()[root_idx];
              if( (leaf==tree_parsing::LHS_NODE_TYPE && root_node.lhs.type_family!=scheduler::COMPOSITE_OPERATION_FAMILY)
                ||(leaf==tree_parsing::RHS_NODE_TYPE && root_node.rhs.type_family!=scheduler::COMPOSITE_OPERATION_FAMILY) )
              {
                mapped_object * obj = mapping_.at(std::make_pair(root_idx,leaf)).get();
                obj->append_kernel_arguments(already_generated_, str_);
              }
          }
      };

      /** @brief functor for generating the prototype of a statement */
      template<class T>
      class set_simd_width_traversal : public tree_parsing::traversal_functor
      {
        private:
          unsigned int simd_width_;
          mapping_type const & mapping_;
        public:
          set_simd_width_traversal(unsigned int simd_width, mapping_type const & mapping) : simd_width_(simd_width), mapping_(mapping) { }

          void operator()(scheduler::statement const & /*statement*/, vcl_size_t root_idx, tree_parsing::leaf_t leaf) const
          {
            mapping_type::const_iterator it = mapping_.find(std::make_pair(root_idx, leaf));
            if(it!=mapping_.end())
              if(T * p = dynamic_cast<T*>(mapping_.at(std::make_pair(root_idx, leaf)).get()))
                p->set_simd_width(simd_width_);
          }
      };

    public:
      class parameters
      {
      private:
        virtual bool invalid_impl(viennacl::ocl::device const & /*dev*/, size_t /*scalartype_size*/) const { return false; }
        virtual unsigned int lmem_used(unsigned int /*scalartype_size*/) const { return 0; }
      public:
        parameters(const char * _scalartype, unsigned int _simd_width, unsigned int _local_size_1, unsigned int _local_size_2, unsigned int _num_kernels) :
          scalartype(_scalartype), simd_width(_simd_width), local_size_0(_local_size_1), local_size_1(_local_size_2), num_kernels(_num_kernels){ }

        const std::string scalartype;

        const unsigned int simd_width;
        const unsigned int local_size_0;
        const unsigned int local_size_1;
        const unsigned int num_kernels;

        /** @brief returns whether or not the profile has undefined behavior on particular device */
        bool is_invalid() const
        {
          bool invalid = false;
          viennacl::ocl::device const & dev = viennacl::ocl::current_device();

          //Query device informations
          size_t lmem_available = static_cast<size_t>(dev.local_mem_size());
          unsigned int scalartype_size = utils::scalartype_size(scalartype);
          invalid |= (lmem_used(scalartype_size)>lmem_available);

          //Invalid work group size
          size_t max_workgroup_size = dev.max_work_group_size();
          std::vector<size_t> max_work_item_sizes = dev.max_work_item_sizes();
          invalid |= local_size_0*local_size_1 > max_workgroup_size
              || local_size_0 > max_work_item_sizes[0]
              || local_size_1 > max_work_item_sizes[1]; // uses too much resources


          //Not warp multiple
          if(dev.type()==CL_DEVICE_TYPE_GPU){
            unsigned int warp_size = 32;
            if(dev.vendor_id()==4098)
              warp_size = 64;
            invalid |= (((local_size_0*local_size_1)%warp_size)>0);
          }

          //Invalid SIMD Width
          invalid |= (simd_width!=1 && simd_width!=2 &&
                      simd_width!=4 && simd_width!=8 &&
                      simd_width!=16);

          return  invalid || invalid_impl(dev, scalartype_size);
        }
      };

    private:

      /** @brief Generates the body of the associated kernel function */
      virtual void core(unsigned int kernel_id, utils::kernel_generation_stream& stream, statements_container const & statements, std::vector<mapping_type> const & mapping) const = 0;
      /** @brief generates the arguments that are global to the kernel (different from the object-specific arguments) */
      virtual void add_kernel_arguments(statements_container const & statements, std::string & arguments_string) const = 0;
      /** @brief configure the local sizes and enqueue the arguments of the kernel */
      virtual void configure_impl(vcl_size_t kernel_id, viennacl::ocl::context & context, statements_container const & statements, viennacl::ocl::kernel & kernel, unsigned int & n_arg)  const = 0;
      /** @brief Returns the effective simd width for a given mapped_object */
      virtual void set_simd_widths(scheduler::statement const & s, mapping_type const & m)
      {
        tree_parsing::traverse(s, s.root(), set_simd_width_traversal<mapped_buffer>(parameters_.simd_width, m), true);
      }

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
        for(sit = statements.data().begin(), mit = mapping.begin() ; mit != mapping.end() ; ++sit, ++mit)
          set_simd_widths(*sit, *mit);
        add_kernel_arguments(statements, prototype);
        for(mit = mapping.begin(), sit = statements.data().begin() ; sit != statements.data().end() ; ++sit, ++mit)
          tree_parsing::traverse(*sit, sit->root(), prototype_generation_traversal(already_generated, prototype, *mit), true);
        prototype.erase(prototype.size()-1); //Last comma pruned

        for(unsigned int i = 0 ; i < parameters_.num_kernels ; ++i)
        {
          stream << " __attribute__((reqd_work_group_size(" << parameters_.local_size_0 << "," << parameters_.local_size_1 << "," << 1 << ")))" << std::endl;
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
        std::vector<viennacl::ocl::kernel*> kernels(parameters_.num_kernels);
        for(current_idx=0, kit = kernels.begin() ; kit != kernels.end() ; ++kit, ++current_idx)
           *kit = &program.get_kernel(kernel_prefix+tools::to_string(current_idx));

        //Configure
        for(current_idx=0, kit = kernels.begin() ; kit != kernels.end() ; ++kit, ++current_idx)
        {
          unsigned int current_arg = 0;
          tools::shared_ptr<symbolic_binder> binder = make_binder(binding_policy_);
          (*kit)->local_work_size(0,parameters_.local_size_0);
          (*kit)->local_work_size(1,parameters_.local_size_1);
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
