#ifndef VIENNACL_GENERATOR_CODE_GENERATION_GENERATOR_BASE
#define VIENNACL_GENERATOR_CODE_GENERATION_GENERATOR_BASE

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
 * Base classes for the generators
*/

#include <list>
#include <set>

#include "viennacl/tools/shared_ptr.hpp"

#include "viennacl/ocl/device.hpp"
#include "viennacl/ocl/infos.hpp"

#include "viennacl/generator/forwards.h"
#include "viennacl/generator/symbolic_types.hpp"

namespace viennacl{

  namespace generator{

    namespace code_generation{

      /** @brief Base class for a generator
       *
       *  Fills a given kernel generation stream
       */
      class generator_base{
        private:
          virtual void generate_body_impl(unsigned int i, utils::kernel_generation_stream& kss) = 0;

          void init(){
            if(shared_infos_.empty())
              for(std::list< tools::shared_ptr<symbolic_binary_expression_tree_infos_base> >::iterator it = expressions_.begin() ; it != expressions_.end() ; ++it)
                (*it)->bind(shared_infos_, *prof_);
          }

          virtual void update_profile_state(){ }

        protected:
          generator_base(unsigned int n_kernels=1) : n_kernels_(n_kernels){          }

        public:

          void add_expression(symbolic_binary_expression_tree_infos_base * p){
            expressions_.push_back(tools::shared_ptr<symbolic_binary_expression_tree_infos_base>(p));
          }

          void set_profile(profile_base const * prof){
            prof_ = prof;
          }

          void enqueue(unsigned int & n, viennacl::ocl::program & pgm){
            for(unsigned int i=0 ; i<n_kernels_ ; ++i){
              prof_->set_state(i);
              viennacl::ocl::kernel& k = pgm.get_kernel("_k"+utils::to_string(n++));
              unsigned int tmp1 = 0;
              std::set<viennacl::backend::mem_handle> tmp2;
              for(std::list< tools::shared_ptr<symbolic_binary_expression_tree_infos_base> >::const_iterator iit = expressions_.begin() ; iit != expressions_.end() ; ++iit){
                (*iit)->enqueue(tmp1, k, tmp2,*prof_);
              }

              prof_->config_nd_range(k,&expressions_.front()->lhs());
              viennacl::ocl::enqueue(k);
            }
          }

          void generate(unsigned int & counter, utils::kernel_generation_stream& kss){
            init();
            for(unsigned int i = 0 ; i < n_kernels_; ++i){
              prof_->set_state(i);

              // Generates headers
              std::string name = "_k"+utils::to_string(counter++);
              kss <<  "__attribute__((reqd_work_group_size(" << prof_->local_work_size().first
                   << "," << prof_->local_work_size().second
                   << ",1)))" << std::endl;
              kss << "__kernel void " << name << '(' << std::endl;
              unsigned int tmp1 = 0;
              std::map<viennacl::backend::mem_handle, unsigned int> tmp2;
              std::ostringstream oss;
              for(std::list< tools::shared_ptr<symbolic_binary_expression_tree_infos_base> >::iterator it = expressions_.begin() ; it != expressions_.end() ; ++it){
                (*it)->generate_header(tmp1, oss, tmp2);
              }
              std::string args = oss.str();
              args.erase(args.size()-1,args.size()-1);
              kss << args << std::endl;
              kss << ')' << std::endl;

              //Generates body
              kss << '{' << std::endl;
              generate_body_impl(i,kss);
              kss << '}' << std::endl;

              kss << std::endl;
            }
          }

          std::list< tools::shared_ptr<symbolic_binary_expression_tree_infos_base> > const & expressions(){
            return expressions_;
          }

          unsigned int n_kernels(){
            return n_kernels_;
          }

          virtual ~generator_base(){ }
        private:
          std::vector< std::pair<symbolic_datastructure *, tools::shared_ptr<shared_symbolic_infos_t> > > shared_infos_;
        protected:
          profile_base const * prof_;
          std::list< tools::shared_ptr<symbolic_binary_expression_tree_infos_base> >  expressions_;
          unsigned int n_kernels_;
      };


    }

  }

}

#endif
