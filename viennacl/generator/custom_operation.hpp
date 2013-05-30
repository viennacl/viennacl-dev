#ifndef VIENNACL_GENERATOR_CUSTOM_OPERATION_HPP
#define VIENNACL_GENERATOR_CUSTOM_OPERATION_HPP

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


/** @file viennacl/generator/custom_operation.hpp
 *
 *  The main interfacing class. Can add operations, compiles kernels if not previously done, infers arguments and executes.
*/

#include "viennacl/generator/code_generation.hpp"
#include "viennacl/generator/symbolic_types.hpp"
#include "viennacl/generator/overloads.hpp"
#include "viennacl/tools/shared_ptr.hpp"
#include <bitset>

namespace viennacl
{
  namespace generator
  {


    /** @brief Interface class for using the kernel generator */
    class custom_operation
    {

      private:
        void compile_program(std::string const & pgm_name) const{
#ifdef VIENNACL_DEBUG_BUILD
          std::cout << "Building " << pgm_name << "..." << std::endl;
          std::cout << source_code_ << std::endl;
#endif
          assert(!source_code_.empty() && " Custom Operation not initialized ");
          viennacl::ocl::program& program = viennacl::ocl::current_context().add_program(source_code_, pgm_name);
          for(std::map<std::string, generator::code_generation::kernel_wrapper>::const_iterator it = kernels_infos_.begin() ; it !=kernels_infos_.end() ; ++it){
            program.add_kernel(it->first);
          }
        }

        void init() {
          if(source_code_.empty()) source_code_ = operations_manager_.get_source_code(kernels_infos_);
        }


      public :

        /** @brief Default Constructor */
        custom_operation(){ }

        /** @brief Creates a custom operation from one operation */
        template<class T0>
        custom_operation(T0 const & op0){
          add(op0);
        }

        /** @brief Add an operation to the operations list */
        template<class T>
        void add(T const & op){
          operations_manager_.add(op);
        }

        /** @brief Forces the code generator to use a particular profile to generate the operations corresponding to T
         *
         * @tparam T profile type
         */
        template<class T>
        void override_model(T const & o){
          operations_manager_.override_model(o);
        }


        /** @brief Returns the list of the operations handled at the time */
        std::list<code_generation::kernel_wrapper> kernels_list(){
          return operations_manager_.get_kernels_list();
        }

        /** @brief Returns the corresponding program. Compiles the operation if not previously done */
        viennacl::ocl::program & program(){
          init();
          std::string program_name_ = operations_manager_.repr();
          if(!viennacl::ocl::current_context().has_program(program_name_)){
            compile_program(program_name_);
          }
          return viennacl::ocl::current_context().get_program(program_name_);
        }

        /** @brief Executes the given operation
         *  Compiles the program if not previously done
         */
        void execute(){
          init();
          viennacl::ocl::program & pgm = program();
          for(std::map<std::string, generator::code_generation::kernel_wrapper>::iterator it = kernels_infos_.begin() ; it != kernels_infos_.end() ; ++it){
            viennacl::ocl::kernel& k = pgm.get_kernel(it->first);
            it->second.enqueue(k);
            it->second.config_nd_range(k);
            viennacl::ocl::enqueue(k);
          }
        }


        /** @brief Returns the source code of the given operation
         *  Empty if not compiled yet
         */
        std::string source_code() const{
          return source_code_;
        }

      private:
        code_generation::operations_handler operations_manager_;
        std::map<std::string, generator::code_generation::kernel_wrapper> kernels_infos_;
        std::string source_code_;
        std::string program_name_;

    };
  }
}
#endif // CUSTOM_OPERATION_HPP
