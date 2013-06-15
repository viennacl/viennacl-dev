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
      public :

        /** @brief Default Constructor */
        custom_operation(){ }

        /** @brief Add an operation to the operations list */
        template<class T>
        void add(T const & op){
          operations_manager_.add(op);
        }

//        /** @brief Forces the code generator to use a particular profile to generate the operations corresponding to T
//         *
//         * @tparam T profile type
//         */
//        template<class T>
//        void override_model(T const & o){
//          operations_manager_.override_model(o);
//        }

        std::string source_code() const{
          return operations_manager_.get_source_code();
        }

        viennacl::ocl::program & program(){
          std::string program_name = operations_manager_.representation();
          if(!viennacl::ocl::current_context().has_program(program_name)){
            std::string source_code = operations_manager_.get_source_code();
#ifdef VIENNACL_DEBUG_BUILD
            std::cout << "Building " << program_name << "..." << std::endl;
            std::cout << source_code << std::endl;
#endif
            viennacl::ocl::current_context().add_program(source_code, program_name);
          }
          return viennacl::ocl::current_context().get_program(program_name);
        }

        /** @brief Executes the given operation
         *  Compiles the program if not previously done
         */
        void execute(){
            viennacl::ocl::program & pgm = program();
            operations_manager_.enqueue(pgm);
//            std::cout << operations_manager_.get_source_code() << std::endl;
//          operations_manager_.bind_arguments(kernels_infos_);
//          viennacl::ocl::program & pgm = program();
//          for(std::map<std::string, generator::code_generation::kernel_wrapper>::iterator it = kernels_infos_.begin() ; it != kernels_infos_.end() ; ++it){
//            viennacl::ocl::kernel& k = pgm.get_kernel(it->first);
//            it->second.enqueue(k);
//            it->second.config_nd_range(k);
//            viennacl::ocl::enqueue(k);
//          }
        }

      private:
        code_generation::operations_handler operations_manager_;
    };
  }
}
#endif // CUSTOM_OPERATION_HPP
