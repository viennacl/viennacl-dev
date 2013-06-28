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
#include "builtin_database.hpp"
#include <bitset>

namespace viennacl
{
  namespace generator
  {


    /** @brief Interface class for using the kernel generator */
    class custom_operation
    {
      public :
        /** @brief Add an operation to the operations list */
        template<class T>
        void add(T const & op){
          operations_manager_.add(op);
        }

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

        /** @brief Force a profile for a given operation */
        template<class PROF>
        void force_profile(code_generation::profile_id const & id, PROF const & prof){
          operations_manager_.force_profile(id,prof);
        }

        /** @brief Executes the given operation
         *  Compiles the program if not previously done
         */
        void execute(bool force_compilation = false){
          std::string program_name = operations_manager_.representation();
          if(force_compilation){
            viennacl::ocl::current_context().delete_program(program_name);
          }
          viennacl::ocl::program & pgm = program();
          operations_manager_.enqueue(pgm);
        }

      private:
        code_generation::operations_handler operations_manager_;
    };
  }
}
#endif // CUSTOM_OPERATION_HPP
