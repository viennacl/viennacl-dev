#ifndef VIENNACL_DEVICE_SPECIFIC_EXECUTE_HPP
#define VIENNACL_DEVICE_SPECIFIC_EXECUTE_HPP

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


/** @file viennacl/generator/execute.hpp
    @brief the user interface for the code generator
*/

#include <cstring>
#include <vector>
#include <typeinfo>

#include "viennacl/scheduler/forwards.h"
#include "viennacl/device_specific/forwards.h"
#include "viennacl/device_specific/generate.hpp"

#include "viennacl/device_specific/tree_parsing/statement_representation.hpp"
#include "viennacl/device_specific/tree_parsing/set_arguments.hpp"
#include "viennacl/device_specific/tree_parsing/map.hpp"

#include "viennacl/tools/tools.hpp"
#include "viennacl/tools/timer.hpp"

namespace viennacl{

  namespace device_specific{

    void enqueue(template_base & tplt, viennacl::ocl::program & program, statements_container const & statements, binding_policy_t binding_policy = BIND_TO_HANDLE){
      std::string prefix = generate::statements_representation(statements, binding_policy);

      //Get the kernels
      std::list<viennacl::ocl::kernel*> kernels;
      for(unsigned int i = 0 ; i < tplt.num_kernels() ; ++i)
          kernels.push_back(&program.get_kernel(prefix+tools::to_string(i)));

      //Configure
      for(std::list<viennacl::ocl::kernel*>::iterator it = kernels.begin() ; it != kernels.end() ; ++it)
      {
        unsigned int current_arg = 0;
        tools::shared_ptr<symbolic_binder> binder = make_binder(binding_policy);
        tplt.configure_range_enqueue_arguments(std::distance(kernels.begin(), it), statements, **it, current_arg);
        for(typename statements_container::data_type::const_iterator itt = statements.data().begin() ; itt != statements.data().end() ; ++itt)
          tree_parsing::traverse(*itt, itt->root(), tree_parsing::set_arguments_functor(*binder,current_arg,**it));

      }

      //Enqueue
      for(std::list<viennacl::ocl::kernel*>::iterator it = kernels.begin() ; it != kernels.end() ; ++it)
        viennacl::ocl::enqueue(**it);
    }

    inline void execute(template_base & tplt, statements_container const & statements, bool force_compilation = false)
    {
      //Generate program name
      std::string program_name = generate::statements_representation(statements);

      //Retrieve/Compile program
      viennacl::ocl::context & ctx = viennacl::ocl::current_context();
      if(force_compilation)
        ctx.delete_program(program_name);
      if(!ctx.has_program(program_name)){
        std::string src;
        //Headers generation
        src+="#if defined(cl_khr_fp64)\n";
        src+="#  pragma OPENCL EXTENSION cl_khr_fp64: enable\n";
        src+="#elif defined(cl_amd_fp64)\n";
        src+="#  pragma OPENCL EXTENSION cl_amd_fp64: enable\n";
        src+="#endif\n";
        src +=generate::opencl_source(tplt, statements);
        ctx.add_program(src, program_name);
      }

      enqueue(tplt, ctx.get_program(program_name), statements);
    }

    inline void execute(template_base & tplt, scheduler::statement const & statement, bool force_recompilation = false){
      execute(tplt, statements_container(statement), force_recompilation);
    }

  }
}
#endif
