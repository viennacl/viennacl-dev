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
#include "viennacl/device_specific/database.hpp"

#include "viennacl/device_specific/tree_parsing/statement_representation.hpp"
#include "viennacl/device_specific/tree_parsing/set_arguments.hpp"
#include "viennacl/device_specific/tree_parsing/map.hpp"

#include "viennacl/tools/tools.hpp"
#include "viennacl/tools/timer.hpp"

namespace viennacl{

  namespace device_specific{

    template<class TemplateT>
    inline void execute(typename TemplateT::parameters const & params, statements_container const & statements, viennacl::ocl::context & ctx = viennacl::ocl::current_context(), bool force_compilation = false)
    {
      TemplateT tplt(params);

      //Generate program name
      std::string program_name = tree_parsing::statements_representation(statements, BIND_TO_HANDLE);

      //Retrieve/Compile program
      if(force_compilation)
        ctx.delete_program(program_name);
      if(!ctx.has_program(program_name))
      {
        std::string src;
        //Headers generation
        src+="#if defined(cl_khr_fp64)\n";
        src+="#  pragma OPENCL EXTENSION cl_khr_fp64: enable\n";
        src+="#elif defined(cl_amd_fp64)\n";
        src+="#  pragma OPENCL EXTENSION cl_amd_fp64: enable\n";
        src+="#endif\n";
        src +=tplt.generate(statements, "kernel");
        std::cout << src << std::endl;
        ctx.add_program(src, program_name);
      }
      tplt.enqueue(ctx.get_program(program_name), statements, "kernel");
    }

  }
}
#endif
