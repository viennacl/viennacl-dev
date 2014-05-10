#ifndef VIENNACL_DEVICE_SPECIFIC_CODE_GENERATOR_HPP
#define VIENNACL_DEVICE_SPECIFIC_CODE_GENERATOR_HPP

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


/** @file viennacl/generator/generate.hpp
    @brief the user interface for the code generator
*/

#include <cstring>
#include <vector>
#include <typeinfo>

#include "viennacl/scheduler/forwards.h"
#include "viennacl/device_specific/forwards.h"

#include "viennacl/device_specific/profiles.hpp"

#include "viennacl/device_specific/tree_parsing/statement_representation.hpp"
#include "viennacl/device_specific/tree_parsing/set_arguments.hpp"
#include "viennacl/device_specific/tree_parsing/map.hpp"

#include "viennacl/tools/tools.hpp"
#include "viennacl/tools/timer.hpp"
namespace viennacl{

  namespace device_specific{

    typedef std::list< std::pair<scheduler::statement, scheduler::statement_node> > statements_type;

    inline void make_program_name(char * program_name, statements_type const & statements)
    {
      unsigned int current_arg = 0;
      void* memory[64] = {NULL};
      for(statements_type::const_iterator it = statements.begin() ; it != statements.end() ; ++it)
          tree_parsing::traverse(it->first, it->second, tree_parsing::statement_representation_functor(memory, current_arg, program_name),true);
      *program_name='\0';
    }

    inline std::string make_opencl_program_string(template_base & t, statements_type const & statements)
    {
      t.bind_statements(&statements);

      utils::kernel_generation_stream stream;

      //Headers generation
      stream << "#if defined(cl_khr_fp64)\n";
      stream <<  "#  pragma OPENCL EXTENSION cl_khr_fp64: enable\n";
      stream <<  "#elif defined(cl_amd_fp64)\n";
      stream <<  "#  pragma OPENCL EXTENSION cl_amd_fp64: enable\n";
      stream <<  "#endif\n";
      stream << std::endl;

      t(stream);

      return stream.str();
    }

    inline std::string opencl_to_cuda(std::string const & opencl_src)
    {
      std::string res = opencl_src;

      viennacl::tools::find_and_replace(res,"__attribute__","//__attribute__");

      //Pointer
      viennacl::tools::find_and_replace(res, "__global float*", "float*");
      viennacl::tools::find_and_replace(res, "__local float*", "float*");

      viennacl::tools::find_and_replace(res, "__global double*", "double*");
      viennacl::tools::find_and_replace(res, "__local double*", "double*");

      //Qualifiers
      viennacl::tools::find_and_replace(res,"__global","__device__");
      viennacl::tools::find_and_replace(res,"__kernel","__global__");
      viennacl::tools::find_and_replace(res,"__constant","__constant__");
      viennacl::tools::find_and_replace(res,"__local","__shared__");

      //Indexing
      viennacl::tools::find_and_replace(res,"get_num_groups(0)","gridDim.x");
      viennacl::tools::find_and_replace(res,"get_num_groups(1)","gridDim.y");

      viennacl::tools::find_and_replace(res,"get_local_size(0)","blockDim.x");
      viennacl::tools::find_and_replace(res,"get_local_size(1)","blockDim.y");

      viennacl::tools::find_and_replace(res,"get_group_id(0)","blockIdx.x");
      viennacl::tools::find_and_replace(res,"get_group_id(1)","blockIdx.y");

      viennacl::tools::find_and_replace(res,"get_local_id(0)","threadIdx.x");
      viennacl::tools::find_and_replace(res,"get_local_id(1)","threadIdx.y");

      viennacl::tools::find_and_replace(res,"get_global_id(0)","(blockIdx.x*blockDim.x + threadIdx.x)");
      viennacl::tools::find_and_replace(res,"get_global_id(1)","(blockIdx.y*blockDim.y + threadIdx.y)");

      //Synchronization
      viennacl::tools::find_and_replace(res,"barrier(CLK_LOCAL_MEM_FENCE)","__syncthreads()");
      viennacl::tools::find_and_replace(res,"barrier(CLK_GLOBAL_MEM_FENCE)","__syncthreads()");


      return res;
    }


    inline void execute(template_base & t, statements_type const & statements, bool force_compilation = false){
      viennacl::tools::timer tim;

      tim.start();
      t.bind_statements(&statements);

      //Make the program name
      std::vector<char> program_name_vector(256);
      char* program_name = program_name_vector.data();
      make_program_name(program_name,statements);

      viennacl::ocl::context & ctx = viennacl::ocl::current_context();
      if(force_compilation)
        ctx.delete_program(program_name);
      if(!ctx.has_program(program_name))
        ctx.add_program(make_opencl_program_string(t,statements), program_name);
      viennacl::ocl::program & program = ctx.get_program(program_name);

      //Add the kernels
      std::list<viennacl::ocl::kernel*> kernels;
      for(unsigned int i = 0 ; i < t.num_kernels() ; ++i){
        //add kernel name
        char str[32];
        std::sprintf(str,"kernel_%u",i);
        viennacl::ocl::kernel & kernel = program.get_kernel(str);
        kernels.push_back(&kernel);
        //Configure ND Range
        unsigned int current_arg = 0;
        t.configure_range_enqueue_arguments(i, kernel, current_arg);
        //Sets the arguments
        std::set<void *> memory;
        for(typename statements_type::const_iterator it = statements.begin() ; it != statements.end() ; ++it)
          tree_parsing::traverse(it->first, it->second, tree_parsing::set_arguments_functor(memory,current_arg,kernel));
      }

      //Executes the kernels
      for(std::list<viennacl::ocl::kernel*>::iterator it = kernels.begin() ; it != kernels.end() ; ++it)
        viennacl::ocl::enqueue(**it);
    }

    inline void execute(template_base & t, scheduler::statement const & statement, bool force_recompilation = false){
      execute(t, statements_type(1, std::make_pair(statement,statement.array()[0])), force_recompilation);
    }

  }
}
#endif
