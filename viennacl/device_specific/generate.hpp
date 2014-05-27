#ifndef VIENNACL_DEVICE_SPECIFIC_GENERATE_HPP
#define VIENNACL_DEVICE_SPECIFIC_GENERATE_HPP

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
#include "viennacl/device_specific/templates/template_base.hpp"
#include "viennacl/device_specific/tree_parsing/statement_representation.hpp"

namespace viennacl{

  namespace device_specific{

    namespace generate{

      inline std::string statements_representation(statements_container const & statements)
      {
          std::vector<char> program_name_vector(256);
          char* program_name = program_name_vector.data();
          unsigned int current_arg = 0;
          void* memory[64] = {NULL};
          for(statements_container::const_iterator it = statements.begin() ; it != statements.end() ; ++it)
              tree_parsing::traverse(it->first, it->second, tree_parsing::statement_representation_functor(memory, current_arg, program_name),true);
          *program_name='\0';
          return std::string(program_name_vector.data());
      }

      inline std::string opencl_source(template_base& tplt, statements_container const & statements)
      {
        tplt.bind_to(&statements);
        std::string prefix = statements_representation(statements);
        return tplt.generate(prefix);
      }

      //CUDA Conversion
      inline std::string opencl_source_to_cuda_source(std::string const & opencl_src)
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
    }

  }
}
#endif
