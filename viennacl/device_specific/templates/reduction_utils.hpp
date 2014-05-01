#ifndef VIENNACL_DEVICE_SPECIFIC_TEMPLATES_REDUCTION_UTILS_HPP
#define VIENNACL_DEVICE_SPECIFIC_TEMPLATES_REDUCTION_UTILS_HPP

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


/** @file viennacl/generator/vector_reduction.hpp
 *
 * Kernel template for the vector reduction operation
*/

#include <vector>

#include "viennacl/scheduler/forwards.h"
#include "viennacl/device_specific/utils.hpp"


namespace viennacl{

  namespace device_specific{

    inline void compute_reduction(utils::kernel_generation_stream & os, std::string const & acc, std::string const & val, scheduler::op_element const & op){
        os << acc << "=";
        if(op.type_subfamily==scheduler::OPERATION_ELEMENTWISE_FUNCTION_TYPE_SUBFAMILY)
            os << tree_parsing::generate(op.type) << "(" << acc << "," << val << ")";
        else
            os << "(" << acc << ")" << tree_parsing::generate(op.type)  << "(" << val << ")";
        os << ";" << std::endl;
    }

    inline void reduce_1d_local_memory(utils::kernel_generation_stream & stream, std::size_t size, std::vector<std::string> const & bufs, std::vector<scheduler::op_element> const & rops)
    {
        //Reduce local memory
        for(std::size_t stride = size/2 ; stride>0 ; stride /=2){
          stream << "barrier(CLK_LOCAL_MEM_FENCE); " << std::endl;
          stream << "if(lid < " << stride << "){" << std::endl;
          stream.inc_tab();
          for(std::size_t k = 0 ; k < bufs.size() ; ++k){
              std::string acc = bufs[k] + "[lid]";
              std::string str = bufs[k] + "[lid + " + utils::to_string(stride) + "]";
              compute_reduction(stream,acc,str,rops[k]);
          }
          stream.dec_tab();
          stream << "}" << std::endl;
        }
    }

    inline std::string neutral_element(scheduler::op_element const & op){
      switch(op.type){
        case scheduler::OPERATION_BINARY_ADD_TYPE : return "0";
        case scheduler::OPERATION_BINARY_MULT_TYPE : return "1";
        case scheduler::OPERATION_BINARY_DIV_TYPE : return "1";
        case scheduler::OPERATION_BINARY_ELEMENT_FMAX_TYPE : return "-INFINITY";
        case scheduler::OPERATION_BINARY_ELEMENT_FMIN_TYPE : return "INFINITY";
        default: throw generator_not_supported_exception("Unsupported reduction operator : no neutral element known");
      }
    }

    inline scheduler::operation_node_type_subfamily get_subfamily(scheduler::operation_node_type const & op){
      if(op==scheduler::OPERATION_BINARY_ELEMENT_FMAX_TYPE || op==scheduler::OPERATION_BINARY_ELEMENT_FMIN_TYPE)
        return scheduler::OPERATION_ELEMENTWISE_FUNCTION_TYPE_SUBFAMILY;
      else
        return scheduler::OPERATION_ELEMENTWISE_OPERATOR_TYPE_SUBFAMILY;
    }

  }
}

#endif
