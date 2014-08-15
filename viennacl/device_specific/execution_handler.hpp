#ifndef VIENNACL_DEVICE_SPECIFIC_EXECUTION_HANDLER_HPP
#define VIENNACL_DEVICE_SPECIFIC_EXECUTION_HANDLER_HPP

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


/** @file viennacl/generator/execution_handler.hpp
    @brief Helper for handling fallbacks, lazy compilation, input-dependent kernels, etc
*/

#include <map>

#include "viennacl/tools/shared_ptr.hpp"

#include "viennacl/device_specific/lazy_program_compiler.hpp"
#include "viennacl/device_specific/templates/template_base.hpp"

namespace viennacl
{

  namespace device_specific
  {

    class execution_handler
    {
    public:
      typedef std::map< std::string, tools::shared_ptr<template_base> > container_type;

    private:
      std::string append_prefix(std::string const & str)
      {
        return "_" + str;
      }

      std::string define_extension(std::string const & ext)
      {
        return    "#ifdef " + ext + "\n"
                  "#pragma OPENCL EXTENSION " + ext + " : enable\n"
                  "#endif\n";
      }

      void init_program_compiler(std::string const & name)
      {
        lazy_programs_.push_back(lazy_program_compiler(&ctx_, name));
        lazy_programs_.back().add(define_extension(device_.double_support_extension()));
      }

    public:
      execution_handler(std::string const & program_name_base, viennacl::ocl::context & ctx, viennacl::ocl::device const & device) : ctx_(ctx), device_(device), program_names_(2), init_done_(false)
      {
        lazy_programs_.reserve(2);
        init_program_compiler(program_name_base + "_0");
        init_program_compiler(program_name_base + "_1");
      }

      void add(std::string const & key, template_base * ptr, statements_container const & statements)
      {
        if(kernels_.insert(container_type::value_type(key, tools::shared_ptr<template_base>(ptr))).second)
        {
          std::vector<std::string> sources = ptr->generate(append_prefix(key), statements, device_);
          assert(sources.size()<=2);
          for(unsigned int i = 0 ; i < sources.size() ; ++i)
            lazy_programs_[i].add(sources[i]);
        }
      }

      template_base * template_of(std::string const & key)
      {
        return kernels_.at(key).get();
      }

      void execute(container_type::key_type const & key, statements_container const & statements)
      {
        tools::shared_ptr<template_base> & template_pointer = kernels_.at(key);
        template_pointer->enqueue(append_prefix(key), lazy_programs_, statements);
      }

    private:
      viennacl::ocl::context & ctx_;
      viennacl::ocl::device const & device_;
      container_type kernels_;
      std::vector<std::string> program_names_;
      std::vector<lazy_program_compiler> lazy_programs_;
      bool init_done_;
    };

  }
}
#endif
