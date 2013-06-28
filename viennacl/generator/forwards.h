#ifndef VIENNACL_GENERATOR_FORWARDS_H
#define VIENNACL_GENERATOR_FORWARDS_H

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


/** @file viennacl/generator/forwards.h
 * Forward declarations for the generator
*/


namespace viennacl{

  namespace generator{

    class custom_operation;
    class symbolic_expression_tree_base;
    class symbolic_kernel_argument;
    class symbolic_datastructure;

    template<typename ScalarType> class vector;
    template<typename ScalarType> class scalar;
    template<class VCL_MATRIX> class matrix;

    namespace utils{
      class kernel_generation_stream;
    }

  }

}
#endif // FORWARDS_H
