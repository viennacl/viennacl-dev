#ifndef VIENNACL_GENERATOR_FORWARDS_H
#define VIENNACL_GENERATOR_FORWARDS_H

/* =========================================================================
   Copyright (c) 2010-2012, Institute for Microelectronics,
                            Institute for Analysis and Scientific Computing,
                            TU Wien.

                            -----------------
                  ViennaCL - The Vienna Computing Library
                            -----------------

   Project Head:    Karl Rupp                   rupp@iue.tuwien.ac.at

   (A list of authors and contributors can be found in the PDF manual)

   License:         MIT (X11), see file LICENSE in the base directory
============================================================================= */

/** @file viennacl/generator/forwards.h
 *  @brief Forward declarations of the important structures for the kernel generator
 *
 *  Generator code contributed by Philippe Tillet
 */

#include <string>
#include "viennacl/forwards.h"

namespace viennacl
{
  namespace generator
  {

    template<class LHS, class OP_TYPE, class RHS>
    class compound_node;

    template<class T>
    struct inner_prod_impl_t;

    struct prod_type;

    struct inner_prod_type;

    template< unsigned int ID, typename SCALARTYPE, unsigned int ALIGNMENT = 1>
    class symbolic_vector;

    template<unsigned int ID,
             typename SCALARTYPE, class F = viennacl::row_major, unsigned int ALIGNMENT = 1>
    class symbolic_matrix;

    template<class REF>
    class tmp_symbolic_matrix;

    template<unsigned int ID,typename SCALARTYPE>
    class cpu_symbolic_scalar;

    template<unsigned int ID,typename SCALARTYPE>
    class gpu_symbolic_scalar;

    template<class Expr, class OP, class Assigned>
    struct MatVecToken;

    template<class Expr, class OP, class Assigned>
    struct MatMatToken;

    template<class Expr,unsigned int NestedLevel>
    struct InProdToken;

    template<class Expr>
    struct ArithmeticToken;

    template<class T>
    struct is_inner_product_leaf;

    template<class T>
    struct is_inner_product_impl;

  }
}
#endif
