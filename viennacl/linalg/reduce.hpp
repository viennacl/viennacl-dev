#ifndef VIENNACL_LINALG_REDUCE_HPP_
#define VIENNACL_LINALG_REDUCE_HPP_

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

/** @file viennacl/linalg/reduce.hpp
    @brief Generic interface for the computation of inner products. See viennacl/linalg/vector_operations.hpp for implementations.
*/

#include "viennacl/forwards.h"
#include "viennacl/tools/tools.hpp"
#include "viennacl/meta/enable_if.hpp"
#include "viennacl/meta/tag_of.hpp"
#include "viennacl/meta/result_of.hpp"

namespace viennacl
{

  namespace linalg
  {

    template<typename OP, typename NumericT>
    viennacl::scalar_expression< const vector_base<NumericT>, const vector_base<NumericT>, viennacl::op_reduce_vector<OP> >
    reduce(vector_base<NumericT> const & vector)
    {
      return viennacl::scalar_expression< const vector_base<NumericT>,
                                          const vector_base<NumericT>,
                                          viennacl::op_reduce_vector<OP> >(vector, vector);
    }

    template< typename ROP, typename LHS, typename RHS, typename OP>
    viennacl::scalar_expression< const viennacl::vector_expression<LHS, RHS, OP>,
                                 const viennacl::vector_expression<LHS, RHS, OP>,
                                 viennacl::op_reduce_vector<ROP> >
    reduce(viennacl::vector_expression<LHS, RHS, OP> const & vector)
    {
        return  viennacl::scalar_expression< const viennacl::vector_expression<LHS, RHS, OP>,
                                            const viennacl::vector_expression<LHS, RHS, OP>,
                                            viennacl::op_reduce_vector<ROP> >(vector,vector);
    }

    //row-wise reduction
    template<typename ROP, typename NumericT>
    viennacl::vector_expression<const matrix_base<NumericT>, const matrix_base<NumericT>, viennacl::op_reduce_rows<ROP> >
    reduce_rows(matrix_base<NumericT> const & mat)
    {
      return viennacl::vector_expression<const matrix_base<NumericT>, const matrix_base<NumericT>, viennacl::op_reduce_rows<ROP> >(mat, mat);
    }

    //column-wise reduction
    template<typename ROP, typename NumericT>
    viennacl::vector_expression<const matrix_base<NumericT>, const matrix_base<NumericT>, viennacl::op_reduce_columns<ROP> >
    reduce_columns(matrix_base<NumericT> const & mat)
    {
      return viennacl::vector_expression<const matrix_base<NumericT>, const matrix_base<NumericT>, viennacl::op_reduce_columns<ROP> >(mat, mat);
    }


  } // end namespace linalg
} // end namespace viennacl
#endif


