#ifndef VIENNACL_TOOLS_MATRIX_SOLVE_KERNEL_CLASS_DEDUCER_HPP_
#define VIENNACL_TOOLS_MATRIX_SOLVE_KERNEL_CLASS_DEDUCER_HPP_

/* =========================================================================
   Copyright (c) 2010-2012, Institute for Microelectronics,
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

/** @file matrix_solve_kernel_class_deducer.hpp
    @brief Implementation of a helper meta class for deducing the correct kernels for the dense matrix solver
*/

#include <string>
#include <fstream>
#include <sstream>
#include "viennacl/forwards.h"
#include "viennacl/linalg/kernels/matrix_solve_col_col_kernels.h"
#include "viennacl/linalg/kernels/matrix_solve_col_row_kernels.h"
#include "viennacl/linalg/kernels/matrix_solve_row_col_kernels.h"
#include "viennacl/linalg/kernels/matrix_solve_row_row_kernels.h"

#include <vector>
#include <map>

namespace viennacl
{
  namespace tools
  {
    /** @brief deduces kernel type for A \ B, where A, B, C are MatrixType1 and MatrixType2 */
    template <typename MatrixType1, typename MatrixType2>
    struct MATRIX_SOLVE_KERNEL_CLASS_DEDUCER
    {};
    
    template <typename SCALARTYPE, unsigned int ALIGNMENT>
    struct MATRIX_SOLVE_KERNEL_CLASS_DEDUCER< viennacl::matrix<SCALARTYPE, viennacl::row_major, ALIGNMENT>,
                                              viennacl::matrix<SCALARTYPE, viennacl::row_major, ALIGNMENT> >
    {
      typedef viennacl::linalg::kernels::matrix_solve_row_row<SCALARTYPE, ALIGNMENT>     ResultType;
    };

    template <typename SCALARTYPE, unsigned int ALIGNMENT>
    struct MATRIX_SOLVE_KERNEL_CLASS_DEDUCER< viennacl::matrix<SCALARTYPE, viennacl::row_major, ALIGNMENT>,
                                              viennacl::matrix<SCALARTYPE, viennacl::column_major, ALIGNMENT> >
    {
      typedef viennacl::linalg::kernels::matrix_solve_row_col<SCALARTYPE, ALIGNMENT>     ResultType;
    };

    template <typename SCALARTYPE, unsigned int ALIGNMENT>
    struct MATRIX_SOLVE_KERNEL_CLASS_DEDUCER< viennacl::matrix<SCALARTYPE, viennacl::column_major, ALIGNMENT>,
                                              viennacl::matrix<SCALARTYPE, viennacl::row_major, ALIGNMENT> >
    {
      typedef viennacl::linalg::kernels::matrix_solve_col_row<SCALARTYPE, ALIGNMENT>     ResultType;
    };

    template <typename SCALARTYPE, unsigned int ALIGNMENT>
    struct MATRIX_SOLVE_KERNEL_CLASS_DEDUCER< viennacl::matrix<SCALARTYPE, viennacl::column_major, ALIGNMENT>,
                                              viennacl::matrix<SCALARTYPE, viennacl::column_major, ALIGNMENT> >
    {
      typedef viennacl::linalg::kernels::matrix_solve_col_col<SCALARTYPE, ALIGNMENT>     ResultType;
    };

    // handle matrix_range and matrix_slice:
    template <typename M1, typename M2>
    struct MATRIX_SOLVE_KERNEL_CLASS_DEDUCER< viennacl::matrix_range<M1>, M2>
    {
      typedef typename MATRIX_SOLVE_KERNEL_CLASS_DEDUCER<M1, M2>::ResultType     ResultType;
    };

    template <typename M1, typename M2>
    struct MATRIX_SOLVE_KERNEL_CLASS_DEDUCER< viennacl::matrix_slice<M1>, M2>
    {
      typedef typename MATRIX_SOLVE_KERNEL_CLASS_DEDUCER<M1, M2>::ResultType     ResultType;
    };

    template <typename M1, typename M2>
    struct MATRIX_SOLVE_KERNEL_CLASS_DEDUCER< M1, viennacl::matrix_range<M2> >
    {
      typedef typename MATRIX_SOLVE_KERNEL_CLASS_DEDUCER<M1, M2>::ResultType     ResultType;
    };

    template <typename M1, typename M2>
    struct MATRIX_SOLVE_KERNEL_CLASS_DEDUCER< M1, viennacl::matrix_slice<M2> >
    {
      typedef typename MATRIX_SOLVE_KERNEL_CLASS_DEDUCER<M1, M2>::ResultType     ResultType;
    };

    // resolve ambiguities;
    template <typename M1, typename M2>
    struct MATRIX_SOLVE_KERNEL_CLASS_DEDUCER< viennacl::matrix_range<M1>,
                                              viennacl::matrix_slice<M2> >
    {
      typedef typename MATRIX_SOLVE_KERNEL_CLASS_DEDUCER<M1, M2>::ResultType     ResultType;
    };

    template <typename M1, typename M2>
    struct MATRIX_SOLVE_KERNEL_CLASS_DEDUCER< viennacl::matrix_slice<M1>,
                                              viennacl::matrix_range<M2> >
    {
      typedef typename MATRIX_SOLVE_KERNEL_CLASS_DEDUCER<M1, M2>::ResultType     ResultType;
    };
    
    template <typename M1, typename M2>
    struct MATRIX_SOLVE_KERNEL_CLASS_DEDUCER< viennacl::matrix_range<M1>,
                                              viennacl::matrix_range<M2> >
    {
      typedef typename MATRIX_SOLVE_KERNEL_CLASS_DEDUCER<M1, M2>::ResultType     ResultType;
    };

    template <typename M1, typename M2>
    struct MATRIX_SOLVE_KERNEL_CLASS_DEDUCER< viennacl::matrix_slice<M1>,
                                              viennacl::matrix_slice<M2> >
    {
      typedef typename MATRIX_SOLVE_KERNEL_CLASS_DEDUCER<M1, M2>::ResultType     ResultType;
    };
    
    
  }

}

#endif
