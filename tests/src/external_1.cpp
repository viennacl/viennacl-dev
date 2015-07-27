/* =========================================================================
   Copyright (c) 2010-2015, Institute for Microelectronics,
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

/** \file tests/src/external_1.cpp  Test for external linkage.
*   \test A check for the absence of external linkage (otherwise, library is not truly 'header-only')
**/


//#define VIENNACL_WITH_EIGEN
#define VIENNACL_WITH_UBLAS

//
// *** System
//
#include <iostream>

//
// *** ViennaCL
//
#include "viennacl/scalar.hpp"
#include "viennacl/vector.hpp"
#include "viennacl/matrix.hpp"
#include "viennacl/compressed_matrix.hpp"
#include "viennacl/coordinate_matrix.hpp"
#include "viennacl/ell_matrix.hpp"
#include "viennacl/fft.hpp"
#include "viennacl/hyb_matrix.hpp"
#include "viennacl/sliced_ell_matrix.hpp"
#ifdef VIENNACL_WITH_OPENCL
  #include "viennacl/circulant_matrix.hpp"
  #include "viennacl/hankel_matrix.hpp"
  #include "viennacl/toeplitz_matrix.hpp"
  #include "viennacl/vandermonde_matrix.hpp"
#endif

#include "viennacl/linalg/bicgstab.hpp"
#include "viennacl/linalg/bisect.hpp"
#include "viennacl/linalg/bisect_gpu.hpp"
#include "viennacl/linalg/cg.hpp"
#include "viennacl/linalg/direct_solve.hpp"
#include "viennacl/linalg/gmres.hpp"
#include "viennacl/linalg/ichol.hpp"
#include "viennacl/linalg/ilu.hpp"
#include "viennacl/linalg/inner_prod.hpp"
#include "viennacl/linalg/jacobi_precond.hpp"
#include "viennacl/linalg/norm_1.hpp"
#include "viennacl/linalg/norm_2.hpp"
#include "viennacl/linalg/norm_inf.hpp"
#include "viennacl/linalg/norm_frobenius.hpp"
#include "viennacl/linalg/lanczos.hpp"
#include "viennacl/linalg/qr.hpp"
#include "viennacl/linalg/qr-method.hpp"
#include "viennacl/linalg/row_scaling.hpp"
#include "viennacl/linalg/sum.hpp"
#include "viennacl/linalg/tql2.hpp"


#include "viennacl/misc/bandwidth_reduction.hpp"

#ifdef VIENNACL_WITH_OPENCL
  #include "viennacl/linalg/amg.hpp"
  #include "viennacl/linalg/spai.hpp"
  #include "viennacl/linalg/svd.hpp"
  #include "viennacl/device_specific/execute.hpp"
#endif

#include "viennacl/io/matrix_market.hpp"
#include "viennacl/scheduler/execute.hpp"



//defined in external_2.cpp
void other_func();

//
// -------------------------------------------------------------
//
int main()
{
  typedef float   NumericType;

  //doing nothing but instantiating a few types
  viennacl::scalar<NumericType>  s;
  viennacl::vector<NumericType>  v(10);
  viennacl::matrix<NumericType>  m(10, 10);
  viennacl::compressed_matrix<NumericType>  compr(10, 10);
  viennacl::coordinate_matrix<NumericType>  coord(10, 10);

  //this is the external linkage check:
  other_func();

   std::cout << std::endl;
   std::cout << "------- Test completed --------" << std::endl;
   std::cout << std::endl;


  return EXIT_SUCCESS;
}
