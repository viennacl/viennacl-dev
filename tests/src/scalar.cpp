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


/** \file tests/src/scalar.cpp  Tests operations for viennacl::scalar objects.
*   \test Tests operations for viennacl::scalar objects.
**/

//
// *** System
//
#include <iostream>
#include <algorithm>
#include <cmath>

//
// *** ViennaCL
//
#include "viennacl/scalar.hpp"

//
// -------------------------------------------------------------
//
template<typename ScalarType>
ScalarType diff(ScalarType & s1, viennacl::scalar<ScalarType> & s2)
{
   viennacl::backend::finish();
   if (std::fabs(s1 - s2) > 0)
      return (s1 - s2) / std::max(std::fabs(s1), std::fabs(s2));
   return 0;
}
//
// -------------------------------------------------------------
//
template< typename NumericT, typename Epsilon >
int test(Epsilon const& epsilon)
{
   int retval = EXIT_SUCCESS;

   NumericT s1 = NumericT(3.1415926);
   NumericT s2 = NumericT(2.71763);
   NumericT s3 = NumericT(42);

   viennacl::scalar<NumericT> vcl_s1;
   viennacl::scalar<NumericT> vcl_s2;
   viennacl::scalar<NumericT> vcl_s3 = 1.0;

   vcl_s1 = s1;
   if ( std::fabs(diff(s1, vcl_s1)) > epsilon )
   {
      std::cout << "# Error at operation: vcl_s1 = s1;" << std::endl;
      std::cout << "  diff: " << fabs(diff(s1, vcl_s1)) << std::endl;
      retval = EXIT_FAILURE;
   }

   vcl_s2 = s2;
   if ( std::fabs(diff(s2, vcl_s2)) > epsilon )
   {
      std::cout << "# Error at operation: vcl_s2 = s2;" << std::endl;
      std::cout << "  diff: " << fabs(diff(s2, vcl_s2)) << std::endl;
      retval = EXIT_FAILURE;
   }

   vcl_s3 = s3;
   if ( std::fabs(diff(s3, vcl_s3)) > epsilon )
   {
      std::cout << "# Error at operation: vcl_s3 = s3;" << std::endl;
      std::cout << "  diff: " << s3 - vcl_s3 << std::endl;
      retval = EXIT_FAILURE;
   }

   NumericT tmp = s2;
   s2 = s1;
   s1 = tmp;
   viennacl::linalg::swap(vcl_s1, vcl_s2);
   if ( fabs(diff(s1, vcl_s1)) > epsilon )
   {
      std::cout << "# Error at operation: swap " << std::endl;
      std::cout << "  diff: " << fabs(diff(s1, vcl_s1)) << std::endl;
      retval = EXIT_FAILURE;
   }

   s1 += s2;
   vcl_s1 += vcl_s2;
   if ( fabs(diff(s1, vcl_s1)) > epsilon )
   {
      std::cout << "# Error at operation: += " << std::endl;
      std::cout << "  diff: " << fabs(diff(s1, vcl_s1)) << std::endl;
      retval = EXIT_FAILURE;
   }

   s1 *= s3;
   vcl_s1 *= vcl_s3;

   if ( fabs(diff(s1, vcl_s1)) > epsilon )
   {
      std::cout << "# Error at operation: *= " << std::endl;
      std::cout << "  diff: " << fabs(diff(s1, vcl_s1)) << std::endl;
      retval = EXIT_FAILURE;
   }

   s1 -= s2;
   vcl_s1 -= vcl_s2;
   if ( fabs(diff(s1, vcl_s1)) > epsilon )
   {
      std::cout << "# Error at operation: -= " << std::endl;
      std::cout << "  diff: " << fabs(diff(s1, vcl_s1)) << std::endl;
      retval = EXIT_FAILURE;
   }

   s1 /= s3;
   vcl_s1 /= vcl_s3;

   if ( fabs(diff(s1, vcl_s1)) > epsilon )
   {
      std::cout << "# Error at operation: /= " << std::endl;
      std::cout << "  diff: " << fabs(diff(s1, vcl_s1)) << std::endl;
      retval = EXIT_FAILURE;
   }

   s1 = vcl_s1;

   s1 = s2 + s3;
   vcl_s1 = vcl_s2 + vcl_s3;
   if ( fabs(diff(s1, vcl_s1)) > epsilon )
   {
      std::cout << "# Error at operation: s1 = s2 + s3 " << std::endl;
      std::cout << "  diff: " << fabs(diff(s1, vcl_s1)) << std::endl;
      retval = EXIT_FAILURE;
   }

   s1 += s2 + s3;
   vcl_s1 += vcl_s2 + vcl_s3;
   if ( fabs(diff(s1, vcl_s1)) > epsilon )
   {
      std::cout << "# Error at operation: s1 += s2 + s3 " << std::endl;
      std::cout << "  diff: " << fabs(diff(s1, vcl_s1)) << std::endl;
      retval = EXIT_FAILURE;
   }

   s1 -= s2 + s3;
   vcl_s1 -= vcl_s2 + vcl_s3;
   if ( fabs(diff(s1, vcl_s1)) > epsilon )
   {
      std::cout << "# Error at operation: s1 -= s2 + s3 " << std::endl;
      std::cout << "  diff: " << fabs(diff(s1, vcl_s1)) << std::endl;
      retval = EXIT_FAILURE;
   }

   s1 = s2 - s3;
   vcl_s1 = vcl_s2 - vcl_s3;
   if ( fabs(diff(s1, vcl_s1)) > epsilon )
   {
      std::cout << "# Error at operation: s1 = s2 - s3 " << std::endl;
      std::cout << "  diff: " << fabs(diff(s1, vcl_s1)) << std::endl;
      retval = EXIT_FAILURE;
   }

   s1 += s2 - s3;
   vcl_s1 += vcl_s2 - vcl_s3;
   if ( fabs(diff(s1, vcl_s1)) > epsilon )
   {
      std::cout << "# Error at operation: s1 += s2 - s3 " << std::endl;
      std::cout << "  diff: " << fabs(diff(s1, vcl_s1)) << std::endl;
      retval = EXIT_FAILURE;
   }

   s1 -= s2 - s3;
   vcl_s1 -= vcl_s2 - vcl_s3;
   if ( fabs(diff(s1, vcl_s1)) > epsilon )
   {
      std::cout << "# Error at operation: s1 -= s2 - s3 " << std::endl;
      std::cout << "  diff: " << fabs(diff(s1, vcl_s1)) << std::endl;
      retval = EXIT_FAILURE;
   }

   s1 = s2 * s3;
   vcl_s1 = vcl_s2 * vcl_s3;
   if ( fabs(diff(s1, vcl_s1)) > epsilon )
   {
      std::cout << "# Error at operation: s1 = s2 * s3 " << std::endl;
      std::cout << "  diff: " << fabs(diff(s1, vcl_s1)) << std::endl;
      retval = EXIT_FAILURE;
   }

   s1 += s2 * s3;
   vcl_s1 += vcl_s2 * vcl_s3;
   if ( fabs(diff(s1, vcl_s1)) > epsilon )
   {
      std::cout << "# Error at operation: s1 += s2 * s3 " << std::endl;
      std::cout << "  diff: " << fabs(diff(s1, vcl_s1)) << std::endl;
      retval = EXIT_FAILURE;
   }

   s1 -= s2 * s3;
   vcl_s1 -= vcl_s2 * vcl_s3;
   if ( fabs(diff(s1, vcl_s1)) > epsilon )
   {
      std::cout << "# Error at operation: s1 -= s2 * s3 " << std::endl;
      std::cout << "  diff: " << fabs(diff(s1, vcl_s1)) << std::endl;
      retval = EXIT_FAILURE;
   }

   s1 = s2 / s3;
   vcl_s1 = vcl_s2 / vcl_s3;
   if ( fabs(diff(s1, vcl_s1)) > epsilon )
   {
      std::cout << "# Error at operation: s1 = s2 / s3 " << std::endl;
      std::cout << "  diff: " << fabs(diff(s1, vcl_s1)) << std::endl;
      retval = EXIT_FAILURE;
   }

   s1 += s2 / s3;
   vcl_s1 += vcl_s2 / vcl_s3;
   if ( fabs(diff(s1, vcl_s1)) > epsilon )
   {
      std::cout << "# Error at operation: s1 += s2 / s3 " << std::endl;
      std::cout << "  diff: " << fabs(diff(s1, vcl_s1)) << std::endl;
      retval = EXIT_FAILURE;
   }

   s1 -= s2 / s3;
   vcl_s1 -= vcl_s2 / vcl_s3;
   if ( fabs(diff(s1, vcl_s1)) > epsilon )
   {
      std::cout << "# Error at operation: s1 -= s2 / s3 " << std::endl;
      std::cout << "  diff: " << fabs(diff(s1, vcl_s1)) << std::endl;
      retval = EXIT_FAILURE;
   }

   // addition with factors, =
   vcl_s1 = s1;

   s1 = s2 * s2 + s3 * s3;
   vcl_s1 = vcl_s2 * s2 + vcl_s3 * s3;
   if ( fabs(diff(s1, vcl_s1)) > epsilon )
   {
      std::cout << "# Error at operation: s1 = s2 * s2 + s3 * s3 " << std::endl;
      std::cout << "  diff: " << fabs(diff(s1, vcl_s1)) << std::endl;
      retval = EXIT_FAILURE;
   }
   vcl_s1 = vcl_s2 * vcl_s2 + vcl_s3 * vcl_s3;
   if ( fabs(diff(s1, vcl_s1)) > epsilon )
   {
      std::cout << "# Error at operation: s1 = s2 * s2 + s3 * s3, second test " << std::endl;
      std::cout << "  diff: " << fabs(diff(s1, vcl_s1)) << std::endl;
      retval = EXIT_FAILURE;
   }

   s1 = s2 * s2 + s3 / s3;
   vcl_s1 = vcl_s2 * s2 + vcl_s3 / s3;
   if ( fabs(diff(s1, vcl_s1)) > epsilon )
   {
      std::cout << "# Error at operation: s1 = s2 * s2 + s3 / s3 " << std::endl;
      std::cout << "  diff: " << fabs(diff(s1, vcl_s1)) << std::endl;
      retval = EXIT_FAILURE;
   }
   vcl_s1 = vcl_s2 * vcl_s2 + vcl_s3 / vcl_s3;
   if ( fabs(diff(s1, vcl_s1)) > epsilon )
   {
      std::cout << "# Error at operation: s1 = s2 * s2 + s3 / s3, second test " << std::endl;
      std::cout << "  diff: " << fabs(diff(s1, vcl_s1)) << std::endl;
      retval = EXIT_FAILURE;
   }

   s1 = s2 / s2 + s3 * s3;
   vcl_s1 = vcl_s2 / s2 + vcl_s3 * s3;
   if ( fabs(diff(s1, vcl_s1)) > epsilon )
   {
      std::cout << "# Error at operation: s1 = s2 / s2 + s3 * s3 " << std::endl;
      std::cout << "  diff: " << fabs(diff(s1, vcl_s1)) << std::endl;
      retval = EXIT_FAILURE;
   }
   vcl_s1 = vcl_s2 / vcl_s2 + vcl_s3 * vcl_s3;
   if ( fabs(diff(s1, vcl_s1)) > epsilon )
   {
      std::cout << "# Error at operation: s1 = s2 / s2 + s3 * s3, second test " << std::endl;
      std::cout << "  diff: " << fabs(diff(s1, vcl_s1)) << std::endl;
      retval = EXIT_FAILURE;
   }

   s1 = s2 / s2 + s3 / s3;
   vcl_s1 = vcl_s2 / s2 + vcl_s3 / s3;
   if ( fabs(diff(s1, vcl_s1)) > epsilon )
   {
      std::cout << "# Error at operation: s1 = s2 / s2 + s3 / s3 " << std::endl;
      std::cout << "  diff: " << fabs(diff(s1, vcl_s1)) << std::endl;
      retval = EXIT_FAILURE;
   }
   vcl_s1 = vcl_s2 / vcl_s2 + vcl_s3 / vcl_s3;
   if ( fabs(diff(s1, vcl_s1)) > epsilon )
   {
      std::cout << "# Error at operation: s1 = s2 / s2 + s3 / s3, second test " << std::endl;
      std::cout << "  diff: " << fabs(diff(s1, vcl_s1)) << std::endl;
      retval = EXIT_FAILURE;
   }

   // addition with factors, +=
   vcl_s1 = s1;

   s1 += s2 * s2 + s3 * s3;
   vcl_s1 += vcl_s2 * s2 + vcl_s3 * s3;
   if ( fabs(diff(s1, vcl_s1)) > epsilon )
   {
      std::cout << "# Error at operation: s1 += s2 * s2 + s3 * s3 " << std::endl;
      std::cout << "  diff: " << fabs(diff(s1, vcl_s1)) << std::endl;
      retval = EXIT_FAILURE;
   }

   s1 += s2 * s2 + s3 / s3;
   vcl_s1 += vcl_s2 * s2 + vcl_s3 / s3;
   if ( fabs(diff(s1, vcl_s1)) > epsilon )
   {
      std::cout << "# Error at operation: s1 += s2 * s2 + s3 / s3 " << std::endl;
      std::cout << "  diff: " << fabs(diff(s1, vcl_s1)) << std::endl;
      retval = EXIT_FAILURE;
   }

   s1 += s2 / s2 + s3 * s3;
   vcl_s1 += vcl_s2 / s2 + vcl_s3 * s3;
   if ( fabs(diff(s1, vcl_s1)) > epsilon )
   {
      std::cout << "# Error at operation: s1 += s2 / s2 + s3 * s3 " << std::endl;
      std::cout << "  diff: " << fabs(diff(s1, vcl_s1)) << std::endl;
      retval = EXIT_FAILURE;
   }

   s1 += s2 / s2 + s3 / s3;
   vcl_s1 += vcl_s2 / s2 + vcl_s3 / s3;
   if ( fabs(diff(s1, vcl_s1)) > epsilon )
   {
      std::cout << "# Error at operation: s1 += s2 / s2 + s3 / s3 " << std::endl;
      std::cout << "  diff: " << fabs(diff(s1, vcl_s1)) << std::endl;
      retval = EXIT_FAILURE;
   }

   // addition with factors, -=
   vcl_s1 = s1;

   s1 -= s2 * s2 + s3 * s3;
   vcl_s1 -= vcl_s2 * s2 + vcl_s3 * s3;
   if ( fabs(diff(s1, vcl_s1)) > epsilon )
   {
      std::cout << "# Error at operation: s1 -= s2 * s2 + s3 * s3 " << std::endl;
      std::cout << "  diff: " << fabs(diff(s1, vcl_s1)) << std::endl;
      retval = EXIT_FAILURE;
   }

   s1 -= s2 * s2 + s3 / s3;
   vcl_s1 -= vcl_s2 * s2 + vcl_s3 / s3;
   if ( fabs(diff(s1, vcl_s1)) > epsilon )
   {
      std::cout << "# Error at operation: s1 -= s2 * s2 + s3 / s3 " << std::endl;
      std::cout << "  diff: " << fabs(diff(s1, vcl_s1)) << std::endl;
      retval = EXIT_FAILURE;
   }

   s1 -= s2 / s2 + s3 * s3;
   vcl_s1 -= vcl_s2 / s2 + vcl_s3 * s3;
   if ( fabs(diff(s1, vcl_s1)) > epsilon )
   {
      std::cout << "# Error at operation: s1 -= s2 / s2 + s3 * s3 " << std::endl;
      std::cout << "  diff: " << fabs(diff(s1, vcl_s1)) << std::endl;
      retval = EXIT_FAILURE;
   }

   s1 -= s2 / s2 + s3 / s3;
   vcl_s1 -= vcl_s2 / s2 + vcl_s3 / s3;
   if ( fabs(diff(s1, vcl_s1)) > epsilon )
   {
      std::cout << "# Error at operation: s1 -= s2 / s2 + s3 / s3 " << std::endl;
      std::cout << "  diff: " << fabs(diff(s1, vcl_s1)) << std::endl;
      retval = EXIT_FAILURE;
   }


   // lenghty expression:

   s1 = s2 + s3 * s2 - s3 / s1;
   vcl_s1 = vcl_s2 + vcl_s3 * vcl_s2 - vcl_s3 / vcl_s1;
   if ( fabs(diff(s1, vcl_s1)) > epsilon )
   {
      std::cout << "# Error at operation: + * - / " << std::endl;
      std::cout << "  diff: " << fabs(diff(s1, vcl_s1)) << std::endl;
      retval = EXIT_FAILURE;
   }

   return retval;
}
//
// -------------------------------------------------------------
//
int main()
{
   std::cout << std::endl;
   std::cout << "----------------------------------------------" << std::endl;
   std::cout << "----------------------------------------------" << std::endl;
   std::cout << "## Test :: Scalar" << std::endl;
   std::cout << "----------------------------------------------" << std::endl;
   std::cout << "----------------------------------------------" << std::endl;
   std::cout << std::endl;

   int retval = EXIT_SUCCESS;

   std::cout << std::endl;
   std::cout << "----------------------------------------------" << std::endl;
   std::cout << std::endl;
   {
      typedef float NumericT;
      NumericT epsilon = NumericT(1.0E-5);
      std::cout << "# Testing setup:" << std::endl;
      std::cout << "  eps:     " << epsilon << std::endl;
      std::cout << "  numeric: float" << std::endl;
      retval = test<NumericT>(epsilon);
      if ( retval == EXIT_SUCCESS )
         std::cout << "# Test passed" << std::endl;
      else
         return retval;
   }
   std::cout << std::endl;
   std::cout << "----------------------------------------------" << std::endl;
   std::cout << std::endl;
#ifdef VIENNACL_WITH_OPENCL
   if ( viennacl::ocl::current_device().double_support() )
#endif
   {
      {
         typedef double NumericT;
         NumericT epsilon = 1.0E-10;
         std::cout << "# Testing setup:" << std::endl;
         std::cout << "  eps:     " << epsilon << std::endl;
         std::cout << "  numeric: double" << std::endl;
         retval = test<NumericT>(epsilon);
         if ( retval == EXIT_SUCCESS )
           std::cout << "# Test passed" << std::endl;
         else
           return retval;
      }
      std::cout << std::endl;
   }

  std::cout << std::endl;
  std::cout << "------- Test completed --------" << std::endl;
  std::cout << std::endl;


   return retval;
}
//
// -------------------------------------------------------------
//

