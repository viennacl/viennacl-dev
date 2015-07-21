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

/** \file tests/src/iterators.cpp  Tests the iterators in ViennaCL.
*   \test Tests the iterators in ViennaCL.
**/

//
// *** System
//
#include <iostream>
#include <stdlib.h>

//
// *** ViennaCL
//
//#define VCL_BUILD_INFO
//#define VIENNACL_WITH_UBLAS 1
#include "viennacl/matrix.hpp"
#include "viennacl/vector.hpp"

//
// -------------------------------------------------------------
//
template< typename NumericT >
int test()
{
   int retval = EXIT_SUCCESS;
   // --------------------------------------------------------------------------
   typedef viennacl::vector<NumericT>  VclVector;

   VclVector vcl_cont(3);
   vcl_cont[0] = 1;
   vcl_cont[1] = 2;
   vcl_cont[2] = 3;

   //typename VclVector::const_iterator const_iter_def_const;
   //typename VclVector::iterator       iter_def_const;

   for (typename VclVector::const_iterator iter = vcl_cont.begin();
       iter != vcl_cont.end(); iter++)
   {
      std::cout << *iter << std::endl;
   }

   for (typename VclVector::iterator iter = vcl_cont.begin();
       iter != vcl_cont.end(); iter++)
   {
      std::cout << *iter << std::endl;
   }

   // --------------------------------------------------------------------------
   return retval;
}

int main()
{
   std::cout << std::endl;
   std::cout << "----------------------------------------------" << std::endl;
   std::cout << "----------------------------------------------" << std::endl;
   std::cout << "## Test :: Iterators" << std::endl;
   std::cout << "----------------------------------------------" << std::endl;
   std::cout << "----------------------------------------------" << std::endl;
   std::cout << std::endl;

   int retval = EXIT_SUCCESS;

   std::cout << std::endl;
   std::cout << "----------------------------------------------" << std::endl;
   std::cout << std::endl;
   {
      typedef float NumericT;
      std::cout << "# Testing setup:" << std::endl;
      std::cout << "  numeric: float" << std::endl;
      retval = test<NumericT>();
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
         std::cout << "# Testing setup:" << std::endl;
         std::cout << "  numeric: double" << std::endl;
         retval = test<NumericT>();
            if ( retval == EXIT_SUCCESS )
              std::cout << "# Test passed" << std::endl;
            else
              return retval;
      }
      std::cout << std::endl;
      std::cout << "----------------------------------------------" << std::endl;
      std::cout << std::endl;
   }

   std::cout << std::endl;
   std::cout << "------- Test completed --------" << std::endl;
   std::cout << std::endl;


   return retval;
}
