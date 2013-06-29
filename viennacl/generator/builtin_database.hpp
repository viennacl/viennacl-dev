#ifndef VIENNACL_GENERATOR_CODE_GENERATION_BUILTIN_DATABASE_HPP
#define VIENNACL_GENERATOR_CODE_GENERATION_BUILTIN_DATABASE_HPP

/* =========================================================================
   Copyright (c) 2010-2013, Institute for Microelectronics,
           Institute for Analysis and Scientific Computing,
           TU Wien.
   Portions of this software are copyright by UChicago Argonne, LLC.

           -----------------
 ViennaCL - The Vienna Computing Library
           -----------------

   Project Head:    Karl Rupp  rupp@iue.tuwien.ac.at

   (A list of authors and contributors can be found in the PDF manual)

   License:         MIT (X11), see file LICENSE in the base directory
============================================================================= */


/** @file viennacl/generator/builtin_database.hpp
 *
 * Vendor-specific parameters for the generated kernels
*/

#include <string>

#include <map>
#include <typeinfo>
#include "viennacl/matrix.hpp"
#include "viennacl/vector.hpp"
#include "viennacl/scalar.hpp"
#include "viennacl/generator/symbolic_types.hpp"
#include "viennacl/generator/overloads.hpp"
#include "viennacl/generator/templates/matrix_product.hpp"
#include "viennacl/generator/templates/vector_reduction.hpp"
#include "viennacl/generator/templates/scalar_reduction.hpp"
#include "viennacl/generator/templates/saxpy_vector.hpp"
#include "viennacl/generator/templates/saxpy_matrix.hpp"
#include "viennacl/tools/shared_ptr.hpp"
#include "viennacl/ocl/forwards.h"


namespace viennacl{

  namespace generator{

    namespace code_generation{


      enum profile_type{
        axpy = 1, aXpY = 2, dot = 3,
        gemvAv = 4, gemvTv = 5,
        gemmAA = 6, gemmTA = 7, gemmAT = 8, gemmTT = 9
      };

      typedef std::pair<profile_type, size_t> profile_id;

      typedef std::map< std::pair<cl_uint, cl_device_type>, std::map<profile_id, profile_base const * > > builtin_database_t;

      static  builtin_database_t make_database(){

        builtin_database_t res;

        //AMD GPUs
        {
          std::map<std::pair<profile_type, size_t>, profile_base const * > tmp;

          //BLAS 1
          tmp.insert(std::make_pair(std::make_pair(axpy,4), new saxpy_vector_profile(1,4,64)));
          tmp.insert(std::make_pair(std::make_pair(dot,4), new scalar_reduction_profile(8,128,128)));

          tmp.insert(std::make_pair(std::make_pair(axpy,8), new saxpy_vector_profile(2,1,64)));
          tmp.insert(std::make_pair(std::make_pair(dot,8), new scalar_reduction_profile(2,256,64)));

          //BLAS2
          tmp.insert(std::make_pair(std::make_pair(gemvAv,4), new vector_reduction_profile(1,256,1024)));
          tmp.insert(std::make_pair(std::make_pair(gemvTv,4), new vector_reduction_profile(32,8,256)));

          tmp.insert(std::make_pair(std::make_pair(gemvAv,8), new vector_reduction_profile(1,256,1024)));
          tmp.insert(std::make_pair(std::make_pair(gemvTv,8), new vector_reduction_profile(64,4,256)));

          //BLAS 3
          tmp.insert(std::make_pair(std::make_pair(gemmAA,4), new matrix_product_profile(16,64,256,4,2,4,1,0,4,1)));
          tmp.insert(std::make_pair(std::make_pair(gemmTA,4), new matrix_product_profile(32,256,256,4,4,8,0,0,4,1)));
          tmp.insert(std::make_pair(std::make_pair(gemmAT,4), new matrix_product_profile(64,32,64,4,8,4,1,0,4,1)));
          tmp.insert(std::make_pair(std::make_pair(gemmTT,4), new matrix_product_profile(128,64,32,8,4,8,0,0,4,1)));

          tmp.insert(std::make_pair(std::make_pair(gemmAA,8), new matrix_product_profile(32,64,128,4,4,4,1,0,4,1)));
          tmp.insert(std::make_pair(std::make_pair(gemmTA,8), new matrix_product_profile(32,128,128,4,2,4,0,0,2,1)));


          res.insert(std::make_pair(std::make_pair(4098,CL_DEVICE_TYPE_GPU),tmp));
        }

        //NVidia GPUs
        {
          std::map<std::pair<profile_type, size_t>, profile_base const * > tmp;

          //BLAS1
          tmp.insert(std::make_pair(std::make_pair(axpy,4), new saxpy_vector_profile(1,1,256)));
          tmp.insert(std::make_pair(std::make_pair(dot,4), new scalar_reduction_profile(4,64,512)));

          tmp.insert(std::make_pair(std::make_pair(axpy,8), new saxpy_vector_profile(2,1,64)));
          tmp.insert(std::make_pair(std::make_pair(dot,8), new scalar_reduction_profile(2,64,512)));

          //BLAS2
          tmp.insert(std::make_pair(std::make_pair(gemvAv,4), new vector_reduction_profile(1,256,1023)));
          tmp.insert(std::make_pair(std::make_pair(gemvTv,4), new vector_reduction_profile(64,4,64)));

          tmp.insert(std::make_pair(std::make_pair(gemvAv,8), new vector_reduction_profile(1,128,1024)));
          tmp.insert(std::make_pair(std::make_pair(gemvTv,8), new vector_reduction_profile(16,32,1024)));


          //BLAS3
          tmp.insert(std::make_pair(std::make_pair(gemmAA,4), new matrix_product_profile(16,128,128,8,4,2,1,0,1,32)));
          tmp.insert(std::make_pair(std::make_pair(gemmAT,4), new matrix_product_profile(32,32,128,2,4,8,1,1,2,1)));
          tmp.insert(std::make_pair(std::make_pair(gemmTA,4), new matrix_product_profile(32,128,256,8,4,2,1,0,1,1)));
          tmp.insert(std::make_pair(std::make_pair(gemmTT,4), new matrix_product_profile(32,32,128,4,4,4,1,1,4,1)));

          tmp.insert(std::make_pair(std::make_pair(gemmAA,8), new matrix_product_profile(16,64,128,2,2,8,1,0,1,32)));
          tmp.insert(std::make_pair(std::make_pair(gemmTA,8), new matrix_product_profile(256,128,32,2,2,8,0,1,2,1)));


          res.insert(std::make_pair(std::make_pair(4318,CL_DEVICE_TYPE_GPU),tmp));
        }

        //Intel CPUs
        {
          std::map<std::pair<profile_type, size_t>, profile_base const * > tmp;

          //BLAS1
          tmp.insert(std::make_pair(std::make_pair(axpy,4), new saxpy_vector_profile(8,16,256)));
          tmp.insert(std::make_pair(std::make_pair(dot,4), new scalar_reduction_profile(8,8,512)));

          tmp.insert(std::make_pair(std::make_pair(axpy,4), new saxpy_vector_profile(8,16,32)));
          tmp.insert(std::make_pair(std::make_pair(dot,4), new scalar_reduction_profile(8,8,512)));


          //BLAS2
          tmp.insert(std::make_pair(std::make_pair(gemvAv,4), new vector_reduction_profile(2,1,8)));
          tmp.insert(std::make_pair(std::make_pair(gemvTv,4), new vector_reduction_profile(16,8,8)));

          tmp.insert(std::make_pair(std::make_pair(gemvAv,8), new vector_reduction_profile(1,1,8)));
          tmp.insert(std::make_pair(std::make_pair(gemvTv,8), new vector_reduction_profile(8,16,16)));

          //BLAS3
          tmp.insert(std::make_pair(std::make_pair(gemmAA,4), new matrix_product_profile(64,64,128,4,4,128,0,0,4,1)));
          tmp.insert(std::make_pair(std::make_pair(gemmTA,4), new matrix_product_profile(128,64,32,16,4,32,0,0,1,1)));

          tmp.insert(std::make_pair(std::make_pair(gemmAA,8), new matrix_product_profile(128,64,64,8,4,64,0,0,2,1)));
          tmp.insert(std::make_pair(std::make_pair(gemmTA,8), new matrix_product_profile(128,128,32,8,4,16,0,0,1,1)));


          res.insert(std::make_pair(std::make_pair(32902,CL_DEVICE_TYPE_CPU),tmp));
          res.insert(std::make_pair(std::make_pair(4098,CL_DEVICE_TYPE_CPU),tmp));
        }
        return res;
      }

      static std::map<profile_type, profile_base const * > make_default_profiles(){
        std::map<profile_type, profile_base const * > res;
        res.insert(std::make_pair(aXpY, new saxpy_matrix_profile(1,128)));
        res.insert(std::make_pair(axpy, new saxpy_vector_profile(1,1,128)));
        res.insert(std::make_pair(dot, new scalar_reduction_profile(1,128,128)));
        res.insert(std::make_pair(gemvAv, new vector_reduction_profile(1,16,64)));
        res.insert(std::make_pair(gemvTv, new vector_reduction_profile(1,16,64)));
        res.insert(std::make_pair(gemmAA, new matrix_product_profile(32,32,32,4,4,4,true,false,1,1)));
        res.insert(std::make_pair(gemmTA, new matrix_product_profile(32,32,32,4,4,4,true,false,1,1)));
        res.insert(std::make_pair(gemmAT, new matrix_product_profile(32,32,32,4,4,4,true,false,1,1)));
        res.insert(std::make_pair(gemmTT, new matrix_product_profile(32,32,32,4,4,4,true,false,1,1)));
        return res;
      }

      static builtin_database_t builtin_database = make_database();
      static std::map<profile_type, profile_base const * > default_profiles = make_default_profiles();

    }

  }

}


#endif

