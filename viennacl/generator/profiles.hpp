#ifndef VIENNACL_GENERATOR_PROFILES_HPP
#define VIENNACL_GENERATOR_PROFILES_HPP

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

#include "viennacl/ocl/forwards.h"
#include "viennacl/generator/forwards.h"

#include "viennacl/generator/generate_template_base.hpp"
#include "viennacl/generator/generate_saxpy.hpp"
#include "viennacl/generator/generate_scalar_reduction.hpp"
#include "viennacl/generator/generate_vector_reduction.hpp"
#include "viennacl/generator/generate_matrix_product.hpp"

namespace viennacl{

  namespace generator{

    namespace profiles{

      typedef std::pair<expression_descriptor, std::size_t> expression_key;
      typedef std::map<expression_key, profile_base *> database_type;

//      static database_type init_global_gpu_default_database(){
//        database_type res;

//        ///SCALARTYPE_SIZE = 4
//        //Vector SAXPY
//        map.insert(std::make_pair(expression_key(VECTOR_SAXPY_TYPE, 4), new vector_saxpy::profile(1,128,128,true)));
//        //Matrix SAXPY
//        map.insert(std::make_pair(expression_key(MATRIX_SAXPY_TYPE, 4), new matrix_saxpy::profile(1,16,16,16,16,true)));
//        //Scalar Reduce
//        map.insert(std::make_pair(expression_key(SCALAR_REDUCE_TYPE, 4), new scalar_reduction::profile(1, 128, 128, true)));
//        //Vector Reduce
//        map.insert(std::make_pair(expression_key(VECTOR_REDUCE_Ax_TYPE, 4), new vector_reduction::profile(1, 1, 256, 32)));
//        map.insert(std::make_pair(expression_key(VECTOR_REDUCE_Tx_TYPE, 4), new vector_reduction::profile(1, 1, 256, 32)));
//        //GEMM
//        map.insert(std::make_pair(expression_key(MATRIX_PRODUCT_AA_TYPE, 4), new matrix_product::profile(1,32,32,32,4,4,4,1,0,1)));
//        map.insert(std::make_pair(expression_key(MATRIX_PRODUCT_TA_TYPE, 4), new matrix_product::profile(1,32,32,32,4,4,4,0,0,1)));
//        map.insert(std::make_pair(expression_key(MATRIX_PRODUCT_AT_TYPE, 4), new matrix_product::profile(1,32,32,32,4,4,4,1,0,1)));
//        map.insert(std::make_pair(expression_key(MATRIX_PRODUCT_TT_TYPE, 4), new matrix_product::profile(1,32,32,32,4,4,4,0,0,1)));

//        ///SCALARTYPE_SIZE = 8
//        //Vector SAXPY
//        map.insert(std::make_pair(expression_key(VECTOR_SAXPY_TYPE, 8), new vector_saxpy::profile(1,128,128,true)));
//        //Matrix SAXPY
//        map.insert(std::make_pair(expression_key(MATRIX_SAXPY_TYPE, 8), new matrix_saxpy::profile(1,16,16,16,16,true)));
//        //Scalar Reduce
//        map.insert(std::make_pair(expression_key(SCALAR_REDUCE_TYPE, 8), new scalar_reduction::profile(1, 128, 128, true)));
//        //Vector Reduce
//        map.insert(std::make_pair(expression_key(VECTOR_REDUCE_Ax_TYPE, 8), new vector_reduction::profile(1, 1, 256, 32)));
//        map.insert(std::make_pair(expression_key(VECTOR_REDUCE_Tx_TYPE, 8), new vector_reduction::profile(1, 1, 256, 32)));
//        //GEMM
//        map.insert(std::make_pair(expression_key(MATRIX_PRODUCT_AA_TYPE, 8), new matrix_product::profile(1,32,32,32,4,4,4,1,0,1)));
//        map.insert(std::make_pair(expression_key(MATRIX_PRODUCT_TA_TYPE, 8), new matrix_product::profile(1,32,32,32,4,4,4,0,0,1)));
//        map.insert(std::make_pair(expression_key(MATRIX_PRODUCT_AT_TYPE, 8), new matrix_product::profile(1,32,32,32,4,4,4,1,0,1)));
//        map.insert(std::make_pair(expression_key(MATRIX_PRODUCT_TT_TYPE, 8), new matrix_product::profile(1,32,32,32,4,4,4,0,0,1)));

//        return res;
//      }
//      static database_type global_gpu_default_database = init_global_gpu_default_database();



//      static void init_amd_default_profiles(profile_database_type & map){
//        ///SCALARTYPE_SIZE = 4
//        //vector SAXPY
//        map.insert(std::make_pair(expression_key(VECTOR_SAXPY_TYPE, 4), new vector_saxpy::profile(1,4,64,true)));
//        //matrix SAXPY

//        //scalar REDUCE
//        map.insert(std::make_pair(expression_key(SCALAR_REDUCE_TYPE, 4), new scalar_reduction::profile(8, 128, 128, true)));
//        //vector REDUCE
//        map.insert(std::make_pair(expression_key(VECTOR_REDUCE_Ax_TYPE, 4), new vector_reduction::profile(1, 1, 256, 1024)));
//        map.insert(std::make_pair(expression_key(VECTOR_REDUCE_Tx_TYPE, 4), new vector_reduction::profile(1, 32, 8, 256)));
//        //GEMM
//        map.insert(std::make_pair(expression_key(MATRIX_PRODUCT_AA_TYPE, 4), new matrix_product::profile(4,16,64,128,4,4,4,1,0,1)));
//        map.insert(std::make_pair(expression_key(MATRIX_PRODUCT_TA_TYPE, 4), new matrix_product::profile(4,32,128,128,4,4,8,0,0,1)));
//        map.insert(std::make_pair(expression_key(MATRIX_PRODUCT_AT_TYPE, 4), new matrix_product::profile(4,64,32,64,4,8,4,1,0,1)));
//        map.insert(std::make_pair(expression_key(MATRIX_PRODUCT_TT_TYPE, 4), new matrix_product::profile(4,128,64,32,8,4,8,0,0,1)));

//        ///SCALARTYPE_SIZE = 8
//        //vector SAXPY
//        map.insert(std::make_pair(expression_key(VECTOR_SAXPY_TYPE, 8), new vector_saxpy::profile(2,1,64,true)));
//        //matrix SAXPY

//        //scalar REDUCE
//        map.insert(std::make_pair(expression_key(SCALAR_REDUCE_TYPE, 8), new scalar_reduction::profile(2, 256, 64, true)));
//        //vector REDUCE
//        map.insert(std::make_pair(expression_key(VECTOR_REDUCE_Ax_TYPE, 8), new vector_reduction::profile(1, 1, 256, 1024)));
//        map.insert(std::make_pair(expression_key(VECTOR_REDUCE_Tx_TYPE, 8), new vector_reduction::profile(1, 64, 4, 256)));
//        //GEMM
//        map.insert(std::make_pair(expression_key(MATRIX_PRODUCT_AA_TYPE, 8), new matrix_product::profile(4,32,64,128,4,4,4,1,0,1)));
//        map.insert(std::make_pair(expression_key(MATRIX_PRODUCT_TA_TYPE, 8), new matrix_product::profile(2,32,128,128,4,2,4,0,0,1)));
//      }


//      enum vendor_id{
//        intel_id = 32902,
//        nvidia_id = 4318,
//        amd_id = 4098
//      };

//      profile_base::profile * get(viennacl::ocl::device const & device, detail::expression_descriptor const & descriptor){
//          return global_gpu_default_database.at(descriptor);
//        cl_uint vendor_id = device.vendor_id();
//        cl_device_type device_type = device.type();
//        switch(vendor_id){
//          case intel_id:
//            return get_intel(device,descriptor);

//          case nvidia_id:
//            return get_nvidia(device,descriptor);

//          case amd_id:
//            return get_amd(device,descriptor);

//          default:
//            return get_default(descriptor);
//        }
      }

    }

  }

}


#endif

