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

      enum vendor_id_type{
        intel_id = 32902,
        nvidia_id = 4318,
        amd_id = 4098,
        unknown = 0
      };

      enum device_type{
        gpu_type = CL_DEVICE_TYPE_GPU,
        cpu_type = CL_DEVICE_TYPE_CPU,
        all_type = CL_DEVICE_TYPE_ALL
      };

      typedef std::string device_name_type;
      typedef std::map<std::size_t, profile_base *> scalartype_map;
      typedef std::map<expression_type, scalartype_map> expression_map;
      typedef std::map<device_name_type, expression_map> device_name_map;
      typedef std::map<device_type, device_name_map> device_type_map;
      typedef std::map<vendor_id_type, device_type_map> database_type;


      /*---------------------------*/
      /*     Global GPU Defaults   */
      /*---------------------------*/
      static database_type init_database(){
        database_type map;

        map[unknown][gpu_type][""][VECTOR_SAXPY_TYPE][4] = new vector_saxpy(1,128,128,true);
        map[unknown][gpu_type][""][MATRIX_SAXPY_TYPE][4] = new matrix_saxpy(1,16,16,16,16,true);
        map[unknown][gpu_type][""][SCALAR_REDUCE_TYPE][4] = new scalar_reduction(1, 128, 128, true);
        map[unknown][gpu_type][""][VECTOR_REDUCE_Ax_TYPE][4] = new vector_reduction(1, 1, 256, 32);
        map[unknown][gpu_type][""][VECTOR_REDUCE_Tx_TYPE][4] = new vector_reduction(1, 1, 256, 32);
        map[unknown][gpu_type][""][MATRIX_PRODUCT_AA_TYPE][4] = new matrix_product(1,32,32,32,4,4,4,1,0,1);
        map[unknown][gpu_type][""][MATRIX_PRODUCT_TA_TYPE][4] = new matrix_product(1,32,32,32,4,4,4,1,0,1);
        map[unknown][gpu_type][""][MATRIX_PRODUCT_AT_TYPE][4] = new matrix_product(1,32,32,32,4,4,4,1,0,1);
        map[unknown][gpu_type][""][MATRIX_PRODUCT_TT_TYPE][4] = new matrix_product(1,32,32,32,4,4,4,1,0,1);


        map[unknown][gpu_type][""][VECTOR_SAXPY_TYPE][8] = new vector_saxpy(1,128,128,true);
        map[unknown][gpu_type][""][MATRIX_SAXPY_TYPE][8] = new matrix_saxpy(1,16,16,16,16,true);
        map[unknown][gpu_type][""][SCALAR_REDUCE_TYPE][8] = new scalar_reduction(1, 128, 128, true);
        map[unknown][gpu_type][""][VECTOR_REDUCE_Ax_TYPE][8] = new vector_reduction(1, 1, 256, 32);
        map[unknown][gpu_type][""][VECTOR_REDUCE_Tx_TYPE][8] = new vector_reduction(1, 1, 256, 32);
        map[unknown][gpu_type][""][MATRIX_PRODUCT_AA_TYPE][8] = new matrix_product(1,32,32,32,4,4,4,1,0,1);
        map[unknown][gpu_type][""][MATRIX_PRODUCT_TA_TYPE][8] = new matrix_product(1,32,32,32,4,4,4,1,0,1);
        map[unknown][gpu_type][""][MATRIX_PRODUCT_AT_TYPE][8] = new matrix_product(1,32,32,32,4,4,4,1,0,1);
        map[unknown][gpu_type][""][MATRIX_PRODUCT_TT_TYPE][8] = new matrix_product(1,32,32,32,4,4,4,1,0,1);

        return map;
      }
      static database_type database = init_database();


      static profile_base * get(viennacl::ocl::device const & device, expression_descriptor const & descriptor){
        return database.at(unknown).at(gpu_type).at("").at(descriptor.type).at(descriptor.scalartype_size);
      }

//      /*---------------------------*/
//      /*     Global CPU Defaults   */
//      /*---------------------------*/
//      static database_type init_global_cpu_default_database(){
//        database_type map;

//        ///SCALARTYPE_SIZE = 4
//        //Vector SAXPY
//        map.insert(std::make_pair(expression_key_type(VECTOR_SAXPY_TYPE, 4), new vector_saxpy(8,16,256,true)));
//        //Matrix SAXPY
//        map.insert(std::make_pair(expression_key_type(MATRIX_SAXPY_TYPE, 4), new matrix_saxpy(1,16,16,16,16,true)));
//        //Scalar Reduce
//        map.insert(std::make_pair(expression_key_type(SCALAR_REDUCE_TYPE, 4), new scalar_reduction(8,8,512)));
//        //Vector Reduce
//        map.insert(std::make_pair(expression_key_type(VECTOR_REDUCE_Ax_TYPE, 4), new vector_reduction(2,1,8)));
//        map.insert(std::make_pair(expression_key_type(VECTOR_REDUCE_Tx_TYPE, 4), new vector_reduction(16,8,8)));
//        //GEMM
//        map.insert(std::make_pair(expression_key_type(MATRIX_PRODUCT_AA_TYPE, 4), new matrix_product(64,64,128,4,4,128,0,0,4,1)));
//        map.insert(std::make_pair(expression_key_type(MATRIX_PRODUCT_TA_TYPE, 4), new matrix_product(128,64,32,16,4,32,0,0,1,1)));
//        map.insert(std::make_pair(expression_key_type(MATRIX_PRODUCT_AT_TYPE, 4), new matrix_product(1,32,32,32,4,4,4,0,0,1)));
//        map.insert(std::make_pair(expression_key_type(MATRIX_PRODUCT_TT_TYPE, 4), new matrix_product(1,32,32,32,4,4,4,0,0,1)));

//        ///SCALARTYPE_SIZE = 8
//        //Vector SAXPY
//        map.insert(std::make_pair(expression_key_type(VECTOR_SAXPY_TYPE, 8), new vector_saxpy(8,16,32)));
//        //Matrix SAXPY
//        map.insert(std::make_pair(expression_key_type(MATRIX_SAXPY_TYPE, 8), new matrix_saxpy(1,16,16,16,16,true)));
//        //Scalar Reduce
//        map.insert(std::make_pair(expression_key_type(SCALAR_REDUCE_TYPE, 8), new scalar_reduction(8,8,512)));
//        //Vector Reduce
//        map.insert(std::make_pair(expression_key_type(VECTOR_REDUCE_Ax_TYPE, 8), new vector_reduction(1,1,8)));
//        map.insert(std::make_pair(expression_key_type(VECTOR_REDUCE_Tx_TYPE, 8), new vector_reduction(8,16,16)));
//        //GEMM
//        map.insert(std::make_pair(expression_key_type(MATRIX_PRODUCT_AA_TYPE, 8), new matrix_product(128,64,64,8,4,64,0,0,2,1)));
//        map.insert(std::make_pair(expression_key_type(MATRIX_PRODUCT_TA_TYPE, 8), new matrix_product(128,128,32,8,4,16,0,0,1,1)));
//        map.insert(std::make_pair(expression_key_type(MATRIX_PRODUCT_AT_TYPE, 8), new matrix_product(1,32,32,32,4,4,4,0,0,1)));
//        map.insert(std::make_pair(expression_key_type(MATRIX_PRODUCT_TT_TYPE, 8), new matrix_product(1,32,32,32,4,4,4,0,0,1)));

//        return map;
//      }
//      static database_type global_gpu_default_database = init_global_cpu_default_database();


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

//      static profile_base * get_nvidia(viennacl::ocl::device const & device, expression_descriptor const & descriptor){

//      }


//      static profile_base * get(viennacl::ocl::device const & device, expression_descriptor const & descriptor){
//        cl_uint vendor_id = device.vendor_id();
//        cl_device_type device_type = device.type();
//        expression_key_type key(descriptor.type, descriptor.scalartype_size);
//        switch (vendor_id) {
//          //NVidia
//          case nvidia_id:
//            if(device_type==CL_DEVICE_TYPE_GPU)
//              return get_nvidia_gpu(device,descriptor);
//            else
//              return global_cpu_default_database.at(key);

//          //AMD
//          case amd_id:
//            if(device_type==CL_DEVICE_TYPE_GPU)
//              return get_amd_gpu(device,descriptor);
//            else
//              return get_amd_cpu(device,descriptor);

//          //Intel
//          case intel_id:
//            if(device_type==CL_DEVICE_TYPE_GPU)
//              return get_intel_gpu(device,descriptor);
//            else
//              return get_intel_cpu(device,descriptor);

//          //Other
//          default:
//            if(device_type==CL_DEVICE_TYPE_GPU)
//              return global_gpu_default_database.at(key);
//            else
//              return global_cpu_default_database.at(key);
//        }
//      }

    }

  }

}


#endif

