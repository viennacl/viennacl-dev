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

      static const unsigned int intel_id = 32902;
      static const unsigned int nvidia_id = 4318;
      static const unsigned int amd_id = 4098;
      static const unsigned int unknown_id = 0;

      typedef cl_uint vendor_id_type;
      typedef cl_device_type device_type;

      typedef std::string device_name_type;
      typedef std::pair<expression_type, std::size_t> expression_key_type;
      typedef std::map<expression_key_type, profile_base*> expression_map;
      typedef std::map<device_name_type, expression_map> device_name_map;
      typedef std::map<device_type, device_name_map> device_type_map;
      typedef std::map<vendor_id_type, device_type_map> database_type;



      static database_type init_database(){
        database_type map;

        /*---------------------------*/
        /*     GPU Defaults          */
        /*---------------------------*/
        map[unknown_id][CL_DEVICE_TYPE_GPU][""][std::make_pair(VECTOR_SAXPY_TYPE,4)] = new vector_saxpy(1,128,128,true);
        map[unknown_id][CL_DEVICE_TYPE_GPU][""][std::make_pair(MATRIX_SAXPY_TYPE,4)] = new matrix_saxpy(1,16,16,16,16,true);
        map[unknown_id][CL_DEVICE_TYPE_GPU][""][std::make_pair(SCALAR_REDUCE_TYPE,4)] = new scalar_reduction(1, 128, 128, true);
        map[unknown_id][CL_DEVICE_TYPE_GPU][""][std::make_pair(VECTOR_REDUCE_Ax_TYPE,4)] = new vector_reduction(1, 1, 256, 32);
        map[unknown_id][CL_DEVICE_TYPE_GPU][""][std::make_pair(VECTOR_REDUCE_Tx_TYPE,4)] = new vector_reduction(1, 1, 256, 32);
        map[unknown_id][CL_DEVICE_TYPE_GPU][""][std::make_pair(MATRIX_PRODUCT_AA_TYPE,4)] = new matrix_product(1,32,32,32,4,4,4,1,0,1);
        map[unknown_id][CL_DEVICE_TYPE_GPU][""][std::make_pair(MATRIX_PRODUCT_TA_TYPE,4)] = new matrix_product(1,32,32,32,4,4,4,1,0,1);
        map[unknown_id][CL_DEVICE_TYPE_GPU][""][std::make_pair(MATRIX_PRODUCT_AT_TYPE,4)] = new matrix_product(1,32,32,32,4,4,4,1,0,1);
        map[unknown_id][CL_DEVICE_TYPE_GPU][""][std::make_pair(MATRIX_PRODUCT_TT_TYPE,4)] = new matrix_product(1,32,32,32,4,4,4,1,0,1);


        map[unknown_id][CL_DEVICE_TYPE_GPU][""][std::make_pair(VECTOR_SAXPY_TYPE,8)] = new vector_saxpy(1,128,128,true);
        map[unknown_id][CL_DEVICE_TYPE_GPU][""][std::make_pair(MATRIX_SAXPY_TYPE,8)] = new matrix_saxpy(1,16,16,16,16,true);
        map[unknown_id][CL_DEVICE_TYPE_GPU][""][std::make_pair(SCALAR_REDUCE_TYPE,8)] = new scalar_reduction(1, 128, 128, true);
        map[unknown_id][CL_DEVICE_TYPE_GPU][""][std::make_pair(VECTOR_REDUCE_Ax_TYPE,8)] = new vector_reduction(1, 1, 256, 32);
        map[unknown_id][CL_DEVICE_TYPE_GPU][""][std::make_pair(VECTOR_REDUCE_Tx_TYPE,8)] = new vector_reduction(1, 1, 256, 32);
        map[unknown_id][CL_DEVICE_TYPE_GPU][""][std::make_pair(MATRIX_PRODUCT_AA_TYPE,8)] = new matrix_product(1,32,32,32,4,4,4,1,0,1);
        map[unknown_id][CL_DEVICE_TYPE_GPU][""][std::make_pair(MATRIX_PRODUCT_TA_TYPE,8)] = new matrix_product(1,32,32,32,4,4,4,1,0,1);
        map[unknown_id][CL_DEVICE_TYPE_GPU][""][std::make_pair(MATRIX_PRODUCT_AT_TYPE,8)] = new matrix_product(1,32,32,32,4,4,4,1,0,1);
        map[unknown_id][CL_DEVICE_TYPE_GPU][""][std::make_pair(MATRIX_PRODUCT_TT_TYPE,8)] = new matrix_product(1,32,32,32,4,4,4,1,0,1);

        /*---------------------------*/
        /*     CPU Defaults          */
        /*---------------------------*/
        map[unknown_id][CL_DEVICE_TYPE_CPU][""][std::make_pair(VECTOR_SAXPY_TYPE,4)] = new vector_saxpy(8,16,256,true);
        map[unknown_id][CL_DEVICE_TYPE_CPU][""][std::make_pair(MATRIX_SAXPY_TYPE,4)] = new matrix_saxpy(1,16,16,16,16,true);
        map[unknown_id][CL_DEVICE_TYPE_CPU][""][std::make_pair(SCALAR_REDUCE_TYPE,4)] = new scalar_reduction(8,8,512,true);
        map[unknown_id][CL_DEVICE_TYPE_CPU][""][std::make_pair(VECTOR_REDUCE_Ax_TYPE,4)] = new vector_reduction(1,2,1,8);
        map[unknown_id][CL_DEVICE_TYPE_CPU][""][std::make_pair(VECTOR_REDUCE_Tx_TYPE,4)] = new vector_reduction(1,16,8,8);
        map[unknown_id][CL_DEVICE_TYPE_CPU][""][std::make_pair(MATRIX_PRODUCT_AA_TYPE,4)] = new matrix_product(4,64,64,128,4,4,128,0,0,1);
        map[unknown_id][CL_DEVICE_TYPE_CPU][""][std::make_pair(MATRIX_PRODUCT_TA_TYPE,4)] = new matrix_product(1,128,64,32,16,4,32,0,0,1);
        map[unknown_id][CL_DEVICE_TYPE_CPU][""][std::make_pair(MATRIX_PRODUCT_AT_TYPE,4)] = new matrix_product(1,32,32,32,4,4,4,0,0,1);
        map[unknown_id][CL_DEVICE_TYPE_CPU][""][std::make_pair(MATRIX_PRODUCT_TT_TYPE,4)] = new matrix_product(1,32,32,32,4,4,4,0,0,1);


        map[unknown_id][CL_DEVICE_TYPE_CPU][""][std::make_pair(VECTOR_SAXPY_TYPE,8)] = new vector_saxpy(8,16,32,true);
        map[unknown_id][CL_DEVICE_TYPE_CPU][""][std::make_pair(MATRIX_SAXPY_TYPE,8)] = new matrix_saxpy(1,16,16,16,16,true);
        map[unknown_id][CL_DEVICE_TYPE_CPU][""][std::make_pair(SCALAR_REDUCE_TYPE,8)] = new scalar_reduction(8,8,512,true);
        map[unknown_id][CL_DEVICE_TYPE_CPU][""][std::make_pair(VECTOR_REDUCE_Ax_TYPE,8)] = new vector_reduction(1,1,1,8);
        map[unknown_id][CL_DEVICE_TYPE_CPU][""][std::make_pair(VECTOR_REDUCE_Tx_TYPE,8)] = new vector_reduction(1,8,16,16);
        map[unknown_id][CL_DEVICE_TYPE_CPU][""][std::make_pair(MATRIX_PRODUCT_AA_TYPE,8)] = new matrix_product(2,128,64,64,8,4,64,0,0,1);
        map[unknown_id][CL_DEVICE_TYPE_CPU][""][std::make_pair(MATRIX_PRODUCT_TA_TYPE,8)] = new matrix_product(1,128,128,32,8,4,16,0,0,1);
        map[unknown_id][CL_DEVICE_TYPE_CPU][""][std::make_pair(MATRIX_PRODUCT_AT_TYPE,8)] = new matrix_product(1,32,32,32,4,4,4,0,0,1);
        map[unknown_id][CL_DEVICE_TYPE_CPU][""][std::make_pair(MATRIX_PRODUCT_TT_TYPE,8)] = new matrix_product(1,32,32,32,4,4,4,0,0,1);


        /*---------------------------*/
        /*     ACCELERATOR Defaults  */
        /*---------------------------*/
        //same as CPU for now
        map[unknown_id][CL_DEVICE_TYPE_ACCELERATOR][""][std::make_pair(VECTOR_SAXPY_TYPE,4)] = new vector_saxpy(8,16,256,true);
        map[unknown_id][CL_DEVICE_TYPE_ACCELERATOR][""][std::make_pair(MATRIX_SAXPY_TYPE,4)] = new matrix_saxpy(1,16,16,16,16,true);
        map[unknown_id][CL_DEVICE_TYPE_ACCELERATOR][""][std::make_pair(SCALAR_REDUCE_TYPE,4)] = new scalar_reduction(8,8,512,true);
        map[unknown_id][CL_DEVICE_TYPE_ACCELERATOR][""][std::make_pair(VECTOR_REDUCE_Ax_TYPE,4)] = new vector_reduction(1,2,1,8);
        map[unknown_id][CL_DEVICE_TYPE_ACCELERATOR][""][std::make_pair(VECTOR_REDUCE_Tx_TYPE,4)] = new vector_reduction(1,16,8,8);
        map[unknown_id][CL_DEVICE_TYPE_ACCELERATOR][""][std::make_pair(MATRIX_PRODUCT_AA_TYPE,4)] = new matrix_product(64,64,128,4,4,128,0,0,4,1);
        map[unknown_id][CL_DEVICE_TYPE_ACCELERATOR][""][std::make_pair(MATRIX_PRODUCT_TA_TYPE,4)] = new matrix_product(128,64,32,16,4,32,0,0,1,1);
        map[unknown_id][CL_DEVICE_TYPE_ACCELERATOR][""][std::make_pair(MATRIX_PRODUCT_AT_TYPE,4)] = new matrix_product(1,32,32,32,4,4,4,0,0,1);
        map[unknown_id][CL_DEVICE_TYPE_ACCELERATOR][""][std::make_pair(MATRIX_PRODUCT_TT_TYPE,4)] = new matrix_product(1,32,32,32,4,4,4,0,0,1);


        map[unknown_id][CL_DEVICE_TYPE_ACCELERATOR][""][std::make_pair(VECTOR_SAXPY_TYPE,8)] = new vector_saxpy(8,16,32,true);
        map[unknown_id][CL_DEVICE_TYPE_ACCELERATOR][""][std::make_pair(MATRIX_SAXPY_TYPE,8)] = new matrix_saxpy(1,16,16,16,16,true);
        map[unknown_id][CL_DEVICE_TYPE_ACCELERATOR][""][std::make_pair(SCALAR_REDUCE_TYPE,8)] = new scalar_reduction(8,8,512,true);
        map[unknown_id][CL_DEVICE_TYPE_ACCELERATOR][""][std::make_pair(VECTOR_REDUCE_Ax_TYPE,8)] = new vector_reduction(1,1,1,8);
        map[unknown_id][CL_DEVICE_TYPE_ACCELERATOR][""][std::make_pair(VECTOR_REDUCE_Tx_TYPE,8)] = new vector_reduction(1,8,16,16);
        map[unknown_id][CL_DEVICE_TYPE_ACCELERATOR][""][std::make_pair(MATRIX_PRODUCT_AA_TYPE,8)] = new matrix_product(128,64,64,8,4,64,0,0,2,1);
        map[unknown_id][CL_DEVICE_TYPE_ACCELERATOR][""][std::make_pair(MATRIX_PRODUCT_TA_TYPE,8)] = new matrix_product(128,128,32,8,4,16,0,0,1,1);
        map[unknown_id][CL_DEVICE_TYPE_ACCELERATOR][""][std::make_pair(MATRIX_PRODUCT_AT_TYPE,8)] = new matrix_product(1,32,32,32,4,4,4,0,0,1);
        map[unknown_id][CL_DEVICE_TYPE_ACCELERATOR][""][std::make_pair(MATRIX_PRODUCT_TT_TYPE,8)] = new matrix_product(1,32,32,32,4,4,4,0,0,1);



        /*---------------------------*/
        /*     AMD  GPU DEFAULT      */
        /*---------------------------*/
        map[amd_id][CL_DEVICE_TYPE_GPU][""][std::make_pair(VECTOR_SAXPY_TYPE,4)] = new vector_saxpy(1,4,64,true);
        map[amd_id][CL_DEVICE_TYPE_GPU][""][std::make_pair(MATRIX_SAXPY_TYPE,4)] = new matrix_saxpy(1,16,16,16,16,true);
        map[amd_id][CL_DEVICE_TYPE_GPU][""][std::make_pair(SCALAR_REDUCE_TYPE,4)] = new scalar_reduction(8,128,128,true);
        map[amd_id][CL_DEVICE_TYPE_GPU][""][std::make_pair(VECTOR_REDUCE_Ax_TYPE,4)] = new vector_reduction(1,1,256,1024);
        map[amd_id][CL_DEVICE_TYPE_GPU][""][std::make_pair(VECTOR_REDUCE_Tx_TYPE,4)] = new vector_reduction(1,32,8,256);
        map[amd_id][CL_DEVICE_TYPE_GPU][""][std::make_pair(MATRIX_PRODUCT_AA_TYPE,4)] = new matrix_product(4,16,64,128,4,4,4,1,0,1);
        map[amd_id][CL_DEVICE_TYPE_GPU][""][std::make_pair(MATRIX_PRODUCT_TA_TYPE,4)] = new matrix_product(4,32,128,128,4,4,8,0,0,1);
        map[amd_id][CL_DEVICE_TYPE_GPU][""][std::make_pair(MATRIX_PRODUCT_AT_TYPE,4)] = new matrix_product(4,64,32,64,4,8,4,1,0,1);
        map[amd_id][CL_DEVICE_TYPE_GPU][""][std::make_pair(MATRIX_PRODUCT_TT_TYPE,4)] = new matrix_product(4,128,64,32,8,4,8,0,0,1);


        map[amd_id][CL_DEVICE_TYPE_GPU][""][std::make_pair(VECTOR_SAXPY_TYPE,8)] = new vector_saxpy(2,1,64,true);
        map[amd_id][CL_DEVICE_TYPE_GPU][""][std::make_pair(MATRIX_SAXPY_TYPE,8)] = new matrix_saxpy(1,16,16,16,16,true);
        map[amd_id][CL_DEVICE_TYPE_GPU][""][std::make_pair(SCALAR_REDUCE_TYPE,8)] = new scalar_reduction(2,256,64,true);
        map[amd_id][CL_DEVICE_TYPE_GPU][""][std::make_pair(VECTOR_REDUCE_Ax_TYPE,8)] = new vector_reduction(1,1,256,1024);
        map[amd_id][CL_DEVICE_TYPE_GPU][""][std::make_pair(VECTOR_REDUCE_Tx_TYPE,8)] = new vector_reduction(1,64,4,256);
        map[amd_id][CL_DEVICE_TYPE_GPU][""][std::make_pair(MATRIX_PRODUCT_AA_TYPE,8)] = new matrix_product(128,64,64,8,4,64,0,0,2,1);
        map[amd_id][CL_DEVICE_TYPE_GPU][""][std::make_pair(MATRIX_PRODUCT_TA_TYPE,8)] = new matrix_product(128,128,32,8,4,16,0,0,1,1);
        map[amd_id][CL_DEVICE_TYPE_GPU][""][std::make_pair(MATRIX_PRODUCT_AT_TYPE,8)] = new matrix_product(4,32,64,128,4,4,4,1,0,1);
        map[amd_id][CL_DEVICE_TYPE_GPU][""][std::make_pair(MATRIX_PRODUCT_TT_TYPE,8)] = new matrix_product(2,32,128,128,4,2,4,0,0,1);



        /*---------------------------*/
        /*     NVidia  GPU DEFAULT   */
        /*---------------------------*/
        map[nvidia_id][CL_DEVICE_TYPE_GPU][""][std::make_pair(VECTOR_SAXPY_TYPE,4)] = new vector_saxpy(1,1,256,true);
        map[nvidia_id][CL_DEVICE_TYPE_GPU][""][std::make_pair(MATRIX_SAXPY_TYPE,4)] = new matrix_saxpy(1,16,16,16,16,true);
        map[nvidia_id][CL_DEVICE_TYPE_GPU][""][std::make_pair(SCALAR_REDUCE_TYPE,4)] = new scalar_reduction(4,64,512,true);
        map[nvidia_id][CL_DEVICE_TYPE_GPU][""][std::make_pair(VECTOR_REDUCE_Ax_TYPE,4)] = new vector_reduction(1,1,256,1024);
        map[nvidia_id][CL_DEVICE_TYPE_GPU][""][std::make_pair(VECTOR_REDUCE_Tx_TYPE,4)] = new vector_reduction(1,64,4,64);
        map[nvidia_id][CL_DEVICE_TYPE_GPU][""][std::make_pair(MATRIX_PRODUCT_AA_TYPE,4)] = new matrix_product(1,16,128,128,4,4,4,1,0,1);
        map[nvidia_id][CL_DEVICE_TYPE_GPU][""][std::make_pair(MATRIX_PRODUCT_TA_TYPE,4)] = new matrix_product(1,32,32,128,4,4,8,0,0,1);
        map[nvidia_id][CL_DEVICE_TYPE_GPU][""][std::make_pair(MATRIX_PRODUCT_AT_TYPE,4)] = new matrix_product(1,32,128,128,4,8,4,1,0,1);
        map[nvidia_id][CL_DEVICE_TYPE_GPU][""][std::make_pair(MATRIX_PRODUCT_TT_TYPE,4)] = new matrix_product(1,32,32,128,8,4,8,0,0,1);


        map[nvidia_id][CL_DEVICE_TYPE_GPU][""][std::make_pair(VECTOR_SAXPY_TYPE,8)] = new vector_saxpy(2,1,64,true);
        map[nvidia_id][CL_DEVICE_TYPE_GPU][""][std::make_pair(MATRIX_SAXPY_TYPE,8)] = new matrix_saxpy(1,16,16,16,16,true);
        map[nvidia_id][CL_DEVICE_TYPE_GPU][""][std::make_pair(SCALAR_REDUCE_TYPE,8)] = new scalar_reduction(2,64,512,true);
        map[nvidia_id][CL_DEVICE_TYPE_GPU][""][std::make_pair(VECTOR_REDUCE_Ax_TYPE,8)] = new vector_reduction(1,1,128,1024);
        map[nvidia_id][CL_DEVICE_TYPE_GPU][""][std::make_pair(VECTOR_REDUCE_Tx_TYPE,8)] = new vector_reduction(1,16,32,1024);
        map[nvidia_id][CL_DEVICE_TYPE_GPU][""][std::make_pair(MATRIX_PRODUCT_AA_TYPE,8)] = new matrix_product(1,16,64,128,2,2,8,1,0,1);
        map[nvidia_id][CL_DEVICE_TYPE_GPU][""][std::make_pair(MATRIX_PRODUCT_TA_TYPE,8)] = new matrix_product(1,128,128,32,2,2,8,0,1,1);
        map[nvidia_id][CL_DEVICE_TYPE_GPU][""][std::make_pair(MATRIX_PRODUCT_AT_TYPE,8)] = new matrix_product(1,32,64,128,4,4,4,1,0,1);
        map[nvidia_id][CL_DEVICE_TYPE_GPU][""][std::make_pair(MATRIX_PRODUCT_TT_TYPE,8)] = new matrix_product(1,32,128,128,4,2,4,0,0,1);

        return map;
      }
      static database_type database = init_database();


      static profile_base * get(viennacl::ocl::device const & device, expression_descriptor const & descriptor){
        device_type dev_type = device.type();
        vendor_id_type vendor_id = device.vendor_id();
        std::string const & device_name = device.name();
        expression_key_type expression_key(descriptor.type, descriptor.scalartype_size);

        //std::cout << "Looking up vendor ID..." << std::endl;
        /*-Vendor ID-*/
        database_type::iterator vendor_it = database.find(vendor_id);
        //Vendor not recognized => global default:
        if(vendor_it==database.end())
          return database.at(unknown_id).at(dev_type).at("").at(expression_key);

        //std::cout << "Looking up device type..." << std::endl;
        /*-Device Type-*/
        device_type_map::iterator device_type_it = vendor_it->second.find(dev_type);
        //Device type not recognized for this vendor => global default
        if(device_type_it==vendor_it->second.end())
          return database.at(unknown_id).at(dev_type).at("").at(expression_key);

        //std::cout << "Looking up device name..." << std::endl;
        /*-Device Name-*/
        device_name_map::iterator device_name_it = device_type_it->second.find(device_name);
        //Name not found => Vendor default
        if(device_name_it==device_type_it->second.end())
          return database.at(vendor_id).at(dev_type).at("").at(expression_key);

        //std::cout << "Looking up expression name.." << std::endl;
        /*-Expression-*/
        expression_map::iterator expression_it = device_name_it->second.find(expression_key);
        //Expression not found => Vendor default
        if(expression_it==device_name_it->second.end())
          return database.at(vendor_id).at(dev_type).at("").at(expression_key);

        //std::cout << "Device found in the database! Getting profile..." << std::endl;
        //Everything okay. Return specific profile//
        return database.at(vendor_id).at(dev_type).at(device_name).at(expression_key);
      }

    }

  }

}


#endif

