#ifndef VIENNACL_DEVICE_SPECIFIC_DATABASE_HPP_
#define VIENNACL_DEVICE_SPECIFIC_DATABASE_HPP_

#include "viennacl/ocl/device_utils.hpp"
#include "viennacl/device_specific/forwards.h"
#include "viennacl/device_specific/templates/vector_axpy_template.hpp"
#include "viennacl/device_specific/templates/reduction_template.hpp"
#include "viennacl/scheduler/forwards.h"

namespace viennacl{

  namespace device_specific{

    namespace database{


//      static database_type<vector_axpy_template::parameters> init_axpy_database(){
//        database_type<vector_axpy_template::parameters> map;

//        //GPU Defaults
//        map[unknown_id][CL_DEVICE_TYPE_GPU][UNKNOWN][""][FLOAT_TYPE] = vector_axpy_template::parameters("float",1,128,128,true);
//        map[unknown_id][CL_DEVICE_TYPE_GPU][UNKNOWN][""][DOUBLE_TYPE] = vector_axpy_template::parameters("double",1,128,128,true);
//        //CPU Defaults
//        map[unknown_id][CL_DEVICE_TYPE_CPU][UNKNOWN][""][FLOAT_TYPE] = vector_axpy_template::parameters("float",8,16,256,true);
//        map[unknown_id][CL_DEVICE_TYPE_CPU][UNKNOWN][""][DOUBLE_TYPE] = vector_axpy_template::parameters("double",8,16,32,true);
//        //Accelerator Defaults
//        map[unknown_id][CL_DEVICE_TYPE_ACCELERATOR][UNKNOWN][""][FLOAT_TYPE] = vector_axpy_template::parameters("float",8,16,256,true);
//        map[unknown_id][CL_DEVICE_TYPE_ACCELERATOR][UNKNOWN][""][DOUBLE_TYPE] = vector_axpy_template::parameters("double",8,16,32,true);

//        /* AMD */
//        //Evergreen
//        map[amd_id][CL_DEVICE_TYPE_GPU][Evergreen][""][FLOAT_TYPE] = vector_axpy_template::parameters("float",1,4,64,true);
//        map[amd_id][CL_DEVICE_TYPE_GPU][Evergreen][""][DOUBLE_TYPE] = vector_axpy_template::parameters("double",2,1,64,true);
//        //Southern Islands
//        map[amd_id][CL_DEVICE_TYPE_GPU][SouthernIslands][""][FLOAT_TYPE] = vector_axpy_template::parameters("float",1,4,64,true);
//        map[amd_id][CL_DEVICE_TYPE_GPU][SouthernIslands][""][DOUBLE_TYPE] = vector_axpy_template::parameters("double",2,1,64,true);
//        //Volcanic Islands
//        map[amd_id][CL_DEVICE_TYPE_GPU][VolcanicIslands][""][FLOAT_TYPE] = vector_axpy_template::parameters("float",1,4,64,true);
//        map[amd_id][CL_DEVICE_TYPE_GPU][VolcanicIslands][""][DOUBLE_TYPE] = vector_axpy_template::parameters("double",2,1,64,true);

//        /* NVidia */
//        //Fermi
//        map[nvidia_id][CL_DEVICE_TYPE_GPU][Fermi][""][FLOAT_TYPE]      =    vector_axpy_template::parameters("float",1,1,256,true);
//        map[nvidia_id][CL_DEVICE_TYPE_GPU][Fermi][""][DOUBLE_TYPE]      =    vector_axpy_template::parameters("double",2,1,64,true);
//        //Kepler
//        map[nvidia_id][CL_DEVICE_TYPE_GPU][Kepler][""][FLOAT_TYPE]      =    vector_axpy_template::parameters("float",1,1,256,true);
//        map[nvidia_id][CL_DEVICE_TYPE_GPU][Kepler][""][DOUBLE_TYPE]      =    vector_axpy_template::parameters("double",2,1,64,true);

//        return map;
//      }


//      static database_type<reduction_template::parameters> init_reduction_database(){
//        database_type<reduction_template::parameters> map;

//        //GPU Defaults
//        map[unknown_id][CL_DEVICE_TYPE_GPU][UNKNOWN][""][FLOAT_TYPE] = reduction_template::parameters("float",1,128,128,true);
//        map[unknown_id][CL_DEVICE_TYPE_GPU][UNKNOWN][""][DOUBLE_TYPE] = reduction_template::parameters("double",1,128,128,true);
//        //CPU Defaults
//        map[unknown_id][CL_DEVICE_TYPE_CPU][UNKNOWN][""][FLOAT_TYPE] = reduction_template::parameters("float",1,16,256,true);
//        map[unknown_id][CL_DEVICE_TYPE_CPU][UNKNOWN][""][DOUBLE_TYPE] = reduction_template::parameters("double",1,16,32,true);
//        //Accelerator Defaults
//        map[unknown_id][CL_DEVICE_TYPE_ACCELERATOR][UNKNOWN][""][FLOAT_TYPE] = reduction_template::parameters("float",1,16,256,true);
//        map[unknown_id][CL_DEVICE_TYPE_ACCELERATOR][UNKNOWN][""][DOUBLE_TYPE] = reduction_template::parameters("double",1,16,32,true);

//        return map;
//      }
//      database_type<reduction_template::parameters> reduction = init_reduction_database();

      /////////////////////
      /// Vector AXPY
      ////////////////////
      database_type<vector_axpy_template::parameters> axpy = database_type<vector_axpy_template::parameters>
          (unknown_id, CL_DEVICE_TYPE_GPU, UNKNOWN, "", FLOAT_TYPE, vector_axpy_template::parameters("float",1,128,128,true))
          (unknown_id, CL_DEVICE_TYPE_GPU, UNKNOWN, "", DOUBLE_TYPE, vector_axpy_template::parameters("double",1,128,128,true));


      /////////////////////
      /// Reduction
      ////////////////////
      database_type<reduction_template::parameters> reduction = database_type<reduction_template::parameters>
          (unknown_id, CL_DEVICE_TYPE_GPU, UNKNOWN, "", FLOAT_TYPE, reduction_template::parameters("float",1,128,128,true))
          (unknown_id, CL_DEVICE_TYPE_GPU, UNKNOWN, "", DOUBLE_TYPE, reduction_template::parameters("double",1,128,128,true));


      /** @brief If the fallback is too harsh, use a very conservative profile */
      template<class ParamT>
      inline ParamT const & handle_failure(database_type<ParamT> const & database, viennacl::ocl::device const & device, scheduler::statement_node_numeric_type numeric_type, ParamT const & params)
      {
        //Returns default if the profile is invalid
        if(params.is_invalid())
          return database.map.at(ocl::unknown_id).at(device.type()).at(ocl::UNKNOWN).at("").at(numeric_type);
        return params;
      }

      /** @brief Get the profile for a device and a descriptor */
      template<class T, class ParamT>
      inline ParamT const & get(database_type<ParamT> const & database)
      {
        viennacl::ocl::device const & device = viennacl::ocl::current_device();
        scheduler::statement_node_numeric_type numeric_type = scheduler::statement_node_numeric_type(scheduler::result_of::numeric_type_id<T>::value);

        device_type dev_type = device.type();
        vendor_id_type vendor_id = device.vendor_id();
        ocl::device_architecture_family device_architecture = device.architecture_family();
        std::string const & device_name = device.name();

        //std::cout << "Looking up vendor ID..." << std::endl;
        /*-Vendor ID-*/
        typename database_type<ParamT>::map_type::const_iterator vendor_it = database.map.find(vendor_id);
        //Vendor not recognized => global default:
        if(vendor_it==database.map.end())
          return handle_failure(database, device, numeric_type, database.map.at(ocl::unknown_id).at(dev_type).at(ocl::UNKNOWN).at("").at(numeric_type));

        /*-Device Type-*/
        //std::cout << "Looking up device type..." << std::endl;
        typename database_type<ParamT>::device_type_map::const_iterator device_type_it = vendor_it->second.find(dev_type);
        //Device type not recognized for this vendor => global default
        if(device_type_it==vendor_it->second.end())
          return handle_failure(database, device, numeric_type, database.map.at(ocl::unknown_id).at(dev_type).at(ocl::UNKNOWN).at("").at(numeric_type));

        /*-Device Architecture-*/
        //std::cout << "Looking up device architecture..." << std::endl;
        typename database_type<ParamT>::device_architecture_map::const_iterator architecture_it = device_type_it->second.find(device_architecture);
        if(architecture_it==device_type_it->second.end())
          return handle_failure(database, device, numeric_type, database.map.at(ocl::unknown_id).at(dev_type).at(ocl::UNKNOWN).at("").at(numeric_type));

        /*-Device Name-*/
        //std::cout << "Looking up device name..." << std::endl;
        typename database_type<ParamT>::device_name_map::const_iterator device_name_it = architecture_it->second.find(device_name);
        //Name not found => Vendor default
        if(device_name_it==architecture_it->second.end())
          return handle_failure(database, device, numeric_type, database.map.at(vendor_id).at(dev_type).at(device_architecture).at("").at(numeric_type));

        //std::cout << "Looking up expression name.." << std::endl;
        /*-Expression-*/
        typename database_type<ParamT>::expression_map::const_iterator expression_it = device_name_it->second.find(numeric_type);
        //Expression not found => Vendor default
        if(expression_it==device_name_it->second.end())
          return handle_failure(database, device, numeric_type, database.map.at(vendor_id).at(dev_type).at(device_architecture).at("").at(numeric_type));

        //std::cout << "Device found in the database! Getting profile..." << std::endl;
        //Everything okay. Return specific profile//
        return handle_failure(database, device, numeric_type, database.map.at(vendor_id).at(dev_type).at(device_architecture).at(device_name).at(numeric_type));
      }


    }

  }

}
#endif
