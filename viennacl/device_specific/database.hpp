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

      /////////////////////
      /// AXPY
      ////////////////////
      static database_type init_axpy_database(){
        database_type map;

        //GPU Defaults
        map[unknown_id][CL_DEVICE_TYPE_GPU][UNKNOWN][""][FLOAT_TYPE] = tools::shared_ptr<template_base>(new vector_axpy_template("float",1,128,128,true));
        map[unknown_id][CL_DEVICE_TYPE_GPU][UNKNOWN][""][DOUBLE_TYPE] = tools::shared_ptr<template_base>(new vector_axpy_template("double",1,128,128,true));
        //CPU Defaults
        map[unknown_id][CL_DEVICE_TYPE_CPU][UNKNOWN][""][FLOAT_TYPE] = tools::shared_ptr<template_base>(new vector_axpy_template("float",8,16,256,true));
        map[unknown_id][CL_DEVICE_TYPE_CPU][UNKNOWN][""][DOUBLE_TYPE] = tools::shared_ptr<template_base>(new vector_axpy_template("double",8,16,32,true));
        //Accelerator Defaults
        map[unknown_id][CL_DEVICE_TYPE_ACCELERATOR][UNKNOWN][""][FLOAT_TYPE] = tools::shared_ptr<template_base>(new vector_axpy_template("float",8,16,256,true));
        map[unknown_id][CL_DEVICE_TYPE_ACCELERATOR][UNKNOWN][""][DOUBLE_TYPE] = tools::shared_ptr<template_base>(new vector_axpy_template("double",8,16,32,true));

        /* AMD */
        //Evergreen
        map[amd_id][CL_DEVICE_TYPE_GPU][Evergreen][""][FLOAT_TYPE] = tools::shared_ptr<template_base>(new vector_axpy_template("float",1,4,64,true));
        map[amd_id][CL_DEVICE_TYPE_GPU][Evergreen][""][DOUBLE_TYPE] = tools::shared_ptr<template_base>(new vector_axpy_template("double",2,1,64,true));
        //Southern Islands
        map[amd_id][CL_DEVICE_TYPE_GPU][SouthernIslands][""][FLOAT_TYPE] = tools::shared_ptr<template_base>(new vector_axpy_template("float",1,4,64,true));
        map[amd_id][CL_DEVICE_TYPE_GPU][SouthernIslands][""][DOUBLE_TYPE] = tools::shared_ptr<template_base>(new vector_axpy_template("double",2,1,64,true));
        //Volcanic Islands
        map[amd_id][CL_DEVICE_TYPE_GPU][VolcanicIslands][""][FLOAT_TYPE] = tools::shared_ptr<template_base>(new vector_axpy_template("float",1,4,64,true));
        map[amd_id][CL_DEVICE_TYPE_GPU][VolcanicIslands][""][DOUBLE_TYPE] = tools::shared_ptr<template_base>(new vector_axpy_template("double",2,1,64,true));

        /* NVidia */
        //Fermi
        map[nvidia_id][CL_DEVICE_TYPE_GPU][Fermi][""][FLOAT_TYPE]      =    tools::shared_ptr<template_base>(new vector_axpy_template("float",1,1,256,true));
        map[nvidia_id][CL_DEVICE_TYPE_GPU][Fermi][""][DOUBLE_TYPE]      =    tools::shared_ptr<template_base>(new vector_axpy_template("double",2,1,64,true));
        //Kepler
        map[nvidia_id][CL_DEVICE_TYPE_GPU][Kepler][""][FLOAT_TYPE]      =    tools::shared_ptr<template_base>(new vector_axpy_template("float",1,1,256,true));
        map[nvidia_id][CL_DEVICE_TYPE_GPU][Kepler][""][DOUBLE_TYPE]      =    tools::shared_ptr<template_base>(new vector_axpy_template("double",2,1,64,true));

        return map;
      }
      database_type axpy = init_axpy_database();

      /////////////////////
      /// Reduction
      ////////////////////
      static database_type init_reduction_database(){
        database_type map;

        //GPU Defaults
        map[unknown_id][CL_DEVICE_TYPE_GPU][UNKNOWN][""][FLOAT_TYPE] = tools::shared_ptr<template_base>(new reduction_template("float",1,128,128,true));
        map[unknown_id][CL_DEVICE_TYPE_GPU][UNKNOWN][""][DOUBLE_TYPE] = tools::shared_ptr<template_base>(new reduction_template("double",1,128,128,true));
        //CPU Defaults
        map[unknown_id][CL_DEVICE_TYPE_CPU][UNKNOWN][""][FLOAT_TYPE] = tools::shared_ptr<template_base>(new reduction_template("float",1,16,256,true));
        map[unknown_id][CL_DEVICE_TYPE_CPU][UNKNOWN][""][DOUBLE_TYPE] = tools::shared_ptr<template_base>(new reduction_template("double",1,16,32,true));
        //Accelerator Defaults
        map[unknown_id][CL_DEVICE_TYPE_ACCELERATOR][UNKNOWN][""][FLOAT_TYPE] = tools::shared_ptr<template_base>(new reduction_template("float",1,16,256,true));
        map[unknown_id][CL_DEVICE_TYPE_ACCELERATOR][UNKNOWN][""][DOUBLE_TYPE] = tools::shared_ptr<template_base>(new reduction_template("double",1,16,32,true));

        return map;
      }
      database_type reduction = init_reduction_database();


      /** @brief If the fallback is too harsh, use a very conservative profile */
      static template_base & handle_failure(database_type & database, viennacl::ocl::device const & device, scheduler::statement_node_numeric_type numeric_type, tools::shared_ptr<template_base> const & profile){
        //Returns default if the profile is invalid
        if(profile->is_invalid())
          return *database.map.at(ocl::unknown_id).map.at(device.type()).map.at(ocl::UNKNOWN).map.at("").map.at(numeric_type).get();
        return *profile.get();
      }

      /** @brief Get the profile for a device and a descriptor */
      template<class T>
      static template_base & get(database_type & database){
        viennacl::ocl::device const & device = viennacl::ocl::current_device();
        scheduler::statement_node_numeric_type numeric_type = scheduler::statement_node_numeric_type(scheduler::result_of::numeric_type_id<T>::value);

        device_type dev_type = device.type();
        vendor_id_type vendor_id = device.vendor_id();
        ocl::device_architecture_family device_architecture = device.architecture_family();
        std::string const & device_name = device.name();

        //std::cout << "Looking up vendor ID..." << std::endl;
        /*-Vendor ID-*/
        database_type::map_type::iterator vendor_it = database.map.find(vendor_id);
        //Vendor not recognized => global default:
        if(vendor_it==database.map.end())
          return handle_failure(database, device, numeric_type, database.map.at(ocl::unknown_id).map.at(dev_type).map.at(ocl::UNKNOWN).map.at("").map.at(numeric_type));

        /*-Device Type-*/
        //std::cout << "Looking up device type..." << std::endl;
        device_type_map::map_type::iterator device_type_it = vendor_it->second.map.find(dev_type);
        //Device type not recognized for this vendor => global default
        if(device_type_it==vendor_it->second.map.end())
          return handle_failure(database, device, numeric_type, database.map.at(ocl::unknown_id).map.at(dev_type).map.at(ocl::UNKNOWN).map.at("").map.at(numeric_type));

        /*-Device Architecture-*/
        //std::cout << "Looking up device architecture..." << std::endl;
        device_architecture_map::map_type::iterator architecture_it = device_type_it->second.map.find(device_architecture);
        if(architecture_it==device_type_it->second.map.end())
          return handle_failure(database, device, numeric_type, database.map.at(ocl::unknown_id).map.at(dev_type).map.at(ocl::UNKNOWN).map.at("").map.at(numeric_type));

        /*-Device Name-*/
        //std::cout << "Looking up device name..." << std::endl;
        device_name_map::map_type::iterator device_name_it = architecture_it->second.map.find(device_name);
        //Name not found => Vendor default
        if(device_name_it==architecture_it->second.map.end())
          return handle_failure(database, device, numeric_type, database.map.at(vendor_id).map.at(dev_type).map.at(device_architecture).map.at("").map.at(numeric_type));

        //std::cout << "Looking up expression name.." << std::endl;
        /*-Expression-*/
        expression_map::map_type::iterator expression_it = device_name_it->second.map.find(numeric_type);
        //Expression not found => Vendor default
        if(expression_it==device_name_it->second.map.end())
          return handle_failure(database, device, numeric_type, database.map.at(vendor_id).map.at(dev_type).map.at(device_architecture).map.at("").map.at(numeric_type));

        //std::cout << "Device found in the database! Getting profile..." << std::endl;
        //Everything okay. Return specific profile//
        return handle_failure(database, device, numeric_type, database.map.at(vendor_id).map.at(dev_type).map.at(device_architecture).map.at(device_name).map.at(numeric_type));
      }


    }

  }

}
#endif
