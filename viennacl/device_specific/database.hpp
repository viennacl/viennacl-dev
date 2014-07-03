#ifndef VIENNACL_DEVICE_SPECIFIC_DATABASE_HPP_
#define VIENNACL_DEVICE_SPECIFIC_DATABASE_HPP_

#include "viennacl/ocl/device_utils.hpp"
#include "viennacl/device_specific/forwards.h"
#include "viennacl/device_specific/templates/vector_axpy_template.hpp"
#include "viennacl/device_specific/templates/matrix_axpy_template.hpp"
#include "viennacl/device_specific/templates/reduction_template.hpp"
#include "viennacl/device_specific/templates/row_wise_reduction_template.hpp"
#include "viennacl/scheduler/forwards.h"

namespace viennacl{

  namespace device_specific{

    namespace database{


      /////////////////////
      /// Vector AXPY
      ////////////////////
      static database_type<vector_axpy_template::parameters> vector_axpy = database_type<vector_axpy_template::parameters>
          (unknown_id, CL_DEVICE_TYPE_GPU, UNKNOWN, "", UINT_TYPE, vector_axpy_template::parameters("uint",1,128,128,true))
          (unknown_id, CL_DEVICE_TYPE_GPU, UNKNOWN, "", INT_TYPE, vector_axpy_template::parameters("int",1,128,128,true))
          (unknown_id, CL_DEVICE_TYPE_GPU, UNKNOWN, "", ULONG_TYPE, vector_axpy_template::parameters("ulong",1,128,128,true))
          (unknown_id, CL_DEVICE_TYPE_GPU, UNKNOWN, "", LONG_TYPE, vector_axpy_template::parameters("long",1,128,128,true))
          (unknown_id, CL_DEVICE_TYPE_GPU, UNKNOWN, "", FLOAT_TYPE, vector_axpy_template::parameters("float",1,128,128,true))
          (unknown_id, CL_DEVICE_TYPE_GPU, UNKNOWN, "", DOUBLE_TYPE, vector_axpy_template::parameters("double",1,128,128,true));


      /////////////////////
      /// Reduction
      ////////////////////
      static database_type<reduction_template::parameters> reduction = database_type<reduction_template::parameters>
          (unknown_id, CL_DEVICE_TYPE_GPU, UNKNOWN, "", UINT_TYPE, reduction_template::parameters("uint",1,128,128,true))
          (unknown_id, CL_DEVICE_TYPE_GPU, UNKNOWN, "", INT_TYPE, reduction_template::parameters("int",1,128,128,true))
          (unknown_id, CL_DEVICE_TYPE_GPU, UNKNOWN, "", ULONG_TYPE, reduction_template::parameters("ulong",1,128,128,true))
          (unknown_id, CL_DEVICE_TYPE_GPU, UNKNOWN, "", LONG_TYPE, reduction_template::parameters("long",1,128,128,true))
          (unknown_id, CL_DEVICE_TYPE_GPU, UNKNOWN, "", FLOAT_TYPE, reduction_template::parameters("float",1,128,128,true))
          (unknown_id, CL_DEVICE_TYPE_GPU, UNKNOWN, "", DOUBLE_TYPE, reduction_template::parameters("double",1,128,128,true));

      /////////////////////
      /// Matrix AXPY
      ////////////////////
      static database_type<matrix_axpy_template::parameters> matrix_axpy = database_type<matrix_axpy_template::parameters>
          (unknown_id, CL_DEVICE_TYPE_GPU, UNKNOWN, "", UINT_TYPE, matrix_axpy_template::parameters("uint",1,8,8,8,8,true))
          (unknown_id, CL_DEVICE_TYPE_GPU, UNKNOWN, "", INT_TYPE, matrix_axpy_template::parameters("int",1,8,8,8,8,true))
          (unknown_id, CL_DEVICE_TYPE_GPU, UNKNOWN, "", ULONG_TYPE, matrix_axpy_template::parameters("ulong",1,8,8,8,8,true))
          (unknown_id, CL_DEVICE_TYPE_GPU, UNKNOWN, "", LONG_TYPE, matrix_axpy_template::parameters("long",1,8,8,8,8,true))
          (unknown_id, CL_DEVICE_TYPE_GPU, UNKNOWN, "", FLOAT_TYPE, matrix_axpy_template::parameters("float",1,8,8,8,8,true))
          (unknown_id, CL_DEVICE_TYPE_GPU, UNKNOWN, "", DOUBLE_TYPE, matrix_axpy_template::parameters("double",1,8,8,8,8,true));

      /////////////////////
      /// Row-wise Reduction
      ////////////////////
      static database_type<row_wise_reduction_template::parameters> row_wise_reduction = database_type<row_wise_reduction_template::parameters>
          (unknown_id, CL_DEVICE_TYPE_GPU, UNKNOWN, "", UINT_TYPE, row_wise_reduction_template::parameters("uint",'N',1,8,8,1))
          (unknown_id, CL_DEVICE_TYPE_GPU, UNKNOWN, "", INT_TYPE, row_wise_reduction_template::parameters("int",'N',1,8,8,1))
          (unknown_id, CL_DEVICE_TYPE_GPU, UNKNOWN, "", ULONG_TYPE, row_wise_reduction_template::parameters("ulong",'N',1,8,8,1))
          (unknown_id, CL_DEVICE_TYPE_GPU, UNKNOWN, "", LONG_TYPE, row_wise_reduction_template::parameters("long",'N',1,8,8,1))
          (unknown_id, CL_DEVICE_TYPE_GPU, UNKNOWN, "", FLOAT_TYPE, row_wise_reduction_template::parameters("float",'N',1,8,8,1))
          (unknown_id, CL_DEVICE_TYPE_GPU, UNKNOWN, "", DOUBLE_TYPE, row_wise_reduction_template::parameters("double",'N',1,8,8,1))

          (unknown_id, CL_DEVICE_TYPE_GPU, UNKNOWN, "", UINT_TYPE, row_wise_reduction_template::parameters("uint",'T',1,8,8,1))
          (unknown_id, CL_DEVICE_TYPE_GPU, UNKNOWN, "", INT_TYPE, row_wise_reduction_template::parameters("int",'T',1,8,8,1))
          (unknown_id, CL_DEVICE_TYPE_GPU, UNKNOWN, "", ULONG_TYPE, row_wise_reduction_template::parameters("ulong",'T',1,8,8,1))
          (unknown_id, CL_DEVICE_TYPE_GPU, UNKNOWN, "", LONG_TYPE, row_wise_reduction_template::parameters("long",'T',1,8,8,1))
          (unknown_id, CL_DEVICE_TYPE_GPU, UNKNOWN, "", FLOAT_TYPE, row_wise_reduction_template::parameters("float",'T',1,8,8,1))
          (unknown_id, CL_DEVICE_TYPE_GPU, UNKNOWN, "", DOUBLE_TYPE, row_wise_reduction_template::parameters("double",'T',1,8,8,1));

      /** @brief If the fallback is too harsh, use a very conservative profile */
      template<class ParamT>
      inline ParamT const & handle_failure(database_type<ParamT> const & database, viennacl::ocl::device const & device, scheduler::statement_node_numeric_type numeric_type, ParamT const & params)
      {
        //Returns default if the profile is invalid
        if(params.is_invalid())
          return database.at(ocl::unknown_id, device.type(), ocl::UNKNOWN, "", numeric_type);
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
        typename database_type<ParamT>::type::map_t::const_iterator vendor_it = database.map.d.find(vendor_id);
        //Vendor not recognized => global default:
        if(vendor_it==database.map.d.end())
          return handle_failure(database, device, numeric_type, database.at(ocl::unknown_id, dev_type, ocl::UNKNOWN, "", numeric_type));

        /*-Device Type-*/
        //std::cout << "Looking up device type..." << std::endl;
        typename database_type<ParamT>::device_type_t::map_t::const_iterator device_type_it = vendor_it->second.d.find(dev_type);
        //Device type not recognized for this vendor => global default
        if(device_type_it==vendor_it->second.d.end())
          return handle_failure(database, device, numeric_type, database.at(ocl::unknown_id, dev_type, ocl::UNKNOWN, "", numeric_type));

        /*-Device Architecture-*/
        //std::cout << "Looking up device architecture..." << std::endl;
        typename database_type<ParamT>::device_architecture_t::map_t::const_iterator architecture_it = device_type_it->second.d.find(device_architecture);
        if(architecture_it==device_type_it->second.d.end())
          return handle_failure(database, device, numeric_type, database.at(ocl::unknown_id, dev_type, ocl::UNKNOWN, "", numeric_type));

        /*-Device Name-*/
        //std::cout << "Looking up device name..." << std::endl;
        typename database_type<ParamT>::device_name_t::map_t::const_iterator device_name_it = architecture_it->second.d.find(device_name);
        //Name not found => Vendor default
        if(device_name_it==architecture_it->second.d.end())
          return handle_failure(database, device, numeric_type, database.at(vendor_id, dev_type, device_architecture, "", numeric_type));

        //std::cout << "Looking up expression name.." << std::endl;
        /*-Expression-*/
        typename database_type<ParamT>::expression_t::map_t::const_iterator expression_it = device_name_it->second.d.find(numeric_type);
        //Expression not found => Vendor default
        if(expression_it==device_name_it->second.d.end())
          return handle_failure(database, device, numeric_type, database.at(vendor_id, dev_type, device_architecture, "", numeric_type));

        //std::cout << "Device found in the database! Getting profile..." << std::endl;
        //Everything okay. Return specific profile//
        return handle_failure(database, device, numeric_type, database.at(vendor_id, dev_type, device_architecture, device_name, numeric_type));
      }


    }

  }

}
#endif
