/* =========================================================================
   Copyright (c) 2010-2016, Institute for Microelectronics,
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

/** \example viennacl-info.cpp
*
*   This tutorial prints informations about the OpenCL backend and therefore only works with OpenCL enabled.
*   See the viennacl::ocl::device class for full information about which information can be queried from devices.
*
*   We start with including required headers:
**/

// include necessary system headers
#include <iostream>
#include <cstdlib>

//include ViennaCL headers
#include "viennacl/ocl/device.hpp"
#include "viennacl/ocl/platform.hpp"
#include "viennacl/device_specific/builtin_database/common.hpp"

/**
*  In the main() routine we iterate over all OpenCL platforms and print the full device information for each OpenCL device found.
**/
int main()
{
  /**
  *  Retrieve the platforms and iterate:
  **/
  typedef std::vector< viennacl::ocl::platform > platforms_type;
  platforms_type platforms = viennacl::ocl::get_platforms();

  bool is_first_element = true;
  for (platforms_type::iterator platform_iter  = platforms.begin();
                                platform_iter != platforms.end();
                              ++platform_iter)
  {
    typedef std::vector<viennacl::ocl::device> devices_type;
    devices_type devices = platform_iter->devices(CL_DEVICE_TYPE_ALL);

    /**
    *  Print some platform information
    **/
    std::cout << "# =========================================" << std::endl;
    std::cout << "#         Platform Information             " << std::endl;
    std::cout << "# =========================================" << std::endl;

    std::cout << "#" << std::endl;
    std::cout << "# Vendor and version: " << platform_iter->info() << std::endl;
    std::cout << "#" << std::endl;

    if (is_first_element)
    {
      std::cout << "# ViennaCL uses this OpenCL platform by default." << std::endl;
      is_first_element = false;
    }


    /**
    *  Traverse the devices and print all information available using the convenience member function full_info():
    **/
    std::cout << "# " << std::endl;
    std::cout << "# Available Devices: " << std::endl;
    std::cout << "# " << std::endl;
    for (devices_type::iterator iter = devices.begin(); iter != devices.end(); iter++)
    {
      std::cout << std::endl;

      std::cout << "  -----------------------------------------" << std::endl;
      std::cout << iter->full_info();
      std::cout << "ViennaCL Device Architecture:  " << iter->architecture_family() << std::endl;
      std::cout << "ViennaCL Database Mapped Name: " << viennacl::device_specific::builtin_database::get_mapped_device_name(iter->name(), iter->vendor_id()) << std::endl;
      std::cout << "  -----------------------------------------" << std::endl;
    }
    std::cout << std::endl;
    std::cout << "###########################################" << std::endl;
    std::cout << std::endl;
  }

  return EXIT_SUCCESS;
}



