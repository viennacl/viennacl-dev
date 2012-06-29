/* =========================================================================
   Copyright (c) 2010-2012, Institute for Microelectronics,
                            Institute for Analysis and Scientific Computing,
                            TU Wien.

                            -----------------
                  ViennaCL - The Vienna Computing Library
                            -----------------

   Project Head:    Karl Rupp                   rupp@iue.tuwien.ac.at
               
   (A list of authors and contributors can be found in the PDF manual)

   License:         MIT (X11), see file LICENSE in the base directory
============================================================================= */

// include necessary system headers
#include <iostream>

//include ViennaCL headers
#include "viennacl/ocl/device.hpp"
#include "viennacl/ocl/platform.hpp"


int main()
{
   //
   //  retrieve the devices
   //
   viennacl::ocl::platform pf;
   typedef std::vector<viennacl::ocl::device> devices_type;
   devices_type devices = pf.devices(CL_DEVICE_TYPE_ALL);
   
   //
   // print some platform info
   //
   std::cout << "#" << std::endl;
   std::cout << "# ViennaCL uses the OpenCL platform from " << pf.info() << " on this machine." << std::endl;
   std::cout << "#" << std::endl;
   
   //
   //  traverse the devices and print the information
   //
   std::cout << "# Available Devices: " << std::endl;
   std::cout << "# =========================================" << std::endl;
   for(devices_type::iterator iter = devices.begin(); iter != devices.end(); iter++)
   {
      std::cout << std::endl;

      std::cout << "  -----------------------------------------" << std::endl;
      std::cout << "  No.:              " << std::distance(devices.begin(), iter) << std::endl;
      std::cout << "  Name:             " << iter->name() << std::endl;
      std::cout << "  Compute Units:    " << iter->max_compute_units() << std::endl;
      std::cout << "  Workgroup Size:   " << iter->max_workgroup_size() << std::endl;
      std::cout << "  Global Memory:    " << iter->global_memory()/(1024*1024) << " MB" << std::endl;
      std::cout << "  Local Memory:     " << iter->local_memory()/1024 << " KB" << std::endl;
      std::cout << "  Max-alloc Memory: " << iter->max_allocable_memory()/(1024*1024) << " MB" << std::endl;
      std::cout << "  Double Support:   " << iter->double_support() << std::endl;
      std::cout << "  Driver Version:   " << iter->driver_version() << std::endl;
      std::cout << "  -----------------------------------------" << std::endl;
   }
   std::cout << std::endl;
   std::cout << "# =========================================" << std::endl;

   return 0;
}



