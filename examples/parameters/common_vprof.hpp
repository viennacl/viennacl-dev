/* =========================================================================
   Copyright (c) 2010-2013, Institute for Microelectronics,
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

#include "viennaprofiler/mysqldb.hpp"
#include "viennaprofiler/timer/precisetimer.hpp"
#include "viennaprofiler/host.hpp"
#include "viennaprofiler/profiler.hpp"


template <typename TimingType>
void write_viennaprofiler(TimingType & timings, std::string function_prefix, std::string kernel_name)
{
  ViennaProfiler::MySQLDB dbConn("myhost.example.com", "database", "user", "password");
  ViennaProfiler::PreciseTimer timer; // choose a timer for measuring the execution time
  
  //ViennaProfiler::Host host = dbConn.getHost("pcrupp"); // create a host
  ViennaProfiler::Profiler<ViennaProfiler::MySQLDB, ViennaProfiler::PreciseTimer> myTest(dbConn, timer, "my_machine_name"); // create a Profiler
  myTest.setCollection("ViennaCL");
  myTest.setFunction(function_prefix + " " + kernel_name);
  myTest.setImplementation("default");
  myTest.setSourceCode("not available");
  myTest.setOperations(0);
  
  //do a dummy start (otherwise, date is not written properly)
  myTest.start();
  myTest.stop();

  for (typename TimingType::iterator it = timings.begin();
       it != timings.end(); ++it)
  {
    myTest.addParameter("work groups", it->second.first);
    myTest.addParameter("work group size", it->second.second);
    myTest.setExternalTiming(it->first, BENCHMARK_RUNS);
    myTest.send();
  }
  
  std::cout << "Optimization for " << kernel_name << " written to ViennaProfiler." << std::endl;
}


