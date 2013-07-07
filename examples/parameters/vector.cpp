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

#ifndef VIENNACL_WITH_OPENCL
  #define VIENNACL_WITH_OPENCL
#endif


#include "viennacl/scalar.hpp"
#include "viennacl/vector.hpp"
#include "viennacl/linalg/inner_prod.hpp"
#include "viennacl/linalg/norm_1.hpp"
#include "viennacl/linalg/norm_2.hpp"
#include "viennacl/linalg/norm_inf.hpp"

#include <iostream>
#include <vector>
#include <stdlib.h>
#include "benchmark-utils.hpp"

template <typename ScalarType, typename VectorType>
class test_data;

#include "common.hpp"
#ifdef ENABLE_VIENNAPROFILER
 #include "common_vprof.hpp"
#endif
#include "vector_functors.hpp"

/*
*   Auto-Tuning for vectors
*/

#define BENCHMARK_VECTOR_SIZE   1000000

//a helper container that holds the objects used during benchmarking
template <typename ScalarType, typename VectorType>
class test_data
{
  public:
    typedef typename VectorType::value_type::value_type   value_type;

    test_data(ScalarType & s1_,
              VectorType & v1_,
              VectorType & v2_,
              VectorType & v3_) : s1(s1_), v1(v1_), v2(v2_), v3(v3_)  {}

    ScalarType & s1;
    VectorType & v1;
    VectorType & v2;
    VectorType & v3;
};



////////////////////// some functions that aid testing to follow /////////////////////////////////



template<typename ScalarType>
int run_vector_benchmark(test_config & config, viennacl::io::parameter_database& paras)
{
  typedef viennacl::scalar<ScalarType>   VCLScalar;
  typedef viennacl::vector<ScalarType>   VCLVector;

  ////////////////////////////////////////////////////////////////////
  //set up a little bit of data to play with:
  //ScalarType std_result = 0;

  ScalarType std_factor1 = static_cast<ScalarType>(3.1415);
  ScalarType std_factor2 = static_cast<ScalarType>(42.0);
  viennacl::scalar<ScalarType> vcl_factor1(std_factor1);
  viennacl::scalar<ScalarType> vcl_factor2(std_factor2);

  std::vector<ScalarType> std_vec1(BENCHMARK_VECTOR_SIZE);  //used to set all values to zero
  VCLVector vcl_vec1(BENCHMARK_VECTOR_SIZE);
  VCLVector vcl_vec2(BENCHMARK_VECTOR_SIZE);
  VCLVector vcl_vec3(BENCHMARK_VECTOR_SIZE);

  viennacl::copy(std_vec1, vcl_vec1); //initialize vectors with all zeros (no need to worry about overflows then)
  viennacl::copy(std_vec1, vcl_vec2); //initialize vectors with all zeros (no need to worry about overflows then)

  typedef test_data<VCLScalar, VCLVector>   TestDataType;
  test_data<VCLScalar, VCLVector> data(vcl_factor1, vcl_vec1, vcl_vec2, vcl_vec3);

  //////////////////////////////////////////////////////////
  ///////////// Start parameter recording  /////////////////
  //////////////////////////////////////////////////////////

  typedef std::map< double, std::pair<unsigned int, unsigned int> >   TimingType;
  std::map< std::string, TimingType > all_timings;

  // vector addition
  std::cout << "------- Related to vector addition ----------" << std::endl;
  config.kernel_name("add");                    optimize_full(paras, all_timings[config.kernel_name()], vector_add<TestDataType>, config, data);
  config.kernel_name("inplace_add");            optimize_full(paras, all_timings[config.kernel_name()], vector_inplace_add<TestDataType>, config, data);
  config.kernel_name("mul_add");                optimize_full(paras, all_timings[config.kernel_name()], vector_mul_add<TestDataType>, config, data);
  config.kernel_name("cpu_mul_add");            optimize_full(paras, all_timings[config.kernel_name()], vector_cpu_mul_add<TestDataType>, config, data);
  config.kernel_name("inplace_mul_add");        optimize_full(paras, all_timings[config.kernel_name()], vector_inplace_mul_add<TestDataType>, config, data);
  config.kernel_name("cpu_inplace_mul_add");    optimize_full(paras, all_timings[config.kernel_name()], vector_cpu_inplace_mul_add<TestDataType>, config, data);
  config.kernel_name("inplace_div_add");        optimize_full(paras, all_timings[config.kernel_name()], vector_inplace_div_add<TestDataType>, config, data);

  std::cout << "------- Related to vector subtraction ----------" << std::endl;
  config.kernel_name("sub");                    optimize_full(paras, all_timings[config.kernel_name()], vector_sub<TestDataType>, config, data);
  config.kernel_name("inplace_sub");            optimize_full(paras, all_timings[config.kernel_name()], vector_inplace_sub<TestDataType>, config, data);
  config.kernel_name("mul_sub");                optimize_full(paras, all_timings[config.kernel_name()], vector_mul_sub<TestDataType>, config, data);
  config.kernel_name("inplace_mul_sub");        optimize_full(paras, all_timings[config.kernel_name()], vector_inplace_mul_sub<TestDataType>, config, data);
  config.kernel_name("inplace_div_sub");        optimize_full(paras, all_timings[config.kernel_name()], vector_inplace_div_sub<TestDataType>, config, data);

  std::cout << "------- Related to vector scaling (mult/div) ----------" << std::endl;
  config.kernel_name("mult");                   optimize_full(paras, all_timings[config.kernel_name()], vector_mult<TestDataType>, config, data);
  config.kernel_name("inplace_mult");           optimize_full(paras, all_timings[config.kernel_name()], vector_inplace_mult<TestDataType>, config, data);
  config.kernel_name("cpu_mult");               optimize_full(paras, all_timings[config.kernel_name()], vector_cpu_mult<TestDataType>, config, data);
  config.kernel_name("cpu_inplace_mult");       optimize_full(paras, all_timings[config.kernel_name()], vector_cpu_inplace_mult<TestDataType>, config, data);
  config.kernel_name("divide");                 optimize_full(paras, all_timings[config.kernel_name()], vector_divide<TestDataType>, config, data);
  config.kernel_name("inplace_divide");         optimize_full(paras, all_timings[config.kernel_name()], vector_inplace_divide<TestDataType>, config, data);

  std::cout << "------- Others ----------" << std::endl;
  config.kernel_name("inner_prod");             optimize_full(paras, all_timings[config.kernel_name()], vector_inner_prod<TestDataType>, config, data);
  config.kernel_name("swap");                   optimize_full(paras, all_timings[config.kernel_name()], vector_swap<TestDataType>, config, data);
  config.kernel_name("clear");                  optimize_full(paras, all_timings[config.kernel_name()], vector_clear<TestDataType>, config, data);
  config.kernel_name("plane_rotation");         optimize_full(paras, all_timings[config.kernel_name()], vector_plane_rotation<TestDataType>, config, data);

  //config.max_work_groups(32); //otherwise failures on 8500 GT
  config.kernel_name("norm_1");                 optimize_restricted(paras, all_timings[config.kernel_name()], vector_norm_1<TestDataType>, config, data);
  config.kernel_name("norm_2");                 optimize_restricted(paras, all_timings[config.kernel_name()], vector_norm_2<TestDataType>, config, data);
  config.kernel_name("norm_inf");               optimize_restricted(paras, all_timings[config.kernel_name()], vector_norm_inf<TestDataType>, config, data);


  //restricted optimizations:
  config.kernel_name("index_norm_inf");         optimize_restricted(paras, all_timings[config.kernel_name()], vector_index_norm_inf<TestDataType>, config, data);


  return 0;
}

int main()
{
  std::cout << std::endl;
  std::cout << "----------------------------------------------" << std::endl;
  std::cout << "               Device Info" << std::endl;
  std::cout << "----------------------------------------------" << std::endl;

  viennacl::ocl::device dev = viennacl::ocl::current_device();

  std::cout << dev.info() << std::endl;

  // -----------------------------------------
  viennacl::io::parameter_database  paras;
  // -----------------------------------------

  std::string devname   = dev.name();
  std::string driver    = dev.driver_version();
  cl_uint compunits     = dev.max_compute_units();
  std::size_t wgsize    = dev.max_work_group_size();

  // -----------------------------------------
   paras.add_device();
   paras.add_data_node(viennacl::io::tag::name, devname);
   paras.add_data_node(viennacl::io::tag::driver, driver);
   paras.add_data_node(viennacl::io::tag::compun, compunits);
   paras.add_data_node(viennacl::io::tag::workgrp, wgsize);
  // -----------------------------------------

  //set up test config:
  test_config conf;
  conf.max_local_size(dev.max_work_group_size());

  // GPU specific test setup:
  if (dev.type() == CL_DEVICE_TYPE_GPU)
  {
    unsigned int units = 1;
    while (2 * units < dev.max_compute_units())
      units *= 2;
    conf.min_work_groups(units);
    conf.max_work_groups(512); //reasonable upper limit on current GPUs

    conf.min_local_size(16); //less than 16 threads per work group is unlikely to have any impact
    //conf.min_local_size(dev.max_workgroup_size()); //testing only
  }
  else if (dev.type() == CL_DEVICE_TYPE_CPU)// CPU specific test setup
  {
    conf.min_work_groups(1);
    conf.max_work_groups(2*dev.max_compute_units()); //reasonable upper limit on current CPUs - more experience needed here!

    conf.min_local_size(1);
  }
  else
  {
    std::cerr << "Unknown device type (neither CPU nor GPU)! Aborting..." << std::endl;
    exit(0);
  }

  std::cout << std::endl;
  std::cout << "----------------------------------------------" << std::endl;
  std::cout << "----------------------------------------------" << std::endl;
  std::cout << "## Benchmark :: Vector" << std::endl;
  std::cout << "----------------------------------------------" << std::endl;

  std::cout << std::endl;
  std::cout << "   -------------------------------" << std::endl;
  std::cout << "   # benchmarking single-precision" << std::endl;
  std::cout << "   -------------------------------" << std::endl;

  // -----------------------------------------
   paras.add_test();
   paras.add_data_node(viennacl::io::tag::name,    viennacl::io::val::vec);
   paras.add_data_node(viennacl::io::tag::numeric, viennacl::io::val::fl);
   paras.add_data_node(viennacl::io::tag::alignment, "1");
  // -----------------------------------------

  //set up test config:
  conf.program_name(viennacl::linalg::opencl::kernels::vector<float>::program_name());

  run_vector_benchmark<float>(conf, paras);

  if( viennacl::ocl::current_device().double_support() )
  {
    std::cout << std::endl;
    std::cout << "   -------------------------------" << std::endl;
    std::cout << "   # benchmarking double-precision" << std::endl;
    std::cout << "   -------------------------------" << std::endl;
  // -----------------------------------------
   paras.add_test();
   paras.add_data_node(viennacl::io::tag::name,    viennacl::io::val::vec);
   paras.add_data_node(viennacl::io::tag::numeric, viennacl::io::val::dbl);
   paras.add_data_node(viennacl::io::tag::alignment, "1");

    conf.program_name(viennacl::linalg::opencl::kernels::vector<double>::program_name());
   // -----------------------------------------
    run_vector_benchmark<double>(conf, paras);
  }
  // -----------------------------------------
  //paras.dump(); // dump to terminal
  paras.dump("vector_parameters.xml"); // dump to outputfile
  //std::ofstream stream; paras.dump(stream);   // dump to stream
  // -----------------------------------------

  std::cout << std::endl;
  std::cout << "//////////////////////////////////////////////////////////////////////" << std::endl;
  std::cout << "// Parameter evaluation for viennacl::vector finished successfully! //" << std::endl;
  std::cout << "//////////////////////////////////////////////////////////////////////" << std::endl;
  return 0;
}

