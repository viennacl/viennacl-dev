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
#include "viennacl/compressed_matrix.hpp"
#include "viennacl/linalg/prod.hpp"

#include <iostream>
#include <vector>
#include <stdlib.h>
#include "benchmark-utils.hpp"

template <typename ScalarType, typename VectorType, typename MatrixType>
class test_data;

#include "common.hpp"
#ifdef ENABLE_VIENNAPROFILER
 #include "common_vprof.hpp"
#endif

template <typename TestData>
void matrix_vec_mul(TestData & data)
{
  data.v2 = viennacl::linalg::prod(data.mat, data.v1);
}

/*
*   Auto-Tuning for dense matrix kernels
*/

#define BENCHMARK_MATRIX_SIZE   123456

//a helper container that holds the objects used during benchmarking
template <typename ScalarType, typename VectorType, typename MatrixType>
class test_data
{
  public:
    typedef typename VectorType::value_type::value_type   value_type;

    test_data(ScalarType & s1_,
              VectorType & v1_,
              VectorType & v2_,
              MatrixType & mat_) : s1(s1_), v1(v1_), v2(v2_), mat(mat_)  {}

    ScalarType & s1;
    VectorType & v1;
    VectorType & v2;
    MatrixType & mat;
};




////////////////////// some functions that aid testing to follow /////////////////////////////////


template<typename ScalarType>
int run_matrix_benchmark(test_config & config, viennacl::io::parameter_database& paras)
{
  typedef viennacl::scalar<ScalarType>   VCLScalar;
  typedef viennacl::vector<ScalarType>   VCLVector;
  typedef viennacl::compressed_matrix<ScalarType>   VCLMatrix;

  ////////////////////////////////////////////////////////////////////
  //set up a little bit of data to play with:
  //ScalarType std_result = 0;

  ScalarType std_factor1 = static_cast<ScalarType>(3.1415);
  ScalarType std_factor2 = static_cast<ScalarType>(42.0);
  viennacl::scalar<ScalarType> vcl_factor1(std_factor1);
  viennacl::scalar<ScalarType> vcl_factor2(std_factor2);

  std::vector<ScalarType> std_vec1(BENCHMARK_MATRIX_SIZE);  //used to set all values to zero
  std::vector< std::map< unsigned int, ScalarType> > stl_mat(BENCHMARK_MATRIX_SIZE);  //store identity matrix here
  VCLVector vcl_vec1(BENCHMARK_MATRIX_SIZE);
  VCLVector vcl_vec2(BENCHMARK_MATRIX_SIZE);
  VCLMatrix vcl_mat(BENCHMARK_MATRIX_SIZE, BENCHMARK_MATRIX_SIZE);

  for (int i=0; i<BENCHMARK_MATRIX_SIZE; ++i)
  {
      if (i > 10)
      {
          stl_mat[i][i - 10] = 1.0;
          stl_mat[i][i - 7] = 1.0;
          stl_mat[i][i - 4] = 1.0;
          stl_mat[i][i - 2] = 1.0;
      }
      stl_mat[i][i] = 1.0;
      if (i + 10 < BENCHMARK_MATRIX_SIZE)
      {
          stl_mat[i][i + 5] = 1.0;
          stl_mat[i][i + 7] = 1.0;
          stl_mat[i][i + 9] = 1.0;
          stl_mat[i][i + 10] = 1.0;
      }
  }

  viennacl::copy(std_vec1, vcl_vec1); //initialize vectors with all zeros (no need to worry about overflows then)
  viennacl::copy(std_vec1, vcl_vec2); //initialize vectors with all zeros (no need to worry about overflows then)
  viennacl::copy(stl_mat, vcl_mat);

  typedef test_data<VCLScalar, VCLVector, VCLMatrix>   TestDataType;
  test_data<VCLScalar, VCLVector, VCLMatrix> data(vcl_factor1, vcl_vec1, vcl_vec2, vcl_mat);

  //////////////////////////////////////////////////////////
  ///////////// Start parameter recording  /////////////////
  //////////////////////////////////////////////////////////

  typedef std::map< double, std::pair<unsigned int, unsigned int> >   TimingType;
  std::map< std::string, TimingType > all_timings;


  //other kernels:
  std::cout << "------- Related to other operations ----------" << std::endl;

  config.kernel_name("vec_mul");
  optimize_full(paras, all_timings[config.kernel_name()],
                      matrix_vec_mul<TestDataType>, config, data);


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
    //conf.min_local_size(dev.max_work_group_size()); //testing only
  }
  else if (dev.type() == CL_DEVICE_TYPE_CPU)// CPU specific test setup
  {
    conf.min_work_groups(1);
    conf.max_work_groups(256); //CPUs don't behave much different from GPUs w.r.t. OpenCL, so test large work group ranges as well

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
  std::cout << "## Benchmark :: Matrix" << std::endl;
  std::cout << "----------------------------------------------" << std::endl;

  std::cout << std::endl;
  std::cout << "   -------------------------------" << std::endl;
  std::cout << "   # benchmarking single-precision" << std::endl;
  std::cout << "   -------------------------------" << std::endl;

  // -----------------------------------------
   paras.add_test();
   paras.add_data_node(viennacl::io::tag::name,    viennacl::io::val::compmat);
   paras.add_data_node(viennacl::io::tag::numeric, viennacl::io::val::fl);
   paras.add_data_node(viennacl::io::tag::alignment, "1");
  // -----------------------------------------

  //set up test config:
  conf.program_name(viennacl::linalg::kernels::compressed_matrix<float, 1>::program_name());

  run_matrix_benchmark<float>(conf, paras);

  if( viennacl::ocl::current_device().double_support() )
  {
    std::cout << std::endl;
    std::cout << "   -------------------------------" << std::endl;
    std::cout << "   # benchmarking double-precision" << std::endl;
    std::cout << "   -------------------------------" << std::endl;
  // -----------------------------------------
   paras.add_test();
   paras.add_data_node(viennacl::io::tag::name,    viennacl::io::val::compmat);
   paras.add_data_node(viennacl::io::tag::numeric, viennacl::io::val::dbl);
   paras.add_data_node(viennacl::io::tag::alignment, "1");

    conf.program_name(viennacl::linalg::kernels::compressed_matrix<double, 1>::program_name());
   // -----------------------------------------
    run_matrix_benchmark<double>(conf, paras);
  }
  // -----------------------------------------
  //paras.dump(); // dump to terminal
  paras.dump("sparse_parameters.xml"); // dump to outputfile
  //std::ofstream stream; paras.dump(stream);   // dump to stream
  // -----------------------------------------

  std::cout << std::endl;
  std::cout << "/////////////////////////////////////////////////////////////////////////////////" << std::endl;
  std::cout << "// Parameter evaluation for viennacl::compressed_matrix finished successfully! //" << std::endl;
  std::cout << "/////////////////////////////////////////////////////////////////////////////////" << std::endl;
  return 0;
}

