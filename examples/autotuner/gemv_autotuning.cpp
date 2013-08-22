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

//#define VIENNACL_DEBUG_BUILD
//#define VIENNACL_WITH_OPENCL
//#define VIENNACL_DEBUG_ALL

#include <iostream>

#include "viennacl/linalg/prod.hpp"
#include "viennacl/vector.hpp"
#include "viennacl/matrix.hpp"
#include "boost/numeric/ublas/matrix.hpp"
#include "viennacl/generator/generate.hpp"
#include "viennacl/generator/autotune.hpp"

//#define N_RUNS 5

using namespace viennacl::generator;

typedef std::vector< viennacl::ocl::platform > platforms_type;
typedef std::vector<viennacl::ocl::device> devices_type;
typedef std::vector<cl_device_id> cl_devices_type;

static const unsigned int size = 2048;


template<class ScalarType>
struct blas2_config{
    typedef vector_reduction profile_type;
    static profile_type create_profile(std::map<std::string, autotune::tuning_param> const & params){
      return profile_type(params.at("vectorization").current(), params.at("local_size1").current(),params.at("local_size2").current(),params.at("num_groups").current());
    }
    static bool is_invalid(viennacl::ocl::device const & dev, std::map<std::string, autotune::tuning_param> const & params){
        profile_type prof = create_profile(params);
        return prof.is_invalid(dev, sizeof(ScalarType));
    }
};



template<class ScalarType>
void run_autotune(std::string const & dump_name, bool trans){
    std::vector<ScalarType> cpu_v1(size), cpu_v3(size);
    boost::numeric::ublas::matrix<ScalarType, boost::numeric::ublas::row_major> cpu_m2(size,size);
    for(unsigned int i=0; i<size; ++i){
        cpu_v1[i]=0;
        cpu_v3[i]=rand()/(ScalarType)RAND_MAX;
        for(unsigned int j=0; j<size; ++j){
            cpu_m2(i,j)=rand()/(ScalarType)RAND_MAX;
        }
    }

    viennacl::vector<ScalarType> v1(size), v3(size);
    viennacl::matrix<ScalarType, viennacl::row_major> m2(size,size);
    viennacl::copy(cpu_v1,v1);
    viennacl::copy(cpu_m2,m2);
    viennacl::copy(cpu_v3,v3);
    viennacl::backend::finish();


    size_t max_size = viennacl::ocl::info<CL_DEVICE_MAX_WORK_GROUP_SIZE>(viennacl::ocl::current_device().id());
    std::map<double,typename blas2_config<ScalarType>::profile_type> timings;
    autotune::tuning_config< blas2_config<ScalarType> > conf;

    std::vector<int> vectorization;
    std::vector<int> local_sizes1;
    std::vector<int> local_sizes2;
    std::vector<int> num_groups;

    vectorization.push_back(1);
    for(unsigned int s = 1 ; s <= max_size ; s*=2)
      local_sizes1.push_back(s);
    for(unsigned int s = 1 ; s <= max_size ; s*=2)
      local_sizes2.push_back(s);
    for(unsigned int g = 16 ; g <= 1024 ; g+=16)
      num_groups.push_back(g);

    conf.add_tuning_param("vectorization",vectorization);
    conf.add_tuning_param("local_size1",local_sizes1);
    conf.add_tuning_param("local_size2",local_sizes2);
    conf.add_tuning_param("num_groups",num_groups);
    std::ofstream stream(dump_name.c_str());
    std::size_t scalartype_size = sizeof(ScalarType);
    code_generator::forced_profile_key_type keyTx(VECTOR_REDUCE_Tx_TYPE, scalartype_size);
    code_generator::forced_profile_key_type keyAx(VECTOR_REDUCE_Ax_TYPE, scalartype_size);
    if(trans)
      autotune::benchmark(&timings,viennacl::scheduler::statement(v1,viennacl::op_assign(), viennacl::linalg::prod(viennacl::trans(m2), v3)),keyTx,conf,&stream);
    else
      autotune::benchmark(&timings,viennacl::scheduler::statement(v1,viennacl::op_assign(), viennacl::linalg::prod(m2, v3)),keyAx,conf,&stream);
    std::cout << std::endl;
    std::cout << " ============" << std::endl;
    std::cout << " Best Profile : " << std::scientific << timings.begin()->first << " => " << timings.begin()->second << std::endl;
    std::cout << " ============" << std::endl;
    std::cout << std::endl;
}


int main(int argc, char* argv[]){
  typedef std::vector< viennacl::ocl::platform > platforms_type;
  std::vector<std::string> args(argv, argv+argc);
  if(argc<2){
    std::cerr << "USAGE : PROGRAM_NAME DEVICE" << std::endl;
    exit(EXIT_FAILURE);
  }
  unsigned int requested_device = atoi(args[1].c_str());
  std::size_t current_device = 0;
  platforms_type platforms = viennacl::ocl::get_platforms();
  for (platforms_type::iterator platform_iter  = platforms.begin();
       platform_iter != platforms.end();
       ++platform_iter)
  {
    typedef std::vector<viennacl::ocl::device> devices_type;
    devices_type devices = platform_iter->devices(CL_DEVICE_TYPE_ALL);
    for(devices_type::iterator iter = devices.begin(); iter != devices.end(); iter++)
    {
      if(current_device++==requested_device){
        viennacl::ocl::setup_context(current_device,*iter);
        viennacl::ocl::switch_context(current_device);
        viennacl::ocl::device const & device = viennacl::ocl::current_device();
        std::string device_name = device.name();
        std::transform(device_name.begin(), device_name.end(), device_name.begin(), ::tolower);
        std::replace(device_name.begin(), device_name.end(),' ', '_');
        std::cout << "-------------------" << std::endl;
        std::cout << device.info()<< std::endl;
        std::cout << "GEMV" << std::endl;
        std::cout << "-------------------" << std::endl;
        std::cout << "scalartype : float" << std::endl;
        std::cout << "-- Av " << std::endl;
        run_autotune<float>("gemv_av_float_"+device_name + ".dat", false);
        std::cout << "-- Tv" << std::endl;
        run_autotune<float>("gemv_tv_float_"+device_name + ".dat", true);

        std::cout << "-----------------" << std::endl;

        std::cout << "scalartype : double" << std::endl;
        std::cout << "-- Av " << std::endl;
        run_autotune<double>("gemv_av_double_"+device_name + ".dat", false);
        std::cout << "-- Tv" << std::endl;
        run_autotune<double>("gemv_tv_double_"+device_name + ".dat", true);
      }
    }
  }
}
