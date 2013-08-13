//#define VIENNACL_DEBUG_BUILD
#define VIENNACL_WITH_OPENCL
//#define VIENNACL_DEBUG_ALL

#include <iostream>
#include "CL/cl.hpp"

#include "viennacl/linalg/prod.hpp"
#include "viennacl/vector.hpp"
#include "viennacl/matrix.hpp"
#include "boost/numeric/ublas/matrix.hpp"
#include "viennacl/generator/generate.hpp"
#include "viennacl/generator/autotune.hpp"

#define N_RUNS 5

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

int main(){
    platforms_type platforms = viennacl::ocl::get_platforms();
    size_t num_platforms = platforms.size();
    for(unsigned int k=0 ; k < num_platforms ; ++k)
  {
    viennacl::ocl::platform pf(k);
    viennacl::ocl::set_context_platform_index(k,k);
    viennacl::ocl::switch_context(k);
    devices_type dev = viennacl::ocl::current_context().devices();
    for(devices_type::iterator it = dev.begin() ; it != dev.end() ; ++it){
      viennacl::ocl::switch_device(*it);
            std::cout << "-------------------" << std::endl;
            std::cout << it->name()<< std::endl;
      std::cout << "-------------------" << std::endl;

            std::cout << "scalartype : float" << std::endl;
            std::cout << "-- Av " << std::endl;
            run_autotune<float>("BLAS2 AV Float "+it->name(), false);
            std::cout << "-- Tv" << std::endl;
            run_autotune<float>("BLAS2 TV Float "+it->name(), true);

            std::cout << "-----------------" << std::endl;

            std::cout << "scalartype : double" << std::endl;
            std::cout << "-- Av " << std::endl;
            run_autotune<double>("BLAS2 AV Double "+it->name(), false);
            std::cout << "-- Tv" << std::endl;
            run_autotune<double>("BLAS2 TV Double "+it->name(), true);

    }
  }


}
