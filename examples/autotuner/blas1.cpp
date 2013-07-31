#define VIENNACL_WITH_OPENCL
//#define VIENNACL_DEBUG_ALL
//#define VIENNACL_DEBUG_BUILD

#include <iostream>
#include "CL/cl.hpp"

#include "viennacl/linalg/inner_prod.hpp"

#include "viennacl/scheduler/forwards.h"

#include "viennacl/vector.hpp"
#include "viennacl/generator/generate.hpp"
#include "viennacl/generator/autotune.hpp"
#include "viennacl/linalg/norm_2.hpp"

#define N_RUNS 5

typedef std::vector< viennacl::ocl::platform > platforms_type;
typedef std::vector<viennacl::ocl::device> devices_type;
typedef std::vector<cl_device_id> cl_devices_type;

static const unsigned int size = 1024*1024;


template<class ScalarType>
struct dot_config{
    typedef viennacl::generator::scalar_reduction::profile profile_t;
    static profile_t create_profile(std::map<std::string, viennacl::generator::autotune::tuning_param> const & params){
      return profile_t(params.at("vectorization").current(),params.at("group_size").current(),params.at("num_groups").current(), params.at("global_decomposition").current());
    }
    static bool is_invalid(viennacl::ocl::device const & dev, std::map<std::string, viennacl::generator::autotune::tuning_param> const & params){
        profile_t prof = create_profile(params);
        return prof.is_invalid(dev, sizeof(ScalarType));
    }
};


template<class ScalarType>
void autotune(){
    std::vector<ScalarType> cpu_v1(size), cpu_v2(size), cpu_v3(size), cpu_v4(size);
    for(unsigned int i=0; i<size; ++i){
        cpu_v1[i]=0;
        cpu_v2[i]=rand()/(ScalarType)RAND_MAX;
        cpu_v3[i]=rand()/(ScalarType)RAND_MAX;
        cpu_v4[i]=rand()/(ScalarType)RAND_MAX;
    }

    viennacl::vector<ScalarType> v1(size), v2(size), v3(size), v4(size);
    viennacl::copy(cpu_v1,v1);
    viennacl::copy(cpu_v2,v2);
    viennacl::copy(cpu_v3,v3);
    viennacl::copy(cpu_v4,v3);
    viennacl::backend::finish();

    viennacl::scalar<ScalarType> s = 0;

    std::map<double, typename dot_config<ScalarType>::profile_t> timings;
    std::cout << "* Tuning DOT" << std::endl;
    viennacl::generator::autotune::tuning_config<dot_config<ScalarType> > conf;
    std::vector<int> vectorizations;
    std::vector<int> group_sizes;
    std::vector<int> num_groups;
    std::vector<int> global_decompositions;
    for(unsigned int a = 1; a <= 8 ; a*=2)
      vectorizations.push_back(a);
    for(unsigned int g = 16 ; g <= 1024 ; g *= 2)
      num_groups.push_back(g);
    for(unsigned int i = 16; i <= viennacl::ocl::current_device().max_work_group_size() ; i*=2)
      group_sizes.push_back(i);
    global_decompositions.push_back(0); global_decompositions.push_back(1);
    conf.add_tuning_param("vectorization",vectorizations);
    conf.add_tuning_param("group_size",group_sizes);
    conf.add_tuning_param("num_groups",num_groups);
    conf.add_tuning_param("global_decomposition", global_decompositions);
    viennacl::generator::autotune::benchmark(timings,viennacl::scheduler::statement(s, viennacl::op_assign(), viennacl::linalg::inner_prod(v1, v2)),conf,sizeof(ScalarType));
    std::cout << std::endl;
    std::cout << "Best Profile: " << timings.begin()->first << "s" << std::endl;
    std::cout << "Vectorization : " << timings.begin()->second.vectorization() << std::endl;
    std::cout << "Num Groups : " << timings.begin()->second.num_groups() << std::endl;
    std::cout << "Group Size : " << timings.begin()->second.group_size() << std::endl;
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
            std::cout << "float:" << std::endl;
            autotune<float>();
            std::cout << "-------------------" << std::endl;
            std::cout << "double:" << std::endl;
            autotune<double>();
    }
  }


}
