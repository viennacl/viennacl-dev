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

using namespace viennacl::generator;

typedef std::vector< viennacl::ocl::platform > platforms_type;
typedef std::vector<viennacl::ocl::device> devices_type;
typedef std::vector<cl_device_id> cl_devices_type;

static const unsigned int size = 1024*1024;


template<class ScalarType>
struct dot_config{
    typedef scalar_reduction profile_type;
    static profile_type create_profile(std::map<std::string, autotune::tuning_param> const & params){
      return profile_type(params.at("vectorization").current(),params.at("group_size").current(),params.at("num_groups").current(), params.at("global_decomposition").current());
    }
    static bool is_invalid(viennacl::ocl::device const & dev, std::map<std::string, autotune::tuning_param> const & params){
        profile_type prof = create_profile(params);
        return prof.is_invalid(dev, sizeof(ScalarType));
    }
    static std::string state_representation_format(){
        return "V" "\t" "GS" "\t" "NG" "\t" "GD";
    }
    static std::string current_state_representation(profile_type const profile){
        std::ostringstream oss;
        oss << profile.vectorization() << "\t" << profile.group_size() << "\t" << profile.num_groups() << "\t" << profile.global_decomposition();
        return oss.str();
    }
};


template<class ScalarType>
void run_autotune(std::string const & dump_name){
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

    std::map<double, typename dot_config<ScalarType>::profile_type> timings;
    std::cout << "* Tuning DOT" << std::endl;
    autotune::tuning_config<dot_config<ScalarType> > conf;
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
    std::ofstream stream(dump_name.c_str());
    std::size_t scalartype_size = sizeof(ScalarType);
    code_generator::forced_profile_key_type key(SCALAR_REDUCE_TYPE, scalartype_size);
    autotune::benchmark(&timings,viennacl::scheduler::statement(s, viennacl::op_assign(), viennacl::linalg::inner_prod(v1, v2)),key,conf,&stream);
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
            std::cout << "float:" << std::endl;
            run_autotune<float>("BLAS1 Float "+it->name());
            std::cout << "-------------------" << std::endl;
            std::cout << "double:" << std::endl;
            run_autotune<double>("BLAS1 Double_"+it->name());
    }
  }


}
