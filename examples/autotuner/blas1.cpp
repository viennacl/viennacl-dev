//#define VIENNACL_DEBUG_BUILD
#define VIENNACL_WITH_OPENCL
//#define VIENNACL_DEBUG_ALL

#include <iostream>
#include "CL/cl.hpp"

#include "viennacl/linalg/inner_prod.hpp"

#include "viennacl/vector.hpp"
#include "viennacl/generator/custom_operation.hpp"
#include "viennacl/generator/autotune.hpp"
#include "viennacl/linalg/norm_2.hpp"

#define N_RUNS 5

typedef std::vector< viennacl::ocl::platform > platforms_type;
typedef std::vector<viennacl::ocl::device> devices_type;
typedef std::vector<cl_device_id> cl_devices_type;

static const unsigned int size = 1024*1024;


template<class ScalarType>
struct dot_config{
    typedef viennacl::generator::code_generation::scalar_reduction_profile profile_t;
    static profile_t create_profile(std::map<std::string, viennacl::generator::autotune::tuning_param> const & params){
        return profile_t(params.at("alignment").current(),params.at("group_size").current(),params.at("num_groups").current());
    }
    static bool is_invalid(viennacl::ocl::device const & dev, std::map<std::string, viennacl::generator::autotune::tuning_param> const & params){
        profile_t prof = create_profile(params);
        return prof.is_invalid(dev, sizeof(ScalarType));
    }
};


template<class ScalarType>
void autotune(){
    typedef viennacl::generator::vector<ScalarType> vec;
    typedef viennacl::generator::scalar<ScalarType> scal;

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
    std::vector<int> alignments;
    std::vector<int> group_sizes;
    std::vector<int> num_groups;
    for(unsigned int a = 1; a <= 8 ; a*=2)
      alignments.push_back(a);
    for(unsigned int i = 16; i <= viennacl::ocl::info<CL_DEVICE_MAX_WORK_GROUP_SIZE>(viennacl::ocl::current_device().id()) ; i*=2)
      group_sizes.push_back(i);
    for(unsigned int g = 16 ; g <= 1024 ; g *= 2)
      num_groups.push_back(g);
    conf.add_tuning_param("alignment",alignments);
    conf.add_tuning_param("group_size",group_sizes);
    conf.add_tuning_param("num_groups",num_groups);
    viennacl::generator::autotune::benchmark(timings,scal(s) = viennacl::generator::inner_prod(vec(v1), vec(v2)),std::make_pair(viennacl::generator::code_generation::dot,sizeof(ScalarType)),conf);
    std::cout << std::endl;
    std::cout << "Best Profile: " << timings.begin()->first << "s" << std::endl;
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
