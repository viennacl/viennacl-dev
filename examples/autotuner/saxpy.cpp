//~ #define VIENNACL_DEBUG_BUILD
#define VIENNACL_WITH_OPENCL
//#define VIENNACL_DEBUG_ALL

#include <iostream>
#include "CL/cl.hpp"

#include "viennacl/vector.hpp"
#include "viennacl/generator/custom_operation.hpp"
#include "viennacl/generator/dummy_types.hpp"
#include "viennacl/generator/autotune/autotune.hpp"
#include "viennacl/linalg/norm_2.hpp"

#define N_RUNS 5

typedef float ScalarType;
typedef std::vector< viennacl::ocl::platform > platforms_type;
typedef std::vector<viennacl::ocl::device> devices_type;
typedef std::vector<cl_device_id> cl_devices_type;

static const unsigned int size = 1024*1024;


class config{
public:
    typedef viennacl::generator::code_generation::saxpy::profile profile_t;

    config(std::pair<unsigned int, unsigned int> minmax_a
             ,std::pair<unsigned int, unsigned int> minmax_u
           ,std::pair<unsigned int, unsigned int> minmax_ls): minmax_a_(minmax_a), minmax_u_(minmax_u), minmax_ls_(minmax_ls){
        current_a_ = minmax_a_.first;
        current_u_ = minmax_u_.first;
        current_ls_ = minmax_ls_.first;

        has_next_ = true;
    }


    bool has_next() const{
        return current_a_<minmax_a_.second ||
                current_u_<minmax_u_.second ||
                current_ls_<minmax_ls_.second;
    }

    void update(){
        current_a_*=2;
        if(current_a_>minmax_a_.second){
            current_a_=minmax_a_.first;
            current_u_*=2;
            if(current_u_>minmax_u_.second){
                current_u_=minmax_u_.first;
                current_ls_*=2;
                if(current_ls_>minmax_ls_.second){
                    current_ls_=minmax_ls_.first;
                }
            }
        }
    }

    profile_t get_current(){
        return profile_t(current_a_,current_u_,current_ls_);
    }

private:
    unsigned int current_u_;
    unsigned int current_a_;
    unsigned int current_ls_;

    std::pair<unsigned int, unsigned int> minmax_a_;
    std::pair<unsigned int, unsigned int> minmax_u_;
    std::pair<unsigned int, unsigned int> minmax_ls_;

    bool has_next_;
};

void autotune(){
    std::map<double, config::profile_t> timings;

    std::vector<ScalarType> cpu_v1(size), cpu_v2(size), cpu_v3(size), cpu_v4(size);
    for(unsigned int i=0; i<size; ++i){
        cpu_v1[i]=i;
        cpu_v2[i]=2*i;
        cpu_v3[i]=3*i;
        cpu_v4[i]=4*i;
    }

    viennacl::vector<ScalarType> v1(size), v2(size), v3(size), v4(size);
    viennacl::copy(cpu_v1,v1);
    viennacl::copy(cpu_v2,v2);
    viennacl::copy(cpu_v3,v3);
    viennacl::copy(cpu_v4,v4);


    typedef viennacl::generator::dummy_vector<ScalarType> dv;
    config conf(std::make_pair(1,4)
                ,std::make_pair(1,16)
                ,std::make_pair(64,viennacl::ocl::info<CL_DEVICE_MAX_WORK_GROUP_SIZE>(viennacl::ocl::current_device().id())));
    viennacl::generator::autotune::benchmark(timings,dv(v1) = dv(v2) + dv(v3),conf);
    std::cout << std::endl;
    std::cout << "Best Profile: " << timings.begin()->first << " <=> " << timings.begin()->second << std::endl;
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
            autotune();
		}
	}
	
	
}
