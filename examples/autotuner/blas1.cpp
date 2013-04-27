//#define VIENNACL_DEBUG_BUILD
#define VIENNACL_WITH_OPENCL
//#define VIENNACL_DEBUG_ALL

#include <iostream>
#include "CL/cl.hpp"

#include "viennacl/linalg/inner_prod.hpp"

#include "viennacl/vector.hpp"
#include "viennacl/generator/custom_operation.hpp"
#include "viennacl/generator/dummy_types.hpp"
#include "viennacl/generator/autotune/autotune.hpp"
#include "viennacl/linalg/norm_2.hpp"

#define N_RUNS 5

typedef double ScalarType;
typedef std::vector< viennacl::ocl::platform > platforms_type;
typedef std::vector<viennacl::ocl::device> devices_type;
typedef std::vector<cl_device_id> cl_devices_type;

static const unsigned int size = 1024*1024;


class saxpy_config{
public:
    typedef viennacl::generator::code_generation::saxpy::profile profile_t;

    saxpy_config(std::pair<unsigned int, unsigned int> minmax_a
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

class inprod_config{
public:
    typedef viennacl::generator::code_generation::inner_product::profile profile_t;

    inprod_config(std::pair<unsigned int, unsigned int> minmax_a
             ,std::pair<unsigned int, unsigned int> minmax_groupsize
           ,std::pair<unsigned int, unsigned int> minmax_numgroups): minmax_a_(minmax_a), minmax_groupsize_(minmax_groupsize), minmax_numgroups_(minmax_numgroups){
        current_a_ = minmax_a_.first;
        current_groupsize_ = minmax_groupsize_.first;
        current_numgroups_ = minmax_numgroups_.first;
    }

    bool has_next() const{
        return current_a_<minmax_a_.second ||
                current_numgroups_<minmax_numgroups_.second ||
                current_groupsize_<minmax_groupsize_.second;
    }

    void update(){
        current_a_*=2;
        if(current_a_>minmax_a_.second){
            current_a_=minmax_a_.first;
            current_numgroups_*=2;
            if(current_numgroups_>minmax_numgroups_.second){
                current_numgroups_=minmax_numgroups_.first;
                current_groupsize_*=2;
                if(current_groupsize_>minmax_groupsize_.second){
                    current_groupsize_=minmax_groupsize_.first;
                }
            }
        }
    }

    profile_t get_current(){
        return profile_t(current_a_,current_groupsize_,current_numgroups_);
    }

private:
    unsigned int current_a_;
    unsigned int current_groupsize_;
    unsigned int current_numgroups_;

    std::pair<unsigned int, unsigned int> minmax_a_;
    std::pair<unsigned int, unsigned int> minmax_groupsize_;
    std::pair<unsigned int, unsigned int> minmax_numgroups_;
};

void autotune(){
    typedef viennacl::generator::vector<ScalarType> vec;
    typedef viennacl::generator::scalar<ScalarType> scal;

    std::vector<ScalarType> cpu_v1(size), cpu_v2(size), cpu_v3(size);
    for(unsigned int i=0; i<size; ++i){
        cpu_v1[i]=0;
        cpu_v2[i]=rand()/(ScalarType)RAND_MAX;
        cpu_v3[i]=rand()/(ScalarType)RAND_MAX;
    }

    viennacl::vector<ScalarType> v1(size), v2(size), v3(size);
    viennacl::copy(cpu_v1,v1);
    viennacl::copy(cpu_v2,v2);
    viennacl::copy(cpu_v3,v3);
    viennacl::backend::finish();

    viennacl::scalar<ScalarType> s = 0;



    {
        std::map<double, saxpy_config::profile_t> timings;
        std::cout << "* Tuning SAXPY" << std::endl;
        saxpy_config conf(std::make_pair(1,8) ,std::make_pair(1,16),std::make_pair(32,viennacl::ocl::info<CL_DEVICE_MAX_WORK_GROUP_SIZE>(viennacl::ocl::current_device().id())));
        viennacl::generator::autotune::benchmark(timings,vec(v1) = vec(v2) + vec(v3),conf);
        std::cout << std::endl;
        std::cout << "Best Profile: " << timings.begin()->second << " => " << timings.begin()->first << std::endl;
    }

    {
        std::map<double, inprod_config::profile_t> timings;
        std::cout << "* Tuning Inner Product" << std::endl;
        inprod_config conf(std::make_pair(1,8)
                    ,std::make_pair(32,viennacl::ocl::info<CL_DEVICE_MAX_WORK_GROUP_SIZE>(viennacl::ocl::current_device().id()))
                    ,std::make_pair(32,viennacl::ocl::info<CL_DEVICE_MAX_WORK_GROUP_SIZE>(viennacl::ocl::current_device().id())));
        viennacl::generator::autotune::benchmark(timings,scal(s) = viennacl::generator::inner_prod(vec(v1), vec(v2)),conf);
        std::cout << std::endl;
        std::cout << "Best Profile: " << timings.begin()->second << " => " << timings.begin()->first << std::endl;


        viennacl::generator::autotune::Timer t;
        s=viennacl::linalg::inner_prod(v1,v2);
        viennacl::backend::finish();

        t.start();
        s= viennacl::linalg::inner_prod(v1,v2);
        viennacl::backend::finish();
        double vcl_time = t.get();

        std::cout << "ViennaCL : " << vcl_time << std::endl;

    }
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
