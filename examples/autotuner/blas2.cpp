//#define VIENNACL_DEBUG_BUILD
#define VIENNACL_WITH_OPENCL
//#define VIENNACL_DEBUG_ALL

#include <iostream>
#include "CL/cl.hpp"

#include "viennacl/linalg/prod.hpp"
#include "viennacl/vector.hpp"
#include "viennacl/matrix.hpp"
#include "boost/numeric/ublas/matrix.hpp"
#include "viennacl/generator/custom_operation.hpp"
#include "viennacl/generator/autotune/autotune.hpp"

#define N_RUNS 5

typedef float ScalarType;
typedef std::vector< viennacl::ocl::platform > platforms_type;
typedef std::vector<viennacl::ocl::device> devices_type;
typedef std::vector<cl_device_id> cl_devices_type;

static const unsigned int size = 4096;


class blas2_config{
public:
    typedef viennacl::generator::code_generation::gemv::profile profile_t;

    blas2_config(std::pair<unsigned int, unsigned int> minmax_a
             ,std::pair<unsigned int, unsigned int> minmax_k
           ,std::pair<unsigned int, unsigned int> minmax_m
           ,std::pair<unsigned int, unsigned int> minmax_numgroups): minmax_a_(minmax_a)
                                                                    , minmax_k_(minmax_k)
                                                                    , minmax_m_(minmax_m)
                                                                  ,minmax_numgroups_(minmax_numgroups){
        current_a_ = minmax_a_.first;
        current_k_ = minmax_k_.first;
        current_m_ = minmax_m_.first;
        current_numgroups_ = minmax_numgroups_.first;

        has_next_ = true;
    }


    bool has_next() const{
        return current_a_<minmax_a_.second ||
                current_k_<minmax_k_.second ||
                current_m_<minmax_m_.second ||
                current_numgroups_<minmax_numgroups_.second;
    }

    void update(){
        current_a_*=2;
        if(current_a_>minmax_a_.second){
            current_a_=minmax_a_.first;
            current_k_*=2;
            if(current_k_>minmax_k_.second){
                current_k_=minmax_k_.first;
                current_m_*=2;
                if(current_m_>minmax_m_.second){
                    current_m_=minmax_m_.first;
                    current_numgroups_*=2;
                    if(current_numgroups_>minmax_numgroups_.second){
                        current_numgroups_=minmax_numgroups_.first;
                    }
                }
            }
        }
    }

    size_t local_memory_used(){ return current_m_*(current_k_+1)*sizeof(ScalarType); }
    profile_t get_current(){ return profile_t(current_m_, current_k_, current_numgroups_); }

private:
    unsigned int current_a_;
    unsigned int current_k_;
    unsigned int current_m_;
    unsigned int current_numgroups_;

    std::pair<unsigned int, unsigned int> minmax_a_;
    std::pair<unsigned int, unsigned int> minmax_k_;
    std::pair<unsigned int, unsigned int> minmax_m_;
    std::pair<unsigned int, unsigned int> minmax_numgroups_;

    bool has_next_;
};

template<class Layout, class BoostLayout>
void autotune(){
    typedef viennacl::generator::vector<ScalarType> vec;
    typedef viennacl::generator::matrix<viennacl::matrix<ScalarType, Layout> > mat;

    std::vector<ScalarType> cpu_v1(size), cpu_v3(size);
    boost::numeric::ublas::matrix<ScalarType, BoostLayout> cpu_m2(size,size);
    for(unsigned int i=0; i<size; ++i){
        cpu_v1[i]=0;
        cpu_v3[i]=rand()/(ScalarType)RAND_MAX;
        for(unsigned int j=0; j<size; ++j){
            cpu_m2(i,j)=rand()/(ScalarType)RAND_MAX;
        }
    }

    viennacl::vector<ScalarType> v1(size), v3(size);
    viennacl::matrix<ScalarType, Layout> m2(size,size);
    viennacl::copy(cpu_v1,v1);
    viennacl::copy(cpu_m2,m2);
    viennacl::copy(cpu_v3,v3);
    viennacl::backend::finish();


    std::map<double, blas2_config::profile_t> timings;
    blas2_config conf(std::make_pair(1,1)
                      ,std::make_pair(2,128)
                      ,std::make_pair(2,128)
                      ,std::make_pair(32,viennacl::ocl::info<CL_DEVICE_MAX_WORK_GROUP_SIZE>(viennacl::ocl::current_device().id())));
    viennacl::generator::autotune::benchmark(timings,vec(v1) = prod(mat(m2),vec(v3)),conf);
    std::cout << std::endl;
    std::cout << "Best Profile: " << timings.begin()->second << " => " << timings.begin()->first << std::endl;
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
            autotune<viennacl::column_major, boost::numeric::ublas::column_major>();
		}
	}
	
	
}
