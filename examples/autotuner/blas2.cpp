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
#include "viennacl/generator/autotune.hpp"

#define N_RUNS 5

typedef std::vector< viennacl::ocl::platform > platforms_type;
typedef std::vector<viennacl::ocl::device> devices_type;
typedef std::vector<cl_device_id> cl_devices_type;

static const unsigned int size = 2048;


template<class ScalarType>
struct blas2_config{
    typedef viennacl::generator::code_generation::vector_reduction_profile profile_t;
    static profile_t create_profile(std::map<std::string, viennacl::generator::autotune::tuning_param> const & params){
        return profile_t(params.at("local_size1").current(),params.at("local_size2").current(),params.at("num_groups").current());
    }
    static bool is_invalid(viennacl::ocl::device const & dev, std::map<std::string, viennacl::generator::autotune::tuning_param> const & params){
        profile_t prof = create_profile(params);
        return prof.is_invalid(dev, sizeof(ScalarType));
    }
};



template<class ScalarType>
void autotune(bool trans){
    typedef viennacl::generator::vector<ScalarType> vec;
    typedef viennacl::generator::matrix<viennacl::matrix<ScalarType, viennacl::row_major> > mat;
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
    std::map<double,typename blas2_config<ScalarType>::profile_t> timings;
    viennacl::generator::autotune::tuning_config< blas2_config<ScalarType> > conf;

    std::vector<int> alignments;
    std::vector<int> local_sizes1;
    std::vector<int> local_sizes2;
    std::vector<int> num_groups;

    alignments.push_back(1);
    for(unsigned int s = 1 ; s <= max_size ; s*=2)
      local_sizes1.push_back(s);
    for(unsigned int s = 1 ; s <= max_size ; s*=2)
      local_sizes2.push_back(s);
    for(unsigned int g = 16 ; g <= 1024 ; g+=16)
      num_groups.push_back(g);

    conf.add_tuning_param("alignment",alignments);
    conf.add_tuning_param("local_size1",local_sizes1);
    conf.add_tuning_param("local_size2",local_sizes2);
    conf.add_tuning_param("num_groups",num_groups);

    if(trans)
      viennacl::generator::autotune::benchmark(timings,vec(v1) = prod(viennacl::generator::trans(mat(m2)),vec(v3)),std::make_pair(viennacl::generator::code_generation::gemvTv, sizeof(ScalarType)),conf);
    else
      viennacl::generator::autotune::benchmark(timings,vec(v1) = prod(mat(m2),vec(v3)),std::make_pair(viennacl::generator::code_generation::gemvAv, sizeof(ScalarType)),conf);
    std::cout << std::endl;
    std::cout << "Best Profile: " << std::setprecision(5) <<  timings.begin()->first << "s" << "\t|\t" << 1e-9*size*(2*size-1)/timings.begin()->first << " GFLOPs" << std::endl;
    std::cout << "M : " << timings.begin()->second.m() << std::endl;
    std::cout << "K : " << timings.begin()->second.k() << std::endl;
    std::cout << "Num Groups 0 : " << timings.begin()->second.num_groups_0() << std::endl;
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
            autotune<float>(false);
            std::cout << "-- Tv" << std::endl;
            autotune<float>(true);

            std::cout << "-----------------" << std::endl;

            std::cout << "scalartype : double" << std::endl;
            std::cout << "-- Av" << std::endl;
            autotune<double>(false);
            std::cout << "-- Tv" << std::endl;
            autotune<double>(true);
    }
  }


}
