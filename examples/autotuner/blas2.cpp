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
    typedef viennacl::generator::code_generation::gemv::profile profile_t;
    static profile_t create_profile(std::map<std::string, viennacl::generator::autotune::tuning_param> const & params){
        return profile_t(params.at("local_size1").current(),params.at("local_size2").current(),params.at("num_groups").current());
    }
    static bool is_invalid(viennacl::ocl::device const & dev, std::map<std::string, viennacl::generator::autotune::tuning_param> const & params){
        profile_t prof = create_profile(params);
        return prof.is_invalid(dev, sizeof(ScalarType));
    }
};



template<class ScalarType, class Layout, class BoostLayout>
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


    size_t max_size = viennacl::ocl::info<CL_DEVICE_MAX_WORK_GROUP_SIZE>(viennacl::ocl::current_device().id());
    std::map<double,typename blas2_config<ScalarType>::profile_t> timings;
    viennacl::generator::autotune::tuning_config< blas2_config<ScalarType> > conf;
    //conf.add_tuning_param("alignment",1,1);
    conf.add_tuning_param("local_size1",1,max_size,&viennacl::generator::autotune::inc::mul_by_two);
    conf.add_tuning_param("local_size2",1,max_size,&viennacl::generator::autotune::inc::mul_by_two);
    conf.add_tuning_param("num_groups",4,1024,&viennacl::generator::autotune::inc::mul_by_two);
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

            std::cout << "scalartype : float" << std::endl;
            std::cout << "-- Layout : row-major" << std::endl;
            autotune<float, viennacl::row_major, boost::numeric::ublas::row_major>();
            std::cout << "-- Layout : column-major" << std::endl;
            autotune<float, viennacl::column_major, boost::numeric::ublas::column_major>();

            std::cout << "scalartype : double" << std::endl;
            std::cout << "-- Layout : row-major" << std::endl;
            autotune<double, viennacl::row_major, boost::numeric::ublas::row_major>();
            std::cout << "-- Layout : column-major" << std::endl;
            autotune<double, viennacl::column_major, boost::numeric::ublas::column_major>();
    }
  }


}
