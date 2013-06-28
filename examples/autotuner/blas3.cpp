#define VIENNACL_WITH_OPENCL

//#define VIENNACL_DEBUG_BUILD
//#define VIENNACL_DEBUG_ALL

#define NDEBUG

#include <iostream>
#include "CL/cl.hpp"
#include <sys/time.h>

#include "boost/numeric/ublas/matrix.hpp"

#include "viennacl/matrix.hpp"
#include "viennacl/generator/custom_operation.hpp"
#include "viennacl/generator/autotune.hpp"
#include "viennacl/linalg/prod.hpp"
#include "viennacl/linalg/norm_2.hpp"

#include "../tutorial/Random.hpp"


typedef std::vector< viennacl::ocl::platform > platforms_type;
typedef std::vector<viennacl::ocl::device> devices_type;
typedef std::vector<cl_device_id> cl_devices_type;

template<class ScalarType>
struct blas3_config{
    typedef viennacl::generator::code_generation::matrix_product_profile profile_t;
    static profile_t create_profile(std::map<std::string, viennacl::generator::autotune::tuning_param> const & params){
        return profile_t(params.at("ml").current(), params.at("kl").current(), params.at("nl").current(),
                         params.at("ms").current(), params.at("ks").current(), params.at("ns").current(),
                         static_cast<bool>(params.at("lhs_storage").current()),static_cast<bool>(params.at("rhs_storage").current()),
                         params.at("vector").current(),
                         params.at("unroll").current());
    }
    static bool is_invalid(viennacl::ocl::device const & dev, std::map<std::string, viennacl::generator::autotune::tuning_param> const & params){
        profile_t prof = create_profile(params);
        return prof.is_invalid(dev, sizeof(ScalarType));
    }
};


template<class NumericT, class MatTypeA, class MatTypeB, class MatTypeC>
void fill_matrix(MatTypeA & A, MatTypeB & B, MatTypeC & C){
    typedef NumericT ScalarTypeA;
    typedef NumericT ScalarTypeB;
    typedef NumericT ScalarTypeC;

    boost::numeric::ublas::matrix<ScalarTypeA> cpu_A(A.size1(),A.size2());
    boost::numeric::ublas::matrix<ScalarTypeB> cpu_B(B.size1(),B.size2());
    boost::numeric::ublas::matrix<ScalarTypeC> cpu_C(C.size1(),C.size1());

    srand(time(NULL));
    for(unsigned int i=0; i<A.size1(); ++i){
        for(unsigned int j=0 ; j<A.size2() ; ++j){
            cpu_A(i,j)=0;
            cpu_B(i,j) =static_cast<ScalarTypeB>(rand())/static_cast<ScalarTypeB>(RAND_MAX);
            cpu_C(i,j) =static_cast<ScalarTypeB>(rand())/static_cast<ScalarTypeB>(RAND_MAX);
        }
    }

    viennacl::copy(cpu_A,A);
    viennacl::copy(cpu_B,B);
    viennacl::copy(cpu_C,C);
    viennacl::backend::finish();
}


template<class NumericT, class OpT, class ConfigT, class MatType>

void benchmark(OpT const & operation, viennacl::generator::code_generation::profile_id const & id, ConfigT conf, MatType & A, MatType & B, MatType & C,
                    std::list<viennacl::generator::code_generation::matrix_product_profile> & fastest_firsts){
    typedef std::map<double, viennacl::generator::code_generation::matrix_product_profile> timings_t;
    timings_t timings;
    unsigned int size;

    std::list<std::pair<unsigned int, unsigned int> > rounds_config;
    rounds_config.push_back(std::make_pair(512,70));
    rounds_config.push_back(std::make_pair(4096,20));
    for(std::list<std::pair<unsigned int, unsigned int> >::iterator it = rounds_config.begin() ; it!= rounds_config.end(); ++it){
        unsigned int k = std::distance(rounds_config.begin(),it);
        timings.clear();
        size=it->first;
        unsigned int n_keep=it->second;
        A.resize(size,size,false);
        B.resize(size,size,false);
        C.resize(size,size,false);
        viennacl::backend::finish();
        fill_matrix<NumericT>(A,B,C);
        viennacl::backend::finish();
        if(k==0)
            viennacl::generator::autotune::benchmark(timings,operation,id,conf);
        else{
            viennacl::generator::autotune::benchmark(timings,operation,id,fastest_firsts);
        }
        fastest_firsts.clear();
        viennacl::backend::finish();
        for(timings_t::iterator itt = timings.begin(); itt!=timings.end() ; ++itt){
            unsigned int n = std::distance(timings.begin(),itt);
            if(n>n_keep) break;
            fastest_firsts.push_back(itt->second);
            if(std::distance(rounds_config.begin(),it)==(int)rounds_config.size()-1){
//                std::cout << std::distance(timings.begin(),itt) << "th Best : " << itt->first << "s | " << 2*std::pow((double)size/1000,3)/itt->first << " GFlops : " << itt->second << std::endl;
            }
        }
    }
}



template<class NumericT>
void run_autotune(bool is_lhs_trans, bool is_rhs_trans){

    viennacl::generator::autotune::tuning_config<blas3_config<NumericT> > conf;

    conf.add_tuning_param("ml",16,256,&viennacl::generator::autotune::inc::mul_by_two);
    conf.add_tuning_param("kl",16,256,&viennacl::generator::autotune::inc::mul_by_two);
    conf.add_tuning_param("nl",16,256,&viennacl::generator::autotune::inc::mul_by_two);
    conf.add_tuning_param("ms",2,16,&viennacl::generator::autotune::inc::mul_by_two);
    conf.add_tuning_param("ks",2,16,&viennacl::generator::autotune::inc::mul_by_two);
    conf.add_tuning_param("ns",2,16,&viennacl::generator::autotune::inc::mul_by_two);
    conf.add_tuning_param("vector",1,4,&viennacl::generator::autotune::inc::mul_by_two);
    conf.add_tuning_param("lhs_storage",1,1,&viennacl::generator::autotune::inc::add_one);
    conf.add_tuning_param("rhs_storage",0,0,&viennacl::generator::autotune::inc::add_one);
    conf.add_tuning_param("unroll",1,1,&viennacl::generator::autotune::inc::mul_by_two);

    viennacl::matrix<NumericT> vcl_A(1,1);
    viennacl::matrix<NumericT> vcl_B(1,1);
    viennacl::matrix<NumericT> vcl_C(1,1);

    viennacl::generator::matrix<viennacl::matrix<NumericT> > A(vcl_A);
    viennacl::generator::matrix<viennacl::matrix<NumericT> > B(vcl_B);
    viennacl::generator::matrix<viennacl::matrix<NumericT> > C(vcl_C);

    std::list<viennacl::generator::code_generation::matrix_product_profile> fastest_firsts;

    using namespace viennacl::generator::code_generation;

    if(is_lhs_trans)
      if(is_rhs_trans)
        benchmark<NumericT>(A = viennacl::generator::prod(viennacl::generator::trans(B), viennacl::generator::trans(C)), std::make_pair(gemmTT,sizeof(NumericT)),conf,vcl_A,vcl_B,vcl_C,fastest_firsts);
      else
        benchmark<NumericT>(A = viennacl::generator::prod(viennacl::generator::trans(B), C), std::make_pair(gemmTA,sizeof(NumericT)),conf,vcl_A,vcl_B,vcl_C,fastest_firsts);
    else
      if(is_rhs_trans)
        benchmark<NumericT>(A = viennacl::generator::prod(B, viennacl::generator::trans(C)), std::make_pair(gemmAT,sizeof(NumericT)),conf,vcl_A,vcl_B,vcl_C,fastest_firsts);
      else
        benchmark<NumericT>(A = viennacl::generator::prod(B, C), std::make_pair(gemmAA,sizeof(NumericT)),conf,vcl_A,vcl_B,vcl_C,fastest_firsts);

}

int main(int argc, char* argv[]){
    std::vector<std::string> args(argv,argv+argc);
    if(argc<3){
        std::cerr << "USAGE : PROGRAM_NAME DEVICE LAYOUT SCALARTYPE" << std::endl;
        exit(1);
    }


    unsigned int current_device=0;
    unsigned int requested_device = atoi(args[1].c_str());
    unsigned int layout = atoi(args[2].c_str());
    std::string scalartype = args[3];
    platforms_type platforms = viennacl::ocl::get_platforms();
    size_t num_platforms = platforms.size();
    for(unsigned int k=0 ; k < num_platforms ; ++k)
    {
        viennacl::ocl::platform pf(k);
        viennacl::ocl::set_context_platform_index(k,k);
        viennacl::ocl::set_context_device_type(k,CL_DEVICE_TYPE_ALL);
        viennacl::ocl::switch_context(k);
        devices_type dev = viennacl::ocl::current_context().devices();

        for(devices_type::iterator it = dev.begin() ; it != dev.end() ; ++it){

            if(current_device++==requested_device){
                viennacl::ocl::switch_device(*it);

                std::string devname = viennacl::ocl::current_device().name();

                std::cout << "-------------------" << std::endl;
                std::cout << "Recording timings for : " << devname << std::endl;
                std::cout << "Vendor ID : " << viennacl::ocl::info<CL_DEVICE_VENDOR_ID>(viennacl::ocl::current_device().id()) << std::endl;

                std::cout << "Matrix - Matrix Multiplication " << std::endl;
                std::cout << "-------------------" << std::endl;
                std::cout << " Scalartype : " << scalartype << std::endl;
                std::cout << "-------------------" << std::endl;
                switch(layout){
                case 0:
                    std::cout << "====== Step 1 : AA =====" << std::endl;
                    if(scalartype=="float") run_autotune<float>(false,false);
                    else if(scalartype=="double") run_autotune<double>(false,false);
                    break;


                case 1:
                    std::cout << "====== Step 3 : TA =====" << std::endl;
                    if(scalartype=="float") run_autotune<float>(true,false);
                    else if(scalartype=="double") run_autotune<double>(true,false);
                    break;


                case 2:
                    std::cout << "====== Step 2 : AT =====" << std::endl;
                    if(scalartype=="float") run_autotune<float>(false,true);
                    else if(scalartype=="double") run_autotune<double>(false,true);
                    break;

                case 3:
                    std::cout << "====== Step 4 : TT =====" << std::endl;
                    if(scalartype=="float") run_autotune<float>(true,true);
                    else if(scalartype=="double") run_autotune<double>(true,true);
                    break;
                }

                exit(0);

            }
        }


    }
}
