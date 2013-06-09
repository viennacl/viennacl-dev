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
    typedef viennacl::generator::code_generation::gemm::profile profile_t;
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


template<class NumericT, class OpT, class ConfigT, class MatTypeA, class MatTypeB, class MatTypeC>

void benchmark(OpT const & operation, ConfigT conf, MatTypeA & A, MatTypeB & B, MatTypeC & C,
                    std::list<viennacl::generator::code_generation::gemm::profile> & fastest_firsts){
    typedef std::map<double, viennacl::generator::code_generation::gemm::profile> timings_t;
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
            viennacl::generator::autotune::benchmark(timings,operation,conf);
        else{
            viennacl::generator::autotune::benchmark(timings,operation,fastest_firsts);
        }
        fastest_firsts.clear();
        viennacl::backend::finish();
        for(timings_t::iterator itt = timings.begin(); itt!=timings.end() ; ++itt){
            unsigned int n = std::distance(timings.begin(),itt);
            if(n>n_keep) break;
            fastest_firsts.push_back(itt->second);
            if(std::distance(rounds_config.begin(),it)==(int)rounds_config.size()-1){
                std::cout << std::distance(timings.begin(),itt) << "th Best : " << itt->first << "s | " << 2*std::pow((double)size/1000,3)/itt->first << " GFlops : " << itt->second << std::endl;
            }
        }
    }
}


template<class F>
struct opposite_layout;

template<>
struct opposite_layout<viennacl::row_major> { typedef viennacl::column_major type;};

template<>
struct opposite_layout<viennacl::column_major> { typedef viennacl::row_major type;};



template<class NumericT, class FB, class FC>
void run_autotune(){

    using viennacl::generator::prod;

    typedef viennacl::matrix<NumericT, FB> VclMatB1;
    typedef viennacl::matrix<NumericT, FC> VclMatC1;

    typedef viennacl::matrix<NumericT, typename opposite_layout<FB>::type> VclMatB2;
    typedef viennacl::matrix<NumericT, typename opposite_layout<FC>::type> VclMatC2;

    typedef viennacl::matrix<NumericT, viennacl::row_major> VclMatA1;
    typedef viennacl::matrix<NumericT, viennacl::column_major> VclMatA2;

    typedef viennacl::generator::matrix<VclMatA1> dma1_t;
    typedef viennacl::generator::matrix<VclMatB1> dmb1_t;
    typedef viennacl::generator::matrix<VclMatC1> dmc1_t;

    typedef viennacl::generator::matrix<VclMatA2> dma2_t;
    typedef viennacl::generator::matrix<VclMatB2> dmb2_t;
    typedef viennacl::generator::matrix<VclMatC2> dmc2_t;

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

    VclMatA1 A1(1,1);
    VclMatB1 B1(1,1);
    VclMatC1 C1(1,1);

    VclMatA2 A2(1,1);
    VclMatB2 B2(1,1);
    VclMatC2 C2(1,1);

    std::list<viennacl::generator::code_generation::gemm::profile> fastest_firsts;


    //------------AA------------
    std::cout << "Getting best parameters..." << std::endl;
    benchmark<NumericT>(dma1_t(A1) = prod(dmb1_t(B1),dmc1_t(C1)),conf,A1,B1,C1,fastest_firsts);
    //--------------------------


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
                    std::cout << "====== Step 1 : Row - Row and alikes =====" << std::endl;
                    if(scalartype=="float") run_autotune<float,viennacl::row_major,viennacl::row_major>();
                    else if(scalartype=="double") run_autotune<double,viennacl::row_major,viennacl::row_major>();
                    break;


                case 1:
                    std::cout << "====== Step 3 : Column - Row and alikes =====" << std::endl;
                    if(scalartype=="float") run_autotune<float,viennacl::column_major,viennacl::row_major>();
                    else if(scalartype=="double") run_autotune<double,viennacl::column_major,viennacl::row_major>();
                    break;


                case 2:
                    std::cout << "====== Step 2 : Row - Column and alikes =====" << std::endl;
                    if(scalartype=="float") run_autotune<float,viennacl::row_major,viennacl::column_major>();
                    else if(scalartype=="double") run_autotune<double,viennacl::row_major,viennacl::column_major>();
                    break;

                case 3:
                    std::cout << "====== Step 4 : Column - Column and alikes =====" << std::endl;
                    if(scalartype=="float") run_autotune<float,viennacl::column_major,viennacl::column_major>();
                    else if(scalartype=="double") run_autotune<double,viennacl::column_major,viennacl::column_major>();
                    break;
                }

                exit(0);

            }
        }


    }
}
