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


template<class NumericT>
void run_autotune(bool is_lhs_trans, bool is_rhs_trans){
    using namespace viennacl::generator::code_generation;
    typedef std::map<double, viennacl::generator::code_generation::matrix_product_profile> timings_t;

    viennacl::generator::autotune::tuning_config<blas3_config<NumericT> > conf;

    std::vector<int> ml; for(unsigned int i=16 ; i<=128 ; i+=16) ml.push_back(i);
    std::vector<int> kl; for(unsigned int i=16 ; i<=128 ; i+=16) kl.push_back(i);
    std::vector<int> nl; for(unsigned int i=16 ; i<=128 ; i+=16) nl.push_back(i);
    std::vector<int> ms; for(unsigned int i=1 ; i<= 8 ; i*=2) ms.push_back(i);
    std::vector<int> ks; for(unsigned int i=1 ; i<= 8 ; i*=2) ks.push_back(i);
    std::vector<int> ns; for(unsigned int i=1 ; i<= 8 ; i*=2) ns.push_back(i);
    std::vector<int> vector; for(unsigned int i=1 ; i<=4 ; i*=2) vector.push_back(i);
    std::vector<int> lhs_storage; for(unsigned int i=1 ; i<=1 ; ++i) lhs_storage.push_back(i);
    std::vector<int> rhs_storage; for(unsigned int i=0 ; i<=0 ; ++i) rhs_storage.push_back(i);
    std::vector<int> unroll; unroll.push_back(1);

    conf.add_tuning_param("ml",ml);
    conf.add_tuning_param("kl",kl);
    conf.add_tuning_param("nl",nl);
    conf.add_tuning_param("ms",ms);
    conf.add_tuning_param("ks",ks);
    conf.add_tuning_param("ns",ns);
    conf.add_tuning_param("vector",vector);
    conf.add_tuning_param("lhs_storage",lhs_storage);
    conf.add_tuning_param("rhs_storage",rhs_storage);
    conf.add_tuning_param("unroll",unroll);


    timings_t timings;
    std::list<viennacl::generator::code_generation::matrix_product_profile> fastest_firsts;

    std::list<std::pair<unsigned int, unsigned int> > rounds_config;
    rounds_config.push_back(std::make_pair(512,10));
//    rounds_config.push_back(std::make_pair(2048,10));

    for(std::list<std::pair<unsigned int, unsigned int> >::iterator it = rounds_config.begin() ; it!= rounds_config.end(); ++it){
        unsigned int k = std::distance(rounds_config.begin(),it);
        timings.clear();
        unsigned int size=it->first;
        unsigned int n_keep=it->second;
        viennacl::matrix<NumericT> vcl_A(size,size);
        viennacl::matrix<NumericT> vcl_B(size,size);
        viennacl::matrix<NumericT> vcl_C(size,size);

        fill_matrix<NumericT>(vcl_A,vcl_B,vcl_C);

        viennacl::generator::matrix<viennacl::matrix<NumericT> > A(vcl_A);
        viennacl::generator::matrix<viennacl::matrix<NumericT> > B(vcl_B);
        viennacl::generator::matrix<viennacl::matrix<NumericT> > C(vcl_C);
        viennacl::backend::finish();

        if(k==0){
          if(is_lhs_trans)
            if(is_rhs_trans)
              viennacl::generator::autotune::benchmark(timings,A = viennacl::generator::prod(viennacl::generator::trans(B), viennacl::generator::trans(C)), std::make_pair(gemmTT,sizeof(NumericT)),conf);
            else
              viennacl::generator::autotune::benchmark(timings,A = viennacl::generator::prod(viennacl::generator::trans(B), C), std::make_pair(gemmTA,sizeof(NumericT)),conf);
          else
            if(is_rhs_trans)
              viennacl::generator::autotune::benchmark(timings,A = viennacl::generator::prod(B, viennacl::generator::trans(C)), std::make_pair(gemmAT,sizeof(NumericT)),conf);
            else
              viennacl::generator::autotune::benchmark(timings,A = viennacl::generator::prod(B, C), std::make_pair(gemmAA,sizeof(NumericT)),conf);
        }
        else{
          if(is_lhs_trans)
            if(is_rhs_trans)
              viennacl::generator::autotune::benchmark(timings,A = viennacl::generator::prod(viennacl::generator::trans(B), viennacl::generator::trans(C)), std::make_pair(gemmTT,sizeof(NumericT)),fastest_firsts, sizeof(NumericT));
            else
              viennacl::generator::autotune::benchmark(timings,A = viennacl::generator::prod(viennacl::generator::trans(B), C), std::make_pair(gemmTA,sizeof(NumericT)),fastest_firsts, sizeof(NumericT));
          else
            if(is_rhs_trans)
              viennacl::generator::autotune::benchmark(timings,A = viennacl::generator::prod(B, viennacl::generator::trans(C)), std::make_pair(gemmAT,sizeof(NumericT)),fastest_firsts, sizeof(NumericT));
            else
              viennacl::generator::autotune::benchmark(timings,A = viennacl::generator::prod(B, C), std::make_pair(gemmAA,sizeof(NumericT)),fastest_firsts, sizeof(NumericT));
        }
        fastest_firsts.clear();
        viennacl::backend::finish();
        for(timings_t::iterator itt = timings.begin(); itt!=timings.end() ; ++itt){
            unsigned int n = std::distance(timings.begin(),itt);
            if(n>n_keep) break;
            fastest_firsts.push_back(itt->second);
            if(std::distance(rounds_config.begin(),it)==(int)rounds_config.size()-1){
                std::cout << "-----------" << std::endl;
                std::cout << std::distance(timings.begin(),itt) << "th Best : " << itt->first << "s | " << 2*std::pow((double)size/1000,3)/itt->first << " GFlops : " << std::endl;
                std::cout << "ML : " << itt->second.ml() << std::endl;
                std::cout << "NL : " << itt->second.nl() << std::endl;
                std::cout << "KL : " << itt->second.kl() << std::endl;

                std::cout << "MS : " << itt->second.ms() << std::endl;
                std::cout << "NS : " << itt->second.ns() << std::endl;
                std::cout << "KS : " << itt->second.ks() << std::endl;

                std::cout << "Unroll : " << itt->second.unroll() << std::endl;

                std::cout << "LHS Shared : " << std::boolalpha << itt->second.use_LHS_shared() << std::endl;
                std::cout << "RHS Shared : " << std::boolalpha << itt->second.use_RHS_shared() << std::endl;

            }
        }
    }
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
