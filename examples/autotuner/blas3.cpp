#define VIENNACL_WITH_OPENCL

//#define VIENNACL_DEBUG_BUILD
//#define VIENNACL_DEBUG_ALL

#define NDEBUG

#include <iostream>
#include "CL/cl.hpp"
#include <sys/time.h>

#include "boost/numeric/ublas/matrix.hpp"

#include "viennacl/matrix.hpp"
#include "viennacl/generator/generate.hpp"
#include "viennacl/generator/autotune.hpp"
#include "viennacl/linalg/prod.hpp"
#include "viennacl/linalg/norm_2.hpp"

#include "../tutorial/Random.hpp"


using namespace viennacl::generator;

typedef std::vector< viennacl::ocl::platform > platforms_type;
typedef std::vector<viennacl::ocl::device> devices_type;
typedef std::vector<cl_device_id> cl_devices_type;


template<class ScalarType>
struct blas3_config{
    typedef matrix_product profile_type;
    static profile_type create_profile(std::map<std::string, autotune::tuning_param> const & params){
        return profile_type(params.at("vector").current()
                        , params.at("local_size1").current()
                        , params.at("cache_width").current()
                        , params.at("local_size2").current()
                        , params.at("ms").current()
                        , params.at("ks").current()
                        , params.at("ns").current(),
                         static_cast<bool>(params.at("lhs_storage").current()),static_cast<bool>(params.at("rhs_storage").current()),
                         params.at("unroll").current());
    }
    static bool is_invalid(viennacl::ocl::device const & dev, std::map<std::string, autotune::tuning_param> const & params){
        profile_type prof = create_profile(params);
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
void run_autotune(std::string const & dump_name, bool is_lhs_trans, bool is_rhs_trans){
    typedef std::map<double, matrix_product> timings_t;
    typedef viennacl::matrix<NumericT> MatrixT;

    autotune::tuning_config<blas3_config<NumericT> > conf;

    std::vector<int> local_size1; for(unsigned int i=2 ; i<=64 ; i*=2) local_size1.push_back(i);
    std::vector<int> cache_width; for(unsigned int i=16 ; i<=128 ; i*=2) cache_width.push_back(i);
    std::vector<int> local_size2; for(unsigned int i=2 ; i<=64 ; i*=2) local_size2.push_back(i);
    std::vector<int> ms; for(unsigned int i=1 ; i<= 8 ; i*=2) ms.push_back(i);
    std::vector<int> ks; for(unsigned int i=1 ; i<= 8 ; i*=2) ks.push_back(i);
    std::vector<int> ns; for(unsigned int i=1 ; i<= 8 ; i*=2) ns.push_back(i);
    std::vector<int> vector; for(unsigned int i=1 ; i<=4 ; i*=2) vector.push_back(i);
    std::vector<int> lhs_storage; for(unsigned int i=1 ; i<=1 ; ++i) lhs_storage.push_back(i);
    std::vector<int> rhs_storage; for(unsigned int i=0 ; i<=0 ; ++i) rhs_storage.push_back(i);
    std::vector<int> unroll; unroll.push_back(1);

    conf.add_tuning_param("local_size1",local_size1);
    conf.add_tuning_param("cache_width",cache_width);
    conf.add_tuning_param("local_size2",local_size2);
    conf.add_tuning_param("ms",ms);
    conf.add_tuning_param("ks",ks);
    conf.add_tuning_param("ns",ns);
    conf.add_tuning_param("vector",vector);
    conf.add_tuning_param("lhs_storage",lhs_storage);
    conf.add_tuning_param("rhs_storage",rhs_storage);
    conf.add_tuning_param("unroll",unroll);


    timings_t timings;
    std::list<matrix_product> fastest_firsts;

    std::list<std::pair<unsigned int, unsigned int> > rounds_config;

    rounds_config.push_back(std::make_pair(1280,100));
    rounds_config.push_back(std::make_pair(2304,100));
    rounds_config.push_back(std::make_pair(3584,100));
    rounds_config.push_back(std::make_pair(4608,100));

    std::ofstream stream(dump_name.c_str());

    for(std::list<std::pair<unsigned int, unsigned int> >::iterator it = rounds_config.begin() ; it!= rounds_config.end(); ++it){
        unsigned int k = std::distance(rounds_config.begin(),it);
        timings.clear();
        unsigned int size=it->first;
        unsigned int n_keep=it->second;
        std::cout << "Round " << k << " : Tuning for size " << size << std::endl;
        MatrixT A(size,size);
        MatrixT B(size,size);
        MatrixT C(size,size);

        fill_matrix<NumericT>(A,B,C);

        viennacl::backend::finish();

        std::size_t scalartype_size = sizeof(NumericT);
        code_generator::forced_profile_key_type keyAA(MATRIX_PRODUCT_AA_TYPE, scalartype_size);
        code_generator::forced_profile_key_type keyTA(MATRIX_PRODUCT_TA_TYPE, scalartype_size);
        code_generator::forced_profile_key_type keyAT(MATRIX_PRODUCT_AT_TYPE, scalartype_size);
        code_generator::forced_profile_key_type keyTT(MATRIX_PRODUCT_AA_TYPE, scalartype_size);

        if(k==0){
          if(is_lhs_trans)
              if(is_rhs_trans)
                  autotune::benchmark(&timings,viennacl::scheduler::statement(A, viennacl::op_assign(), viennacl::linalg::prod(trans(B), trans(C))),keyTT,conf,&stream);
              else
                  autotune::benchmark(&timings,viennacl::scheduler::statement(A, viennacl::op_assign(), viennacl::linalg::prod(trans(B), C)),keyTA,conf,&stream);
          else
              if(is_rhs_trans)
                  autotune::benchmark(&timings,viennacl::scheduler::statement(A, viennacl::op_assign(), viennacl::linalg::prod(B, trans(C))),keyAT,conf,&stream);
              else
                  autotune::benchmark(&timings,viennacl::scheduler::statement(A, viennacl::op_assign(), viennacl::linalg::prod(B,C)),keyAA,conf,&stream);
        }
        else{
          if(is_lhs_trans)
            if(is_rhs_trans)
              autotune::benchmark(&timings,viennacl::scheduler::statement(A, viennacl::op_assign(), viennacl::linalg::prod(trans(B), trans(C))),keyTT,fastest_firsts);
            else
              autotune::benchmark(&timings,viennacl::scheduler::statement(A, viennacl::op_assign(), viennacl::linalg::prod(trans(B), C)),keyTA,fastest_firsts);
          else
            if(is_rhs_trans)
              autotune::benchmark(&timings,viennacl::scheduler::statement(A, viennacl::op_assign(), viennacl::linalg::prod(B, trans(C))),keyAT,fastest_firsts);
            else
              autotune::benchmark(&timings,viennacl::scheduler::statement(A, viennacl::op_assign(), viennacl::linalg::prod(B,C)),keyAA,fastest_firsts);
        }
        fastest_firsts.clear();
        viennacl::backend::finish();
        for(timings_t::iterator itt = timings.begin(); itt!=timings.end() ; ++itt){
            unsigned int n = std::distance(timings.begin(),itt);
            if(n>n_keep) break;
            fastest_firsts.push_back(itt->second);
        }
        std::cout << "Best : " << 2*std::pow((double)size/1000,3)/timings.begin()->first << " GFlops : " << timings.begin()->second << std::endl;
        std::cout << "-----------" << std::endl;
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
                    std::cout << "Layout : AA" << std::endl;
                    if(scalartype=="float")
                        run_autotune<float>("BLAS3 AA Float" + it->name(),false,false);
                    else if(scalartype=="double")
                        run_autotune<double>("BLAS3 AA Double" + it->name(),false,false);
                    break;


                case 1:
                    std::cout << "Layout : TA" << std::endl;
                    if(scalartype=="float")
                        run_autotune<float>("BLAS3 TA Float" + it->name(), true, false);
                    else if(scalartype=="double")
                        run_autotune<double>("BLAS3 TA Double" + it->name(), true, false);
                    break;


                case 2:
                    std::cout << "Layout : AT" << std::endl;
                    if(scalartype=="float")
                        run_autotune<float>("BLAS3 AT Float" + it->name(), false, true);
                    else if(scalartype=="double")
                        run_autotune<double>("BLAS3 AT Double" + it->name(), false, true);
                    break;

                case 3:
                    std::cout << "Layout : TT" << std::endl;
                    if(scalartype=="float")
                        run_autotune<float>("BLAS3 TT Float" + it->name(),true,true);
                    else if(scalartype=="double")
                        run_autotune<double>("BLAS3 TT Double" + it->name(), true, true);
                    break;
                }

                exit(0);

            }
        }


    }
}
