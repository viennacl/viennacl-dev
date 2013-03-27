//#define VIENNACL_DEBUG_BUILD
#define VIENNACL_WITH_OPENCL
//#define VIENNACL_DEBUG_ALL
//#define VIENNACL_USE_SCHEDULER

#define NDEBUG

#include <iostream>
#include "CL/cl.hpp"
#include <sys/time.h>

#include "boost/numeric/ublas/matrix.hpp"

#include "viennacl/matrix.hpp"
#include "viennacl/generator/custom_operation.hpp"
#include "viennacl/generator/dummy_types.hpp"
#include "viennacl/generator/autotune/autotune.hpp"
#include "viennacl/linalg/prod.hpp"
#include "viennacl/linalg/norm_2.hpp"

#include "viennacl/io/kernel_parameters.hpp"

#include "../tutorial/Random.hpp"


typedef std::vector< viennacl::ocl::platform > platforms_type;
typedef std::vector<viennacl::ocl::device> devices_type;
typedef std::vector<cl_device_id> cl_devices_type;



struct config{

    unsigned int size;
    unsigned int n_runs;

    unsigned int ml_min;
    unsigned int kl_min;
    unsigned int nl_min;
    unsigned int ms_min;
    unsigned int ks_min;
    unsigned int ns_min;

    unsigned int ml_max;
    unsigned int kl_max;
    unsigned int nl_max;
    unsigned int ms_max;
    unsigned int ks_max;
    unsigned int ns_max;

    unsigned int alignment_min;
    unsigned int alignment_max;

    unsigned int min_unroll;
    unsigned int max_unroll;

    std::vector<bool> LHS_storages;
    std::vector<bool> RHS_storages;
};

template<class OpT>
void add_profile(OpT const & OP, viennacl::generator::code_generation::blas3_optimization_profile const & prof, viennacl::io::parameter_database & paras){
    viennacl::generator::custom_operation op;
    op.add(OP);

    paras.add_kernel();
    paras.add_data_node(viennacl::io::tag::type, viennacl::io::val::blas3::blas3);
    paras.add_data_node(viennacl::io::tag::name, op.operations_manager().get_kernels_list().front().trees().front()->repr());

    paras.add_parameter();
    paras.add_data_node(viennacl::io::tag::name, viennacl::io::tag::blas3::ml);
    paras.add_data_node(viennacl::io::tag::value, prof.ml());
    paras.add_parameter();
    paras.add_data_node(viennacl::io::tag::name, viennacl::io::tag::blas3::kl);
    paras.add_data_node(viennacl::io::tag::value, prof.kl());
    paras.add_parameter();
    paras.add_data_node(viennacl::io::tag::name, viennacl::io::tag::blas3::nl);
    paras.add_data_node(viennacl::io::tag::value, prof.nl());
    paras.add_parameter();
    paras.add_data_node(viennacl::io::tag::name, viennacl::io::tag::blas3::ms);
    paras.add_data_node(viennacl::io::tag::value, prof.ms());
    paras.add_parameter();
    paras.add_data_node(viennacl::io::tag::name, viennacl::io::tag::blas3::ks);
    paras.add_data_node(viennacl::io::tag::value, prof.ks());
    paras.add_parameter();
    paras.add_data_node(viennacl::io::tag::name, viennacl::io::tag::blas3::ns);
    paras.add_data_node(viennacl::io::tag::value, prof.ns());
    paras.add_parameter();
    paras.add_data_node(viennacl::io::tag::name, viennacl::io::tag::blas3::lhs_storage);
    paras.add_data_node(viennacl::io::tag::value, prof.use_LHS_shared());
    paras.add_parameter();
    paras.add_data_node(viennacl::io::tag::name, viennacl::io::tag::blas3::rhs_storage);
    paras.add_data_node(viennacl::io::tag::value, prof.use_RHS_shared());
    paras.add_parameter();
    paras.add_data_node(viennacl::io::tag::name, viennacl::io::tag::alignment);
    paras.add_data_node(viennacl::io::tag::value, prof.alignment());
}

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
    viennacl::ocl::get_queue().finish();
}


template<class NumericT, class OpT, class MatTypeA, class MatTypeB, class MatTypeC>

void benchmark(OpT const & operation, config conf, MatTypeA & A, MatTypeB & B, MatTypeC & C,
                    std::list<viennacl::generator::code_generation::blas3_optimization_profile> & fastest_firsts){
    viennacl::generator::autotune::timings_t timings;
    unsigned int size;

    std::list<std::pair<unsigned int, unsigned int> > rounds_config;
    rounds_config.push_back(std::make_pair(512,70));
    rounds_config.push_back(std::make_pair(2048,20));
    for(std::list<std::pair<unsigned int, unsigned int> >::iterator it = rounds_config.begin() ; it!= rounds_config.end(); ++it){
        unsigned int k = std::distance(rounds_config.begin(),it);
        timings.clear();
        size=it->first;
        unsigned int n_keep=it->second;
        A.resize(size,size,false);
        B.resize(size,size,false);
        C.resize(size,size,false);
        viennacl::ocl::get_queue().finish();
        fill_matrix<NumericT>(A,B,C);
        viennacl::ocl::get_queue().finish();
        if(k==0)
            viennacl::generator::autotune::benchmark_blas3(timings,operation,conf);
        else{
            viennacl::generator::autotune::benchmark_blas3(timings,operation,fastest_firsts);
        }
        fastest_firsts.clear();
        viennacl::ocl::get_queue().finish();
        for(viennacl::generator::autotune::timings_t::iterator itt = timings.begin(); itt!=timings.end() ; ++itt){
            unsigned int n = std::distance(timings.begin(),itt);
            if(n>n_keep) break;
            fastest_firsts.push_back(*static_cast<viennacl::generator::code_generation::blas3_optimization_profile* >(itt->second.get()));
            if(std::distance(rounds_config.begin(),it)==(int)rounds_config.size()-1){
                std::cout << std::distance(timings.begin(),itt) << "th Best : " << itt->first << "s | " << 2*std::pow((double)size/1000,3)/itt->first << " GFlops : " << *itt->second << std::endl;
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
void run_autotune(viennacl::io::parameter_database & paras){

    using viennacl::generator::prod;

    typedef viennacl::matrix<NumericT, FB> VclMatB1;
    typedef viennacl::matrix<NumericT, FC> VclMatC1;

    typedef viennacl::matrix<NumericT, typename opposite_layout<FB>::type> VclMatB2;
    typedef viennacl::matrix<NumericT, typename opposite_layout<FC>::type> VclMatC2;

    typedef viennacl::matrix<NumericT, viennacl::row_major> VclMatA1;
    typedef viennacl::matrix<NumericT, viennacl::column_major> VclMatA2;

    typedef viennacl::generator::dummy_matrix<VclMatA1> dma1_t;
    typedef viennacl::generator::dummy_matrix<VclMatB1> dmb1_t;
    typedef viennacl::generator::dummy_matrix<VclMatC1> dmc1_t;

    typedef viennacl::generator::dummy_matrix<VclMatA2> dma2_t;
    typedef viennacl::generator::dummy_matrix<VclMatB2> dmb2_t;
    typedef viennacl::generator::dummy_matrix<VclMatC2> dmc2_t;

    config conf;

    if(viennacl::ocl::info<CL_DEVICE_TYPE>(viennacl::ocl::current_device().id()) == CL_DEVICE_TYPE_CPU){
        conf.n_runs = 2;
        conf.ml_min = 32; conf.ml_max=256;
        conf.kl_min = 32; conf.kl_max=256;
        conf.nl_min = 32; conf.nl_max=256;
        conf.ms_min = 2; conf.ms_max=16;
        conf.ks_min = 2; conf.ks_max=8;
        conf.ns_min = 2; conf.ns_max=16;
        conf.alignment_min = 1 ; conf.alignment_max = 8 ;
    //    conf.LHS_storages.push_back(true);
        conf.LHS_storages.push_back(false);
    //    conf.RHS_storages.push_back(true);
        conf.RHS_storages.push_back(false);
        conf.min_unroll = 1;
        conf.max_unroll = 1;
    }
    else{
        //64,128,32,2,2,8,0,0,1

        conf.n_runs = 2;
        conf.ml_min = 32; conf.ml_max=256;
        conf.kl_min = 32; conf.kl_max=256;
        conf.nl_min = 32; conf.nl_max=256;
        conf.ms_min = 2; conf.ms_max=8;
        conf.ks_min = 2; conf.ks_max=8;
        conf.ns_min = 2; conf.ns_max=8;
        conf.alignment_min = 1 ; conf.alignment_max = 4 ;
        conf.LHS_storages.push_back(true);
        conf.LHS_storages.push_back(false);
        conf.RHS_storages.push_back(true);
        conf.RHS_storages.push_back(false);
        conf.min_unroll = 1;
        conf.max_unroll = 1;
    }


    VclMatA1 A1(1,1);
    VclMatB1 B1(1,1);
    VclMatC1 C1(1,1);

    VclMatA2 A2(1,1);
    VclMatB2 B2(1,1);
    VclMatC2 C2(1,1);

    std::list<viennacl::generator::code_generation::blas3_optimization_profile> fastest_firsts;


    //------------AA------------
    std::cout << "Getting best parameters..." << std::endl;
    benchmark<NumericT>(dma1_t(A1) = prod(dmb1_t(B1),dmc1_t(C1)),conf,A1,B1,C1,fastest_firsts);
    //--------------------------

    std::cout << "Saving..." << std::endl;
    //-----------Saves best parameters-------------//
    add_profile(dma1_t(A1) = prod(dmb1_t(B1),dmc1_t(C1)),fastest_firsts.front(),paras);
    add_profile(dma2_t(A2) = prod(dmb1_t(B1),dmc1_t(C1)),fastest_firsts.front(),paras);

    //-----------Changing layout and transposing is equivalent to doing nothing-------------//
    add_profile(dma1_t(A1) = prod(trans(dmb2_t(B2)),dmc1_t(C1)),fastest_firsts.front(),paras);
    add_profile(dma2_t(A2) = prod(trans(dmb2_t(B2)),dmc1_t(C1)),fastest_firsts.front(),paras);

    add_profile(dma1_t(A1) = prod(trans(dmb2_t(B2)),trans(dmc2_t(C2))),fastest_firsts.front(),paras);
    add_profile(dma2_t(A2) = prod(trans(dmb2_t(B2)),trans(dmc2_t(C2))),fastest_firsts.front(),paras);

    add_profile(dma1_t(A1) = prod(dmb1_t(B1),trans(dmc2_t(C2))),fastest_firsts.front(),paras);
    add_profile(dma2_t(A2) = prod(dmb1_t(B1),trans(dmc2_t(C2))),fastest_firsts.front(),paras);


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
                viennacl::io::parameter_database  paras;
                viennacl::ocl::switch_device(*it);
                cl_device_id dev_id = it->id();
                paras.add_device();
                paras.add_data_node(viennacl::io::tag::name, viennacl::ocl::info<CL_DEVICE_NAME>(dev_id));
                paras.add_data_node(viennacl::io::tag::driver, viennacl::ocl::info<CL_DRIVER_VERSION>(dev_id));

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
                    if(scalartype=="float") run_autotune<float,viennacl::row_major,viennacl::row_major>(paras);
                    else if(scalartype=="double") run_autotune<double,viennacl::row_major,viennacl::row_major>(paras);
                    break;


                case 1:
                    std::cout << "====== Step 3 : Column - Row and alikes =====" << std::endl;
                    if(scalartype=="float") run_autotune<float,viennacl::column_major,viennacl::row_major>(paras);
                    else if(scalartype=="double") run_autotune<double,viennacl::column_major,viennacl::row_major>(paras);
                    break;


                case 2:
                    std::cout << "====== Step 2 : Row - Column and alikes =====" << std::endl;
                    if(scalartype=="float") run_autotune<float,viennacl::row_major,viennacl::column_major>(paras);
                    else if(scalartype=="double") run_autotune<double,viennacl::row_major,viennacl::column_major>(paras);
                    break;

                case 3:
                    std::cout << "====== Step 4 : Column - Column and alikes =====" << std::endl;
                    if(scalartype=="float") run_autotune<float,viennacl::column_major,viennacl::column_major>(paras);
                    else if(scalartype=="double") run_autotune<double,viennacl::column_major,viennacl::column_major>(paras);
                    break;
                }

                paras.dump("parameters_gemm_"+devname+"_"+args[1] + "_" + args[2] + "_" + args[3] +".xml");

                exit(0);

            }
        }


    }
}
