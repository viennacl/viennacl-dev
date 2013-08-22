/* =========================================================================
   Copyright (c) 2010-2013, Institute for Microelectronics,
                            Institute for Analysis and Scientific Computing,
                            TU Wien.
   Portions of this software are copyright by UChicago Argonne, LLC.

                            -----------------
                  ViennaCL - The Vienna Computing Library
                            -----------------

   Project Head:    Karl Rupp                   rupp@iue.tuwien.ac.at

   (A list of authors and contributors can be found in the PDF manual)

   License:         MIT (X11), see file LICENSE in the base directory
============================================================================= */

#define SIZE_INC 128
#define MAX_SIZE 3072
#define N_RUNS 2

//#define VIENNACL_DEBUG_BUILD
//#define VIENNACL_DEBUG_ALL

#include <algorithm>
#include <string>
#include <iostream>
#include "viennacl/ocl/platform.hpp"
#include "viennacl/ocl/device.hpp"

#include <sys/time.h>

#include "boost/numeric/ublas/matrix.hpp"

#include "viennacl/matrix.hpp"
#include "viennacl/generator/generate.hpp"
#include "viennacl/generator/autotune.hpp"
#include "viennacl/linalg/prod.hpp"
#include "viennacl/linalg/norm_2.hpp"

#include "../tutorial/Random.hpp"
#include "../benchmarks/benchmark-utils.hpp"

using namespace viennacl::generator;

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

template<class MatA, class MatB, class MatC>
viennacl::scheduler::statement * allocate_statement(bool is_lhs_trans, bool is_rhs_trans, MatA const & A, MatB const & B, MatC const & C){
    if(is_lhs_trans)
      if(is_rhs_trans)
          return new viennacl::scheduler::statement(C, viennacl::op_assign(), viennacl::linalg::prod(trans(A),trans(B)));
      else
          return new viennacl::scheduler::statement(C, viennacl::op_assign(), viennacl::linalg::prod(trans(A),B));
    else
      if(is_rhs_trans)
          return new viennacl::scheduler::statement(C, viennacl::op_assign(), viennacl::linalg::prod(A,trans(B)));
      else
          return new viennacl::scheduler::statement(C, viennacl::op_assign(), viennacl::linalg::prod(A,B));

}

template<typename ScalarType>
double run_benchmark(size_t size, bool is_lhs_trans, bool is_rhs_trans, code_generator::forced_profile_key_type const & key, typename blas3_config<ScalarType>::profile_type const & profile)
{    //viennacl::ocl::current_context().build_options("-cl-mad-enable -cl-fast-relaxed-math");   //uncomment for additional optimizations
    //viennacl::ocl::current_context().build_options("-cl-opt-disable");                        //uncomment to get poor performance
    viennacl::matrix<ScalarType> A(size, size);
    viennacl::matrix<ScalarType> B(size, size);
    viennacl::matrix<ScalarType> C(size, size);
    viennacl::scheduler::statement * statement = allocate_statement(is_lhs_trans, is_rhs_trans,A,B,C);
    viennacl::generator::code_generator gen;
    gen.add(*statement,statement->array()[0]);
    gen.force_profile(key, profile);
    viennacl::generator::enqueue(gen);
    viennacl::generator::enqueue(gen);
    viennacl::backend::finish();
    Timer timer;
    timer.start();
    for(unsigned int r = 0 ; r < N_RUNS ; ++r){
      viennacl::generator::enqueue(gen);
    }
    viennacl::backend::finish();
    double time = timer.get()/(double)N_RUNS;
    delete statement;
    return 2*pow(size/static_cast<double>(1000.0),3)/time;
}

template<class NumericT>
void run_autotune(std::string const & dump_name, bool light_tuning, viennacl::ocl::device const & device, bool is_lhs_trans, bool is_rhs_trans){
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
    std::vector<int> lhs_storage; std::vector<int> rhs_storage;
    if(light_tuning){
        lhs_storage.push_back(1);
        rhs_storage.push_back(0);
    }
    else{
        for(unsigned int i=0 ; i<=1 ; ++i) lhs_storage.push_back(i);
        for(unsigned int i=0 ; i<=1 ; ++i) rhs_storage.push_back(i);
    }
    std::vector<int> unroll; unroll.push_back(0); unroll.push_back(1);

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
    rounds_config.push_back(std::make_pair(2432,100));

    std::ofstream stream(dump_name.c_str());
    std::size_t scalartype_size = sizeof(NumericT);

    code_generator::forced_profile_key_type * key = NULL;
    if(is_lhs_trans)
        if(is_rhs_trans) key = new code_generator::forced_profile_key_type(MATRIX_PRODUCT_TT_TYPE, scalartype_size);
        else  key = new code_generator::forced_profile_key_type(MATRIX_PRODUCT_TA_TYPE, scalartype_size);
    else
        if(is_rhs_trans) key = new code_generator::forced_profile_key_type(MATRIX_PRODUCT_AT_TYPE, scalartype_size);
        else key = new code_generator::forced_profile_key_type(MATRIX_PRODUCT_AA_TYPE, scalartype_size);


    stream << "#" << expression_type_to_string(key->first) << " | Scalartype Size : " << key->second << std::endl;
    stream << "#----------------------" << std::endl;
    stream << "#----------------------" << std::endl;
    stream << "#----------------------" << std::endl;
    stream << device.full_info(1,'#');
    stream << "#----------------------" << std::endl;
    stream << "#tuning for size : " << rounds_config.front().first << std::endl;

    for(std::list<std::pair<unsigned int, unsigned int> >::iterator it = rounds_config.begin() ; it!= rounds_config.end(); ++it){
        unsigned int k = std::distance(rounds_config.begin(),it);
        timings.clear();
        unsigned int size=it->first;
        unsigned int n_keep=it->second;

        MatrixT A(size,size);
        MatrixT B(size,size);
        MatrixT C(size,size);

        fill_matrix<NumericT>(A,B,C);

        viennacl::backend::finish();
        viennacl::scheduler::statement * statement = allocate_statement(is_lhs_trans, is_rhs_trans,A,B,C);

        if(k==0)
          autotune::benchmark(&timings,*statement,*key,conf,&stream);
        else{
          unsigned int n=0;
          for(typename std::list<typename blas3_config<NumericT>::profile_type>::const_iterator it = fastest_firsts.begin(); it!=fastest_firsts.end(); ++it){
            double percent = (double)n++*100/fastest_firsts.size();
            std::cout << '\r' << "Determining best profile for size " << size << "..." << "[" << std::setprecision(2) << std::setfill (' ') << std::setw(6) << std::fixed  << percent << "%" << "]" << std::flush;
            double exec_time = autotune::benchmark_impl(*statement,*key,*it);
            timings.insert(std::make_pair(exec_time, *it));
          }
          std::cout << std::endl;
        }
        fastest_firsts.clear();
        viennacl::backend::finish();
        for(timings_t::iterator itt = timings.begin(); itt!=timings.end() ; ++itt){
            unsigned int n = std::distance(timings.begin(),itt);
            if(n>n_keep) break;
            fastest_firsts.push_back(itt->second);
        }
        stream << "# " << " Size : " << size << " | Best : " << 2*std::pow((double)size/1000,3)/timings.begin()->first << " GFlops : " << timings.begin()->second << std::endl;

        //Update default profile
        viennacl::generator::code_generator dummy;
        dummy.add(*statement,statement->array()[0]);
        dummy.force_profile(*key, timings.begin()->second);
        viennacl::generator::enqueue(dummy,true);
        viennacl::backend::finish();

        delete statement;
    }

    stream << "#Benchmarking " << timings.begin()->second << "..." << std::endl;
    stream << "##Size\tGFLOP/s" << std::endl;
    for(unsigned int size = SIZE_INC ; size <= MAX_SIZE ; size += SIZE_INC){
        double percent = (double)size/MAX_SIZE*100;
        std::cout << '\r' << "Benchmarking..." << "[" << std::setprecision(2) << std::setfill (' ') << std::setw(6) << std::fixed  << percent << "%" << "]" << std::flush;
        stream << size << "\t" << run_benchmark<NumericT>(size,is_lhs_trans,is_rhs_trans,*key,timings.begin()->second) << std::endl;
    }

    delete key;
}


int main(int argc, char* argv[]){
    typedef std::vector< viennacl::ocl::platform > platforms_type;
  std::vector<std::string> args(argv, argv+argc);
  if(argc<4){
      std::cerr << "USAGE : PROGRAM_NAME DEVICE LAYOUT SCALARTYPE LIGHT_TUNING" << std::endl;
      exit(1);
  }
  unsigned int current_device=0;
  unsigned int requested_device = atoi(args[1].c_str());
  unsigned int layout = atoi(args[2].c_str());
  std::string scalartype = args[3];
  unsigned int light_tuning = atoi(args[4].c_str());

  platforms_type platforms = viennacl::ocl::get_platforms();
  for (platforms_type::iterator platform_iter  = platforms.begin();
       platform_iter != platforms.end();
       ++platform_iter)
  {
    typedef std::vector<viennacl::ocl::device> devices_type;
    devices_type devices = platform_iter->devices(CL_DEVICE_TYPE_ALL);
    for(devices_type::iterator iter = devices.begin(); iter != devices.end(); iter++)
    {
      if(current_device++==requested_device){
        viennacl::ocl::setup_context(current_device,*iter);
        viennacl::ocl::switch_context(current_device);
        viennacl::ocl::device const & device = viennacl::ocl::current_device();
        std::string device_name = device.name();
        std::transform(device_name.begin(), device_name.end(), device_name.begin(), ::tolower);
        std::replace(device_name.begin(), device_name.end(),' ', '_');
        std::cout << "-------------------" << std::endl;
        std::cout << device.info() << std::endl;
        std::cout << "GEMM" << std::endl;
        std::cout << "-------------------" << std::endl;
        std::cout << " Scalartype : " << scalartype << std::endl;
        std::cout << "-------------------" << std::endl;
        switch(layout){
          case 0:
            std::cout << "Layout : AA" << std::endl;
            if(scalartype=="float")
              run_autotune<float>("gemm_aa_float_" + device_name + ".dat",light_tuning,device,false,false);
            else if(scalartype=="double")
              run_autotune<double>("gemm_aa_double_" + device_name + ".dat",light_tuning,device,false,false);
            break;


          case 1:
            std::cout << "Layout : TA" << std::endl;
            if(scalartype=="float")
              run_autotune<float>("gemm_ta_float_" + device_name + ".dat",light_tuning,device, true, false);
            else if(scalartype=="double")
              run_autotune<double>("gemm_ta_double_" + device_name + ".dat",light_tuning,device, true, false);
            break;


          case 2:
            std::cout << "Layout : AT" << std::endl;
            if(scalartype=="float")
              run_autotune<float>("gemm_at_float_" + device_name + ".dat",light_tuning,device, false, true);
            else if(scalartype=="double")
              run_autotune<double>("gemm_at_double_" + device_name + ".dat",light_tuning,device, false, true);
            break;

          case 3:
            std::cout << "Layout : TT" << std::endl;
            if(scalartype=="float")
              run_autotune<float>("gemm_tt_float_" + device_name + ".dat",light_tuning,device,true,true);
            else if(scalartype=="double")
              run_autotune<double>("gemm_tt_double_" + device_name + ".dat",light_tuning,device, true, true);
            break;
        }
      }
    }
  }
  std::cout << std::endl;
  std::cout << "Autotuning Complete!" << std::endl;
  return EXIT_SUCCESS;
}
