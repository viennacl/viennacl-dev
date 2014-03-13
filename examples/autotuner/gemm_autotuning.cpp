/* =========================================================================
   Copyright (c) 2010-2014, Institute for Microelectronics,
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

//#define VIENNACL_DEBUG_BUILD
//#define VIENNACL_DEBUG_ALL

#include <algorithm>
#include <string>
#include <iostream>
#include "viennacl/ocl/platform.hpp"
#include "viennacl/ocl/device.hpp"

#include "viennacl/matrix.hpp"
#include "viennacl/linalg/prod.hpp"

#include "viennacl/generator/generate.hpp"
#include "viennacl/generator/autotuning/autotune.hpp"

#include "viennacl/tools/timer.hpp"

#include "command-line-utils.hpp"

using namespace viennacl::generator;

static const unsigned int n_runs = 10;

struct autotuner_options{

    std::string layout;
    std::string scalartype;
    std::string output_name;

    unsigned int requested_device;

    std::string ms_interval;
    std::string ks_interval;
    std::string ns_interval;

    std::string ls0_interval;
    std::string kl_interval;
    std::string ls1_interval;

    std::string simd_width_interval;

    std::string A_fetch_method;
    std::string B_fetch_method;

    std::string local_fetch0_interval;
    std::string local_fetch1_interval;

};

autotuner_options get_options(int argc, char* argv[]){
    try{
        autotuner_options options;

        TCLAP::CmdLine cmd("GEMM Autotuner", ' ', "0.1");


        pow_2_interval_constraint pow_2_interval_cstrt;

        //Layouts
        std::vector<std::string> allowed_layouts;
        allowed_layouts.push_back("NN");
        allowed_layouts.push_back("TN");
        allowed_layouts.push_back("NT");
        allowed_layouts.push_back("TT");
        TCLAP::ValuesConstraint<std::string> allowed_layouts_constraint( allowed_layouts);
        TCLAP::ValueArg<std::string> layout_arg("l","layout","Layout to tune the hardware for",true,"NN",&allowed_layouts_constraint,cmd);

        //Scalartype
        std::vector<std::string> allowed_scalartypes;
        allowed_scalartypes.push_back("float");
        allowed_scalartypes.push_back("double");
        TCLAP::ValuesConstraint<std::string> allowed_scalartypes_constraint( allowed_scalartypes);
        TCLAP::ValueArg<std::string> scalartype_arg("s","scalartype","Scalartype to tune the hardware for",true,"float",&allowed_scalartypes_constraint,cmd);

        //Output data file
        TCLAP::ValueArg<std::string> output_name_arg("o","output","Name of the output data file",true,"gemm_autotuning.dat","string",cmd);

        //Device id
        TCLAP::ValueArg<unsigned int> requested_device_arg("d","device","ID of the device to use for the autotuning procedure",false,0,"unsigned int",cmd);

        //Small blocks
        TCLAP::ValueArg<std::string> ms_interval_arg("","ms","Number of row in each block processed by each work-item. Specify min,max both power of two.",false,"2,8",&pow_2_interval_cstrt,cmd);
        TCLAP::ValueArg<std::string> ks_interval_arg("","ks","Increment size for each small block calculation. Specify min,max both power of two.",false,"2,8",&pow_2_interval_cstrt,cmd);
        TCLAP::ValueArg<std::string> ns_interval_arg("","ns","Number of column in each block processed by each work-item. Specify min,max both power of two.",false,"2,8",&pow_2_interval_cstrt,cmd);


        //Large blocks
        TCLAP::ValueArg<std::string> ls0_interval_arg("","ls-0","Number of work-item rows in each work-group. Specify min,max both power of two.",false,"8,16",&pow_2_interval_cstrt,cmd);
        TCLAP::ValueArg<std::string> kl_interval_arg("","kl","Increment size for each Large block calculation. Specify min,max both power of two.",false,"8,32",&pow_2_interval_cstrt,cmd);
        TCLAP::ValueArg<std::string> ls1_interval_arg("","ls-1","Number of work-item columns in each work-group. Specify min,max both power of two.",false,"8,16",&pow_2_interval_cstrt,cmd);



        //Vector
        TCLAP::ValueArg<std::string> vector_interval_arg("","vector","Vector type used in the kernel",false,"1,4",&pow_2_interval_cstrt,cmd);

        //Storage Type
        std::vector<std::string> allowed_fetch_method;
        allowed_fetch_method.push_back("local");
        allowed_fetch_method.push_back("global");
        allowed_fetch_method.push_back("all");
        TCLAP::ValuesConstraint<std::string> allowed_fetch_method_constraint(allowed_fetch_method);

        TCLAP::ValueArg<std::string> A_fetch_method_arg("","A-fetch","Method to fetch A.",false,"all",&allowed_fetch_method_constraint,cmd);
        TCLAP::ValueArg<std::string> B_fetch_method_arg("","B-fetch","Method to fetch B.",false,"all",&allowed_fetch_method_constraint,cmd);

        TCLAP::ValueArg<std::string> local_fetch0_interval("","local-fetch0","Internal size0 for fetching A to local memory",false,"4,32",&pow_2_interval_cstrt,cmd);
        TCLAP::ValueArg<std::string> local_fetch1_interval("","local-fetch1","Internal size1 for fetching A to local memory",false,"4,32",&pow_2_interval_cstrt,cmd);

        cmd.parse(argc,argv);
        options.layout = layout_arg.getValue();
        options.scalartype = scalartype_arg.getValue();
        options.output_name = output_name_arg.getValue();
        options.requested_device = requested_device_arg.getValue();
        options.ms_interval = ms_interval_arg.getValue();
        options.ks_interval = ks_interval_arg.getValue();
        options.ns_interval = ns_interval_arg.getValue();
        options.ls0_interval = ls0_interval_arg.getValue();
        options.kl_interval = kl_interval_arg.getValue();
        options.ls1_interval = ls1_interval_arg.getValue();
        options.simd_width_interval = vector_interval_arg.getValue();
        options.A_fetch_method = A_fetch_method_arg.getValue();
        options.local_fetch0_interval = local_fetch0_interval.getValue();
        options.local_fetch1_interval = local_fetch1_interval.getValue();
        options.B_fetch_method = B_fetch_method_arg.getValue();

        return options;
    }
    catch (TCLAP::ArgException &e){
        std::cerr << "error: " << "\"" << e.error() << "\"" << " [for arg " << e.argId() << "]" << std::endl;
        exit(EXIT_FAILURE);
    }
}

template<class ScalarType, char TransA, char TransB>
struct config{
    typedef matrix_product<TransA, TransB> profile_type;

    static profile_type create_profile(std::map<std::string, autotune::tuning_param> const & params){
       int vector = viennacl::generator::at(params, std::string("vector")).current();
       int ls0 = viennacl::generator::at(params, std::string("local_size1")).current();
       int ls1 = viennacl::generator::at(params, std::string("local_size2")).current();
       int kl = viennacl::generator::at(params, std::string("kl")).current();
       int ms = viennacl::generator::at(params, std::string("ms")).current();
       int ks = viennacl::generator::at(params, std::string("ks")).current();
       int ns = viennacl::generator::at(params, std::string("ns")).current();

       bool A_local = (viennacl::generator::at(params, std::string("A_storage")).current()==1);
       bool B_local = (viennacl::generator::at(params, std::string("B_storage")).current()==1);

       int local_fetch0 = viennacl::generator::at(params, std::string("local_fetch0")).current();
       int local_fetch1 = viennacl::generator::at(params, std::string("local_fetch1")).current();

       profile_type res(vector,ls0,kl,ls1,ms,ks,ns,A_local,B_local,local_fetch0,local_fetch1);

       return res;
    }
    static bool is_invalid(viennacl::ocl::device const & dev, std::map<std::string, autotune::tuning_param> const & params){
        profile_type prof = create_profile(params);
        return prof.is_invalid(dev, sizeof(ScalarType));
    }
};

viennacl::generator::code_generator::forced_profile_key_type make_key(std::string const & layout, std::size_t scalartype_size){
    if(layout=="TT")
        return code_generator::forced_profile_key_type(MATRIX_PRODUCT_TT_TYPE, scalartype_size);
    else if(layout=="TN")
        return code_generator::forced_profile_key_type(MATRIX_PRODUCT_TN_TYPE, scalartype_size);
    else if(layout=="NT")
        return code_generator::forced_profile_key_type(MATRIX_PRODUCT_NT_TYPE, scalartype_size);
    else
        return code_generator::forced_profile_key_type(MATRIX_PRODUCT_NN_TYPE, scalartype_size);
}

template<class MatA, class MatB, class MatC>
viennacl::scheduler::statement make_statement(std::string const & layout, MatA const & A, MatB const & B, MatC const & C){
    if(layout=="TT")
        return viennacl::scheduler::statement(C, viennacl::op_assign(), viennacl::linalg::prod(trans(A),trans(B)));
    else if(layout=="TN")
        return viennacl::scheduler::statement(C, viennacl::op_assign(), viennacl::linalg::prod(trans(A),B));
    else if(layout=="NT")
        return viennacl::scheduler::statement(C, viennacl::op_assign(), viennacl::linalg::prod(A,trans(B)));
    else
       return  viennacl::scheduler::statement(C, viennacl::op_assign(), viennacl::linalg::prod(A,B));
}

template<typename ScalarType, char TransA, char TransB>
unsigned int run_benchmark(size_t size, std::string layout, std::size_t scalartype_size, typename config<ScalarType, TransA, TransB>::profile_type const & profile)
{
    //viennacl::ocl::current_context().build_options("-cl-mad-enable -cl-fast-relaxed-math");   //uncomment for additional optimizations
    //viennacl::ocl::current_context().build_options("-cl-opt-disable");                        //uncomment to get poor performance
    viennacl::matrix<ScalarType, viennacl::column_major> A(size, size);
    viennacl::matrix<ScalarType, viennacl::column_major> B(size, size);
    viennacl::matrix<ScalarType, viennacl::column_major> C(size, size);
    viennacl::scheduler::statement statement = make_statement(layout,A,B,C);
    viennacl::generator::code_generator gen;
    gen.add(statement,statement.array()[0]);
    gen.force_profile(make_key(layout,scalartype_size), profile);
    viennacl::generator::enqueue(gen);
    viennacl::backend::finish();
    viennacl::tools::timer timer;
    timer.start();
    for(unsigned int r = 0 ; r < n_runs ; ++r){
      viennacl::generator::enqueue(gen);
    }
    viennacl::backend::finish();
    double time = timer.get()/(double)n_runs;
    return static_cast<unsigned int>(2*pow(size/static_cast<double>(1000.0),3)/time);
}

template<class ItType>
void print_parameters(std::string const & name, ItType begin, ItType end){
    std::cout << name << " : [";
    for(ItType it = begin ; it != end ; ++it)
        std::cout << ((it==begin)?"":",") << *it;
    std::cout << "]" << std::endl;
}

template<class ScalarType, char TransA, char TransB>
void run_autotune(autotuner_options options){
    typedef std::map<double, matrix_product<TransA, TransB> > timings_t;
    typedef viennacl::matrix<ScalarType, viennacl::column_major> MatrixT;
    typedef config<ScalarType, TransA, TransB> config_type;
    typedef typename config_type::profile_type profile_type;

    viennacl::ocl::device const &  device = viennacl::ocl::current_device();

    autotune::tuning_config<config_type> conf;
    timings_t timings;
    std::list<matrix_product<TransA, TransB> > fastest_firsts;
    std::ofstream stream(options.output_name.c_str());

    std::list<std::pair<unsigned int, unsigned int> > rounds_config;
    rounds_config.push_back(std::make_pair(2400,50));

    std::vector<unsigned int> tmp;
    tmp = get_values_in_commas(options.ls0_interval); std::vector<int> ls0; for(unsigned int i=tmp[0] ; i<=tmp[1]; i*=2) ls0.push_back(i);
    tmp = get_values_in_commas(options.kl_interval); std::vector<int> kl; for(unsigned int i=tmp[0] ; i<=tmp[1]; i*=2) kl.push_back(i);
    tmp = get_values_in_commas(options.ls1_interval); std::vector<int> ls1; for(unsigned int i=tmp[0] ; i<=tmp[1]; i*=2) ls1.push_back(i);
    tmp = get_values_in_commas(options.ms_interval); std::vector<int> ms; for(unsigned int i=tmp[0] ; i<=tmp[1]; i*=2) ms.push_back(i);
    tmp = get_values_in_commas(options.ks_interval); std::vector<int> ks; for(unsigned int i=tmp[0] ; i<=tmp[1]; i*=2) ks.push_back(i);
    tmp = get_values_in_commas(options.ns_interval); std::vector<int> ns; for(unsigned int i=tmp[0] ; i<=tmp[1]; i*=2) ns.push_back(i);
    tmp = get_values_in_commas(options.simd_width_interval); std::vector<int> simd_width; for(unsigned int i=tmp[0] ; i<=tmp[1]; i*=2) simd_width.push_back(i);
    
    ms.push_back(6);
    ns.push_back(6);


    std::vector<int> A_storage;
    if(options.A_fetch_method=="global")
        A_storage.push_back(0);
    else if(options.A_fetch_method=="local")
        A_storage.push_back(1);
    else{
        A_storage.push_back(0);
        A_storage.push_back(1);
    }
    std::vector<int> B_storage;
    if(options.B_fetch_method=="global")
        B_storage.push_back(0);
    else if(options.B_fetch_method=="local")
        B_storage.push_back(1);
    else{
        B_storage.push_back(0);
        B_storage.push_back(1);
    }

    std::vector<int> local_fetch0, local_fetch1;
    if(std::find(A_storage.begin(), A_storage.end(), 1)!=A_storage.end() || std::find(B_storage.begin(), B_storage.end(), 1)!=B_storage.end()){
      tmp = get_values_in_commas(options.local_fetch0_interval); for(unsigned int i=tmp[0] ; i<=tmp[1]; i*=2) local_fetch0.push_back(i);
      tmp = get_values_in_commas(options.local_fetch1_interval); for(unsigned int i=tmp[0] ; i<=tmp[1]; i*=2) local_fetch1.push_back(i);
    }
    else{
      local_fetch0.push_back(0);
      local_fetch1.push_back(0);
    }

    
    std::cout << "-------------------" << std::endl;
    print_parameters("local size 1", ls0.begin(), ls0.end());
    print_parameters("local size 2", ls1.begin(), ls1.end());
    print_parameters("kl", kl.begin(), kl.end());
    print_parameters("ms", ms.begin(), ms.end());
    print_parameters("ks", ks.begin(), ks.end());
    print_parameters("ns", ns.begin(), ns.end());
    print_parameters("SIMD width", simd_width.begin(), simd_width.end());
    print_parameters("A fetch method", A_storage.begin(), A_storage.end());
    print_parameters("B fetch method", B_storage.begin(), B_storage.end());

    print_parameters("local fetch0", local_fetch0.begin(), local_fetch0.end());
    print_parameters("local fetch1", local_fetch1.begin(), local_fetch1.end());
    std::cout << "-------------------" << std::endl;

    conf.add_tuning_param("local_size1",ls0);
    conf.add_tuning_param("kl",kl);
    conf.add_tuning_param("local_size2",ls1);
    conf.add_tuning_param("ms",ms);
    conf.add_tuning_param("ks",ks);
    conf.add_tuning_param("ns",ns);
    conf.add_tuning_param("vector",simd_width);
    conf.add_tuning_param("A_storage",A_storage);
    conf.add_tuning_param("B_storage",B_storage);

    conf.add_tuning_param("local_fetch0",local_fetch0);
    conf.add_tuning_param("local_fetch1",local_fetch1);


    stream << "# ---- GEMM ----" << std::endl;
    stream << "#" << options.layout << " | Scalartype : " << options.scalartype << std::endl;
    stream << "#----------------------" << std::endl;
    stream << "#----------------------" << std::endl;
    stream << "#----------------------" << std::endl;
    stream << device.full_info(1,'#');
    stream << "#----------------------" << std::endl;
    stream << "#tuning for size : " << rounds_config.front().first << std::endl;

    code_generator::forced_profile_key_type key = make_key(options.layout,sizeof(ScalarType));
    for(std::list<std::pair<unsigned int, unsigned int> >::iterator it = rounds_config.begin() ; it!= rounds_config.end(); ++it){
        timings.clear();
        unsigned int k = static_cast<unsigned int>(std::distance(rounds_config.begin(),it));
        unsigned int size=it->first;
        unsigned int n_keep=it->second;
        MatrixT A(size,size);
        MatrixT B(size,size);
        MatrixT C(size,size);
        viennacl::backend::finish();
        viennacl::scheduler::statement statement = make_statement(options.layout,A,B,C);
        stream << "#time" << "," << profile_type::csv_format() << std::endl;
        if(k==0){
          autotune::benchmark(&timings,statement,key,conf,n_runs,&stream);
        }
        else{
          unsigned int n=0;
          for(typename std::list<profile_type>::const_iterator it = fastest_firsts.begin(); it!=fastest_firsts.end(); ++it){
            double percent = (double)n++*100/fastest_firsts.size();
            std::cout << '\r' << "Determining best profile for size " << size << "..." << "[" << std::setprecision(2) << std::setfill (' ') << std::setw(6) << std::fixed  << percent << "%" << "]" << std::flush;
            double exec_time = autotune::benchmark_impl(statement,key,*it,n_runs);
            timings.insert(std::make_pair(exec_time, *it));
            stream <<  std::scientific << exec_time << "," << it->csv_representation() << std::endl;
          }
          std::cout << std::endl;
        }
        fastest_firsts.clear();
        viennacl::backend::finish();
        for(typename timings_t::iterator itt = timings.begin(); itt!=timings.end() ; ++itt){
            unsigned int n = static_cast<unsigned int>(std::distance(timings.begin(),itt));
            if(n>n_keep) break;
            fastest_firsts.push_back(itt->second);
        }
        stream << "# " << " Size : " << size << " | Best : " << 2*std::pow((double)size/1000,3)/timings.begin()->first << " GFlops : " << timings.begin()->second.csv_representation() << std::endl;

        //Recompiles for the best profile
        profile_type best_profile = timings.begin()->second;
        viennacl::generator::code_generator dummy;
        dummy.add(statement,statement.array()[0]);
        dummy.force_profile(key, best_profile);
        viennacl::generator::enqueue(dummy,true);
        viennacl::backend::finish();
    }

//    stream << "#Benchmarking " << timings.begin()->second.csv_representation() << "..." << std::endl;
//    stream << "##Size\tGFLOP/s" << std::endl;
//    for(unsigned int size = 128 ; size <= 3072 ; size += 128){
//        double percent = (double)size/3072*100;
//        std::cout << '\r' << "Benchmarking..." << "[" << std::setprecision(2) << std::setfill (' ') << std::setw(6) << std::fixed  << percent << "%" << "]" << std::flush;
//        stream << "#" << size << "\t" << run_benchmark<ScalarType, TransA, TransB>(size,options.layout,sizeof(ScalarType),timings.begin()->second) << std::endl;
//    }
//    std::cout << '\r' << "Benchmarking...[100.00%]" << std::endl;
}



int main(int argc, char* argv[]){
  typedef std::vector< viennacl::ocl::platform > platforms_type;
  typedef std::vector<viennacl::ocl::device> devices_type;
  autotuner_options options = get_options(argc,argv);
  unsigned int counter=0;
  platforms_type platforms = viennacl::ocl::get_platforms();
  for (platforms_type::iterator platform_iter  = platforms.begin();
       platform_iter != platforms.end();
       ++platform_iter)
  {
    devices_type devices = platform_iter->devices(CL_DEVICE_TYPE_ALL);
    for(devices_type::iterator iter = devices.begin(); iter != devices.end(); iter++)
    {
      if(counter++==options.requested_device){
        viennacl::ocl::setup_context(counter,*iter);
        viennacl::ocl::switch_context(counter);
        viennacl::ocl::device const & device = viennacl::ocl::current_device();
        std::string device_name = device.name();
        std::transform(device_name.begin(), device_name.end(), device_name.begin(), ::tolower);
        std::replace(device_name.begin(), device_name.end(),' ', '_');
        std::cout << "-------------------" << std::endl;
        std::cout << device.info() << std::endl;
        std::cout << "Operation : GEMM" << std::endl;
        std::cout << "-------------------" << std::endl;
        std::cout << "layout : " << options.layout << std::endl;
        std::cout << "scalartype : " << options.scalartype << std::endl;
        std::cout << "-------------------" << std::endl;
        if(options.scalartype=="float"){
            if(options.layout=="NN") run_autotune<float,'N','N'>(options);
            else if(options.layout=="NT") run_autotune<float,'N','T'>(options);
            else if(options.layout=="TN") run_autotune<float,'T','N'>(options);
            else if(options.layout=="TT") run_autotune<float,'T','T'>(options);
        }
        else if(options.scalartype=="double"){
            if(options.layout=="NN") run_autotune<double,'N','N'>(options);
            else if(options.layout=="NT") run_autotune<double,'N','T'>(options);
            else if(options.layout=="TN") run_autotune<double,'T','N'>(options);
            else if(options.layout=="TT") run_autotune<double,'T','T'>(options);
        }
      }
    }
  }
  std::cout << "Autotuning Complete!" << std::endl;
  return EXIT_SUCCESS;
}
