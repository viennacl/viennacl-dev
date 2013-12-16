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

//#define VIENNACL_DEBUG_ALL
//#define VIENNACL_DEBUG_BUILD

#include <iostream>
#include <algorithm>
#include <string>

#include "viennacl/linalg/inner_prod.hpp"
#include "viennacl/scheduler/forwards.h"
#include "viennacl/vector.hpp"
#include "viennacl/generator/generate.hpp"
#include "viennacl/generator/autotune.hpp"
#include "viennacl/linalg/norm_2.hpp"

#include "command-line-utils.hpp"

static const unsigned int n_runs = 10;

using namespace viennacl::generator;

typedef std::vector< viennacl::ocl::platform > platforms_type;
typedef std::vector<viennacl::ocl::device> devices_type;
typedef std::vector<cl_device_id> cl_devices_type;


struct autotuner_options{
    unsigned int tuning_size;

    std::string scalartype;
    std::string output_name;

    unsigned int requested_device;

    std::string vector_interval;

    std::string local_size_interval;
    std::string num_groups_interval;

    std::string decomposition;
};

autotuner_options get_options(int argc, char* argv[]){
    try{
        autotuner_options options;

        TCLAP::CmdLine cmd("GEMM Autotuner", ' ', "0.1");


        pow_2_interval_constraint pow_2_interval_cstrt;
        min_max_inc_constraint min_max_inc_cstrt;

        TCLAP::ValueArg<unsigned int> tuning_size_arg("","tuning-size","Size to use for the autotuning procedure",false,1024*1024,"unsigned int",cmd);

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

        //Vector
        TCLAP::ValueArg<std::string> vector_interval_arg("","vector","Vector type used in the kernel",false,"1,1",&pow_2_interval_cstrt,cmd);

        //Large blocks
        TCLAP::ValueArg<std::string> local_size_interval_arg("","local-size","Number of work-item in each work-group. Specify min,max both power of two.",false,"16,1024",&pow_2_interval_cstrt,cmd);
        TCLAP::ValueArg<std::string> num_groups_interval_arg("","num-groups","Number of work groups required.",false,"16,1024,16",&min_max_inc_cstrt,cmd);

        //Decomposition
        std::vector<std::string> allowed_decomposition_method;
        allowed_decomposition_method.push_back("local");
        allowed_decomposition_method.push_back("global");
        allowed_decomposition_method.push_back("all");
        TCLAP::ValuesConstraint<std::string> allowed_decomposition_method_constraint(allowed_decomposition_method);
        TCLAP::ValueArg<std::string> decomposition_method_arg("","decomposition","Work decomposition method. If set to \"local\" , the work items within a work group will access contiguous data.",false,"all",&allowed_decomposition_method_constraint,cmd);

        cmd.parse(argc,argv);
        options.tuning_size = tuning_size_arg.getValue();
        options.scalartype = scalartype_arg.getValue();
        options.output_name = output_name_arg.getValue();
        options.requested_device = requested_device_arg.getValue();
        options.vector_interval = vector_interval_arg.getValue();
        options.local_size_interval = local_size_interval_arg.getValue();
        options.num_groups_interval = num_groups_interval_arg.getValue();
        options.decomposition = decomposition_method_arg.getValue();
        return options;
    }
    catch (TCLAP::ArgException &e){
        std::cerr << "error: " << "\"" << e.error() << "\"" << " [for arg " << e.argId() << "]" << std::endl;
        exit(EXIT_FAILURE);
    }
}

template<class ScalarType>
struct config{
    typedef scalar_reduction profile_type;
    static profile_type create_profile(std::map<std::string, autotune::tuning_param> const & params){
      return profile_type(viennacl::generator::at(params, std::string("vector")).current(),
                          viennacl::generator::at(params, std::string("local_size")).current(),
                          viennacl::generator::at(params, std::string("num_groups")).current(),
                          viennacl::generator::at(params, std::string("decomposition")).current());
    }
    static bool is_invalid(viennacl::ocl::device const & dev, std::map<std::string, autotune::tuning_param> const & params){
        profile_type prof = create_profile(params);
        return prof.is_invalid(dev, sizeof(ScalarType));
    }
};

template<class ScalarType>
code_generator::forced_profile_key_type make_key(autotuner_options /*options*/){
    return code_generator::forced_profile_key_type(SCALAR_REDUCE_TYPE,sizeof(ScalarType));
}

template<class ScalarType>
viennacl::scheduler::statement make_statement(autotuner_options /*options*/, viennacl::scalar<ScalarType> const & s, viennacl::vector<ScalarType> const & x, viennacl::vector<ScalarType> const & y){
    return viennacl::scheduler::statement(s, viennacl::op_assign(), viennacl::linalg::inner_prod(x, y));
}

template<typename ScalarType>
double run_benchmark(size_t size, autotuner_options options, typename config<ScalarType>::profile_type const & profile)
{
    //viennacl::ocl::current_context().build_options("-cl-mad-enable -cl-fast-relaxed-math");   //uncomment for additional optimizations
    //viennacl::ocl::current_context().build_options("-cl-opt-disable");                        //uncomment to get poor performance
    viennacl::vector<ScalarType> y(size);
    viennacl::vector<ScalarType> x(size);
    viennacl::scalar<ScalarType> s = 0;
    viennacl::scheduler::statement statement = make_statement(options,s,x,y);
    viennacl::generator::code_generator gen;
    gen.add(statement,statement.array()[0]);
    gen.force_profile(make_key<ScalarType>(options), profile);
    viennacl::generator::enqueue(gen);
    viennacl::generator::enqueue(gen);
    viennacl::backend::finish();
    viennacl::tools::timer timer;
    timer.start();
    static const unsigned int n_runs = 1;
    for(unsigned int r = 0 ; r < n_runs; ++r)
      viennacl::generator::enqueue(gen);
    viennacl::backend::finish();
    double time = timer.get()/(double)n_runs;
    return 1e-9*2.0*static_cast<double>(size*sizeof(ScalarType))/time;
}

template<class ScalarType>
void run_autotune(autotuner_options const & options){
    typedef config<ScalarType> config_type;
    typedef typename config_type::profile_type profile_type;

    viennacl::ocl::device const &  device = viennacl::ocl::current_device();

    viennacl::vector<ScalarType> v1(options.tuning_size), v2(options.tuning_size), v3(options.tuning_size), v4(options.tuning_size);
    viennacl::backend::finish();
    autotune::tuning_config<config<ScalarType> > conf;
    std::map<double, typename config<ScalarType>::profile_type> timings;
    viennacl::scalar<ScalarType> s = 0;

    std::vector<unsigned int> tmp;
    tmp = get_values_in_commas(options.local_size_interval); std::vector<int> local_size; for(unsigned int i=tmp[0] ; i<=tmp[1]; i*=2) local_size.push_back(i);
    tmp = get_values_in_commas(options.num_groups_interval); std::vector<int> num_groups; for(unsigned int i=tmp[0] ; i<=tmp[1]; i+=tmp[2]) { num_groups.push_back(i); }
    tmp = get_values_in_commas(options.vector_interval); std::vector<int> vector; for(unsigned int i=tmp[0] ; i<=tmp[1]; i*=2) vector.push_back(i);
    std::vector<int> decomposition;
    if(options.decomposition=="global")
        decomposition.push_back(0);
    else if(options.decomposition=="local")
        decomposition.push_back(1);
    else{
        decomposition.push_back(0);
        decomposition.push_back(1);
    }

    conf.add_tuning_param("vector",vector);
    conf.add_tuning_param("local_size",local_size);
    conf.add_tuning_param("num_groups",num_groups);
    conf.add_tuning_param("decomposition", decomposition);
    std::ofstream stream(options.output_name.c_str());


    stream << "# ---- DOT AUTOTUNING ----" << std::endl;
    stream << "#" << "Scalartype : " << options.scalartype << std::endl;
    stream << "#----------------------" << std::endl;
    stream << "#----------------------" << std::endl;
    stream << "#----------------------" << std::endl;
    stream << device.full_info(1,'#');
    stream << "#----------------------" << std::endl;
    stream << "#tuning for size : " << options.tuning_size << std::endl;

    code_generator::forced_profile_key_type key(SCALAR_REDUCE_TYPE, sizeof(ScalarType));
    viennacl::scheduler::statement statement(s, viennacl::op_assign(), viennacl::linalg::inner_prod(v1, v2));
    autotune::benchmark(&timings,statement,key,conf,n_runs,&stream);

    //Recompiles for the best profile
    profile_type best_profile = timings.begin()->second;
    viennacl::generator::code_generator dummy;
    dummy.add(statement,statement.array()[0]);
    dummy.force_profile(key, best_profile);
    viennacl::generator::enqueue(dummy,true);
    viennacl::backend::finish();

    stream << "#Benchmarking " << timings.begin()->second << "..." << std::endl;
    stream << "##Size\tGB/s" << std::endl;
    for(unsigned int size = 1024 ; size <= 5e7 ; size *=2){
        double percent = (double)size/1e7*100;
        std::cout << '\r' << "Benchmarking..." << "[" << std::setprecision(2) << std::setfill (' ') << std::setw(6) << std::fixed  << percent << "%" << "]" << std::flush;
        stream << "#" << size << "\t" << run_benchmark<ScalarType>(size,options,best_profile) << std::endl;
    }
    std::cout << '\r' << "Benchmarking...[100.00%]" << std::endl;
}

int main(int argc, char* argv[]){
  typedef std::vector< viennacl::ocl::platform > platforms_type;
  typedef std::vector<viennacl::ocl::device> devices_type;
  autotuner_options options = get_options(argc,argv);
  std::size_t device_counter = 0;
  platforms_type platforms = viennacl::ocl::get_platforms();
  for (platforms_type::iterator platform_iter  = platforms.begin();
       platform_iter != platforms.end();
       ++platform_iter)
  {
    devices_type devices = platform_iter->devices(CL_DEVICE_TYPE_ALL);
    for(devices_type::iterator iter = devices.begin(); iter != devices.end(); iter++)
    {
      if(device_counter++==options.requested_device){
        viennacl::ocl::setup_context(options.requested_device,*iter);
        viennacl::ocl::switch_context(options.requested_device);
        viennacl::ocl::device const & device = viennacl::ocl::current_device();
        std::string device_name = device.name();
        std::transform(device_name.begin(), device_name.end(), device_name.begin(), ::tolower);
        std::replace(device_name.begin(), device_name.end(),' ', '_');
        std::cout << "-------------------" << std::endl;
        std::cout << device.info() << std::endl;
        std::cout << "Operation : DOT" << std::endl;
        std::cout << "-------------------" << std::endl;
        std::cout << "scalatype : " << options.scalartype << std::endl;
        std::cout << "vector : [" << options.vector_interval << "]" << std::endl;
        std::cout << "local size : [" << options.local_size_interval << "]" << std::endl;
        std::cout << "number of groups : [" << options.num_groups_interval << "]" << std::endl;
        std::cout << "decomposition : [" << options.decomposition << "]" << std::endl;
        std::cout << "tuning size : " << options.tuning_size << std::endl;
        std::cout << "-------------------" << std::endl;
        if(options.scalartype=="float")
            run_autotune<float>(options);
        else if(options.scalartype=="double")
            run_autotune<double>(options);
      }
    }
  }
  std::cout << "Autotuning complete! Check \"" << options.output_name << "\" for results." << std::endl;
}
