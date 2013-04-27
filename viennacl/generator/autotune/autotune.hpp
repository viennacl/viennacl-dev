#ifndef VIENNACL_GENERATOR_AUTOTUNE_HPP
#define VIENNACL_GENERATOR_AUTOTUNE_HPP

#include "viennacl/generator/autotune/benchmark-utils.hpp"
#include "ctime"
#include "viennacl/generator/forwards.h"
#include "viennacl/generator/code_generation/frontend.hpp"
#include "viennacl/ocl/kernel.hpp"
#include "viennacl/ocl/infos.hpp"
#include "iomanip"
#include "cmath"

namespace viennacl{

namespace generator{

namespace autotune{


template<class OpT, class ProfileT>
void benchmark_impl(std::map<double, ProfileT> & timings, viennacl::ocl::device const & dev, OpT const & operation, ProfileT const & prof){

    Timer t;

    unsigned int n_runs = 5;

    viennacl::generator::custom_operation op(operation);
    op.operations_manager().override_model(prof);
    viennacl::ocl::program & pgm = op.program();
    viennacl::ocl::kernel & k = pgm.get_kernel("_k0");


    //Anticipates kernel failure
    size_t max_workgroup_size = viennacl::ocl::kernel::info<CL_KERNEL_WORK_GROUP_SIZE>(k,dev);
    if(prof.local_work_size().first*prof.local_work_size().second > max_workgroup_size)  return;

    //Doesn't execute because it would likelily be a waste of time
    size_t prefered_workgroup_size_multiple = viennacl::ocl::kernel::info<CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE>(k,dev);
    if( (prof.local_work_size().first*prof.local_work_size().second) % prefered_workgroup_size_multiple > 0) return;

    op.execute();
    viennacl::backend::finish();

    double exec_time = 0;
    t.start();
    for(unsigned int n=0; n<n_runs ; ++n){
        op.execute();
        viennacl::backend::finish();
    }
    exec_time = t.get()/(float)n_runs;
    timings.insert(std::make_pair(exec_time, ProfileT(prof)));
}

template<class OpT, class ConfigT>
void benchmark(std::map<double, typename ConfigT::profile_t> & timings, OpT const & op, ConfigT & config){

    viennacl::ocl::device const & dev = viennacl::ocl::current_device();
    benchmark_impl(timings,dev,op,config.get_current());
    while(config.has_next()){
        std::cout << '.' << std::flush;
        config.update();
        benchmark_impl(timings,dev,op,config.get_current());
    }
}

template<class OpT, class ProfT>
void benchmark(std::map<double, ProfT> & timings, OpT const & op, std::list<ProfT> const & profiles){
    viennacl::ocl::device const & dev = viennacl::ocl::current_device();
    for(typename std::list<ProfT>::const_iterator it = profiles.begin(); it!=profiles.end(); ++it){
        std::cout << '.' << std::flush;
        benchmark_impl<OpT>(timings,dev,op,*it);

    }
    std::cout << std::endl;
}



}

}

}
#endif // AUTOTUNE_HPP
