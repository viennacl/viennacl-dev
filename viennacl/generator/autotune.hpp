#ifndef VIENNACL_GENERATOR_AUTOTUNE_HPP
#define VIENNACL_GENERATOR_AUTOTUNE_HPP

#include <ctime>
#include <iomanip>
#include <cmath>

#include "viennacl/generator/benchmark-utils.hpp"
#include "viennacl/generator/forwards.h"
#include "viennacl/generator/code_generation/frontend.hpp"
#include "viennacl/ocl/kernel.hpp"
#include "viennacl/ocl/infos.hpp"


namespace viennacl{

namespace generator{

namespace autotune{

struct inc{
    static void mul_by_two(unsigned int & val) { val*=2 ; }
    static void add_one(unsigned int & val) { val+=1; }
};

struct tuning_param{
public:
    tuning_param(unsigned int min, unsigned int max, void (*inc)(unsigned int &)) : current_(min), min_max_(min,max), inc_(inc){ }
    bool is_max() const { return current_ >= min_max_.second; }
    bool inc(){
        inc_(current_);
        if(current_<=min_max_.second) return false;
        current_=min_max_.first;
        return true; //has been reset
    }
    unsigned int current() const{ return current_; }
    void reset() { current_ = min_max_.first; }
private:
    unsigned int current_;
    std::pair<unsigned int, unsigned int> min_max_;
    void (*inc_)(unsigned int &);
};

template<class ConfigT>
class tuning_config{
private:
    typedef std::map<std::string, viennacl::generator::autotune::tuning_param> params_t;
public:
    typedef typename ConfigT::profile_t profile_t;
    tuning_config(){ }
    void add_tuning_param(std::string const & name, unsigned int min, unsigned int max, void (*inc)(unsigned int &)){
        params_.insert(std::make_pair(name,tuning_param(min,max,inc)));
    }
    bool has_next() const{
        bool res = false;
        for(typename params_t::const_iterator it = params_.begin() ; it != params_.end() ; ++it) res = res || !it->second.is_max();
        return res;
    }
    void update(){
        for(typename params_t::iterator it = params_.begin() ; it != params_.end() ; ++it) if(it->second.inc()==false) break;
    }
    bool is_invalid(viennacl::ocl::device const & dev) const{
        return ConfigT::is_invalid(dev,params_);
    }
    typename ConfigT::profile_t get_current(){
        return ConfigT::create_profile(params_);
    }
    void reset(){
        for(params_t::iterator it = params_.begin() ; it != params_.end() ; ++it){
            it->second.reset();
        }
    }

private:
    params_t params_;
};

template<class OpT, class ProfileT>
void benchmark_impl(std::map<double, ProfileT> & timings, viennacl::ocl::device const & dev, OpT const & operation, ProfileT const & prof){

    Timer t;

    unsigned int n_runs = 10;

    //Skips if use too much local memory.
    viennacl::generator::custom_operation op(operation);
    op.operations_manager().override_model(prof);
    viennacl::ocl::program & pgm = op.program();
    viennacl::ocl::kernel & k = pgm.get_kernel("_k0");


    //Anticipates kernel failure
    size_t max_workgroup_size = viennacl::ocl::info<CL_KERNEL_WORK_GROUP_SIZE>(k,dev);
    std::pair<size_t,size_t> work_group_sizes = prof.local_work_size();
    if(work_group_sizes.first*work_group_sizes.second > max_workgroup_size)  return;

    //Doesn't execute because it would likelily be a waste of time
    size_t prefered_workgroup_size_multiple = viennacl::ocl::info<CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE>(k,dev);
    if( (prof.local_work_size().first*prof.local_work_size().second) % prefered_workgroup_size_multiple > 0) return;

    op.execute();
    viennacl::backend::finish();

    double exec_time = 0;
    t.start();
    for(unsigned int n=0; n<n_runs ; ++n){
        op.execute();
    }
    viennacl::backend::finish();
    exec_time = t.get()/(float)n_runs;
    timings.insert(std::make_pair(exec_time, ProfileT(prof)));
}

template<class OpT, class ConfigT>
void benchmark(std::map<double, typename ConfigT::profile_t> & timings, OpT const & op, ConfigT & config){
    viennacl::ocl::device const & dev = viennacl::ocl::current_device();
    if(config.is_invalid(dev)==false)  benchmark_impl(timings,dev,op,config.get_current());

    unsigned int n=0, n_conf = 0;
    while(config.has_next()){
        config.update();
        if(config.is_invalid(dev)) continue;
        ++n_conf;
    }

    std::cout << "Benchmarking over " << n_conf << " valid kernels" << std::endl;

    config.reset();
    while(config.has_next()){
        config.update();
        if(config.is_invalid(dev)) continue;
        std::cout << '\r' << (float)(n++)*100/n_conf << "%" << std::flush;
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
