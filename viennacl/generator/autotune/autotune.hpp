#ifndef VIENNACL_GENERATOR_AUTOTUNE_HPP
#define VIENNACL_GENERATOR_AUTOTUNE_HPP

#include "viennacl/generator/autotune/benchmark-utils.hpp"
#include "ctime"
#include "viennacl/generator/forwards.h"
#include "viennacl/generator/code_generation/frontend.hpp"
#include "viennacl/ocl/infos.hpp"
#include "iomanip"
#include "cmath"

namespace viennacl{

namespace generator{

namespace autotune{

typedef std::map<double, viennacl::tools::shared_ptr<viennacl::generator::code_generation::optimization_profile> > timings_t;

template<class OpT>
void benchmark_blas3_profile(timings_t & timings, viennacl::ocl::device const & dev, OpT const & operation, viennacl::generator::code_generation::blas3_optimization_profile const & prof){


    bool lhs_storage = prof.use_LHS_shared(); bool rhs_storage = prof.use_RHS_shared();
    unsigned int ml = prof.ml(); unsigned int ms = prof.ms();
    unsigned int kl = prof.kl(); unsigned int ks = prof.ks();
    unsigned int nl = prof.nl(); unsigned int ns = prof.ns();
    unsigned int alignment = prof.alignment();
    unsigned int unroll = prof.unroll();




    std::ostringstream oss;
    viennacl::generator::custom_operation op(operation);
    op.operations_manager().override_blas3_model(prof);

    matrix_expression_infos_base * expr = static_cast<matrix_expression_infos_base *>(op.kernels_list().front().trees().front());
    matrix_expression_infos_base * prod = static_cast<matrix_expression_infos_base *>(&expr->rhs());
    mat_infos_base * lhs = static_cast<mat_infos_base*>(&prod->lhs());
    mat_infos_base * rhs = static_cast<mat_infos_base*>(&prod->rhs());


    unsigned int n_runs = 2;

    if(alignment>ms || alignment>ks || alignment>ns) return;
    if(unroll > kl/ks) return;

    unsigned int lmem_size = 0;
    if(lhs_storage){
        lmem_size += (kl+1)*(ml+1)*lhs->scalartype_size();
    }

    if(rhs_storage){
        lmem_size += (nl+1)*(kl+1)*rhs->scalartype_size();
    }

    if( lmem_size > viennacl::ocl::info<CL_DEVICE_LOCAL_MEM_SIZE>(viennacl::ocl::current_device().id()) ) return;
    if(prof.local_work_size().first*prof.local_work_size().second > dev.max_workgroup_size()) return;


    viennacl::ocl::program & pgm = op.program();
    viennacl::ocl::kernel & k = pgm.get_kernel("_k0");


    //Anticipates kernel failure
    size_t max_workgroup_size = viennacl::ocl::kernel::info<CL_KERNEL_WORK_GROUP_SIZE>(k,dev);
    if(prof.local_work_size().first*prof.local_work_size().second > max_workgroup_size)
        return;

    //Doesn't execute because it would likelily be a waste of time
    size_t prefered_workgroup_size_multiple = viennacl::ocl::kernel::info<CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE>(k,dev);
    if( (prof.local_work_size().first*prof.local_work_size().second) % prefered_workgroup_size_multiple > 0)
        return;

    op.execute();
    viennacl::ocl::get_queue().finish();
    op.execute();
    viennacl::ocl::get_queue().finish();


    double exec_time = 0;
    for(unsigned int n=0; n<n_runs ; ++n){
        op.execute();
        Timer t;
        t.start();
        viennacl::ocl::get_queue().finish();
        exec_time+=t.get();
    }
    exec_time = exec_time/n_runs;

    if(exec_time < 1e-4){
        std::cout << prof << " " << exec_time << std::endl;
        exit(EXIT_FAILURE);
    }
    timings.insert(std::make_pair(exec_time, new viennacl::generator::code_generation::blas3_optimization_profile(prof)));
}

template<class OpT, class ConfigT>
void benchmark_blas3(timings_t & timings, OpT const & op, ConfigT const & config){

    viennacl::ocl::device const & dev = viennacl::ocl::current_device();

    float total = std::log(config.ml_max/config.ml_min)/std::log(2)+1;
    total*=std::log(config.kl_max/config.kl_min)/std::log(2)+1;
    total*=std::log(config.nl_max/config.nl_min)/std::log(2)+1;
    total*=std::log(config.ms_max/config.ms_min)/std::log(2)+1;
    total*=std::log(config.ks_max/config.ks_min)/std::log(2)+1;
    total*=std::log(config.ns_max/config.ns_min)/std::log(2)+1;
    total*=std::log(config.alignment_max/config.alignment_min)/std::log(2)+1;
    total*=config.LHS_storages.size();
    total*=config.RHS_storages.size();

    float perc=0;
    float prev_perc;

    for(unsigned int ml = config.ml_min ; ml <= config.ml_max; ml*=2){
        for(unsigned int kl = config.kl_min ; kl <= config.kl_max; kl*=2){
            for(unsigned int nl = config.nl_min ; nl <= config.nl_max; nl*=2){
                for(unsigned int ms = config.ms_min ; ms <= config.ms_max; ms*=2){
                    for(unsigned int ks = config.ks_min ; ks <= config.ks_max; ks*=2){
                        for(unsigned int ns = config.ns_min ; ns <= config.ns_max; ns*=2){
                            for(unsigned int alignment = config.alignment_min ; alignment <= config.alignment_max; alignment *=2){
                                for(std::vector<bool>::const_iterator lhs_storage = config.LHS_storages.begin(); lhs_storage!=config.LHS_storages.end(); ++lhs_storage){
                                    for(std::vector<bool>::const_iterator rhs_storage = config.RHS_storages.begin(); rhs_storage!=config.RHS_storages.end(); ++rhs_storage){
                                        for(unsigned int unroll = config.min_unroll; unroll < config.max_unroll ; unroll *= 2){
                                            std::cout << '.' << std::flush;
                                            prev_perc=perc;
                                            perc += (float)100/total;
                                            if((int)prev_perc!=(int)perc) std::cout << '\n' << perc << "%" << std::endl;
                                            viennacl::generator::code_generation::blas3_optimization_profile prof(ml,kl,nl,ms,ks,ns,*lhs_storage,*rhs_storage,alignment,unroll);
                                            benchmark_blas3_profile(timings,dev,op,prof);
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    std::cout << std::endl;
}

template<class OpT>
void benchmark_blas3(timings_t & timings, OpT const & op, std::list<viennacl::generator::code_generation::blas3_optimization_profile> const & profiles){
    viennacl::ocl::device const & dev = viennacl::ocl::current_device();
    float perc=0;
    float prev_perc;
    for(std::list<viennacl::generator::code_generation::blas3_optimization_profile>::const_iterator it = profiles.begin(); it!=profiles.end(); ++it){

        //        std::cout << '.' << std::flush;
        prev_perc=perc;
        perc += (float)100/profiles.size();
        if((int)prev_perc != (int)perc) std::cout << '\r' << perc << "%" << std::flush ;
        benchmark_blas3_profile<OpT>(timings,dev,op,*it);

    }
    std::cout << std::endl;
}



}

}

}
#endif // AUTOTUNE_HPP
