#ifndef VIENNACL_GENERATOR_CODE_GENERATION_TEMPLATES_INNER_PRODUCT_HPP
#define VIENNACL_GENERATOR_CODE_GENERATION_TEMPLATES_INNER_PRODUCT_HPP

#include "viennacl/generator/symbolic_types.hpp"
#include "viennacl/generator/code_generation/optimization_profile.hpp"
#include "viennacl/generator/code_generation/utils.hpp"

namespace viennacl{

namespace generator{

namespace code_generation{

namespace inner_product{

class profile : public optimization_profile{
public:
    profile(){
        group_size_=128;
        num_groups_=256;
        vectorization_=4;
    }

    profile(unsigned int vectorization, unsigned int group_size, unsigned int num_groups) : optimization_profile(vectorization){
        group_size_ = group_size;
        num_groups_ = num_groups;
    }

    std::pair<size_t,size_t> local_work_size() const{  return std::make_pair(group_size_,1); }

    unsigned int group_size() const { return group_size_; }

    unsigned int num_groups() const { return num_groups_; }

    void config_nd_range(viennacl::ocl::kernel & k, infos_base* p){
        k.local_work_size(0,group_size_);
        k.global_work_size(0,group_size_*num_groups_);
    }

    std::string repr() const{
        std::ostringstream oss;
        oss << "V" << vectorization_
            << "GS" << group_size_
            << "NG" << num_groups_;
        return oss.str();
    }
private:
    unsigned int group_size_;
    unsigned int num_groups_;
};

class generator: public code_generation::generator{
private:
    void compute_reductions_samesize(kernel_generation_stream& kss, std::map<binary_op_infos_base const *, local_memory<1> > const & lmems){
       unsigned int size = lmems.begin()->second.size();
       for(unsigned int stride = size/2 ; stride>0 ; stride /=2){
           kss << "barrier(CLK_LOCAL_MEM_FENCE); ";
           for(std::map<binary_op_infos_base const *, local_memory<1> >::const_iterator it = lmems.begin(); it != lmems.end() ; ++it){
               kss <<  it->second.access("lid") <<  " = " << it->first->generate(it->second.access("lid"), "((lid < " + to_string(stride) + ")?" + it->second.access("lid+" + to_string(stride)) + " : 0)" ) << ";" << std::endl;
           }
       }
    }
public:
    generator(std::list<binary_scalar_expression_infos_base *> const & expressions, profile * kernel_config): expressions_(expressions), profile_(kernel_config)
    {
        for(std::list<binary_scalar_expression_infos_base*>::const_iterator it=expressions_.begin() ; it!=expressions_.end() ; ++it){
            extract_as(*it,vectors_,utils::is_type<vec_infos_base>());
            extract_as(*it,gpu_scalars_,utils::is_type<gpu_scal_infos_base>());
            extract_as(*it,inner_prods_,utils::is_type<inner_product_infos_base>());
        }
    }


    void operator()(kernel_generation_stream& kss){
        kss << "unsigned int lid = get_local_id(0);" << std::endl;
        unsigned int alignment = (*vectors_.begin())->alignment();
        bool is_computed = (*inner_prods_.begin())->is_computed();
        if(is_computed){
            std::map<binary_op_infos_base const *, local_memory<1> > local_mems;
            for( std::list<inner_product_infos_base *>::const_iterator it = inner_prods_.begin(); it != inner_prods_.end() ; ++it){
                local_memory<1> lmem = local_memory<1>((*it)->name()+"_local",profile_->group_size(),(*it)->scalartype());
                local_mems.insert(std::make_pair(&(*it)->op_reduce(),lmem));
                kss << lmem.declare() << ";" << std::endl;
                kss << lmem.access("get_local_id(0)") << " = " << (*it)->name() << "[lid];" << ";" << std::endl;
                (*it)->access_name(lmem.access("0"));
            }
            compute_reductions_samesize(kss,local_mems);
            for(std::list<binary_scalar_expression_infos_base*>::iterator it = expressions_.begin() ; it!=expressions_.end() ; ++it){
                kss << (*it)->generate(0) << ";" << std::endl;
            }
            for(std::list<inner_product_infos_base *>::iterator it=inner_prods_.begin() ; it!=inner_prods_.end();++it){
                (*it)->reset_state();
            }
        }
        else{
            for(std::list<inner_product_infos_base*>::iterator it = inner_prods_.begin() ; it!=inner_prods_.end() ; ++it){
                std::string sum_name = (*it)->name() + "_reduced";
                kss << (*it)->scalartype() << " " << sum_name << " = 0;" << std::endl;
            }
            std::string size = (*vectors_.begin())->size();
            kss << "unsigned int chunk_size = " << size << "/get_num_groups(0) ;" << std::endl;
            kss << "unsigned int chunk_start = get_group_id(0)*chunk_size;" << std::endl;
            kss << "unsigned int chunk_end = min((int)((1+get_group_id(0))*chunk_size), (int)" << size << ");" << std::endl;
            kss << "for(unsigned int i = chunk_start + get_local_id(0) ; i < chunk_end ; i += get_local_size(0)){" << std::endl;
            kss.inc_tab();

            //Set access index
            for(std::list<inner_product_infos_base*>::iterator it = inner_prods_.begin() ; it!=inner_prods_.end() ; ++it){
                (*it)->access_index(0,"i","0");
                (*it)->fetch(0,kss);
            }
            for(std::list<inner_product_infos_base*>::iterator it=inner_prods_.begin() ; it!=inner_prods_.end();++it){
                    std::string sum_name = (*it)->name() + "_reduced";
                    for(unsigned int a=0; a<alignment;++a){
                        kss << sum_name << " = " << (*it)->op_reduce().generate(sum_name, (*it)->binary_scalar_expression_infos_base::generate(0,a)) << ";" << std::endl;
                    }
            }
            kss.dec_tab();
            kss << "}" << std::endl;
            std::map<binary_op_infos_base const *, local_memory<1> > local_mems;
            for( std::list<inner_product_infos_base *>::const_iterator it = inner_prods_.begin(); it != inner_prods_.end() ; ++it){
                std::string sum_name = (*it)->name() + "_reduced";
                local_memory<1> lmem = local_memory<1>((*it)->name()+"_local",profile_->group_size(),(*it)->scalartype());
                local_mems.insert(std::make_pair(&(*it)->op_reduce(),lmem));
                kss << lmem.declare() << ";" << std::endl;
                kss << lmem.access("lid") << " = " << sum_name << ";" << std::endl;
            }
            compute_reductions_samesize(kss,local_mems);
            for(std::list<inner_product_infos_base *>::iterator it=inner_prods_.begin() ; it!=inner_prods_.end();++it){
                (*it)->set_computed();
                kss << "if(lid==0) " << (*it)->name() << "[get_group_id(0)]" << "=" << (*it)->name()+"_local" << "[0]" << ";" << std::endl;
            }
        }
    }

private:
    std::list<binary_scalar_expression_infos_base* >  expressions_;
    std::list<inner_product_infos_base*>  inner_prods_;
    std::list<vec_infos_base *>  vectors_;
    std::list<gpu_scal_infos_base *> gpu_scalars_;
    profile * profile_;
};

}

}

}

}

#endif
