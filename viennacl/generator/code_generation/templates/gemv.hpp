#ifndef VIENNACL_GENERATOR_CODE_GENERATION_TEMPLATES_GEMV_HPP
#define VIENNACL_GENERATOR_CODE_GENERATION_TEMPLATES_GEMV_HPP

#include "viennacl/generator/code_generation/optimization_profile.hpp"

namespace viennacl{

namespace generator{

namespace code_generation{

namespace gemv{

class profile : public optimization_profile{
public:

    profile(){
        m_ = 2;
        k_ = 128;
        num_groups_0_ = 1024;
    }

    std::pair<size_t,size_t> local_work_size() const{
        return std::make_pair(m_,k_);
    }

    void config_nd_range(viennacl::ocl::kernel & k, infos_base* p){
        k.local_work_size(0,m_);
        k.local_work_size(1,k_);
        k.global_work_size(0,m_*num_groups_0_);
        k.global_work_size(1,k_);
    }

    unsigned int m() const { return m_; }
    unsigned int k() const { return k_; }
    unsigned int num_groups_0() const { return num_groups_0_; }

    std::string repr() const{
        std::ostringstream oss;
        oss << "V" << vectorization_;
        oss << "M" << m_;
        oss << "K" << k_;
        oss << "NG0" << num_groups_0_;
        return oss.str();
    }

private:
    unsigned int m_;
    unsigned int k_;
    unsigned int num_groups_0_;
};


class generator : public code_generation::generator{
public:
    generator(std::list<infos_base * > const & expressions
              , profile * prof): expressions_(expressions), profile_(prof)
    {
        for(std::list<infos_base*>::const_iterator it=expressions_.begin() ; it!=expressions_.end() ; ++it){
            extract_as(*it, prods_, utils::is_type<matvec_prod_infos_base>());
            extract_as(*it, gpu_scalars_,  utils::is_type<gpu_scal_infos_base>());
            extract_as(*it, matrices_, utils::is_type<mat_infos_base>());
            extract_as(*it, vectors_, utils::is_type<vec_infos_base>());
        }
    }

    void operator()(utils::kernel_generation_stream& kss){
            mat_infos_base* first_matrix = *matrices_.begin();
            vec_infos_base* first_vector = *vectors_.begin();
            matvec_prod_infos_base * first_prod = *prods_.begin();
            binary_vector_expression_infos_base * first_vec_expr = dynamic_cast<binary_vector_expression_infos_base*>(expressions_.front());
            vec_infos_base * assigned = dynamic_cast<vec_infos_base*>(&first_vec_expr->lhs());
            unsigned int m = profile_->m();
            unsigned int k = profile_->k();
            local_memory<2> lmem("block",m,k+1,first_matrix->scalartype());
            kss << "unsigned int lid0 = get_local_id(0);" << std::endl;
            kss << "unsigned int lid1 = get_local_id(1);" << std::endl;
            kss << lmem.declare() <<";" << std::endl;
            kss << "for(unsigned int r = get_global_id(0) ; r < " << first_matrix->internal_size1() << " ; r += get_global_size(0)){" << std::endl;
            kss.inc_tab();
            kss << first_matrix->scalartype() << " sum = 0;" << std::endl;
            kss << "for(unsigned int c = get_global_id(1) ; c < " << first_matrix->internal_size2() << " ; c += get_global_size(1)){" << std::endl;
            kss.inc_tab();
            kss << "sum += " << first_matrix->name() << "[c + r*" << first_matrix->internal_size2() << "]" << "*" << first_vector->name() << "[c];" << std::endl;
            kss.dec_tab();
            kss << "}" << std::endl;
            kss << lmem.access("lid0", "lid1")<< " = sum;" << std::endl;
            for(unsigned int stride = k/2 ; stride>0 ; stride /=2){
                kss << "barrier(CLK_LOCAL_MEM_FENCE); " << std::endl;
                kss <<  "if(lid1 < " << to_string(stride) << ")" << lmem.access("lid0", "lid1") <<  " += " <<   lmem.access("lid0", "lid1+" + to_string(stride)) << ";" << std::endl;
            }
            kss << "if(lid1==0)" << assigned->name() << "[r] = " << lmem.access("lid0","0") << ";" << std::endl;
            kss.dec_tab();
            kss << "}" << std::endl;
    }

private:
    std::list<infos_base* >  expressions_;
    std::set<matvec_prod_infos_base*, deref_less>  prods_;
    std::set<vec_infos_base *, viennacl::generator::deref_less >  vectors_;
    std::set<mat_infos_base *, viennacl::generator::deref_less >  matrices_;
    std::set<gpu_scal_infos_base *, viennacl::generator::deref_less > gpu_scalars_;
    profile * profile_;
};

}

}

}

}

#endif
