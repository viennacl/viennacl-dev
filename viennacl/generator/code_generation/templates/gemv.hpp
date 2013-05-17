#ifndef VIENNACL_GENERATOR_CODE_GENERATION_TEMPLATES_GEMV_HPP
#define VIENNACL_GENERATOR_CODE_GENERATION_TEMPLATES_GEMV_HPP

#include "viennacl/generator/code_generation/optimization_profile.hpp"
#include "viennacl/generator/symbolic_types.hpp"

namespace viennacl{

namespace generator{

namespace code_generation{

namespace gemv{

class profile : public optimization_profile{
public:

    profile(){
        m_ = 64;
        k_ = 16;
        num_groups_0_ = 64;
    }

    profile(unsigned int m, unsigned int k, unsigned int num_groups_0) : m_(m), k_(k), num_groups_0_(num_groups_0){ }

    std::pair<size_t,size_t> local_work_size() const{
        return std::make_pair(m_,k_);
    }

    void config_nd_range(viennacl::ocl::kernel & k, symbolic_expression_tree_base* p){
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

    bool is_invalid(viennacl::ocl::device const & dev, size_t scalartype_size){
        return optimization_profile::is_invalid(dev,m_*(k_+1)*scalartype_size)
                || vectorization_ > m_
                || vectorization_ > k_;
    }


private:
    unsigned int m_;
    unsigned int k_;
    unsigned int num_groups_0_;
};


class generator : public code_generation::generator{
public:
    generator(std::list<symbolic_binary_vector_expression_base * > const & expressions
              , profile * prof): expressions_(expressions), profile_(prof)
    {
        for(std::list<symbolic_binary_vector_expression_base*>::const_iterator it=expressions_.begin() ; it!=expressions_.end() ; ++it){
            extract_as(*it, gpu_scalars_,  utils::is_type<symbolic_pointer_scalar_base>());
            extract_as(*it, matrices_, utils::is_type<symbolic_matrix_base>());
            extract_as(*it, vectors_, utils::is_type<symbolic_vector_base>());
            extract_as(*it, prods_, utils::is_type<symbolic_matrix_vector_product_base>());
        }
    }

    void operator()(utils::kernel_generation_stream& kss){
            symbolic_matrix_base* first_matrix = *matrices_.begin();
            symbolic_matrix_vector_product_base * first_prod = *prods_.begin();
            std::string scalartype = first_matrix->scalartype();
            std::string internal_size1 = first_matrix->internal_size1();
            std::string internal_size2 = first_matrix->internal_size2();

            unsigned int m = profile_->m();
            unsigned int k = profile_->k();

            bool is_lhs_transposed = is_transposed(&first_prod->lhs());
//            bool is_lhs_row_major = first_matrix->is_rowmajor();
            std::map<symbolic_matrix_vector_product_base*, std::pair<std::string,std::pair<symbolic_local_memory<2>, symbolic_vector_base*> > > reductions;
            for(std::list<symbolic_binary_vector_expression_base*>::iterator it = expressions_.begin(); it!=expressions_.end() ; ++it){
                unsigned int id = std::distance(expressions_.begin(),it);
                symbolic_vector_base* assigned = dynamic_cast<symbolic_vector_base*>(&(*it)->lhs());
                symbolic_local_memory<2> lmem("block_"+utils::to_string(id),m,k+1,scalartype);
                std::list<symbolic_matrix_vector_product_base *>  prods;
                extract_as(*it, prods, utils::is_type<symbolic_matrix_vector_product_base>());
                assert(prods.size()==1 && "More than one product involved in the expression");
                reductions.insert(std::make_pair(*prods.begin(),std::make_pair("reduction_"+utils::to_string(id),std::make_pair(lmem,assigned))));
            }
            kss << "unsigned int lid0 = get_local_id(0);" << std::endl;
            kss << "unsigned int lid1 = get_local_id(1);" << std::endl;
            for(std::map<symbolic_matrix_vector_product_base*, std::pair<std::string,std::pair<symbolic_local_memory<2>, symbolic_vector_base*> > >::iterator it = reductions.begin() ; it != reductions.end() ; ++it){
                kss << it->second.second.first.declare() << ";" << std::endl;
            }
            if(is_lhs_transposed)
                kss << "for(unsigned int r = get_global_id(0) ; r < " << internal_size2 << " ; r += get_global_size(0)){" << std::endl;
            else
                kss << "for(unsigned int r = get_global_id(0) ; r < " << internal_size1 << " ; r += get_global_size(0)){" << std::endl;
            kss.inc_tab();

            for(std::map<symbolic_matrix_vector_product_base*, std::pair<std::string,std::pair<symbolic_local_memory<2>, symbolic_vector_base*> > >::iterator it = reductions.begin() ; it != reductions.end() ; ++it){
                symbolic_matrix_vector_product_base* prod = it->first;
                binary_op_infos_base const & op_reduce = prod->op_reduce();
                std::string const & sum_name = it->second.first;
                symbolic_local_memory<2> const & lmem = it->second.second.first;
                symbolic_vector_base * assigned = it->second.second.second;
                kss << scalartype << " " << sum_name << " = 0;" << std::endl;
                if(is_lhs_transposed)
                    kss << "for(unsigned int c = get_local_id(1) ; c < " << internal_size1 << " ; c += get_local_size(1)){" << std::endl;
                else
                    kss << "for(unsigned int c = get_local_id(1) ; c < " << internal_size2 << " ; c += get_local_size(1)){" << std::endl;
                kss.inc_tab();

                prod->lhs().access_index(0,"r","c");
                prod->rhs().access_index(0,"c","0");
                prod->fetch(0,kss);


//                for(unsigned int a=0; a<alignment;++a){
                    kss << sum_name << " = " << op_reduce.generate(sum_name,prod->symbolic_binary_vector_expression_base::generate(0)) << ";" << std::endl;
//                }

                kss.dec_tab();
                kss << "}" << std::endl;

                kss << lmem.access("lid0", "lid1")<< " = " << sum_name << ";" << std::endl;

                for(unsigned int stride = k/2 ; stride>0 ; stride /=2){
                    kss << "barrier(CLK_LOCAL_MEM_FENCE); ";
                    kss <<  "if(lid1 < " << utils::to_string(stride) << ")" << lmem.access("lid0", "lid1") <<  " = " <<   op_reduce.generate(lmem.access("lid0", "lid1"),lmem.access("lid0", "lid1+" + utils::to_string(stride))) << ";" << std::endl;
                }

                it->first->access_name(lmem.access("lid0","0"));
                assigned->access_index(0,"r","0");
                kss << "if(lid1==0)" << expressions_.front()->generate(0) << ";" << std::endl;
            }


            kss.dec_tab();
            kss << "}" << std::endl;
    }



private:
    std::list<symbolic_binary_vector_expression_base* >  expressions_;
    std::list<symbolic_matrix_vector_product_base *>  prods_;
    std::list<symbolic_vector_base *>  vectors_;
    std::list<symbolic_matrix_base *>  matrices_;
    std::list<symbolic_pointer_scalar_base *> gpu_scalars_;
    profile * profile_;
};

}

}

}

}

#endif
