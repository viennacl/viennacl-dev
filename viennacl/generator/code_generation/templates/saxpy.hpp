#ifndef VIENNACL_GENERATOR_CODE_GENERATION_TEMPLATES_SAXPY_HPP
#define VIENNACL_GENERATOR_CODE_GENERATION_TEMPLATES_SAXPY_HPP

#include "viennacl/tools/tools.hpp"

#include "viennacl/generator/code_generation/optimization_profile.hpp"
#include "viennacl/generator/symbolic_types_base.hpp"

namespace viennacl{

namespace generator{

namespace code_generation{

namespace saxpy{

class profile : public optimization_profile{
public:
    profile(){
        loop_unroll_ = 1;
        group_size0_ = 128;
    }

    profile(unsigned int vectorization, unsigned int loop_unroll, size_t group_size0) : optimization_profile(vectorization){
        loop_unroll_ = loop_unroll;
        group_size0_ = group_size0;
    }

    unsigned int loop_unroll() const{
        return loop_unroll_;
    }

    std::pair<size_t,size_t> local_work_size() const{
        return std::make_pair(group_size0_,1);
    }

    void config_nd_range(viennacl::ocl::kernel & k, infos_base* p){
        //k.local_work_size(0,group_size0_);
        if(vec_infos_base* vec = dynamic_cast<vec_infos_base*>(p)){
            //k.global_work_size(0,viennacl::tools::roundUpToNextMultiple<cl_uint>(vec->real_size()/(vectorization_*loop_unroll_),group_size0_)); //Note: now using for-loop for good performance on CPU
        }
        else if(mat_infos_base * mat = dynamic_cast<mat_infos_base*>(p)){
            k.global_work_size(0,viennacl::tools::roundUpToNextMultiple<cl_uint>(mat->real_size1() * mat->real_size2()/(vectorization_*loop_unroll_),group_size0_));
        }
    }


    std::string repr() const{
        std::ostringstream oss;
        oss << "V" << vectorization_
           <<  "U" << loop_unroll_
           << "GROUP" << group_size0_;
        return oss.str();
    }
private:
    unsigned int loop_unroll_;
    unsigned int group_size0_;
};

class generator : public code_generation::generator{
public:
    generator(std::list<binary_vector_expression_infos_base* > const & vector_expressions
              ,std::list<binary_scalar_expression_infos_base *> const & scalar_expressions
              ,std::list<binary_matrix_expression_infos_base * > const & matrix_expressions
              ,profile * kernel_config): vector_expressions_(vector_expressions), matrix_expressions_(matrix_expressions), scalar_expressions_(scalar_expressions), profile_(kernel_config)
    {
        for(std::list<binary_vector_expression_infos_base*>::const_iterator it=vector_expressions.begin() ; it!= vector_expressions.end() ; ++it)
            extract_as(*it,vectors_,utils::is_type<vec_infos_base>());
        for(std::list<binary_scalar_expression_infos_base*>::const_iterator it=scalar_expressions.begin() ; it!= scalar_expressions.end() ; ++it)
            extract_as(*it,gpu_scalars_,utils::is_type<gpu_scal_infos_base>());
        for(std::list<binary_matrix_expression_infos_base*>::const_iterator it=matrix_expressions.begin() ; it!= matrix_expressions.end() ; ++it)
            extract_as(*it,matrices_,utils::is_type<mat_infos_base>());
    }


    void operator()(kernel_generation_stream& kss){

        unsigned int n_unroll = profile_->loop_unroll();

        vec_infos_base * first_vector =  NULL;
        mat_infos_base * first_matrix = NULL;
        if(vectors_.size()) first_vector = *vectors_.begin();
        if(matrices_.size()) first_matrix = *matrices_.begin();
        if(first_vector){
            kss << "uint work_per_item = max((uint) (" + first_vector->size() + " / get_global_size(0)), (uint) 1);" << std::endl;
            kss << "uint row_start = get_global_id(0) * work_per_item;" << std::endl;
            kss << "uint row_stop  = min( (uint) ((get_global_id(0) + 1) * work_per_item), (uint) " + first_vector->size() + ");" << std::endl;
            kss << "for (unsigned int i = row_start; i < row_stop; ++i) {" << std::endl;
            //kss << "unsigned int i = get_global_id(0)" ; if(n_unroll>1) kss << "*" << n_unroll; kss << ";" << std::endl;
            kss.inc_tab();

            //Set access indices
            for(typename std::list<binary_vector_expression_infos_base*>::iterator it=vector_expressions_.begin() ; it!=vector_expressions_.end();++it){
                for(unsigned int j=0 ; j < n_unroll ; ++j){
                    if(j==0)  (*it)->access_index(j,"i","0");
                    else (*it)->access_index(j,"i + " + to_string(j),"0");
                    (*it)->fetch(j,kss);
                }
            }

            //Compute expressions
            for(typename std::list<binary_vector_expression_infos_base*>::iterator it=vector_expressions_.begin() ; it!=vector_expressions_.end();++it){
                for(unsigned int j=0 ; j < n_unroll ; ++j){
                    kss << (*it)->generate(j) << ";" << std::endl;
                }
            }

            for(typename std::list<binary_vector_expression_infos_base*>::iterator it=vector_expressions_.begin() ; it!=vector_expressions_.end();++it){
                for(unsigned int j=0 ; j < n_unroll ; ++j){
                    (*it)->write_back(j,kss);
                }
            }

            kss.dec_tab();
            kss << "}" << std::endl;
        }
        if(first_matrix){
            if(first_matrix->is_rowmajor()){
                kss << "unsigned int r = get_global_id(0)/" << first_matrix->internal_size2() << ";" << std::endl;
                kss << "unsigned int c = get_global_id(0)%" << first_matrix->internal_size2() << ";" << std::endl;
            }
            else{
                kss << "unsigned int r = get_global_id(0)%" << first_matrix->internal_size1() << ";" << std::endl;
                kss << "unsigned int c = get_global_id(0)/" << first_matrix->internal_size1() << ";" << std::endl;
            }
            kss << "if(r < " << first_matrix->internal_size1() << "){" << std::endl;
            kss.inc_tab();

            //Set access indices
            for(typename std::list<binary_matrix_expression_infos_base*>::iterator it=matrix_expressions_.begin() ; it!=matrix_expressions_.end();++it){
                    (*it)->access_index(0,"r","c");
                    (*it)->fetch(0,kss);
            }

            //Compute expressions
            for(std::list<binary_matrix_expression_infos_base*>::iterator it = matrix_expressions_.begin(); it!=matrix_expressions_.end(); ++it)
                kss << (*it)->generate(0) << ";" << std::endl;

            for(typename std::list<binary_matrix_expression_infos_base*>::iterator it=matrix_expressions_.begin() ; it!=matrix_expressions_.end();++it)
                (*it)->write_back(0,kss);


            kss.dec_tab();
            kss << "}" << std::endl;
        }
    }

private:
    std::list<binary_vector_expression_infos_base* >  vector_expressions_;
    std::list<binary_matrix_expression_infos_base* >  matrix_expressions_;
    std::list<binary_scalar_expression_infos_base* >  scalar_expressions_;
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
