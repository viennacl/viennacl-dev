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
        k.local_work_size(0,group_size0_);
        if(vec_infos_base* vec = dynamic_cast<vec_infos_base*>(p)){
            k.global_work_size(0,viennacl::tools::roundUpToNextMultiple<cl_uint>(vec->real_size()/(vectorization_*loop_unroll_),group_size0_));
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

        std::list<vec_infos_base *> assigned_vec;
        for(std::list<binary_vector_expression_infos_base*>::iterator it=vector_expressions_.begin(); it!= vector_expressions_.end();++it){
            if(dynamic_cast<assignment_op_infos_base*>(&(*it)->op())) assigned_vec.push_back(dynamic_cast<vec_infos_base*>(&(*it)->lhs()));
        }

        std::list<mat_infos_base *> assigned_mat;
        for(std::list<binary_matrix_expression_infos_base*>::iterator it=matrix_expressions_.begin(); it!= matrix_expressions_.end();++it){
            if(dynamic_cast<assignment_op_infos_base*>(&(*it)->op())) assigned_mat.push_back(dynamic_cast<mat_infos_base*>(&(*it)->lhs()));
        }


        std::list<gpu_scal_infos_base*> assigned_scal;
        for(std::list<binary_scalar_expression_infos_base*>::iterator it=scalar_expressions_.begin(); it!= scalar_expressions_.end();++it){
            if(dynamic_cast<assignment_op_infos_base*>(&(*it)->op())) assigned_scal.push_back(dynamic_cast<gpu_scal_infos_base*>(&(*it)->lhs()));
        }

        code_generation::utils::cache_manager<vec_infos_base> vector_cache(vectors_,assigned_vec,kss);
        code_generation::utils::cache_manager<mat_infos_base> matrix_cache(matrices_,assigned_mat,kss);
        code_generation::utils::cache_manager<gpu_scal_infos_base> scalar_cache(gpu_scalars_,assigned_scal,kss);
        vec_infos_base * first_vector =  NULL;
        mat_infos_base * first_matrix = NULL;
        if(vectors_.size()) first_vector = *vectors_.begin();
        if(matrices_.size()) first_matrix = *matrices_.begin();
        scalar_cache.fetch_entries(0);
        if(first_vector){
            kss << "unsigned int i = get_global_id(0)" ; if(n_unroll>1) kss << "*" << n_unroll; kss << ";" << std::endl;
            kss << "if(i < " << first_vector->size() << "){" << std::endl;
            kss.inc_tab();

            //Set access indices
            for(typename std::list<binary_vector_expression_infos_base*>::iterator it=vector_expressions_.begin() ; it!=vector_expressions_.end();++it){
                for(unsigned int j=0 ; j < n_unroll ; ++j){
                    (*it)->access_index(j,"i" + to_string(j),"0");
                }
            }

            //Loads into private memory
            for(unsigned int j=0 ; j<n_unroll  ; ++j){
                vector_cache.fetch_entries(j);
            }

            //Compute expressions
            for(typename std::list<binary_vector_expression_infos_base*>::iterator it=vector_expressions_.begin() ; it!=vector_expressions_.end();++it){
                for(unsigned int j=0 ; j < n_unroll ; ++j){
                    kss << (*it)->generate(j) << ";" << std::endl;
                }
            }

            //Write back to global memory
            for(unsigned int j=0 ; j<n_unroll  ; ++j){
                vector_cache.writeback_entries(j);
            }
            kss.dec_tab();
            kss << "}" << std::endl;
        }
        if(first_matrix){
            kss << "unsigned int r = get_global_id(0)/" << first_matrix->internal_size2() << ";" << std::endl;
            kss << "unsigned int c = get_global_id(0)%" << first_matrix->internal_size2() << ";" << std::endl;
            kss << "if(r < " << first_matrix->internal_size1() << "){" << std::endl;
            kss.inc_tab();

            //Set access indices
            for(typename std::list<binary_matrix_expression_infos_base*>::iterator it=matrix_expressions_.begin() ; it!=matrix_expressions_.end();++it)
                (*it)->access_index(0,"r","c");


            //Loads into private memory
            matrix_cache.fetch_entries(0);

            //Compute expressions
            for(std::list<binary_matrix_expression_infos_base*>::iterator it = matrix_expressions_.begin(); it!=matrix_expressions_.end(); ++it)
                kss << (*it)->generate(0) << ";" << std::endl;

            //Write back to global memory
            matrix_cache.writeback_entries(0);
            kss.dec_tab();
            kss << "}" << std::endl;
        }
        scalar_cache.writeback_entries(0);
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
