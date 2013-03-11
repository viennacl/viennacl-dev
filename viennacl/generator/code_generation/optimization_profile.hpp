#ifndef VIENNACL_GENERATOR_CODE_GENERATION_OPTIMIZATION_PROFILE
#define VIENNACL_GENERATOR_CODE_GENERATION_OPTIMIZATION_PROFILE

#include "viennacl/generator/symbolic_types_base.hpp"

namespace viennacl{

namespace generator{

namespace code_generation{

class optimization_profile{
protected:
    typedef unsigned int size_type;
public:

    optimization_profile() : vectorization_(1){ }

    optimization_profile(unsigned int vectorization) : vectorization_(vectorization){ }

    virtual std::string repr() const = 0;

    virtual void config_nd_range(viennacl::ocl::kernel & k, infos_base * p) = 0;

    void apply(std::list<infos_base*> & expressions){
        std::set<kernel_argument*,viennacl::generator::deref_less> kernel_arguments;
        for(std::list<infos_base*>::iterator it = expressions.begin() ; it!=expressions.end() ; ++it){
            extract_as(*it,kernel_arguments,utils::is_type<kernel_argument>());
        }
        for(std::set<kernel_argument*,viennacl::generator::deref_less>::iterator it=kernel_arguments.begin() ; it!= kernel_arguments.end() ; ++it){
            (*it)->alignment(vectorization_);
        }
    }

    unsigned int vectorization() const{ return vectorization_; }
    virtual std::pair<size_t,size_t> local_work_size() const = 0;
    virtual ~optimization_profile(){ }


protected:
    unsigned int vectorization_;
};

class generator{
public:
    virtual void operator()(utils::kernel_generation_stream& kss) = 0;
};


static std::ostream& operator<<(std::ostream& os, optimization_profile const & prof){
    os << prof.repr();
    return os;
}

}

}

}

#endif
