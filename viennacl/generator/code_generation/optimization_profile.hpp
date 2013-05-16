#ifndef VIENNACL_GENERATOR_CODE_GENERATION_OPTIMIZATION_PROFILE
#define VIENNACL_GENERATOR_CODE_GENERATION_OPTIMIZATION_PROFILE

#include "viennacl/generator/forwards.h"
#include <list>
#include <set>

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
    unsigned int vectorization() const{ return vectorization_; }
    virtual std::pair<size_t,size_t> local_work_size() const = 0;
    virtual ~optimization_profile(){ }


protected:
    unsigned int vectorization_;
};

class generator{
public:
    virtual void operator()(kernel_generation_stream& kss) = 0;
    virtual ~generator(){ }
};


static std::ostream& operator<<(std::ostream& os, optimization_profile const & prof){
    os << prof.repr();
    return os;
}

}

}

}

#endif
