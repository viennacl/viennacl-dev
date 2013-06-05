#ifndef VIENNACL_RAND_UTILS_HPP_
#define VIENNACL_RAND_UTILS_HPP_

#ifdef VIENNACL_WITH_OPENCL
#include "viennacl/linalg/kernels/rand_kernels.h"

namespace viennacl{

namespace rand{


template<class SCALARTYPE, class DISTRIBUTION>
struct random_matrix_t{
    typedef size_t size_type;
    random_matrix_t(size_type _size1, unsigned int _size2, DISTRIBUTION const & _distribution) : size1(_size1), size2(_size2), distribution(_distribution){
        #ifdef VIENNACL_WITH_OPENCL
        viennacl::linalg::kernels::rand<SCALARTYPE,1>::init();
        #endif
    }
    size_type size1;
    size_type size2;
    DISTRIBUTION distribution;
};


template<class SCALARTYPE, class DISTRIBUTION>
struct random_vector_t{
    typedef size_t size_type;
    random_vector_t(size_type _size, DISTRIBUTION const & _distribution) : size(_size), distribution(_distribution){
        #ifdef VIENNACL_WITH_OPENCL
        viennacl::linalg::kernels::rand<SCALARTYPE,1>::init();
        #endif
    }
    size_type size;
    DISTRIBUTION distribution;
};

template<class ScalarType, class Distribution>
struct buffer_dumper;


}

}

#endif


#endif
