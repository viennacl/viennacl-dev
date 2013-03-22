#ifndef VIENNACL_RAND_UNIFORM_HPP_
#define VIENNACL_RAND_UNIFORM_HPP_

#include "viennacl/backend/mem_handle.hpp"
#include "viennacl/rand/utils.hpp"
#include "viennacl/linalg/kernels/rand_kernels.h"


namespace viennacl{

namespace rand{

struct uniform_tag{
    uniform_tag(unsigned int _a = 0, unsigned int _b = 1) : a(_a), b(_b){ }
    float a;
    float b;
};

template<class ScalarType>
struct buffer_dumper<ScalarType, uniform_tag>{
  static void dump(viennacl::backend::mem_handle const & buff, uniform_tag tag, cl_uint start, cl_uint size){
    viennacl::ocl::kernel & k = viennacl::ocl::get_kernel(viennacl::linalg::kernels::rand<ScalarType,1>::program_name(),"dump_uniform");
    k.global_work_size(0, viennacl::tools::roundUpToNextMultiple<unsigned int>(size,k.local_work_size(0)));
    viennacl::ocl::enqueue(k(buff.opencl_handle(), start, size, cl_float(tag.a), cl_float(tag.b) , cl_uint(time(0))));
  }
};



}

}

#endif
