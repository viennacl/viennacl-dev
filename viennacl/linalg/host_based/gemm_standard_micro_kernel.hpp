#ifndef VIENNACL_LINALG_HOST_BASED_GEMM_STANDARD_MICRO_KERNEL_HPP_
#define VIENNACL_LINALG_HOST_BASED_GEMM_STANDARD_MICRO_KERNEL_HPP_

#include "viennacl/forwards.h"

namespace viennacl
{
  template<typename NumericT>
  inline void standard_micro_kernel(NumericT const *buffer_A, NumericT const *buffer_B, NumericT *buffer_C,
                                    vcl_size_t num_micro_slivers, vcl_size_t mr, vcl_size_t nr)
  {
    for (vcl_size_t l=0; l<num_micro_slivers; ++l)
    {
      for (vcl_size_t j=0; j<nr; ++j)
      {
        for (vcl_size_t i=0; i<mr; ++i)
        {
          buffer_C[i + j*mr] += buffer_A[i] * buffer_B[j];  
        }
      }
      buffer_A += mr;
      buffer_B += nr;
    }
  }
}//viennacl
#endif
