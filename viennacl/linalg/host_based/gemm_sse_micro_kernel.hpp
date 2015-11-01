#ifndef VIENNACL_LINALG_HOST_BASED_GEMM_SSE_MICRO_KERNEL_HPP_
#define VIENNACL_LINALG_HOST_BASED_GEMM_SSE_MICRO_KERNEL_HPP_

#include "viennacl/linalg/host_based/common.hpp"
#include "immintrin.h"

/* register-block sizes: 
 * D := double, F := float */
#define MR_D (6)
#define NR_D (4)
#define MR_F (8)
#define NR_F (8)

/* addresses for buffer_C */
#define C0_ROW_D(a) (a*NR_D)
#define C1_ROW_D(a) (a*NR_D + NR_D/2)

#define C0_ROW_F(a) (a*NR_F)
#define C1_ROW_F(a) (a*NR_F + NR_F/2)

namespace viennacl
{
  /**
   * @brief general "dummy" template, fully specialized for supported types (double/float)
   */
  template<typename NumericT>
  inline void sse_micro_kernel(NumericT const *buffer_A, NumericT const *buffer_B, NumericT *buffer_C,
                               vcl_size_t num_micro_slivers, vcl_size_t mr, vcl_size_t nr)
  {
    assert(false && bool("called with unsupported numeric type!"));
  }

  /**
   * @brief SSE micro-kernel for floats, calculates a 8x8 block of matrix C from slivers of A and B
   */
  template<>
  inline void sse_micro_kernel<float>(float const *buffer_A, float const *buffer_B, float *buffer_C,
                                      vcl_size_t num_micro_slivers, vcl_size_t mr, vcl_size_t nr)
  {
    assert( (mr == MR_F) && (nr == NR_F) && bool("mr and nr obtained by 'get_block_sizes()' in 'matrix_operations.hpp' and given to 'avx_micro_kernel()' do not match with MR_F/NR_F defined in 'gemm_avx_micro_kernel.hpp' ") );

    __m128 xmm0 , xmm1 , xmm2 , xmm3 ;
    __m128 xmm4 , xmm5 , xmm6 , xmm7 ;
    __m128 xmm8 , xmm9 , xmm10, xmm11;
    __m128 xmm12, xmm13, xmm14, xmm15;

    for (vcl_size_t l=0; l<num_micro_slivers; ++l)
    {
      xmm0 = _mm_load_ps(buffer_B+l*NR_F);
      xmm1 = _mm_load_ps(buffer_B+l*NR_F+4);
      
      xmm2 = _mm_load_ps1(buffer_A+l*MR_F);
      xmm3 = _mm_mul_ps(xmm0, xmm2);
      xmm4 = _mm_mul_ps(xmm1, xmm2);
      
      xmm2 = _mm_load_ps1(buffer_A+l*MR_F+1);
      xmm5 = _mm_mul_ps(xmm0, xmm2);
      xmm6 = _mm_mul_ps(xmm1, xmm2);
      
      xmm2 = _mm_load_ps1(buffer_A+l*MR_F+2);
      xmm7 = _mm_mul_ps(xmm0, xmm2);
      xmm8 = _mm_mul_ps(xmm1, xmm2);
      
      xmm2  = _mm_load_ps1(buffer_A+l*MR_F+3);
      xmm9  = _mm_mul_ps(xmm0, xmm2);
      xmm10 = _mm_mul_ps(xmm1, xmm2);
      
      xmm2  = _mm_load_ps1(buffer_A+l*MR_F+4);
      xmm11 = _mm_mul_ps(xmm0, xmm2);
      xmm12 = _mm_mul_ps(xmm1, xmm2);
      
      xmm2  = _mm_load_ps1(buffer_A+l*MR_F+5);
      xmm13 = _mm_mul_ps(xmm0, xmm2);
      xmm14 = _mm_mul_ps(xmm1, xmm2);

      /* free registers by storing their results */
      xmm15 = _mm_load_ps(buffer_C+C0_ROW_F(0));
      xmm15 = _mm_add_ps(xmm15, xmm3);
      _mm_store_ps(buffer_C+C0_ROW_F(0), xmm15);

      xmm15 = _mm_load_ps(buffer_C+C1_ROW_F(0));
      xmm15 = _mm_add_ps(xmm15, xmm4);
      _mm_store_ps(buffer_C+C1_ROW_F(0), xmm15);

      /* continue calculating */      
      xmm2 = _mm_load_ps1(buffer_A+l*MR_F+6);
      xmm3 = _mm_mul_ps(xmm0, xmm2);
      xmm4 = _mm_mul_ps(xmm1, xmm2);
      
      /* free registers by storing their results */
      xmm15 = _mm_load_ps(buffer_C+C0_ROW_F(1));
      xmm15 = _mm_add_ps(xmm15, xmm5);
      _mm_store_ps(buffer_C+C0_ROW_F(1), xmm15);

      xmm15 = _mm_load_ps(buffer_C+C1_ROW_F(1));
      xmm15 = _mm_add_ps(xmm15, xmm6);
      _mm_store_ps(buffer_C+C1_ROW_F(1), xmm15);
      
      /* continue calculating */
      xmm2 = _mm_load_ps1(buffer_A+l*MR_F+7);
      xmm5 = _mm_mul_ps(xmm0, xmm2);
      xmm6 = _mm_mul_ps(xmm1, xmm2);
      
      /* store the rest of the results */
      xmm15 = _mm_load_ps(buffer_C+C0_ROW_F(2));
      xmm15 = _mm_add_ps(xmm15, xmm7);
      _mm_store_ps(buffer_C+C0_ROW_F(2), xmm15);

      xmm15 = _mm_load_ps(buffer_C+C1_ROW_F(2));
      xmm15 = _mm_add_ps(xmm15, xmm8);
      _mm_store_ps(buffer_C+C1_ROW_F(2), xmm15);

      xmm15 = _mm_load_ps(buffer_C+C0_ROW_F(3));
      xmm15 = _mm_add_ps(xmm15, xmm9);
      _mm_store_ps(buffer_C+C0_ROW_F(3), xmm15);

      xmm15 = _mm_load_ps(buffer_C+C1_ROW_F(3));
      xmm15 = _mm_add_ps(xmm15, xmm10);
      _mm_store_ps(buffer_C+C1_ROW_F(3), xmm15);

      xmm15 = _mm_load_ps(buffer_C+C0_ROW_F(4));
      xmm15 = _mm_add_ps(xmm15, xmm11);
      _mm_store_ps(buffer_C+C0_ROW_F(4), xmm15);

      xmm15 = _mm_load_ps(buffer_C+C1_ROW_F(4));
      xmm15 = _mm_add_ps(xmm15, xmm12);
      _mm_store_ps(buffer_C+C1_ROW_F(4), xmm15);

      xmm15 = _mm_load_ps(buffer_C+C0_ROW_F(5));
      xmm15 = _mm_add_ps(xmm15, xmm13);
      _mm_store_ps(buffer_C+C0_ROW_F(5), xmm15);

      xmm15 = _mm_load_ps(buffer_C+C1_ROW_F(5));
      xmm15 = _mm_add_ps(xmm15, xmm14);
      _mm_store_ps(buffer_C+C1_ROW_F(5), xmm15);

      xmm15 = _mm_load_ps(buffer_C+C0_ROW_F(6));
      xmm15 = _mm_add_ps(xmm15, xmm3);
      _mm_store_ps(buffer_C+C0_ROW_F(6), xmm15);

      xmm15 = _mm_load_ps(buffer_C+C1_ROW_F(6));
      xmm15 = _mm_add_ps(xmm15, xmm4);
      _mm_store_ps(buffer_C+C1_ROW_F(6), xmm15);

      xmm15 = _mm_load_ps(buffer_C+C0_ROW_F(7));
      xmm15 = _mm_add_ps(xmm15, xmm5);
      _mm_store_ps(buffer_C+C0_ROW_F(7), xmm15);

      xmm15 = _mm_load_ps(buffer_C+C1_ROW_F(7));
      xmm15 = _mm_add_ps(xmm15, xmm6);
      _mm_store_ps(buffer_C+C1_ROW_F(7), xmm15);
    }//for
  }//sse_micro_kernel()

  /**
   * @brief SSE micro-kernel for doubles, calculates a 6x4 block of matrix C from slivers of A and B
   */
  template<>
  inline void sse_micro_kernel<double>(double const *buffer_A, double const *buffer_B, double *buffer_C,
                                        vcl_size_t num_micro_slivers, vcl_size_t mr, vcl_size_t nr)
  {
    assert( (mr == MR_D) && (nr == NR_D) && bool("mr and nr obtained by 'get_block_sizes()' in 'matrix_operations.hpp' and given to 'avx_micro_kernel()' do not match with MR_D/NR_D defined in 'gemm_avx_micro_kernel.hpp' ") );

    __m128d xmm0 , xmm1 , xmm2 , xmm3 ;
    __m128d xmm4 , xmm5 , xmm6 , xmm7 ;
    __m128d xmm8 , xmm9 , xmm10, xmm11;
    __m128d xmm12, xmm13, xmm14, xmm15;

    vcl_size_t l;
    
    for (l=0; l<num_micro_slivers; ++l)
    {
      xmm0 = _mm_load_pd(buffer_B+l*NR_D);
      xmm1 = _mm_load_pd(buffer_B+l*NR_D+2);

      xmm2 = _mm_load_pd1(buffer_A+l*MR_D);
      xmm3 = _mm_mul_pd(xmm0, xmm2);
      xmm4 = _mm_mul_pd(xmm1, xmm2);
      
      xmm2 = _mm_load_pd1(buffer_A+l*MR_D+1);
      xmm5 = _mm_mul_pd(xmm0, xmm2);
      xmm6 = _mm_mul_pd(xmm1, xmm2);

      xmm2 = _mm_load_pd1(buffer_A+l*MR_D+2);
      xmm7 = _mm_mul_pd(xmm0, xmm2);
      xmm8 = _mm_mul_pd(xmm1, xmm2);

      xmm2  = _mm_load_pd1(buffer_A+l*MR_D+3);
      xmm9  = _mm_mul_pd(xmm0, xmm2);
      xmm10 = _mm_mul_pd(xmm1, xmm2);

      xmm2  = _mm_load_pd1(buffer_A+l*MR_D+4);
      xmm11 = _mm_mul_pd(xmm0, xmm2);
      xmm12 = _mm_mul_pd(xmm1, xmm2);

      xmm2  = _mm_load_pd1(buffer_A+l*MR_D+5);
      xmm13 = _mm_mul_pd(xmm0, xmm2);
      xmm14 = _mm_mul_pd(xmm1, xmm2);
    
      /* store new entries */
      xmm15 = _mm_load_pd(buffer_C+C0_ROW_D(0));
      xmm15 = _mm_add_pd(xmm15, xmm3);
      _mm_store_pd(buffer_C+C0_ROW_D(0), xmm15);
      
      xmm15 = _mm_load_pd(buffer_C+C1_ROW_D(0));
      xmm15 = _mm_add_pd(xmm15, xmm4);
      _mm_store_pd(buffer_C+C1_ROW_D(0), xmm15);

      xmm15 = _mm_load_pd(buffer_C+C0_ROW_D(1));
      xmm15 = _mm_add_pd(xmm15, xmm5);
      _mm_store_pd(buffer_C+C0_ROW_D(1), xmm15);

      xmm15 = _mm_load_pd(buffer_C+C1_ROW_D(1));
      xmm15 = _mm_add_pd(xmm15, xmm6);
      _mm_store_pd(buffer_C+C1_ROW_D(1), xmm15);

      xmm15 = _mm_load_pd(buffer_C+C0_ROW_D(2));
      xmm15 = _mm_add_pd(xmm15, xmm7);
      _mm_store_pd(buffer_C+C0_ROW_D(2), xmm15);
      
      xmm15 = _mm_load_pd(buffer_C+C1_ROW_D(2));
      xmm15 = _mm_add_pd(xmm15, xmm8);
      _mm_store_pd(buffer_C+C1_ROW_D(2), xmm15);

      xmm15 = _mm_load_pd(buffer_C+C0_ROW_D(3));
      xmm15 = _mm_add_pd(xmm15, xmm9);
      _mm_store_pd(buffer_C+C0_ROW_D(3), xmm15);

      xmm15 = _mm_load_pd(buffer_C+C1_ROW_D(3));
      xmm15 = _mm_add_pd(xmm15, xmm10);
      _mm_store_pd(buffer_C+C1_ROW_D(3), xmm15);

      xmm15 = _mm_load_pd(buffer_C+C0_ROW_D(4));
      xmm15 = _mm_add_pd(xmm15, xmm11);
      _mm_store_pd(buffer_C+C0_ROW_D(4), xmm15);

      xmm15 = _mm_load_pd(buffer_C+C1_ROW_D(4));
      xmm15 = _mm_add_pd(xmm15, xmm12);
      _mm_store_pd(buffer_C+C1_ROW_D(4), xmm15);

      xmm15 = _mm_load_pd(buffer_C+C0_ROW_D(5));
      xmm15 = _mm_add_pd(xmm15, xmm13);
      _mm_store_pd(buffer_C+C0_ROW_D(5), xmm15);

      xmm15 = _mm_load_pd(buffer_C+C1_ROW_D(5));
      xmm15 = _mm_add_pd(xmm15, xmm14);
      _mm_store_pd(buffer_C+C1_ROW_D(5), xmm15);

    }//for
  }//sse_micro_kernel<double>
}//viennacl
#endif
