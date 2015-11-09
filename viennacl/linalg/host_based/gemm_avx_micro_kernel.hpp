#ifndef VIENNACL_LINALG_HOST_BASED_GEMM_AVX_MICRO_KERNEL_HPP_
#define VIENNACL_LINALG_HOST_BASED_GEMM_AVX_MICRO_KERNEL_HPP_

#include "viennacl/linalg/host_based/common.hpp"
#include "immintrin.h"

/* register-block sizes: 
 * D := double, F := float */
#define MR_D ( 6)
#define NR_D ( 8)
#define MR_F ( 8)
#define NR_F (16)

/* addresses for buffer_C, 
 * where C0 is the right block and C1 is the left block of the whole partial result */
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
  inline void avx_micro_kernel(NumericT const *buffer_A, NumericT const *buffer_B, NumericT *buffer_C,
                               vcl_size_t num_micro_slivers, vcl_size_t mr, vcl_size_t nr)
  {
    assert(false && bool("called with unsupported numeric type!"));
  }

  /**
   * @brief AVX micro-kernel for floats, calculates a 8x16 block of matrix C from slivers of A and B
   */
  template<>
  inline void avx_micro_kernel<float>(float const *buffer_A, float const *buffer_B, float *buffer_C,
                                      vcl_size_t num_micro_slivers, vcl_size_t mr, vcl_size_t nr)
  {
    assert( (mr == MR_F) && (nr == NR_F) && bool("mr and nr obtained by 'get_block_sizes()' in 'matrix_operations.hpp' and given to 'avx_micro_kernel()' do not match with MR_F/NR_F defined in 'gemm_avx_micro_kernel.hpp' ") );

    __m256 ymm0 , ymm1 , ymm2 , ymm3 ;
    __m256 ymm4 , ymm5 , ymm6 , ymm7 ;
    __m256 ymm8 , ymm9 , ymm10, ymm11;
    __m256 ymm12, ymm13, ymm14, ymm15;

    /* extension to 8x16 did not yield any speedup for sizes 1000 to 2000 on amdtestbox :( */
    
    for (vcl_size_t l=0; l<num_micro_slivers; ++l)
    {
      ymm0 = _mm256_load_ps(buffer_B+l*NR_F);
      ymm1 = _mm256_load_ps(buffer_B+l*NR_F+8);
      
      ymm2 = _mm256_broadcast_ss(buffer_A+l*MR_F);
      ymm3 = _mm256_mul_ps(ymm0, ymm2);
      ymm4 = _mm256_mul_ps(ymm1, ymm2);
      
      ymm2 = _mm256_broadcast_ss(buffer_A+l*MR_F+1);
      ymm5 = _mm256_mul_ps(ymm0, ymm2);
      ymm6 = _mm256_mul_ps(ymm1, ymm2);
      
      ymm2 = _mm256_broadcast_ss(buffer_A+l*MR_F+2);
      ymm7 = _mm256_mul_ps(ymm0, ymm2);
      ymm8 = _mm256_mul_ps(ymm1, ymm2);
      
      ymm2  = _mm256_broadcast_ss(buffer_A+l*MR_F+3);
      ymm9  = _mm256_mul_ps(ymm0, ymm2);
      ymm10 = _mm256_mul_ps(ymm1, ymm2);
      
      ymm2  = _mm256_broadcast_ss(buffer_A+l*MR_F+4);
      ymm11 = _mm256_mul_ps(ymm0, ymm2);
      ymm12 = _mm256_mul_ps(ymm1, ymm2);
      
      ymm2  = _mm256_broadcast_ss(buffer_A+l*MR_F+5);
      ymm13 = _mm256_mul_ps(ymm0, ymm2);
      ymm14 = _mm256_mul_ps(ymm1, ymm2);

      /* free registers by storing their results */
      ymm15 = _mm256_load_ps(buffer_C+C0_ROW_F(0));
      ymm15 = _mm256_add_ps(ymm15, ymm3);
      _mm256_store_ps(buffer_C+C0_ROW_F(0), ymm15);

      ymm15 = _mm256_load_ps(buffer_C+C1_ROW_F(0));
      ymm15 = _mm256_add_ps(ymm15, ymm4);
      _mm256_store_ps(buffer_C+C1_ROW_F(0), ymm15);

      /* continue calculating */      
      ymm2 = _mm256_broadcast_ss(buffer_A+l*MR_F+6);
      ymm3 = _mm256_mul_ps(ymm0, ymm2);
      ymm4 = _mm256_mul_ps(ymm1, ymm2);
      
      /* free registers by storing their results */
      ymm15 = _mm256_load_ps(buffer_C+C0_ROW_F(1));
      ymm15 = _mm256_add_ps(ymm15, ymm5);
      _mm256_store_ps(buffer_C+C0_ROW_F(1), ymm15);

      ymm15 = _mm256_load_ps(buffer_C+C1_ROW_F(1));
      ymm15 = _mm256_add_ps(ymm15, ymm6);
      _mm256_store_ps(buffer_C+C1_ROW_F(1), ymm15);
      
      /* continue calculating */
      ymm2 = _mm256_broadcast_ss(buffer_A+l*MR_F+7);
      ymm5 = _mm256_mul_ps(ymm0, ymm2);
      ymm6 = _mm256_mul_ps(ymm1, ymm2);
      
      /* store the rest of the results */
      ymm15 = _mm256_load_ps(buffer_C+C0_ROW_F(2));
      ymm15 = _mm256_add_ps(ymm15, ymm7);
      _mm256_store_ps(buffer_C+C0_ROW_F(2), ymm15);

      ymm15 = _mm256_load_ps(buffer_C+C1_ROW_F(2));
      ymm15 = _mm256_add_ps(ymm15, ymm8);
      _mm256_store_ps(buffer_C+C1_ROW_F(2), ymm15);

      ymm15 = _mm256_load_ps(buffer_C+C0_ROW_F(3));
      ymm15 = _mm256_add_ps(ymm15, ymm9);
      _mm256_store_ps(buffer_C+C0_ROW_F(3), ymm15);

      ymm15 = _mm256_load_ps(buffer_C+C1_ROW_F(3));
      ymm15 = _mm256_add_ps(ymm15, ymm10);
      _mm256_store_ps(buffer_C+C1_ROW_F(3), ymm15);

      ymm15 = _mm256_load_ps(buffer_C+C0_ROW_F(4));
      ymm15 = _mm256_add_ps(ymm15, ymm11);
      _mm256_store_ps(buffer_C+C0_ROW_F(4), ymm15);

      ymm15 = _mm256_load_ps(buffer_C+C1_ROW_F(4));
      ymm15 = _mm256_add_ps(ymm15, ymm12);
      _mm256_store_ps(buffer_C+C1_ROW_F(4), ymm15);

      ymm15 = _mm256_load_ps(buffer_C+C0_ROW_F(5));
      ymm15 = _mm256_add_ps(ymm15, ymm13);
      _mm256_store_ps(buffer_C+C0_ROW_F(5), ymm15);

      ymm15 = _mm256_load_ps(buffer_C+C1_ROW_F(5));
      ymm15 = _mm256_add_ps(ymm15, ymm14);
      _mm256_store_ps(buffer_C+C1_ROW_F(5), ymm15);

      ymm15 = _mm256_load_ps(buffer_C+C0_ROW_F(6));
      ymm15 = _mm256_add_ps(ymm15, ymm3);
      _mm256_store_ps(buffer_C+C0_ROW_F(6), ymm15);

      ymm15 = _mm256_load_ps(buffer_C+C1_ROW_F(6));
      ymm15 = _mm256_add_ps(ymm15, ymm4);
      _mm256_store_ps(buffer_C+C1_ROW_F(6), ymm15);

      ymm15 = _mm256_load_ps(buffer_C+C0_ROW_F(7));
      ymm15 = _mm256_add_ps(ymm15, ymm5);
      _mm256_store_ps(buffer_C+C0_ROW_F(7), ymm15);

      ymm15 = _mm256_load_ps(buffer_C+C1_ROW_F(7));
      ymm15 = _mm256_add_ps(ymm15, ymm6);
      _mm256_store_ps(buffer_C+C1_ROW_F(7), ymm15);

    }
  }

  /**
   * @brief AVX micro-kernel for doubles, calculates a 6x8 block of matrix C from slivers of A and B
   */
  template<>
  inline void avx_micro_kernel<double>(double const *buffer_A, double const *buffer_B, double *buffer_C,
                                       vcl_size_t num_micro_slivers, vcl_size_t mr, vcl_size_t nr)
  {
    assert( (mr == MR_D) && (nr == NR_D) && bool("mr and nr obtained by 'get_block_sizes()' in 'matrix_operations.hpp' and given to 'avx_micro_kernel()' do not match with MR_D/NR_D defined in 'gemm_avx_micro_kernel.hpp' ") );

    __m256d ymm0 , ymm1 , ymm2 , ymm3 ;
    __m256d ymm4 , ymm5 , ymm6 , ymm7 ;
    __m256d ymm8 , ymm9 , ymm10, ymm11;
    __m256d ymm12, ymm13, ymm14, ymm15;

    vcl_size_t l;
    
    for (l=0; l<(num_micro_slivers); ++l)
    {
      /* UNROLLING UNDONE, STILL IN COMMENTS 
       * TO RE-ENABLE IT, SIMPLY UNCOMMENT AND REPLACE 'l' with  '4*l'
       * IN FIRST ITERATION (UNROLL0) AND DIVIDE FOR INDEX BY 4 */
      /* UNROLL 0 */
      ymm0 = _mm256_load_pd(buffer_B+l*NR_D);
      ymm1 = _mm256_load_pd(buffer_B+l*NR_D+4);

      ymm2 = _mm256_broadcast_sd(buffer_A+l*MR_D+0);
      ymm3 = _mm256_mul_pd(ymm0, ymm2);
      ymm4 = _mm256_mul_pd(ymm1, ymm2);
      
      ymm2 = _mm256_broadcast_sd(buffer_A+l*MR_D+1);
      ymm5 = _mm256_mul_pd(ymm0, ymm2);
      ymm6 = _mm256_mul_pd(ymm1, ymm2);

      ymm2 = _mm256_broadcast_sd(buffer_A+l*MR_D+2);
      ymm7 = _mm256_mul_pd(ymm0, ymm2);
      ymm8 = _mm256_mul_pd(ymm1, ymm2);

      ymm2  = _mm256_broadcast_sd(buffer_A+l*MR_D+3);
      ymm9  = _mm256_mul_pd(ymm0, ymm2);
      ymm10 = _mm256_mul_pd(ymm1, ymm2);

      ymm2  = _mm256_broadcast_sd(buffer_A+l*MR_D+4);
      ymm11 = _mm256_mul_pd(ymm0, ymm2);
      ymm12 = _mm256_mul_pd(ymm1, ymm2);

      ymm2  = _mm256_broadcast_sd(buffer_A+l*MR_D+5);
      ymm13 = _mm256_mul_pd(ymm0, ymm2);
      ymm14 = _mm256_mul_pd(ymm1, ymm2);
    
      /* store new entries */
      ymm15 = _mm256_load_pd(buffer_C+C0_ROW_D(0));
      ymm15 = _mm256_add_pd(ymm15, ymm3);
      _mm256_store_pd(buffer_C+C0_ROW_D(0), ymm15);
      
      ymm15 = _mm256_load_pd(buffer_C+C1_ROW_D(0));
      ymm15 = _mm256_add_pd(ymm15, ymm4);
      _mm256_store_pd(buffer_C+C1_ROW_D(0), ymm15);

      ymm15 = _mm256_load_pd(buffer_C+C0_ROW_D(1));
      ymm15 = _mm256_add_pd(ymm15, ymm5);
      _mm256_store_pd(buffer_C+C0_ROW_D(1), ymm15);

      ymm15 = _mm256_load_pd(buffer_C+C1_ROW_D(1));
      ymm15 = _mm256_add_pd(ymm15, ymm6);
      _mm256_store_pd(buffer_C+C1_ROW_D(1), ymm15);

      ymm15 = _mm256_load_pd(buffer_C+C0_ROW_D(2));
      ymm15 = _mm256_add_pd(ymm15, ymm7);
      _mm256_store_pd(buffer_C+C0_ROW_D(2), ymm15);
      
      ymm15 = _mm256_load_pd(buffer_C+C1_ROW_D(2));
      ymm15 = _mm256_add_pd(ymm15, ymm8);
      _mm256_store_pd(buffer_C+C1_ROW_D(2), ymm15);

      ymm15 = _mm256_load_pd(buffer_C+C0_ROW_D(3));
      ymm15 = _mm256_add_pd(ymm15, ymm9);
      _mm256_store_pd(buffer_C+C0_ROW_D(3), ymm15);

      ymm15 = _mm256_load_pd(buffer_C+C1_ROW_D(3));
      ymm15 = _mm256_add_pd(ymm15, ymm10);
      _mm256_store_pd(buffer_C+C1_ROW_D(3), ymm15);

      ymm15 = _mm256_load_pd(buffer_C+C0_ROW_D(4));
      ymm15 = _mm256_add_pd(ymm15, ymm11);
      _mm256_store_pd(buffer_C+C0_ROW_D(4), ymm15);
      
      ymm15 = _mm256_load_pd(buffer_C+C1_ROW_D(4));
      ymm15 = _mm256_add_pd(ymm15, ymm12);
      _mm256_store_pd(buffer_C+C1_ROW_D(4), ymm15);

      ymm15 = _mm256_load_pd(buffer_C+C0_ROW_D(5));
      ymm15 = _mm256_add_pd(ymm15, ymm13);
      _mm256_store_pd(buffer_C+C0_ROW_D(5), ymm15);

      ymm15 = _mm256_load_pd(buffer_C+C1_ROW_D(5));
      ymm15 = _mm256_add_pd(ymm15, ymm14);
      _mm256_store_pd(buffer_C+C1_ROW_D(5), ymm15);

      /* UNROLL 1 
         ymm0 = _mm256_load_pd(buffer_B+1*NR_D+4*l*NR_D);
         ymm1 = _mm256_load_pd(buffer_B+1*NR_D+4*l*NR_D+4);

         ymm2 = _mm256_broadcast_sd(buffer_A+1*MR_D+4*l*MR_D);
         ymm3 = _mm256_mul_pd(ymm0, ymm2);
         ymm4 = _mm256_mul_pd(ymm1, ymm2);
      
         ymm2 = _mm256_broadcast_sd(buffer_A+1*MR_D+4*l*MR_D+1);
         ymm5 = _mm256_mul_pd(ymm0, ymm2);
         ymm6 = _mm256_mul_pd(ymm1, ymm2);

         ymm2 = _mm256_broadcast_sd(buffer_A+1*MR_D+4*l*MR_D+2);
         ymm7 = _mm256_mul_pd(ymm0, ymm2);
         ymm8 = _mm256_mul_pd(ymm1, ymm2);

         ymm2  = _mm256_broadcast_sd(buffer_A+1*MR_D+4*l*MR_D+3);
         ymm9  = _mm256_mul_pd(ymm0, ymm2);
         ymm10 = _mm256_mul_pd(ymm1, ymm2);
    
         store new entries
         ymm15 = _mm256_load_pd(buffer_C+C0_ROW_D(0));
         ymm15 = _mm256_add_pd(ymm15, ymm3);
         _mm256_store_pd(buffer_C+C0_ROW_D(0), ymm15);
      
         ymm15 = _mm256_load_pd(buffer_C+C1_ROW_D(0));
         ymm15 = _mm256_add_pd(ymm15, ymm4);
         _mm256_store_pd(buffer_C+C1_ROW_D(0), ymm15);

         ymm15 = _mm256_load_pd(buffer_C+C0_ROW_D(1));
         ymm15 = _mm256_add_pd(ymm15, ymm5);
         _mm256_store_pd(buffer_C+C0_ROW_D(1), ymm15);

         ymm15 = _mm256_load_pd(buffer_C+C1_ROW_D(1));
         ymm15 = _mm256_add_pd(ymm15, ymm6);
         _mm256_store_pd(buffer_C+C1_ROW_D(1), ymm15);

         ymm15 = _mm256_load_pd(buffer_C+C0_ROW_D(2));
         ymm15 = _mm256_add_pd(ymm15, ymm7);
         _mm256_store_pd(buffer_C+C0_ROW_D(2), ymm15);
      
         ymm15 = _mm256_load_pd(buffer_C+C1_ROW_D(2));
         ymm15 = _mm256_add_pd(ymm15, ymm8);
         _mm256_store_pd(buffer_C+C1_ROW_D(2), ymm15);

         ymm15 = _mm256_load_pd(buffer_C+C0_ROW_D(3));
         ymm15 = _mm256_add_pd(ymm15, ymm9);
         _mm256_store_pd(buffer_C+C0_ROW_D(3), ymm15);

         ymm15 = _mm256_load_pd(buffer_C+C1_ROW_D(3));
         ymm15 = _mm256_add_pd(ymm15, ymm10);
         _mm256_store_pd(buffer_C+C1_ROW_D(3), ymm15);

         UNROLL 2
         ymm0 = _mm256_load_pd(buffer_B+2*NR_D+4*l*NR_D);
         ymm1 = _mm256_load_pd(buffer_B+2*NR_D+4*l*NR_D+4);

         ymm2 = _mm256_broadcast_sd(buffer_A+2*MR_D+4*l*MR_D);
         ymm3 = _mm256_mul_pd(ymm0, ymm2);
         ymm4 = _mm256_mul_pd(ymm1, ymm2);
      
         ymm2 = _mm256_broadcast_sd(buffer_A+2*MR_D+4*l*MR_D+1);
         ymm5 = _mm256_mul_pd(ymm0, ymm2);
         ymm6 = _mm256_mul_pd(ymm1, ymm2);

         ymm2 = _mm256_broadcast_sd(buffer_A+2*MR_D+4*l*MR_D+2);
         ymm7 = _mm256_mul_pd(ymm0, ymm2);
         ymm8 = _mm256_mul_pd(ymm1, ymm2);

         ymm2  = _mm256_broadcast_sd(buffer_A+2*MR_D+4*l*MR_D+3);
         ymm9  = _mm256_mul_pd(ymm0, ymm2);
         ymm10 = _mm256_mul_pd(ymm1, ymm2);
    
         store new entries
         ymm15 = _mm256_load_pd(buffer_C+C0_ROW_D(0));
         ymm15 = _mm256_add_pd(ymm15, ymm3);
         _mm256_store_pd(buffer_C+C0_ROW_D(0), ymm15);
      
         ymm15 = _mm256_load_pd(buffer_C+C1_ROW_D(0));
         ymm15 = _mm256_add_pd(ymm15, ymm4);
         _mm256_store_pd(buffer_C+C1_ROW_D(0), ymm15);

         ymm15 = _mm256_load_pd(buffer_C+C0_ROW_D(1));
         ymm15 = _mm256_add_pd(ymm15, ymm5);
         _mm256_store_pd(buffer_C+C0_ROW_D(1), ymm15);

         ymm15 = _mm256_load_pd(buffer_C+C1_ROW_D(1));
         ymm15 = _mm256_add_pd(ymm15, ymm6);
         _mm256_store_pd(buffer_C+C1_ROW_D(1), ymm15);

         ymm15 = _mm256_load_pd(buffer_C+C0_ROW_D(2));
         ymm15 = _mm256_add_pd(ymm15, ymm7);
         _mm256_store_pd(buffer_C+C0_ROW_D(2), ymm15);
      
         ymm15 = _mm256_load_pd(buffer_C+C1_ROW_D(2));
         ymm15 = _mm256_add_pd(ymm15, ymm8);
         _mm256_store_pd(buffer_C+C1_ROW_D(2), ymm15);

         ymm15 = _mm256_load_pd(buffer_C+C0_ROW_D(3));
         ymm15 = _mm256_add_pd(ymm15, ymm9);
         _mm256_store_pd(buffer_C+C0_ROW_D(3), ymm15);

         ymm15 = _mm256_load_pd(buffer_C+C1_ROW_D(3));
         ymm15 = _mm256_add_pd(ymm15, ymm10);
         _mm256_store_pd(buffer_C+C1_ROW_D(3), ymm15);

         UNROLL 3 (last)
         ymm0 = _mm256_load_pd(buffer_B+3*NR_D+4*l*NR_D);
         ymm1 = _mm256_load_pd(buffer_B+3*NR_D+4*l*NR_D+4);

         ymm2 = _mm256_broadcast_sd(buffer_A+3*MR_D+4*l*MR_D);
         ymm3 = _mm256_mul_pd(ymm0, ymm2);
         ymm4 = _mm256_mul_pd(ymm1, ymm2);
      
         ymm2 = _mm256_broadcast_sd(buffer_A+3*MR_D+4*l*MR_D+1);
         ymm5 = _mm256_mul_pd(ymm0, ymm2);
         ymm6 = _mm256_mul_pd(ymm1, ymm2);

         ymm2 = _mm256_broadcast_sd(buffer_A+3*MR_D+4*l*MR_D+2);
         ymm7 = _mm256_mul_pd(ymm0, ymm2);
         ymm8 = _mm256_mul_pd(ymm1, ymm2);

         ymm2  = _mm256_broadcast_sd(buffer_A+3*MR_D+4*l*MR_D+3);
         ymm9  = _mm256_mul_pd(ymm0, ymm2);
         ymm10 = _mm256_mul_pd(ymm1, ymm2);
    
         store new entries
         ymm15 = _mm256_load_pd(buffer_C+C0_ROW_D(0));
         ymm15 = _mm256_add_pd(ymm15, ymm3);
         _mm256_store_pd(buffer_C+C0_ROW_D(0), ymm15);
      
         ymm15 = _mm256_load_pd(buffer_C+C1_ROW_D(0));
         ymm15 = _mm256_add_pd(ymm15, ymm4);
         _mm256_store_pd(buffer_C+C1_ROW_D(0), ymm15);

         ymm15 = _mm256_load_pd(buffer_C+C0_ROW_D(1));
         ymm15 = _mm256_add_pd(ymm15, ymm5);
         _mm256_store_pd(buffer_C+C0_ROW_D(1), ymm15);

         ymm15 = _mm256_load_pd(buffer_C+C1_ROW_D(1));
         ymm15 = _mm256_add_pd(ymm15, ymm6);
         _mm256_store_pd(buffer_C+C1_ROW_D(1), ymm15);

         ymm15 = _mm256_load_pd(buffer_C+C0_ROW_D(2));
         ymm15 = _mm256_add_pd(ymm15, ymm7);
         _mm256_store_pd(buffer_C+C0_ROW_D(2), ymm15);
      
         ymm15 = _mm256_load_pd(buffer_C+C1_ROW_D(2));
         ymm15 = _mm256_add_pd(ymm15, ymm8);
         _mm256_store_pd(buffer_C+C1_ROW_D(2), ymm15);

         ymm15 = _mm256_load_pd(buffer_C+C0_ROW_D(3));
         ymm15 = _mm256_add_pd(ymm15, ymm9);
         _mm256_store_pd(buffer_C+C0_ROW_D(3), ymm15);

         ymm15 = _mm256_load_pd(buffer_C+C1_ROW_D(3));
         ymm15 = _mm256_add_pd(ymm15, ymm10);
         _mm256_store_pd(buffer_C+C1_ROW_D(3), ymm15);
         }//for unrolled 4 times

         for (vcl_size_t x=l*4; x<((l*4)+(num_micro_slivers%4)); ++x)
         {
         ymm0 = _mm256_load_pd(buffer_B+x*NR_D);
         ymm1 = _mm256_load_pd(buffer_B+x*NR_D+4);

         ymm2 = _mm256_broadcast_sd(buffer_A+x*MR_D);
         ymm3 = _mm256_mul_pd(ymm0, ymm2);
         ymm4 = _mm256_mul_pd(ymm1, ymm2);
      
         ymm2 = _mm256_broadcast_sd(buffer_A+x*MR_D+1);
         ymm5 = _mm256_mul_pd(ymm0, ymm2);
         ymm6 = _mm256_mul_pd(ymm1, ymm2);

         ymm2 = _mm256_broadcast_sd(buffer_A+x*MR_D+2);
         ymm7 = _mm256_mul_pd(ymm0, ymm2);
         ymm8 = _mm256_mul_pd(ymm1, ymm2);

         ymm2  = _mm256_broadcast_sd(buffer_A+x*MR_D+3);
         ymm9  = _mm256_mul_pd(ymm0, ymm2);
         ymm10 = _mm256_mul_pd(ymm1, ymm2);
    
         store new entries 
         ymm15 = _mm256_load_pd(buffer_C+C0_ROW_D(0));
         ymm15 = _mm256_add_pd(ymm15, ymm3);
         _mm256_store_pd(buffer_C+C0_ROW_D(0), ymm15);
      
         ymm15 = _mm256_load_pd(buffer_C+C1_ROW_D(0));
         ymm15 = _mm256_add_pd(ymm15, ymm4);
         _mm256_store_pd(buffer_C+C1_ROW_D(0), ymm15);

         ymm15 = _mm256_load_pd(buffer_C+C0_ROW_D(1));
         ymm15 = _mm256_add_pd(ymm15, ymm5);
         _mm256_store_pd(buffer_C+C0_ROW_D(1), ymm15);

         ymm15 = _mm256_load_pd(buffer_C+C1_ROW_D(1));
         ymm15 = _mm256_add_pd(ymm15, ymm6);
         _mm256_store_pd(buffer_C+C1_ROW_D(1), ymm15);

         ymm15 = _mm256_load_pd(buffer_C+C0_ROW_D(2));
         ymm15 = _mm256_add_pd(ymm15, ymm7);
         _mm256_store_pd(buffer_C+C0_ROW_D(2), ymm15);
      
         ymm15 = _mm256_load_pd(buffer_C+C1_ROW_D(2));
         ymm15 = _mm256_add_pd(ymm15, ymm8);
         _mm256_store_pd(buffer_C+C1_ROW_D(2), ymm15);

         ymm15 = _mm256_load_pd(buffer_C+C0_ROW_D(3));
         ymm15 = _mm256_add_pd(ymm15, ymm9);
         _mm256_store_pd(buffer_C+C0_ROW_D(3), ymm15);

         ymm15 = _mm256_load_pd(buffer_C+C1_ROW_D(3));
         ymm15 = _mm256_add_pd(ymm15, ymm10);
         _mm256_store_pd(buffer_C+C1_ROW_D(3), ymm15);
         }//for residuals*/
    }//for without unroll
  }//avx_micro_kernel2<double>
}//viennacl
#endif
