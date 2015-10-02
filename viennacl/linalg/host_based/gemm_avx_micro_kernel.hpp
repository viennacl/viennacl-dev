#ifndef VIENNACL_LINALG_HOST_BASED_GEMM_AVX_MICRO_KERNEL_HPP_
#define VIENNACL_LINALG_HOST_BASED_GEMM_AVX_MICRO_KERNEL_HPP_

#include "viennacl/linalg/host_based/common.hpp"
#include "immintrin.h"

/* register-block sizes: 
 * D := double, F := float */
#define MR_D ( 4)
#define NR_D ( 8)
#define MR_F ( 8)
#define NR_F ( 8)
//#define NR_F (16)//TODO

/* addresses for buffer_C */
#define C0_ROW_D(a) (a*NR_D)
#define C1_ROW_D(a) (a*NR_D + NR_D/2)

#define C0_ROW_F(a) (a*NR_F)
#define C1_ROW_F(a) (a*NR_F + NR_F/2)

/* imm8 values for permute and shuffle instructions */
#define SWAP_128_BIT  (0x01)

#define SWAP_64_BIT_D (0x05)
#define SHUFFLE_D     (0x0A)
#define PERMUTE_D     (0x30)

#define SWAP_64_BIT_F (0x4E)
#define SWAP_32_BIT_F (0xB1)
#define SHUFFLE_F     (0xD8)

    //TODO: do some ascii art to demonstrate how data is moved in registers
/* ymm2    |   ymm3   | ymm4    | ymm5   
 * 0 . . . |  . 1 . . | . . . 3 | . . 2 .
 * . 1 . . |  0 . . . | . . 2 . | . . . 3   ... etc.
 * . . 2 . |  . . . 2 | . 1 . . | 0 . . .
 * . . . 3 |  . . 3 . | 0 . . . | . 1 . .
 */


namespace viennacl
{
  template<typename NumericT>
  inline void avx_micro_kernel(NumericT const *buffer_A, NumericT const *buffer_B, NumericT *buffer_C,
                               vcl_size_t num_micro_slivers, vcl_size_t mr, vcl_size_t nr)
  {
    assert(false && bool("called with unsupported numeric type!"));
  }

  template<>
  inline void avx_micro_kernel<float>(float const *buffer_A, float const *buffer_B, float *buffer_C,
                                      vcl_size_t num_micro_slivers, vcl_size_t mr, vcl_size_t nr)
  {
    assert( (mr == MR_F) && (nr == NR_F) && bool("mr and nr obtained by 'get_block_sizes()' in 'matrix_operations.hpp' and given to 'avx_micro_kernel()' do not match with MR_F/NR_F defined in 'gemm_avx_micro_kernel.hpp' ") );

    __m256 ymm0 , ymm1 , ymm2 , ymm3 ;
    __m256 ymm4 , ymm5 , ymm6 , ymm7 ;
    __m256 ymm8 , ymm9;// , ymm10, ymm11;
    __m256 /*ymm12, ymm13, ymm14,*/ ymm15;

    //    float buffer_debug[MR_F*NR_F];//DEBUG

    for (vcl_size_t l=0; l<num_micro_slivers; ++l)
    {
      /* load data */
      ymm0 = _mm256_broadcast_ss(buffer_A+l*MR_F);
      ymm1 = _mm256_load_ps(buffer_B+l*NR_F);
      //ymm6 = _mm256_load_ps(buffer_B+l*NR_F+4);
      
      ymm2 = _mm256_mul_ps(ymm0, ymm1);
      
      ymm0 = _mm256_broadcast_ss(buffer_A+l*MR_F+1);
      ymm3 = _mm256_mul_ps(ymm0, ymm1);

      ymm0 = _mm256_broadcast_ss(buffer_A+l*MR_F+2);
      ymm4 = _mm256_mul_ps(ymm0, ymm1);
      
      ymm0 = _mm256_broadcast_ss(buffer_A+l*MR_F+3);
      ymm5 = _mm256_mul_ps(ymm0, ymm1);
      
      ymm0 = _mm256_broadcast_ss(buffer_A+l*MR_F+4);
      ymm6 = _mm256_mul_ps(ymm0, ymm1);
      
      ymm0 = _mm256_broadcast_ss(buffer_A+l*MR_F+5);
      ymm7 = _mm256_mul_ps(ymm0, ymm1);
      
      ymm0 = _mm256_broadcast_ss(buffer_A+l*MR_F+6);
      ymm8 = _mm256_mul_ps(ymm0, ymm1);
      
      ymm0 = _mm256_broadcast_ss(buffer_A+l*MR_F+7);
      ymm9 = _mm256_mul_ps(ymm0, ymm1);

      /* store new entries */
      ymm15 = _mm256_load_ps(buffer_C+C0_ROW_F(0));
      ymm15 = _mm256_add_ps(ymm15, ymm2);
      _mm256_store_ps(buffer_C+C0_ROW_F(0), ymm15);
      
      ymm15 = _mm256_load_ps(buffer_C+C0_ROW_F(1));
      ymm15 = _mm256_add_ps(ymm15, ymm3);
      _mm256_store_ps(buffer_C+C0_ROW_F(1), ymm15);

      ymm15 = _mm256_load_ps(buffer_C+C0_ROW_F(2));
      ymm15 = _mm256_add_ps(ymm15, ymm4);
      _mm256_store_ps(buffer_C+C0_ROW_F(2), ymm15);

      ymm15 = _mm256_load_ps(buffer_C+C0_ROW_F(3));
      ymm15 = _mm256_add_ps(ymm15, ymm5);
      _mm256_store_ps(buffer_C+C0_ROW_F(3), ymm15);

      ymm15 = _mm256_load_ps(buffer_C+C0_ROW_F(4));
      ymm15 = _mm256_add_ps(ymm15, ymm6);
      _mm256_store_ps(buffer_C+C0_ROW_F(4), ymm15);

      ymm15 = _mm256_load_ps(buffer_C+C0_ROW_F(5));
      ymm15 = _mm256_add_ps(ymm15, ymm7);
      _mm256_store_ps(buffer_C+C0_ROW_F(5), ymm15);
      
      ymm15 = _mm256_load_ps(buffer_C+C0_ROW_F(6));
      ymm15 = _mm256_add_ps(ymm15, ymm8);
      _mm256_store_ps(buffer_C+C0_ROW_F(6), ymm15);

      ymm15 = _mm256_load_ps(buffer_C+C0_ROW_F(7));
      ymm15 = _mm256_add_ps(ymm15, ymm9);
      _mm256_store_ps(buffer_C+C0_ROW_F(7), ymm15);
      
      //_mm256_store_ps(buffer_debug, ymm1);//DEBUG
      //if (l == 1)//DEBUG
      //std::cout << std::endl <<"ymm1 is: " << buffer_debug[0] << " " << buffer_debug[1] << " " << buffer_debug[2] << " "<< buffer_debug[3] << std::endl;//DEBUG

      
      /*std::cout << "buffer_C is: ";//DEBUG
        for (int i=0; i<MR_F*NR_F; i++)//DEBUG
        std::cout << buffer_C[i] << ", ";//DEBUG
        std::cout << std::endl;//DEBUG*/
    }
  }

  template<>
  inline void avx_micro_kernel<double>(double const *buffer_A, double const *buffer_B, double *buffer_C,
                                        vcl_size_t num_micro_slivers, vcl_size_t mr, vcl_size_t nr)
  {
    assert( (mr == MR_D) && (nr == NR_D) && bool("mr and nr obtained by 'get_block_sizes()' in 'matrix_operations.hpp' and given to 'avx_micro_kernel()' do not match with MR_D/NR_D defined in 'gemm_avx_micro_kernel.hpp' ") );

    __m256d ymm0 , ymm1 , ymm2 , ymm3 ;
    __m256d ymm4 , ymm5 , ymm6 , ymm7 ;
    __m256d ymm8 , ymm9 , ymm10;//, ymm11;
    __m256d /*ymm12, ymm13, ymm14,*/ ymm15;

    for (vcl_size_t l=0; l<num_micro_slivers; ++l)
    {
      ymm0 = _mm256_load_pd(buffer_B+l*NR_D);
      ymm1 = _mm256_load_pd(buffer_B+l*NR_D+4);

      ymm2 = _mm256_broadcast_sd(buffer_A+l*MR_D);
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
    }//for
  }//avx_micro_kernel2<double>
}//viennacl
#endif
