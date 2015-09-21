#ifndef VIENNACL_LINALG_HOST_BASED_GEMM_AVX_MICRO_KERNEL_HPP_
#define VIENNACL_LINALG_HOST_BASED_GEMM_AVX_MICRO_KERNEL_HPP_

//#define _POSIX_C_SOURCE
//#define _XOPEN_SOURCE

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
#define C0_ROW0_D (0*NR_D)
#define C0_ROW1_D (1*NR_D)
#define C0_ROW2_D (2*NR_D)
#define C0_ROW3_D (3*NR_D)
#define C1_ROW0_D (0*NR_D + NR_D/2)
#define C1_ROW1_D (1*NR_D + NR_D/2)
#define C1_ROW2_D (2*NR_D + NR_D/2)
#define C1_ROW3_D (3*NR_D + NR_D/2)

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
    //TODO: FIND OUT WHAT'S THE PROPPER WAY TO DO THIS
    gemm_standard_micro_kernel(buffer_A, buffer_B, buffer_C, num_micro_slivers, mr, nr);
  }

  template<>
  inline void avx_micro_kernel<double>(double const *buffer_A, double const *buffer_B, double *buffer_C,
                                       vcl_size_t num_micro_slivers, vcl_size_t mr, vcl_size_t nr)
  {
    assert( (mr == MR_D) && (nr == NR_D) && bool("mr and nr obtained by 'get_block_sizes()' in 'matrix_operations.hpp' and given to 'avx_micro_kernel()' do not match with MR_D/NR_D defined in 'gemm_avx_micro_kernel.hpp' ") );

    __m256d ymm0 , ymm1 , ymm2 , ymm3 ;
    __m256d ymm4 , ymm5 , ymm6 , ymm7 ;
    __m256d ymm8 , ymm9 , ymm10;//, ymm11;
    __m256d ymm12, ymm13, ymm14, ymm15;

    //    double buffer_debug[MR_D*NR_D];//DEBUG

    for (vcl_size_t l=0; l<num_micro_slivers; ++l)
    {
      //load operands for C0 and C1
      ymm0 = _mm256_load_pd(buffer_A+l*MR_D);
      ymm1 = _mm256_load_pd(buffer_B+l*NR_D);
      ymm6 = _mm256_load_pd(buffer_B+l*NR_D+4);
      
      //_mm256_store_pd(buffer_debug, ymm1);//DEBUG
      //if (l == 1)//DEBUG
      //std::cout << std::endl <<"ymm1 is: " << buffer_debug[0] << " " << buffer_debug[1] << " " << buffer_debug[2] << " "<< buffer_debug[3] << std::endl;//DEBUG
    
      /* calculate C0 */
      ymm2  = _mm256_mul_pd(ymm0, ymm1);
      ymm7  = _mm256_mul_pd(ymm0, ymm6);
      ymm0  = _mm256_permute_pd(ymm0, SWAP_64_BIT_D);
      //_mm256_store_pd(buffer_debug, ymm0);//DEBUG
      //std::cout << "ymm0 is: " << buffer_debug[0] << " " << buffer_debug[1] << " " << buffer_debug[2] << " "<< buffer_debug[3] << std::endl;//DEBUG
      ymm3  = _mm256_mul_pd(ymm0, ymm1);
      ymm8  = _mm256_mul_pd(ymm0, ymm6);
      ymm0  = _mm256_permute2f128_pd(ymm0, ymm0, SWAP_128_BIT);
      //_mm256_store_pd(buffer_debug, ymm0);//DEBUG
      //std::cout << "ymm0 is: " << buffer_debug[0] << " " << buffer_debug[1] << " " << buffer_debug[2] << " "<< buffer_debug[3] << std::endl;//DEBUG
      ymm4  = _mm256_mul_pd(ymm0, ymm1);
      ymm9  = _mm256_mul_pd(ymm0, ymm6);
      ymm0  = _mm256_permute_pd(ymm0, SWAP_64_BIT_D);
      //_mm256_store_pd(buffer_debug, ymm0);//DEBUG
      //std::cout << "ymm0 is: " << buffer_debug[0] << " " << buffer_debug[1] << " " << buffer_debug[2] << " "<< buffer_debug[3] << std::endl;//DEBUG
      ymm5  = _mm256_mul_pd(ymm0, ymm1);
      ymm10 = _mm256_mul_pd(ymm0, ymm6);      


      /*      _mm256_store_pd(buffer_debug+0 , ymm2);//DEBUG
              _mm256_store_pd(buffer_debug+4 , ymm3);//DEBUG
              _mm256_store_pd(buffer_debug+8 , ymm4);//DEBUG
              _mm256_store_pd(buffer_debug+12, ymm5);//DEBUG
              _mm256_store_pd(buffer_debug+16, ymm7);//DEBUG
              _mm256_store_pd(buffer_debug+20, ymm8);//DEBUG
              _mm256_store_pd(buffer_debug+24, ymm9);//DEBUG
              _mm256_store_pd(buffer_debug+28, ymm10);//DEBUG
              std::cout << "ymm2-5 and ymm7-10 are: ";//DEBUG
              for (vcl_size_t i=0; i<32; ++i)//DEBUG
              std::cout << buffer_debug[i] << " ";//DEBUG
              std::cout << std::endl;//DEBUG*/

      
      /* store C0 */
      /* C0: store row 0 and 2 */
      ymm12 = _mm256_shuffle_pd(ymm2 , ymm3 , SHUFFLE_D);
      ymm13 = _mm256_shuffle_pd(ymm5 , ymm4 , SHUFFLE_D);
      ymm14 = _mm256_permute2f128_pd(ymm12, ymm13, PERMUTE_D);
      //_mm256_store_pd(buffer_debug, ymm14);//DEBUG
      //std::cout << "ymm14 after first permute is: " << buffer_debug[0] << " " << buffer_debug[1] << " " << buffer_debug[2] << " "<< buffer_debug[3] << std::endl;//DEBUG
      ymm15 = _mm256_load_pd(buffer_C+C0_ROW0_D);
      ymm14 = _mm256_add_pd(ymm14, ymm15);
      //_mm256_store_pd(buffer_debug, ymm14);//DEBUG
      //std::cout << "row0 is: " << buffer_debug[0] << " " << buffer_debug[1] << " " << buffer_debug[2] << " "<< buffer_debug[3] << std::endl;//DEBUG
      _mm256_store_pd(buffer_C+C0_ROW0_D, ymm14);
      ymm14 = _mm256_permute2f128_pd(ymm13, ymm12, PERMUTE_D);
      //_mm256_store_pd(buffer_debug, ymm14);//DEBUG
      //std::cout << "ymm14 after second permute is: " << buffer_debug[0] << " " << buffer_debug[1] << " " << buffer_debug[2] << " "<< buffer_debug[3] << std::endl;//DEBUG
      ymm15 = _mm256_load_pd(buffer_C+C0_ROW2_D);
      ymm14 = _mm256_add_pd(ymm14, ymm15);
      //_mm256_store_pd(buffer_debug, ymm14);//DEBUG
      //std::cout << "row2 is: " << buffer_debug[0] << " " << buffer_debug[1] << " " << buffer_debug[2] << " "<< buffer_debug[3] << std::endl;//DEBUG
      _mm256_store_pd(buffer_C+C0_ROW2_D, ymm14);
      
    
      /* C0: store row 1 and 3 */
      ymm12 = _mm256_shuffle_pd(ymm3 , ymm2 , SHUFFLE_D);
      ymm13 = _mm256_shuffle_pd(ymm4 , ymm5 , SHUFFLE_D);
      ymm14 = _mm256_permute2f128_pd(ymm12, ymm13, PERMUTE_D);
      
      ymm15 = _mm256_load_pd(buffer_C+C0_ROW1_D);
      ymm14 = _mm256_add_pd(ymm14, ymm15);
      _mm256_store_pd(buffer_C+C0_ROW1_D, ymm14);
      
      ymm14 = _mm256_permute2f128_pd(ymm13, ymm12, PERMUTE_D);
      
      ymm15 = _mm256_load_pd(buffer_C+C0_ROW3_D);
      ymm14 = _mm256_add_pd(ymm14, ymm15);
      _mm256_store_pd(buffer_C+C0_ROW3_D, ymm14);

      /* store C1 */
      /* C1: store row 0 and 2 */
      ymm12 = _mm256_shuffle_pd(ymm7  , ymm8 , SHUFFLE_D);
      ymm13 = _mm256_shuffle_pd(ymm10 , ymm9 , SHUFFLE_D);
      ymm14 = _mm256_permute2f128_pd(ymm12, ymm13, PERMUTE_D);
      //_mm256_store_pd(buffer_debug, ymm14);//DEBUG
      //std::cout << "ymm14 after first permute is: " << buffer_debug[0] << " " << buffer_debug[1] << " " << buffer_debug[2] << " "<< buffer_debug[3] << std::endl;//DEBUG
      ymm15 = _mm256_load_pd(buffer_C+C1_ROW0_D);
      ymm14 = _mm256_add_pd(ymm14, ymm15);
      //_mm256_store_pd(buffer_debug, ymm14);//DEBUG
      //std::cout << "row0 is: " << buffer_debug[0] << " " << buffer_debug[1] << " " << buffer_debug[2] << " "<< buffer_debug[3] << std::endl;//DEBUG
      _mm256_store_pd(buffer_C+C1_ROW0_D, ymm14);
      ymm14 = _mm256_permute2f128_pd(ymm13, ymm12, PERMUTE_D);
      //_mm256_store_pd(buffer_debug, ymm14);//DEBUG
      //std::cout << "ymm14 after second permute is: " << buffer_debug[0] << " " << buffer_debug[1] << " " << buffer_debug[2] << " "<< buffer_debug[3] << std::endl;//DEBUG
      ymm15 = _mm256_load_pd(buffer_C+C1_ROW2_D);
      ymm14 = _mm256_add_pd(ymm14, ymm15);
      //_mm256_store_pd(buffer_debug, ymm14);//DEBUG
      //std::cout << "row2 is: " << buffer_debug[0] << " " << buffer_debug[1] << " " << buffer_debug[2] << " "<< buffer_debug[3] << std::endl;//DEBUG
      _mm256_store_pd(buffer_C+C1_ROW2_D, ymm14);
    
      /* C1: store row 1 and 3 */
      ymm12 = _mm256_shuffle_pd(ymm8, ymm7 , SHUFFLE_D);
      ymm13 = _mm256_shuffle_pd(ymm9, ymm10, SHUFFLE_D);
      ymm14 = _mm256_permute2f128_pd(ymm12, ymm13, PERMUTE_D);
      
      ymm15 = _mm256_load_pd(buffer_C+C1_ROW1_D);
      ymm14 = _mm256_add_pd(ymm14, ymm15);
      _mm256_store_pd(buffer_C+C1_ROW1_D, ymm14);
      
      ymm14 = _mm256_permute2f128_pd(ymm13, ymm12, PERMUTE_D);
      
      ymm15 = _mm256_load_pd(buffer_C+C1_ROW3_D);
      ymm14 = _mm256_add_pd(ymm14, ymm15);
      _mm256_store_pd(buffer_C+C1_ROW3_D, ymm14);
    }
    /*            std::cout << "buffer_C is: ";//DEBUG
                  for (int i=0; i<MR_D*NR_D; i++)//DEBUG
                  std::cout << buffer_C[i] << ", ";//DEBUG
                  std::cout << std::endl;//DEBUG*/
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

  template<typename NumericT>
  NumericT *get_aligned_buffer(vcl_size_t size, bool fill_zeros)
  {
#ifdef VIENNACL_WITH_AVX
    
    void *ptr;
    int error = posix_memalign(&ptr, 32, size*sizeof(NumericT));
    
    if (error)
    {
      //dunno how to handle, throw exception? which one?
    }
    if (fill_zeros)
    {
      for (vcl_size_t i=0; i<size; ++i)
      {
        ((NumericT *)ptr)[i] = NumericT(0);
      }
    }
        
    return (NumericT *)ptr;
    
#else
    std::vector<NumericT> buffer(size);
    
    if (fill_zeros)
      std::fill(buffer.begin(), buffer.end(), NumericT(0));
    
    return (NumericT *)&(buffer[0]);
#endif    
  }

  template<typename NumericT>
  inline void free_aligned_buffer(NumericT *ptr)
  {
    free((void *)ptr);
  }
}//viennacl

#endif
