#ifndef VIENNACL_LINALG_HOST_BASED_GEMM_AVX_MICRO_KERNEL_HPP_
#define VIENNACL_LINALG_HOST_BASED_GEMM_AVX_MICRO_KERNEL_HPP_

#include "viennacl/linalg/host_based/common.hpp"
#include "immintrin.h"

#define MR (4)
#define NR (4)

//TODO: do some ascii art to demonstrate how data is moved in registers
/* ymm2    |   ymm3   | ymm4    | ymm5   
 * 0 . . . |  . 1 . . | . . . 3 | . . 2 .
 * . 1 . . |  0 . . . | . . 2 . | . . . 3   ... etc.
 * . . 2 . |  . . . 2 | . 1 . . | 0 . . .
 * . . . 3 |  . . 3 . | 0 . . . | . 1 . .
 */
#define SWAP_128_BIT (0x01)
#define SWAP_64_BIT  (0x05)
#define SHUFFLE      (0x0A)
#define PERMUTE1     (0x30)
#define PERMUTE2     (0x12)

namespace viennacl
{
  // TODO: define register sizes in common.hpp depending on Compilerflag? => remove mr,nr from argument list*/
  inline void avx_micro_kernel_double(double const *buffer_A, double const *buffer_B, double *buffer_C,
                                      vcl_size_t num_micro_slivers, vcl_size_t mr, vcl_size_t nr)
  {

    // TODO: move this etc?
    assert(mr == MR && nr == NR && bool("register-block size was set incorrectly (by get_block_sizes()) for this architecture!"));
    
    //TODO:  get rid of unaligned stores!
    __m256d ymm0 , ymm1 , ymm2 , ymm3 ;
    __m256d ymm4 , ymm5;// , ymm6 , ymm7 ;
    //__m256d ymm8 , ymm9 , ymm10, ymm11;
    __m256d ymm12, ymm13, ymm14, ymm15;

    //double buffer_debug[MR*NR];//DEBUG

    for (vcl_size_t l=0; l<num_micro_slivers; ++l)
    {
      //load operands for C0
      ymm0 = _mm256_loadu_pd(buffer_A+l*MR);
      ymm1 = _mm256_loadu_pd(buffer_B+l*NR);
      
      //_mm256_storeu_pd(buffer_debug, ymm1);//DEBUG
      //if (l == 1)//DEBUG
      //std::cout << std::endl <<"ymm1 is: " << buffer_debug[0] << " " << buffer_debug[1] << " " << buffer_debug[2] << " "<< buffer_debug[3] << std::endl;//DEBUG
      
      //load operands for C1
      //ymm6 = _mm256_load_pd(buffer_A+4);
      //ymm7 = _mm256_load_pd(buffer_B+4);
    
      //calculate C0
      ymm2 = _mm256_mul_pd(ymm0, ymm1);
      ymm0 = _mm256_permute_pd(ymm0, SWAP_64_BIT);
      //_mm256_storeu_pd(buffer_debug, ymm0);//DEBUG
      //std::cout << "ymm0 is: " << buffer_debug[0] << " " << buffer_debug[1] << " " << buffer_debug[2] << " "<< buffer_debug[3] << std::endl;//DEBUG
      ymm3 = _mm256_mul_pd(ymm0, ymm1);
      ymm0 = _mm256_permute2f128_pd(ymm0, ymm0, SWAP_128_BIT);
      //_mm256_storeu_pd(buffer_debug, ymm0);//DEBUG
      //std::cout << "ymm0 is: " << buffer_debug[0] << " " << buffer_debug[1] << " " << buffer_debug[2] << " "<< buffer_debug[3] << std::endl;//DEBUG
      ymm4 = _mm256_mul_pd(ymm0, ymm1);    
      ymm0 = _mm256_permute_pd(ymm0, SWAP_64_BIT);
      //_mm256_storeu_pd(buffer_debug, ymm0);//DEBUG
      //std::cout << "ymm0 is: " << buffer_debug[0] << " " << buffer_debug[1] << " " << buffer_debug[2] << " "<< buffer_debug[3] << std::endl;//DEBUG
      ymm5 = _mm256_mul_pd(ymm0, ymm1);
    
      //store C0
      //  store row 0 and 2
      ymm12 = _mm256_shuffle_pd(ymm2 , ymm3 , SHUFFLE);
      ymm13 = _mm256_shuffle_pd(ymm5 , ymm4 , SHUFFLE);
      ymm14 = _mm256_permute2f128_pd(ymm12, ymm13, PERMUTE1);
      //_mm256_storeu_pd(buffer_debug, ymm14);//DEBUG
      //std::cout << "ymm14 after first permute is: " << buffer_debug[0] << " " << buffer_debug[1] << " " << buffer_debug[2] << " "<< buffer_debug[3] << std::endl;//DEBUG
      
      ymm15 = _mm256_loadu_pd(buffer_C+0*MR);
      ymm14 = _mm256_add_pd(ymm14, ymm15);
      //_mm256_storeu_pd(buffer_debug, ymm14);//DEBUG
      //std::cout << "row0 is: " << buffer_debug[0] << " " << buffer_debug[1] << " " << buffer_debug[2] << " "<< buffer_debug[3] << std::endl;//DEBUG
      _mm256_storeu_pd(buffer_C+0*MR, ymm14);
      
      ymm14 = _mm256_permute2f128_pd(ymm12, ymm13, PERMUTE2);
      //_mm256_storeu_pd(buffer_debug, ymm14);//DEBUG
      //std::cout << "ymm14 after second permute is: " << buffer_debug[0] << " " << buffer_debug[1] << " " << buffer_debug[2] << " "<< buffer_debug[3] << std::endl;//DEBUG
      
      ymm15 = _mm256_loadu_pd(buffer_C+2*MR);
      ymm14 = _mm256_add_pd(ymm14, ymm15);
      //_mm256_storeu_pd(buffer_debug, ymm14);//DEBUG
      //std::cout << "row2 is: " << buffer_debug[0] << " " << buffer_debug[1] << " " << buffer_debug[2] << " "<< buffer_debug[3] << std::endl;//DEBUG
      _mm256_storeu_pd(buffer_C+2*MR, ymm14);
      
    
      //  store row 1 and 3
      ymm12 = _mm256_shuffle_pd(ymm3 , ymm2 , SHUFFLE);
      ymm13 = _mm256_shuffle_pd(ymm4 , ymm5 , SHUFFLE);
      ymm14 = _mm256_permute2f128_pd(ymm12, ymm13, PERMUTE1);
      
      ymm15 = _mm256_loadu_pd(buffer_C+1*MR);
      ymm14 = _mm256_add_pd(ymm14, ymm15);
      _mm256_storeu_pd(buffer_C+1*MR, ymm14);
      
      ymm14 = _mm256_permute2f128_pd(ymm12, ymm13, PERMUTE2);
      
      ymm15 = _mm256_loadu_pd(buffer_C+3*MR);
      ymm14 = _mm256_add_pd(ymm14, ymm15);
      _mm256_storeu_pd(buffer_C+3*MR, ymm14);
    
      /*      ymm14 = _m256_shuffle_pd(ymm12, ymm13, SHUFFLE3);
              ymm12 = _m256_shuffle_pd(

              ymm8  = _m256_mul_pd(ymm6, ymm7);
              ymm6  = _m256_permute_pd(ymm6, SWAP_64_BIT);
              ymm9  = _m256_mul_pd(ymm6, ymm7);
              ymm6  = _mm256_permute2f128_si256(ymm6, ymm6, SWAP_128_BIT);
              ymm10 = _m256_mul_pd(ymm6, ymm7);
              ymm6  = _m256_permute_pd(ymm6, SWAP_64_BIT);
              ymm11 = _m256_mul_pd(ymm6, ymm7); */
    }

    /*    std::cout << "buffer_C is: ";//DEBUG
    for (int i=0; i<MR*NR; i++)//DEBUG
      std::cout << buffer_C[i] << ", ";//DEBUG
      std::cout << std::endl;//DEBUG*/
    

  }
  

  template<typename MatrixAccT>
  inline void avx_micro_kernel_float(double *buffer_A, double *buffer_B, double *buffer_C, MatrixAccT & C,
                                      double alpha, double beta,
                                      vcl_size_t C2B2_idx, vcl_size_t A2B1_idx, vcl_size_t C1A1_idx, 
                                      vcl_size_t sliver_A_idx, vcl_size_t sliver_B_idx,
                                      vcl_size_t m_size, vcl_size_t k_size, vcl_size_t n_size,
                                      vcl_size_t mc, vcl_size_t kc, vcl_size_t nc,
                                      vcl_size_t mr, vcl_size_t nr)
  {
    
  }
}//viennacl

#endif
