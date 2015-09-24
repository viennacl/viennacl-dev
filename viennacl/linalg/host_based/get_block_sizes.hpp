#ifndef VIENNACL_LINALG_HOST_BASED_GET_BLOCK_SIZES_HPP_
#define VIENNACL_LINALG_HOST_BASED_GET_BLOCK_SIZES_HPP_

#include <string.h>
#include "viennacl/forwards.h"
#include "viennacl/traits/size.hpp"

/* sizes for different architectures in bytes */
#define AVX_REG_SIZE     (256/8)
#define AVX_512_REG_SIZE (512/8)
#define SSE_REG_SIZE     (128/8)

/* CPU-vendor strings */
#define VENDOR_STR_LEN (12)
#define INTEL          "GenuineIntel"
#define AMD            "AuthenticAMD"

/* choose which cache-info is to be read,
 * quotation marks needed by the inline-assembler */
#define AMD_GET_L1      "0x80000005"
#define AMD_GET_L2      "0x80000006"
#define AMD_GET_L3      "0x80000007"
#define INTEL_GET_CACHE "0x00000002"
#define INTEL_GET_L1    "0x00000000"
#define INTEL_GET_L2    "0x00000001"
#define INTEL_GET_L3    "0x00000002"

/* auxilary for cache-info arrays */
#define AMD_CACHE_INFO_SIZE (4)
#define AMD_CACHE_SIZE_IDX  (3)

/* mask accodording bits obtained by calling 'cpuid' */
#define INTEL_CACHE_WAYS       (0xFFC0000000000000)
#define INTEL_CACHE_PARTITIONS (0x003FF00000000000)
#define INTEL_CACHE_LINE_SIZE  (0x00000FFF00000000)
#define INTEL_CACHE_SETS       (0x00000000FFFFFFFF)

/* calculates cache size, Formula found in intels "Intel 64 and IA-32 Architectures Software Developer Manual", p. 198 (pdf:264) */
#define INTEL_CALC_CACHE_SIZE(a) ((a & INTEL_CACHE_WAYS) +1) * ((a & INTEL_CACHE_PARTITIONS) +1) * ((a & INTEL_CACHE_LINE_SIZE)+1) * ((a & INTEL_CACHE_SETS)+1)

/* misc */
#define KILO (1024)
#define MEGA (1024*1024)

namespace viennacl
{
  void get_vendor_string(char *vendor)
  {
    __asm__
      (
        "movl  $0, %%eax         \n\t"
        "cpuid                  \n\t"
        "movl  %%ebx, (%%rdi)    \n\t"
        "movl  %%edx, 4(%%rdi)   \n\t"
        "movl  %%ecx, 8(%%rdi)   \n\t"
        :              \
        :              \
        : "%eax"
        );
  }
  
  void get_cache_amd(uint8_t *l1, uint8_t *l2, uint8_t *l3)
  {
    /* l1 = %rdi, l2 = %rsi, l3 = %rdx, %rcx*/
    __asm__
      (
        /* get L1-cache info */
        "movl  $" AMD_GET_L1 ", %%eax \n\t"
        "cpuid                    \n\t"
        "movl  %%ecx, (%%rdi)     \n\t"
        /* get L2-cache info */
        "movl  $" AMD_GET_L2 ", %%eax \n\t"
        "cpuid                    \n\t"
        "movl  %%ecx, (%%rsi)     \n\t"
        /* get L3-cache info */
        "movl  $" AMD_GET_L3 ", %%eax \n\t"
        "cpuid                    \n\t"
        "movl  %%ecx, (%%rdx)     \n\t"
        
        :              \
        :              \
        : "%eax"
        );
    //    *l1 = 16 * 1024; //16KB
    //*l2 =  1 * 1024; // 1KB
    //*l3 =  0 * 1024; // 0KB
  }
  
  void get_cache_intel(uint64_t *l1, uint64_t *l2, uint64_t *l3)
  {
    /* l1 = %rdi, l2 = %rsi, l3 = %rdx */
    __asm__
      (
        /* get L1-cache info */
        "movl  $" INTEL_GET_CACHE ", %%eax \n\t"
        "movl  $" INTEL_GET_L1 "   , %%ecx \n\t"
        "cpuid                         \n\t"
        "movl  %%ebx,  (%%rdi)          \n\t"
        "movl  %%ecx, 4(%%rdi)          \n\t"
        /* get L2-cache info */
        "movl  $" INTEL_GET_CACHE ", %%eax \n\t"
        "movl  $" INTEL_GET_L2 "   , %%ecx \n\t"
        "cpuid                         \n\t"
        "movl  %%ebx,  (%%rsi)          \n\t"
        "movl  %%ecx, 4(%%rsi)          \n\t"
        /* get L3-cache info */
        "movl  $" INTEL_GET_CACHE ", %%eax \n\t"
        "movl  $" INTEL_GET_L3 "   , %%ecx \n\t"
        "cpuid                         \n\t"
        "movl  %%ebx,  (%%rdx)          \n\t"
        "movl  %%ecx, 4(%%rdx)          \n\t"
        :
        :
        :
        );
  }
  
  void set_cache_sizes(int &l1_size, int &l2_size, int &l3_size)
  {
    /* used to store CPU-vendor string */
    char vendor[VENDOR_STR_LEN];
        
    /* check CPU-Vendor */
    get_vendor_string(vendor);

    if ( strncmp(vendor, INTEL, VENDOR_STR_LEN) == 0 )
    {
      std::cout << "INTEL!" << std::endl;//DEBUG
      /* store cache information (size, associativity, etc.)*/
      uint64_t l1_info;
      uint64_t l2_info;
      uint64_t l3_info;

      get_cache_intel(&l1_info, &l2_info, &l3_info);
      l1_size = INTEL_CALC_CACHE_SIZE(l1_info);
      l2_size = INTEL_CALC_CACHE_SIZE(l2_info);
      l3_size = INTEL_CALC_CACHE_SIZE(l3_info);
    }      
    else if ( strncmp(vendor, AMD, VENDOR_STR_LEN) == 0 )
    {
      std::cout << "AMD!" << std::endl;//DEBUG
      /* store cache information (size, associativity, etc.)*/
      uint8_t l1_info[AMD_CACHE_INFO_SIZE];
      uint8_t l2_info[AMD_CACHE_INFO_SIZE];
      uint8_t l3_info[AMD_CACHE_INFO_SIZE];
      
      /* gets cache info and sets sizes in Bytes */
      get_cache_amd(l1_info,l2_info,l3_info);
      std::cout << "hi" << std::endl;//DEBUG
      l1_size = l1_info[AMD_CACHE_SIZE_IDX] * KILO;
      l2_size = l2_info[AMD_CACHE_SIZE_IDX] * KILO;
      l3_size = l3_info[AMD_CACHE_SIZE_IDX] * KILO;
      std::cout << "AMD! sizes:" << l1_size << " " << l2_size << " " << l3_size <<std::endl;//DEBUG

    }
    else
    {
      /* CPU-vendor unknown, not supported or cpuid could not be read out propperly.
         Assume some (small) cache sizes. */
      l1_size =  8 * KILO;        // 8KB
      l2_size = 16 * KILO;        //16KB
      l3_size =  0 * KILO;        // 0KB
    }
  }

  template<typename NumericT> 
  void get_block_sizes(const vcl_size_t m_size, const vcl_size_t k_size, const vcl_size_t n_size, vcl_size_t & mc, vcl_size_t & kc, vcl_size_t & nc, vcl_size_t & mr, vcl_size_t & nr)
  {
    /* get register-block sizes in bytes and according to NumericT */
#ifdef VIENNACL_WITH_AVX

    mr = AVX_REG_SIZE/sizeof(NumericT);
    
    /* The current microkernel can process a register-block-size 'nr', of 8 for doubles,
     * but not for floats. Hence, the additional term. */
    nr = AVX_REG_SIZE/sizeof(NumericT) + (sizeof(NumericT)/sizeof(double)) * AVX_REG_SIZE/sizeof(NumericT);

    // not implemented yet
    /*#elif VIENNACL_WITH_AVX_512

      mr =   AVX_512_REG_SIZE/sizeof(NumericT);
      nr = 2*AVX_512_REG_SIZE/sizeof(NumericT);

      #elif VIENNACL_WITH_SSE

      mr =   SSE_REG_SIZE/sizeof(NumericT);
      nr = 2*SSE_REG_SIZE/sizeof(NumericT);*/
    
#else
    /* standard case */
    mr = 1;
    nr = 1;
    
#endif

    static int l1, l2, l3;
    static bool cache_sizes_unknown = true;

    /* hardware won't change during run-time (hopefully)
     * ==> determine cache sizes only once */
    if (cache_sizes_unknown)
    {
      set_cache_sizes(l1, l2, l3);
      cache_sizes_unknown = false;
    }
    
    /* Calculate blocksizes for L1 (mc x nr) and L2 (mc * kc) and L3 cache. 
     * Assumed that block in L2 cache should be square and half of cache shuold be empty. */
    // TODO: improve formula? 
    if (l1 == 0)
      kc  = k_size;
    else
      kc  = l1 / (2 * nr * sizeof(NumericT));
    
    if (l1 == 0)
      mc  = m_size;
    else
      mc = kc;
    mc += mr - (mc%mr); //mc must be divisible by mr

    if (l3 == 0)
      nc = n_size;
    else
      nc  = l3 / (2 * mc * sizeof(NumericT));
    nc += nr - (nc%nr); // nc must be divisible by nr

    /*    //DEBUG
          kc = 8;//DEBUG
          mc = kc;//DEBUG
          nc = 16;//DEBUG*/
  }
}//viennacl
#endif
