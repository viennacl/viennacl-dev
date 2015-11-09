#ifndef VIENNACL_LINALG_HOST_BASED_GET_BLOCK_SIZES_HPP_
#define VIENNACL_LINALG_HOST_BASED_GET_BLOCK_SIZES_HPP_

#include <string.h>
#include "viennacl/forwards.h"
#include "viennacl/traits/size.hpp"

/* defines to quickly change what denominator of each cache level should be used (what fraction, half, quarter etc.) */
#define L1_DENOM_AVX (4)
#define L2_DENOM_AVX (2)
#define L3_DENOM_AVX (2)

#define L1_DENOM_SSE (4)
#define L2_DENOM_SSE (4)
#define L3_DENOM_SSE (2)

#define L1_DENOM_STD (4)
#define L2_DENOM_STD (2)
#define L3_DENOM_STD (2)

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
#define AMD_GET_L1            "0x80000005"
#define AMD_GET_L2            "0x80000006"
#define AMD_GET_L3            "0x80000006"
#define INTEL_GET_CACHE_LEAF2 "0x00000002"
#define INTEL_GET_CACHE_LEAF4 "0x00000004"
#define INTEL_GET_L1          "0x00000000"
#define INTEL_GET_L2          "0x00000002"
#define INTEL_GET_L3          "0x00000004"

/* auxilary for cache-info arrays and etc. */
#define INTEL_CACHE_INFO_SIZE    (6)
#define INTEL_L1_CACHE_SIZE_IDX  (0)
#define INTEL_L2_CACHE_SIZE_IDX  (1)
#define INTEL_L3_CACHE_SIZE_IDX  (2)
#define INTEL_CACHE_TYPE_MASK    (0x000F)
#define INTEL_DATA_CACHE_TYPE    (0x0001)
#define INTEL_UNIFIED_CACHE_TYPE (0x0003)
#define INTEL_NO_MORE_CACHE      (0x0000)

/* mask accodording bits obtained by calling 'cpuid' */
#define AMD_L1_CACHE_SIZE_MASK (0xFF000000)
#define AMD_L2_CACHE_SIZE_MASK (0xFFFF0000)
#define AMD_L3_CACHE_SIZE_MASK (0xFFFC0000)
/*#define INTEL_CACHE_WAYS       (0xFFC0000000000000)
#define INTEL_CACHE_PARTITIONS (0x003FF00000000000)
#define INTEL_CACHE_LINE_SIZE  (0x00000FFF00000000)
#define INTEL_CACHE_SETS       (0x00000000FFFFFFFF)*/

#define INTEL_CACHE_WAYS       (0xFFC00000)
#define INTEL_CACHE_PARTITIONS (0x003FF000)
#define INTEL_CACHE_LINE_SIZE  (0x00000FFF)
#define INTEL_CACHE_SETS       (0xFFFFFFFF)

/* calculates cache size, see AMDs "CPUID Specification" */
#define AMD_CALC_L1_CACHE_SIZE(a) ( (a & AMD_L1_CACHE_SIZE_MASK) >> 24 ) * KILO
#define AMD_CALC_L2_CACHE_SIZE(a) ( (a & AMD_L2_CACHE_SIZE_MASK) >> 16 ) * KILO
#define AMD_CALC_L3_CACHE_SIZE(a) ( (a & AMD_L3_CACHE_SIZE_MASK) >> 18 ) * 512 * KILO

/* calculates cache size, Formula found in intels "Intel 64 and IA-32 Architectures Software Developer Manual", p. 198 (pdf:264) */
#define INTEL_CALC_CACHE_SIZE(a,b)                    \
  ( ((a & INTEL_CACHE_WAYS)      >>(22)) +1 ) *      \
  ( ((a & INTEL_CACHE_PARTITIONS)>>(12)) +1 ) *      \
  ( ((a & INTEL_CACHE_LINE_SIZE) >>( 0)) +1 ) *      \
  (  (b & INTEL_CACHE_SETS)              +1 )

/* misc */
#define KILO  (1024)
#define MEGA  (1024*1024)
#define FALSE (0x0000)
#define TRUE  (0x0001)

namespace viennacl
{
  void get_cache_intel_leaf4(uint32_t *l1l2l3)
  {
    uint32_t tmp[2] = {0};
    uint32_t cache_type = 0x0000;
    uint32_t i = 0x0000;
    vcl_size_t l1l2l3_idx = 0;

    do
    {
      __asm__
        (
          "movl  $" INTEL_GET_CACHE_LEAF4 ", %%eax \n\t"
          "movl  %3   , %%ecx                      \n\t"
          "cpuid                                   \n\t"
          "movl  %%ebx,  %0          \n\t"
          "movl  %%ecx,  %1          \n\t"
          "movl  %%eax,  %2                      \n\t"
          : "=r" (tmp[0]), "=r" (tmp[1]), "=r" (cache_type)
          : "r" (i)
          : "%eax", "%ebx", "%ecx", "%edx", "memory"
          );
	
      cache_type = (cache_type & INTEL_CACHE_TYPE_MASK);

      if ( (cache_type == INTEL_DATA_CACHE_TYPE) || (cache_type == INTEL_UNIFIED_CACHE_TYPE) )
      {
        assert((l1l2l3_idx < 6) && bool("Apparantly more than three data caches! get_cache_sizes() expects three levels at max..."));

        l1l2l3[l1l2l3_idx] = tmp[0];
        l1l2l3[l1l2l3_idx+1] = tmp[1];
        l1l2l3_idx += 2;
      }
      ++i;
    } while (cache_type != INTEL_NO_MORE_CACHE);
  }

  /**
   * @brief reads cpuid-information on Intel CPUs and sets cache sizes accordingly
   */
  void set_cache_intel(vcl_size_t &l1_size, vcl_size_t &l2_size, vcl_size_t &l3_size)
  {
    /* every entry saves the state of one of the registers %eax, %ebx, %ecx and %edx
     * which are represent cache and TLB information after cpuid is called */
    uint32_t registers[4];

    /* is set to true if the cache info can't be read with leaf2,
     * therefore use leaf4 instead */
    bool use_cpuid_leaf4 = false;

    /* try to read out cache information with cpuid leaf 2 */
    __asm__
      (
        "movl  $" INTEL_GET_CACHE_LEAF2 ", %%eax \n\t"
        "cpuid                             \n\t"
        "movl  %%eax, %0            \n\t"
        "movl  %%ebx, %1            \n\t"
        "movl  %%ecx, %2            \n\t"
        "movl  %%edx, %3            \n\t"
        : "=r" (registers[0]), "=r" (registers[1]), "=r" (registers[2]), "=r" (registers[3])
        : 
        : "%eax", "%ebx", "%ecx", "%edx"
        );

    /* Iterate over all bytes of registers %eax, %ebx, %ecx and %edx (stored in registers).
     * Least significant byte of %eax should not be checked as it is always set to 0x01,
     * but 0x01 is a TLB descriptor the default case (do nothing) is matched. */
    for (int i=0; i<4; ++i)
    {
      /* Most significant bit is zero if register conatins valid 1-byte-descriptors */
      if ( (registers[i] >> 31) == 0)
      {
        for (int j=0; j<4; ++j)
        {
          /* iterate over all bytes */
          switch ( (registers[i] & (0xFF000000 >> j*8))>>((3-j)*8) )
          {
          case 0x0000000A: l1_size =   8*KILO; break;
          case 0x0000000C: l1_size =  16*KILO; break;
          case 0x0000000D: l1_size =  16*KILO; break;
          case 0x0000000E: l1_size =  24*KILO; break;
          case 0x0000001D: l2_size = 128*KILO; break;
          case 0x00000021: l2_size = 256*KILO; break;
          case 0x00000022: l3_size = 512*KILO; break;
          case 0x00000023: l3_size =   1*MEGA; break;
          case 0x00000024: l2_size =   1*MEGA; break;
          case 0x00000025: l3_size =   2*MEGA; break;
          case 0x00000029: l3_size =   4*MEGA; break;
          case 0x0000002C: l1_size =  32*KILO; break;
          case 0x00000041: l2_size = 128*KILO; break;
          case 0x00000042: l2_size = 256*KILO; break;
          case 0x00000043: l2_size = 512*KILO; break;
          case 0x00000044: l2_size =   1*MEGA; break;
          case 0x00000045: l2_size =   2*MEGA; break;
          case 0x00000046: l3_size =   4*MEGA; break;
          case 0x00000047: l3_size =   8*MEGA; break;
          case 0x00000048: l2_size =   3*MEGA; break;
          case 0x00000049: l2_size =   4*MEGA; l3_size = 4*MEGA; break;
          case 0x0000004A: l3_size =   6*MEGA; break;
          case 0x0000004B: l3_size =   8*MEGA; break;
          case 0x0000004C: l3_size =  12*MEGA; break;
          case 0x0000004D: l3_size =  16*MEGA; break;
          case 0x0000004E: l2_size =   6*MEGA; break;
          case 0x00000060: l1_size =  16*KILO; break;
          case 0x00000066: l1_size =   8*KILO; break;
          case 0x00000067: l1_size =  16*KILO; break;
          case 0x00000068: l1_size =  32*KILO; break;
          case 0x00000078: l2_size =   1*MEGA; break;
          case 0x00000079: l2_size = 128*KILO; break;
          case 0x0000007A: l2_size = 256*KILO; break;
          case 0x0000007B: l2_size = 512*KILO; break;
          case 0x0000007C: l2_size =   1*MEGA; break;
          case 0x0000007D: l2_size =   2*MEGA; break;
          case 0x0000007F: l2_size = 512*KILO; break;
          case 0x00000080: l2_size = 512*KILO; break;
          case 0x00000082: l2_size = 256*KILO; break;
          case 0x00000083: l2_size = 512*KILO; break;
          case 0x00000084: l2_size =   1*MEGA; break;
          case 0x00000085: l2_size =   2*MEGA; break;
          case 0x00000086: l2_size = 512*KILO; break;
          case 0x00000087: l2_size =   1*MEGA; break;
          case 0x000000D0: l3_size = 512*KILO; break;
          case 0x000000D1: l3_size =   1*MEGA; break;	      
          case 0x000000D2: l3_size =   2*MEGA; break;	        
          case 0x000000D6: l3_size =   1*MEGA; break;	        
          case 0x000000D7: l3_size =   2*MEGA; break;	        
          case 0x000000D8: l3_size =   4*MEGA; break;	        
          case 0x000000DC: l3_size =   1*MEGA + MEGA/2; break;  
          case 0x000000DD: l3_size =   3*MEGA; break;                               
          case 0x000000DE: l3_size =   6*MEGA; break;
          case 0x000000E2: l3_size =   2*MEGA; break;
          case 0x000000E3: l3_size =   4*MEGA; break;
          case 0x000000E4: l3_size =   8*MEGA; break;
          case 0x000000EA: l3_size =  12*MEGA; break;
          case 0x000000EB: l3_size =  18*MEGA; break;
          case 0x000000EC: l3_size =  24*MEGA; break;
          case 0x000000FF: use_cpuid_leaf4 = true; goto leaf4; break;
          default: /* matched TLB (or other) type */ break;
          }
        }
      }
    }
  leaf4:
    if (use_cpuid_leaf4)
    {
      uint32_t l1l2l3[INTEL_CACHE_INFO_SIZE];

      get_cache_intel_leaf4(l1l2l3);
      
      l1_size = INTEL_CALC_CACHE_SIZE(l1l2l3[0], l1l2l3[1]);
      l2_size = INTEL_CALC_CACHE_SIZE(l1l2l3[2], l1l2l3[3]);
      l3_size = INTEL_CALC_CACHE_SIZE(l1l2l3[4], l1l2l3[5]);
    }
  
    return;
  }

  void set_cache_sizes(vcl_size_t &l1_size, vcl_size_t &l2_size, vcl_size_t &l3_size)
  {
    /* used to store CPU-vendor string */
    uint32_t vendor[3] = {0};

    /* get vendor-string */
    __asm__
      (
        "movl  $0, %%eax      \n\t"
        "cpuid                \n\t"
        "movl  %%ebx, %0    \n\t"
        "movl  %%edx, %1   \n\t"
        "movl  %%ecx, %2   \n\t"
        : "=r" (vendor[0]), "=r" (vendor[1]), "=r" (vendor[2])
        : 
        : "%eax", "%ebx", "%edx", "%ecx"
       );
    /* check CPU-Vendor */

    if ( strncmp((char *)vendor, INTEL, VENDOR_STR_LEN) == 0 )
    {
      set_cache_intel(l1_size, l2_size, l3_size);
    }      
    else if ( strncmp((char *)vendor, AMD, VENDOR_STR_LEN) == 0 )
    {
      /* store cache information (size, associativity, etc.)*/
      uint32_t registers[3] = {0};
      
      /* gets cache info and sets sizes in Bytes */
    __asm__
      (
        /* get L1-cache info */
        "movl  $" AMD_GET_L1 ", %%eax \n\t"
        "cpuid                    \n\t"
        "movl  %%ecx, %0     \n\t"
        /* get L2-cache info */
        "movl  $" AMD_GET_L2 ", %%eax \n\t"
        "cpuid                    \n\t"
        "movl  %%ecx, %1     \n\t"
        /* get L3-cache info */
        "movl  $" AMD_GET_L3 ", %%eax \n\t"
        "cpuid                    \n\t"
        "movl  %%edx, %2     \n\t"
        : "=r" (registers[0]), "=r" (registers[1]),"=r" (registers[2])
        : 
        : "%eax", "%ebx", "%ecx", "%edx"
       );
      l1_size = AMD_CALC_L1_CACHE_SIZE(registers[0]);
      l2_size = AMD_CALC_L2_CACHE_SIZE(registers[1]);
      l3_size = AMD_CALC_L3_CACHE_SIZE(registers[2]);
    }
    else
    {
      /* CPU-vendor unknown, not supported or cpuid could not be read out propperly.
         Assume some (small) cache sizes. */
      l1_size =  4 * KILO;
      l2_size =  8 * KILO;
      l3_size =  0 * KILO;
    }
  }

  /**
   * @brief calcluates block sizes from cache sizes and number of available threads
   */
  template<typename NumericT> 
  void get_block_sizes(const vcl_size_t m_size, const vcl_size_t k_size, const vcl_size_t n_size, vcl_size_t & mc, vcl_size_t & kc, vcl_size_t & nc, vcl_size_t & mr, vcl_size_t & nr)
  {
    vcl_size_t l1_denom, l2_denom, l3_denom;

    /* get register-block sizes in bytes and according to NumericT */
#ifdef VIENNACL_WITH_AVX

    mr = 6 + 2*(sizeof(float)/sizeof(NumericT));
    nr = 8 + 8*(sizeof(float)/sizeof(NumericT));
    l1_denom = L1_DENOM_AVX;
    l2_denom = L2_DENOM_AVX;
    l3_denom = L3_DENOM_AVX;
#elif VIENNACL_WITH_SSE

    mr = 6 + 2*(sizeof(float)/sizeof(NumericT));
    nr = 4 + 4*(sizeof(float)/sizeof(NumericT));
    l1_denom = L1_DENOM_SSE;
    l2_denom = L2_DENOM_SSE;
    l3_denom = L3_DENOM_SSE;
#else
    /* standard case */
    mr = 1;
    nr = 1;
    l1_denom = L1_DENOM_STD;
    l2_denom = L2_DENOM_STD;
    l3_denom = L3_DENOM_STD;
#endif

    static vcl_size_t l1 = 0;
    static vcl_size_t l2 = 0;
    static vcl_size_t l3 = 0;
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
    vcl_size_t tmp = l1 / (l1_denom * nr * sizeof(NumericT));
    // TODO: improve formula! 
    if (tmp <= 0)
    {
      kc  = k_size;
    }
    else
    {
      kc  = tmp;
    }
    tmp = l2 / (l2_denom * kc * sizeof(NumericT));

    if (tmp <= 0)
    {
      mc  = m_size;
    }
    else
    {
      mc = tmp;
    }
    mc += mr - (mc%mr); //mc must be divisible by mr

    int num_threads = 1;

#ifdef VIENNACL_WITH_OPENMP 
    num_threads = omp_get_max_threads();
#endif

    tmp = std::min(l3/( l3_denom*num_threads*kc*sizeof(NumericT)), (n_size-1)/num_threads+1);

    if (tmp <= 0)
    {
      nc = (n_size-1)/num_threads+1;
    }
    else
    {
      nc = tmp;
    }
    nc += nr - (nc%nr); // nc must be divisible by nr
  }
}//viennacl
#endif
