#ifndef VIENNACL_LINALG_HOST_BASED_GET_BLOCK_SIZES_HPP_
#define VIENNACL_LINALG_HOST_BASED_GET_BLOCK_SIZES_HPP_

#include "viennacl/forwards.h"
#include "viennacl/traits/size.hpp"

/* sizes for different architectures in bytes */
#define AVX_REG_SIZE     (256/8)
#define AVX_512_REG_SIZE (512/8)
#define SSE_REG_SIZE     (128/8)

namespace viennacl
{
  void get_cache_sizes(int & l1, int & l2, int & l3)
  {

    //placeholder, TODO: read out cpuid etc.
    if (true)
    {
      l1 = 16 * 1024;
      l2 =  1 * 1024;
      l3 = 0;
      /*
        l1 =  32 * 1024;
        l2 = 256 * 1024;
        l3 =   8 * 1024 * 1024;*/
    }
    else
    {
      /* if we could not read out cache sizes assume (small) sizes */
      l1 =  8 * 1024;        // 8KB
      l2 = 16 * 1024;        //16KB
      l3 =  1 * 1024 * 1024; // 1MB
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
    static bool cache_sizes_unknown;

    /* hardware won't change during run-time (hopefully)
     * ==> determine cache sizes only once */
    if (cache_sizes_unknown)
    {
      get_cache_sizes(l1, l2, l3);
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
