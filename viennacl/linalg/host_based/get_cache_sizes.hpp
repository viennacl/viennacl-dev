//this was inspired by the euqivalent code of the Eigen library

#ifndef VIENNACL_LINALG_HOST_BASED_GET_CACHE_SIZES_HPP_
#define VIENNACL_LINALG_HOST_BASED_GET_CACHE_SIZES_HPP_

#include "viennacl/forwards.h"
#include "viennacl/traits/size.hpp"
namespace viennacl
{
void get_cache_sizes(int & l1, int & l2, int & l3)
{

//placeholder, TODO: read out cpuid etc.
if (true)
{
l1 =  32 * 1024;
l2 = 256 * 1024;
l3 =   8 * 1024 * 1024;
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
void get_block_sizes(vcl_size_t & mc, vcl_size_t & kc, vcl_size_t & nc, vcl_size_t & mr, vcl_size_t & nr)
{
#ifdef VIENNACL_WITH_AVX
#define REG_SIZE (256/8)
//#elif VIENNACL_WITH_AVX_512  //not supported yet
  //#elif VIENNACL_WITH_AVX_KNC  //not supported yet
  //#elif VIENNACL_WITH_SSE      //not supported yet
#else 
#define REG_SIZE (0) // no vectorization
#endif

  /* assuming block in registers should be square */
  mr = ( REG_SIZE == 0 ? 1 : REG_SIZE/sizeof(NumericT) );
  nr = mr;

  int l1, l2, l3;
  get_cache_sizes(l1, l2, l3);
  
  /* caclulating blocksizes for L1 (mc x nr) and L2 (mc * kc) and L3 cache. 
   * Assumed that block in L2 cache should be square and half of cache shuold be empty. */
  // TODO: improve formula? 
  mc  = l1 / (2 * nr * sizeof(NumericT));
  mc += mc % mr; // mc must be divisible by mr 
  nc  = l3 / (2 * mc);
  nc += nc % nr; // nc must be divisible by nr
  kc  = mc;

}
}//viennacl
#endif
