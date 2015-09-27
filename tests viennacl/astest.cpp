#include <iostream>

/* CPU-vendor strings */
#define VENDOR_STR_LEN (12)
#define INTEL          "GenuineIntel"
#define AMD            "AuthenticAMD"

/* choose which cache-info is to be read,
 * quotation marks needed by the inline-assembler */
#define AMD_GET_L1      "0x80000005"
#define AMD_GET_L2      "0x80000006"
#define AMD_GET_L3      "0x80000006"
#define INTEL_GET_CACHE "0x00000002"
#define INTEL_GET_L1    "0x00000000"
#define INTEL_GET_L2    "0x00000001"
#define INTEL_GET_L3    "0x00000002"

/* auxilary for cache-info arrays */
#define AMD_CACHE_INFO_SIZE     (3)
#define AMD_L1_CACHE_SIZE_IDX   (0)
#define AMD_L2_CACHE_SIZE_IDX   (1)
#define AMD_L3_CACHE_SIZE_IDX   (2)
#define INTEL_CACHE_INFO_SIZE   (3)
#define INTEL_L1_CACHE_SIZE_IDX (0)
#define INTEL_L2_CACHE_SIZE_IDX (1)
#define INTEL_L3_CACHE_SIZE_IDX (2)

/* mask accodording bits obtained by calling 'cpuid' */
#define AMD_L1_CACHE_SIZE_MASK (0xFF000000)
#define AMD_L2_CACHE_SIZE_MASK (0xFFFF0000)
#define AMD_L3_CACHE_SIZE_MASK (0xFFFC0000)
#define INTEL_CACHE_WAYS       (0xFFC0000000000000)
#define INTEL_CACHE_PARTITIONS (0x003FF00000000000)
#define INTEL_CACHE_LINE_SIZE  (0x00000FFF00000000)
#define INTEL_CACHE_SETS       (0x00000000FFFFFFFF)

  /* calculates cache size, see AMDs "CPUID Specification" */
#define AMD_CALC_L1_CACHE_SIZE(a) ( (a[AMD_L1_CACHE_SIZE_IDX] & AMD_L1_CACHE_SIZE_MASK) >> 24 ) * KILO
#define AMD_CALC_L2_CACHE_SIZE(a) ( (a[AMD_L2_CACHE_SIZE_IDX] & AMD_L2_CACHE_SIZE_MASK) >> 16 ) * KILO
#define AMD_CALC_L3_CACHE_SIZE(a) ( (a[AMD_L3_CACHE_SIZE_IDX] & AMD_L3_CACHE_SIZE_MASK) >> 18 ) * 512 * KILO

/* calculates cache size, Formula found in intels "Intel 64 and IA-32 Architectures Software Developer Manual", p. 198 (pdf:264) */
#define INTEL_CALC_CACHE_SIZE(a) ((a & INTEL_CACHE_WAYS) +1) * ((a & INTEL_CACHE_PARTITIONS) +1) * ((a & INTEL_CACHE_LINE_SIZE)+1) * ((a & INTEL_CACHE_SETS)+1)

/* misc */
#define KILO (1024)
#define MEGA (1024*1024)

void get_cache_amd(uint32_t *bla)
{
  /* l1 = %rdi, l2 = %rsi, l3 = %rdx */
    __asm__
    (

      "movl  $" AMD_GET_L1 ", %%eax \n\t"
      "cpuid                    \n\t"
      "movl  %%ecx, (%%rdi)     \n\t"

      "movl  $" AMD_GET_L2 ", %%eax \n\t"
      "cpuid                    \n\t"
      "movl  %%ecx, 4(%%rdi)     \n\t"

      "movl  $" AMD_GET_L3 ", %%eax \n\t"
      "cpuid                    \n\t"
      "movl  %%ecx, 8(%%rdi)     \n\t"
        
      :                                      
      :
      : "%eax", "%ecx", "%ebx"
      );
  //    *l1 = 16 * 1024; //16KB
  //*l2 =  1 * 1024; // 1KB
  //*l3 =  0 * 1024; // 0KB*/
}

void get_cache_intel(uint64_t *l1l2l3)
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
      "movl  %%ebx,  8(%%rdi)          \n\t"
      "movl  %%ecx, 12(%%rdi)          \n\t"
      /* get L3-cache info */
      "movl  $" INTEL_GET_CACHE ", %%eax \n\t"
      "movl  $" INTEL_GET_L3 "   , %%ecx \n\t"
      "cpuid                         \n\t"
      "movl  %%ebx, 16(%%rdi)          \n\t"
      "movl  %%ecx, 20(%%rdi)          \n\t"
      :
      :
      :
      );
}


int main ()
{
  //  uint8_t vendor[4];

  uint32_t bla[3];

  //  std::cout << l1 << std::endl;
    
  get_cache_amd(bla);
  
  //std::cout << (int *)vendor <<std::endl;
  
  for (int i=0; i<4; ++i)
    //    std::cout << (int)l1[i] << " " << (int)l2[i] << " " << (int)l3[i] << " " << std::endl;
  std::cout << std::endl;

  return 0;
}
 
      
