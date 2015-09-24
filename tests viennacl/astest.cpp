#include <iostream>

#define AMD_GET_L1      "0x80000005"
#define AMD_GET_L2      "0x80000006"
#define AMD_GET_L3      "0x80000007"

void get_cache_amd(uint8_t *l1, uint8_t *l2, uint8_t *l3)
{
  /* l1 = %rdi, l2 = %rsi, l3 = %rdx, %rcx*/
  std::cout << (int)l1 << std::endl;
  std::cout << (int)l2 << std::endl;
  std::cout << (int)l3 << std::endl;
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


int main ()
{
  uint8_t vendor[4];

  uint8_t l1[4];
  uint8_t l2[4];
  uint8_t l3[4];

  std::cout << l1 << std::endl;
  std::cout << l2 << std::endl;
  std::cout << l3 << std::endl;
  
  get_cache_amd(l1,l2,l3);
  
  //std::cout << (int *)vendor <<std::endl;
  
  for (int i=0; i<4; ++i)
    std::cout << (int)l1[i] << " " << (int)l2[i] << " " << (int)l3[i] << " " << std::endl;
  std::cout << std::endl;

  return 0;
}
 
      
