#include <iostream>

#define AMD_GET_L1      "0x80000005"
#define AMD_GET_L2      "0x80000006"
#define AMD_GET_L3      "0x80000007"

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
 
      
