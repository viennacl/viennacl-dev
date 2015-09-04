#include "viennacl/forwards.h"
#include "viennacl/matrix.hpp"
#include "viennacl/matrix_proxy.hpp"

#include <stdlib.h>

int main()
{
  viennacl::matrix<double,viennacl::row_major> a(2,3);

  a(0,0) = 0.42;
  a(0,1) = 0.33;
  a(1,2) = 0.12;

  viennacl::matrix<double,viennacl::row_major> at = trans(a);

  std::cout << "a sizes " << a.size1() << " "
            << a.size2() << " " << a.internal_size1() << " " << a.internal_size2() << std::endl;

  std::cout << "at sizes " << at.size1() << " " << at.size2() << " " 
            << at.internal_size1() << " " << at.internal_size2() << std::endl;

  double * p_a = viennacl::linalg::host_based::detail::extract_raw_pointer<double>(a);
  double * p_at = viennacl::linalg::host_based::detail::extract_raw_pointer<double>(at);

  int temp = 0;

  
  for (size_t i=0; i<at.internal_size(); ++i)
  {
    temp += p_a[i] - p_at[i];
  }
  std::cout << temp << std::endl;

  //  /*
   std::cout << "elems of a:" << std::endl;
  for (size_t i=0; i<a.internal_size(); ++i)
  {
    std::cout << p_a[i] << ",";
    }
  std::cout << "elems of at:" << std::endl;
  for (size_t i=0; i<at.internal_size(); ++i)
  {
    std::cout << p_at[i] << ",";
    }
  //  */
  /*  for (size_t i=0; i<a.size1(); ++i)
  {
    std::cout << "(";
    for (size_t j=0; j<a.size2(); ++j)
    {
      std::cout << p_a[i*internal_size1 + j] << ",";
    }
    std::cout << ")";
  }

  std::cout << std::endl << "elems of a:" << std::endl;
  for (size_t i=0; i<at.size1(); ++i)
  {
    std::cout << "(";
    for (size_t j=0; j<at.size2(); ++j)
    {
      std::cout << p_at[i*internal_size1 + j] << ",";
    }
    std::cout << ")";
    }*/
  //std::cout << "a is " << a << std::endl;
  //std::cout << "at is " << at << std::endl;

}
