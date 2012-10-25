/* =========================================================================
   Copyright (c) 2010-2011, Institute for Microelectronics,
                            Institute for Analysis and Scientific Computing,
                            TU Wien.

                            -----------------
                  ViennaCL - The Vienna Computing Library
                            -----------------

   Project Head:    Karl Rupp                   rupp@iue.tuwien.ac.at
               
   (A list of authors and contributors can be found in the PDF manual)

   License:         MIT (X11), see file LICENSE in the base directory
============================================================================= */

//
// *** System
//
#include <iostream>

//
// *** Boost
//
#include <boost/numeric/ublas/io.hpp>
#include <boost/numeric/ublas/vector.hpp>

//
// *** ViennaCL
//
//#define VIENNACL_DEBUG_ALL
#define VIENNACL_HAVE_UBLAS 1
#include "viennacl/vector.hpp"
#include "viennacl/linalg/inner_prod.hpp"
#include "viennacl/linalg/norm_1.hpp"
#include "viennacl/linalg/norm_2.hpp"
#include "viennacl/linalg/norm_inf.hpp"


using namespace boost::numeric;

//
// -------------------------------------------------------------
//
template <class TYPE>
bool readVectorFromFile(const std::string & filename, boost::numeric::ublas::vector<TYPE> & vec)
{
  std::ifstream file(filename.c_str());

  if (!file) return false;

  unsigned int size;
  file >> size;
  
  if (size > 30000)  //keep execution times short
    size = 30000;
  vec.resize(size);

  for (unsigned int i = 0; i < size; ++i)
  {
    TYPE element;
    file >> element;
    vec[i] = element;
  }

  return true;
}

//
// -------------------------------------------------------------
//
template <typename ScalarType>
ScalarType diff(ScalarType & s1, ScalarType & s2) 
{
   if (s1 != s2)
      return (s1 - s2) / std::max(fabs(s1), fabs(s2));
   return 0;
}
//
// -------------------------------------------------------------
//
template <typename ScalarType>
ScalarType diff(ScalarType & s1, viennacl::scalar<ScalarType> & s2) 
{
   if (s1 != s2)
      return (s1 - s2) / std::max(fabs(s1), fabs(s2));
   return 0;
}
//
// -------------------------------------------------------------
//
template <typename ScalarType>
ScalarType diff(ScalarType & s1, viennacl::entry_proxy<ScalarType> const& s2) 
{
   if (s1 != s2)
      return (s1 - s2) / std::max(fabs(s1), fabs(s2));
   return 0;
}
//
// -------------------------------------------------------------
//
template <typename ScalarType>
ScalarType diff(ublas::vector<ScalarType> & v1, viennacl::vector<ScalarType> & v2)
{
   ublas::vector<ScalarType> v2_cpu(v2.size());
   viennacl::fast_copy(v2.begin(), v2.end(), v2_cpu.begin());

   for (unsigned int i=0;i<v1.size(); ++i)
   {
      if ( std::max( fabs(v2_cpu[i]), fabs(v1[i]) ) > 0 )
         v2_cpu[i] = fabs(v2_cpu[i] - v1[i]) / std::max( fabs(v2_cpu[i]), fabs(v1[i]) );
      else
         v2_cpu[i] = 0.0;
   }

   return ublas::norm_inf(v2_cpu);
}
//
// -------------------------------------------------------------
//
template< typename NumericT, typename Epsilon >
int test(Epsilon const& epsilon, std::string rhsfile, std::string /*resultfile*/)
{
   int retval = EXIT_SUCCESS;

   ublas::vector<NumericT> rhs;
   ublas::vector<NumericT> rhs2;

   if (!readVectorFromFile<NumericT>(rhsfile, rhs)) 
   {
      std::cout << "Error reading RHS file" << std::endl;
      retval = EXIT_FAILURE;
   }
   
   std::cout << "Running tests for vector of size " << rhs.size() << std::endl;

//    ublas::vector<NumericT> result;
//    if (!readVectorFromFile<NumericT>(resultfile, result))  
//    {
//       std::cout << "Error reading Result file" << std::endl;
//       retval = EXIT_FAILURE;
//    }

   viennacl::vector<NumericT> vcl_rhs(rhs.size());
   viennacl::fast_copy(rhs.begin(), rhs.end(), vcl_rhs.begin());
   viennacl::vector<NumericT> vcl_rhs2(rhs.size()); 
   viennacl::copy(rhs.begin(), rhs.end(), vcl_rhs2.begin());
   
   NumericT                    cpu_result;
   viennacl::scalar<NumericT>  gpu_result;
   // --------------------------------------------------------------------------
   std::cout << "Testing inner_prod..." << std::endl;
   cpu_result = viennacl::linalg::inner_prod(rhs, rhs);
   gpu_result = viennacl::linalg::inner_prod(vcl_rhs, vcl_rhs);

   if( fabs(diff(cpu_result, gpu_result)) > epsilon )
   {
      std::cout << "# Error at operation: inner product" << std::endl;
      std::cout << "  diff: " << fabs(diff(cpu_result, gpu_result)) << std::endl;
      retval = EXIT_FAILURE;
   }
   
   // --------------------------------------------------------------------------
   std::cout << "Testing norm_1..." << std::endl;
   cpu_result = norm_1(rhs);
   gpu_result = viennacl::linalg::norm_1(vcl_rhs);

   if( fabs(diff(cpu_result, gpu_result)) > epsilon )
   {
      std::cout << "# Error at operation: norm-1" << std::endl;
      std::cout << "  diff: " << fabs(diff(cpu_result, gpu_result)) << std::endl;
      retval = EXIT_FAILURE;
   }
   // --------------------------------------------------------------------------
   std::cout << "Testing norm_2..." << std::endl;
   cpu_result = norm_2(rhs);
   gpu_result = viennacl::linalg::norm_2(vcl_rhs);

   if( fabs(diff(cpu_result, gpu_result)) > epsilon )
   {
      std::cout << "# Error at operation: norm-2" << std::endl;
      std::cout << "  diff: " << fabs(diff(cpu_result, gpu_result)) << std::endl;
      retval = EXIT_FAILURE;
   }
   // --------------------------------------------------------------------------
   std::cout << "Testing norm_inf..." << std::endl;
   cpu_result = norm_inf(rhs);
   gpu_result = viennacl::linalg::norm_inf(vcl_rhs);

   if( fabs(diff(cpu_result, gpu_result)) > epsilon )
   {
      std::cout << "# Error at operation: norm-inf" << std::endl;
      std::cout << "  diff: " << fabs(diff(cpu_result, gpu_result)) << std::endl;
      retval = EXIT_FAILURE;
   }
   // --------------------------------------------------------------------------
   std::cout << "Testing index_norm_inf..." << std::endl;
   size_t cpu_index = index_norm_inf(rhs);
   size_t gpu_index = viennacl::linalg::index_norm_inf(vcl_rhs);

   if( cpu_index != gpu_index )
   {
      std::cout << "# Error at operation: index norm-inf" << std::endl;
      std::cout << "  cpu-index: " << cpu_index << " vs. gpu-index: " << gpu_index << std::endl;
      retval = EXIT_FAILURE;
   }
   // --------------------------------------------------------------------------
   cpu_result = rhs[index_norm_inf(rhs)];
   gpu_result = vcl_rhs[viennacl::linalg::index_norm_inf(vcl_rhs)];

   if( fabs(diff(cpu_result, gpu_result)) > epsilon )
   {
      std::cout << "# Error at operation: value norm-inf" << std::endl;
      std::cout << "  diff: " << fabs(diff(cpu_result, gpu_result)) << std::endl;
      retval = EXIT_FAILURE;
   }
   // --------------------------------------------------------------------------
   ublas::vector<NumericT> x = rhs;
   ublas::vector<NumericT> y = rhs;
   ublas::vector<NumericT> t = rhs;
   t.assign (NumericT(1.1) * x + NumericT(2.3) * y),
   y.assign (- NumericT(2.3) * x + NumericT(1.1) * y),
   x.assign (t);
//   cpu_result = norm_inf(x); 

   copy(rhs, vcl_rhs);
   copy(rhs, vcl_rhs2);
   std::cout << "Testing plane_rotation..." << std::endl;
   viennacl::linalg::plane_rotation(vcl_rhs, vcl_rhs2, NumericT(1.1), NumericT(2.3));
   //gpu_result = viennacl::linalg::norm_inf(vcl_rhs);

   if( fabs(diff(x, vcl_rhs)) > epsilon )
   {
      std::cout << "# Error at operation: plane rotation" << std::endl;
      std::cout << "  diff: " << fabs(diff(x, vcl_rhs)) << std::endl;
      retval = EXIT_FAILURE;
   }
   // --------------------------------------------------------------------------
   viennacl::copy(rhs, vcl_rhs);
   
   std::cout << "Testing cpu_assignments..." << std::endl;
   NumericT val = static_cast<NumericT>(1e-3);
   for (size_t i=0; i < rhs.size(); ++i)
     rhs(i) = val;

   if( fabs(diff(val, rhs(0))) > epsilon )
   {
      std::cout << "# Error at operation: cpu assignment" << std::endl;
      std::cout << "  diff: " << fabs(diff(val, rhs(0))) << std::endl;
      retval = EXIT_FAILURE;
   }

   std::cout << "Testing gpu_assignments..." << std::endl;
   for (size_t i=0; i < vcl_rhs.size(); ++i)
     vcl_rhs(i) = val;

   if( fabs(diff(rhs, vcl_rhs)) > epsilon )
   {
      std::cout << "# Error at operation: gpu assignment" << std::endl;
      std::cout << "  diff: " << fabs(diff(val, vcl_rhs(0))) << std::endl;
      retval = EXIT_FAILURE;
   }
   
   
   //
   // multiplication and division of vectors by scalars
   //
   copy(rhs.begin(), rhs.end(), vcl_rhs.begin());
   rhs2 = rhs;

   std::cout << "Testing scaling with CPU scalar..." << std::endl;
   NumericT alpha = static_cast<NumericT>(3.1415);
   viennacl::scalar<NumericT> gpu_alpha = alpha;

   rhs     *= alpha;
   vcl_rhs *= alpha;
  
   if( fabs(diff(rhs, vcl_rhs)) > epsilon )
   {
      std::cout << "# Error at operation: stretching with CPU scalar" << std::endl;
      std::cout << "  diff: " << fabs(diff(cpu_result, gpu_result)) << std::endl;
      retval = EXIT_FAILURE;
   }  

   std::cout << "Testing scaling with GPU scalar..." << std::endl;
   copy(rhs2.begin(), rhs2.end(), vcl_rhs.begin());
   vcl_rhs *= gpu_alpha;
  
   if( fabs(diff(rhs, vcl_rhs)) > epsilon )
   {
      std::cout << "# Error at operation: stretching with GPU scalar" << std::endl;
      std::cout << "  diff: " << fabs(diff(rhs, vcl_rhs)) << std::endl;
      retval = EXIT_FAILURE;
   }  

   NumericT beta  = static_cast<NumericT>(1.4153);
   viennacl::scalar<NumericT> gpu_beta = beta;
   rhs2 = rhs;
  
   std::cout << "Testing shrinking with CPU scalar..." << std::endl;
   rhs     /= beta;
   vcl_rhs /= beta;  

   if( fabs(diff(rhs, vcl_rhs)) > epsilon )
   {
      std::cout << "# Error at operation: shrinking with CPU scalar" << std::endl;
      std::cout << "  diff: " << fabs(diff(rhs, vcl_rhs)) << std::endl;
      retval = EXIT_FAILURE;
   }    
   
   std::cout << "Testing shrinking with GPU scalar..." << std::endl;
   copy(rhs2.begin(), rhs2.end(), vcl_rhs.begin());
   vcl_rhs /= gpu_beta;

   if( fabs(diff(rhs, vcl_rhs)) > epsilon )
   {
      std::cout << "# Error at operation: shrinking with GPU scalar" << std::endl;
      std::cout << "  diff: " << fabs(diff(rhs, vcl_rhs)) << std::endl;
      retval = EXIT_FAILURE;
   }    
   


   //
   // add and inplace_add of vectors
   //
   std::cout << "Testing add on vector..." << std::endl;
   rhs2 = 42.0 * rhs;
   copy(rhs.begin(), rhs.end(), vcl_rhs.begin());
   copy(rhs2.begin(), rhs2.end(), vcl_rhs2.begin());

   rhs     = rhs + rhs2;
   vcl_rhs = vcl_rhs + vcl_rhs2;

   if( fabs(diff(rhs, vcl_rhs)) > epsilon )
   {
      std::cout << "# Error at operation: add on vector" << std::endl;
      std::cout << "  diff: " << fabs(diff(rhs, vcl_rhs)) << std::endl;
      retval = EXIT_FAILURE;
   }       

   std::cout << "Testing inplace-add on vector..." << std::endl;
   rhs2 = 42.0 * rhs;
   copy(rhs.begin(), rhs.end(), vcl_rhs.begin());
   copy(rhs2.begin(), rhs2.end(), vcl_rhs2.begin());

   rhs     += rhs2;
   vcl_rhs += vcl_rhs2;

   if( fabs(diff(rhs, vcl_rhs)) > epsilon )
   {
      std::cout << "# Error at operation: inplace-add on vector" << std::endl;
      std::cout << "  diff: " << fabs(diff(rhs, vcl_rhs)) << std::endl;
      retval = EXIT_FAILURE;
   }       

   //
   // subtract and inplace_subtract of vectors
   //
   std::cout << "Testing sub on vector..." << std::endl;
   rhs2 = 42.0 * rhs;
   copy(rhs.begin(), rhs.end(), vcl_rhs.begin());
   copy(rhs2.begin(), rhs2.end(), vcl_rhs2.begin());

   rhs     = rhs - rhs2;
   vcl_rhs = vcl_rhs - vcl_rhs2;

   if( fabs(diff(rhs, vcl_rhs)) > epsilon )
   {
      std::cout << "# Error at operation: sub on vector" << std::endl;
      std::cout << "  diff: " << fabs(diff(rhs, vcl_rhs)) << std::endl;
      retval = EXIT_FAILURE;
   }       

   std::cout << "Testing inplace-sub on vector..." << std::endl;
   rhs2 = 42.0 * rhs;
   copy(rhs.begin(), rhs.end(), vcl_rhs.begin());
   copy(rhs2.begin(), rhs2.end(), vcl_rhs2.begin());

   rhs     += rhs2;
   vcl_rhs += vcl_rhs2;

   if( fabs(diff(rhs, vcl_rhs)) > epsilon )
   {
      std::cout << "# Error at operation: inplace-sub on vector" << std::endl;
      std::cout << "  diff: " << fabs(diff(rhs, vcl_rhs)) << std::endl;
      retval = EXIT_FAILURE;
   }       


   
   //
   // multiply-add and multiply-subtract
   //
   std::cout << "Testing multiply-add on vector with CPU scalar..." << std::endl;
   rhs2 = 42.0 * rhs;
   copy(rhs.begin(), rhs.end(), vcl_rhs.begin());
   copy(rhs2.begin(), rhs2.end(), vcl_rhs2.begin());

   rhs     = rhs + alpha * rhs2;
   vcl_rhs = vcl_rhs + alpha * vcl_rhs2;

   if( fabs(diff(rhs, vcl_rhs)) > epsilon )
   {
      std::cout << "# Error at operation: multiply add with CPU scalar" << std::endl;
      std::cout << "  diff: " << fabs(diff(rhs, vcl_rhs)) << std::endl;
      retval = EXIT_FAILURE;
   }       


   std::cout << "Testing inplace multiply-add on vector with CPU scalar..." << std::endl;
   rhs2 = 42.0 * rhs;
   copy(rhs.begin(), rhs.end(), vcl_rhs.begin());
   copy(rhs2.begin(), rhs2.end(), vcl_rhs2.begin());

   rhs     += alpha * rhs2;
   vcl_rhs += alpha * vcl_rhs2;

   if( fabs(diff(rhs, vcl_rhs)) > epsilon )
   {
      std::cout << "# Error at operation: inplace multiply add with CPU scalar" << std::endl;
      std::cout << "  diff: " << fabs(diff(rhs, vcl_rhs)) << std::endl;
      retval = EXIT_FAILURE;
   }       

   std::cout << "Testing multiply-add on vector with GPU scalar..." << std::endl;
   rhs2 = 42.0 * rhs;
   copy(rhs.begin(), rhs.end(), vcl_rhs.begin());
   copy(rhs2.begin(), rhs2.end(), vcl_rhs2.begin());
   
   rhs = rhs + alpha * rhs2;
   vcl_rhs = vcl_rhs + gpu_alpha * vcl_rhs2;

   if( fabs(diff(rhs, vcl_rhs)) > epsilon )
   {
      std::cout << "# Error at operation: multiply add with GPU scalar" << std::endl;
      std::cout << "  diff: " << fabs(diff(rhs, vcl_rhs)) << std::endl;
      retval = EXIT_FAILURE;
   }       
   
   std::cout << "Testing inplace multiply-add on vector with GPU scalar..." << std::endl;
   rhs2 = 42.0 * rhs;
   copy(rhs.begin(), rhs.end(), vcl_rhs.begin());
   copy(rhs2.begin(), rhs2.end(), vcl_rhs2.begin());
   
   rhs += alpha * rhs2;
   vcl_rhs += gpu_alpha * vcl_rhs2;

   if( fabs(diff(rhs, vcl_rhs)) > epsilon )
   {
      std::cout << "# Error at operation: inplace multiply add with GPU scalar" << std::endl;
      std::cout << "  diff: " << fabs(diff(rhs, vcl_rhs)) << std::endl;
      retval = EXIT_FAILURE;
   }       
   


   //
   // multiply-subtract
   //
   std::cout << "Testing multiply-subtract on vector with CPU scalar..." << std::endl;
   rhs2 = 42.0 * rhs;
   copy(rhs.begin(), rhs.end(), vcl_rhs.begin());
   copy(rhs2.begin(), rhs2.end(), vcl_rhs2.begin());

   rhs     = rhs - alpha * rhs2;
   vcl_rhs = vcl_rhs - alpha * vcl_rhs2;

   if( fabs(diff(rhs, vcl_rhs)) > epsilon )
   {
      std::cout << "# Error at operation: multiply-subtract with CPU scalar" << std::endl;
      std::cout << "  diff: " << fabs(diff(rhs, vcl_rhs)) << std::endl;
      retval = EXIT_FAILURE;
   }       


   std::cout << "Testing inplace multiply-subtract on vector with CPU scalar..." << std::endl;
   rhs2 = 42.0 * rhs;
   copy(rhs.begin(), rhs.end(), vcl_rhs.begin());
   copy(rhs2.begin(), rhs2.end(), vcl_rhs2.begin());

   rhs     -= alpha * rhs2;
   vcl_rhs -= alpha * vcl_rhs2;

   if( fabs(diff(rhs, vcl_rhs)) > epsilon )
   {
      std::cout << "# Error at operation: inplace multiply subtract with CPU scalar" << std::endl;
      std::cout << "  diff: " << fabs(diff(rhs, vcl_rhs)) << std::endl;
      retval = EXIT_FAILURE;
   }       

   std::cout << "Testing multiply-subtract on vector with GPU scalar..." << std::endl;
   rhs2 = 42.0 * rhs;
   copy(rhs.begin(), rhs.end(), vcl_rhs.begin());
   copy(rhs2.begin(), rhs2.end(), vcl_rhs2.begin());
   
   rhs     = rhs - alpha * rhs2;
   vcl_rhs = vcl_rhs - gpu_alpha * vcl_rhs2;

   if( fabs(diff(rhs, vcl_rhs)) > epsilon )
   {
      std::cout << "# Error at operation: multiply subtract with GPU scalar" << std::endl;
      std::cout << "  diff: " << fabs(diff(rhs, vcl_rhs)) << std::endl;
      retval = EXIT_FAILURE;
   }       
   
   std::cout << "Testing inplace multiply-subtract on vector with GPU scalar..." << std::endl;
   rhs2 = 42.0 * rhs;
   copy(rhs.begin(), rhs.end(), vcl_rhs.begin());
   copy(rhs2.begin(), rhs2.end(), vcl_rhs2.begin());
   
   rhs -= alpha * rhs2;
   vcl_rhs -= gpu_alpha * vcl_rhs2;

   if( fabs(diff(rhs, vcl_rhs)) > epsilon )
   {
      std::cout << "# Error at operation: inplace multiply subtract with GPU scalar" << std::endl;
      std::cout << "  diff: " << fabs(diff(rhs, vcl_rhs)) << std::endl;
      retval = EXIT_FAILURE;
   }       
   
   
   
   //
   // Misc stuff
   //
   rhs2 = rhs;
   copy(rhs.begin(), rhs.end(), vcl_rhs.begin());
   copy(rhs2.begin(), rhs2.end(), vcl_rhs2.begin());

   std::cout << "Testing several vector additions..." << std::endl;
   rhs     = rhs2 + rhs + rhs2;
   vcl_rhs = vcl_rhs2 + vcl_rhs + vcl_rhs2;
   
   if( fabs(diff(rhs, vcl_rhs)) > epsilon )
   {
      std::cout << "# Error at operation: several additions" << std::endl;
      std::cout << "  diff: " << fabs(diff(rhs, vcl_rhs)) << std::endl;
      retval = EXIT_FAILURE;
   }          
   
   
   
   //
   // Complicated expressions (for ensuring the operator overloads work correctly)
   //
   copy(vcl_rhs.begin(), vcl_rhs.end(), rhs2.begin());
   copy(rhs.begin(), rhs.end(), vcl_rhs.begin());
   copy(rhs2.begin(), rhs2.end(), vcl_rhs2.begin());
   rhs2 = rhs;

   std::cout << "Testing complicated vector expression with CPU scalar..." << std::endl;
   rhs     = beta * (rhs - alpha*rhs2);
   vcl_rhs = beta * (vcl_rhs - alpha*vcl_rhs2);

   if( fabs(diff(rhs, vcl_rhs)) > epsilon )
   {
      std::cout << "# Error at operation: advanced mul diff with CPU scalars" << std::endl;
      std::cout << "  diff: " << fabs(diff(rhs, vcl_rhs)) << std::endl;
      retval = EXIT_FAILURE;
   }          
   
   std::cout << "Testing complicated vector expression with GPU scalar..." << std::endl;
   copy(rhs2.begin(), rhs2.end(), vcl_rhs.begin());
   vcl_rhs = gpu_beta * (vcl_rhs - gpu_alpha*vcl_rhs2);

   if( fabs(diff(rhs, vcl_rhs)) > epsilon )
   {
      std::cout << "# Error at operation: advanced mul diff with GPU scalars" << std::endl;
      std::cout << "  diff: " << fabs(diff(rhs, vcl_rhs)) << std::endl;
      retval = EXIT_FAILURE;
   }          
   
   // --------------------------------------------------------------------------      
   copy(vcl_rhs.begin(), vcl_rhs.end(), rhs2.begin());
   rhs2 = rhs;
   rhs2 *= 3.0;
   copy(rhs.begin(), rhs.end(), vcl_rhs.begin());
   copy(rhs2.begin(), rhs2.end(), vcl_rhs2.begin());

   std::cout << "Testing swap..." << std::endl;
   swap(rhs, rhs2);
   swap(vcl_rhs, vcl_rhs2);

   if( fabs(diff(rhs, vcl_rhs)) > epsilon )
   {
      std::cout << "# Error at operation: swap" << std::endl;
      std::cout << "  diff: " << fabs(diff(rhs, vcl_rhs)) << std::endl;
      retval = EXIT_FAILURE;
   }          
   // --------------------------------------------------------------------------         
   rhs2 = 5.0 * rhs;
   copy(rhs.begin(), rhs.end(), vcl_rhs.begin());
   copy(rhs2.begin(), rhs2.end(), vcl_rhs2.begin());

   std::cout << "Testing another complicated vector expression with CPU scalars..." << std::endl;
   rhs     = rhs2 / alpha + beta * (rhs - alpha*rhs2);
   vcl_rhs = vcl_rhs2 / alpha + beta * (vcl_rhs - alpha*vcl_rhs2);

   if( fabs(diff(rhs, vcl_rhs)) > epsilon )
   {
      std::cout << "# Error at operation: complex vector operations with CPU scalars" << std::endl;
      std::cout << "  diff: " << fabs(diff(rhs, vcl_rhs)) << std::endl;
      retval = EXIT_FAILURE;
   }             
   
   std::cout << "Testing another complicated vector expression with GPU scalars..." << std::endl;
   copy(rhs.begin(), rhs.end(), vcl_rhs.begin());
   copy(rhs2.begin(), rhs2.end(), vcl_rhs2.begin());
   rhs     = rhs2 / alpha + beta * (rhs - alpha*rhs2);
   vcl_rhs = vcl_rhs2 / gpu_alpha + gpu_beta * (vcl_rhs - gpu_alpha*vcl_rhs2);

   if( fabs(diff(rhs, vcl_rhs)) > epsilon )
   {
      std::cout << "# Error at operation: complex vector operations with GPU scalars" << std::endl;
      std::cout << "  diff: " << fabs(diff(rhs, vcl_rhs)) << std::endl;
      retval = EXIT_FAILURE;
   }             

   
   std::cout << "Testing lenghty sum of scaled vectors..." << std::endl;
   copy(rhs.begin(), rhs.end(), vcl_rhs.begin());
   copy(rhs2.begin(), rhs2.end(), vcl_rhs2.begin());
   rhs     = rhs2 / alpha + beta * rhs - alpha*rhs2 + beta * rhs - alpha * rhs;
   vcl_rhs = vcl_rhs2 / gpu_alpha + gpu_beta * vcl_rhs - alpha*vcl_rhs2 + beta * vcl_rhs - alpha * vcl_rhs;

   if( fabs(diff(rhs, vcl_rhs)) > epsilon )
   {
      std::cout << "# Error at operation: complex vector operations with GPU scalars" << std::endl;
      std::cout << "  diff: " << fabs(diff(rhs, vcl_rhs)) << std::endl;
      retval = EXIT_FAILURE;
   }             
   
   // --------------------------------------------------------------------------            
   return retval;
}
//
// -------------------------------------------------------------
//
int main()
{
   std::cout << std::endl;
   std::cout << "----------------------------------------------" << std::endl;
   std::cout << "----------------------------------------------" << std::endl;
   std::cout << "## Test :: Vector" << std::endl;
   std::cout << "----------------------------------------------" << std::endl;
   std::cout << "----------------------------------------------" << std::endl;
   std::cout << std::endl;

   int retval = EXIT_SUCCESS;

   std::string rhsfile("../../examples/testdata/rhs65025.txt");
   std::string resultfile("../../examples/testdata/result65025.txt");

   std::cout << std::endl;
   std::cout << "----------------------------------------------" << std::endl;
   std::cout << std::endl;
   {
      typedef float NumericT;
      NumericT epsilon = static_cast<NumericT>(1.0E-4);
      std::cout << "# Testing setup:" << std::endl;
      std::cout << "  eps:     " << epsilon << std::endl;
      std::cout << "  numeric: float" << std::endl;
      retval = test<NumericT>(epsilon, rhsfile, resultfile);
      if( retval == EXIT_SUCCESS )
         std::cout << "# Test passed" << std::endl;
      else
         return retval;
   }
   std::cout << std::endl;
   std::cout << "----------------------------------------------" << std::endl;
   std::cout << std::endl;
   if( viennacl::ocl::current_device().double_support() )
   {
      {
         typedef double NumericT;
         NumericT epsilon = 1.0E-10;
         std::cout << "# Testing setup:" << std::endl;
         std::cout << "  eps:     " << epsilon << std::endl;
         std::cout << "  numeric: double" << std::endl;
         retval = test<NumericT>(epsilon, rhsfile, resultfile);
         if( retval == EXIT_SUCCESS )
           std::cout << "# Test passed" << std::endl;
         else
           return retval;
      }
      std::cout << std::endl;
      std::cout << "----------------------------------------------" << std::endl;
      std::cout << std::endl;
      {
         typedef double NumericT;
         NumericT epsilon = 1.0E-11;
         std::cout << "# Testing setup:" << std::endl;
         std::cout << "  eps:     " << epsilon << std::endl;
         std::cout << "  numeric: double" << std::endl;
         retval = test<NumericT>(epsilon, rhsfile, resultfile);
         if( retval == EXIT_SUCCESS )
           std::cout << "# Test passed" << std::endl;
         else
           return retval;
      }
      std::cout << std::endl;
      std::cout << "----------------------------------------------" << std::endl;
      std::cout << std::endl;
      {
         typedef double NumericT;
         NumericT epsilon = 1.0E-12;
         std::cout << "# Testing setup:" << std::endl;
         std::cout << "  eps:     " << epsilon << std::endl;
         std::cout << "  numeric: double" << std::endl;
         retval = test<NumericT>(epsilon, rhsfile, resultfile);
         if( retval == EXIT_SUCCESS )
           std::cout << "# Test passed" << std::endl;
         else
           return retval;
      }
      std::cout << std::endl;
      std::cout << "----------------------------------------------" << std::endl;
      std::cout << std::endl;
   }
   
  std::cout << std::endl;
  std::cout << "------- Test completed --------" << std::endl;
  std::cout << std::endl;
   
   
   return retval;
}
