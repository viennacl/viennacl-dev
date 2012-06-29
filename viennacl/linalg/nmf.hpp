#ifndef VIENNACL_LINALG_NMF_HPP
#define VIENNACL_LINALG_NMF_HPP

/* =========================================================================
   Copyright (c) 2010-2012, Institute for Microelectronics,
                            Institute for Analysis and Scientific Computing,
                            TU Wien.

                            -----------------
                  ViennaCL - The Vienna Computing Library
                            -----------------

   Project Head:    Karl Rupp                   rupp@iue.tuwien.ac.at
               
   (A list of authors and contributors can be found in the PDF manual)

   License:         MIT (X11), see file LICENSE in the base directory
============================================================================= */

/** @file viennacl/linalg/nmf.hpp
    @brief Provides a nonnegative matrix factorization implementation.  Experimental in 1.3.x.
    
    Contributed by Volodymyr Kysenko.
*/


#include "viennacl/vector.hpp"
#include "viennacl/matrix.hpp"
#include "viennacl/linalg/prod.hpp"
#include "viennacl/linalg/norm_2.hpp"
#include "viennacl/linalg/kernels/nmf_kernels.h"

namespace viennacl
{
  namespace linalg
  {
    //const std::string NMF_PROGRAM_NAME = "elem_wise_ops";
    const std::string NMF_MUL_DIV_KERNEL = "el_wise_mul_div";
    const std::string NMF_SUB_KERNEL = "sub_wise";


    template <typename ScalarType>
    void nmf(viennacl::matrix<ScalarType> const & v,
             viennacl::matrix<ScalarType> & w,
             viennacl::matrix<ScalarType> & h,
             std::size_t k,
             ScalarType eps = 0.000001,
             std::size_t max_iter = 10000,
             std::size_t check_diff_every_step = 100)
    {
      viennacl::linalg::kernels::nmf<ScalarType, 1>::init();
      
      w.resize(v.size1(), k);
      h.resize(k, v.size2());

      std::vector<ScalarType> stl_w(w.internal_size1() * w.internal_size2());
      std::vector<ScalarType> stl_h(h.internal_size1() * h.internal_size2());

      for (std::size_t j = 0; j < stl_w.size(); j++)
          stl_w[j] = static_cast<ScalarType>(rand()) / RAND_MAX;

      for (std::size_t j = 0; j < stl_h.size(); j++)
          stl_h[j] = static_cast<ScalarType>(rand()) / RAND_MAX;

      viennacl::matrix<ScalarType> wn(v.size1(), k);
      viennacl::matrix<ScalarType> wd(v.size1(), k);
      viennacl::matrix<ScalarType> wtmp(v.size1(), v.size2());

      viennacl::matrix<ScalarType> hn(k, v.size2());
      viennacl::matrix<ScalarType> hd(k, v.size2());
      viennacl::matrix<ScalarType> htmp(k, k);

      viennacl::matrix<ScalarType> appr(v.size1(), v.size2());
      viennacl::vector<ScalarType> diff(v.size1() * v.size2());

      viennacl::fast_copy(&stl_w[0], &stl_w[0] + stl_w.size(), w);
      viennacl::fast_copy(&stl_h[0], &stl_h[0] + stl_h.size(), h);

      ScalarType last_diff = 0.0f;


      
      for (std::size_t i = 0; i < max_iter; i++)
      {
        {
          hn = viennacl::linalg::prod(trans(w), v);
          htmp = viennacl::linalg::prod(trans(w), w);
          hd = viennacl::linalg::prod(htmp, h);

          viennacl::ocl::kernel & mul_div_kernel = viennacl::ocl::get_kernel(viennacl::linalg::kernels::nmf<ScalarType, 1>::program_name(), 
                                                                             NMF_MUL_DIV_KERNEL);
          viennacl::ocl::enqueue(mul_div_kernel(h, hn, hd, cl_uint(stl_h.size())));
        }
        {
          wn = viennacl::linalg::prod(v, trans(h));
          wtmp = viennacl::linalg::prod(w, h);
          wd = viennacl::linalg::prod(wtmp, trans(h));

          viennacl::ocl::kernel & mul_div_kernel = viennacl::ocl::get_kernel(viennacl::linalg::kernels::nmf<ScalarType, 1>::program_name(), 
                                                                             NMF_MUL_DIV_KERNEL);
          
          viennacl::ocl::enqueue(mul_div_kernel(w, wn, wd, cl_uint(stl_w.size())));
        }

        if (i % check_diff_every_step == 0)
        {
          appr = viennacl::linalg::prod(w, h);

         viennacl::ocl::kernel & sub_kernel = viennacl::ocl::get_kernel(viennacl::linalg::kernels::nmf<ScalarType, 1>::program_name(), 
                                                                        NMF_SUB_KERNEL);
          //this is a cheat. i.e save difference of two matrix into vector to get norm_2
          viennacl::ocl::enqueue(sub_kernel(appr, v, diff, cl_uint(v.size1() * v.size2())));
          ScalarType diff_val = viennacl::linalg::norm_2(diff);

          if((diff_val < eps) || (fabs(diff_val - last_diff) < eps))
          {
              //std::cout << "Breaked at diff - " << diff_val << "\n";
              break;
          }

          last_diff = diff_val;

          //printf("Iteration #%lu - %.5f \n", i, diff_val);
        }
      }
      
      
    }
  }
}

#endif