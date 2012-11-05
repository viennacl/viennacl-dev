#ifndef VIENNACL_LINALG_OPENCL_VECTOR_OPERATIONS_HPP_
#define VIENNACL_LINALG_OPENCL_VECTOR_OPERATIONS_HPP_

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

/** @file viennacl/linalg/opencl/vector_operations.hpp
    @brief Implementations of vector operations using OpenCL
*/

#include "viennacl/forwards.h"
#include "viennacl/ocl/device.hpp"
#include "viennacl/ocl/handle.hpp"
#include "viennacl/ocl/kernel.hpp"
#include "viennacl/scalar.hpp"
#include "viennacl/tools/tools.hpp"
#include "viennacl/linalg/kernels/vector_kernels.h"
#include "viennacl/meta/predicate.hpp"
#include "viennacl/meta/enable_if.hpp"
#include "viennacl/traits/size.hpp"
#include "viennacl/traits/start.hpp"
#include "viennacl/traits/handle.hpp"
#include "viennacl/traits/stride.hpp"

namespace viennacl
{
  namespace linalg
  {
    namespace opencl
    {
      //
      // Introductory note: By convention, all dimensions are already checked in the dispatcher frontend. No need to double-check again in here!
      //
      
      
      template <typename V1,
                typename V2, typename ScalarType1>
      typename viennacl::enable_if< viennacl::is_any_dense_nonstructured_vector<V1>::value
                                    && viennacl::is_any_dense_nonstructured_vector<V2>::value
                                    && viennacl::is_any_scalar<ScalarType1>::value
                                  >::type
      av(V1 & vec1, 
         V2 const & vec2, ScalarType1 const & alpha, std::size_t len_alpha, bool reciprocal_alpha, bool flip_sign_alpha) 
      {
        typedef typename viennacl::result_of::cpu_value_type<V1>::type        value_type;
        const unsigned int ALIGNMENT = viennacl::result_of::alignment<V1>::value;
        viennacl::linalg::kernels::vector<value_type, ALIGNMENT>::init();
        
        cl_uint options_alpha =   ((len_alpha > 1) ? (len_alpha << 2) : 0)
                                + (reciprocal_alpha ? 2 : 0)
                                + (flip_sign_alpha ? 1 : 0);
                                
        viennacl::ocl::kernel & k = viennacl::ocl::get_kernel(viennacl::linalg::kernels::vector<value_type, ALIGNMENT>::program_name(),
                                                              (viennacl::is_cpu_scalar<ScalarType1>::value ? "av_cpu" : "av_gpu"));
        k.global_work_size(0, std::min<std::size_t>(128 * k.local_work_size(),
                                                    viennacl::tools::roundUpToNextMultiple<std::size_t>(viennacl::traits::size(vec1), k.local_work_size()) ) );
        
        viennacl::ocl::packed_cl_uint size_vec1;
        size_vec1.start  = cl_uint(viennacl::traits::start(vec1));
        size_vec1.stride = cl_uint(viennacl::traits::stride(vec1));
        size_vec1.size   = cl_uint(viennacl::traits::size(vec1));

        viennacl::ocl::packed_cl_uint size_vec2;
        size_vec2.start  = cl_uint(viennacl::traits::start(vec2));
        size_vec2.stride = cl_uint(viennacl::traits::stride(vec2));
        size_vec2.size   = cl_uint(viennacl::traits::size(vec2));
        
        
        viennacl::ocl::enqueue(k(viennacl::traits::opencl_handle(vec1),
                                 size_vec1,
                                
                                 viennacl::traits::opencl_handle(viennacl::tools::promote_if_host_scalar<value_type>(alpha)),
                                 options_alpha,
                                 viennacl::traits::opencl_handle(vec2),
                                 size_vec2 )
                              );
      }
      
      
      template <typename V1,
                typename V2, typename ScalarType1,
                typename V3, typename ScalarType2>
      typename viennacl::enable_if< viennacl::is_any_dense_nonstructured_vector<V1>::value
                                    && viennacl::is_any_dense_nonstructured_vector<V2>::value
                                    && viennacl::is_any_dense_nonstructured_vector<V3>::value
                                    && viennacl::is_any_scalar<ScalarType1>::value
                                    && viennacl::is_any_scalar<ScalarType2>::value
                                  >::type
      avbv(V1 & vec1, 
          V2 const & vec2, ScalarType1 const & alpha, std::size_t len_alpha, bool reciprocal_alpha, bool flip_sign_alpha,
          V3 const & vec3, ScalarType2 const & beta, std::size_t len_beta, bool reciprocal_beta, bool flip_sign_beta) 
      {
        typedef typename viennacl::result_of::cpu_value_type<V1>::type        value_type;
        const unsigned int ALIGNMENT = viennacl::result_of::alignment<V1>::value;
        viennacl::linalg::kernels::vector<value_type, ALIGNMENT>::init();
        
        std::string kernel_name;
        if (viennacl::is_cpu_scalar<ScalarType1>::value && viennacl::is_cpu_scalar<ScalarType2>::value)
          kernel_name = "avbv_cpu_cpu";
        else if (viennacl::is_cpu_scalar<ScalarType1>::value && !viennacl::is_cpu_scalar<ScalarType2>::value)
          kernel_name = "avbv_cpu_gpu";
        else if (!viennacl::is_cpu_scalar<ScalarType1>::value && viennacl::is_cpu_scalar<ScalarType2>::value)
          kernel_name = "avbv_gpu_cpu";
        else 
          kernel_name = "avbv_gpu_gpu";
          
        cl_uint options_alpha =   ((len_alpha > 1) ? (len_alpha << 2) : 0)
                                + (reciprocal_alpha ? 2 : 0)
                                + (flip_sign_alpha ? 1 : 0);
        cl_uint options_beta =    ((len_beta > 1) ? (len_beta << 2) : 0)
                                + (reciprocal_beta ? 2 : 0)
                                + (flip_sign_beta ? 1 : 0);

        viennacl::ocl::kernel & k = viennacl::ocl::get_kernel(viennacl::linalg::kernels::vector<value_type, ALIGNMENT>::program_name(), kernel_name);
        k.global_work_size(0, std::min<std::size_t>(128 * k.local_work_size(),
                                                    viennacl::tools::roundUpToNextMultiple<std::size_t>(viennacl::traits::size(vec1), k.local_work_size()) ) );
        
        viennacl::ocl::packed_cl_uint size_vec1;
        size_vec1.start  = cl_uint(viennacl::traits::start(vec1));
        size_vec1.stride = cl_uint(viennacl::traits::stride(vec1));
        size_vec1.size   = cl_uint(viennacl::traits::size(vec1));

        viennacl::ocl::packed_cl_uint size_vec2;
        size_vec2.start  = cl_uint(viennacl::traits::start(vec2));
        size_vec2.stride = cl_uint(viennacl::traits::stride(vec2));
        size_vec2.size   = cl_uint(viennacl::traits::size(vec2));

        viennacl::ocl::packed_cl_uint size_vec3;
        size_vec3.start  = cl_uint(viennacl::traits::start(vec3));
        size_vec3.stride = cl_uint(viennacl::traits::stride(vec3));
        size_vec3.size   = cl_uint(viennacl::traits::size(vec3));
        
        viennacl::ocl::enqueue(k(viennacl::traits::opencl_handle(vec1),
                                 size_vec1,
                                
                                 viennacl::traits::opencl_handle(viennacl::tools::promote_if_host_scalar<value_type>(alpha)),
                                 options_alpha,
                                 viennacl::traits::opencl_handle(vec2),
                                 size_vec2,
                                
                                 viennacl::traits::opencl_handle(viennacl::tools::promote_if_host_scalar<value_type>(beta)),
                                 options_beta,
                                 viennacl::traits::opencl_handle(vec3),
                                 size_vec3 )
                              );
      }
      
      
      template <typename V1,
                typename V2, typename ScalarType1,
                typename V3, typename ScalarType2>
      typename viennacl::enable_if< viennacl::is_any_dense_nonstructured_vector<V1>::value
                                    && viennacl::is_any_dense_nonstructured_vector<V2>::value
                                    && viennacl::is_any_dense_nonstructured_vector<V3>::value
                                    && viennacl::is_any_scalar<ScalarType1>::value
                                    && viennacl::is_any_scalar<ScalarType2>::value
                                  >::type
      avbv_v(V1 & vec1,
            V2 const & vec2, ScalarType1 const & alpha, std::size_t len_alpha, bool reciprocal_alpha, bool flip_sign_alpha,
            V3 const & vec3, ScalarType2 const & beta,  std::size_t len_beta,  bool reciprocal_beta,  bool flip_sign_beta) 
      {
        typedef typename viennacl::result_of::cpu_value_type<V1>::type        value_type;
        const unsigned int ALIGNMENT = viennacl::result_of::alignment<V1>::value;
        viennacl::linalg::kernels::vector<value_type, ALIGNMENT>::init();
        
        std::string kernel_name;
        if (viennacl::is_cpu_scalar<ScalarType1>::value && viennacl::is_cpu_scalar<ScalarType2>::value)
          kernel_name = "avbv_v_cpu_cpu";
        else if (viennacl::is_cpu_scalar<ScalarType1>::value && !viennacl::is_cpu_scalar<ScalarType2>::value)
          kernel_name = "avbv_v_cpu_gpu";
        else if (!viennacl::is_cpu_scalar<ScalarType1>::value && viennacl::is_cpu_scalar<ScalarType2>::value)
          kernel_name = "avbv_v_gpu_cpu";
        else 
          kernel_name = "avbv_v_gpu_gpu";
        
        cl_uint options_alpha =   ((len_alpha > 1) ? (len_alpha << 2) : 0)
                                + (reciprocal_alpha ? 2 : 0)
                                + (flip_sign_alpha ? 1 : 0);
        cl_uint options_beta =    ((len_beta > 1) ? (len_beta << 2) : 0)
                                + (reciprocal_beta ? 2 : 0)
                                + (flip_sign_beta ? 1 : 0);

        viennacl::ocl::kernel & k = viennacl::ocl::get_kernel(viennacl::linalg::kernels::vector<value_type, ALIGNMENT>::program_name(), kernel_name);
        k.global_work_size(0, std::min<std::size_t>(128 * k.local_work_size(),
                                                    viennacl::tools::roundUpToNextMultiple<std::size_t>(viennacl::traits::size(vec1), k.local_work_size()) ) );
        
        viennacl::ocl::packed_cl_uint size_vec1;
        size_vec1.start  = cl_uint(viennacl::traits::start(vec1));
        size_vec1.stride = cl_uint(viennacl::traits::stride(vec1));
        size_vec1.size   = cl_uint(viennacl::traits::size(vec1));

        viennacl::ocl::packed_cl_uint size_vec2;
        size_vec2.start  = cl_uint(viennacl::traits::start(vec2));
        size_vec2.stride = cl_uint(viennacl::traits::stride(vec2));
        size_vec2.size   = cl_uint(viennacl::traits::size(vec2));

        viennacl::ocl::packed_cl_uint size_vec3;
        size_vec3.start  = cl_uint(viennacl::traits::start(vec3));
        size_vec3.stride = cl_uint(viennacl::traits::stride(vec3));
        size_vec3.size   = cl_uint(viennacl::traits::size(vec3));
        
        viennacl::ocl::enqueue(k(viennacl::traits::opencl_handle(vec1),
                                 size_vec1,
                                
                                 viennacl::traits::opencl_handle(viennacl::tools::promote_if_host_scalar<value_type>(alpha)),
                                 options_alpha,
                                 viennacl::traits::opencl_handle(vec2),
                                 size_vec2,
                                
                                 viennacl::traits::opencl_handle(viennacl::tools::promote_if_host_scalar<value_type>(beta)),
                                 options_beta,
                                 viennacl::traits::opencl_handle(vec3),
                                 size_vec3 )
                              );
      }
      
      
      /** @brief Assign a constant value to a vector (-range/-slice)
      *
      * @param vec1   The vector to which the value should be assigned
      * @param alpha  The value to be assigned
      */
      template <typename V1, typename S1>
      typename viennacl::enable_if< viennacl::is_any_dense_nonstructured_vector<V1>::value
                                    && viennacl::is_cpu_scalar<S1>::value
                                  >::type
      vector_assign(V1 & vec1, const S1 & alpha)
      {
        typedef typename viennacl::result_of::cpu_value_type<V1>::type        value_type;
        const unsigned int ALIGNMENT = viennacl::result_of::alignment<V1>::value;
        viennacl::linalg::kernels::vector<value_type, ALIGNMENT>::init();
        
        viennacl::ocl::kernel & k = viennacl::ocl::get_kernel(viennacl::linalg::kernels::vector<value_type, ALIGNMENT>::program_name(), "assign_cpu");
        k.global_work_size(0, std::min<std::size_t>(128 * k.local_work_size(),
                                                    viennacl::tools::roundUpToNextMultiple<std::size_t>(viennacl::traits::size(vec1), k.local_work_size()) ) );
        viennacl::ocl::enqueue(k(viennacl::traits::opencl_handle(vec1),
                                 cl_uint(viennacl::traits::start(vec1)),
                                 cl_uint(viennacl::traits::stride(vec1)),
                                 cl_uint(viennacl::traits::size(vec1)),
                                 viennacl::traits::opencl_handle(value_type(alpha)) )
                              );
      }

      
      /** @brief Swaps the contents of two vectors, data is copied
      *
      * @param vec1   The first vector (or -range, or -slice)
      * @param vec2   The second vector (or -range, or -slice)
      */
      template <typename V1, typename V2>
      typename viennacl::enable_if<    viennacl::is_any_dense_nonstructured_vector<V1>::value
                                    && viennacl::is_any_dense_nonstructured_vector<V2>::value
                                  >::type
      vector_swap(V1 & vec1, V2 & vec2)
      {
        typedef typename viennacl::result_of::cpu_value_type<V1>::type        value_type;
        const unsigned int ALIGNMENT = viennacl::result_of::alignment<V1>::value;
        viennacl::linalg::kernels::vector<value_type, ALIGNMENT>::init();
        
        viennacl::ocl::kernel & k = viennacl::ocl::get_kernel(viennacl::linalg::kernels::vector<value_type, ALIGNMENT>::program_name(), "swap");

        viennacl::ocl::enqueue(k(viennacl::traits::opencl_handle(vec1),
                                 cl_uint(viennacl::traits::start(vec1)),
                                 cl_uint(viennacl::traits::stride(vec1)),
                                 cl_uint(viennacl::traits::size(vec1)),
                                 viennacl::traits::opencl_handle(vec2),
                                 cl_uint(viennacl::traits::start(vec2)),
                                 cl_uint(viennacl::traits::stride(vec2)),
                                 cl_uint(viennacl::traits::size(vec2)))
                              );
      }


      ///////////////////////// Norms and inner product ///////////////////


      //implementation of inner product:
      //namespace {
      /** @brief Computes the inner product of two vectors - implementation. Library users should call inner_prod(vec1, vec2).
      *
      * @param vec1 The first vector
      * @param vec2 The second vector
      * @param result The result scalar (on the gpu)
      */
      template <typename V1, typename V2, typename S3>
      typename viennacl::enable_if<    viennacl::is_any_dense_nonstructured_vector<V1>::value
                                    && viennacl::is_any_dense_nonstructured_vector<V2>::value
                                    && viennacl::is_scalar<S3>::value
                                  >::type
      inner_prod_impl(V1 const & vec1,
                      V2 const & vec2,
                      S3 & result)
      {
        typedef typename viennacl::result_of::cpu_value_type<V1>::type        value_type;
        const unsigned int ALIGNMENT = viennacl::result_of::alignment<V1>::value;
        viennacl::linalg::kernels::vector<value_type, ALIGNMENT>::init();
      
        assert( (viennacl::traits::size(vec1) == viennacl::traits::size(vec2))
              && bool("Incompatible vector sizes in inner_prod_impl()!"));
        
        viennacl::ocl::kernel & k = viennacl::ocl::get_kernel(viennacl::linalg::kernels::vector<value_type, ALIGNMENT>::program_name(), "inner_prod");
        //cl_uint size = static_cast<cl_uint>(std::min(vec1.internal_size(), vec2.internal_size()));
        unsigned int work_groups = k.global_work_size() / k.local_work_size();
        
        static viennacl::vector<value_type> temp(work_groups);
        
        //Note: Number of work groups MUST be a power of two!
        //std::cout << work_groups << ", " << k.local_work_size() << ", " << k.global_work_size() << std::endl;
        assert( work_groups * k.local_work_size() == k.global_work_size() );
        assert( (k.global_work_size() / k.local_work_size()) == 1 
                || (k.global_work_size() / k.local_work_size()) == 2 
                || (k.global_work_size() / k.local_work_size()) == 4
                || (k.global_work_size() / k.local_work_size()) == 8
                || (k.global_work_size() / k.local_work_size()) == 16
                || (k.global_work_size() / k.local_work_size()) == 32
                || (k.global_work_size() / k.local_work_size()) == 64
                || (k.global_work_size() / k.local_work_size()) == 128
                || (k.global_work_size() / k.local_work_size()) == 256
                || (k.global_work_size() / k.local_work_size()) == 512 );
                
        viennacl::ocl::enqueue(k(viennacl::traits::opencl_handle(vec1),
                                 cl_uint(viennacl::traits::start(vec1)),
                                 cl_uint(viennacl::traits::stride(vec1)),
                                 cl_uint(viennacl::traits::size(vec1)),
                                 viennacl::traits::opencl_handle(vec2),
                                 cl_uint(viennacl::traits::start(vec2)),
                                 cl_uint(viennacl::traits::stride(vec2)),
                                 cl_uint(viennacl::traits::size(vec2)),
                                 viennacl::ocl::local_mem(sizeof(value_type) * k.local_work_size()),
                                 viennacl::traits::opencl_handle(temp)
                                )
                              );        

        viennacl::ocl::kernel & ksum = viennacl::ocl::get_kernel(viennacl::linalg::kernels::vector<value_type, ALIGNMENT>::program_name(), "sum");
        
        ksum.local_work_size(0, work_groups);
        ksum.global_work_size(0, work_groups);
        viennacl::ocl::enqueue(ksum(viennacl::traits::opencl_handle(temp),
                                    cl_uint(viennacl::traits::start(temp)),
                                    cl_uint(viennacl::traits::stride(temp)),
                                    cl_uint(viennacl::traits::size(temp)),
                                    cl_uint(1),
                                    viennacl::ocl::local_mem(sizeof(value_type) * ksum.local_work_size()),
                                    viennacl::traits::opencl_handle(result) )
                              );
      }

      
      //implementation of inner product:
      //namespace {
      /** @brief Computes the inner product of two vectors - implementation. Library users should call inner_prod(vec1, vec2).
      *
      * @param vec1 The first vector
      * @param vec2 The second vector
      * @param result The result scalar (on the gpu)
      */
      template <typename V1, typename V2, typename S3>
      typename viennacl::enable_if<    viennacl::is_any_dense_nonstructured_vector<V1>::value
                                    && viennacl::is_any_dense_nonstructured_vector<V2>::value
                                    && viennacl::is_cpu_scalar<S3>::value
                                  >::type
      inner_prod_cpu(V1 const & vec1,
                     V2 const & vec2,
                     S3 & result)
      {
        typedef typename viennacl::result_of::cpu_value_type<V1>::type        value_type;
        const unsigned int ALIGNMENT = viennacl::result_of::alignment<V1>::value;
        viennacl::linalg::kernels::vector<value_type, ALIGNMENT>::init();
      
        viennacl::ocl::kernel & k = viennacl::ocl::get_kernel(viennacl::linalg::kernels::vector<value_type, ALIGNMENT>::program_name(), "inner_prod");
        static const unsigned int work_groups = k.global_work_size() / k.local_work_size();
        
        static viennacl::vector<value_type> temp(work_groups);
        
        //Note: Number of work groups MUST be a power of two!
        //std::cout << work_groups << ", " << k.local_work_size() << ", " << k.global_work_size() << std::endl;
        assert( work_groups * k.local_work_size() == k.global_work_size() );
        assert( (k.global_work_size() / k.local_work_size()) == 1 
                || (k.global_work_size() / k.local_work_size()) == 2 
                || (k.global_work_size() / k.local_work_size()) == 4
                || (k.global_work_size() / k.local_work_size()) == 8
                || (k.global_work_size() / k.local_work_size()) == 16
                || (k.global_work_size() / k.local_work_size()) == 32
                || (k.global_work_size() / k.local_work_size()) == 64
                || (k.global_work_size() / k.local_work_size()) == 128
                || (k.global_work_size() / k.local_work_size()) == 256
                || (k.global_work_size() / k.local_work_size()) == 512 );
                
        viennacl::ocl::enqueue(k(viennacl::traits::opencl_handle(vec1),
                                 cl_uint(viennacl::traits::start(vec1)),
                                 cl_uint(viennacl::traits::stride(vec1)),
                                 cl_uint(viennacl::traits::size(vec1)),
                                 viennacl::traits::opencl_handle(vec2),
                                 cl_uint(viennacl::traits::start(vec2)),
                                 cl_uint(viennacl::traits::stride(vec2)),
                                 cl_uint(viennacl::traits::size(vec2)),
                                 viennacl::ocl::local_mem(sizeof(value_type) * k.local_work_size()),
                                 viennacl::traits::opencl_handle(temp)
                                )
                              );        

        // Now copy partial results from GPU back to CPU and run reduction there:
        static std::vector<value_type> temp_cpu(work_groups);
        viennacl::fast_copy(temp.begin(), temp.end(), temp_cpu.begin());
        
        result = 0;
        for (typename std::vector<value_type>::const_iterator it = temp_cpu.begin(); it != temp_cpu.end(); ++it)
          result += *it;
      }
      
      
      /** @brief Computes the l^1-norm of a vector
      *
      * @param vec The vector
      * @param result The result scalar
      */
      template <typename V1, typename S2>
      typename viennacl::enable_if<    viennacl::is_any_dense_nonstructured_vector<V1>::value
                                    && viennacl::is_scalar<S2>::value
                                  >::type
      norm_1_impl(V1 const & vec,
                  S2 & result)
      {
        typedef typename viennacl::result_of::cpu_value_type<V1>::type        value_type;
        const unsigned int ALIGNMENT = viennacl::result_of::alignment<V1>::value;
        viennacl::linalg::kernels::vector<value_type, ALIGNMENT>::init();
        
        viennacl::ocl::kernel & k = viennacl::ocl::get_kernel(viennacl::linalg::kernels::vector<value_type, ALIGNMENT>::program_name(), "norm");
        //cl_uint size = static_cast<cl_uint>(vcl_vec.internal_size());
        
        unsigned int work_groups = k.global_work_size() / k.local_work_size();
        viennacl::vector<value_type> temp(work_groups);

        //Note: Number of work groups MUST be a power of two!
        //std::cout << work_groups << ", " << k.local_work_size() << ", " << k.global_work_size() << std::endl;
        assert( work_groups * k.local_work_size() == k.global_work_size() );
        assert( (k.global_work_size() / k.local_work_size()) == 1 
              || (k.global_work_size() / k.local_work_size()) == 2 
              || (k.global_work_size() / k.local_work_size()) == 4
              || (k.global_work_size() / k.local_work_size()) == 8
              || (k.global_work_size() / k.local_work_size()) == 16
              || (k.global_work_size() / k.local_work_size()) == 32
              || (k.global_work_size() / k.local_work_size()) == 64
              || (k.global_work_size() / k.local_work_size()) == 128
              || (k.global_work_size() / k.local_work_size()) == 256
              || (k.global_work_size() / k.local_work_size()) == 512 );
                
        viennacl::ocl::enqueue(k(viennacl::traits::opencl_handle(vec),
                                 cl_uint(viennacl::traits::start(vec)),
                                 cl_uint(viennacl::traits::stride(vec)),
                                 cl_uint(viennacl::traits::size(vec)),
                                 cl_uint(1),
                                 viennacl::ocl::local_mem(sizeof(value_type) * k.local_work_size()),
                                 viennacl::traits::opencl_handle(temp) )
                              );        
        
        viennacl::ocl::kernel & ksum = viennacl::ocl::get_kernel(viennacl::linalg::kernels::vector<value_type, ALIGNMENT>::program_name(), "sum");
        
        ksum.local_work_size(0, work_groups);
        ksum.global_work_size(0, work_groups);
        viennacl::ocl::enqueue(ksum(viennacl::traits::opencl_handle(temp),
                                    cl_uint(viennacl::traits::start(temp)),
                                    cl_uint(viennacl::traits::stride(temp)),
                                    cl_uint(viennacl::traits::size(temp)),
                                    cl_uint(1),
                                    viennacl::ocl::local_mem(sizeof(value_type) * ksum.local_work_size()),
                                    result)
                              );
      }

      /** @brief Computes the l^2-norm of a vector - implementation
      *
      * @param vec The vector
      * @param result The result scalar
      * @param dummy  Dummy parameter used for SFINAE
      */
      template <typename V1, typename S2>
      typename viennacl::enable_if<    viennacl::is_any_dense_nonstructured_vector<V1>::value
                                    && viennacl::is_scalar<S2>::value
                                  >::type
      norm_2_impl(V1 const & vec,
                  S2 & result)
      {
        typedef typename viennacl::result_of::cpu_value_type<V1>::type        value_type;
        const unsigned int ALIGNMENT = viennacl::result_of::alignment<V1>::value;
        viennacl::linalg::kernels::vector<value_type, ALIGNMENT>::init();
        
        viennacl::ocl::kernel & k = viennacl::ocl::get_kernel(viennacl::linalg::kernels::vector<value_type, ALIGNMENT>::program_name(), "norm");
        //cl_uint size = static_cast<cl_uint>(vcl_vec.internal_size());
        
        unsigned int work_groups = k.global_work_size() / k.local_work_size();
        viennacl::vector<value_type> temp(work_groups);
          
        //Note: Number of work groups MUST be a power of two!
        //std::cout << work_groups << ", " << k.local_work_size() << ", " << k.global_work_size() << std::endl;
        assert( work_groups * k.local_work_size() == k.global_work_size() );
        assert( (k.global_work_size() / k.local_work_size()) == 1 
              || (k.global_work_size() / k.local_work_size()) == 2 
              || (k.global_work_size() / k.local_work_size()) == 4
              || (k.global_work_size() / k.local_work_size()) == 8
              || (k.global_work_size() / k.local_work_size()) == 16
              || (k.global_work_size() / k.local_work_size()) == 32
              || (k.global_work_size() / k.local_work_size()) == 64
              || (k.global_work_size() / k.local_work_size()) == 128
              || (k.global_work_size() / k.local_work_size()) == 256
              || (k.global_work_size() / k.local_work_size()) == 512 );
                
          viennacl::ocl::enqueue(k(viennacl::traits::opencl_handle(vec),
                                   cl_uint(viennacl::traits::start(vec)),
                                   cl_uint(viennacl::traits::stride(vec)),
                                   cl_uint(viennacl::traits::size(vec)),                                 
                                   cl_uint(2),
                                   viennacl::ocl::local_mem(sizeof(value_type) * k.local_work_size()),
                                   temp)
                                );

          viennacl::ocl::kernel & ksum = viennacl::ocl::get_kernel(viennacl::linalg::kernels::vector<value_type, ALIGNMENT>::program_name(), "sum");
          
          ksum.local_work_size(0, work_groups);
          ksum.global_work_size(0, work_groups);
          viennacl::ocl::enqueue( ksum(viennacl::traits::opencl_handle(temp),
                                       cl_uint(viennacl::traits::start(temp)),
                                       cl_uint(viennacl::traits::stride(temp)),
                                       cl_uint(viennacl::traits::size(temp)),
                                       cl_uint(2),
                                       viennacl::ocl::local_mem(sizeof(value_type) * ksum.local_work_size()),
                                       result)
                                );
      }

      /** @brief Computes the supremum-norm of a vector
      *
      * @param vec The vector
      * @param result The result scalar
      */
      template <typename V1, typename S2>
      typename viennacl::enable_if<    viennacl::is_any_dense_nonstructured_vector<V1>::value
                                    && viennacl::is_scalar<S2>::value
                                  >::type
      norm_inf_impl(V1 const & vec,
                    S2 & result)
      {
        typedef typename viennacl::result_of::cpu_value_type<V1>::type        value_type;
        const unsigned int ALIGNMENT = viennacl::result_of::alignment<V1>::value;
        viennacl::linalg::kernels::vector<value_type, ALIGNMENT>::init();
        
        //cl_uint size = static_cast<cl_uint>(vcl_vec.internal_size());
        viennacl::ocl::kernel & k = viennacl::ocl::get_kernel(viennacl::linalg::kernels::vector<value_type, ALIGNMENT>::program_name(), "norm");
        
        unsigned int work_groups = k.global_work_size() / k.local_work_size();
        viennacl::vector<value_type> temp(work_groups);
          
        //Note: Number of work groups MUST be a power of two!
        //std::cout << work_groups << ", " << k.local_work_size() << ", " << k.global_work_size() << std::endl;
        assert( work_groups * k.local_work_size() == k.global_work_size() );
        assert( work_groups == 1 
              || work_groups == 2 
              || work_groups == 4
              || work_groups == 8
              || work_groups == 16
              || work_groups == 32
              || work_groups == 64
              || work_groups == 128
              || work_groups == 256
              || work_groups == 512 );
                
        viennacl::ocl::enqueue(k(viennacl::traits::opencl_handle(vec),
                                 cl_uint(viennacl::traits::start(vec)),
                                 cl_uint(viennacl::traits::stride(vec)),
                                 cl_uint(viennacl::traits::size(vec)),                                 
                                 cl_uint(0),
                                 viennacl::ocl::local_mem(sizeof(value_type) * k.local_work_size()),
                                 temp));
        //viennacl::ocl::get_queue().finish();
        
        //part 2: parallel reduction of reduced kernel:
        viennacl::ocl::kernel & ksum = viennacl::ocl::get_kernel(viennacl::linalg::kernels::vector<value_type, ALIGNMENT>::program_name(), "sum");
        ksum.local_work_size(0, work_groups);
        ksum.global_work_size(0, work_groups);
        
        viennacl::ocl::enqueue( ksum(viennacl::traits::opencl_handle(temp),
                                     cl_uint(viennacl::traits::start(temp)),
                                     cl_uint(viennacl::traits::stride(temp)),
                                     cl_uint(viennacl::traits::size(temp)),
                                     cl_uint(0),
                                     viennacl::ocl::local_mem(sizeof(value_type) * ksum.local_work_size()),
                                     result)
                              );
      }

      //This function should return a CPU scalar, otherwise statements like 
      // vcl_rhs[index_norm_inf(vcl_rhs)] 
      // are ambiguous
      /** @brief Computes the index of the first entry that is equal to the supremum-norm in modulus.
      *
      * @param vec The vector
      * @return The result. Note that the result must be a CPU scalar (unsigned int), since gpu scalars are floating point types.
      */
      template <typename V1>
      typename viennacl::enable_if< viennacl::is_any_dense_nonstructured_vector<V1>::value,
                                    cl_uint
                                  >::type
      index_norm_inf(V1 const & vec)
      {
        typedef typename viennacl::result_of::cpu_value_type<V1>::type        value_type;
        const unsigned int ALIGNMENT = viennacl::result_of::alignment<V1>::value;
        viennacl::linalg::kernels::vector<value_type, ALIGNMENT>::init();
        
        viennacl::ocl::handle<cl_mem> h = viennacl::ocl::current_context().create_memory(CL_MEM_READ_WRITE, sizeof(cl_uint));
        
        viennacl::ocl::kernel & k = viennacl::ocl::get_kernel(viennacl::linalg::kernels::vector<value_type, ALIGNMENT>::program_name(), "index_norm_inf");
        //cl_uint size = static_cast<cl_uint>(vcl_vec.internal_size());

        //TODO: Use multi-group kernel for large vector sizes
        
        k.global_work_size(0, k.local_work_size());
        viennacl::ocl::enqueue(k(viennacl::traits::opencl_handle(vec),
                                 cl_uint(viennacl::traits::start(vec)),
                                 cl_uint(viennacl::traits::stride(vec)),
                                 cl_uint(viennacl::traits::size(vec)),                                 
                                 viennacl::ocl::local_mem(sizeof(value_type) * k.local_work_size()),
                                 viennacl::ocl::local_mem(sizeof(cl_uint) * k.local_work_size()), h));
        
        //read value:
        cl_uint result;
        cl_int err = clEnqueueReadBuffer(viennacl::ocl::get_queue().handle().get(), h.get(), CL_TRUE, 0, sizeof(cl_uint), &result, 0, NULL, NULL);
        VIENNACL_ERR_CHECK(err);
        return result;
      }
      
      //TODO: Special case vec1 == vec2 allows improvement!!
      /** @brief Computes a plane rotation of two vectors.
      *
      * Computes (x,y) <- (alpha * x + beta * y, -beta * x + alpha * y)
      *
      * @param vec1   The first vector
      * @param vec2   The second vector
      * @param alpha  The first transformation coefficient
      * @param beta   The second transformation coefficient
      */
      template <typename V1, typename V2, typename SCALARTYPE>
      typename viennacl::enable_if< viennacl::is_any_dense_nonstructured_vector<V1>::value
                                    && viennacl::is_any_dense_nonstructured_vector<V2>::value
                                    && viennacl::is_cpu_scalar<SCALARTYPE>::value
                                  >::type
      plane_rotation(V1 & vec1,
                    V2 & vec2,
                    SCALARTYPE alpha,
                    SCALARTYPE beta)
      {
        typedef typename viennacl::result_of::cpu_value_type<V1>::type        value_type;
        const unsigned int ALIGNMENT = V1::alignment;
        viennacl::linalg::kernels::vector<value_type, ALIGNMENT>::init();
        
        assert(viennacl::traits::size(vec1) == viennacl::traits::size(vec2));
        viennacl::ocl::kernel & k = viennacl::ocl::get_kernel(viennacl::linalg::kernels::vector<SCALARTYPE, ALIGNMENT>::program_name(), "plane_rotation");

        viennacl::ocl::enqueue(k(viennacl::traits::opencl_handle(vec1),
                                 cl_uint(viennacl::traits::start(vec1)),
                                 cl_uint(viennacl::traits::stride(vec1)),                                 
                                 cl_uint(viennacl::traits::size(vec1)),                                 
                                 viennacl::traits::opencl_handle(vec2),
                                 cl_uint(viennacl::traits::start(vec2)),
                                 cl_uint(viennacl::traits::stride(vec2)),                                 
                                 cl_uint(viennacl::traits::size(vec2)),                                 
                                 alpha,
                                 beta)
                              );
      }

    } //namespace opencl
  } //namespace linalg
} //namespace viennacl


#endif
