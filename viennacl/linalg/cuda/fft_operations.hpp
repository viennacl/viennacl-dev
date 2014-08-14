#ifndef VIENNACL_LINALG_CUDA_FFT_OPERATIONS_HPP_
#define VIENNACL_LINALG_CUDA_FFT_OPERATIONS_HPP_

/* =========================================================================
  Copyright (c) 2010-2014, Institute for Microelectronics,
                            Institute for Analysis and Scientific Computing,
                            TU Wien.
   Portions of this software are copyright by UChicago Argonne, LLC.

                            -----------------
                  ViennaCL - The Vienna Computing Library
                            -----------------

   Project Head:    Karl Rupp                   rupp@iue.tuwien.ac.at

   (A list of authors and contributors can be found in the PDF manual)

   License:         MIT (X11), see file LICENSE in the base directory
============================================================================= */

/** @file viennacl/linalg/cuda/fft_operations.hpp
    @brief Implementations of Fast Furier Transformation using cuda
*/
#include <cmath>
#include <viennacl/matrix.hpp>
#include <viennacl/vector.hpp>

#include "viennacl/forwards.h"
#include "viennacl/scalar.hpp"
#include "viennacl/tools/tools.hpp"
#include "viennacl/linalg/cuda/common.hpp"
#include "viennacl/linalg/host_based/vector_operations.hpp"

namespace viennacl
{
  namespace linalg
  {
    namespace cuda
    {
      namespace detail
      {
        namespace fft
        {
          const vcl_size_t MAX_LOCAL_POINTS_NUM = 512;

          inline vcl_size_t num_bits(vcl_size_t size)
          {
            vcl_size_t bits_datasize = 0;
            vcl_size_t ds = 1;

            while (ds < size)
            {
              ds = ds << 1;
              bits_datasize++;
            }

            return bits_datasize;
          }

          inline vcl_size_t next_power_2(vcl_size_t n)
          {
            n = n - 1;

            vcl_size_t power = 1;

            while (power < sizeof(vcl_size_t) * 8)
            {
              n = n | (n >> power);
              power *= 2;
            }

            return n + 1;
          }

        } //namespace fft
      } //namespace detail

// addition
      inline __host__ __device__ float2 operator+(float2 a, float2 b)
      {
        return make_float2(a.x + b.x, a.y + b.y);
      }

// subtract
      inline __host__ __device__ float2 operator-(float2 a, float2 b)
      {
        return make_float2(a.x - b.x, a.y - b.y);
      }
// division
      template<typename SCALARTYPE>
      inline __host__ __device__ float2 operator/(float2 a,SCALARTYPE b)
      {
        return make_float2(a.x/b, a.y/b);
      }

//multiplikation
      inline __host__ __device__ float2 operator*(float2 in1, float2 in2)
      {
        return make_float2(in1.x * in2.x - in1.y * in2.y, in1.x * in2.y + in1.y * in2.x);
      }

// addition
      inline __host__ __device__ double2 operator+(double2 a, double2 b)
      {
        return make_double2(a.x + b.x, a.y + b.y);
      }

// subtract
      inline __host__ __device__ double2 operator-(double2 a, double2 b)
      {
        return make_double2(a.x - b.x, a.y - b.y);
      }

// division
      template<typename SCALARTYPE>
      inline __host__ __device__ double2 operator/(double2 a,SCALARTYPE b)
      {
        return make_double2(a.x/b, a.y/b);
      }

//multiplikation
      inline __host__ __device__ double2 operator*(double2 in1, double2 in2)
      {
        return make_double2(in1.x * in2.x - in1.y * in2.y, in1.x * in2.y + in1.y * in2.x);
      }

      inline __host__ __device__ unsigned int get_reorder_num(unsigned int v,
          unsigned int bit_size)
      {
        v = ((v >> 1) & 0x55555555) | ((v & 0x55555555) << 1);
        v = ((v >> 2) & 0x33333333) | ((v & 0x33333333) << 2);
        v = ((v >> 4) & 0x0F0F0F0F) | ((v & 0x0F0F0F0F) << 4);
        v = ((v >> 8) & 0x00FF00FF) | ((v & 0x00FF00FF) << 8);
        v = (v >> 16) | (v << 16);
        v = v >> (32 - bit_size);
        return v;
      }

      template<typename T,typename SCALARTYPE>
      __global__ void fft_direct(
          const T * input,
          T * output,
          unsigned int size,
          unsigned int stride,
          unsigned int batch_num,
          SCALARTYPE sign,
          bool is_row_major)
      {

        const SCALARTYPE NUM_PI = 3.14159265358979323846;

        for(unsigned int batch_id = 0; batch_id < batch_num; batch_id++)
        {
          for(unsigned int k = blockIdx.x * blockDim.x + threadIdx.x; k < size; k += gridDim.x * blockDim.x)
          {
            T f;
            f.x = 0;
            f.y = 0;

            for(unsigned int n = 0; n < size; n++)
            {
              T in;
              if (!is_row_major)
              in = input[batch_id * stride + n];   //input index here
              else
              in = input[n * stride + batch_id];//input index here

              SCALARTYPE sn,cs;
              SCALARTYPE arg = sign * 2 * NUM_PI * k / size * n;
              sn = sin(arg);
              cs = cos(arg);

              T ex;
              ex.x = cs;
              ex.y = sn;
              T tmp;
              tmp.x = in.x * ex.x - in.y * ex.y;
              tmp.y = in.x * ex.y + in.y * ex.x;
              f = f + tmp;
            }

            if (!is_row_major)
            output[batch_id * stride + k] = f; // output index here
            else
            output[k * stride + batch_id] = f;// output index here
          }
        }
      }

      /**
       * @brief Direct 1D algorithm for computing Fourier transformation.
       *
       * Works on any sizes of data.
       * Serial implementation has o(n^2) complexity
       */
      template<class SCALARTYPE, unsigned int ALIGNMENT>
      void direct(const viennacl::vector<SCALARTYPE, ALIGNMENT>& in,
          viennacl::vector<SCALARTYPE, ALIGNMENT>& out, vcl_size_t size, vcl_size_t stride,
          vcl_size_t batch_num, SCALARTYPE sign = -1.0f,
          viennacl::linalg::host_based::detail::fft::FFT_DATA_ORDER::DATA_ORDER data_order =
              viennacl::linalg::host_based::detail::fft::FFT_DATA_ORDER::ROW_MAJOR)
      {
        fft_direct<<<128,128>>>(
			  reinterpret_cast<const typename viennacl::linalg::cuda::detail::type_to_type2<SCALARTYPE>::type *>(detail::cuda_arg<SCALARTYPE>(in)),
			  reinterpret_cast<typename viennacl::linalg::cuda::detail::type_to_type2<SCALARTYPE>::type *>(detail::cuda_arg<SCALARTYPE>(out)),
			  static_cast<unsigned int>(size),
			  static_cast<unsigned int>(stride),
			  static_cast<unsigned int>(batch_num),
			  sign,
			  static_cast<bool>(data_order));
	VIENNACL_CUDA_LAST_ERROR_CHECK("fft_direct");

      }

      /**
       * @brief Direct 2D algorithm for computing Fourier transformation.
       *
       * Works on any sizes of data.
       * Serial implementation has o(n^2) complexity
       */
      template<class SCALARTYPE, unsigned int ALIGNMENT>
      void direct(const viennacl::matrix<SCALARTYPE, viennacl::row_major, ALIGNMENT>& in,
          viennacl::matrix<SCALARTYPE, viennacl::row_major, ALIGNMENT>& out, vcl_size_t size,
          vcl_size_t stride, vcl_size_t batch_num, SCALARTYPE sign = -1.0f,
          viennacl::linalg::host_based::detail::fft::FFT_DATA_ORDER::DATA_ORDER data_order =
              viennacl::linalg::host_based::detail::fft::FFT_DATA_ORDER::ROW_MAJOR)
      {

        fft_direct<<<128,128>>>(
			reinterpret_cast<const typename viennacl::linalg::cuda::detail::type_to_type2<SCALARTYPE>::type *>(detail::cuda_arg<SCALARTYPE>(in)),
			reinterpret_cast<typename viennacl::linalg::cuda::detail::type_to_type2<SCALARTYPE>::type *>(detail::cuda_arg<SCALARTYPE>(out)),
			static_cast<unsigned int>(size),
			static_cast<unsigned int>(stride),
			static_cast<unsigned int>(batch_num),
			sign,
			static_cast<bool>(data_order));
	VIENNACL_CUDA_LAST_ERROR_CHECK("fft_direct");

      }

      template<typename T>
      __global__ void fft_reorder(
          T * input,
          unsigned int bit_size,
          unsigned int size,
          unsigned int stride,
          unsigned int batch_num,
          bool is_row_major)
      {

        unsigned int glb_id = blockIdx.x * blockDim.x + threadIdx.x;
        unsigned int glb_sz = gridDim.x * blockDim.x;

        for(unsigned int batch_id = 0; batch_id < batch_num; batch_id++)
        {
          for(unsigned int i = glb_id; i < size; i += glb_sz)
          {
            unsigned int v = get_reorder_num(i, bit_size);

            if(i < v)
            {
              if (!is_row_major)
              {
                T tmp = input[batch_id * stride + i];    // index
                input[batch_id * stride + i] = input[batch_id * stride + v];//index
                input[batch_id * stride + v] = tmp;//index
              }
              else
              {
                T tmp = input[i * stride + batch_id];
                input[i * stride + batch_id] = input[v * stride + batch_id];
                input[v * stride + batch_id] = tmp;
              }
            }
          }
        }
      }

      /*
       * This function performs reorder of input data. Indexes are sorted in bit-reversal order.
       * Such reordering should be done before in-place FFT.
       */
      template<class SCALARTYPE, unsigned int ALIGNMENT>
      void reorder(viennacl::vector<SCALARTYPE, ALIGNMENT>& in, vcl_size_t size, vcl_size_t stride,
          vcl_size_t bits_datasize, vcl_size_t batch_num,
          viennacl::linalg::host_based::detail::fft::FFT_DATA_ORDER::DATA_ORDER data_order =
              viennacl::linalg::host_based::detail::fft::FFT_DATA_ORDER::ROW_MAJOR)
      {

        fft_reorder<<<128,128>>>(
			reinterpret_cast<typename viennacl::linalg::cuda::detail::type_to_type2<SCALARTYPE>::type *>(detail::cuda_arg<SCALARTYPE>(in)),
			static_cast<unsigned int>(bits_datasize),
			static_cast<unsigned int>(size),
			static_cast<unsigned int>(stride),
			static_cast<unsigned int>(batch_num),
			static_cast<bool>(data_order));
	VIENNACL_CUDA_LAST_ERROR_CHECK("fft_reorder");
      }

      template<typename T,typename SCALARTYPE>
      __global__ void fft_radix2_local(
          T * input,
          unsigned int bit_size,
          unsigned int size,
          unsigned int stride,
          unsigned int batch_num,
          SCALARTYPE sign,
          bool is_row_major)
      {
        __shared__ T lcl_input[1024];
        unsigned int grp_id = blockIdx.x;
        unsigned int grp_num = gridDim.x;

        unsigned int lcl_sz = blockDim.x;
        unsigned int lcl_id = threadIdx.x;
        const SCALARTYPE NUM_PI = 3.14159265358979323846;

        for(unsigned int batch_id = grp_id; batch_id < batch_num; batch_id += grp_num)
        {
          for(unsigned int p = lcl_id; p < size; p += lcl_sz)
          {
            unsigned int v = get_reorder_num(p, bit_size);
            if(!is_row_major)
            lcl_input[v] = input[batch_id * stride + p];
            else
            lcl_input[v] = input[p * stride + batch_id];
          }

          __syncthreads();

          //performs Cooley-Tukey FFT on local arrayfft
          for(unsigned int s = 0; s < bit_size; s++)
          {
            unsigned int ss = 1 << s;
            SCALARTYPE cs, sn;
            for(unsigned int tid = lcl_id; tid < size; tid += lcl_sz)
            {
              unsigned int group = (tid & (ss - 1));
              unsigned int pos = ((tid >> s) << (s + 1)) + group;

              T in1 = lcl_input[pos];
              T in2 = lcl_input[pos + ss];

              SCALARTYPE arg = group * sign * NUM_PI / ss;

              sn = sin(arg);
              cs = cos(arg);
              T ex;
              ex.x = cs;
              ex.y = sn;

              T tmp;
              tmp.x = in2.x * ex.x - in2.y * ex.y;
              tmp.y = in2.x * ex.y + in2.y * ex.x;

              lcl_input[pos + ss] = in1 - tmp;
              lcl_input[pos] = in1 + tmp;
            }
            __syncthreads();
          }

          //copy local array back to global memory
          for(unsigned int p = lcl_id; p < size; p += lcl_sz)
          {
            if(!is_row_major)
            input[batch_id * stride + p] = lcl_input[p];   //index
            else
            input[p * stride + batch_id] = lcl_input[p];
          }

        }
      }

      template<typename T,typename SCALARTYPE>
      __global__ void fft_radix2(
          T * input,
          unsigned int s,
          unsigned int bit_size,
          unsigned int size,
          unsigned int stride,
          unsigned int batch_num,
          SCALARTYPE sign,
          bool is_row_major)
      {

        unsigned int ss = 1 << s;
        unsigned int half_size = size >> 1;

        SCALARTYPE cs, sn;
        const SCALARTYPE NUM_PI = 3.14159265358979323846;

        unsigned int glb_id = blockIdx.x * blockDim.x + threadIdx.x;
        unsigned int glb_sz = gridDim.x * blockDim.x;

        for(unsigned int batch_id = 0; batch_id < batch_num; batch_id++)
        {
          for(unsigned int tid = glb_id; tid < half_size; tid += glb_sz)
          {
            unsigned int group = (tid & (ss - 1));
            unsigned int pos = ((tid >> s) << (s + 1)) + group;
            T in1;
            T in2;
            unsigned int offset;
            if (!is_row_major)
            {
              offset = batch_id * stride + pos;
              in1 = input[offset];   //index
              in2 = input[offset + ss];//index
            }
            else
            {
              offset = pos * stride + batch_id;
              in1 = input[offset];   //index
              in2 = input[offset + ss * stride];//index
            }

            SCALARTYPE arg = group * sign * NUM_PI / ss;

            sn = sin(arg);
            cs = cos(arg);

            T ex;
            ex.x = cs;
            ex.y = sn;

            T tmp;
            tmp.x = in2.x * ex.x - in2.y * ex.y;
            tmp.y = in2.x * ex.y + in2.y * ex.x;

            if (!is_row_major)
            {
              input[offset + ss] = in1 - tmp;  //index

            }
            else
            {
              input[offset + ss * stride] = in1 - tmp;  //index
            }
            input[offset] = in1 + tmp;  //index
          }
        }
      }

      /**
       * @brief Radix-2 1D algorithm for computing Fourier transformation.
       *
       * Works only on power-of-two sizes of data.
       * Serial implementation has o(n * lg n) complexity.
       * This is a Cooley-Tukey algorithm
       */
      template<class SCALARTYPE, unsigned int ALIGNMENT>
      void radix2(viennacl::vector<SCALARTYPE, ALIGNMENT>& in, vcl_size_t size, vcl_size_t stride,
          vcl_size_t batch_num, SCALARTYPE sign = -1.0f,
          viennacl::linalg::host_based::detail::fft::FFT_DATA_ORDER::DATA_ORDER data_order =
              viennacl::linalg::host_based::detail::fft::FFT_DATA_ORDER::ROW_MAJOR)
      {

        unsigned int bit_size = viennacl::linalg::cuda::detail::fft::num_bits(size);

        if (size <= viennacl::linalg::cuda::detail::fft::MAX_LOCAL_POINTS_NUM)
        {
          fft_radix2_local<<<128,128>>>(
			reinterpret_cast<typename viennacl::linalg::cuda::detail::type_to_type2<SCALARTYPE>::type *>(detail::cuda_arg<SCALARTYPE>(in)),
			static_cast<unsigned int>(bit_size),
			static_cast<unsigned int>(size),
			static_cast<unsigned int>(stride),
			static_cast<unsigned int>(batch_num),
			static_cast<SCALARTYPE>(sign),
			static_cast<bool>(data_order));
	 VIENNACL_CUDA_LAST_ERROR_CHECK("fft_radix2_local");

        } else
        {
          fft_reorder<<<128,128>>>(
			reinterpret_cast<typename viennacl::linalg::cuda::detail::type_to_type2<SCALARTYPE>::type *>(detail::cuda_arg<SCALARTYPE>(in)),
			static_cast<unsigned int>(bit_size),
			static_cast<unsigned int>(size),
			static_cast<unsigned int>(stride),
			static_cast<unsigned int>(batch_num),
			static_cast<bool>(data_order));
	  VIENNACL_CUDA_LAST_ERROR_CHECK("fft_reorder");

          for (vcl_size_t step = 0; step < bit_size; step++)
          {
            fft_radix2<<<128,128>>>(
			reinterpret_cast<typename viennacl::linalg::cuda::detail::type_to_type2<SCALARTYPE>::type *>(detail::cuda_arg<SCALARTYPE>(in)),
			static_cast<unsigned int>(step),
			static_cast<unsigned int>(bit_size),
			static_cast<unsigned int>(size),
			static_cast<unsigned int>(stride),
			static_cast<unsigned int>(batch_num),
			sign,
			static_cast<bool>(data_order));
	    VIENNACL_CUDA_LAST_ERROR_CHECK("fft_radix2");
          }
        }
      }

      /**
       * @brief Radix-2 2D algorithm for computing Fourier transformation.
       *
       * Works only on power-of-two sizes of data.
       * Serial implementation has o(n * lg n) complexity.
       * This is a Cooley-Tukey algorithm
       */
      template<class SCALARTYPE, unsigned int ALIGNMENT>
      void radix2(viennacl::matrix<SCALARTYPE, viennacl::row_major, ALIGNMENT>& in, vcl_size_t size,
          vcl_size_t stride, vcl_size_t batch_num, SCALARTYPE sign = -1.0f,
          viennacl::linalg::host_based::detail::fft::FFT_DATA_ORDER::DATA_ORDER data_order =
              viennacl::linalg::host_based::detail::fft::FFT_DATA_ORDER::ROW_MAJOR)
      {

        unsigned int bit_size = viennacl::linalg::cuda::detail::fft::num_bits(size);

        if (size <= viennacl::linalg::cuda::detail::fft::MAX_LOCAL_POINTS_NUM)
        {
          fft_radix2_local<<<128,128>>>(
			reinterpret_cast<typename viennacl::linalg::cuda::detail::type_to_type2<SCALARTYPE>::type *>(detail::cuda_arg<SCALARTYPE>(in)),
			static_cast<unsigned int>(bit_size),
			static_cast<unsigned int>(size),
			static_cast<unsigned int>(stride),
			static_cast<unsigned int>(batch_num),
			sign,
			static_cast<bool>(data_order));
	  VIENNACL_CUDA_LAST_ERROR_CHECK("fft_radix2_local");
        } else
        {
          fft_reorder<<<128,128>>>(
			reinterpret_cast<typename viennacl::linalg::cuda::detail::type_to_type2<SCALARTYPE>::type *>(detail::cuda_arg<SCALARTYPE>(in)),
			static_cast<unsigned int>(bit_size),
			static_cast<unsigned int>(size),
			static_cast<unsigned int>(stride),
			static_cast<unsigned int>(batch_num),
			static_cast<bool>(data_order));
	  VIENNACL_CUDA_LAST_ERROR_CHECK("fft_reorder");
          for (vcl_size_t step = 0; step < bit_size; step++)
          {
            fft_radix2<<<128,128>>>(
			reinterpret_cast<typename viennacl::linalg::cuda::detail::type_to_type2<SCALARTYPE>::type *>(detail::cuda_arg<SCALARTYPE>(in)),
			static_cast<unsigned int>(step),
			static_cast<unsigned int>(bit_size),
			static_cast<unsigned int>(size),
			static_cast<unsigned int>(stride),
			static_cast<unsigned int>(batch_num),
			sign,
			static_cast<bool>(data_order));
	   VIENNACL_CUDA_LAST_ERROR_CHECK("fft_radix2");
          }
        }
      }

      template<typename T,typename SCALARTYPE>
      __global__ void bluestein_post(
          T * Z,
          T * out,
          unsigned int size,
          SCALARTYPE sign)
      {
        unsigned int glb_id = blockIdx.x * blockDim.x + threadIdx.x;
        unsigned int glb_sz =gridDim.x * blockDim.x;

        unsigned int double_size = size << 1;
        SCALARTYPE sn_a, cs_a;
        const SCALARTYPE NUM_PI = 3.14159265358979323846;

        for(unsigned int i = glb_id; i < size; i += glb_sz)
        {
          unsigned int rm = i * i % (double_size);
          SCALARTYPE angle = (SCALARTYPE )rm / size * (-NUM_PI);

          sn_a = sin(angle);
          cs_a= cos(angle);

          T b_i;
          b_i.x = cs_a;
          b_i.y = sn_a;
          out[i].x =Z[i].x * b_i.x - Z[i].y * b_i.y;
          out[i].y = Z[i].x * b_i.y + Z[i].y * b_i.x;
        }
      }

      template<typename T,typename SCALARTYPE>
      __global__ void bluestein_pre(
          T * input,
          T * A,
          T * B,
          unsigned int size,
          unsigned int ext_size,
          SCALARTYPE sign)
      {
        unsigned int glb_id = blockIdx.x * blockDim.x + threadIdx.x;
        unsigned int glb_sz =gridDim.x * blockDim.x;

        unsigned int double_size = size << 1;

        SCALARTYPE sn_a, cs_a;
        const SCALARTYPE NUM_PI = 3.14159265358979323846;

        for(unsigned int i = glb_id; i < size; i += glb_sz)
        {
          unsigned int rm = i * i % (double_size);
          SCALARTYPE angle = (SCALARTYPE)rm / size * NUM_PI;

          sn_a = sin(-angle);
          cs_a= cos(-angle);

          T a_i;
          a_i.x =cs_a;
          a_i.y =sn_a;

          T b_i;
          b_i.x =cs_a;
          b_i.y =-sn_a;

          A[i].x = input[i].x * a_i.x - input[i].y * a_i.y;
          A[i].y = input[i].x * a_i.y + input[i].y * a_i.x;
          B[i] = b_i;

          // very bad instruction, to be fixed
          if(i)
          B[ext_size - i] = b_i;
        }
      }

      template<typename T>
      __global__ void zero2(
          T * input1,
          T * input2,
          unsigned int size)
      {
        for (unsigned int i = blockIdx.x * blockDim.x + threadIdx.x; i < size; i += gridDim.x * blockDim.x)
        {
          input1[i].x = 0;
          input1[i].y = 0;

          input2[i].x = 0;
          input2[i].y = 0;

        }
      }

      /**
       * @brief Bluestein's algorithm for computing Fourier transformation.
       *
       * Currently,  Works only for sizes of input data which less than 2^16.
       * Uses a lot of additional memory, but should be fast for any size of data.
       * Serial implementation has something about o(n * lg n) complexity
       */
      template<class SCALARTYPE, unsigned int ALIGNMENT>
      void bluestein(viennacl::vector<SCALARTYPE, ALIGNMENT>& in,
          viennacl::vector<SCALARTYPE, ALIGNMENT>& out, vcl_size_t /*batch_num*/)
      {

        vcl_size_t size = in.size() >> 1;
        vcl_size_t ext_size = viennacl::linalg::cuda::detail::fft::next_power_2(2 * size - 1);

        viennacl::vector<SCALARTYPE, ALIGNMENT> A(ext_size << 1);
        viennacl::vector<SCALARTYPE, ALIGNMENT> B(ext_size << 1);

        viennacl::vector<SCALARTYPE, ALIGNMENT> Z(ext_size << 1);

        zero2<<<128,128>>>(
			reinterpret_cast<typename viennacl::linalg::cuda::detail::type_to_type2<SCALARTYPE>::type *>(detail::cuda_arg<SCALARTYPE>(A)),
			reinterpret_cast<typename viennacl::linalg::cuda::detail::type_to_type2<SCALARTYPE>::type *>(detail::cuda_arg<SCALARTYPE>(B)),
			static_cast<unsigned int>(ext_size));
	VIENNACL_CUDA_LAST_ERROR_CHECK("zero2");
	
	bluestein_pre<<<128,128>>>(
			reinterpret_cast<typename viennacl::linalg::cuda::detail::type_to_type2<SCALARTYPE>::type *>(detail::cuda_arg<SCALARTYPE>(in)),
			reinterpret_cast<typename viennacl::linalg::cuda::detail::type_to_type2<SCALARTYPE>::type *>(detail::cuda_arg<SCALARTYPE>(A)),
			reinterpret_cast<typename viennacl::linalg::cuda::detail::type_to_type2<SCALARTYPE>::type *>(detail::cuda_arg<SCALARTYPE>(B)),
			static_cast<unsigned int>(size),
			static_cast<unsigned int>(ext_size),
			SCALARTYPE(1));
	VIENNACL_CUDA_LAST_ERROR_CHECK("bluestein_pre");

        viennacl::linalg::convolve_i(A, B, Z);

        bluestein_post<<<128,128>>>(
			reinterpret_cast<typename viennacl::linalg::cuda::detail::type_to_type2<SCALARTYPE>::type *>(detail::cuda_arg<SCALARTYPE>(Z)),
			reinterpret_cast<typename viennacl::linalg::cuda::detail::type_to_type2<SCALARTYPE>::type *>(detail::cuda_arg<SCALARTYPE>(out)),
			static_cast<unsigned int>(size),
			SCALARTYPE(1));
	VIENNACL_CUDA_LAST_ERROR_CHECK("bluestein_post");

      }

      template<typename T>
      __global__ void fft_mult_vec(
          const T * input1,
          const T * input2,
          T * output,
          unsigned int size)
      {
        for (unsigned int i = blockIdx.x * blockDim.x + threadIdx.x; i < size; i += gridDim.x * blockDim.x)
        {
          T in1 = input1[i];
          T in2 = input2[i];
          output[i] = in1 * in2;
        }
      }

      /**
       * @brief Mutiply two complex vectors and store result in output
       */
      template<class SCALARTYPE, unsigned int ALIGNMENT>
      void multiply_complex(viennacl::vector<SCALARTYPE, ALIGNMENT> const & input1,
          viennacl::vector<SCALARTYPE, ALIGNMENT> const & input2,
          viennacl::vector<SCALARTYPE, ALIGNMENT> & output)
      {
        vcl_size_t size = input1.size() >> 1;
        fft_mult_vec<<<128,128>>>(
			reinterpret_cast<const typename viennacl::linalg::cuda::detail::type_to_type2<SCALARTYPE>::type *>(detail::cuda_arg<SCALARTYPE>(input1)),
			reinterpret_cast<const typename viennacl::linalg::cuda::detail::type_to_type2<SCALARTYPE>::type *>(detail::cuda_arg<SCALARTYPE>(input2)),
			reinterpret_cast<typename viennacl::linalg::cuda::detail::type_to_type2<SCALARTYPE>::type *>(detail::cuda_arg<SCALARTYPE>(output)),
			static_cast<unsigned int>(size));
	VIENNACL_CUDA_LAST_ERROR_CHECK("fft_mult_vec");

      }

      template<typename T,typename SCALARTYPE>
      __global__ void fft_div_vec_scalar(
          T * input1,
          unsigned int size,
          SCALARTYPE factor)
      {
        for (unsigned int i = blockIdx.x * blockDim.x + threadIdx.x; i < size; i += gridDim.x*blockDim.x)
        input1[i] = input1[i]/factor;
      }

      /**
       * @brief Normalize vector on with his own size
       */
      template<class SCALARTYPE, unsigned int ALIGNMENT>
      void normalize(viennacl::vector<SCALARTYPE, ALIGNMENT> & input)
      {

        vcl_size_t size = input.size() >> 1;
        SCALARTYPE norm_factor = static_cast<SCALARTYPE>(size);
        fft_div_vec_scalar<<<128,128>>>(
			reinterpret_cast<typename viennacl::linalg::cuda::detail::type_to_type2<SCALARTYPE>::type *>(detail::cuda_arg<SCALARTYPE>(input)),
			static_cast<unsigned int>(size),
			norm_factor);
	VIENNACL_CUDA_LAST_ERROR_CHECK("fft_div_vec_scalar");

      }

      template<typename T>
      __global__ void transpose(
          const T * input,
          T * output,
          unsigned int row_num,
          unsigned int col_num)
      {
        unsigned int size = row_num * col_num;
        for(unsigned int i =blockIdx.x * blockDim.x + threadIdx.x; i < size; i+= gridDim.x * blockDim.x)
        {
          unsigned int row = i / col_num;
          unsigned int col = i - row*col_num;
          unsigned int new_pos = col * row_num + row;
          output[new_pos] = input[i];
        }
      }

      /**
       * @brief Transpose matrix
       */
      template<class SCALARTYPE, unsigned int ALIGNMENT>
      void transpose(viennacl::matrix<SCALARTYPE, viennacl::row_major, ALIGNMENT> const & input,
          viennacl::matrix<SCALARTYPE, viennacl::row_major, ALIGNMENT> & output)
      {

        transpose<<<128,128>>>(
			reinterpret_cast<const typename viennacl::linalg::cuda::detail::type_to_type2<SCALARTYPE>::type *>(detail::cuda_arg<SCALARTYPE>(input)),
			reinterpret_cast<typename viennacl::linalg::cuda::detail::type_to_type2<SCALARTYPE>::type *>(detail::cuda_arg<SCALARTYPE>(output)),
			static_cast<unsigned int>(input.internal_size1()>>1),
			static_cast<unsigned int>(input.internal_size2()>>1));
	VIENNACL_CUDA_LAST_ERROR_CHECK("transpose");

      }

      template<typename T>
      __global__ void transpose_inplace(
          T * input,
          unsigned int row_num,
          unsigned int col_num)
      {
        unsigned int size = row_num * col_num;
        for(unsigned int i = blockIdx.x * blockDim.x + threadIdx.x; i < size; i+= gridDim.x * blockDim.x)
        {
          unsigned int row = i / col_num;
          unsigned int col = i - row*col_num;
          unsigned int new_pos = col * row_num + row;
          if(i < new_pos)
          {
            T val = input[i];
            input[i] = input[new_pos];
            input[new_pos] = val;
          }
        }
      }

      /**
       * @brief Inplace_transpose matrix
       */
      template<class SCALARTYPE, unsigned int ALIGNMENT>
      void transpose(viennacl::matrix<SCALARTYPE, viennacl::row_major, ALIGNMENT> & input)
      {
        transpose_inplace<<<128,128>>>(
				reinterpret_cast<typename viennacl::linalg::cuda::detail::type_to_type2<SCALARTYPE>::type *>(detail::cuda_arg<SCALARTYPE>(input)),
				static_cast<unsigned int>(input.internal_size1()>>1),
				static_cast<unsigned int>(input.internal_size2() >> 1));
	VIENNACL_CUDA_LAST_ERROR_CHECK("transpose_inplace");

      }

      template<typename T2,typename T>
      __global__ void real_to_complex(
          const T * in,
          T2 * out,
          unsigned int size)
      {
        for (unsigned int i = blockIdx.x * blockDim.x + threadIdx.x; i < size; i += gridDim.x * blockDim.x)
        {
          T2 val;
          val.x = 0;
          val.y = 0;
          val.x = in[i];
          out[i] = val;
        }
      }

      /**
       * @brief Create complex vector from real vector (even elements(2*k) = real part, odd elements(2*k+1) = imaginary part)
       */
      template<class SCALARTYPE>
      void real_to_complex(viennacl::vector_base<SCALARTYPE> const & in,
          viennacl::vector_base<SCALARTYPE> & out, vcl_size_t size)
      {
        real_to_complex<<<128,128>>>(
        			detail::cuda_arg<SCALARTYPE>(in),
				reinterpret_cast<typename viennacl::linalg::cuda::detail::type_to_type2<SCALARTYPE>::type *>(detail::cuda_arg<SCALARTYPE>(out)),
				static_cast<unsigned int>(size));
	VIENNACL_CUDA_LAST_ERROR_CHECK("real_to_complex");

      }

      template<typename T,typename T2>
      __global__ void complex_to_real(
          const T2 * in,
          T * out,
          unsigned int size)
      {
        for (unsigned int i = blockIdx.x * blockDim.x + threadIdx.x; i < size; i += gridDim.x * blockDim.x)
        out[i] = in[i].x;
      }

      /**
       * @brief Create real vector from complex vector (even elements(2*k) = real part, odd elements(2*k+1) = imaginary part)
       */
      template<class SCALARTYPE>
      void complex_to_real(viennacl::vector_base<SCALARTYPE> const & in,
          viennacl::vector_base<SCALARTYPE>& out, vcl_size_t size)
      {

        complex_to_real<<<128,128>>>(
				reinterpret_cast<const typename viennacl::linalg::cuda::detail::type_to_type2<SCALARTYPE>::type *>(detail::cuda_arg<SCALARTYPE>(in)),
				detail::cuda_arg<SCALARTYPE>(out),
				static_cast<unsigned int>(size));
	VIENNACL_CUDA_LAST_ERROR_CHECK("complex_to_real");

      }

      template<typename T>
      __global__ void reverse_inplace(
          T * vec,
          uint size)
      {

        for(uint i = blockIdx.x * blockDim.x + threadIdx.x; i < (size >> 1); i+=gridDim.x * blockDim.x)
        {
          T val1 = vec[i];
          T val2 = vec[size - i - 1];
          vec[i] = val2;
          vec[size - i - 1] = val1;
        }
      }

      /**
       * @brief Reverse vector to oposite order and save it in input vector
       */
      template<class SCALARTYPE>
      void reverse(viennacl::vector_base<SCALARTYPE>& in)
      {

        vcl_size_t size = in.size();
        reverse_inplace<<<128,128>>>(
				detail::cuda_arg<SCALARTYPE>(in),
				static_cast<unsigned int>(size));
	VIENNACL_CUDA_LAST_ERROR_CHECK("reverse_inplace");

      }

    }  //namespace cuda
  }  //namespace linalg
}  //namespace viennacl

#endif /* FFT_OPERATIONS_HPP_ */
