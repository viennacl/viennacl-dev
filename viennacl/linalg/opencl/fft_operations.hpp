#ifndef VIENNACL_LINALG_OPENCL_FFT_OPERATIONS_HPP_
#define VIENNACL_LINALG_OPENCL_FFT_OPERATIONS_HPP_

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

/** @file viennacl/linalg/opencl/fft_operations.hpp
 @brief Implementations of Fast Furier Transformation using OpenCL
 */

#include "viennacl/forwards.h"
#include "viennacl/ocl/device.hpp"
#include "viennacl/ocl/kernel.hpp"
#include "viennacl/traits/handle.hpp"
#include "viennacl/traits/stride.hpp"
#include "viennacl/linalg/opencl/kernels/fft.hpp"

#include <viennacl/vector.hpp>
#include <viennacl/matrix.hpp>

#include <cmath>
#include <stdexcept>

namespace viennacl
{
  namespace linalg
  {
    namespace detail
    {
      namespace fft
      {

        const vcl_size_t MAX_LOCAL_POINTS_NUM = 512;

        /**
         * @brief Get number of bits
         */
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

        /**
         * @brief Find next power of two
         */
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

      } //namespce fft
    } //namespace detail

    namespace opencl
    {

      /**
       * @brief Direct algorithm for computing Fourier transformation.
       *
       * Works on any sizes of data.
       * Serial implementation has o(n^2) complexity
       */
      template<class SCALARTYPE>
      void direct(const viennacl::ocl::handle<cl_mem>& in, const viennacl::ocl::handle<cl_mem>& out,
          vcl_size_t size, vcl_size_t stride, vcl_size_t batch_num, SCALARTYPE sign = -1.0f,
          viennacl::linalg::host_based::detail::fft::FFT_DATA_ORDER::DATA_ORDER data_order =
              viennacl::linalg::host_based::detail::fft::FFT_DATA_ORDER::ROW_MAJOR)
      {
        viennacl::ocl::context & ctx = const_cast<viennacl::ocl::context &>(in.context());
        viennacl::linalg::opencl::kernels::fft<SCALARTYPE>::init(ctx);

        std::string program_string =
            viennacl::linalg::opencl::kernels::matrix_legacy<SCALARTYPE, row_major>::program_name();
        if (data_order == viennacl::linalg::host_based::detail::fft::FFT_DATA_ORDER::COL_MAJOR)
        {
          viennacl::linalg::opencl::kernels::matrix_legacy<SCALARTYPE, column_major>::init(ctx);
          program_string =
              viennacl::linalg::opencl::kernels::matrix_legacy<SCALARTYPE, column_major>::program_name();
        } else
          viennacl::linalg::opencl::kernels::matrix_legacy<SCALARTYPE, row_major>::init(ctx);

        viennacl::ocl::kernel& kernel = ctx.get_kernel(program_string, "fft_direct");
        viennacl::ocl::enqueue(
            kernel(in, out, static_cast<cl_uint>(size), static_cast<cl_uint>(stride),
                static_cast<cl_uint>(batch_num), sign));
      }

      /*
       * This function performs reorder of input data. Indexes are sorted in bit-reversal order.
       * Such reordering should be done before in-place FFT.
       */
      template<typename SCALARTYPE>
      void reorder(const viennacl::ocl::handle<cl_mem>& in, vcl_size_t size, vcl_size_t stride,
          vcl_size_t bits_datasize, vcl_size_t batch_num,
          viennacl::linalg::host_based::detail::fft::FFT_DATA_ORDER::DATA_ORDER data_order =
              viennacl::linalg::host_based::detail::fft::FFT_DATA_ORDER::ROW_MAJOR)
      {
        viennacl::ocl::context & ctx = const_cast<viennacl::ocl::context &>(in.context());
        viennacl::linalg::opencl::kernels::fft<SCALARTYPE>::init(ctx);

        std::string program_string =
            viennacl::linalg::opencl::kernels::matrix_legacy<SCALARTYPE, row_major>::program_name();
        if (data_order == viennacl::linalg::host_based::detail::fft::FFT_DATA_ORDER::COL_MAJOR)
        {
          viennacl::linalg::opencl::kernels::matrix_legacy<SCALARTYPE, column_major>::init(ctx);
          program_string =
              viennacl::linalg::opencl::kernels::matrix_legacy<SCALARTYPE, column_major>::program_name();
        } else
          viennacl::linalg::opencl::kernels::matrix_legacy<SCALARTYPE, row_major>::init(ctx);

        viennacl::ocl::kernel& kernel = ctx.get_kernel(program_string, "fft_reorder");
        viennacl::ocl::enqueue(
            kernel(in, static_cast<cl_uint>(bits_datasize), static_cast<cl_uint>(size),
                static_cast<cl_uint>(stride), static_cast<cl_uint>(batch_num)));
      }
      /**
       * @brief Radix-2 algorithm for computing Fourier transformation.
       *
       * Works only on power-of-two sizes of data.
       * Serial implementation has o(n * lg n) complexity.
       * This is a Cooley-Tukey algorithm
       */
      template<class SCALARTYPE>
      void radix2(const viennacl::ocl::handle<cl_mem>& in, vcl_size_t size, vcl_size_t stride,
          vcl_size_t batch_num, SCALARTYPE sign = -1.0f,
          viennacl::linalg::host_based::detail::fft::FFT_DATA_ORDER::DATA_ORDER data_order =
              viennacl::linalg::host_based::detail::fft::FFT_DATA_ORDER::ROW_MAJOR)
      {

        viennacl::ocl::context & ctx = const_cast<viennacl::ocl::context &>(in.context());
        viennacl::linalg::opencl::kernels::fft<SCALARTYPE>::init(ctx);

        assert(batch_num != 0);

        std::string program_string =
            viennacl::linalg::opencl::kernels::matrix_legacy<SCALARTYPE, row_major>::program_name();
        if (data_order == viennacl::linalg::host_based::detail::fft::FFT_DATA_ORDER::COL_MAJOR)
        {
          viennacl::linalg::opencl::kernels::matrix_legacy<SCALARTYPE, column_major>::init(ctx);
          program_string =
              viennacl::linalg::opencl::kernels::matrix_legacy<SCALARTYPE, column_major>::program_name();
        } else
          viennacl::linalg::opencl::kernels::matrix_legacy<SCALARTYPE, row_major>::init(ctx);

        vcl_size_t bits_datasize = viennacl::linalg::detail::fft::num_bits(size);
        if (size <= viennacl::linalg::detail::fft::MAX_LOCAL_POINTS_NUM)
        {

          viennacl::ocl::kernel& kernel = ctx.get_kernel(program_string, "fft_radix2_local");

          viennacl::ocl::enqueue(
              kernel(in, viennacl::ocl::local_mem((size * 4) * sizeof(SCALARTYPE)),
                  static_cast<cl_uint>(bits_datasize), static_cast<cl_uint>(size),
                  static_cast<cl_uint>(stride), static_cast<cl_uint>(batch_num), sign));

        } else
        {

          viennacl::linalg::opencl::reorder<SCALARTYPE>(in, size, stride, bits_datasize, batch_num);

          for (vcl_size_t step = 0; step < bits_datasize; step++)
          {
            viennacl::ocl::kernel& kernel = ctx.get_kernel(program_string, "fft_radix2");
            viennacl::ocl::enqueue(
                kernel(in, static_cast<cl_uint>(step), static_cast<cl_uint>(bits_datasize),
                    static_cast<cl_uint>(size), static_cast<cl_uint>(stride),
                    static_cast<cl_uint>(batch_num), sign));
          }

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
        viennacl::ocl::context & ctx =
            const_cast<viennacl::ocl::context &>(viennacl::traits::opencl_handle(in).context());
        viennacl::linalg::opencl::kernels::fft<SCALARTYPE>::init(ctx);

        vcl_size_t size = in.size() >> 1;
        vcl_size_t ext_size = viennacl::linalg::detail::fft::next_power_2(2 * size - 1);

        viennacl::vector<SCALARTYPE, ALIGNMENT> A(ext_size << 1);
        viennacl::vector<SCALARTYPE, ALIGNMENT> B(ext_size << 1);

        viennacl::vector<SCALARTYPE, ALIGNMENT> Z(ext_size << 1);

        {
          viennacl::ocl::kernel& kernel = ctx.get_kernel(
              viennacl::linalg::opencl::kernels::fft<SCALARTYPE>::program_name(), "zero2");
          viennacl::ocl::enqueue(kernel(A, B, static_cast<cl_uint>(ext_size)));

        }
        {
          viennacl::ocl::kernel& kernel = ctx.get_kernel(
              viennacl::linalg::opencl::kernels::fft<SCALARTYPE>::program_name(), "bluestein_pre");
          viennacl::ocl::enqueue(
              kernel(in, A, B, static_cast<cl_uint>(size), static_cast<cl_uint>(ext_size)));
        }

        viennacl::linalg::convolve_i(A, B, Z);

        {
          viennacl::ocl::kernel& kernel = ctx.get_kernel(
              viennacl::linalg::opencl::kernels::fft<SCALARTYPE>::program_name(), "bluestein_post");
          viennacl::ocl::enqueue(kernel(Z, out, static_cast<cl_uint>(size)));
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
        viennacl::ocl::context & ctx =
            const_cast<viennacl::ocl::context &>(viennacl::traits::opencl_handle(input1).context());
        viennacl::linalg::opencl::kernels::fft<SCALARTYPE>::init(ctx);
        vcl_size_t size = input1.size() >> 1;
        viennacl::ocl::kernel& kernel = ctx.get_kernel(
            viennacl::linalg::opencl::kernels::fft<SCALARTYPE>::program_name(), "fft_mult_vec");
        viennacl::ocl::enqueue(kernel(input1, input2, output, static_cast<cl_uint>(size)));
      }

      /**
       * @brief Normalize vector on with his own size
       */
      template<class SCALARTYPE, unsigned int ALIGNMENT>
      void normalize(viennacl::vector<SCALARTYPE, ALIGNMENT> & input)
      {
        viennacl::ocl::context & ctx =
            const_cast<viennacl::ocl::context &>(viennacl::traits::opencl_handle(input).context());
        viennacl::linalg::opencl::kernels::fft<SCALARTYPE>::init(ctx);

        viennacl::ocl::kernel& kernel = ctx.get_kernel(
            viennacl::linalg::opencl::kernels::fft<SCALARTYPE>::program_name(),
            "fft_div_vec_scalar");
        vcl_size_t size = input.size() >> 1;
        SCALARTYPE norm_factor = static_cast<SCALARTYPE>(size);
        viennacl::ocl::enqueue(kernel(input, static_cast<cl_uint>(size), norm_factor));
      }

      /**
       * @brief Inplace_transpose matrix
       */
      template<class SCALARTYPE, unsigned int ALIGNMENT>
      void transpose(viennacl::matrix<SCALARTYPE, viennacl::row_major, ALIGNMENT> & input)
      {
        viennacl::ocl::context & ctx =
            const_cast<viennacl::ocl::context &>(viennacl::traits::opencl_handle(input).context());
        viennacl::linalg::opencl::kernels::fft<SCALARTYPE>::init(ctx);

        viennacl::ocl::kernel& kernel = ctx.get_kernel(
            viennacl::linalg::opencl::kernels::fft<SCALARTYPE>::program_name(),
            "transpose_inplace");
        viennacl::ocl::enqueue(
            kernel(input, static_cast<cl_uint>(input.internal_size1() >> 1),
                static_cast<cl_uint>(input.internal_size2()) >> 1));
      }

      /**
       * @brief Transpose matrix
       */
      template<class SCALARTYPE, unsigned int ALIGNMENT>
      void transpose(viennacl::matrix<SCALARTYPE, viennacl::row_major, ALIGNMENT> const & input,
          viennacl::matrix<SCALARTYPE, viennacl::row_major, ALIGNMENT> & output)
      {
        viennacl::ocl::context & ctx =
            const_cast<viennacl::ocl::context &>(viennacl::traits::opencl_handle(input).context());
        viennacl::linalg::opencl::kernels::fft<SCALARTYPE>::init(ctx);

        viennacl::ocl::kernel& kernel = ctx.get_kernel(
            viennacl::linalg::opencl::kernels::fft<SCALARTYPE>::program_name(), "transpose");
        viennacl::ocl::enqueue(
            kernel(input, output, static_cast<cl_uint>(input.internal_size1() >> 1),
                static_cast<cl_uint>(input.internal_size2() >> 1)));
      }

      /**
       * @brief Create complex vector from real vector (even elements(2*k) = real part, odd elements(2*k+1) = imaginary part)
       */
      template<class SCALARTYPE>
      void real_to_complex(viennacl::vector_base<SCALARTYPE> const & in,
          viennacl::vector_base<SCALARTYPE> & out, vcl_size_t size)
      {
        viennacl::ocl::context & ctx =
            const_cast<viennacl::ocl::context &>(viennacl::traits::opencl_handle(in).context());
        viennacl::linalg::opencl::kernels::fft<SCALARTYPE>::init(ctx);
        viennacl::ocl::kernel & kernel = ctx.get_kernel(
            viennacl::linalg::opencl::kernels::fft<SCALARTYPE>::program_name(), "real_to_complex");
        viennacl::ocl::enqueue(kernel(in, out, static_cast<cl_uint>(size)));
      }

      /**
       * @brief Create real vector from complex vector (even elements(2*k) = real part, odd elements(2*k+1) = imaginary part)
       */
      template<class SCALARTYPE>
      void complex_to_real(viennacl::vector_base<SCALARTYPE> const & in,
          viennacl::vector_base<SCALARTYPE>& out, vcl_size_t size)
      {
        viennacl::ocl::context & ctx =
            const_cast<viennacl::ocl::context &>(viennacl::traits::opencl_handle(in).context());
        viennacl::linalg::opencl::kernels::fft<SCALARTYPE>::init(ctx);
        viennacl::ocl::kernel& kernel = ctx.get_kernel(
            viennacl::linalg::opencl::kernels::fft<SCALARTYPE>::program_name(), "complex_to_real");
        viennacl::ocl::enqueue(kernel(in, out, static_cast<cl_uint>(size)));
      }

      /**
       * @brief Reverse vector to oposite order and save it in input vector
       */
      template<class SCALARTYPE>
      void reverse(viennacl::vector_base<SCALARTYPE>& in)
      {
        viennacl::ocl::context & ctx =
            const_cast<viennacl::ocl::context &>(viennacl::traits::opencl_handle(in).context());
        viennacl::linalg::opencl::kernels::fft<SCALARTYPE>::init(ctx);
        vcl_size_t size = in.size();
        viennacl::ocl::kernel& kernel = ctx.get_kernel(
            viennacl::linalg::opencl::kernels::fft<SCALARTYPE>::program_name(), "reverse_inplace");
        viennacl::ocl::enqueue(kernel(in, static_cast<cl_uint>(size)));
      }
    } //namespace opencl
  } //namespace linalg
} //namespace viennacl

#endif /* FFT_OPERATIONS_HPP_ */

