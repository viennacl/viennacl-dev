#ifndef VIENNACL_LINALG_HOST_BASED_FFT_OPERATIONS_HPP_
#define VIENNACL_LINALG_HOST_BASED_FFT_OPERATIONS_HPP_

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

/** @file viennacl/linalg/host_based/fft_operations.hpp
 @brief Implementations of Fast Furier Transformation using a plain single-threaded or OpenMP-enabled execution on CPU
 */

//TODO openom Conditions
#include <viennacl/vector.hpp>
#include <viennacl/matrix.hpp>

#include "viennacl/linalg/host_based/vector_operations.hpp"

#include <stdexcept>
#include <math.h>
#include <complex>

namespace viennacl
{
  namespace linalg
  {
    namespace host_based
    {
      namespace detail
      {
        namespace fft
        {
          const vcl_size_t MAX_LOCAL_POINTS_NUM = 512;

          namespace FFT_DATA_ORDER
          {

            enum DATA_ORDER
            {
              ROW_MAJOR, COL_MAJOR
            };
          }

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

          inline unsigned int get_reorder_num(unsigned int v, unsigned int bit_size)
          {
            v = ((v >> 1) & 0x55555555) | ((v & 0x55555555) << 1);
            v = ((v >> 2) & 0x33333333) | ((v & 0x33333333) << 2);
            v = ((v >> 4) & 0x0F0F0F0F) | ((v & 0x0F0F0F0F) << 4);
            v = ((v >> 8) & 0x00FF00FF) | ((v & 0x00FF00FF) << 8);
            v = (v >> 16) | (v << 16);
            v = v >> (32 - bit_size);
            return v;
          }

          template<class SCALARTYPE, unsigned int ALIGNMENT>
          void copy_to_complex_array(std::complex<SCALARTYPE> * input_complex,
              const viennacl::vector<SCALARTYPE, ALIGNMENT>& in, int size)
          {
#ifdef VIENNACL_WITH_OPENMP
#pragma omp parallel for if (size > VIENNACL_OPENMP_VECTOR_MIN_SIZE)
#endif
            for (unsigned int i = 0; i < size * 2; i += 2)
            { //change array to complex array
              input_complex[i / 2] = std::complex<SCALARTYPE>(in[i], in[i + 1]);
            }
          }

          template<class SCALARTYPE>
          void copy_to_complex_array(std::complex<SCALARTYPE> * input_complex,
              const viennacl::vector_base<SCALARTYPE> & in, int size)
          {
#ifdef VIENNACL_WITH_OPENMP
#pragma omp parallel for if (size > VIENNACL_OPENMP_VECTOR_MIN_SIZE)
#endif
            for (unsigned int i = 0; i < size * 2; i += 2)
            { //change array to complex array
              input_complex[i / 2] = std::complex<SCALARTYPE>(in[i], in[i + 1]);
            }
          }

          template<class SCALARTYPE, unsigned int ALIGNMENT>
          void copy_to_vector(std::complex<SCALARTYPE> * input_complex,
              viennacl::vector<SCALARTYPE, ALIGNMENT>& in, unsigned int size)
          {
#ifdef VIENNACL_WITH_OPENMP
#pragma omp parallel for if (size > VIENNACL_OPENMP_VECTOR_MIN_SIZE)
#endif
            for (unsigned int i = 0; i < size; i += 1)
            {
              in(i * 2) = (SCALARTYPE) std::real(input_complex[i]);
              in(i * 2 + 1) = std::imag(input_complex[i]);
            }
          }

          template<class SCALARTYPE>
          void copy_to_complex_array(std::complex<SCALARTYPE> * input_complex,
              const SCALARTYPE * in, int size)
          {
#ifdef VIENNACL_WITH_OPENMP
#pragma omp parallel for if (size > VIENNACL_OPENMP_VECTOR_MIN_SIZE)
#endif
            for (unsigned int i = 0; i < size * 2; i += 2)
            { //change array to complex array
              input_complex[i / 2] = std::complex<SCALARTYPE>(in[i], in[i + 1]);
            }
          }

          template<class SCALARTYPE>
          void copy_to_vector(std::complex<SCALARTYPE> * input_complex, SCALARTYPE * in,
              unsigned int size)
          {
#ifdef VIENNACL_WITH_OPENMP
#pragma omp parallel for if (size > VIENNACL_OPENMP_VECTOR_MIN_SIZE)
#endif
            for (unsigned int i = 0; i < size; i += 1)
            {
              in[i * 2] = (SCALARTYPE) std::real(input_complex[i]);
              in[i * 2 + 1] = std::imag(input_complex[i]);
            }
          }

          template<class SCALARTYPE>
          void copy_to_vector(std::complex<SCALARTYPE> * input_complex,
              viennacl::vector_base<SCALARTYPE> & in, unsigned int size)
          {
            std::vector<SCALARTYPE> temp(2 * size);
#ifdef VIENNACL_WITH_OPENMP
#pragma omp parallel for if (size > VIENNACL_OPENMP_VECTOR_MIN_SIZE)
#endif
            for (unsigned int i = 0; i < size; i += 1)
            {
              temp[i * 2] = (SCALARTYPE) std::real(input_complex[i]);
              temp[i * 2 + 1] = std::imag(input_complex[i]);
            }
            viennacl::copy(temp, in);
          }

          template<class SCALARTYPE>
          void zero2(SCALARTYPE *input1, SCALARTYPE *input2, unsigned int size)
          {
#ifdef VIENNACL_WITH_OPENMP
#pragma omp parallel for if (size > VIENNACL_OPENMP_VECTOR_MIN_SIZE)
#endif
            for (unsigned int i = 0; i < size; i += 1)
            {
              input1[i] = 0;
              input2[i] = 0;
            }
          }

        } //namespace fft

      } //namespace detail

      /**
       * @brief Direct algoritm kenrnel
       */
      template<class SCALARTYPE>
      void fft_direct(std::complex<SCALARTYPE> * input_complex, std::complex<SCALARTYPE> * output,
          unsigned int size, unsigned int stride, unsigned int batch_num, SCALARTYPE sign,
          viennacl::linalg::host_based::detail::fft::FFT_DATA_ORDER::DATA_ORDER data_order =
              viennacl::linalg::host_based::detail::fft::FFT_DATA_ORDER::ROW_MAJOR)
      {
        const SCALARTYPE NUM_PI = 3.14159265358979323846;
#ifdef VIENNACL_WITH_OPENMP
#pragma omp parallel
#endif
        for (unsigned int batch_id = 0; batch_id < batch_num; batch_id++)
        {

          for (unsigned int k = 0; k < size; k += 1)
          {
            std::complex<SCALARTYPE> f = 0;
            for (unsigned int n = 0; n < size; n++)
            {
              std::complex<SCALARTYPE> input;
              if (!data_order)
                input = input_complex[batch_id * stride + n]; //input index here
              else
                input = input_complex[n * stride + batch_id];
              SCALARTYPE sn, cs;
              SCALARTYPE arg = sign * 2 * NUM_PI * k / size * n;
              sn = sin(arg);
              cs = cos(arg);

              std::complex<SCALARTYPE> ex(cs, sn);
              std::complex<SCALARTYPE> tmp(input.real() * ex.real() - input.imag() * ex.imag(),
                  input.real() * ex.imag() + input.imag() * ex.real());
              f = f + tmp;
            }
            if (!data_order)
              output[batch_id * stride + k] = f;   // output index here
            else
              output[k * stride + batch_id] = f;
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

        std::complex<SCALARTYPE> input_complex[size * batch_num];
        std::complex<SCALARTYPE> output[size * batch_num];

        viennacl::linalg::host_based::detail::fft::copy_to_complex_array(input_complex, in,
            size * batch_num);

        fft_direct(input_complex, output, size, stride, batch_num, sign, data_order);

        viennacl::linalg::host_based::detail::fft::copy_to_vector(output, out, size * batch_num);
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
        unsigned int row_num = in.internal_size1();
        unsigned int col_num = in.internal_size2() >> 1;

        unsigned int size_mat = row_num * col_num;

        std::complex<SCALARTYPE> input_complex[size_mat];
        std::complex<SCALARTYPE> output[size_mat];

        const SCALARTYPE * data_A = detail::extract_raw_pointer<SCALARTYPE>(in);
        SCALARTYPE * data_B = detail::extract_raw_pointer<SCALARTYPE>(out);

        viennacl::linalg::host_based::detail::fft::copy_to_complex_array(input_complex, data_A,
            size_mat);

        fft_direct(input_complex, output, size, stride, batch_num, sign, data_order);

        viennacl::linalg::host_based::detail::fft::copy_to_vector(output, data_B, size_mat);
      }

      /*
       * This function performs reorder of 1D input  data. Indexes are sorted in bit-reversal order.
       * Such reordering should be done before in-place FFT.
       */
      template<class SCALARTYPE, unsigned int ALIGNMENT>
      void reorder(viennacl::vector<SCALARTYPE, ALIGNMENT>& in, vcl_size_t size, vcl_size_t stride,
          vcl_size_t bits_datasize, vcl_size_t batch_num,
          viennacl::linalg::host_based::detail::fft::FFT_DATA_ORDER::DATA_ORDER data_order =
              viennacl::linalg::host_based::detail::fft::FFT_DATA_ORDER::ROW_MAJOR)
      {
        std::complex<SCALARTYPE> input[size * batch_num];
        viennacl::linalg::host_based::detail::fft::copy_to_complex_array(input, in,
            size * batch_num);
#ifdef VIENNACL_WITH_OPENMP
#pragma omp parallel for
#endif
        for (unsigned int batch_id = 0; batch_id < batch_num; batch_id++)
        {
          for (unsigned int i = 0; i < size; i++)
          {
            unsigned int v = viennacl::linalg::host_based::detail::fft::get_reorder_num(i,
                bits_datasize);
            if (i < v)
            {
              if (!data_order)
              {
                std::complex<SCALARTYPE> tmp = input[batch_id * stride + i]; // index
                input[batch_id * stride + i] = input[batch_id * stride + v]; //index
                input[batch_id * stride + v] = tmp;      //index
              } else
              {
                std::complex<SCALARTYPE> tmp = input[i * stride + batch_id]; // index
                input[i * stride + batch_id] = input[v * stride + batch_id]; //index
                input[v * stride + batch_id] = tmp;      //index
              }
            }
          }
        }
        viennacl::linalg::host_based::detail::fft::copy_to_vector(input, in, size * batch_num);
      }

      /*
       * This function performs reorder of 2D input  data. Indexes are sorted in bit-reversal order.
       * Such reordering should be done before in-place FFT.
       */
      template<class SCALARTYPE, unsigned int ALIGNMENT>
      void reorder(viennacl::matrix<SCALARTYPE, viennacl::row_major, ALIGNMENT>& in,
          vcl_size_t size, vcl_size_t stride, vcl_size_t bits_datasize, vcl_size_t batch_num,
          viennacl::linalg::host_based::detail::fft::FFT_DATA_ORDER::DATA_ORDER data_order =
              viennacl::linalg::host_based::detail::fft::FFT_DATA_ORDER::ROW_MAJOR)
      {

        SCALARTYPE * data = detail::extract_raw_pointer<SCALARTYPE>(in);
        unsigned int row_num = in.internal_size1();
        unsigned int col_num = in.internal_size2() >> 1;
        unsigned int size_mat = row_num * col_num;

        std::complex<SCALARTYPE> input[size_mat];

        viennacl::linalg::host_based::detail::fft::copy_to_complex_array(input, data, size_mat);

#ifdef VIENNACL_WITH_OPENMP
#pragma omp parallel for
#endif
        for (unsigned int batch_id = 0; batch_id < batch_num; batch_id++)
        {
          for (unsigned int i = 0; i < size; i++)
          {
            unsigned int v = viennacl::linalg::host_based::detail::fft::get_reorder_num(i,
                bits_datasize);
            if (i < v)
            {
              if (!data_order)
              {
                std::complex<SCALARTYPE> tmp = input[batch_id * stride + i]; // index
                input[batch_id * stride + i] = input[batch_id * stride + v]; //index
                input[batch_id * stride + v] = tmp;      //index
              } else
              {
                std::complex<SCALARTYPE> tmp = input[i * stride + batch_id]; // index
                input[i * stride + batch_id] = input[v * stride + batch_id]; //index
                input[v * stride + batch_id] = tmp;      //index
              }
            }
          }
        }
        viennacl::linalg::host_based::detail::fft::copy_to_vector(input, data, size_mat);
      }

      /**
       * @brief Radix-2 algorithm for computing Fourier transformation.
       * Kernel for computing smaller amount of data
       */
      template<class SCALARTYPE>
      void fft_radix2(std::complex<SCALARTYPE> * input_complex, unsigned int batch_num,
          unsigned int bit_size, unsigned int size, unsigned int stride, SCALARTYPE sign,
          viennacl::linalg::host_based::detail::fft::FFT_DATA_ORDER::DATA_ORDER data_order =
              viennacl::linalg::host_based::detail::fft::FFT_DATA_ORDER::ROW_MAJOR)
      {
        const SCALARTYPE NUM_PI = 3.14159265358979323846;

        for (vcl_size_t step = 0; step < bit_size; step++)
        {
          unsigned int ss = 1 << step;
          unsigned int half_size = size >> 1;
          SCALARTYPE cs, sn;
#ifdef VIENNACL_WITH_OPENMP
#pragma omp parallel for private(cs,sn) shared(ss,half_size,step)
#endif
          for (unsigned int batch_id = 0; batch_id < batch_num; batch_id++)
          {
#ifdef VIENNACL_WITH_OPENMP
#pragma omp parallel for private(cs,sn) shared(ss,half_size)
#endif
            for (unsigned int tid = 0; tid < half_size; tid++)
            {
              unsigned int group = (tid & (ss - 1));
              unsigned int pos = ((tid >> step) << (step + 1)) + group;
              std::complex<SCALARTYPE> in1;
              std::complex<SCALARTYPE> in2;
              unsigned int offset;
              if (!data_order)
              {
                offset = batch_id * stride + pos;
                in1 = input_complex[offset];
                in2 = input_complex[offset + ss];
              } else
              {
                offset = pos * stride + batch_id;
                in1 = input_complex[offset];
                in2 = input_complex[offset + ss * stride];
              }
              SCALARTYPE arg = group * sign * NUM_PI / ss;
              sn = sin(arg);
              cs = cos(arg);
              std::complex<SCALARTYPE> ex(cs, sn);
              std::complex<SCALARTYPE> tmp(in2.real() * ex.real() - in2.imag() * ex.imag(),
                  in2.real() * ex.imag() + in2.imag() * ex.real());
              if (!data_order)
                input_complex[offset + ss] = in1 - tmp;
              else
                input_complex[offset + ss * stride] = in1 - tmp;
              input_complex[offset] = in1 + tmp;

            }
          }
        }

      }

      /**
       * @brief Radix-2 algorithm for computing Fourier transformation.
       * Kernel for computing bigger amount of data
       */
      template<class SCALARTYPE>
      void fft_radix2_local(std::complex<SCALARTYPE> * input_complex,
          std::complex<SCALARTYPE> * lcl_input, unsigned int batch_num, unsigned int bit_size,
          unsigned int size, unsigned int stride, SCALARTYPE sign,
          viennacl::linalg::host_based::detail::fft::FFT_DATA_ORDER::DATA_ORDER data_order =
              viennacl::linalg::host_based::detail::fft::FFT_DATA_ORDER::ROW_MAJOR)
      {
        const SCALARTYPE NUM_PI = 3.14159265358979323846;

        for (unsigned int batch_id = 0; batch_id < batch_num; batch_id++)
        {
#ifdef VIENNACL_WITH_OPENMP
#pragma omp parallel for
#endif
          for (unsigned int p = 0; p < size; p += 1)
          {
            unsigned int v = viennacl::linalg::host_based::detail::fft::get_reorder_num(p,
                bit_size);

            if (!data_order)
              lcl_input[v] = input_complex[batch_id * stride + p]; //index
            else
              lcl_input[v] = input_complex[p * stride + batch_id];
          }

          for (unsigned int s = 0; s < bit_size; s++)
          {
            unsigned int ss = 1 << s;
            SCALARTYPE cs, sn;
#ifdef VIENNACL_WITH_OPENMP
#pragma omp parallel for private(cs,sn) shared(ss,s)
#endif
            for (unsigned int tid = 0; tid < size; tid++)
            {
              unsigned int group = (tid & (ss - 1));
              unsigned int pos = ((tid >> s) << (s + 1)) + group;

              std::complex<SCALARTYPE> in1 = lcl_input[pos];
              std::complex<SCALARTYPE> in2 = lcl_input[pos + ss];

              SCALARTYPE arg = group * sign * NUM_PI / ss;

              sn = sin(arg);
              cs = cos(arg);
              std::complex<SCALARTYPE> ex(cs, sn);

              std::complex<SCALARTYPE> tmp(in2.real() * ex.real() - in2.imag() * ex.imag(),
                  in2.real() * ex.imag() + in2.imag() * ex.real());

              lcl_input[pos + ss] = in1 - tmp;
              lcl_input[pos] = in1 + tmp;
            }

          }
#ifdef VIENNACL_WITH_OPENMP
#pragma omp parallel for shared(batch_id)
#endif
          //copy local array back to global memory
          for (unsigned int p = 0; p < size; p += 1)
          {
            if (!data_order)
              input_complex[batch_id * stride + p] = lcl_input[p];
            else
              input_complex[p * stride + batch_id] = lcl_input[p];

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

        unsigned int bit_size = viennacl::linalg::host_based::detail::fft::num_bits(size);

        std::complex<SCALARTYPE> input_complex[size * batch_num];
        std::complex<SCALARTYPE> lcl_input[size * batch_num];
        viennacl::linalg::host_based::detail::fft::copy_to_complex_array(input_complex, in,
            size * batch_num);

        if (size <= viennacl::linalg::host_based::detail::fft::MAX_LOCAL_POINTS_NUM)
        {
          viennacl::linalg::host_based::fft_radix2_local(input_complex, lcl_input, batch_num,
              bit_size, size, stride, sign, data_order);

        } else
        {

          viennacl::linalg::host_based::reorder<SCALARTYPE>(in, size, stride, bit_size, batch_num,
              data_order);

          viennacl::linalg::host_based::detail::fft::copy_to_complex_array(input_complex, in,
              size * batch_num);

          viennacl::linalg::host_based::fft_radix2(input_complex, batch_num, bit_size, size, stride,
              sign, data_order);

        }

        viennacl::linalg::host_based::detail::fft::copy_to_vector(input_complex, in,
            size * batch_num);
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

        unsigned int bit_size = viennacl::linalg::host_based::detail::fft::num_bits(size);

        SCALARTYPE * data = detail::extract_raw_pointer<SCALARTYPE>(in);

        unsigned int row_num = in.internal_size1();
        unsigned int col_num = in.internal_size2() >> 1;
        unsigned int size_mat = row_num * col_num;

        std::complex<SCALARTYPE> input_complex[size_mat];

        viennacl::linalg::host_based::detail::fft::copy_to_complex_array(input_complex, data,
            size_mat);
        if (size <= viennacl::linalg::host_based::detail::fft::MAX_LOCAL_POINTS_NUM)
        {
          //std::cout<<bit_size<<","<<size<<","<<stride<<","<<batch_num<<","<<size<<","<<sign<<","<<data_order<<std::endl;
          std::complex<SCALARTYPE> lcl_input[size_mat];
          viennacl::linalg::host_based::fft_radix2_local(input_complex, lcl_input, batch_num,
              bit_size, size, stride, sign, data_order);

        } else
        {

          viennacl::linalg::host_based::reorder<SCALARTYPE>(in, size, stride, bit_size, batch_num,
              data_order);
          std::cout << "in" << in << std::endl;

          viennacl::linalg::host_based::detail::fft::copy_to_complex_array(input_complex, data,
              size_mat);

          viennacl::linalg::host_based::fft_radix2(input_complex, batch_num, bit_size, size, stride,
              sign, data_order);
        }

        viennacl::linalg::host_based::detail::fft::copy_to_vector(input_complex, data, size_mat);

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
        vcl_size_t ext_size = viennacl::linalg::host_based::detail::fft::next_power_2(2 * size - 1);

        viennacl::vector<SCALARTYPE, ALIGNMENT> A(ext_size << 1);
        viennacl::vector<SCALARTYPE, ALIGNMENT> B(ext_size << 1);
        viennacl::vector<SCALARTYPE, ALIGNMENT> Z(ext_size << 1);

        std::complex<SCALARTYPE> input_complex[size];
        std::complex<SCALARTYPE> output_complex[size];

        std::complex<SCALARTYPE> A_complex[ext_size];
        std::complex<SCALARTYPE> B_complex[ext_size];
        std::complex<SCALARTYPE> Z_complex[ext_size];

        viennacl::linalg::host_based::detail::fft::copy_to_complex_array(input_complex, in, size);
#ifdef VIENNACL_WITH_OPENMP
#pragma omp parallel for
#endif
        for (unsigned int i = 0; i < ext_size; i++)
        {
          A_complex[i] = 0;
          B_complex[i] = 0;
        }

        unsigned int double_size = size << 1;

        SCALARTYPE sn_a, cs_a;
        const SCALARTYPE NUM_PI = 3.14159265358979323846;
#ifdef VIENNACL_WITH_OPENMP
#pragma omp parallel for private(sn_a,cs_a)
#endif
        for (unsigned int i = 0; i < size; i++)
        {
          unsigned int rm = i * i % (double_size);
          SCALARTYPE angle = (SCALARTYPE) rm / size * NUM_PI;

          sn_a = sin(-angle);
          cs_a = cos(-angle);

          std::complex<SCALARTYPE> a_i(cs_a, sn_a);
          std::complex<SCALARTYPE> b_i(cs_a, -sn_a);

          A_complex[i] = std::complex<SCALARTYPE>(
              input_complex[i].real() * a_i.real() - input_complex[i].imag() * a_i.imag(),
              input_complex[i].real() * a_i.imag() + input_complex[i].imag() * a_i.real());
          B_complex[i] = b_i;

          // very bad instruction, to be fixed
          if (i)
            B_complex[ext_size - i] = b_i;
        }

        viennacl::linalg::host_based::detail::fft::copy_to_vector(input_complex, in, size);
        viennacl::linalg::host_based::detail::fft::copy_to_vector(A_complex, A, ext_size);
        viennacl::linalg::host_based::detail::fft::copy_to_vector(B_complex, B, ext_size);

        viennacl::linalg::convolve_i(A, B, Z);

        viennacl::linalg::host_based::detail::fft::copy_to_complex_array(Z_complex, Z, ext_size);

#ifdef VIENNACL_WITH_OPENMP
#pragma omp parallel for private(sn_a,cs_a)
#endif
        for (unsigned int i = 0; i < size; i++)
        {
          unsigned int rm = i * i % (double_size);
          SCALARTYPE angle = (SCALARTYPE) rm / size * (-NUM_PI);
          sn_a = sin(angle);
          cs_a = cos(angle);
          std::complex<SCALARTYPE> b_i(cs_a, sn_a);
          output_complex[i] = std::complex<SCALARTYPE>(
              Z_complex[i].real() * b_i.real() - Z_complex[i].imag() * b_i.imag(),
              Z_complex[i].real() * b_i.imag() + Z_complex[i].imag() * b_i.real());
        }
        viennacl::linalg::host_based::detail::fft::copy_to_vector(output_complex, out, size);

      }

      /**
       * @brief Normalize vector with his own size
       */
      template<class SCALARTYPE, unsigned int ALIGNMENT>
      void normalize(viennacl::vector<SCALARTYPE, ALIGNMENT> & input)
      {
        vcl_size_t size = input.size() >> 1;
        SCALARTYPE norm_factor = static_cast<SCALARTYPE>(size);
        for (unsigned int i = 0; i < size * 2; i++)
        {
          input[i] /= norm_factor;
        }

      }

      /**
       * @brief Complex multiplikation of two vectors
       */
      template<class SCALARTYPE, unsigned int ALIGNMENT>
      void multiply_complex(viennacl::vector<SCALARTYPE, ALIGNMENT> const & input1,
          viennacl::vector<SCALARTYPE, ALIGNMENT> const & input2,
          viennacl::vector<SCALARTYPE, ALIGNMENT> & output)
      {

        vcl_size_t size = input1.size() >> 1;

        std::complex<SCALARTYPE> input1_complex[size];
        std::complex<SCALARTYPE> input2_complex[size];
        std::complex<SCALARTYPE> output_complex[size];
        viennacl::linalg::host_based::detail::fft::copy_to_complex_array(input1_complex, input1,
            size);
        viennacl::linalg::host_based::detail::fft::copy_to_complex_array(input2_complex, input2,
            size);

#ifdef VIENNACL_WITH_OPENMP
#pragma omp parallel for
#endif
        for (unsigned int i = 0; i < size; i++)
        {
          std::complex<SCALARTYPE> in1 = input1_complex[i];
          std::complex<SCALARTYPE> in2 = input2_complex[i];
          output_complex[i] = std::complex<SCALARTYPE>(
              in1.real() * in2.real() - in1.imag() * in2.imag(),
              in1.real() * in2.imag() + in1.imag() * in2.real());
        }
        viennacl::linalg::host_based::detail::fft::copy_to_vector(output_complex, output, size);

      }
      /**
       * @brief Inplace transpose of matrix
       */
      template<class SCALARTYPE, unsigned int ALIGNMENT>
      void transpose(viennacl::matrix<SCALARTYPE, viennacl::row_major, ALIGNMENT> & input)
      {
        unsigned int row_num = input.internal_size1() / 2;
        unsigned int col_num = static_cast<unsigned int>(input.internal_size2()) / 2;

        unsigned int size = row_num * col_num;

        SCALARTYPE * data = detail::extract_raw_pointer<SCALARTYPE>(input);

        std::complex<SCALARTYPE> input_complex[size];

        viennacl::linalg::host_based::detail::fft::copy_to_complex_array(input_complex, data, size);
#ifdef VIENNACL_WITH_OPENMP
#pragma omp parallel for shared(row_num,col_num)
#endif
        for (unsigned int i = 0; i < size; i++)
        {
          unsigned int row = i / col_num;
          unsigned int col = i - row * col_num;
          unsigned int new_pos = col * row_num + row;

          if (i < new_pos)
          {
            std::complex<SCALARTYPE> val = input_complex[i];
            input_complex[i] = input_complex[new_pos];
            input_complex[new_pos] = val;
          }
        }
        viennacl::linalg::host_based::detail::fft::copy_to_vector(input_complex, data, size);

      }

      /**
       * @brief Transpose matrix
       */
      template<class SCALARTYPE, unsigned int ALIGNMENT>
      void transpose(viennacl::matrix<SCALARTYPE, viennacl::row_major, ALIGNMENT> const & input,
          viennacl::matrix<SCALARTYPE, viennacl::row_major, ALIGNMENT> & output)
      {

        unsigned int row_num = input.internal_size1() / 2;
        unsigned int col_num = input.internal_size2() / 2;
        unsigned int size = row_num * col_num;

        const SCALARTYPE * data_A = detail::extract_raw_pointer<SCALARTYPE>(input);
        SCALARTYPE * data_B = detail::extract_raw_pointer<SCALARTYPE>(output);

        std::complex<SCALARTYPE> input_complex[size];
        viennacl::linalg::host_based::detail::fft::copy_to_complex_array(input_complex, data_A,
            size);

        std::complex<SCALARTYPE> output_complex[size];
#ifdef VIENNACL_WITH_OPENMP
#pragma omp parallel for shared(row_num,col_num)
#endif
        for (unsigned int i = 0; i < size; i++)
        {
          unsigned int row = i / col_num;
          unsigned int col = i % col_num;
          unsigned int new_pos = col * row_num + row;
          output_complex[new_pos] = input_complex[i];
        }
        viennacl::linalg::host_based::detail::fft::copy_to_vector(output_complex, data_B, size);
      }

      /**
       * @brief Create complex vector from real vector (even elements(2*k) = real part, odd elements(2*k+1) = imaginary part)
       */
      template<class SCALARTYPE>
      void real_to_complex(viennacl::vector_base<SCALARTYPE> const & in,
          viennacl::vector_base<SCALARTYPE> & out, vcl_size_t size)
      {

        std::complex<SCALARTYPE> out_complex[size];

#ifdef VIENNACL_WITH_OPENMP
#pragma omp parallel for if (size > VIENNACL_OPENMP_VECTOR_MIN_SIZE)
#endif
        for (unsigned int i = 0; i < size; i++)
        {
          std::complex<SCALARTYPE> val = 0;
          val.real() = in[i];
          out_complex[i] = val;
        }
        viennacl::linalg::host_based::detail::fft::copy_to_vector(out_complex, out, int(size));
      }

      /**
       * @brief Create real vector from complex vector (even elements(2*k) = real part, odd elements(2*k+1) = imaginary part)
       */
      template<class SCALARTYPE>
      void complex_to_real(viennacl::vector_base<SCALARTYPE> const & in,
          viennacl::vector_base<SCALARTYPE> & out, vcl_size_t size)
      {
        std::complex<SCALARTYPE> input1_complex[size];
        viennacl::linalg::host_based::detail::fft::copy_to_complex_array(input1_complex, in, size);
#ifdef VIENNACL_WITH_OPENMP
#pragma omp parallel for if (size > VIENNACL_OPENMP_VECTOR_MIN_SIZE)
#endif

        for (unsigned int i = 0; i < size; i++)
        {
          out[i] = input1_complex[i].real();
        }
      }

      /**
       * @brief Reverse vector to oposite order and save it in input vector
       */
      template<class SCALARTYPE>
      void reverse(viennacl::vector_base<SCALARTYPE>& in)
      {
        unsigned int size = in.size();
#ifdef VIENNACL_WITH_OPENMP
#pragma omp parallel for if (size > VIENNACL_OPENMP_VECTOR_MIN_SIZE)
#endif
        for (unsigned int i = 0; i < size; i++)
        {
          SCALARTYPE val1 = in[i];
          SCALARTYPE val2 = in[size - i - 1];
          in[i] = val2;
          in[size - i - 1] = val1;
        }
      }

    }      //namespace host_based
  }      //namespace linalg
}      //namespace viennacl

#endif /* FFT_OPERATIONS_HPP_ */
