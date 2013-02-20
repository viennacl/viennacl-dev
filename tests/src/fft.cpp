
/* =========================================================================
   Copyright (c) 2010-2013, Institute for Microelectronics,
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

#include <iostream>
#include <vector>
#include <cmath>
#include <complex>
#include <fstream>
#include <algorithm>

//#define VIENNACL_BUILD_INFO

#include "viennacl/fft.hpp"

typedef float ScalarType;

const ScalarType EPS = 0.06;  //use smaller values in double precision

typedef ScalarType (*test_function_ptr)(std::vector<ScalarType>&,
                                        std::vector<ScalarType>&,
                                        unsigned int,
                                        unsigned int,
                                        unsigned int);
typedef void (*input_function_ptr)(std::istream&,
                                   std::vector<ScalarType>&,
                                   std::vector<ScalarType>&,
                                   unsigned int&,
                                   unsigned int&,
                                   unsigned int&);

void read_vectors_pair(std::istream& str,
                      std::vector<ScalarType>& input,
                      std::vector<ScalarType>& output,
                      unsigned int& rows,
                      unsigned int& cols,
                      unsigned int& batch_size) 
{
    rows = 1;

    str >> cols >> batch_size;
    input.resize(2 * cols * batch_size);
    output.resize(2 * cols * batch_size);

    for(unsigned int i = 0; i < input.size(); i++) 
        str >> input[i];

    for(unsigned int i = 0; i < output.size(); i++) 
        str >> output[i];
}

void read_matrices_pair(std::istream& str,
                        std::vector<ScalarType>& input,
                        std::vector<ScalarType>& output,
                        unsigned int& rows,
                        unsigned int& cols,
                        unsigned int& batch_size) 
{
    batch_size = 1;
    str >> rows >> cols;

    input.resize(2 * rows * cols);
    output.resize(2 * rows * cols);

    for(unsigned int i = 0; i < input.size(); i++) {
        str >> input[i];
    }

    for(unsigned int i = 0; i < output.size(); i++) {
        str >> output[i];
    }
}

template <typename ScalarType>
ScalarType diff(std::vector<ScalarType>& vec, std::vector<ScalarType>& ref) 
{
    ScalarType df = 0.0;
    ScalarType norm_ref = 0;

    for(std::size_t i = 0; i < vec.size(); i++) 
    {
        df = df + pow(vec[i] - ref[i], 2);
        norm_ref += ref[i] * ref[i];
    }

    return sqrt(df / norm_ref) ;
}

template <typename ScalarType>
ScalarType diff_max(std::vector<ScalarType>& vec, std::vector<ScalarType>& ref) 
{
  ScalarType df = 0.0;
  ScalarType mx = 0.0;
  ScalarType norm_max = 0;
  
  for (std::size_t i = 0; i < vec.size(); i++) 
  {
    df = std::max<ScalarType>(fabs(vec[i] - ref[i]), df);
    mx = std::max<ScalarType>(fabs(vec[i]), mx);
    
    if (mx > 0)
    {
      if (norm_max < df / mx)
        norm_max = df / mx;
    }
  }
  
  return norm_max;
}

void convolve_ref(std::vector<ScalarType>& in1,
                  std::vector<ScalarType>& in2,
                  std::vector<ScalarType>& out) 
{
    out.resize(in1.size());
    unsigned int data_size = in1.size() >> 1;

    for(unsigned int n = 0; n < data_size; n++) {
        std::complex<ScalarType> el;
        for(unsigned int k = 0; k < data_size; k++) {
            int offset = (n - k);
            if(offset < 0) offset += data_size;
            std::complex<ScalarType> m1(in1[2*k], in1[2*k + 1]);
            std::complex<ScalarType> m2(in2[2*offset], in2[2*offset + 1]);
//            std::cout << offset << " " << m1 << " " << m2 << "\n";
            el = el +  m1 * m2 ;
        }
        //std::cout << "Answer - " << el << "\n";
        out[2*n] = el.real();
        out[2*n + 1] = el.imag();
    }
}

ScalarType opencl_fft(std::vector<ScalarType>& in,
                      std::vector<ScalarType>& out,
                      unsigned int /*row*/, unsigned int /*col*/, unsigned int batch_size)
{
    viennacl::vector<ScalarType> input(in.size());
    viennacl::vector<ScalarType> output(in.size());

    std::vector<ScalarType> res(in.size());

    viennacl::fast_copy(in, input);

    viennacl::fft(input, output, batch_size);

    viennacl::backend::finish();
    viennacl::fast_copy(output, res);

    return diff_max(res, out);
}

ScalarType opencl_2d_fft_1arg(std::vector<ScalarType>& in,
                              std::vector<ScalarType>& out,
                              unsigned int row, unsigned int col, unsigned int /*batch_size*/)
{
    viennacl::matrix<ScalarType> input(row, 2 * col);

    std::vector<ScalarType> res(in.size());

    viennacl::fast_copy(&in[0], &in[0] + in.size(), input);
    //std::cout << input << "\n";
    viennacl::inplace_fft(input);
    //std::cout << input << "\n";
    viennacl::backend::finish();
    viennacl::fast_copy(input, &res[0]);

    return diff_max(res, out);
}

ScalarType opencl_2d_fft_2arg(std::vector<ScalarType>& in,
                              std::vector<ScalarType>& out,
                              unsigned int row, unsigned int col, unsigned int /*batch_size*/)
{
    viennacl::matrix<ScalarType> input(row, 2 * col);
    viennacl::matrix<ScalarType> output(row, 2 * col);

    std::vector<ScalarType> res(in.size());

    viennacl::fast_copy(&in[0], &in[0] + in.size(), input);
    //std::cout << input << "\n";
    viennacl::fft(input, output);
    //std::cout << input << "\n";
    viennacl::backend::finish();
    viennacl::fast_copy(output, &res[0]);

    return diff_max(res, out);
}

ScalarType opencl_direct(std::vector<ScalarType>& in,
                         std::vector<ScalarType>& out,
                         unsigned int /*row*/, unsigned int /*col*/, unsigned int batch_num)
{
    viennacl::vector<ScalarType> input(in.size());
    viennacl::vector<ScalarType> output(in.size());

    std::vector<ScalarType> res(in.size());

    viennacl::fast_copy(in, input);

    unsigned int size = (input.size() >> 1) / batch_num;

    viennacl::detail::fft::direct<ScalarType>(input.handle().opencl_handle(), output.handle().opencl_handle(), size, size, batch_num);

    viennacl::backend::finish();
    viennacl::fast_copy(output, res);

    return diff_max(res, out);
}

ScalarType opencl_bluestein(std::vector<ScalarType>& in,
                            std::vector<ScalarType>& out,
                            unsigned int /*row*/, unsigned int /*col*/, unsigned int batch_size)
{
    viennacl::vector<ScalarType> input(in.size());
    viennacl::vector<ScalarType> output(in.size());

    std::vector<ScalarType> res(in.size());

    viennacl::fast_copy(in, input);

    viennacl::detail::fft::bluestein(input, output, batch_size);

    viennacl::backend::finish();
    viennacl::fast_copy(output, res);

    return diff_max(res, out);
}

ScalarType opencl_radix2(std::vector<ScalarType>& in,
                         std::vector<ScalarType>& out,
                         unsigned int /*row*/, unsigned int /*col*/, unsigned int batch_num)
{
    viennacl::vector<ScalarType> input(in.size());
    viennacl::vector<ScalarType> output(in.size());

    std::vector<ScalarType> res(in.size());

    viennacl::fast_copy(in, input);

    unsigned int size = (input.size() >> 1) / batch_num;

    viennacl::detail::fft::radix2<ScalarType>(input.handle().opencl_handle(), size, size, batch_num);

    viennacl::backend::finish();
    viennacl::fast_copy(input, res);

    return diff_max(res, out);
}

ScalarType opencl_convolve(std::vector<ScalarType>& in1,
                           std::vector<ScalarType>& in2,
                           unsigned int /*row*/, unsigned int /*col*/, unsigned int /*batch_size*/)
{
    //if(in1.size() > 2048) return -1;
    viennacl::vector<ScalarType> input1(in1.size());
    viennacl::vector<ScalarType> input2(in2.size());
    viennacl::vector<ScalarType> output(in1.size());

    viennacl::fast_copy(in1, input1);
    viennacl::fast_copy(in2, input2);

    viennacl::linalg::convolve(input1, input2, output);

    viennacl::backend::finish();
    std::vector<ScalarType> res(in1.size());
    viennacl::fast_copy(output, res);

    std::vector<ScalarType> ref(in1.size());
    convolve_ref(in1, in2, ref);

    return diff_max(res, ref);
}

int test_correctness(const std::string& log_tag,
                                const std::string& filename,
                                input_function_ptr input_function,
                                test_function_ptr func) {

    std::vector<ScalarType> input;
    std::vector<ScalarType> output;

    std::fstream fstr;

    fstr.open(filename.c_str());

    std::cout << "*****************" << log_tag << "***************************\n";

    unsigned int test_size = 0;

    fstr >> test_size;
    
    std::cout << "Test size: " << test_size << std::endl;

    for(unsigned int i = 0; i < test_size; i++) {
        unsigned int batch_size;
        unsigned int rows_num, cols_num;
        input_function(fstr, input, output, rows_num, cols_num, batch_size);
        ScalarType df = func(input, output, rows_num, cols_num, batch_size);
        printf("%7s NX=%6d NY=%6d; BATCH=%3d; DIFF=%3.15f;\n", ((fabs(df) < EPS)?"[Ok]":"[Fail]"), rows_num, cols_num, batch_size, df);
        if (df > EPS)
          return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}



int main() 
{
  std::cout << "*" << std::endl;
  std::cout << "* ViennaCL test: FFT" << std::endl;
  std::cout << "*" << std::endl;

  //1D FFT tests
  if (test_correctness("fft::direct", "../non-release/testdata/cufft.data", read_vectors_pair, &opencl_direct) == EXIT_FAILURE)
    return EXIT_FAILURE;
  if (test_correctness("fft::fft", "../non-release/testdata/cufft.data", read_vectors_pair, &opencl_fft) == EXIT_FAILURE)
    return EXIT_FAILURE;
  if (test_correctness("fft::batch::direct", "../non-release/testdata/batch_radix.data", read_vectors_pair, &opencl_direct) == EXIT_FAILURE)
    return EXIT_FAILURE;
  if (test_correctness("fft::radix2", "../non-release/testdata/radix2.data", read_vectors_pair, &opencl_radix2) == EXIT_FAILURE)
    return EXIT_FAILURE;
  if (test_correctness("fft::batch::radix2", "../non-release/testdata/batch_radix.data", read_vectors_pair, &opencl_radix2) == EXIT_FAILURE)
    return EXIT_FAILURE;
  if (test_correctness("fft::batch::fft", "../non-release/testdata/batch_radix.data", read_vectors_pair, &opencl_fft) == EXIT_FAILURE)
    return EXIT_FAILURE;
  if (test_correctness("fft::convolve::1", "../non-release/testdata/cufft.data", read_vectors_pair, &opencl_convolve) == EXIT_FAILURE)
    return EXIT_FAILURE;
  if (test_correctness("fft::convolve::2", "../non-release/testdata/radix2.data", read_vectors_pair, &opencl_convolve) == EXIT_FAILURE)
    return EXIT_FAILURE;
  if (test_correctness("fft::bluestein::1", "../non-release/testdata/cufft.data", read_vectors_pair, &opencl_bluestein) == EXIT_FAILURE)
    return EXIT_FAILURE;
  if (test_correctness("fft::bluestein::2", "../non-release/testdata/radix2.data", read_vectors_pair, &opencl_bluestein) == EXIT_FAILURE)
    return EXIT_FAILURE;

  //2D FFT tests
  if (test_correctness("fft:2d::radix2::sml::1_arg", 
                        "../non-release/testdata/fft2d_radix2.data", read_matrices_pair, &opencl_2d_fft_1arg) == EXIT_FAILURE)
    return EXIT_FAILURE;
  if (test_correctness("fft:2d::direct::sml::1_arg",
                        "../non-release/testdata/fft2d_direct.data", read_matrices_pair, &opencl_2d_fft_1arg) == EXIT_FAILURE)
    return EXIT_FAILURE;
  if (test_correctness("fft:2d::direct::big::1_arg",
                        "../non-release/testdata/fft2d_direct_big.data", read_matrices_pair, &opencl_2d_fft_1arg) == EXIT_FAILURE)
    return EXIT_FAILURE;

  if (test_correctness("fft:2d::radix2::sml::2_arg", 
                        "../non-release/testdata/fft2d_radix2.data", read_matrices_pair, &opencl_2d_fft_2arg) == EXIT_FAILURE)
    return EXIT_FAILURE;
  if (test_correctness("fft:2d::direct::sml::2_arg",
                        "../non-release/testdata/fft2d_direct.data", read_matrices_pair, &opencl_2d_fft_2arg) == EXIT_FAILURE)
    return EXIT_FAILURE;
  if (test_correctness("fft:2d::direct::bscalarig::2_arg", 
                        "../non-release/testdata/fft2d_direct_big.data", read_matrices_pair, &opencl_2d_fft_2arg) == EXIT_FAILURE)
    return EXIT_FAILURE;

  std::cout << std::endl;
  std::cout << "------- Test completed --------" << std::endl;
  std::cout << std::endl;
   

  return EXIT_SUCCESS;
}
