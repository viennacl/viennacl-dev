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

#include <iostream>
#include <vector>
#include <cmath>
#include <complex>
#include <algorithm>

//#define VIENNACL_BUILD_INFO
#include "viennacl/linalg/host_based/fft_operations.hpp"

#ifdef VIENNACL_WITH_OPENCL
#include "viennacl/linalg/opencl/fft_operations.hpp"
#include "viennacl/linalg/opencl/kernels/fft.hpp"
#endif

#ifdef VIENNACL_WITH_CUDA
#include "viennacl/linalg/cuda/fft_operations.hpp"
#endif
#include "viennacl/linalg/fft_operations.hpp"
#include "viennacl/fft.hpp"

typedef float ScalarType;

const ScalarType EPS = ScalarType(0.06f); //use smaller values in double precision

typedef ScalarType (*test_function_ptr)(std::vector<ScalarType>&, std::vector<ScalarType>&,
    unsigned int, unsigned int, unsigned int);
typedef void (*input_function_ptr)(std::vector<ScalarType>&, std::vector<ScalarType>&,
    unsigned int&, unsigned int&, unsigned int&, const std::string&);

struct testData
{
  float input[2048];
  float output[2048];
  unsigned int batch_num;
  unsigned int row_num;
  unsigned int col_num;
};

testData direct_2d = { { 0.120294, 0.839315, 0.890936, 0.775417, 0.375051, 0.775645, 0.367671, 0.309852, 0.551154, 0.166495, 0.174865, 0.340252, 0.393914, 0.439817, 0.523974, 0.291109, 0.181803,
    0.811176, 0.490668, 0.234881, 0.611783, 0.098058, 0.106492, 0.399059, 0.974164, 0.403960, 0.324111, 0.772581, 0.609412, 0.917312, 0.538254, 0.729706, 0.756627, 0.429191, 0.505123, 0.131678,
    0.204836, 0.872794, 0.441530, 0.755990, 0.039289, 0.616395, 0.096242, 0.433203, 0.056212, 0.620216, 0.724312, 0.238015 }, { 10.058718, 12.402115, 3.306907, 0.570050, -0.527832, -1.052828,
    -0.309640, 1.578631, 0.027247, 1.441292, -2.396150, 0.396048, -2.490234, -0.923666, -0.890061, 1.154475, -2.485666, -0.029132, -1.617884, -0.788678, 0.008640, -0.751211, -0.245883, 2.815872,
    2.316608, 0.780692, 0.437285, -0.798080, 0.304596, -0.176831, 1.481121, -0.633767, -0.177035, 0.302556, -1.388328, 0.109418, 0.034794, 0.568763, 0.053167, -0.332043, 0.074045, -1.350742,
    -1.101494, 1.267548, -1.288304, 2.578995, -0.297569, 1.014074 }, 1, 4, 6 };

testData radix2_2d = { { 0.860600, 0.020071, 0.756794, 0.472348, 0.604630, 0.445387, 0.738811, 0.644715, 0.840903, 0.746019, 0.629334, 0.682880, 0.516268, 0.235386, 0.800333, 0.175785, 0.974124,
    0.485907, 0.492256, 0.696148, 0.230253, 0.600575, 0.138786, 0.136737, 0.114667, 0.516912, 0.173743, 0.899410, 0.891824, 0.704459, 0.450209, 0.752424, 0.724530, 0.207003, 0.224772, 0.329161,
    0.652390, 0.963583, 0.973876, 0.493293, 0.709602, 0.603211, 0.176173, 0.225870, 0.838596, 0.976507, 0.401655, 0.812721, 0.462413, 0.893911, 0.508869, 0.692667, 0.494486, 0.647656, 0.829403,
    0.609152, 0.164568, 0.003146, 0.508563, 0.056392, 0.707605, 0.958771, 0.808816, 0.432136 }, { 18.399853, 17.120342, 1.194352, 0.639568, -0.086731, -0.384759, 1.241270, -2.175158, 1.175068,
    0.896665, 0.753659, 0.780709, -0.082556, -3.727531, 1.578434, -0.294704, 1.544822, -0.169894, 0.570453, -1.065756, 1.432534, -1.146827, -1.713843, 2.376111, -2.141517, -3.200578, -1.061705,
    -1.680550, 0.656694, 2.493567, -1.462913, -3.195214, 2.498683, -1.052464, -1.144435, -4.022502, 0.301723, 0.550845, -1.033154, -0.872973, 0.916475, -0.175878, 0.123236, -1.495021, 1.962570,
    -0.616791, -2.436357, -1.537166, 0.547337, -2.207615, 1.563801, -0.916862, 2.013805, 1.934075, 0.940849, -0.143010, -0.361511, 0.364330, -0.161776, 1.245928, -1.553198, 1.579960, 1.363282,
    0.741429 }, 1, 4, 8 };

testData direct_2d_big = { { 0.475679, 0.408864, 0.313085, 0.387599, 0.767833, 0.015767, 0.832733, 0.764867, 0.850312, 0.782744, 0.355199, 0.308463, 0.496935, 0.043339, 0.309902, 0.030681, 0.497275,
    0.237185, 0.229802, 0.606489, 0.720393, 0.848826, 0.704500, 0.845834, 0.451885, 0.339276, 0.523190, 0.688469, 0.646792, 0.975192, 0.933888, 0.122471, 0.384056, 0.246973, 0.510070, 0.151889,
    0.262739, 0.342803, 0.916756, 0.113051, 0.125547, 0.271954, 0.421514, 0.622482, 0.315293, 0.731416, 0.653164, 0.812568, 0.968601, 0.882965, 0.419057, 0.688994, 0.731792, 0.123557, 0.534827,
    0.183676, 0.462833, 0.058017, 0.872145, 0.109626, 0.033209, 0.806033, 0.232097, 0.417265, 0.053006, 0.742167, 0.569154, 0.315745, 0.084970, 0.485910, 0.428796, 0.210517, 0.757864, 0.850311,
    0.832999, 0.073158, 0.581726, 0.486163, 0.885726, 0.550328, 0.369128, 0.304783, 0.239321, 0.100920 }, { 21.755795, 18.089336, -1.248233, -0.179035, 1.307578, 1.589876, -1.680055, 1.879153,
    0.500297, 0.839735, 0.046095, -0.177522, 0.742587, -0.786261, -3.427422, -0.445572, -1.376776, 1.221333, 0.334313, -0.588123, -2.070653, 1.297694, -1.879930, -2.445690, 1.692045, 0.251480,
    0.435994, 0.257269, 1.513737, 0.859310, 0.538316, -3.698363, -3.243739, 2.342074, 1.255018, -1.052454, 0.450322, 3.684811, -0.951320, 2.863686, -0.170055, 1.501932, -0.800708, 2.040001,
    -0.229112, -0.175461, -5.128507, -2.872447, -2.125049, -2.656515, 0.632609, -2.080163, 2.527745, -1.830541, 0.086613, -1.402300, -0.900261, -1.355287, -0.909127, 2.822799, 2.142723, -0.882929,
    -3.627774, 0.180693, -0.073456, 0.783774, 2.144351, -0.252458, 0.090970, -0.007880, 3.457415, 0.527979, 0.505462, 0.978198, -1.807562, -2.692160, 2.556900, -1.385276, 3.526823, 0.247212,
    1.879590, 0.288942, 1.504963, -0.408566 }, 1, 7, 6 };

testData transposeMatrix= {{0.139420,0.539278,0.547922,0.672097,0.528360,0.158671,0.596258,0.432662,0.445432,0.597279,0.966011,0.707923,0.705743,0.282214,0.100677,0.143657,0.040120,0.346660,0.279002,
    0.568480,0.505332,0.875261,0.001142,0.237294,0.673498,0.699611,0.990521,0.379241,0.981826,0.091198,0.522898,0.637506}, {0.13942,0.539278,0.445432,0.597279,0.04012,0.34666,0.673498,0.699611,
    0.547922,0.672097,0.966011,0.707923,0.279002,0.56848,0.990521,0.379241,0.52836,0.158671,0.705743,0.282214,0.505332,0.875261,0.981826,0.091198,0.596258,0.432662,0.100677,0.143657,0.001142,
    0.237294,0.522898,0.637506},1,4,4};

void set_values_struct(std::vector<ScalarType>& input, std::vector<ScalarType>& output,
    unsigned int& rows, unsigned int& cols, unsigned int& batch_size, testData& data)
{
  unsigned int size = data.col_num * data.batch_num * 2 * data.row_num;
  input.resize(size);
  output.resize(size);
  rows = data.row_num;
  cols = data.col_num;
  batch_size = data.batch_num;
  for (unsigned int i = 0; i < size; i++)
  {
    input[i] = data.input[i];
    output[i] = data.output[i];
  }

}

void read_matrices_pair(std::vector<ScalarType>& input, std::vector<ScalarType>& output,
    unsigned int& rows, unsigned int& cols, unsigned int& batch_size, const std::string& log_tag)
{
  if (log_tag == "fft:2d::direct::1_arg")
    set_values_struct(input, output, rows, cols, batch_size, direct_2d);
  if (log_tag == "fft:2d::radix2::1_arg")
    set_values_struct(input, output, rows, cols, batch_size, radix2_2d);
  if (log_tag == "fft:2d::direct::big::2_arg")
    set_values_struct(input, output, rows, cols, batch_size, direct_2d_big);
  if (log_tag == "fft::transpose" || log_tag == "fft::transpose_inplace")
      set_values_struct(input, output, rows, cols, batch_size, transposeMatrix);

}

template<typename ScalarType>
ScalarType diff(std::vector<ScalarType>& vec, std::vector<ScalarType>& ref)
{
  ScalarType df = 0.0;
  ScalarType norm_ref = 0;

  for (std::size_t i = 0; i < vec.size(); i++)
  {
    df = df + pow(vec[i] - ref[i], 2);
    norm_ref += ref[i] * ref[i];
  }

  return sqrt(df / norm_ref);
}

template<typename ScalarType>
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

void copy_vector_to_matrix(viennacl::matrix<ScalarType> & input, std::vector<ScalarType> & in,
    unsigned int row, unsigned int col)
{
  std::vector<std::vector<ScalarType> > my_matrix(row, std::vector<ScalarType>(col * 2));
  for (unsigned int i = 0; i < row; i++)
    for (unsigned int j = 0; j < col * 2; j++)
      my_matrix[i][j] = in[i * col * 2 + j];
  viennacl::copy(my_matrix, input);

}

void copy_matrix_to_vector(viennacl::matrix<ScalarType> & input, std::vector<ScalarType> & in,
    unsigned int row, unsigned int col)
{
  std::vector<std::vector<ScalarType> > my_matrix(row, std::vector<ScalarType>(col * 2));
  viennacl::copy(input, my_matrix);
  for (unsigned int i = 0; i < row; i++)
    for (unsigned int j = 0; j < col * 2; j++)
      in[i * col * 2 + j] = my_matrix[i][j];
}

ScalarType fft_2d_1arg(std::vector<ScalarType>& in, std::vector<ScalarType>& out, unsigned int row,
    unsigned int col, unsigned int /*batch_size*/)
{
  viennacl::matrix<ScalarType> input(row, 2 * col);

  std::vector<ScalarType> res(in.size());

  copy_vector_to_matrix(input, in, row, col);

  viennacl::inplace_fft(input);
  //std::cout << input << "\n";
  viennacl::backend::finish();

  copy_matrix_to_vector(input, res, row, col);

  return diff_max(res, out);
}

ScalarType transpose_inplace(std::vector<ScalarType>& in, std::vector<ScalarType>& out, unsigned int row,
    unsigned int col, unsigned int /*batch_size*/)
{
  viennacl::matrix<ScalarType> input(row, 2 * col);

  std::vector<ScalarType> res(in.size());

  copy_vector_to_matrix(input, in, row, col);

  viennacl::linalg::transpose(input);

  viennacl::backend::finish();

  copy_matrix_to_vector(input, res, row, col);

  return diff_max(res, out);
}

ScalarType transpose(std::vector<ScalarType>& in, std::vector<ScalarType>& out, unsigned int row,
    unsigned int col, unsigned int /*batch_size*/)
{
  viennacl::matrix<ScalarType> input(row, 2 * col);
  viennacl::matrix<ScalarType> output(row, 2 * col);


  std::vector<ScalarType> res(in.size());

  copy_vector_to_matrix(input, in, row, col);

  viennacl::linalg::transpose(input,output);

  viennacl::backend::finish();

  copy_matrix_to_vector(output, res, row, col);

  return diff_max(res, out);
}

ScalarType fft_2d_2arg(std::vector<ScalarType>& in, std::vector<ScalarType>& out, unsigned int row,
    unsigned int col, unsigned int /*batch_size*/)
{
  viennacl::matrix<ScalarType> input(row, 2 * col);
  viennacl::matrix<ScalarType> output(row, 2 * col);

  std::vector<ScalarType> res(in.size());

  copy_vector_to_matrix(input, in, row, col);

  //std::cout << input << "\n";
  viennacl::fft(input, output);
  //std::cout << input << "\n";
  viennacl::backend::finish();

  copy_matrix_to_vector(output, res, row, col);

  return diff_max(res, out);
}

int test_correctness(const std::string& log_tag, input_function_ptr input_function,
    test_function_ptr func)
{

  std::vector<ScalarType> input;
  std::vector<ScalarType> output;

  std::cout << std::endl;
  std::cout << "*****************" << log_tag << "***************************\n";

  unsigned int batch_size;
  unsigned int rows_num, cols_num;

  input_function(input, output, rows_num, cols_num, batch_size, log_tag);
  ScalarType df = func(input, output, rows_num, cols_num, batch_size);
  printf("%7s ROWS=%6d COLS=%6d; BATCH=%3d; DIFF=%3.15f;\n", ((fabs(df) < EPS) ? "[Ok]" : "[Fail]"),
      rows_num, cols_num, batch_size, df);
  std::cout << std::endl;

  if (df > EPS)
    return EXIT_FAILURE;

  return EXIT_SUCCESS;
}

int main()
{
  std::cout << "*" << std::endl;
  std::cout << "* ViennaCL test: FFT" << std::endl;
  std::cout << "*" << std::endl;

  //2D FFT tests
  if (test_correctness("fft:2d::radix2::1_arg", read_matrices_pair, &fft_2d_1arg) == EXIT_FAILURE)
    return EXIT_FAILURE;
  if (test_correctness("fft:2d::direct::1_arg", read_matrices_pair, &fft_2d_1arg) == EXIT_FAILURE)
    return EXIT_FAILURE;
  if (test_correctness("fft:2d::direct::big::2_arg", read_matrices_pair,
      &fft_2d_2arg) == EXIT_FAILURE)
    return EXIT_FAILURE;
  if (test_correctness("fft::transpose_inplace", read_matrices_pair, &transpose_inplace) == EXIT_FAILURE)
    return EXIT_FAILURE;
  if (test_correctness("fft::transpose", read_matrices_pair, &transpose) == EXIT_FAILURE)
      return EXIT_FAILURE;

  std::cout << std::endl;
  std::cout << "------- Test completed --------" << std::endl;
  std::cout << std::endl;

  return EXIT_SUCCESS;
}

