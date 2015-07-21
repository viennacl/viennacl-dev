/* =========================================================================
   Copyright (c) 2010-2015, Institute for Microelectronics,
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

/** \file tests/src/fft_2d.cpp  Tests the two-dimensional FFT routines.
*   \test Tests the two-dimensional FFT routines.
**/

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

static testData direct_2d = { { 0.120294f, 0.839315f, 0.890936f, 0.775417f, 0.375051f, 0.775645f, 0.367671f, 0.309852f, 0.551154f, 0.166495f, 0.174865f, 0.340252f, 0.393914f, 0.439817f, 0.523974f, 0.291109f, 0.181803f,
    0.811176f, 0.490668f, 0.234881f, 0.611783f, 0.098058f, 0.106492f, 0.399059f, 0.974164f, 0.403960f, 0.324111f, 0.772581f, 0.609412f, 0.917312f, 0.538254f, 0.729706f, 0.756627f, 0.429191f, 0.505123f, 0.131678f,
    0.204836f, 0.872794f, 0.441530f, 0.755990f, 0.039289f, 0.616395f, 0.096242f, 0.433203f, 0.056212f, 0.620216f, 0.724312f, 0.238015f }, { 10.058718f, 12.402115f, 3.306907f, 0.570050f, -0.527832f, -1.052828f,
    -0.309640f, 1.578631f, 0.027247f, 1.441292f, -2.396150f, 0.396048f, -2.490234f, -0.923666f, -0.890061f, 1.154475f, -2.485666f, -0.029132f, -1.617884f, -0.788678f, 0.008640f, -0.751211f, -0.245883f, 2.815872f,
    2.316608f, 0.780692f, 0.437285f, -0.798080f, 0.304596f, -0.176831f, 1.481121f, -0.633767f, -0.177035f, 0.302556f, -1.388328f, 0.109418f, 0.034794f, 0.568763f, 0.053167f, -0.332043f, 0.074045f, -1.350742f,
    -1.101494f, 1.267548f, -1.288304f, 2.578995f, -0.297569f, 1.014074f }, 1, 4, 6 };

static testData radix2_2d = { { 0.860600f, 0.020071f, 0.756794f, 0.472348f, 0.604630f, 0.445387f, 0.738811f, 0.644715f, 0.840903f, 0.746019f, 0.629334f, 0.682880f, 0.516268f, 0.235386f, 0.800333f, 0.175785f, 0.974124f,
    0.485907f, 0.492256f, 0.696148f, 0.230253f, 0.600575f, 0.138786f, 0.136737f, 0.114667f, 0.516912f, 0.173743f, 0.899410f, 0.891824f, 0.704459f, 0.450209f, 0.752424f, 0.724530f, 0.207003f, 0.224772f, 0.329161f,
    0.652390f, 0.963583f, 0.973876f, 0.493293f, 0.709602f, 0.603211f, 0.176173f, 0.225870f, 0.838596f, 0.976507f, 0.401655f, 0.812721f, 0.462413f, 0.893911f, 0.508869f, 0.692667f, 0.494486f, 0.647656f, 0.829403f,
    0.609152f, 0.164568f, 0.003146f, 0.508563f, 0.056392f, 0.707605f, 0.958771f, 0.808816f, 0.432136f }, { 18.399853f, 17.120342f, 1.194352f, 0.639568f, -0.086731f, -0.384759f, 1.241270f, -2.175158f, 1.175068f,
    0.896665f, 0.753659f, 0.780709f, -0.082556f, -3.727531f, 1.578434f, -0.294704f, 1.544822f, -0.169894f, 0.570453f, -1.065756f, 1.432534f, -1.146827f, -1.713843f, 2.376111f, -2.141517f, -3.200578f, -1.061705f,
    -1.680550f, 0.656694f, 2.493567f, -1.462913f, -3.195214f, 2.498683f, -1.052464f, -1.144435f, -4.022502f, 0.301723f, 0.550845f, -1.033154f, -0.872973f, 0.916475f, -0.175878f, 0.123236f, -1.495021f, 1.962570f,
    -0.616791f, -2.436357f, -1.537166f, 0.547337f, -2.207615f, 1.563801f, -0.916862f, 2.013805f, 1.934075f, 0.940849f, -0.143010f, -0.361511f, 0.364330f, -0.161776f, 1.245928f, -1.553198f, 1.579960f, 1.363282f,
    0.741429f }, 1, 4, 8 };

static testData direct_2d_big = { { 0.475679f, 0.408864f, 0.313085f, 0.387599f, 0.767833f, 0.015767f, 0.832733f, 0.764867f, 0.850312f, 0.782744f, 0.355199f, 0.308463f, 0.496935f, 0.043339f, 0.309902f, 0.030681f, 0.497275f,
    0.237185f, 0.229802f, 0.606489f, 0.720393f, 0.848826f, 0.704500f, 0.845834f, 0.451885f, 0.339276f, 0.523190f, 0.688469f, 0.646792f, 0.975192f, 0.933888f, 0.122471f, 0.384056f, 0.246973f, 0.510070f, 0.151889f,
    0.262739f, 0.342803f, 0.916756f, 0.113051f, 0.125547f, 0.271954f, 0.421514f, 0.622482f, 0.315293f, 0.731416f, 0.653164f, 0.812568f, 0.968601f, 0.882965f, 0.419057f, 0.688994f, 0.731792f, 0.123557f, 0.534827f,
    0.183676f, 0.462833f, 0.058017f, 0.872145f, 0.109626f, 0.033209f, 0.806033f, 0.232097f, 0.417265f, 0.053006f, 0.742167f, 0.569154f, 0.315745f, 0.084970f, 0.485910f, 0.428796f, 0.210517f, 0.757864f, 0.850311f,
    0.832999f, 0.073158f, 0.581726f, 0.486163f, 0.885726f, 0.550328f, 0.369128f, 0.304783f, 0.239321f, 0.100920f }, { 21.755795f, 18.089336f, -1.248233f, -0.179035f, 1.307578f, 1.589876f, -1.680055f, 1.879153f,
    0.500297f, 0.839735f, 0.046095f, -0.177522f, 0.742587f, -0.786261f, -3.427422f, -0.445572f, -1.376776f, 1.221333f, 0.334313f, -0.588123f, -2.070653f, 1.297694f, -1.879930f, -2.445690f, 1.692045f, 0.251480f,
    0.435994f, 0.257269f, 1.513737f, 0.859310f, 0.538316f, -3.698363f, -3.243739f, 2.342074f, 1.255018f, -1.052454f, 0.450322f, 3.684811f, -0.951320f, 2.863686f, -0.170055f, 1.501932f, -0.800708f, 2.040001f,
    -0.229112f, -0.175461f, -5.128507f, -2.872447f, -2.125049f, -2.656515f, 0.632609f, -2.080163f, 2.527745f, -1.830541f, 0.086613f, -1.402300f, -0.900261f, -1.355287f, -0.909127f, 2.822799f, 2.142723f, -0.882929f,
    -3.627774f, 0.180693f, -0.073456f, 0.783774f, 2.144351f, -0.252458f, 0.090970f, -0.007880f, 3.457415f, 0.527979f, 0.505462f, 0.978198f, -1.807562f, -2.692160f, 2.556900f, -1.385276f, 3.526823f, 0.247212f,
    1.879590f, 0.288942f, 1.504963f, -0.408566f }, 1, 7, 6 };

static testData transposeMatrix= {{0.139420f,0.539278f,0.547922f,0.672097f,0.528360f,0.158671f,0.596258f,0.432662f,0.445432f,0.597279f,0.966011f,0.707923f,0.705743f,0.282214f,0.100677f,0.143657f,0.040120f,0.346660f,0.279002f,
    0.568480f,0.505332f,0.875261f,0.001142f,0.237294f,0.673498f,0.699611f,0.990521f,0.379241f,0.981826f,0.091198f,0.522898f,0.637506f}, {0.13942f,0.539278f,0.445432f,0.597279f,0.04012f,0.34666f,0.673498f,0.699611f,
    0.547922f,0.672097f,0.966011f,0.707923f,0.279002f,0.56848f,0.990521f,0.379241f,0.52836f,0.158671f,0.705743f,0.282214f,0.505332f,0.875261f,0.981826f,0.091198f,0.596258f,0.432662f,0.100677f,0.143657f,0.001142f,
    0.237294f,0.522898f,0.637506f},1,4,4};

void set_values_struct(std::vector<ScalarType>& input, std::vector<ScalarType>& output,
    unsigned int& rows, unsigned int& cols, unsigned int& batch_size, testData& data);

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
    unsigned int& rows, unsigned int& cols, unsigned int& batch_size, const std::string& log_tag);

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
    df = std::max<ScalarType>(std::fabs(vec[i] - ref[i]), df);
    mx = std::max<ScalarType>(std::fabs(vec[i]), mx);

    if (mx > 0)
    {
      if (norm_max < df / mx)
        norm_max = df / mx;
    }
  }

  return norm_max;
}


void copy_vector_to_matrix(viennacl::matrix<ScalarType> & input, std::vector<ScalarType> & in,
    unsigned int row, unsigned int col);

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
    unsigned int row, unsigned int col);

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
    unsigned int col, unsigned int /*batch_size*/);

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
    unsigned int col, unsigned int /*batch_size*/);

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
    unsigned int col, unsigned int /*batch_size*/);

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
    unsigned int col, unsigned int /*batch_size*/);

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
    test_function_ptr func);

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

