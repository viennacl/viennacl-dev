/*
* Generates BLAS level 3 routines
*/

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>

#include <iostream>
#include <stdlib.h>

//generate code for C = op1(A) * op2(B), where A, B, C can have different storage layouts and opX(D) = D or trans(D)
void printMatrixMatrixProduct(bool row_major_A, bool row_major_B, bool row_major_C,
                              bool transpose_A, bool transpose_B, bool write_cuda)
{
  //write header:
  if (!write_cuda)
    std::cout << "// file automatically generated - do not edit!" << std::endl;
  std::cout << "// matrix-matrix multiplication C = ";
  if (transpose_A)
    std::cout << "A^T * ";
  else
    std::cout << "A * ";
  if (transpose_B)
    std::cout << "B^T" << std::endl;
  else
    std::cout << "B" << std::endl;
  std::cout << "// matrix layouts: ";
  if (row_major_C)
    std::cout << "C...row_major, ";
  else
    std::cout << "C...col_major, ";
  if (row_major_A)
    std::cout << "A...row_major, ";
  else
    std::cout << "A...col_major, ";
  if (row_major_B)
    std::cout << "B...row_major" << std::endl;
  else
    std::cout << "B...col_major" << std::endl;

  if (write_cuda)
    std::cout << "template <typename T>" << std::endl;

  //start OpenCL code:
  if (write_cuda)
    std::cout << "__global__ void prod_";
  else
    std::cout << "__kernel void prod_";
  if (transpose_A)
    std::cout << "T";
  else
    std::cout << "A";
  if (transpose_B)
    std::cout << "T";
  else
    std::cout << "A";

  std::cout << "(" << std::endl;
  if (write_cuda)
  {
    std::cout << "          T alpha," << std::endl;
    std::cout << "          const T * A," << std::endl;
  }
  else
  {
    std::cout << "          float alpha," << std::endl;
    std::cout << "          __global const float * A," << std::endl;
  }
  std::cout << "          unsigned int A_row_start," << std::endl;
  std::cout << "          unsigned int A_col_start," << std::endl;
  std::cout << "          unsigned int A_row_inc," << std::endl;
  std::cout << "          unsigned int A_col_inc," << std::endl;
  std::cout << "          unsigned int A_row_size," << std::endl;   //number of elements starting from row_start!
  std::cout << "          unsigned int A_col_size," << std::endl;
  std::cout << "          unsigned int A_internal_rows," << std::endl;
  std::cout << "          unsigned int A_internal_cols," << std::endl;
  if (write_cuda)
    std::cout << "          const T * B,  " << std::endl;
  else
    std::cout << "          __global const float * B,  " << std::endl;
  std::cout << "          unsigned int B_row_start," << std::endl;
  std::cout << "          unsigned int B_col_start," << std::endl;
  std::cout << "          unsigned int B_row_inc," << std::endl;
  std::cout << "          unsigned int B_col_inc," << std::endl;
  std::cout << "          unsigned int B_row_size," << std::endl;
  std::cout << "          unsigned int B_col_size," << std::endl;
  std::cout << "          unsigned int B_internal_rows," << std::endl;
  std::cout << "          unsigned int B_internal_cols," << std::endl;
  if (write_cuda)
    std::cout << "          T beta," << std::endl;
  else
    std::cout << "          float beta," << std::endl;
  if (write_cuda)
    std::cout << "          T * C," << std::endl;
  else
    std::cout << "          __global float * C," << std::endl;
  std::cout << "          unsigned int C_row_start," << std::endl;
  std::cout << "          unsigned int C_col_start," << std::endl;
  std::cout << "          unsigned int C_row_inc," << std::endl;
  std::cout << "          unsigned int C_col_inc," << std::endl;
  std::cout << "          unsigned int C_row_size," << std::endl;
  std::cout << "          unsigned int C_col_size," << std::endl;
  std::cout << "          unsigned int C_internal_rows," << std::endl;
  std::cout << "          unsigned int C_internal_cols) " << std::endl;
  std::cout << "{ " << std::endl;
  std::cout << std::endl;
  if (write_cuda)
  {
    std::cout << "  __shared__ T bufA[" << 16 * 17 << "];" << std::endl;
    std::cout << "  __shared__ T bufB[" << 16 * 17 << "];" << std::endl;
  }
  else
  {
    std::cout << "  __local float bufA[" << 16 * 17 << "];" << std::endl;
    std::cout << "  __local float bufB[" << 16 * 17 << "];" << std::endl;
  }
  std::cout << std::endl;
  //do not forgot to change block_size !!!
  std::cout << "  size_t block_size = 16;//get_local_size(0);" << std::endl;
  if (write_cuda)
  {
    std::cout << "  size_t row_block_id = blockIdx.x;" << std::endl;
    std::cout << "  size_t col_block_id = blockIdx.y;" << std::endl;
    std::cout << "  size_t row_thread_id = threadIdx.x;" << std::endl;
    std::cout << "  size_t col_thread_id = threadIdx.y;" << std::endl;
  }
  else
  {
    std::cout << "  size_t row_block_id = get_group_id(0);" << std::endl;
    std::cout << "  size_t col_block_id = get_group_id(1);" << std::endl;
    std::cout << "  size_t row_thread_id = get_local_id(0);" << std::endl;
    std::cout << "  size_t col_thread_id = get_local_id(1);" << std::endl;
  }

  //traverse block row of A (taking mem layout and transpose operation into account)
  if (row_major_A && transpose_A)
  {
    std::cout << "  size_t aBegin = (row_block_id * block_size * A_col_inc + A_col_start) + A_row_start * A_internal_cols;" << std::endl;
    std::cout << "  size_t aStep = block_size * A_row_inc * A_internal_cols;" << std::endl;
  }
  else if (row_major_A && !transpose_A)
  {
    std::cout << "  size_t aBegin = (row_block_id * block_size * A_row_inc + A_row_start) * A_internal_cols + A_col_start;" << std::endl;
    std::cout << "  size_t aStep = block_size * A_col_inc;" << std::endl;
  }
  else if (!row_major_A && transpose_A)
  {
    std::cout << "  size_t aBegin = (row_block_id * block_size * A_col_inc + A_col_start) * A_internal_rows + A_row_start;" << std::endl;
    std::cout << "  size_t aStep = block_size * A_row_inc;" << std::endl;
  }
  else if (!row_major_A && !transpose_A)
  {
    std::cout << "  size_t aBegin = (row_block_id * block_size * A_row_inc + A_row_start) + A_col_start * A_internal_rows;" << std::endl;
    std::cout << "  size_t aStep = block_size * A_col_inc * A_internal_rows;" << std::endl;
  }


  if (row_major_B && transpose_B)
  {
    std::cout << "  size_t bBegin = (col_block_id * block_size * B_row_inc + B_row_start) * B_internal_cols + B_col_start;" << std::endl;
    std::cout << "  size_t bStep = block_size * B_col_inc;" << std::endl;
  }
  else if (row_major_B && !transpose_B)
  {
    std::cout << "  size_t bBegin = (col_block_id * block_size * B_col_inc + B_col_start) + B_row_start * B_internal_cols;" << std::endl;
    std::cout << "  size_t bStep = block_size * B_internal_cols * B_row_inc;" << std::endl;
  }
  else if (!row_major_B && transpose_B)
  {
    std::cout << "  size_t bBegin = (col_block_id * block_size * B_row_inc + B_row_start) + B_col_start * B_internal_rows;" << std::endl;
    std::cout << "  size_t bStep = block_size * B_internal_rows * B_col_inc;" << std::endl;
  }
  else if (!row_major_B && !transpose_B)
  {
    std::cout << "  size_t bBegin = (col_block_id * block_size * B_col_inc + B_col_start) * B_internal_rows + B_row_start;" << std::endl;
    std::cout << "  size_t bStep = block_size * B_row_inc;" << std::endl;
  }


  if (transpose_A)
    std::cout << "  size_t block_num = (A_row_size + block_size - 1) / block_size;" << std::endl;
  else
    std::cout << "  size_t block_num = (A_col_size + block_size - 1) / block_size;" << std::endl;

  if (write_cuda)
    std::cout << "  T Csub = 0;" << std::endl;
  else
    std::cout << "  float Csub = 0;" << std::endl;

  //offset of the the memory access by the thread relative to the beginning of the block:
  if (row_major_A)
    std::cout << "  size_t aOffset = row_thread_id * A_col_inc + col_thread_id * A_row_inc * A_internal_cols;" << std::endl;
  else
    std::cout << "  size_t aOffset = row_thread_id * A_row_inc + col_thread_id * A_col_inc * A_internal_rows;" << std::endl;

  if (row_major_B)
    std::cout << "  size_t bOffset = row_thread_id * B_col_inc + col_thread_id * B_row_inc * B_internal_cols;" << std::endl;
  else
    std::cout << "  size_t bOffset = row_thread_id * B_row_inc + col_thread_id * B_col_inc *  B_internal_rows;" << std::endl;

  std::cout << std::endl;

  std::cout << "  size_t row_thread_id_times_block_size = row_thread_id * (block_size + 1);" << std::endl;
  std::cout << "  size_t col_thread_id_times_block_size = col_thread_id * (block_size + 1);" << std::endl;

  std::cout << "  for (size_t block = 0;" << std::endl;
  std::cout << "           block < block_num;" << std::endl;
  std::cout << "           ++block)" << std::endl;
  std::cout << "  {" << std::endl;

  //read block from A and check for access within matrix:

  if (transpose_A && row_major_A)
    std::cout << "    bufA[row_thread_id_times_block_size + col_thread_id] = ((block * block_size + col_thread_id < A_row_size) && (row_block_id * block_size + row_thread_id < A_col_size)) ? A[aBegin + aOffset] : 0;" << std::endl;
  else if (transpose_A && !row_major_A)
    std::cout << "    bufA[col_thread_id_times_block_size + row_thread_id] = ((block * block_size + row_thread_id < A_row_size) && (row_block_id * block_size + col_thread_id < A_col_size)) ? A[aBegin + aOffset] : 0;" << std::endl;
  else if (!transpose_A && row_major_A)
    std::cout << "    bufA[col_thread_id_times_block_size + row_thread_id] = ((block * block_size + row_thread_id < A_col_size) && (row_block_id * block_size + col_thread_id < A_row_size)) ? A[aBegin + aOffset] : 0;" << std::endl;
  else if (!transpose_A && !row_major_A)
    std::cout << "    bufA[row_thread_id_times_block_size + col_thread_id] = ((block * block_size + col_thread_id < A_col_size) && (row_block_id * block_size + row_thread_id < A_row_size)) ? A[aBegin + aOffset] : 0;" << std::endl;


  if (transpose_B && row_major_B)
    std::cout << "    bufB[col_thread_id_times_block_size + row_thread_id] = ((block * block_size + row_thread_id < B_col_size) && (col_block_id * block_size + col_thread_id < B_row_size)) ? B[bBegin + bOffset] : 0;" << std::endl;
  else if (transpose_B && !row_major_B)
    std::cout << "    bufB[row_thread_id_times_block_size + col_thread_id] = ((block * block_size + col_thread_id < B_col_size) && (col_block_id * block_size + row_thread_id < B_row_size)) ? B[bBegin + bOffset] : 0;" << std::endl;
  else if (!transpose_B && row_major_B)
    std::cout << "    bufB[row_thread_id_times_block_size + col_thread_id] = ((block * block_size + col_thread_id < B_row_size) && (col_block_id * block_size + row_thread_id < B_col_size)) ? B[bBegin + bOffset] : 0;" << std::endl;
  else if (!transpose_B && !row_major_B)
    std::cout << "    bufB[col_thread_id_times_block_size + row_thread_id] = ((block * block_size + row_thread_id < B_row_size) && (col_block_id * block_size + col_thread_id < B_col_size)) ? B[bBegin + bOffset] : 0;" << std::endl;

  //computation of block-matrix-matrix product is the same for all cases:
  if (write_cuda)
    std::cout << "    __syncthreads();" << std::endl;
  else
    std::cout << "    barrier(CLK_LOCAL_MEM_FENCE);" << std::endl;
  //std::cout << "    for (size_t k = 0; k < block_size; ++k)" << std::endl;
  //std::cout << "      Csub += bufA[row_thread_id_times_block_size + k] * bufB[k * block_size + col_thread_id];" << std::endl;
  //loop unrolling:
  if (write_cuda)
  {
    std::cout << "    __shared__ T * bufAptr = bufA + row_thread_id_times_block_size;" << std::endl;
    std::cout << "    __shared__ T * bufBptr = bufB + col_thread_id_times_block_size;" << std::endl;
  }
  else
  {
    std::cout << "    __local float * bufAptr = bufA + row_thread_id_times_block_size;" << std::endl;
    std::cout << "    __local float * bufBptr = bufB + col_thread_id_times_block_size;" << std::endl;
  }
  //std::cout << "      Csub += bufA[row_thread_id_times_block_size] * bufB[col_thread_id * block_size];" << std::endl;
  // code in following line depends on block size and must be changed in case of block_size changes
  //std::cout << "      for(int i = 0; i < 4; i++) {" << std::endl;
  for (size_t unroll = 0; unroll < 16; ++unroll) {
    std::cout << "      Csub += (*bufAptr) * (*bufBptr); ++bufAptr; ++bufBptr;" << std::endl;
  }
  //std::cout << "     }" << std::endl;
    //std::cout << "      Csub += bufAptr[" << i << "] * bufB[" << i << "  + col_thread_id * block_size];" << std::endl;
    //std::cout << "      Csub += bufAptr[" << i << "] * bufB[" << i << " * block_size + col_thread_id];" << std::endl;
    //std::cout << "      Csub += bufAptr[" << i << "] * bufB[" << i << "];" << std::endl;
  if (write_cuda)
    std::cout << "    __syncthreads();" << std::endl;
  else
    std::cout << "    barrier(CLK_LOCAL_MEM_FENCE);" << std::endl;
  std::cout << "    aBegin += aStep;" << std::endl;
  std::cout << "    bBegin += bStep;" << std::endl;
  std::cout << "  }" << std::endl;


  if (transpose_A)
  {
    if (write_cuda)
      std::cout << "  if ((blockIdx.x * blockDim.x + threadIdx.x) < A_col_size && ";
    else
      std::cout << "  if (get_global_id(0) < A_col_size && ";
  }
  else
  {
    if (write_cuda)
      std::cout << "  if ((blockIdx.x * blockDim.x + threadIdx.x) < A_row_size && ";
    else
      std::cout << "  if (get_global_id(0) < A_row_size && ";
  }

  if (transpose_B)
  {
    if (write_cuda)
      std::cout << "(blockIdx.y * blockDim.y + threadIdx.y) < B_row_size)" << std::endl;
    else
      std::cout << "get_global_id(1) < B_row_size)" << std::endl;
  }
  else
  {
    if (write_cuda)
      std::cout << "(blockIdx.y * blockDim.y + threadIdx.y) < B_col_size)" << std::endl;
    else
      std::cout << "get_global_id(1) < B_col_size)" << std::endl;
  }

  if (row_major_C)
  {
    if (write_cuda)
      std::cout << "    C[((blockIdx.x * blockDim.x + threadIdx.x) * C_row_inc + C_row_start) * C_internal_cols + (blockIdx.y * blockDim.y + threadIdx.y) * C_col_inc + C_col_start] = (beta == 0) ? alpha * Csub : alpha * Csub + beta * C[((blockIdx.x * blockDim.x + threadIdx.x) * C_row_inc + C_row_start) * C_internal_cols + (blockIdx.y * blockDim.y + threadIdx.y) * C_col_inc + C_col_start];" << std::endl;
    else
      std::cout << "    C[(get_global_id(0) * C_row_inc + C_row_start) * C_internal_cols + get_global_id(1) * C_col_inc + C_col_start] = (beta == 0) ? alpha * Csub : alpha * Csub + beta * C[(get_global_id(0) * C_row_inc + C_row_start) * C_internal_cols + get_global_id(1) * C_col_inc + C_col_start];" << std::endl;
  }
  else
  {
    if (write_cuda)
      std::cout << "    C[(blockIdx.x * blockDim.x + threadIdx.x) * C_row_inc + C_row_start + ((blockIdx.y * blockDim.y + threadIdx.y) * C_col_inc + C_col_start) * C_internal_rows] = (beta == 0) ? alpha * Csub : alpha * Csub + beta * C[(blockIdx.x * blockDim.x + threadIdx.x) * C_row_inc + C_row_start + ((blockIdx.y * blockDim.y + threadIdx.y) * C_col_inc + C_col_start) * C_internal_rows];" << std::endl;
    else
      std::cout << "    C[get_global_id(0) * C_row_inc + C_row_start + (get_global_id(1) * C_col_inc + C_col_start) * C_internal_rows] = (beta == 0) ? alpha * Csub : alpha * Csub + beta * C[get_global_id(0) * C_row_inc + C_row_start + (get_global_id(1) * C_col_inc + C_col_start) * C_internal_rows];" << std::endl;
  }
  std::cout << "}" << std::endl;

  if (write_cuda)
    std::cout << std::endl;
}

void printUsage()
{
  std::cout << "Must have five parameters for C = A * B:" << std::endl;
  std::cout << " 0/1 : storage layout for A (column_major/row_major)" << std::endl;
  std::cout << " 0/1 : storage layout for B (column_major/row_major)" << std::endl;
  std::cout << " 0/1 : storage layout for C (column_major/row_major)" << std::endl;
  std::cout << " 0/1 : transpose for A (no/yes)" << std::endl;
  std::cout << " 0/1 : transpose for B (no/yes)" << std::endl;
  std::cout << " 0/1 : write CUDA output (no/yes)" << std::endl;
}

void readParameter(bool & param, char input)
{
  if (input == '0')
    param = false;
  else if (input == '1')
    param = true;
  else
  {
    printUsage();
    exit(EXIT_FAILURE);
  }
}

int main(int args, char * argsv[])
{
  if (args != 7)
  {
    printUsage();
    exit(EXIT_FAILURE);
  }

  //the following flags are 'true' for row_major layout
  bool layout_A;
  bool layout_B;
  bool layout_C;

  readParameter(layout_A, argsv[1][0]);
  readParameter(layout_B, argsv[2][0]);
  readParameter(layout_C, argsv[3][0]);

  bool transpose_A;
  bool transpose_B;
  readParameter(transpose_A, argsv[4][0]);
  readParameter(transpose_B, argsv[5][0]);

  bool writeCuda;
  readParameter(writeCuda, argsv[6][0]);


  printMatrixMatrixProduct(layout_A, layout_B, layout_C, transpose_A, transpose_B, writeCuda);
}
