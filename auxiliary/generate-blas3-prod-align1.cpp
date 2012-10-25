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
                              bool transpose_A, bool transpose_B)
{
  //write header:
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
  
  //start OpenCL code:
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
  std::cout << "          float alpha," << std::endl;
  std::cout << "          __global const float * A," << std::endl;
  std::cout << "          unsigned int A_row_start," << std::endl;
  std::cout << "          unsigned int A_col_start," << std::endl;
  std::cout << "          unsigned int A_row_inc," << std::endl;
  std::cout << "          unsigned int A_col_inc," << std::endl;
  std::cout << "          unsigned int A_row_size," << std::endl;   //number of elements starting from row_start!
  std::cout << "          unsigned int A_col_size," << std::endl;
  std::cout << "          unsigned int A_internal_rows," << std::endl;
  std::cout << "          unsigned int A_internal_cols," << std::endl;
  std::cout << "          __global const float * B,  " << std::endl;
  std::cout << "          unsigned int B_row_start," << std::endl;
  std::cout << "          unsigned int B_col_start," << std::endl;
  std::cout << "          unsigned int B_row_inc," << std::endl;
  std::cout << "          unsigned int B_col_inc," << std::endl;
  std::cout << "          unsigned int B_row_size," << std::endl;
  std::cout << "          unsigned int B_col_size," << std::endl;
  std::cout << "          unsigned int B_internal_rows," << std::endl;
  std::cout << "          unsigned int B_internal_cols," << std::endl;
  std::cout << "          float beta," << std::endl;
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
  std::cout << "  __local float bufA[" << 16 * 17 << "];" << std::endl;
  std::cout << "  __local float bufB[" << 16 * 17 << "];" << std::endl;
  std::cout << std::endl;
  //do not forgot to change block_size !!!
  std::cout << "  size_t block_size = 16;//get_local_size(0);" << std::endl;
  std::cout << "  size_t row_block_id = get_group_id(0);" << std::endl;
  std::cout << "  size_t col_block_id = get_group_id(1);" << std::endl;
  std::cout << "  size_t row_thread_id = get_local_id(0);" << std::endl;
  std::cout << "  size_t col_thread_id = get_local_id(1);" << std::endl;
  
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
/*  if (transpose_A)
    std::cout << "    if (block * block_size + col_thread_id < A_rows && get_global_id(0) < A_cols)" << std::endl;
  else 
    std::cout << "    if (block * block_size + col_thread_id < A_cols && get_global_id(0) < A_rows)" << std::endl;
  
  std::cout << "      bufA[row_thread_id * block_size + col_thread_id] = A[aBegin + aOffset];" << std::endl;
  std::cout << "    else" << std::endl;
  std::cout << "      bufA[row_thread_id * block_size + col_thread_id] = 0;" << std::endl;*/

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
  std::cout << "    barrier(CLK_LOCAL_MEM_FENCE);" << std::endl;
  //std::cout << "    for (size_t k = 0; k < block_size; ++k)" << std::endl;
  //std::cout << "      Csub += bufA[row_thread_id_times_block_size + k] * bufB[k * block_size + col_thread_id];" << std::endl;
  //loop unrolling:
  std::cout << "    __local float * bufAptr = bufA + row_thread_id_times_block_size;" << std::endl;
  std::cout << "    __local float * bufBptr = bufB + col_thread_id_times_block_size;" << std::endl;
  //std::cout << "      Csub += bufA[row_thread_id_times_block_size] * bufB[col_thread_id * block_size];" << std::endl;
  // code in following line depends on block size and must be changed in case of block_size changes
  std::cout << "      for(int i = 0; i < 4; i++) {" << std::endl;
  for (size_t unroll = 0; unroll < 4; ++unroll) {
    std::cout << "      Csub += (*bufAptr) * (*bufBptr); ++bufAptr; ++bufBptr;" << std::endl;
  }
  std::cout << "     }" << std::endl;
    //std::cout << "      Csub += bufAptr[" << i << "] * bufB[" << i << "  + col_thread_id * block_size];" << std::endl;
    //std::cout << "      Csub += bufAptr[" << i << "] * bufB[" << i << " * block_size + col_thread_id];" << std::endl;
    //std::cout << "      Csub += bufAptr[" << i << "] * bufB[" << i << "];" << std::endl;
  std::cout << "    barrier(CLK_LOCAL_MEM_FENCE);" << std::endl;
  std::cout << "    aBegin += aStep;" << std::endl;
  std::cout << "    bBegin += bStep;" << std::endl;
  std::cout << "  }" << std::endl;
  
  
  if (transpose_A)
    std::cout << "  if (get_global_id(0) < A_col_size && ";
  else
    std::cout << "  if (get_global_id(0) < A_row_size && ";
  
  if (transpose_B)
    std::cout << "get_global_id(1) < B_row_size)" << std::endl;
  else
    std::cout << "get_global_id(1) < B_col_size)" << std::endl;
  
  if (row_major_C)
    std::cout << "    C[(get_global_id(0) * C_row_inc + C_row_start) * C_internal_cols + get_global_id(1) * C_col_inc + C_col_start] = (beta == 0) ? alpha * Csub : alpha * Csub + beta * C[(get_global_id(0) * C_row_inc + C_row_start) * C_internal_cols + get_global_id(1) * C_col_inc + C_col_start];" << std::endl;
  else
    std::cout << "    C[get_global_id(0) * C_row_inc + C_row_start + (get_global_id(1) * C_col_inc + C_col_start) * C_internal_rows] = (beta == 0) ? alpha * Csub : alpha * Csub + beta * C[get_global_id(0) * C_row_inc + C_row_start + (get_global_id(1) * C_col_inc + C_col_start) * C_internal_rows];" << std::endl;
  std::cout << "}" << std::endl;
  
}

void printUsage()
{
  std::cout << "Must have five parameters for C = A * B:" << std::endl;
  std::cout << " 0/1 : storage layout for A (column_major/row_major)" << std::endl;
  std::cout << " 0/1 : storage layout for B (column_major/row_major)" << std::endl;
  std::cout << " 0/1 : storage layout for C (column_major/row_major)" << std::endl;
  std::cout << " 0/1 : transpose for A (no/yes)" << std::endl;
  std::cout << " 0/1 : transpose for B (no/yes)" << std::endl;
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
  if (args != 6)
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
  
  
  printMatrixMatrixProduct(layout_A, layout_B, layout_C, transpose_A, transpose_B);
}
