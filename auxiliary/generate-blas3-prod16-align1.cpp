/*
* Generates BLAS level 3 routines
*/

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>

#include <iostream>
#include <stdlib.h>

//generate code for C = alpha * op1(A) * op2(B) + beta * C, where A, B, C can have different storage layouts and opX(D) = D or trans(D)
void printMatrixMatrixProduct(bool row_major_A, bool row_major_B, bool row_major_C,
                              bool transpose_A, bool transpose_B)
{
  std::size_t vector_size =  4;
  std::size_t block_size  = 16;

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
  std::cout << "__kernel void prod16_";
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
  std::cout << "          unsigned int A_row_size," << std::endl;   //number of elements starting from row_start, using an increment of A_row_inc
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
  //do not forgot to change block_size !!!
  std::cout << "  size_t row_block_id = get_group_id(1);" << std::endl;    //refers to the row index in op(A), op(B)
  std::cout << "  size_t col_block_id = get_group_id(0);" << std::endl;    //refers to the col index in op(A), op(B)
  std::cout << "  size_t row_thread_id = get_local_id(1);" << std::endl;
  std::cout << "  size_t col_thread_id = get_local_id(0);" << std::endl;
  std::cout << std::endl;
  std::cout << "  __local float As[" << block_size * block_size << "];" << std::endl;
  std::cout << std::endl;
  std::cout << "  float cv[" << block_size << "] = {";
  for (std::size_t i=0; i<block_size-1; ++i)
    std::cout << "0,";
  std::cout << "0};" << std::endl;

  //traverse block row of A (taking mem layout and transpose operation into account)
  if (row_major_A && transpose_A)
  {
    std::cout << "  size_t aBegin = (row_block_id * " << block_size << " * A_col_inc + A_col_start) + A_row_start * A_internal_cols;" << std::endl;
    std::cout << "  size_t aStep = " << block_size << " * A_internal_cols * A_row_inc;" << std::endl;
    std::cout << "  size_t aEnd = aBegin + A_internal_cols * A_row_inc * A_row_size;" << std::endl;
  }
  else if (row_major_A && !transpose_A)
  {
    std::cout << "  size_t aBegin = (row_block_id * " << block_size << " * A_row_inc + A_row_start) * A_internal_cols + A_col_start;" << std::endl;
    std::cout << "  size_t aStep = " << block_size << " * A_col_inc;" << std::endl;
    std::cout << "  size_t aEnd = aBegin + A_col_inc * A_col_size;" << std::endl;
  }
  else if (!row_major_A && transpose_A)
  {
    std::cout << "  size_t aBegin = (row_block_id * " << block_size << " * A_col_inc + A_col_start) * A_internal_rows + A_row_start;" << std::endl;
    std::cout << "  size_t aStep = " << block_size << " * A_row_inc;" << std::endl;
    std::cout << "  size_t aEnd = aBegin + A_row_inc * A_row_size;" << std::endl;
  }
  else if (!row_major_A && !transpose_A)
  {
    std::cout << "  size_t aBegin = (row_block_id * " << block_size << " * A_row_inc + A_row_start) + A_col_start * A_internal_rows;" << std::endl;
    std::cout << "  size_t aStep = " << block_size << " * A_internal_rows * A_col_inc;" << std::endl;
    std::cout << "  size_t aEnd = aBegin + A_internal_rows * A_col_inc * A_col_size;" << std::endl;
  }


  if (row_major_B && transpose_B)
  {
    std::cout << "  size_t bBegin = (col_block_id * " << block_size * vector_size << " * B_row_inc + B_row_start) * B_internal_cols + B_col_start;" << std::endl;
    std::cout << "  size_t bStep = " << block_size << " * B_col_inc;" << std::endl;
  }
  else if (row_major_B && !transpose_B)
  {
    std::cout << "  size_t bBegin = (col_block_id * " << block_size * vector_size << " * B_col_inc + B_col_start) + B_row_start * B_internal_cols;" << std::endl;
    std::cout << "  size_t bStep = " << block_size << " * B_row_inc * B_internal_cols;" << std::endl;
  }
  else if (!row_major_B && transpose_B)
  {
    std::cout << "  size_t bBegin = (col_block_id * " << block_size * vector_size << " * B_row_inc + B_row_start) + B_col_start * B_internal_rows;" << std::endl;
    std::cout << "  size_t bStep = " << block_size << " * B_col_inc * B_internal_rows;" << std::endl;
  }
  else if (!row_major_B && !transpose_B)
  {
    std::cout << "  size_t bBegin = (col_block_id * " << block_size * vector_size << " * B_col_inc + B_col_start) * B_internal_rows + B_row_start;" << std::endl;
    std::cout << "  size_t bStep = " << block_size << " * B_row_inc;" << std::endl;
  }

  std::cout << "  for(size_t a = aBegin, b = bBegin; a < aEnd; a += aStep, b += bStep) { " << std::endl;

  // copy blocks of op(A) to shared memory (op(A) is column-major in shared memory then)
  std::cout << "    for(size_t i = 0; i < " << vector_size << "; i++)  " << std::endl;
  if (row_major_A && transpose_A)
    std::cout << "      As[ (i*" << vector_size << " + row_thread_id) + " << block_size << " * col_thread_id] = (A[a + A_col_inc * (i * " << vector_size << " + row_thread_id) + A_internal_cols * A_row_inc * col_thread_id]);"  << std::endl;
  else if (row_major_A && !transpose_A)
    std::cout << "      As[ (i*" << vector_size << " + row_thread_id) + " << block_size << " * col_thread_id] = (A[a + A_internal_cols * A_row_inc * (i * " << vector_size << " + row_thread_id) + A_col_inc * col_thread_id]);"  << std::endl;
  else if (!row_major_A && transpose_A)
    std::cout << "      As[ (i*" << vector_size << " + row_thread_id) + " << block_size << " * col_thread_id] = (A[a + A_internal_rows * A_col_inc * (i * " << vector_size << " + row_thread_id) + A_row_inc * col_thread_id]);"  << std::endl;
  else if (!row_major_A && !transpose_A)
    std::cout << "      As[ (i*" << vector_size << " + row_thread_id) + " << block_size << " * col_thread_id] = (A[a + A_row_inc * (i * " << vector_size << " + row_thread_id) + A_internal_rows * A_col_inc * col_thread_id]);"  << std::endl;
  std::cout << std::endl;
  std::cout << "    barrier(CLK_LOCAL_MEM_FENCE); " << std::endl;

  // initialize memory pointers
  std::cout << std::endl;
  std::cout << "    __local  float *ap = As; " << std::endl;
  if (row_major_B && transpose_B)
    std::cout << "    __global const float *bp = B + (b + (" << block_size << " * row_thread_id + col_thread_id) * B_row_inc * B_internal_cols); " << std::endl;
  else if (row_major_B && !transpose_B)
    std::cout << "    __global const float *bp = B + (b + (" << block_size << " * row_thread_id + col_thread_id) * B_col_inc); " << std::endl;
  else if (!row_major_B && transpose_B)
    std::cout << "    __global const float *bp = B + (b + (" << block_size << " * row_thread_id + col_thread_id) * B_row_inc); " << std::endl;
  else if (!row_major_B && !transpose_B)
    std::cout << "    __global const float *bp = B + (b + (" << block_size << " * row_thread_id + col_thread_id) * B_col_inc * B_internal_rows); " << std::endl;
  std::cout << std::endl;

  // run computations
  std::cout << "    for(size_t i = 0; i < " << block_size << "; i++) { " << std::endl;
  if (row_major_B && transpose_B)
    std::cout << "      float bv = bp[i * B_col_inc]; " << std::endl;
  else if (row_major_B && !transpose_B)
    std::cout << "      float bv = bp[i * B_row_inc * B_internal_cols]; " << std::endl;
  else if (!row_major_B && transpose_B)
    std::cout << "      float bv = bp[i * B_col_inc * B_internal_rows]; " << std::endl;
  else if (!row_major_B && !transpose_B)
    std::cout << "      float bv = bp[i * B_row_inc]; " << std::endl;
  std::cout << std::endl;
  std::cout << "      for(size_t k = 0; k < " << block_size << "; k++)  " << std::endl;
  std::cout << "	    cv[k] += ap[k] * bv; " << std::endl;
  std::cout << std::endl;
  std::cout << "      ap += " << block_size << "; " << std::endl;
  std::cout << "    } " << std::endl;
  std::cout << std::endl;
  std::cout << "    barrier(CLK_LOCAL_MEM_FENCE); " << std::endl;
  std::cout << "  } " << std::endl;

  // write to C
  if (row_major_C)
  {
      std::cout << "  int c = C_internal_cols * (C_row_inc * " << block_size << " * row_block_id + C_row_start) + "  //block row index
                              << vector_size * block_size << " * C_col_inc * col_block_id + C_col_start " << std::endl;  //block column index
      std::cout << "          + C_col_inc * (" << block_size << " * row_thread_id + col_thread_id); " << std::endl;
  }
  else
  {
      std::cout << "  int c = C_row_inc * " << block_size << " * row_block_id + C_row_start + ("   // block row index
                              << vector_size * block_size << " * C_col_inc * col_block_id + C_col_start) * C_internal_rows " << std::endl;   // block column index
      std::cout << "          + C_internal_rows * C_col_inc * (" << block_size << " * row_thread_id + col_thread_id); " << std::endl;
  }

  std::cout << "  for(size_t i = 0; i < " << block_size << "; i++) { " << std::endl;

  if (row_major_C)
  {
    std::cout << "    C[c] = (beta == 0) ? alpha * cv[i] : alpha * cv[i] + beta * C[c]; " << std::endl;
    std::cout << "      c += C_internal_cols * C_row_inc; " << std::endl;
  }
  else
  {
    std::cout << "    C[c] = (beta == 0) ? alpha * cv[i] : alpha * cv[i] + beta * C[c]; " << std::endl;
    std::cout << "      c += C_row_inc; " << std::endl;
  }

  std::cout << "  } " << std::endl;
  std::cout << "} " << std::endl;



//  if (row_major_C)
//    std::cout << "    C[(get_global_id(0) * C_row_inc + C_row_start) * C_internal_cols + get_global_id(1) * C_col_inc + C_col_start] = Csub;" << std::endl;
//  else
//    std::cout << "    C[get_global_id(0) * C_row_inc + C_row_start + (get_global_id(1) * C_col_inc + C_col_start) * C_internal_rows] = Csub;" << std::endl;

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
