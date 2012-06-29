/*
* Generates BLAS level 3 routines for direct solve
*/

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>

#include <iostream>
#include <stdlib.h>

//generate code for inplace_solve(op1(A), op2(B), tag) where A and B can have different storage layouts and opX(D) = D or trans(D)
void printMatrixMatrixSolve(bool row_major_A, bool row_major_B,
                            bool transpose_A, bool transpose_B,
                            bool upper_solve, bool unit_diagonal)
{
  //write header:
  std::cout << "// file automatically generated - do not edit!" << std::endl;
  std::cout << "// inplace solve ";
  if (transpose_A)
    std::cout << "A^T \\\\ ";
  else
    std::cout << "A \\\\ ";
  if (transpose_B)
    std::cout << "B^T" << std::endl;
  else
    std::cout << "B" << std::endl;
  std::cout << "// matrix layouts: ";
  if (row_major_A)
    std::cout << "A...row_major, ";
  else
    std::cout << "A...col_major, ";
  if (row_major_B)
    std::cout << "B...row_major" << std::endl;
  else
    std::cout << "B...col_major" << std::endl;
  
  //start OpenCL code:
  std::cout << "__kernel void ";
  if (transpose_A)
    std::cout << "trans_";
  if (unit_diagonal)
    std::cout << "unit_";
  if (upper_solve)
    std::cout << "upper_";
  else
    std::cout << "lower_";
  if (transpose_B)
    std::cout << "trans_";
  std::cout << "solve";
  
  std::cout << "(" << std::endl;
  std::cout << "          __global const float * A," << std::endl;
  std::cout << "          unsigned int A_rows," << std::endl;
  std::cout << "          unsigned int A_cols," << std::endl;
  std::cout << "          unsigned int A_internal_rows," << std::endl;
  std::cout << "          unsigned int A_internal_cols," << std::endl;
  std::cout << "          __global float * B,  " << std::endl;
  std::cout << "          unsigned int B_rows," << std::endl;
  std::cout << "          unsigned int B_cols," << std::endl;
  std::cout << "          unsigned int B_internal_rows," << std::endl;
  std::cout << "          unsigned int B_internal_cols)" << std::endl;
  std::cout << "{ " << std::endl;
  std::cout << "  float temp; " << std::endl;
  if (upper_solve)
  {
    //Note: A is square, thus A_rows == A_cols and no dispatch for transposedness needed
    std::cout << "  for (int row = A_rows-1; row > -1; --row) " << std::endl;
  }
  else //lower triangular solve
  {
    std::cout << "  for (int row = 0; row < A_rows; ++row) " << std::endl;
  }
  std::cout << "  { " << std::endl;
  if (!unit_diagonal)
  {
    std::cout << "    barrier(CLK_GLOBAL_MEM_FENCE); " << std::endl;
    std::cout << "    if (get_local_id(0) == 0) " << std::endl;
    //Note: A is square, thus A_internal_rows == A_internal_cols and no dispatch for transposedness needed
    if (row_major_B && transpose_B)
      std::cout << "      B[row + get_group_id(0) * B_internal_cols] /= A[row + row*A_internal_cols]; " << std::endl;
    else if (row_major_B && !transpose_B)
      std::cout << "      B[row * B_internal_cols + get_group_id(0)] /= A[row + row*A_internal_cols]; " << std::endl;
    else if (!row_major_B && transpose_B)
      std::cout << "      B[row * B_internal_rows + get_group_id(0)] /= A[row + row*A_internal_cols]; " << std::endl;
    else if (!row_major_B && !transpose_B)
      std::cout << "      B[row + get_group_id(0) * B_internal_rows] /= A[row + row*A_internal_cols]; " << std::endl;
  }
  
  std::cout << "    barrier(CLK_GLOBAL_MEM_FENCE); " << std::endl;
  
  if (row_major_B && transpose_B)
    std::cout << "      temp = B[row + get_group_id(0) * B_internal_cols]; " << std::endl;
  else if (row_major_B && !transpose_B)
    std::cout << "      temp = B[row * B_internal_cols + get_group_id(0)]; " << std::endl;
  else if (!row_major_B && transpose_B)
    std::cout << "      temp = B[row * B_internal_rows + get_group_id(0)]; " << std::endl;
  else if (!row_major_B && !transpose_B)
    std::cout << "      temp = B[row + get_group_id(0) * B_internal_rows]; " << std::endl;

  std::cout << "    //eliminate column of op(A) with index 'row' in parallel: " << std::endl;
  if (upper_solve)
    std::cout << "    for  (int elim = get_local_id(0); elim < row; elim += get_local_size(0)) " << std::endl;
  else
    std::cout << "    for  (int elim = row + get_local_id(0) + 1; elim < A_rows; elim += get_local_size(0)) " << std::endl;
  
  if (row_major_B && transpose_B)
    std::cout << "      B[elim + get_group_id(0) * B_internal_cols] -= temp * ";
  else if (row_major_B && !transpose_B)
    std::cout << "      B[elim * B_internal_cols + get_group_id(0)] -= temp * ";
  else if (!row_major_B && transpose_B)
    std::cout << "      B[elim * B_internal_rows + get_group_id(0)] -= temp * ";
  else if (!row_major_B && !transpose_B)
    std::cout << "      B[elim + get_group_id(0) * B_internal_rows] -= temp * ";
  
  if (row_major_A && transpose_A)
    std::cout << "A[elim + row * A_internal_cols];" << std::endl;
  else if (row_major_A && !transpose_A)
    std::cout << "A[elim * A_internal_cols + row];" << std::endl;
  else if (!row_major_A && transpose_A)
    std::cout << "A[elim * A_internal_rows + row];" << std::endl;
  else if (!row_major_A && !transpose_A)
    std::cout << "A[elim + row * A_internal_rows];" << std::endl;
  
  std::cout << "   }" << std::endl;
  std::cout << "}" << std::endl;
  
}

void printUsage()
{
  std::cout << "Must have six parameters for A \\ B:" << std::endl;
  std::cout << " 0/1 : storage layout for A (column_major/row_major)" << std::endl;
  std::cout << " 0/1 : storage layout for B (column_major/row_major)" << std::endl;
  std::cout << " 0/1 : transpose for A (no/yes)" << std::endl;
  std::cout << " 0/1 : transpose for B (no/yes)" << std::endl;
  std::cout << " 0/1 : upper triangular system (no/yes)" << std::endl;
  std::cout << " 0/1 : has unit diagonal (no/yes)" << std::endl;
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

  readParameter(layout_A, argsv[1][0]);
  readParameter(layout_B, argsv[2][0]);
  
  bool transpose_A;
  bool transpose_B;
  readParameter(transpose_A, argsv[3][0]);
  readParameter(transpose_B, argsv[4][0]);
  
  bool upper_solve;
  bool unit_diagonal;
  readParameter(upper_solve,   argsv[5][0]);
  readParameter(unit_diagonal, argsv[6][0]);
  
  printMatrixMatrixSolve(layout_A, layout_B,
                         transpose_A, transpose_B,
                         upper_solve, unit_diagonal);
}
