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
  std::cout << "          unsigned int A_start1, unsigned int A_start2," << std::endl;
  std::cout << "          unsigned int A_inc1,   unsigned int A_inc2," << std::endl;
  std::cout << "          unsigned int A_size1,  unsigned int A_size2," << std::endl;
  std::cout << "          unsigned int A_internal_size1, unsigned int A_internal_size2," << std::endl;
  std::cout << "          __global float * B," << std::endl;
  std::cout << "          unsigned int B_start1, unsigned int B_start2," << std::endl;
  std::cout << "          unsigned int B_inc1,   unsigned int B_inc2," << std::endl;
  std::cout << "          unsigned int B_size1,  unsigned int B_size2," << std::endl;
  std::cout << "          unsigned int B_internal_size1, unsigned int B_internal_size2)" << std::endl;
  std::cout << "{ " << std::endl;
  std::cout << "  float temp; " << std::endl;
  if (upper_solve)
  {
    //Note: A is square, thus A_rows == A_cols and no dispatch for transposedness needed
    std::cout << "  for (unsigned int row_cnt = 0; row_cnt < A_size1; ++row_cnt) " << std::endl;
    std::cout << "  { " << std::endl;
    std::cout << "    unsigned int row = A_size1 - 1 - row_cnt;" << std::endl;
  }
  else //lower triangular solve
  {
    std::cout << "  for (unsigned int row = 0; row < A_size1; ++row) " << std::endl;
    std::cout << "  { " << std::endl;
  }

  if (!unit_diagonal)
  {
    std::cout << "    barrier(CLK_GLOBAL_MEM_FENCE); " << std::endl;
    std::cout << "    if (get_local_id(0) == 0) " << std::endl;
    //Note: A is square, thus A_internal_rows == A_internal_cols and no dispatch for transposedness needed
    if (row_major_B && transpose_B)
      std::cout << "      B[(get_group_id(0) * B_inc1 + B_start1) * B_internal_size2 + (row * B_inc2 + B_start2)] /= ";
    else if (row_major_B && !transpose_B)
      std::cout << "      B[(row * B_inc1 + B_start1) * B_internal_size2 + (get_group_id(0) * B_inc2 + B_start2)] /= ";
    else if (!row_major_B && transpose_B)
      std::cout << "      B[(get_group_id(0) * B_inc1 + B_start1) + (row * B_inc2 + B_start2) * B_internal_size1] /= ";
    else if (!row_major_B && !transpose_B)
      std::cout << "      B[(row * B_inc1 + B_start1) + (get_group_id(0) * B_inc2 + B_start2) * B_internal_size1] /= ";

    if (row_major_A)
      std::cout << "A[(row * A_inc1 + A_start1) * A_internal_size2 + (row * A_inc2 + A_start2)];" << std::endl;
    else
      std::cout << "A[(row * A_inc1 + A_start1) + (row * A_inc2 + A_start2)*A_internal_size1];" << std::endl;
  }

  std::cout << "    barrier(CLK_GLOBAL_MEM_FENCE); " << std::endl;

  if (row_major_B && transpose_B)
    std::cout << "    temp = B[(get_group_id(0) * B_inc1 + B_start1) * B_internal_size2 + (row * B_inc2 + B_start2)]; " << std::endl;
  else if (row_major_B && !transpose_B)
    std::cout << "    temp = B[(row * B_inc1 + B_start1) * B_internal_size2 + (get_group_id(0) * B_inc2 + B_start2)]; " << std::endl;
  else if (!row_major_B && transpose_B)
    std::cout << "    temp = B[(get_group_id(0) * B_inc1 + B_start1) + (row * B_inc2 + B_start2) * B_internal_size1]; " << std::endl;
  else if (!row_major_B && !transpose_B)
    std::cout << "    temp = B[(row * B_inc1 + B_start1) + (get_group_id(0) * B_inc2 + B_start2) * B_internal_size1]; " << std::endl;

  std::cout << "    //eliminate column of op(A) with index 'row' in parallel: " << std::endl;
  if (upper_solve)
    std::cout << "    for  (unsigned int elim = get_local_id(0); elim < row; elim += get_local_size(0)) " << std::endl;
  else
    std::cout << "    for  (unsigned int elim = row + get_local_id(0) + 1; elim < A_size1; elim += get_local_size(0)) " << std::endl;

  if (row_major_B && transpose_B)
    std::cout << "      B[(get_group_id(0) * B_inc1 + B_start1) * B_internal_size2 + (elim * B_inc2 + B_start2)] -= temp * ";
  else if (row_major_B && !transpose_B)
    std::cout << "      B[(elim * B_inc1 + B_start1) * B_internal_size2 + (get_group_id(0) * B_inc2 + B_start2)] -= temp * ";
  else if (!row_major_B && transpose_B)
    std::cout << "      B[(get_group_id(0) * B_inc1 + B_start1) + (elim * B_inc2 + B_start2) * B_internal_size1] -= temp * ";
  else if (!row_major_B && !transpose_B)
    std::cout << "      B[(elim * B_inc1 + B_start1) + (get_group_id(0) * B_inc2 + B_start2) * B_internal_size1] -= temp * ";

  if (row_major_A && transpose_A)
    std::cout << "A[(row * A_inc1 + A_start1) * A_internal_size2 + (elim * A_inc2 + A_start2)];" << std::endl;
  else if (row_major_A && !transpose_A)
    std::cout << "A[(elim * A_inc1 + A_start1) * A_internal_size2 + (row * A_inc2 + A_start2)];" << std::endl;
  else if (!row_major_A && transpose_A)
    std::cout << "A[(row * A_inc1 + A_start1) + (elim * A_inc2 + A_start2) * A_internal_size1];" << std::endl;
  else if (!row_major_A && !transpose_A)
    std::cout << "A[(elim * A_inc1 + A_start1) + (row * A_inc2 + A_start2) * A_internal_size1];" << std::endl;

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
