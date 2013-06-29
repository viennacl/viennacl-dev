/*
* Generates element-wise vector operations
*/

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>

#include <iostream>
#include <stdlib.h>

//generate code for C = op1(A) * op2(B), where A, B, C can have different storage layouts and opX(D) = D or trans(D)
void print_unary_operation(std::string funcname, bool is_row_major, std::string op, std::string op_name)
{
  //start OpenCL code:

  std::cout << "__kernel void " << funcname << "_" << op_name << "(" << std::endl;
  std::cout << "          __global float * A," << std::endl;
  std::cout << "          unsigned int A_start1, unsigned int A_start2," << std::endl;
  std::cout << "          unsigned int A_inc1,   unsigned int A_inc2," << std::endl;
  std::cout << "          unsigned int A_size1,  unsigned int A_size2," << std::endl;
  std::cout << "          unsigned int A_internal_size1,  unsigned int A_internal_size2," << std::endl;

  std::cout << "          __global const float * B," << std::endl;
  std::cout << "          unsigned int B_start1, unsigned int B_start2," << std::endl;
  std::cout << "          unsigned int B_inc1,   unsigned int B_inc2," << std::endl;
  std::cout << "          unsigned int B_internal_size1,  unsigned int B_internal_size2) {" << std::endl;

  if (is_row_major)
  {
    std::cout << "  unsigned int row_gid = get_global_id(0) / get_local_size(0);" << std::endl;
    std::cout << "  unsigned int col_gid = get_global_id(0) % get_local_size(0);" << std::endl;

    std::cout << "  for (unsigned int row = row_gid; row < A_size1; row += get_num_groups(0))" << std::endl;
    std::cout << "    for (unsigned int col = col_gid; col < A_size2; col += get_local_size(0))" << std::endl;
    std::cout << "      A[(row * A_inc1 + A_start1) * A_internal_size2 + col * A_inc2 + A_start2]" << std::endl;
    std::cout << "        " << op << " " << funcname << "(B[(row * B_inc1 + B_start1) * B_internal_size2 + col * B_inc2 + B_start2]);" << std::endl;
  }
  else
  {
    std::cout << "  unsigned int row_gid = get_global_id(0) % get_local_size(0);" << std::endl;
    std::cout << "  unsigned int col_gid = get_global_id(0) / get_local_size(0);" << std::endl;

    std::cout << "  for (unsigned int col = col_gid; col < A_size2; col += get_num_groups(0))" << std::endl;
    std::cout << "    for (unsigned int row = row_gid; row < A_size1; row += get_local_size(0))" << std::endl;
    std::cout << "      A[(row * A_inc1 + A_start1) + (col * A_inc2 + A_start2) * A_internal_size1]" << std::endl;
    std::cout << "        " << op << " " << funcname << "(B[(row * B_inc1 + B_start1) + (col * B_inc2 + B_start2) * B_internal_size1]);" << std::endl;
  }
  std::cout << "}" << std::endl;
}

void print_unary_operation(std::string funcname, bool is_row_major)
{
  print_unary_operation(funcname, is_row_major, "=", "assign");
  print_unary_operation(funcname, is_row_major, "+=", "plus");
  print_unary_operation(funcname, is_row_major, "-=", "minus");
}

int main(int argc, char **argv)
{
  bool row_major = true;
  if (argc == 2)
    row_major = bool(atol(argv[1]));

  //print_unary_operation("abs");
  print_unary_operation("acos", row_major);
  print_unary_operation("asin", row_major);
  print_unary_operation("atan", row_major);
  print_unary_operation("ceil", row_major);
  print_unary_operation("cos",  row_major);
  print_unary_operation("cosh", row_major);
  print_unary_operation("exp",  row_major);
  print_unary_operation("fabs", row_major);
  print_unary_operation("floor", row_major);
  print_unary_operation("log",  row_major);
  print_unary_operation("log10", row_major);
  print_unary_operation("sin",  row_major);
  print_unary_operation("sinh", row_major);
  print_unary_operation("sqrt", row_major);
  print_unary_operation("tan",  row_major);
  print_unary_operation("tanh", row_major);
}

