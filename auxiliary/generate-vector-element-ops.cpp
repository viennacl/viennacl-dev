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
void print_unary_operation(std::string funcname, std::string op, std::string op_name)
{
  bool write_cuda = false;

  //start OpenCL code:
  if (write_cuda)
    std::cout << "template <typename T>" << std::endl;

  std::cout << (write_cuda ? "__global__" : "__kernel") << " void " << funcname << "_" << op_name << "(" << std::endl;
  if (write_cuda)
    std::cout << "          T * vec1," << std::endl;
  else
    std::cout << "          __global float * vec1," << std::endl;
  std::cout << "          uint4 size1," << std::endl;

  if (write_cuda)
    std::cout << "          const T * vec2," << std::endl;
  else
    std::cout << "          __global float * vec2," << std::endl;
  std::cout << "          uint4 size2) {" << std::endl;

  if (write_cuda)
    std::cout << "for (unsigned int i = blockDim.x * blockIdx.x + threadIdx.x; i < size1.z; i += gridDim.x * blockDim.x)" << std::endl;
  else
    std::cout << "for (unsigned int i = get_global_id(0); i < size1.z; i += get_global_size(0))" << std::endl;

  std::cout << "  vec1[i*size1.y+size1.x] " << op << " " << funcname << "(vec2[i*size2.y+size2.x]);" << std::endl;
  std::cout << "}" << std::endl;
}

void print_unary_operation(std::string funcname)
{
  print_unary_operation(funcname, "=", "assign");
  print_unary_operation(funcname, "+=", "plus");
  print_unary_operation(funcname, "-=", "minus");
}

int main()
{
  //print_unary_operation("abs");
  print_unary_operation("acos");
  print_unary_operation("asin");
  print_unary_operation("atan");
  print_unary_operation("ceil");
  print_unary_operation("cos");
  print_unary_operation("cosh");
  print_unary_operation("exp");
  print_unary_operation("fabs");
  print_unary_operation("floor");
  print_unary_operation("log");
  print_unary_operation("log10");
  print_unary_operation("sin");
  print_unary_operation("sinh");
  print_unary_operation("sqrt");
  print_unary_operation("tan");
  print_unary_operation("tanh");
}

