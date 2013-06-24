
__kernel void el_wise_mul_div(
          __global float * matrix1,
          __global const float * matrix2,
          __global const float * matrix3,
          unsigned int size)
{
  for (unsigned int i = get_global_id(0); i < size; i += get_global_size(0))
  {
    float val = matrix1[i] * matrix2[i];
    float divisor = matrix3[i];
    matrix1[i] = (divisor > 0.00001) ? (val / divisor) : 0;
  }
}
