
__kernel void sub_wise(
          __global const float * matrix1,
          __global const float * matrix2,
          __global float * result,
          unsigned int size)
{
  for (unsigned int i = get_global_id(0); i < size; i += get_global_size(0))
    result[i] = matrix1[i] - matrix2[i];
}
