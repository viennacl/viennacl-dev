
__kernel void cpu_mult(
          __global const float16 * vec,
          unsigned int start1,
          unsigned int inc1,
          unsigned int size1,
          float factor, 
          __global float16 * result,
          unsigned int start2,
          unsigned int inc2,
          unsigned int size2) 
{ 
  unsigned int i_end = size1/16;
  for (unsigned int i = get_global_id(0); i < i_end; i += get_global_size(0))
    result[i*inc2+start2] = vec[i*inc1+start1] * factor;
}

