
__kernel void sub(
          __global const float16 * vec1,
          unsigned int start1,
          unsigned int inc1,
          unsigned int size1,
          __global const float16 * vec2, 
          unsigned int start2,
          unsigned int inc2,
          unsigned int size2,
          __global float16 * result,
          unsigned int start3,
          unsigned int inc3,
          unsigned int size3)
{ 
  unsigned int i_end = size1 / 16;
  for (unsigned int i = get_global_id(0); i < i_end; i += get_global_size(0))
    result[i*inc3+start3] = vec1[i*inc1+start1] - vec2[i*inc2+start2];
}


