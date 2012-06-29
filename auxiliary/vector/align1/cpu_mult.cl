
__kernel void cpu_mult(
          __global const float * vec,
          unsigned int start1,
          unsigned int inc1,
          unsigned int size1,
          float factor, 
          __global float * result,
          unsigned int start2,
          unsigned int inc2,
          unsigned int size2) 
{ 
  for (unsigned int i = get_global_id(0); i < size1; i += get_global_size(0))
    result[i*inc2+start2] = vec[i*inc1+start1] * factor;
}


