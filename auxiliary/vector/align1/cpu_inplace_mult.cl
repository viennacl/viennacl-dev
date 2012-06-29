
__kernel void cpu_inplace_mult(
          __global float * vec,
          unsigned int start1,
          unsigned int inc1,
          unsigned int size1,
          float factor) 
{ 
  for (unsigned int i = get_global_id(0); i < size1; i += get_global_size(0))
    vec[i*inc1+start1] *= factor;
}

