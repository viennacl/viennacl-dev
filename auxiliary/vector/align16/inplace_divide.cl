

__kernel void inplace_divide(
          __global float16 * vec,
          unsigned int start1,
          unsigned int inc1,
          unsigned int size1,
          __global const float * fac)  //note: CPU variant is mapped to prod_scalar
{ 
  float factor = *fac;
  unsigned int i_end = size1/16;
  for (unsigned int i = get_global_id(0); i < i_end; i += get_global_size(0))
    vec[i*inc1+start1] /= factor;
}

