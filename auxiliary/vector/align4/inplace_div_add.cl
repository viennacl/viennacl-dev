
__kernel void inplace_div_add(
          __global float4 * vec1,
          unsigned int start1,
          unsigned int inc1,
          unsigned int size1,
          __global const float4 * vec2,
          unsigned int start2,
          unsigned int inc2,
          unsigned int size2,
          __global const float * fac)   //CPU variant is mapped to mult_add
{ 
  float factor = *fac;
  unsigned int i_end = size1 / 4;
  for (unsigned int i = get_global_id(0); i < i_end; i += get_global_size(0))
    vec1[i*inc1+start1] -= vec2[i*inc2+start2] / factor;
}



