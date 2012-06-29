
__kernel void inplace_mul_sub(
          __global float * vec1,
          unsigned int start1,
          unsigned int inc1,
          unsigned int size1,
          __global const float * vec2,
          unsigned int start2,
          unsigned int inc2,
          unsigned int size2,
          __global const float * fac)   //CPU variant is mapped to mult_add
{ 
  float factor = *fac;
  for (unsigned int i = get_global_id(0); i < size1; i += get_global_size(0))
    vec1[i*inc1+start1] -= vec2[i*inc2+start2] * factor;
}


