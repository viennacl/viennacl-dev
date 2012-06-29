
///// multiply subtract:
__kernel void mul_sub(
          __global const float * vec1,
          unsigned int start1,
          unsigned int inc1,
          unsigned int size1,
          __global const float * fac,
          __global const float * vec2,
          unsigned int start2,
          unsigned int inc2,
          unsigned int size2,
          __global float * result,
          unsigned int start3,
          unsigned int inc3,
          unsigned int size3
          ) 
{ 
  float factor = *fac;
  for (unsigned int i = get_global_id(0); i < size1; i += get_global_size(0))
    result[i*inc3+start3] = vec1[i*inc1+start1] * factor - vec2[i*inc2+start2];
}

