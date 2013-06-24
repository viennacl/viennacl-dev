
// generic kernel for the vector operation v1 = alpha * v2 + beta * v3, where v1, v2, v3 are not necessarily distinct vectors
__kernel void element_op(
          __global float * vec1,
          unsigned int start1,
          unsigned int inc1,
          unsigned int size1,

          __global const float * vec2,
          unsigned int start2,
          unsigned int inc2,

          __global const float * vec3,
          unsigned int start3,
          unsigned int inc3,

          unsigned int is_division
          )
{
  if (is_division)
  {
    for (unsigned int i = get_global_id(0); i < size1; i += get_global_size(0))
      vec1[i*inc1+start1] = vec2[i*inc2+start2] / vec3[i*inc3+start3];
  }
  else
  {
    for (unsigned int i = get_global_id(0); i < size1; i += get_global_size(0))
      vec1[i*inc1+start1] = vec2[i*inc2+start2] * vec3[i*inc3+start3];
  }
}
