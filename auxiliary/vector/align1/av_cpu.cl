
// generic kernel for the vector operation v1 = alpha * v2, where v1, v2 are not necessarily distinct vectors
__kernel void av_cpu(
          __global float * vec1,
          unsigned int start1,
          unsigned int inc1,          
          unsigned int size1,
          
          float fac2,
          unsigned int options2,
          __global const float * vec2,
          unsigned int start2,
          unsigned int inc2)
{ 
  float alpha = fac2;
  if (options2 & (1 << 0))
    alpha = -alpha;
  if (options2 & (1 << 1))
    alpha = ((float)(1)) / alpha;

  for (unsigned int i = get_global_id(0); i < size1; i += get_global_size(0))
    vec1[i*inc1+start1] = vec2[i*inc2+start2] * alpha;
}
