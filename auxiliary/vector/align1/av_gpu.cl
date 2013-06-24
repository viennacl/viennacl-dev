
// generic kernel for the vector operation v1 = alpha * v2, where v1, v2 are not necessarily distinct vectors
__kernel void av_gpu(
          __global float * vec1,
          uint4 size1,

          __global const float * fac2,
          unsigned int options2,  // 0: no action, 1: flip sign, 2: take inverse, 3: flip sign and take inverse
          __global const float * vec2,
          uint4 size2)
{
  float alpha = fac2[0];
  if ((options2 >> 2) > 1)
  {
    for (unsigned int i=1; i<(options2 >> 2); ++i)
      alpha += fac2[i];
  }
  if (options2 & (1 << 0))
    alpha = -alpha;
  if (options2 & (1 << 1))
    alpha = ((float)(1)) / alpha;

  for (unsigned int i = get_global_id(0); i < size1.z; i += get_global_size(0))
    vec1[i*size1.y+size1.x] = vec2[i*size2.y+size2.x] * alpha;
}

