
// generic kernel for the scalar operation s1 = alpha * s2, where s1, s2 are not necessarily distinct GPU scalars
__kernel void as_gpu(
          __global float * s1,
          __global const float * fac2,
          unsigned int options2,  // 0: no action, 1: flip sign, 2: take inverse, 3: flip sign and take inverse
          __global const float * s2)
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

  *s1 = *s2 * alpha;
}

