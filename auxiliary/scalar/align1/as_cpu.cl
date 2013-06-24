
// generic kernel for the scalar operation s1 = alpha * s2, where s1, s2 are not necessarily distinct GPU scalars
__kernel void as_cpu(
          __global float * s1,
          float fac2,
          unsigned int options2,
          __global const float * s2)
{
  float alpha = fac2;
  if (options2 & (1 << 0))
    alpha = -alpha;
  if (options2 & (1 << 1))
    alpha = ((float)(1)) / alpha;

  *s1 = *s2 * alpha;
}
