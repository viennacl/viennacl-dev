
// generic kernel for the scalar operation s1 += alpha * s2 + beta * s3 + gamma * v4, where v1, v2, v3 are not necessarily distinct vectors
__kernel void asbs_s_cpu_cpu(
          __global float * s1,

          float fac2,
          unsigned int options2,  // 0: no action, 1: flip sign, 2: take inverse, 3: flip sign and take inverse
          __global const float * s2,

          float fac3,
          unsigned int options3,  // 0: no action, 1: flip sign, 2: take inverse, 3: flip sign and take inverse
          __global const float * s3)
{
  float alpha = fac2;
  if (options2 & (1 << 0))
    alpha = -alpha;
  if (options2 & (1 << 1))
    alpha = ((float)(1)) / alpha;

  float beta = fac3;
  if (options3 & (1 << 0))
    beta = -beta;
  if (options3 & (1 << 1))
    beta = ((float)(1)) / beta;

  *s1 += *s2 * alpha + *s3 * beta;
}
