
// generic kernel for the matrix operation A = alpha * B + beta * C, where A, B, C are not necessarily distinct matrices
__kernel void ambm_m_cpu_cpu(
          __global float * A,
          unsigned int A_start1, unsigned int A_start2,
          unsigned int A_inc1,   unsigned int A_inc2,
          unsigned int A_size1,  unsigned int A_size2,
          unsigned int A_internal_size1,  unsigned int A_internal_size2,

          float fac2,
          unsigned int options2,
          __global const float * B,
          unsigned int B_start1, unsigned int B_start2,
          unsigned int B_inc1,   unsigned int B_inc2,
          unsigned int B_internal_size1,  unsigned int B_internal_size2,

          float fac3,
          unsigned int options3,
          __global const float * C,
          unsigned int C_start1, unsigned int C_start2,
          unsigned int C_inc1,   unsigned int C_inc2,
          unsigned int C_internal_size1,  unsigned int C_internal_size2)
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

  unsigned int row_gid = get_global_id(0) / get_local_size(0);
  unsigned int col_gid = get_global_id(0) % get_local_size(0);

  for (unsigned int row = row_gid; row < A_size1; row += get_num_groups(0))
    for (unsigned int col = col_gid; col < A_size2; col += get_local_size(0))
      A[(row * A_inc1 + A_start1) * A_internal_size2 + col * A_inc2 + A_start2]
   += B[(row * B_inc1 + B_start1) * B_internal_size2 + col * B_inc2 + B_start2] * alpha
    + C[(row * C_inc1 + C_start1) * C_internal_size2 + col * C_inc2 + C_start2] * beta;
}
