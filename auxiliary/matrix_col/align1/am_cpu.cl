
// generic kernel for the matrix operation A = alpha * B, where A, B are not necessarily distinct matrices
__kernel void am_cpu(
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
          unsigned int B_internal_size1,  unsigned int B_internal_size2)
{
  float alpha = fac2;
  if (options2 & (1 << 0))
    alpha = -alpha;
  if (options2 & (1 << 1))
    alpha = ((float)(1)) / alpha;

  unsigned int row_gid = get_global_id(0) % get_local_size(0);
  unsigned int col_gid = get_global_id(0) / get_local_size(0);

  for (unsigned int col = col_gid; col < A_size2; col += get_num_groups(0))
    for (unsigned int row = row_gid; row < A_size1; row += get_local_size(0))
      A[(row * A_inc1 + A_start1) + (col * A_inc2 + A_start2) * A_internal_size1] = B[(row * B_inc1 + B_start1) + (col * B_inc2 + B_start2) * B_internal_size1] * alpha;
}
