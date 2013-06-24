
__kernel void scaled_rank1_update_gpu(
          __global float * A,
          unsigned int A_start1, unsigned int A_start2,
          unsigned int A_inc1,   unsigned int A_inc2,
          unsigned int A_size1,  unsigned int A_size2,
          unsigned int A_internal_size1,  unsigned int A_internal_size2,

          __global const float * fac2,
          unsigned int options2,

          __global const float * vec1,
          unsigned int start1,
          unsigned int inc1,
          unsigned int size1,

          __global const float * vec2,
          unsigned int start2,
          unsigned int inc2,
          unsigned int size2)
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

  unsigned int row_gid = get_global_id(0) / get_local_size(0);
  unsigned int col_gid = get_global_id(0) % get_local_size(0);

  for (unsigned int row = row_gid; row < A_size1; row += get_num_groups(0))
  {
    float tmp = alpha * vec1[row * inc1 + start1];
    for (unsigned int col = col_gid; col < A_size2; col += get_local_size(0))
      A[(row * A_inc1 + A_start1) * A_internal_size2 + col * A_inc2 + A_start2] += tmp * vec2[col * inc2 + start2];
  }
}


