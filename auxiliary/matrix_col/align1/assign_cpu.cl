
__kernel void assign_cpu(
          __global float * A,
          unsigned int start1,          unsigned int start2,
          unsigned int inc1,            unsigned int inc2,
          unsigned int size1,           unsigned int size2,
          unsigned int internal_size1,  unsigned int internal_size2,
          float alpha)
{
  unsigned int row_gid = get_global_id(0) % get_local_size(0);
  unsigned int col_gid = get_global_id(0) / get_local_size(0);

  for (unsigned int col = col_gid; col < size2; col += get_num_groups(0))
    for (unsigned int row = row_gid; row < size1; row += get_local_size(0))
      A[(row * inc1 + start1) + (col * inc2 + start2) * internal_size1] = alpha;
}

