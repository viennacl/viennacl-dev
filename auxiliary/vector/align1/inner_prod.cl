
__kernel void inner_prod(
          __global const float * vec1,
          unsigned int start1,
          unsigned int inc1,
          unsigned int size1,
          __global const float * vec2,
          unsigned int start2,
          unsigned int inc2,
          unsigned int size2,
          __local float * tmp_buffer,
          __global float * group_buffer)
{
  unsigned int entries_per_group = get_local_size(0) * (size1-1) / get_global_size(0) + 1;
  entries_per_group = (entries_per_group == 0) ? 1 : entries_per_group;
  unsigned int group_start1 = get_group_id(0) * entries_per_group * inc1 + start1;
  unsigned int group_start2 = get_group_id(0) * entries_per_group * inc2 + start2;

  unsigned int group_size = entries_per_group;
  if (get_group_id(0) * entries_per_group > size1)
    group_size = 0;
  else if ((get_group_id(0) + 1) * entries_per_group > size1)
    group_size = size1 - get_group_id(0) * entries_per_group;

  // compute partial results within group:
  float tmp = 0;
  for (unsigned int i = get_local_id(0); i < group_size; i += get_local_size(0))
    tmp += vec1[i*inc1 + group_start1] * vec2[i*inc2 + group_start2];
  tmp_buffer[get_local_id(0)] = tmp;

  // now run reduction:
  for (unsigned int stride = get_local_size(0)/2; stride > 0; stride /= 2)
  {
    barrier(CLK_LOCAL_MEM_FENCE);
    if (get_local_id(0) < stride)
      tmp_buffer[get_local_id(0)] += tmp_buffer[get_local_id(0)+stride];
  }

  if (get_local_id(0) == 0)
    group_buffer[get_group_id(0)] = tmp_buffer[get_local_id(0)];
}

