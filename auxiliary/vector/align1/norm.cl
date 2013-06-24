//helper:
void helper_norm_parallel_reduction( __local float * tmp_buffer )
{
  for (unsigned int stride = get_local_id(0)/2; stride > 0; stride /= 2)
  {
    barrier(CLK_LOCAL_MEM_FENCE);
    if (get_local_id(0) < stride)
      tmp_buffer[get_local_id(0)] += tmp_buffer[get_local_id(0)+stride];
  }
}

float impl_norm(
          __global const float * vec,
          unsigned int start1,
          unsigned int inc1,
          unsigned int size1,
          unsigned int norm_selector,
          __local float * tmp_buffer)
{
  float tmp = 0;
  if (norm_selector == 1) //norm_1
  {
    for (unsigned int i = get_local_id(0); i < size1; i += get_local_size(0))
      tmp += fabs(vec[i*inc1 + start1]);
  }
  else if (norm_selector == 2) //norm_2
  {
    float vec_entry = 0;
    for (unsigned int i = get_local_id(0); i < size1; i += get_local_size(0))
    {
      vec_entry = vec[i*inc1 + start1];
      tmp += vec_entry * vec_entry;
    }
  }
  else if (norm_selector == 0) //norm_inf
  {
    for (unsigned int i = get_local_id(0); i < size1; i += get_local_size(0))
      tmp = fmax(fabs(vec[i*inc1 + start1]), tmp);
  }

  tmp_buffer[get_local_id(0)] = tmp;

  if (norm_selector > 0) //norm_1 or norm_2:
  {
    for (unsigned int stride = get_local_size(0)/2; stride > 0; stride /= 2)
    {
      barrier(CLK_LOCAL_MEM_FENCE);
      if (get_local_id(0) < stride)
        tmp_buffer[get_local_id(0)] += tmp_buffer[get_local_id(0)+stride];
    }
    return tmp_buffer[0];
  }

  //norm_inf:
  for (unsigned int stride = get_local_size(0)/2; stride > 0; stride /= 2)
  {
    barrier(CLK_LOCAL_MEM_FENCE);
    if (get_local_id(0) < stride)
      tmp_buffer[get_local_id(0)] = fmax(tmp_buffer[get_local_id(0)], tmp_buffer[get_local_id(0)+stride]);
  }

  return tmp_buffer[0];
};

__kernel void norm(
          __global const float * vec,
          unsigned int start1,
          unsigned int inc1,
          unsigned int size1,
          unsigned int norm_selector,
          __local float * tmp_buffer,
          __global float * group_buffer)
{
  float tmp = impl_norm(vec,
                        (        get_group_id(0)  * size1) / get_num_groups(0) * inc1 + start1,
                        inc1,
                        (   (1 + get_group_id(0)) * size1) / get_num_groups(0)
                      - (        get_group_id(0)  * size1) / get_num_groups(0),
                        norm_selector,
                        tmp_buffer);

  if (get_local_id(0) == 0)
    group_buffer[get_group_id(0)] = tmp;
}

