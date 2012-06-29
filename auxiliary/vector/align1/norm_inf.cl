
////// norm_inf
float impl_norm_inf(
          __global const float * vec,
          unsigned int start1,
          unsigned int inc1,
          unsigned int size1,
          __local float * tmp_buffer)
{
  float tmp = 0;
  for (unsigned int i = get_local_id(0); i < size1; i += get_local_size(0))
    tmp = fmax(fabs(vec[i*inc1 + start1]), tmp);
  tmp_buffer[get_local_id(0)] = tmp;
  
  //step 2: parallel reduction:
  for (unsigned int stride = get_global_size(0)/2; stride > 0; stride /= 2)
  {
    barrier(CLK_LOCAL_MEM_FENCE);
    if (get_global_id(0) < stride)
      tmp_buffer[get_global_id(0)] = fmax(tmp_buffer[get_global_id(0)], tmp_buffer[get_global_id(0)+stride]);
  }
  
  return tmp_buffer[0];
}

__kernel void norm_inf(
          __global const float * vec,
          unsigned int start1,
          unsigned int inc1,
          unsigned int size1,
          __local float * tmp_buffer,
          global float * group_buffer)
{
  float tmp = impl_norm_inf(vec,
                          (      get_group_id(0) * size1) / get_num_groups(0) + start1,
                          inc1, 
                          ((get_group_id(0) + 1) * size1) / get_num_groups(0)
                            - (      get_group_id(0) * size1) / get_num_groups(0),
                          tmp_buffer);
  
  if (get_local_id(0) == 0)
    group_buffer[get_group_id(0)] = tmp;  
}
