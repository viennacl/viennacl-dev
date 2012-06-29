
//helper:
void helper_inner_prod_parallel_reduction( __local float * tmp_buffer )
{
  for (unsigned int stride = get_local_size(0)/2; stride > 0; stride /= 2)
  {
    barrier(CLK_LOCAL_MEM_FENCE);
    if (get_local_id(0) < stride)
      tmp_buffer[get_local_id(0)] += tmp_buffer[get_local_id(0)+stride];
  }
}

//////// inner products:
float impl_inner_prod(
          __global const float * vec1,
          unsigned int start1,
          unsigned int inc1,
          unsigned int size1,
          __global const float * vec2,
          unsigned int start2,
          unsigned int inc2,
          unsigned int size2,
          __local float * tmp_buffer)
{
  float tmp = 0;
  for (unsigned int i = get_local_id(0); i < size1; i += get_local_size(0))
    tmp += vec1[i*inc1+start1] * vec2[i*inc2+start2];
  tmp_buffer[get_local_id(0)] = tmp;
  
  helper_inner_prod_parallel_reduction(tmp_buffer);
  
  return tmp_buffer[0];
}


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
          global float * group_buffer)
{
  float tmp = impl_inner_prod(vec1,
                              (      get_group_id(0) * size1) / get_num_groups(0) * inc1 + start1,
                              inc1,
                              ((get_group_id(0) + 1) * size1) / get_num_groups(0)
                                - (      get_group_id(0) * size1) / get_num_groups(0),
                              vec2,
                              (      get_group_id(0) * size2) / get_num_groups(0) * inc2 + start2,
                              inc2,
                              ((get_group_id(0) + 1) * size2) / get_num_groups(0)
                                - (      get_group_id(0) * size2) / get_num_groups(0),
                              tmp_buffer);
  
  if (get_local_id(0) == 0)
    group_buffer[get_group_id(0)] = tmp;
  
}

