void helper_bicgstab_kernel1_parallel_reduction( __local float * tmp_buffer )
{
  for (unsigned int stride = get_local_size(0)/2; stride > 0; stride /= 2)
  {
    barrier(CLK_LOCAL_MEM_FENCE);
    if (get_local_id(0) < stride)
      tmp_buffer[get_local_id(0)] += tmp_buffer[get_local_id(0)+stride];
  }
}

//////// inner products:
float bicgstab_kernel1_inner_prod(
          __global const float * vec1,
          __global const float * vec2,
          unsigned int size,
          __local float * tmp_buffer)
{
  float tmp = 0;
  unsigned int i_end = ((size - 1) / get_local_size(0) + 1) * get_local_size(0);
  for (unsigned int i = get_local_id(0); i < i_end; i += get_local_size(0))
  {
    if (i < size)
      tmp += vec1[i] * vec2[i];
  }
  tmp_buffer[get_local_id(0)] = tmp;
  
  helper_bicgstab_kernel1_parallel_reduction(tmp_buffer);

  barrier(CLK_LOCAL_MEM_FENCE);

  return tmp_buffer[0];
}


__kernel void bicgstab_kernel1(
          __global const float * tmp0,
          __global const float * r0star, 
          __global const float * residual,
          __global float * s,
          __global float * alpha,
          __global const float * ip_rr0star,
          __local float * tmp_buffer,
          unsigned int size) 
{ 
  float alpha_local = ip_rr0star[0] / bicgstab_kernel1_inner_prod(tmp0, r0star, size, tmp_buffer);
  
  for (unsigned int i = get_local_id(0); i < size; i += get_local_size(0))
    s[i] = residual[i] - alpha_local * tmp0[i];
  
  if (get_global_id(0) == 0)
    alpha[0] = alpha_local;
}


