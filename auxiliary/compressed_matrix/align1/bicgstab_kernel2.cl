void helper_bicgstab_kernel2_parallel_reduction( __local float * tmp_buffer )
{
  for (unsigned int stride = get_local_size(0)/2; stride > 0; stride /= 2)
  {
    barrier(CLK_LOCAL_MEM_FENCE);
    if (get_local_id(0) < stride)
      tmp_buffer[get_local_id(0)] += tmp_buffer[get_local_id(0)+stride];
  }
}

//////// inner products:
float bicgstab_kernel2_inner_prod(
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
  
  helper_bicgstab_kernel2_parallel_reduction(tmp_buffer);

  barrier(CLK_LOCAL_MEM_FENCE);

  return tmp_buffer[0];
}


__kernel void bicgstab_kernel2(
          __global const float * tmp0,
          __global const float * tmp1,
          __global const float * r0star, 
          __global const float * s, 
          __global float * p, 
          __global float * result,
          __global float * residual,
          __global const float * alpha,
          __global float * ip_rr0star,
          __global float * error_estimate,
          __local float * tmp_buffer,
          unsigned int size) 
{ 
  float omega_local = bicgstab_kernel2_inner_prod(tmp1, s, size, tmp_buffer) / bicgstab_kernel2_inner_prod(tmp1, tmp1, size, tmp_buffer);
  float alpha_local = alpha[0];
  
  //result += alpha * p + omega * s;
  for (unsigned int i = get_local_id(0); i < size; i += get_local_size(0))
    result[i] += alpha_local * p[i] + omega_local * s[i];

  //residual = s - omega * tmp1;
  for (unsigned int i = get_local_id(0); i < size; i += get_local_size(0))
    residual[i] = s[i] - omega_local * tmp1[i];

  //new_ip_rr0star = viennacl::linalg::inner_prod(residual, r0star);
  float new_ip_rr0star = bicgstab_kernel2_inner_prod(residual, r0star, size, tmp_buffer);
  float beta = (new_ip_rr0star / ip_rr0star[0]) * (alpha_local / omega_local);
  
  //p = residual + beta * (p - omega*tmp0);
  for (unsigned int i = get_local_id(0); i < size; i += get_local_size(0))
    p[i] = residual[i] + beta * (p[i] - omega_local * tmp0[i]);

  //compute norm of residual:
  float new_error_estimate = bicgstab_kernel2_inner_prod(residual, residual, size, tmp_buffer);

  barrier(CLK_GLOBAL_MEM_FENCE);

  //update values:
  if (get_global_id(0) == 0)
  {
    error_estimate[0] = new_error_estimate;
    ip_rr0star[0] = new_ip_rr0star;
  }
}


