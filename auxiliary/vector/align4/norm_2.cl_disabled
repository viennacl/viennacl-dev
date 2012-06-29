
//helper:
void helper_norm24_float_parallel_reduction( __local float * tmp_buffer )
{
  for (unsigned int stride = get_global_size(0)/2; stride > 0; stride /= 2)
  {
    barrier(CLK_LOCAL_MEM_FENCE);
    if (get_global_id(0) < stride)
      tmp_buffer[get_global_id(0)] += tmp_buffer[get_global_id(0)+stride];
  }
};


////// norm_2
float float_vector4_norm_2_impl(
          __global const float4 * vec,
          unsigned int size,
          __local float * tmp_buffer)
{
  //step 1: fill buffer:
  float tmp = 0;
  float4 veci;
  unsigned int steps = size/4;
  for (unsigned int i = get_global_id(0); i < steps; i += get_global_size(0))
  {
    veci = vec[i];
    tmp += dot(veci, veci);
  }
  tmp_buffer[get_global_id(0)] = tmp;
  
  //step 2: parallel reduction:
  helper_norm24_float_parallel_reduction(tmp_buffer);
  
  return tmp_buffer[0];
};

__kernel void norm_2(
          __global float4 * vec,
          unsigned int size,
          __local float * tmp_buffer,
          global float * result) 
{ 
  float tmp = float_vector4_norm_2_impl(vec, size, tmp_buffer);
  if (get_global_id(0) == 0) *result = sqrt(tmp);
};


