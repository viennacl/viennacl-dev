//index_norm_inf:
unsigned int float_vector1_index_norm_inf_impl(
          __global const float * vec,
          unsigned int start1,
          unsigned int inc1,
          unsigned int size1,
          __local float * float_buffer,
          __local unsigned int * index_buffer)
{
  //step 1: fill buffer:
  float cur_max = 0.0f;
  float tmp;
  for (unsigned int i = get_global_id(0); i < size1; i += get_global_size(0))
  {
    tmp = fabs(vec[i*inc1+start1]);
    if (cur_max < tmp)
    {
      float_buffer[get_global_id(0)] = tmp;
      index_buffer[get_global_id(0)] = i;
      cur_max = tmp;
    }
  }

  //step 2: parallel reduction:
  for (unsigned int stride = get_global_size(0)/2; stride > 0; stride /= 2)
  {
    barrier(CLK_LOCAL_MEM_FENCE);
    if (get_global_id(0) < stride)
    {
      //find the first occurring index
      if (float_buffer[get_global_id(0)] < float_buffer[get_global_id(0)+stride])
      {
        index_buffer[get_global_id(0)] = index_buffer[get_global_id(0)+stride];
        float_buffer[get_global_id(0)] = float_buffer[get_global_id(0)+stride];
      }

      //index_buffer[get_global_id(0)] = float_buffer[get_global_id(0)] < float_buffer[get_global_id(0)+stride] ? index_buffer[get_global_id(0)+stride] : index_buffer[get_global_id(0)];
      //float_buffer[get_global_id(0)] = max(float_buffer[get_global_id(0)], float_buffer[get_global_id(0)+stride]);
    }
  }

  return index_buffer[0];
}

__kernel void index_norm_inf(
          __global float * vec,
          unsigned int start1,
          unsigned int inc1,
          unsigned int size1,
          __local float * float_buffer,
          __local unsigned int * index_buffer,
          __global unsigned int * result)
{
  float_buffer[get_global_id(0)] = 0;
  index_buffer[get_global_id(0)] = 0;
  unsigned int tmp = float_vector1_index_norm_inf_impl(vec, start1, inc1, size1, float_buffer, index_buffer);
  if (get_global_id(0) == 0) *result = tmp;
}


