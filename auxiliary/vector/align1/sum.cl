
// sums the array 'vec1' and writes to result. Makes use of a single work-group only.
__kernel void sum(
          __global float * vec1,
          unsigned int start1,
          unsigned int inc1,
          unsigned int size1,
          unsigned int option, //0: use fmax, 1: just sum, 2: sum and return sqrt of sum
          __local float * tmp_buffer,
          __global float * result)
{
  float thread_sum = 0;
  for (unsigned int i = get_local_id(0); i<size1; i += get_local_size(0))
  {
    if (option > 0)
      thread_sum += vec1[i*inc1+start1];
    else
      thread_sum = fmax(thread_sum, fabs(vec1[i*inc1+start1]));
  }

  tmp_buffer[get_local_id(0)] = thread_sum;

  for (unsigned int stride = get_local_size(0)/2; stride > 0; stride /= 2)
  {
    if (get_local_id(0) < stride)
    {
      if (option > 0)
        tmp_buffer[get_local_id(0)] += tmp_buffer[get_local_id(0) + stride];
      else
        tmp_buffer[get_local_id(0)] = fmax(tmp_buffer[get_local_id(0)], tmp_buffer[get_local_id(0) + stride]);
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }

  if (get_global_id(0) == 0)
  {
    if (option == 2)
      *result = sqrt(tmp_buffer[0]);
    else
      *result = tmp_buffer[0];
  }
}

