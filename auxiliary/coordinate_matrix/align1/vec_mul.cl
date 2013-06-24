

__kernel void vec_mul(
          __global const uint2 * coords, //(row_index, column_index)
          __global const float * elements,
          __global const uint  * group_boundaries,
          __global const float * vector,
          __global float * result,
          __local unsigned int * shared_rows,
          __local float * inter_results)
{
  uint2 tmp;
  float val;
  uint last_index  = get_local_size(0) - 1;
  uint group_start = group_boundaries[get_group_id(0)];
  uint group_end   = group_boundaries[get_group_id(0) + 1];
  uint k_end = (group_end > group_start) ? 1 + (group_end - group_start - 1) / get_local_size(0) : 0;   // -1 in order to have correct behavior if group_end - group_start == j * get_local_size(0)

  uint local_index = 0;

  for (uint k = 0; k < k_end; ++k)
  {
    local_index = group_start + k * get_local_size(0) + get_local_id(0);

    tmp = (local_index < group_end) ? coords[local_index] : (uint2) 0;
    val = (local_index < group_end) ? elements[local_index] * vector[tmp.y] : 0;

    //check for carry from previous loop run:
    if (get_local_id(0) == 0 && k > 0)
    {
      if (tmp.x == shared_rows[last_index])
        val += inter_results[last_index];
      else
        result[shared_rows[last_index]] = inter_results[last_index];
    }

    //segmented parallel reduction begin
    barrier(CLK_LOCAL_MEM_FENCE);
    shared_rows[get_local_id(0)] = tmp.x;
    inter_results[get_local_id(0)] = val;
    float left = 0;
    barrier(CLK_LOCAL_MEM_FENCE);

    for (unsigned int stride = 1; stride < get_local_size(0); stride *= 2)
    {
      left = (get_local_id(0) >= stride && tmp.x == shared_rows[get_local_id(0) - stride]) ? inter_results[get_local_id(0) - stride] : 0;
      barrier(CLK_LOCAL_MEM_FENCE);
      inter_results[get_local_id(0)] += left;
      barrier(CLK_LOCAL_MEM_FENCE);
    }
    //segmented parallel reduction end

    if (get_local_id(0) != last_index &&
        shared_rows[get_local_id(0)] != shared_rows[get_local_id(0) + 1] &&
        inter_results[get_local_id(0)] != 0)
    {
      result[tmp.x] = inter_results[get_local_id(0)];
    }

    barrier(CLK_LOCAL_MEM_FENCE);
  } //for k

  if (get_local_id(0) == last_index && inter_results[last_index] != 0)
    result[tmp.x] = inter_results[last_index];
}
