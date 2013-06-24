

__kernel void vec_mul_cpu(
          __global const unsigned int * row_indices,
          __global const unsigned int * column_indices,
          __global const float * elements,
          __global const float * vector,
          __global float * result,
          unsigned int size)
{
  unsigned int work_per_item = max((uint) (size / get_global_size(0)), (uint) 1);
  unsigned int row_start = get_global_id(0) * work_per_item;
  unsigned int row_stop  = min( (uint) ((get_global_id(0) + 1) * work_per_item), (uint) size);
  for (unsigned int row = row_start; row < row_stop; ++row)
  {
    float dot_prod = 0.0f;
    unsigned int row_end = row_indices[row+1];
    for (unsigned int i = row_indices[row]; i < row_end; ++i)
      dot_prod += elements[i] * vector[column_indices[i]];
    result[row] = dot_prod;
  }
}


