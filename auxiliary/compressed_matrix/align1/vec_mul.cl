

__kernel void vec_mul(
          __global const unsigned int * row_indices,
          __global const unsigned int * column_indices,
          __global const float * elements,
          __global const float * x,
          uint4 layout_x,
          __global float * result,
          uint4 layout_result)
{
  for (unsigned int row = get_global_id(0); row < layout_result.z; row += get_global_size(0))
  {
    float dot_prod = 0;
    unsigned int row_end = row_indices[row+1];
    for (unsigned int i = row_indices[row]; i < row_end; ++i)
      dot_prod += elements[i] * x[column_indices[i] * layout_x.y + layout_x.x];
    result[row * layout_result.y + layout_result.x] = dot_prod;
  }
}


