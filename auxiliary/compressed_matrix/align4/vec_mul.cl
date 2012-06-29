

__kernel void vec_mul(
          __global const unsigned int * row_indices,
          __global const uint4 * column_indices, 
          __global const float4 * elements,
          __global const float * vector,  
          __global float * result,
          unsigned int size)
{ 
  float dot_prod;
  unsigned int start, next_stop;
  uint4 col_idx;
  float4 tmp_vec;
  float4 tmp_entries;

  for (unsigned int row = get_global_id(0); row < size; row += get_global_size(0))
  {
    dot_prod = 0.0f;
    start = row_indices[row] / 4;
    next_stop = row_indices[row+1] / 4;

    for (unsigned int i = start; i < next_stop; ++i)
    {
      col_idx = column_indices[i];

      tmp_entries = elements[i];
      tmp_vec.x = vector[col_idx.x];
      tmp_vec.y = vector[col_idx.y];
      tmp_vec.z = vector[col_idx.z];
      tmp_vec.w = vector[col_idx.w];

      dot_prod += dot(tmp_entries, tmp_vec);
    }
    result[row] = dot_prod;
  }
}
