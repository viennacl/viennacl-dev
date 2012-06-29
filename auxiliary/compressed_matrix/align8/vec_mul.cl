

__kernel void vec_mul(
          __global const unsigned int * row_indices,
          __global const uint8 * column_indices, 
          __global const float8 * elements,
          __global const float * vector,  
          __global float * result,
          unsigned int size)
{ 
  float dot_prod;
  unsigned int start, next_stop;
  uint8 col_idx;
  float8 tmp_vec;
  float8 tmp_entries;

  for (unsigned int row = get_global_id(0); row < size; row += get_global_size(0))
  {
    dot_prod = 0.0f;
    start = row_indices[row] / 8;
    next_stop = row_indices[row+1] / 8;

    for (unsigned int i = start; i < next_stop; ++i)
    {
      col_idx = column_indices[i];

      tmp_entries = elements[i];
      tmp_vec.s0 = vector[col_idx.s0];
      tmp_vec.s1 = vector[col_idx.s1];
      tmp_vec.s2 = vector[col_idx.s2];
      tmp_vec.s3 = vector[col_idx.s3];
      tmp_vec.s4 = vector[col_idx.s4];
      tmp_vec.s5 = vector[col_idx.s5];
      tmp_vec.s6 = vector[col_idx.s6];
      tmp_vec.s7 = vector[col_idx.s7];

      dot_prod += dot(tmp_entries.lo, tmp_vec.lo);
      dot_prod += dot(tmp_entries.hi, tmp_vec.hi);
    }
    result[row] = dot_prod;
  }
}
