
__kernel void clear( // A <- 0
          __global float * A,
          unsigned int A_row_start,
          unsigned int A_col_start,
          unsigned int A_row_inc,
          unsigned int A_col_inc,
          unsigned int A_row_size,
          unsigned int A_col_size,
          unsigned int A_internal_rows,
          unsigned int A_internal_cols) 
{ 
  for (unsigned int i = get_global_id(0); i < A_row_size; i += get_global_size(0))
    for (unsigned int j = get_global_id(1); j < A_col_size; j += get_global_size(1))
      A[i * A_row_inc + A_row_start + (j * A_col_inc + A_col_start) * A_internal_rows] = 0;
}
