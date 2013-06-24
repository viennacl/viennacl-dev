
__kernel void vec_mul(
          __global const float * A,
          unsigned int A_row_start,
          unsigned int A_col_start,
          unsigned int A_row_inc,
          unsigned int A_col_inc,
          unsigned int A_row_size,
          unsigned int A_col_size,
          unsigned int A_internal_rows,
          unsigned int A_internal_cols,
          __global const float * v,
          unsigned int v_start,
          unsigned int v_inc,
          unsigned int v_size,
          __global float * result,
          unsigned int result_start,
          unsigned int result_inc,
          unsigned int result_size,
          __local float * work)
{
  for (unsigned int row = get_global_id(0); row < A_row_size; row += get_global_size(0))
  {
    float dot_prod = 0;
    for (unsigned int col = 0; col < A_col_size; ++col)
      dot_prod += A[(row * A_row_inc + A_row_start) + (col * A_col_inc + A_col_start) * A_internal_rows] * v[v_start + v_inc * col];
    result[row * result_inc + result_start] = dot_prod;
  }
}
