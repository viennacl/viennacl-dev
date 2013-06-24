
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
  unsigned int row_gid = get_global_id(0) / get_local_size(0);
  unsigned int col_gid = get_global_id(0) % get_local_size(0);
  unsigned int lid = get_local_id(0);

  for (unsigned int row = row_gid; row < A_row_size; row += get_num_groups(0))
  {
    float dot_prod = 0;
    for (unsigned int col = col_gid; col < A_col_size; col+=get_local_size(0))
      dot_prod += A[(row * A_row_inc + A_row_start) * A_internal_cols + col * A_col_inc + A_col_start] * v[v_start + v_inc * col];
    work[lid] = dot_prod;

    for(unsigned int stride=get_local_size(0)/2 ; stride>0 ; stride>>=1){
      barrier(CLK_LOCAL_MEM_FENCE);
      if(lid < stride)
        work[lid] += work[lid+stride];
    }

    if(lid == 0)
      result[row * result_inc + result_start] = work[0];
  }
}
