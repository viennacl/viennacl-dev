
__kernel void d_mat_mul(
          __global const unsigned int * sp_mat_row_indices,
          __global const unsigned int * sp_mat_col_indices, 
          __global const float * sp_mat_elements,
          __global const float * d_mat,
          unsigned int d_mat_row_start,
          unsigned int d_mat_col_start,
          unsigned int d_mat_row_inc,
          unsigned int d_mat_col_inc,
          unsigned int d_mat_row_size,
          unsigned int d_mat_col_size,
          unsigned int d_mat_internal_rows,
          unsigned int d_mat_internal_cols,
          __global float * result,
          unsigned int result_row_start,
          unsigned int result_col_start,
          unsigned int result_row_inc,
          unsigned int result_col_inc,
          unsigned int result_row_size,
          unsigned int result_col_size,
          unsigned int result_internal_rows,
          unsigned int result_internal_cols,
          __local float* sh_elements,
          __local unsigned int* sh_col_indices ) {

  // split work rows (sparse matrix rows) to thread groups 
  for (unsigned int row = get_group_id(0); row < result_row_size; row += get_num_groups(0)) {

    unsigned int row_start = sp_mat_row_indices[row];
    unsigned int row_end = sp_mat_row_indices[row+1];

    // load work rows to shared memory
    for (unsigned int x = get_local_id(0); x < (row_end-row_start); x += get_local_size(0)) {
        sh_elements[x] = sp_mat_elements[x + row_start];
        sh_col_indices[x] = sp_mat_col_indices[x + row_start];
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    // split result cols between threads in a thread group
    for ( unsigned int col = get_local_id(0); col < result_col_size; col += get_local_size(0) ) {

      float temp = 0;
      for (unsigned int k = 0; k < (row_end - row_start); k ++) {
        temp += sh_elements[k] * 
          d_mat[ (d_mat_row_start + sh_col_indices[k] * d_mat_row_inc) * d_mat_internal_cols + 
                 d_mat_col_start + col * d_mat_col_inc ];
      }

      result[ (result_row_start + row * result_row_inc) * result_internal_cols + 
        result_col_start + col * result_col_inc ] = temp;
    }
  }

}

