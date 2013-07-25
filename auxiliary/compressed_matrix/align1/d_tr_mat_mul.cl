
__kernel void d_tr_mat_mul(
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
          unsigned int result_internal_cols ) {

  // split work rows (sparse matrix rows) to thread groups 
  for (unsigned int row = get_group_id(0); row < result_row_size; row += get_num_groups(0)) {

    unsigned int row_start = sp_mat_row_indices[row];
    unsigned int row_end = sp_mat_row_indices[row+1];

    // split result cols between threads in a thread group
    for ( unsigned int col = get_local_id(0); col < result_col_size; col += get_local_size(0) ) {

      float r = 0;
      // TODO check to see if uncoalesced read from d_mat can be avoided
      // possible solution is that threads in a group unroll the k loop (danger: lots of threads idle in many blocks)
      for (unsigned int k = row_start; k < row_end; k ++) {

        unsigned int j = sp_mat_col_indices[k];
        float x = sp_mat_elements[k];

        float y = d_mat[ (d_mat_row_start + col * d_mat_row_inc) * d_mat_internal_cols +
                          d_mat_col_start + j * d_mat_col_inc ];
        r += x * y;
      }

      result[ (result_row_start + row * result_row_inc) * result_internal_cols + 
        result_col_start + col * result_col_inc ] = r;
    }
  }

}
