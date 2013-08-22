
__kernel void d_tr_mat_mul(
    __global const unsigned int * sp_mat_coords,
    __global const float * sp_mat_elems,
    unsigned int sp_mat_row_num,
    unsigned int sp_mat_col_num,
    unsigned int sp_mat_internal_row_num,
    unsigned int sp_mat_items_per_row,
    unsigned int sp_mat_aligned_items_per_row,
    __global const float* d_mat,
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
    unsigned int result_internal_cols) {

    uint glb_id = get_global_id(0);
    uint glb_sz = get_global_size(0);

    for( uint rc = glb_id; rc < (sp_mat_row_num * d_mat_row_size); rc += glb_sz) {
      uint row = rc % sp_mat_row_num;
      uint col = rc / sp_mat_row_num;

      uint offset = row;
      float r = (float)0;

      for( uint k = 0; k < sp_mat_items_per_row; k++, offset += sp_mat_internal_row_num) {

        uint j = sp_mat_coords[offset];
        float x = sp_mat_elems[offset];

        if(x != (float)0) {
          float y = d_mat[ (d_mat_row_start + col * d_mat_row_inc) * d_mat_internal_cols +
                            d_mat_col_start + j * d_mat_col_inc ];
          r += x*y;
        }
      }
      result[ (result_row_start + row * result_row_inc) * result_internal_cols +
               result_col_start + col * result_col_inc ] = r;
    }
}

