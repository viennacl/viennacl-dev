void assemble_bv(__global float * g_bv_r, __global float * g_bv, unsigned int col_n){
  for(unsigned int i = 0; i < col_n; ++i){
    g_bv_r[i] = g_bv[ i];
  }
}

void assemble_bv_block(__global float * g_bv_r, __global float * g_bv, unsigned int col_n,
             __global float * g_bv_u, unsigned int col_n_u)
{
  assemble_bv(g_bv_r, g_bv, col_n);
  assemble_bv(g_bv_r + col_n, g_bv_u, col_n_u);

}

__kernel void block_bv_assembly(__global float * g_bv,
            __global unsigned int * start_bv_ind,
            __global unsigned int * matrix_dimensions,
            __global float * g_bv_u,
            __global unsigned int * start_bv_u_ind,
            __global unsigned int * matrix_dimensions_u,
            __global float * g_bv_r,
            __global unsigned int * start_bv_r_ind,
            __global unsigned int * matrix_dimensions_r,
            __global unsigned int * g_is_update,
            //__local  float * local_gb,
            unsigned int  block_elems_num)
{
  for(unsigned int i  = get_global_id(0); i < block_elems_num; i += get_global_size(0)){
    if((matrix_dimensions[2*i] > 0) && (matrix_dimensions[2*i + 1] > 0) && g_is_update[i] > 0){
      assemble_bv_block(g_bv_r + start_bv_r_ind[i], g_bv + start_bv_ind[i], matrix_dimensions[2*i + 1], g_bv_u + start_bv_u_ind[i], matrix_dimensions_u[2*i + 1]);
    }
  }
}
