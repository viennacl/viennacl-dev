void assemble_r(__global float * gR, unsigned int row_n_r, unsigned int col_n_r, __global float * R,
        unsigned int row_n, unsigned int col_n)
{
  for(unsigned int i = 0; i < col_n; ++i){
     for(unsigned int j = 0; j < row_n; ++j){
    gR[i*row_n_r + j] = R[i*row_n + j ];
     }
  }
}

void assemble_r_u(__global float * gR,
          unsigned int row_n_r, unsigned int col_n_r, __global float * R_u, unsigned int row_n_u, unsigned int col_n_u,
          unsigned int col_n)
{
  for(unsigned int i = 0; i < col_n_u; ++i){
    for(unsigned int j = 0; j < col_n; ++j){
      gR[ (i+col_n)*row_n_r + j] = R_u[ i*row_n_u + j];
    }
  }
}


void assemble_r_u_u(__global float * gR,  unsigned int row_n_r, unsigned int col_n_r, __global float * R_u_u, unsigned int row_n_u_u,
          unsigned int col_n_u_u, unsigned int col_n)
{
  for(unsigned int i = 0; i < col_n_u_u; ++i){
    for(unsigned int j = 0; j < row_n_u_u; ++j){
      gR[(col_n+i)*row_n_r + j + col_n] = R_u_u[i*row_n_u_u + j];
    }
  }
}

void assemble_r_block(__global float * gR, unsigned int row_n_r, unsigned int col_n_r, __global float * R, unsigned int row_n,
        unsigned int col_n, __global float * R_u, unsigned int row_n_u, unsigned int col_n_u, __global float * R_u_u,
        unsigned int row_n_u_u, unsigned int col_n_u_u){
        assemble_r(gR, row_n_r, col_n_r, R, row_n, col_n);
        assemble_r_u(gR, row_n_r, col_n_r, R_u, row_n_u, col_n_u, col_n);
        assemble_r_u_u(gR, row_n_r, col_n_r, R_u_u, row_n_u_u, col_n_u_u, col_n);
}


__kernel void block_r_assembly(
                    __global float * R,
                  __global unsigned int * block_ind,
                __global unsigned int * matrix_dimensions,
                __global float * R_u,
                  __global unsigned int * block_ind_u,
                __global unsigned int * matrix_dimensions_u,
                __global float * R_u_u,
                __global unsigned int * block_ind_u_u,
                __global unsigned int * matrix_dimensions_u_u,
                __global float * g_R,
                __global unsigned int * block_ind_r,
                __global unsigned int * matrix_dimensions_r,
                __global unsigned int * g_is_update,
                    //__local  float * local_gR,
                    unsigned int  block_elems_num)
{
    for(unsigned int i  = get_global_id(0); i < block_elems_num; i += get_global_size(0)){
        if((matrix_dimensions[2*i] > 0) && (matrix_dimensions[2*i + 1] > 0) && g_is_update[i] > 0){

            assemble_r_block(g_R + block_ind_r[i], matrix_dimensions_r[2*i], matrix_dimensions_r[2*i + 1], R + block_ind[i], matrix_dimensions[2*i],
              matrix_dimensions[2*i + 1], R_u + block_ind_u[i], matrix_dimensions_u[2*i], matrix_dimensions_u[2*i + 1],
              R_u_u + block_ind_u_u[i], matrix_dimensions_u_u[2*i], matrix_dimensions_u_u[2*i + 1]);

        }
    }
}
