void assemble_upper_part(__global float * R_q,
            unsigned int row_n_q, unsigned int col_n_q, __global float * R_u,
            unsigned int row_n_u, unsigned int col_n_u,
            unsigned int col_n, unsigned int diff){
            for(unsigned int i = 0; i < col_n_q; ++i){
                for(unsigned int j = 0; j < diff; ++j){
          R_q[ i*row_n_q + j] = R_u[ i*row_n_u + j + col_n ];
                }
            }
        }


void assemble_lower_part(__global float * R_q, unsigned int row_n_q, unsigned int col_n_q, __global float * R_u_u,
             unsigned int row_n_u_u, unsigned int col_n_u_u,
             unsigned int diff){
  for(unsigned int i = 0; i < col_n_u_u; ++i){
    for(unsigned int j = 0; j < row_n_u_u; ++j){
      R_q[i*row_n_q + j + diff] = R_u_u[i*row_n_u_u + j];
    }
  }
}



void assemble_qr_block(__global float * R_q, unsigned int row_n_q, unsigned int col_n_q, __global float * R_u, unsigned int row_n_u,
            unsigned int col_n_u, __global float * R_u_u, unsigned int row_n_u_u, unsigned int col_n_u_u, unsigned int col_n){
            unsigned int diff = row_n_u - col_n;
            assemble_upper_part(R_q, row_n_q, col_n_q, R_u, row_n_u, col_n_u, col_n, diff);
            if(diff > 0){
              assemble_lower_part(R_q, row_n_q, col_n_q, R_u_u, row_n_u_u, col_n_u_u, diff);
            }
}

__kernel void block_qr_assembly(
          __global unsigned int * matrix_dimensions,
        __global float * R_u,
      __global unsigned int * block_ind_u,
      __global unsigned int * matrix_dimensions_u,
      __global float * R_u_u,
        __global unsigned int * block_ind_u_u,
      __global unsigned int * matrix_dimensions_u_u,
      __global float * R_q,
      __global unsigned int * block_ind_q,
      __global unsigned int * matrix_dimensions_q,
      __global unsigned int * g_is_update,
          //__local  float * local_R_q,
            unsigned int  block_elems_num)
{
    for(unsigned int i  = get_global_id(0); i < block_elems_num; i += get_global_size(0)){
        if((matrix_dimensions[2*i] > 0) && (matrix_dimensions[2*i + 1] > 0) && g_is_update[i] > 0){
      //
            assemble_qr_block(R_q + block_ind_q[i], matrix_dimensions_q[2*i], matrix_dimensions_q[2*i + 1], R_u + block_ind_u[i], matrix_dimensions_u[2*i],
              matrix_dimensions_u[2*i + 1], R_u_u + block_ind_u_u[i], matrix_dimensions_u_u[2*i], matrix_dimensions_u_u[2*i + 1], matrix_dimensions[2*i + 1]);

        }
    }
}
