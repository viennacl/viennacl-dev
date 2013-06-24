
void custom_dot_prod(__global float * A, unsigned int row_n, __local float * v, unsigned int ind, float *res){
            *res = 0.0;
            for(unsigned int j = ind; j < row_n; ++j){
                if(j == ind){
                    *res += v[j];
                }else{
                    *res += A[j + ind*row_n]*v[j];
                }
            }
        }

void apply_q_trans_vec(__global float * R, unsigned int row_n, unsigned int col_n, __global float * b_v, __local float * y){
            float inn_prod = 0;
            for(unsigned int i = 0; i < col_n; ++i){
                custom_dot_prod(R, row_n, y, i, &inn_prod);
                for(unsigned int j = i; j < row_n; ++j){
                    if(i == j){
                        y[j] -= b_v[ i]*inn_prod;
                    }
                    else{
                        y[j] -= b_v[ i]*inn_prod*R[ j + i*row_n];
                    }
                }
            }
        }

void q_mult(__global float * R, unsigned int row_n, unsigned int col_n, __global float * b_v, __local float * R_u, unsigned int col_n_u){
        for(unsigned int i = get_local_id(0); i < col_n_u; i+= get_local_size(0)){
          apply_q_trans_vec(R, row_n, col_n, b_v, R_u + row_n*i);
        }
}

void matrix_from_global_to_local(__global float* g_M, __local float* l_M, unsigned int row_n, unsigned int col_n, unsigned int mat_start_ind){
  for(unsigned int i = get_local_id(0); i < col_n; i+= get_local_size(0)){
    for(unsigned int j = 0; j < row_n; ++j){
      l_M[i*row_n + j] = g_M[mat_start_ind + i*row_n + j];
    }
  }
}

void matrix_from_local_to_global(__global float* g_M, __local float* l_M, unsigned int row_n, unsigned int col_n, unsigned int mat_start_ind){
  for(unsigned int i = get_local_id(0); i < col_n; i+= get_local_size(0)){
    for(unsigned int j = 0; j < row_n; ++j){
      g_M[mat_start_ind + i*row_n + j] = l_M[i*row_n + j];
    }
  }
}



__kernel void block_q_mult(__global float * global_R,
  __global unsigned int * block_ind,
  __global float * global_R_u,
  __global unsigned int *block_ind_u,
  __global float * b_v,
  __global unsigned int * start_bv_inds,
  __global unsigned int * matrix_dimensions,
  __global unsigned int * matrix_dimensions_u,
  __global unsigned int * g_is_update,
  __local  float * local_R_u,
    unsigned int  block_elems_num){
    for(unsigned int i  = get_group_id(0); i < block_elems_num; i += get_num_groups(0)){
          if((matrix_dimensions[2*i] > 0) && (matrix_dimensions[2*i + 1] > 0) && (g_is_update[i] > 0)){
        //matrix_from_global_to_local(R, local_buff_R, matrix_dimensions[2*i], matrix_dimensions[2*i + 1], start_matrix_inds[i]);
        matrix_from_global_to_local(global_R_u, local_R_u, matrix_dimensions_u[2*i], matrix_dimensions_u[2*i+ 1], block_ind_u[i]);
        barrier(CLK_LOCAL_MEM_FENCE);
              q_mult(global_R + block_ind[i], matrix_dimensions[2*i], matrix_dimensions[2*i + 1], b_v + start_bv_inds[i], local_R_u,
             matrix_dimensions_u[2*i + 1]);
        barrier(CLK_LOCAL_MEM_FENCE);
              matrix_from_local_to_global(global_R_u, local_R_u, matrix_dimensions_u[2*i], matrix_dimensions_u[2*i + 1], block_ind_u[i]);
          }
      }
}
