
void custom_dot_prod_ls(__global float * A, unsigned int row_n, __global float * v, unsigned int ind, float *res){
            *res = 0.0;
            for(unsigned int j = ind; j < row_n; ++j){
                if(j == ind){
                    *res += v[ j];
                }else{
                    *res += A[ j + ind*row_n]*v[ j];
                }
            }
        }

void backwardSolve(__global float * R,  unsigned int row_n, unsigned int col_n, __global float * y, __global float * x){
  for (int i = col_n-1; i >= 0 ; i--) {
    x[ i] = y[ i];
    for (int j = i+1; j < col_n; ++j) {
      x[ i] -= R[ i + j*row_n]*x[ j];
    }
    x[i] /= R[ i + i*row_n];
  }

}


void apply_q_trans_vec_ls(__global float * R, unsigned int row_n, unsigned int col_n, __global const float * b_v,  __global float * y){
            float inn_prod = 0;
            for(unsigned int i = 0; i < col_n; ++i){
                custom_dot_prod_ls(R, row_n, y, i, &inn_prod);
                for(unsigned int j = i; j < row_n; ++j){
                    if(i == j){
                        y[ j] -= b_v[ i]*inn_prod;
                    }
                    else{
                        y[j] -= b_v[ i]*inn_prod*R[ j +i*row_n];
                    }
                }
                //std::cout<<y<<std::endl;
            }
        }

void ls(__global float * R, unsigned int row_n, unsigned int col_n, __global float * b_v, __global float * m_v, __global float * y_v){

  apply_q_trans_vec_ls(R, row_n, col_n, b_v, y_v);
  //m_new - is m_v now
  backwardSolve(R, row_n, col_n, y_v, m_v);
}

__kernel void block_least_squares(
          __global float * global_R,
      __global unsigned int * block_ind,
          __global float * b_v,
        __global unsigned int * start_bv_inds,
      __global float * m_v,
      __global float * y_v,
      __global unsigned int * start_y_inds,
        __global unsigned int * matrix_dimensions,
        __global unsigned int * g_is_update,
          //__local  float * local_R,
            unsigned int  block_elems_num)
{
    for(unsigned int i  = get_global_id(0); i < block_elems_num; i += get_global_size(0)){
        if((matrix_dimensions[2*i] > 0) && (matrix_dimensions[2*i + 1] > 0) && g_is_update[i] > 0){

            ls(global_R + block_ind[i], matrix_dimensions[2*i], matrix_dimensions[2*i + 1], b_v +start_bv_inds[i], m_v + start_bv_inds[i], y_v + start_y_inds[i] );

        }
    }
}
