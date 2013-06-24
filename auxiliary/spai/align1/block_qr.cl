void dot_prod(__local const float* A, unsigned int n, unsigned int beg_ind, float* res){
    *res = 0;
    for(unsigned int i = beg_ind; i < n; ++i){
        *res += A[(beg_ind-1)*n + i]*A[(beg_ind-1)*n + i];
    }
}

void vector_div(__global float* v, unsigned int beg_ind, float b, unsigned int n){
    for(unsigned int i = beg_ind; i < n; ++i){
        v[i] /= b;
    }
}

void copy_vector(__local const float* A, __global float* v, const unsigned int beg_ind, const unsigned int n){
    for(unsigned int i = beg_ind; i < n; ++i){
        v[i] = A[(beg_ind-1)*n + i];
    }
}


void householder_vector(__local const float* A, unsigned int j, unsigned int n, __global float* v, __global float* b){
    float sg;
    dot_prod(A, n, j+1, &sg);
    copy_vector(A, v, j+1, n);
    float mu;
    v[j] = 1.0;
    //print_contigious_vector(v, v_start_ind, n);
    if(sg == 0){
        *b = 0;
    }
    else{
        mu = sqrt(A[j*n + j]*A[ j*n + j] + sg);
        if(A[ j*n + j] <= 0){
            v[j] = A[ j*n + j] - mu;
        }else{
            v[j] = -sg/(A[ j*n + j] + mu);
        }
    *b = 2*(v[j]*v[j])/(sg + v[j]*v[j]);
        //*b = (2*v[j]*v[j])/(sg + (v[j])*(v[j]));
        vector_div(v, j, v[j], n);
        //print_contigious_vector(v, v_start_ind, n);
    }
}

void custom_inner_prod(__local const float* A, __global float* v, unsigned int col_ind, unsigned int row_num, unsigned int start_ind, float* res){
    for(unsigned int i = start_ind; i < row_num; ++i){
        *res += A[col_ind*row_num + i]*v[i];
    }
}
//
void apply_householder_reflection(__local float* A,  unsigned int row_n, unsigned int col_n, unsigned int iter_cnt, __global float* v, float b){
    float in_prod_res;
    for(unsigned int i= iter_cnt + get_local_id(0); i < col_n; i+=get_local_size(0)){
        in_prod_res = 0.0;
        custom_inner_prod(A, v, i, row_n, iter_cnt, &in_prod_res);
        for(unsigned int j = iter_cnt; j < row_n; ++j){
            A[ i*row_n + j] -= b*in_prod_res* v[j];
        }
    }

}

void store_householder_vector(__local float* A,  unsigned int ind, unsigned int n, __global float* v){
    for(unsigned int i = ind; i < n; ++i){
        A[ (ind-1)*n + i] = v[i];
    }
}

void single_qr( __local float* R, __global unsigned int* matrix_dimensions, __global float* b_v, __global float* v, unsigned int matrix_ind){
            //matrix_dimensions[0] - number of rows
              //matrix_dimensions[1] - number of columns
  unsigned int col_n = matrix_dimensions[2*matrix_ind + 1];
  unsigned int row_n = matrix_dimensions[2*matrix_ind];

  if((col_n == row_n)&&(row_n == 1)){
    b_v[0] = 0.0;
      return;
  }
       for(unsigned int i = 0; i < col_n; ++i){
        if(get_local_id(0) == 0){
                  householder_vector(R, i, row_n, v, b_v + i);
        }
        barrier(CLK_LOCAL_MEM_FENCE);
                apply_householder_reflection(R, row_n, col_n, i, v, b_v[i]);
                barrier(CLK_LOCAL_MEM_FENCE);
        if(get_local_id(0) == 0){
                  if(i < matrix_dimensions[2*matrix_ind]){
                      store_householder_vector(R, i+1, row_n, v);
                  }
        }
           }
}

void matrix_from_global_to_local_qr(__global float* g_M, __local float* l_M, unsigned int row_n, unsigned int col_n, unsigned int mat_start_ind){
  for(unsigned int i = get_local_id(0); i < col_n; i+= get_local_size(0)){
    for(unsigned int j = 0; j < row_n; ++j){
      l_M[i*row_n + j] = g_M[mat_start_ind + i*row_n + j];
    }
  }
}
void matrix_from_local_to_global_qr(__global float* g_M, __local float* l_M, unsigned int row_n, unsigned int col_n, unsigned int mat_start_ind){
  for(unsigned int i = get_local_id(0); i < col_n; i+= get_local_size(0)){
    for(unsigned int j = 0; j < row_n; ++j){
      g_M[mat_start_ind + i*row_n + j] = l_M[i*row_n + j];
    }
  }
}


__kernel void block_qr(
      __global float* R,
      __global unsigned int* matrix_dimensions,
      __global float* b_v,
      __global float* v,
      __global unsigned int* start_matrix_inds,
      __global unsigned int* start_bv_inds,
      __global unsigned int* start_v_inds,
      __global unsigned int * g_is_update,
      __local float* local_buff_R,
      unsigned int block_elems_num){
    for(unsigned int i  = get_group_id(0); i < block_elems_num; i += get_num_groups(0)){
        if((matrix_dimensions[2*i] > 0) && (matrix_dimensions[2*i + 1] > 0) && g_is_update[i] > 0){
      matrix_from_global_to_local_qr(R, local_buff_R, matrix_dimensions[2*i], matrix_dimensions[2*i + 1], start_matrix_inds[i]);
      barrier(CLK_LOCAL_MEM_FENCE);
            single_qr(local_buff_R, matrix_dimensions, b_v + start_bv_inds[i], v + start_v_inds[i], i);
      barrier(CLK_LOCAL_MEM_FENCE);
            matrix_from_local_to_global_qr(R, local_buff_R, matrix_dimensions[2*i], matrix_dimensions[2*i + 1], start_matrix_inds[i]);
        }
    }
}
