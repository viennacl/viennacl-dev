
__kernel void scaled_rank1_update(
          __global float * matrix,
          unsigned int matrix_rows,
          unsigned int matrix_cols,
          unsigned int matrix_internal_rows,
          unsigned int matrix_internal_cols,
          float val,
          __global const float * vector1,  
          __global const float * vector2) 
{ 
  float tmp;

  for (unsigned int row = get_global_id(0); row < matrix_rows; row += get_global_size(0))
  {
    tmp = val * vector1[row];
    for (unsigned int col = 0; col < matrix_cols; ++col)
      matrix[row + col*matrix_internal_rows] += tmp * vector2[col];
  }
}


