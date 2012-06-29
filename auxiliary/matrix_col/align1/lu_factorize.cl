
__kernel void lu_factorize(
          __global float * matrix,
          unsigned int matrix_rows,
          unsigned int matrix_cols,
          unsigned int matrix_internal_rows,
          unsigned int matrix_internal_cols) 
{ 
  float temp;
  for (unsigned int i=1; i<matrix_rows; ++i)
  {
    for (unsigned int k=0; k<i; ++k)
    {
      if (get_global_id(0) == 0)
        matrix[i + k*matrix_internal_rows] /= matrix[k + k*matrix_internal_rows];

      barrier(CLK_GLOBAL_MEM_FENCE);
      temp = matrix[i + k*matrix_internal_rows];
      
      //parallel subtraction:
      for (unsigned int j=k+1 + get_global_id(0); j<matrix_cols; j += get_global_size(0))
        matrix[i + j*matrix_internal_rows] -= temp * matrix[k + j*matrix_internal_rows];
    }
  }
} 


