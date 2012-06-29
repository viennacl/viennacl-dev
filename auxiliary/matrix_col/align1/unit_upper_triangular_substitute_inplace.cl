

__kernel void unit_upper_triangular_substitute_inplace( 
          __global const float * matrix, 
          unsigned int matrix_rows,
          unsigned int matrix_cols,
          unsigned int matrix_internal_rows,
          unsigned int matrix_internal_cols,
          __global float * vector) 
{ 
  float temp; 
  for (int row = matrix_rows-1; row > -1; --row) 
  { 
    barrier(CLK_GLOBAL_MEM_FENCE); 
    
    temp = vector[row]; 
    //eliminate column with index 'row' in parallel: 
    for  (int elim = get_global_id(0); elim < row; elim += get_global_size(0)) 
      vector[elim] -= temp * matrix[elim + row  * matrix_internal_rows]; 
  } 
   
}

