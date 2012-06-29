// file automatically generated - do not edit!
// inplace solve A^T \\ B
// matrix layouts: A...col_major, B...row_major
__kernel void trans_upper_solve(
          __global const float * A,
          unsigned int A_rows,
          unsigned int A_cols,
          unsigned int A_internal_rows,
          unsigned int A_internal_cols,
          __global float * B,  
          unsigned int B_rows,
          unsigned int B_cols,
          unsigned int B_internal_rows,
          unsigned int B_internal_cols)
{ 
  float temp; 
  for (int row = A_rows-1; row > -1; --row) 
  { 
    barrier(CLK_GLOBAL_MEM_FENCE); 
    if (get_local_id(0) == 0) 
      B[row * B_internal_cols + get_group_id(0)] /= A[row + row*A_internal_cols]; 
    barrier(CLK_GLOBAL_MEM_FENCE); 
      temp = B[row * B_internal_cols + get_group_id(0)]; 
    //eliminate column of op(A) with index 'row' in parallel: 
    for  (int elim = get_local_id(0); elim < row; elim += get_local_size(0)) 
      B[elim * B_internal_cols + get_group_id(0)] -= temp * A[elim * A_internal_rows + row];
   }
}
