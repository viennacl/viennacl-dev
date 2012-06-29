// file automatically generated - do not edit!
// inplace solve A^T \\ B^T
// matrix layouts: A...row_major, B...col_major
__kernel void trans_unit_lower_trans_solve(
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
  for (int row = 0; row < A_rows; ++row) 
  { 
    barrier(CLK_GLOBAL_MEM_FENCE); 
      temp = B[row * B_internal_rows + get_group_id(0)]; 
    //eliminate column of op(A) with index 'row' in parallel: 
    for  (int elim = row + get_local_id(0) + 1; elim < A_rows; elim += get_local_size(0)) 
      B[elim * B_internal_rows + get_group_id(0)] -= temp * A[elim + row * A_internal_cols];
   }
}
