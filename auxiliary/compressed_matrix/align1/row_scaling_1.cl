

__kernel void row_scaling_1(
          __global const unsigned int * row_indices,
          __global const unsigned int * column_indices, 
          __global const float * elements,
          __global float * diag_M_inv,
          unsigned int size) 
{ 
  for (unsigned int row = get_global_id(0); row < size; row += get_global_size(0))
  {
    float dot_prod = 0.0f;
    unsigned int row_end = row_indices[row+1];
    for (unsigned int i = row_indices[row]; i < row_end; ++i)
      dot_prod += fabs(elements[i]);
    diag_M_inv[row] = 1.0f / dot_prod;
  }
}


