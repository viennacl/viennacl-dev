

__kernel void jacobi(
 __global const unsigned int * row_indices,
 __global const unsigned int * column_indices,
 __global const float * elements,
 float weight,
 __global const float * old_result,
 __global float * new_result,
 __global const float * rhs,
 unsigned int size)
 {
  float sum, diag=1;
  int col;
  for (unsigned int i = get_global_id(0); i < size; i += get_global_size(0))
  {
    sum = 0;
    for (unsigned int j = row_indices[i]; j<row_indices[i+1]; j++)
    {
      col = column_indices[j];
      if (i == col)
  diag = elements[j];
      else
  sum += elements[j] * old_result[col];
    }
      new_result[i] = weight * (rhs[i]-sum) / diag + (1-weight) * old_result[i];
   }
 }
