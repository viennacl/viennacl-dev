

__kernel void diagonal(
          __global const unsigned int * row_indices,
          __global const unsigned int * column_indices, 
          __global const float * elements,
          __global float * result,
          unsigned int size) 
{ 
  for (unsigned int row = get_global_id(0); row < size; row += get_global_size(0))
  {
    float diag = 0.0f;
    unsigned int row_end = row_indices[row+1];
    for (unsigned int i = row_indices[row]; i < row_end; ++i)
    {
      if (column_indices[i] == row)
      {
        diag = elements[i];
        break;
      }
    }
    result[row] = diag;
  }
}


