



// compute y in Ly = z for incomplete LU factorizations of a sparse matrix in compressed format
__kernel void trans_unit_lu_forward_slow(
          __global const unsigned int * row_indices,
          __global const unsigned int * column_indices,
          __global const float * elements,
          __global float * vector,
          unsigned int size)
{
  for (unsigned int row = 0; row < size; ++row)
  {
    float result_entry = vector[row];

    unsigned int row_start = row_indices[row];
    unsigned int row_stop  = row_indices[row + 1];
    for (unsigned int entry_index = row_start + get_local_id(0); entry_index < row_stop; entry_index += get_local_size(0))
    {
      unsigned int col_index = column_indices[entry_index];
      if (col_index > row)
        vector[col_index] -= result_entry * elements[entry_index];
    }

    barrier(CLK_GLOBAL_MEM_FENCE);
  }
}


