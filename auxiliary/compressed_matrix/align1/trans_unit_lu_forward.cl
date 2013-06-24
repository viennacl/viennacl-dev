



// compute y in Ly = z for incomplete LU factorizations of a sparse matrix in compressed format
__kernel void trans_unit_lu_forward(
          __global const unsigned int * row_indices,
          __global const unsigned int * column_indices,
          __global const float * elements,
          __global float * vector,
          unsigned int size)
{
  __local unsigned int row_index_lookahead[256];
  __local unsigned int row_index_buffer[256];

  unsigned int row_index;
  unsigned int col_index;
  float matrix_entry;
  unsigned int nnz = row_indices[size];
  unsigned int row_at_window_start = 0;
  unsigned int row_at_window_end = 0;
  unsigned int loop_end = ( (nnz - 1) / get_local_size(0) + 1) * get_local_size(0);

  for (unsigned int i = get_local_id(0); i < loop_end; i += get_local_size(0))
  {
    col_index    = (i < nnz) ? column_indices[i] : 0;
    matrix_entry = (i < nnz) ? elements[i]       : 0;
    row_index_lookahead[get_local_id(0)] = (row_at_window_start + get_local_id(0) < size) ? row_indices[row_at_window_start + get_local_id(0)] : size - 1;

    barrier(CLK_LOCAL_MEM_FENCE);

    if (i < nnz)
    {
      unsigned int row_index_inc = 0;
      while (i >= row_index_lookahead[row_index_inc + 1])
        ++row_index_inc;
      row_index = row_at_window_start + row_index_inc;
      row_index_buffer[get_local_id(0)] = row_index;
    }
    else
    {
      row_index = size+1;
      row_index_buffer[get_local_id(0)] = size - 1;
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    row_at_window_start = row_index_buffer[0];
    row_at_window_end   = row_index_buffer[get_local_size(0) - 1];

    //forward elimination
    for (unsigned int row = row_at_window_start; row <= row_at_window_end; ++row)
    {
      float result_entry = vector[row];

      if ( (row_index == row) && (col_index > row) )
        vector[col_index] -= result_entry * matrix_entry;

      barrier(CLK_GLOBAL_MEM_FENCE);
    }

    row_at_window_start = row_at_window_end;
  }
}


