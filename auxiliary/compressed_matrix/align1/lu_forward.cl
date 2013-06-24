



// compute y in Ly = z for incomplete LU factorizations of a sparse matrix in compressed format
__kernel void lu_forward(
          __global const unsigned int * row_indices,
          __global const unsigned int * column_indices,
          __global const float * elements,
          __global float * vector,
          unsigned int size)
{
  __local unsigned int col_index_buffer[128];
  __local float element_buffer[128];
  __local float vector_buffer[128];

  unsigned int nnz = row_indices[size];
  unsigned int current_row = 0;
  unsigned int row_at_window_start = 0;
  float current_vector_entry = vector[0];
  float diagonal_entry;
  unsigned int loop_end = (nnz / get_local_size(0) + 1) * get_local_size(0);
  unsigned int next_row = row_indices[1];

  for (unsigned int i = get_local_id(0); i < loop_end; i += get_local_size(0))
  {
    //load into shared memory (coalesced access):
    if (i < nnz)
    {
      element_buffer[get_local_id(0)] = elements[i];
      unsigned int tmp = column_indices[i];
      col_index_buffer[get_local_id(0)] = tmp;
      vector_buffer[get_local_id(0)] = vector[tmp];
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    //now a single thread does the remaining work in shared memory:
    if (get_local_id(0) == 0)
    {
      // traverse through all the loaded data:
      for (unsigned int k=0; k<get_local_size(0); ++k)
      {
        if (current_row < size && i+k == next_row) //current row is finished. Write back result
        {
          vector[current_row] = current_vector_entry / diagonal_entry;
          ++current_row;
          if (current_row < size) //load next row's data
          {
            next_row = row_indices[current_row+1];
            current_vector_entry = vector[current_row];
          }
        }

        if (current_row < size && col_index_buffer[k] < current_row) //substitute
        {
          if (col_index_buffer[k] < row_at_window_start) //use recently computed results
            current_vector_entry -= element_buffer[k] * vector_buffer[k];
          else if (col_index_buffer[k] < current_row) //use buffered data
            current_vector_entry -= element_buffer[k] * vector[col_index_buffer[k]];
        }
        else if (col_index_buffer[k] == current_row)
          diagonal_entry = element_buffer[k];

      } // for k

      row_at_window_start = current_row;
    } // if (get_local_id(0) == 0)

    barrier(CLK_GLOBAL_MEM_FENCE);
  } //for i
}


