

// compute x in Ux = y for incomplete LU factorizations of a sparse matrix in compressed format
__kernel void lu_backward(
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
  unsigned int current_row = size-1;
  unsigned int row_at_window_start = size-1;
  float current_vector_entry = vector[size-1];
  float diagonal_entry = 0;
  unsigned int loop_end = ( (nnz - 1) / get_local_size(0)) * get_local_size(0);
  unsigned int next_row = row_indices[size-1];

  unsigned int i = loop_end + get_local_id(0);
  while (1)
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
      // traverse through all the loaded data from back to front:
      for (unsigned int k2=0; k2<get_local_size(0); ++k2)
      {
        unsigned int k = (get_local_size(0) - k2) - 1;

        if (i+k >= nnz)
          continue;

        if (col_index_buffer[k] > row_at_window_start) //use recently computed results
          current_vector_entry -= element_buffer[k] * vector_buffer[k];
        else if (col_index_buffer[k] > current_row) //use buffered data
          current_vector_entry -= element_buffer[k] * vector[col_index_buffer[k]];
        else if (col_index_buffer[k] == current_row)
          diagonal_entry = element_buffer[k];

        if (i+k == next_row) //current row is finished. Write back result
        {
          vector[current_row] = current_vector_entry / diagonal_entry;
          if (current_row > 0) //load next row's data
          {
            --current_row;
            next_row = row_indices[current_row];
            current_vector_entry = vector[current_row];
          }
        }


      } // for k

      row_at_window_start = current_row;
    } // if (get_local_id(0) == 0)

    barrier(CLK_GLOBAL_MEM_FENCE);

    if (i < get_local_size(0))
      break;

    i -= get_local_size(0);
  } //for i
}

