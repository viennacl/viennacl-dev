
__kernel void level_scheduling_substitute(
          __global const unsigned int * row_index_array,
          __global const unsigned int * row_indices,
          __global const unsigned int * column_indices,
          __global const float * elements,
          __global float * vec,
          unsigned int size)
{
  for (unsigned int row  = get_global_id(0);
                    row  < size;
                    row += get_global_size(0))
  {
    unsigned int eq_row = row_index_array[row];
    float vec_entry = vec[eq_row];
    unsigned int row_end = row_indices[row+1];

    for (unsigned int j = row_indices[row]; j < row_end; ++j)
      vec_entry -= vec[column_indices[j]] * elements[j];

    vec[eq_row] = vec_entry;
  }
}

