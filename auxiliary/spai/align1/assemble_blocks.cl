
float get_element(__global const unsigned int * row_indices,
           __global const unsigned int * column_indices,
           __global const float * elements,
           unsigned int row,
           unsigned int col
           )
{
  unsigned int row_end = row_indices[row+1];
  for(unsigned int i = row_indices[row]; i < row_end; ++i){
    if(column_indices[i] == col)
      return elements[i];
    if(column_indices[i] > col)
      return 0.0;
  }
  return 0.0;
}

void block_assembly(__global const unsigned int * row_indices,
          __global const unsigned int * column_indices,
          __global const float * elements,
          __global const unsigned int * matrix_dimensions,
          __global const unsigned int * set_I,
          __global const unsigned int * set_J,
          unsigned int matrix_ind,
          __global float * com_A_I_J)
{
  unsigned int row_n = matrix_dimensions[2*matrix_ind];
  unsigned int col_n = matrix_dimensions[2*matrix_ind + 1];

  for(unsigned int i = 0; i < col_n; ++i){
        //start row index
        for(unsigned int j = 0; j < row_n; j++){
          com_A_I_J[ i*row_n + j] = get_element(row_indices, column_indices, elements, set_I[j], set_J[i]);
        }
      }

}

__kernel void assemble_blocks(
          __global const unsigned int * row_indices,
          __global const unsigned int * column_indices,
          __global const float * elements,
          __global const unsigned int * set_I,
        __global const unsigned int * set_J,
      __global const unsigned int * i_ind,
      __global const unsigned int * j_ind,
        __global const unsigned int * block_ind,
        __global const unsigned int * matrix_dimensions,
      __global float * com_A_I_J,
      __global unsigned int * g_is_update,
                   unsigned int  block_elems_num)
{
    for(unsigned int i  = get_global_id(0); i < block_elems_num; i += get_global_size(0)){
        if((matrix_dimensions[2*i] > 0) && (matrix_dimensions[2*i + 1] > 0) && g_is_update[i] > 0){

            block_assembly(row_indices, column_indices, elements, matrix_dimensions, set_I + i_ind[i], set_J + j_ind[i], i, com_A_I_J + block_ind[i]);
        }
    }
}
