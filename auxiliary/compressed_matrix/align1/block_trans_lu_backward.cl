
 __kernel void block_trans_lu_backward(
           __global const unsigned int * row_jumper_U,      //U part (note that U is transposed in memory)
           __global const unsigned int * column_indices_U,
           __global const float * elements_U,
           __global const float * diagonal_U,
           __global const unsigned int * block_offsets,
           __global float * result,
           unsigned int size)
 {
   unsigned int col_start = block_offsets[2*get_group_id(0)];
   unsigned int col_stop  = block_offsets[2*get_group_id(0)+1];
   unsigned int row_start;
   unsigned int row_stop;
   float result_entry = 0;

   if (col_start >= col_stop)
     return;

   //backward elimination, using U and diagonal_U
   for (unsigned int iter = 0; iter < col_stop - col_start; ++iter)
   {
     unsigned int col = (col_stop - iter) - 1;
     result_entry = result[col] / diagonal_U[col];
     row_start = row_jumper_U[col];
     row_stop  = row_jumper_U[col + 1];
     for (unsigned int buffer_index = row_start + get_local_id(0); buffer_index < row_stop; buffer_index += get_local_size(0))
       result[column_indices_U[buffer_index]] -= result_entry * elements_U[buffer_index];
     barrier(CLK_GLOBAL_MEM_FENCE);
   }

   //divide result vector by diagonal:
   for (unsigned int col = col_start + get_local_id(0); col < col_stop; col += get_local_size(0))
     result[col] /= diagonal_U[col];
 };

