
 __kernel void block_trans_unit_lu_forward(
           __global const unsigned int * row_jumper_L,      //L part (note that L is transposed in memory)
           __global const unsigned int * column_indices_L,
           __global const float * elements_L,
           __global const unsigned int * block_offsets,
           __global float * result,
           unsigned int size)
 {
   unsigned int col_start = block_offsets[2*get_group_id(0)];
   unsigned int col_stop  = block_offsets[2*get_group_id(0)+1];
   unsigned int row_start = row_jumper_L[col_start];
   unsigned int row_stop;
   float result_entry = 0;

   if (col_start >= col_stop)
     return;

   //forward elimination, using L:
   for (unsigned int col = col_start; col < col_stop; ++col)
   {
     result_entry = result[col];
     row_stop = row_jumper_L[col + 1];
     for (unsigned int buffer_index = row_start + get_local_id(0); buffer_index < row_stop; buffer_index += get_local_size(0))
       result[column_indices_L[buffer_index]] -= result_entry * elements_L[buffer_index];
     row_start = row_stop; //for next iteration (avoid unnecessary loads from GPU RAM)
     barrier(CLK_GLOBAL_MEM_FENCE);
   }

 };

