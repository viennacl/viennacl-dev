
 __kernel void block_ilu_substitute(
           __global const unsigned int * row_jumper_L,      //L part (note that L is transposed in memory)
           __global const unsigned int * column_indices_L, 
           __global const float * elements_L,
           __global const unsigned int * row_jumper_U,      //U part (note that U is transposed in memory)
           __global const unsigned int * column_indices_U,
           __global const float * elements_U,
           __global const float * elements_D,              //diagonal
           __global const unsigned int * block_offsets,
           __global float * result,
           unsigned int size)
 {
   unsigned int col_start = block_offsets[2*get_group_id(0)];
   unsigned int col_stop  = block_offsets[2*get_group_id(0)+1];
   unsigned int row_start = row_jumper_L[col_start];
   unsigned int row_stop;
   float result_entry = 0;

   if (col_start <= col_stop)
     return;

   //forward elimination, using L:
   for (unsigned int col = col_start; col < col_stop; ++col)
   {
     result_entry = result[col];
     row_stop = row_jumper_L[col + 1];
     for (unsigned int row_index = row_start + get_local_id(0); row_index < row_stop; ++row_index) 
       result[column_indices_L[row_index]] -= result_entry * elements_L[row_index]; 
     row_start = row_stop; //for next iteration (avoid unnecessary loads from GPU RAM)
     barrier(CLK_GLOBAL_MEM_FENCE);
   } 

   //backward elimination, using U and D: 
   for (unsigned int iter = 0; iter < col_stop - col_start; ++iter) 
   { 
     result_entry = result[col_stop - iter - 1] / elements_D[col_stop - iter - 1]; 
     row_start = row_jumper_U[col_stop - iter - 1]; 
     row_stop  = row_jumper_U[col_stop - iter]; 
     for (unsigned int row_index = row_start + get_local_id(0); row_index < row_stop; ++row_index) 
       result[column_indices_U[row_index]] -= result_entry * elements_U[row_index]; 
     barrier(CLK_GLOBAL_MEM_FENCE); 
     if (get_local_id(0) == 0) 
       result[col_stop - iter - 1] = result_entry; 
   } 

 };

