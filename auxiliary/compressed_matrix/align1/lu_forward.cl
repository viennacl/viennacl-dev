


 
// compute y in Ly = z for incomplete LU factorizations of a sparse matrix in compressed format
__kernel void lu_forward(
          __global const unsigned int * row_indices,
          __global const unsigned int * column_indices, 
          __global const float * elements,
          __local  int * buffer,                              
          __local  float * vec_entries,   //a memory block from vector
          __global float * vector,
          unsigned int size) 
{
  int waiting_for; //block index that must be finished before the current thread can start
  unsigned int waiting_for_index;
  int block_offset;
  unsigned int col;
  unsigned int row;
  unsigned int row_index_end;
  
  //backward substitution: one thread per row in blocks of get_global_size(0)
  for (unsigned int block_num = 0; block_num <= size / get_global_size(0); ++block_num)
  {
    block_offset = block_num * get_global_size(0);
    row = block_offset + get_global_id(0);
    buffer[get_global_id(0)] = 0; //set flag to 'undone'
    waiting_for = -1;

    if (row < size)
    {
      vec_entries[get_global_id(0)] = vector[row];
      waiting_for_index = row_indices[row];
      row_index_end = row_indices[row+1];
    }
    
    if (get_global_id(0) == 0)
      buffer[get_global_size(0)] = 1;


    //try to eliminate all lines in the block. 
    //in worst case scenarios, in each step only one line can be substituted, thus loop
    for (unsigned int k = 0; k<get_global_size(0); ++k)
    {
      barrier(CLK_LOCAL_MEM_FENCE);
      if (row < size) //valid index?
      {
        if (waiting_for >= 0)
        {
          if (buffer[waiting_for] == 1)
            waiting_for = -1;
        }
        
        if (waiting_for == -1) //substitution not yet done, check whether possible
        {
          //check whether reduction is possible:
          for (unsigned int j = waiting_for_index; j < row_index_end; ++j)
          {
            col = column_indices[j];
            if (col < block_offset) //index valid, but not from current block
              vec_entries[get_global_id(0)] -= elements[j] * vector[col];
            else if (col < row)  //index is from current block
            {
              if (buffer[col - block_offset] == 0) //entry is not yet calculated
              {
                waiting_for = col - block_offset;
                waiting_for_index = j;
                break;
              }
              else  //updated entry is available in shared memory:
                vec_entries[get_global_id(0)] -= elements[j] * vec_entries[col - block_offset];
            }
          }
          
          if (waiting_for == -1)  //this row is done
          {
            buffer[get_global_id(0)] = 1;
            waiting_for = -2; //magic number: thread is finished
          }
        } 
      } //row < size
      else
        buffer[get_global_id(0)] = 1; //work done (because there is no work to be done at all...)

      ///////// check whether all threads are done. If yes, exit loop /////////////
      
      if (buffer[get_global_id(0)] == 0)
        buffer[get_global_size(0)] = 0;
      barrier(CLK_LOCAL_MEM_FENCE);
      
      if (buffer[get_global_size(0)] > 0)  //all threads break this loop simultaneously
        break;

      if (get_global_id(0) == 0)
        buffer[get_global_size(0)] = 1;

    } //for k
    
    //write to vector:
    if (row < size)
      vector[row] = vec_entries[get_global_id(0)];
    
    barrier(CLK_GLOBAL_MEM_FENCE);
  } //for block_num
}


