
//segmented parallel reduction. At present restricted to up to 256 threads
void segmented_parallel_reduction(unsigned int row, 
                                  float val, 
                                  __local unsigned int * shared_rows, 
                                  __local float * inter_results) 
{ 
  //barrier(CLK_LOCAL_MEM_FENCE); 
  shared_rows[get_local_id(0)] = row; 
  inter_results[get_local_id(0)] = val; 
  float left = 0;
 
  barrier(CLK_LOCAL_MEM_FENCE); 
  if( get_local_id(0) >=  1 && row == shared_rows[get_local_id(0) -  1] ) { left = inter_results[get_local_id(0) -  1]; }  
  barrier(CLK_LOCAL_MEM_FENCE); 
  inter_results[get_local_id(0)] += left; left = 0;
  barrier(CLK_LOCAL_MEM_FENCE); 

  if( get_local_id(0) >=  2 && row == shared_rows[get_local_id(0) -  2] ) { left = inter_results[get_local_id(0) -  2]; } 
  barrier(CLK_LOCAL_MEM_FENCE); 
  inter_results[get_local_id(0)] += left; left = 0;
  barrier(CLK_LOCAL_MEM_FENCE); 

  if( get_local_id(0) >=  4 && row == shared_rows[get_local_id(0) -  4] ) { left = inter_results[get_local_id(0) -  4]; } 
  barrier(CLK_LOCAL_MEM_FENCE); 
  inter_results[get_local_id(0)] += left; left = 0;
  barrier(CLK_LOCAL_MEM_FENCE); 

  if( get_local_id(0) >=  8 && row == shared_rows[get_local_id(0) -  8] ) { left = inter_results[get_local_id(0) -  8]; } 
  barrier(CLK_LOCAL_MEM_FENCE); 
  inter_results[get_local_id(0)] += left; left = 0;
  barrier(CLK_LOCAL_MEM_FENCE); 

  if( get_local_id(0) >= 16 && row == shared_rows[get_local_id(0) - 16] ) { left = inter_results[get_local_id(0) - 16]; } 
  barrier(CLK_LOCAL_MEM_FENCE); 
  inter_results[get_local_id(0)] += left; left = 0;
  barrier(CLK_LOCAL_MEM_FENCE); 

  if( get_local_id(0) >= 32 && row == shared_rows[get_local_id(0) - 32] ) { left = inter_results[get_local_id(0) - 32]; } 
  barrier(CLK_LOCAL_MEM_FENCE); 
  inter_results[get_local_id(0)] += left; left = 0;
  barrier(CLK_LOCAL_MEM_FENCE); 

  if( get_local_id(0) >= 64 && row == shared_rows[get_local_id(0) - 64] ) { left = inter_results[get_local_id(0) - 64]; } 
  barrier(CLK_LOCAL_MEM_FENCE); 
  inter_results[get_local_id(0)] += left; left = 0;
  barrier(CLK_LOCAL_MEM_FENCE); 

  if( get_local_id(0) >= 128 && row == shared_rows[get_local_id(0) - 128] ) { left = inter_results[get_local_id(0) - 128]; } 
  barrier(CLK_LOCAL_MEM_FENCE); 
  inter_results[get_local_id(0)] += left; left = 0;
  barrier(CLK_LOCAL_MEM_FENCE); 

  //if( get_local_id(0) >= 256 && row == shared_rows[get_local_id(0) - 256] ) { left = inter_results[get_local_id(0) - 256]; } 
  //barrier(CLK_LOCAL_MEM_FENCE);  
  //inter_results[get_local_id(0)] += left; left = 0;
  //barrier(CLK_LOCAL_MEM_FENCE); 
}


__kernel void vec_mul( 
          __global const uint2 * coords, //(row_index, column_index) 
          __global const float * elements, 
          __global const uint  * group_boundaries,
          __global const float * vector,  
          __global float * result, 
          __local unsigned int * shared_rows, 
          __local float * inter_results) 
{ 
  uint2 tmp; 
  float val;
  uint last_index = get_local_size(0) - 1;
  uint group_start = group_boundaries[get_group_id(0)];
  uint group_end = group_boundaries[get_group_id(0) + 1];
  uint k_end = 1 + (group_end - group_start - 1) / get_local_size(0);   // -1 in order to have correct behavior if group_end - group_start == j * get_local_size(0)

  uint local_index = 0;

  for (uint k = 0; k < k_end; ++k)
  { 
    barrier(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE); 
    
    local_index = group_start + k * get_local_size(0) + get_local_id(0); 
  
    if (local_index < group_end)
    {
      tmp = coords[local_index]; 
      val = elements[local_index] * vector[tmp.y]; 
    }
    else
    {
      tmp.x = 0;
      tmp.y = 0;
      val = 0;
    }

    barrier(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE); 

    //check for carry from previous loop run: 
    if (get_local_id(0) == 0 && k > 0)
    { 
      if (tmp.x == shared_rows[last_index]) 
        val += inter_results[last_index]; 
      else 
        result[shared_rows[last_index]] += inter_results[last_index]; 
    } 

    barrier(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE); 

    segmented_parallel_reduction(tmp.x, val, shared_rows, inter_results); //all threads have to enter this function

    barrier(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE); 

    if (get_local_id(0) != last_index &&
        shared_rows[get_local_id(0)] != shared_rows[get_local_id(0) + 1] &&
        inter_results[get_local_id(0)] != 0) 
    { 
      result[tmp.x] += inter_results[get_local_id(0)]; 
    }
   
    barrier(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE); 
  } //for k
   
  if (get_local_id(0) == last_index && inter_results[last_index] != 0) 
    result[tmp.x] += inter_results[last_index]; 
}