
// helper kernel for norm_2
__kernel void sqrt_sum(
          __global float * vec1,
          unsigned int start1,  
          unsigned int inc1,
          unsigned int size1,
          __global float * result) 
{ 
  //parallel reduction on global memory: (make sure get_global_size(0) is a power of 2)
  for (unsigned int stride = get_global_size(0)/2; stride > 0; stride /= 2)
  {
    if (get_global_id(0) < stride)
      vec1[get_global_id(0)*inc1+start1] += vec1[(get_global_id(0)+stride)*inc1+start1];
    barrier(CLK_GLOBAL_MEM_FENCE);
  }
  
  if (get_global_id(0) == 0)
    *result = sqrt(vec1[start1]);
  
}

