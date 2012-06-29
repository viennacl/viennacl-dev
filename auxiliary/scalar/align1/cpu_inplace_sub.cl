 
__kernel void cpu_inplace_sub(
          __global float * val1,
          float val2) 
{ 
  if (get_global_id(0) == 0)
    *val1 -= val2;
}
 

