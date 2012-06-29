 
__kernel void inplace_add(
          __global float * val1,
          __global const float * val2) 
{ 
  if (get_global_id(0) == 0)
    *val1 += *val2;
}
 
