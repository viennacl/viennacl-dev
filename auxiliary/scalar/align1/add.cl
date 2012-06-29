 
__kernel void add(
          __global const float * val1,
          __global const float * val2, 
          __global float * result) 
{ 
  if (get_global_id(0) == 0)
    *result = *val1 + *val2;
}
 
