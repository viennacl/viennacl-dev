 
// note: 'div' seems to produce some name clashes with the OpenCL jit-compiler, thus using 'divide'
__kernel void divide(
          __global const float * val1,
          __global const float * val2, 
          __global float * result) 
{ 
  if (get_global_id(0) == 0)
    *result = *val1 / *val2;
}

 
