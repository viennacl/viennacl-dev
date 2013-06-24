
__kernel void diag_precond(
          __global const float * diag_A_inv,
          unsigned int start1,
          unsigned int inc1,
          unsigned int size1,
          __global float * x,
          unsigned int start2,
          unsigned int inc2,
          unsigned int size2)
{
  for (unsigned int i = get_global_id(0); i < size1; i += get_global_size(0))
    x[i*inc2+start2] *= diag_A_inv[i*inc1+start1];
}
