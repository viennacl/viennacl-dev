
__kernel void assign_cpu(
          __global float * vec1,
          unsigned int start1,
          unsigned int inc1,
          unsigned int size1,
          unsigned int internal_size1,
          float alpha)
{

  for (unsigned int i = get_global_id(0); i < internal_size1; i += get_global_size(0))
    vec1[i*inc1+start1] = (i < size1) ? alpha : 0;
}

