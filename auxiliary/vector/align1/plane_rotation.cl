
////// plane rotation: (x,y) <- (\alpha x + \beta y, -\beta x + \alpha y)
__kernel void plane_rotation(
          __global float * vec1,
          unsigned int start1,
          unsigned int inc1,
          unsigned int size1,
          __global float * vec2,
          unsigned int start2,
          unsigned int inc2,
          unsigned int size2,
          float alpha,
          float beta)
{
  float tmp1 = 0;
  float tmp2 = 0;

  for (unsigned int i = get_global_id(0); i < size1; i += get_global_size(0))
  {
    tmp1 = vec1[i*inc1+start1];
    tmp2 = vec2[i*inc2+start2];

    vec1[i*inc1+start1] = alpha * tmp1 + beta * tmp2;
    vec2[i*inc2+start2] = alpha * tmp2 - beta * tmp1;
  }

}

