

__kernel void row_info_extractor(
          __global const unsigned int * row_indices,
          __global const unsigned int * column_indices,
          __global const float * elements,
          __global float * result,
          unsigned int size,
          unsigned int option
          )
{
  for (unsigned int row = get_global_id(0); row < size; row += get_global_size(0))
  {
    float value = 0;
    unsigned int row_end = row_indices[row+1];

    switch (option)
    {
      case 0: //inf-norm
        for (unsigned int i = row_indices[row]; i < row_end; ++i)
          value = max(value, fabs(elements[i]));
        break;

      case 1: //1-norm
        for (unsigned int i = row_indices[row]; i < row_end; ++i)
          value += fabs(elements[i]);
        break;

      case 2: //2-norm
        for (unsigned int i = row_indices[row]; i < row_end; ++i)
          value += elements[i] * elements[i];
        value = sqrt(value);
        break;

      case 3: //diagonal entry
        for (unsigned int i = row_indices[row]; i < row_end; ++i)
        {
          if (column_indices[i] == row)
          {
            value = elements[i];
            break;
          }
        }
        break;

      default:
        break;
    }
    result[row] = value;
  }
}


