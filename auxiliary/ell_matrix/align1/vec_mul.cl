

__kernel void vec_mul(
    __global const unsigned int * coords,
    __global const float * elements,
    __global const float * vector,
    __global float * result,
    unsigned int row_num,
    unsigned int col_num,
    unsigned int internal_row_num,
    unsigned int items_per_row,
    unsigned int aligned_items_per_row
    )
{
    uint glb_id = get_global_id(0);
    uint glb_sz = get_global_size(0);

    for(uint row_id = glb_id; row_id < row_num; row_id += glb_sz)
    {
        float sum = 0;

        uint offset = row_id;
        for(uint item_id = 0; item_id < items_per_row; item_id++, offset += internal_row_num)
        {
            float val = elements[offset];


            if(val != 0.0f)
            {
                int col = coords[offset];
                sum += (vector[col] * val);
            }

        }

        result[row_id] = sum;
    }
}
