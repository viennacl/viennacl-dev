
__kernel void vec_mul(
    const __global int* ell_coords,
    const __global float* ell_elements,
    const __global uint* csr_rows,
    const __global uint* csr_cols,
    const __global float* csr_elements,
    const __global float * vector,

    __global float * result,

    unsigned int row_num,
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
            float val = ell_elements[offset];


            if(val != 0.0f)
            {
                int col = ell_coords[offset];
                sum += (vector[col] * val);
            }

        }

        uint col_begin = csr_rows[row_id];
        uint col_end   = csr_rows[row_id + 1];

        for(uint item_id = col_begin; item_id < col_end; item_id++)
        {
            sum += (vector[csr_cols[item_id]] * csr_elements[item_id]);
        }

        result[row_id] = sum;
    }
}
