
// calculates a sum of local array elements
void col_reduce_lcl_array(__local float* sums, uint lcl_id, uint lcl_sz) {
    uint step = lcl_sz >> 1;

    while(step > 0) {
        if(lcl_id < step) {
            sums[lcl_id] += sums[lcl_id + step];
        }
        step >>= 1;
        barrier(CLK_LOCAL_MEM_FENCE);
    }
}

__kernel void house_col(__global float* A,
                        __global float* QL,
                        __constant float* V, //householder vector
                        uint row_start,
                        uint col_start,
                        uint size1,
                        uint size2,
                        uint stride,
                        uint strideQ,
                        __local float* sums
                        ) {
    uint glb_id = get_global_id(0);
    uint glb_sz = get_global_size(0);

    uint grp_id = get_group_id(0);
    uint grp_nm = get_num_groups(0);

    uint lcl_id = get_local_id(0);
    uint lcl_sz = get_local_size(0);

    float ss = 0.0f;
    // update of left matrix
    for(uint i = grp_id; i < size1; i += grp_nm) {
        ss = 0.0f;
        for(uint j = lcl_id; j < size1; j += lcl_sz) ss = ss + (V[j] * QL[i * strideQ + j]);
        sums[lcl_id] = ss;

        barrier(CLK_LOCAL_MEM_FENCE);
        col_reduce_lcl_array(sums, lcl_id, lcl_sz);
        barrier(CLK_LOCAL_MEM_FENCE);

        float sum_Qv = sums[0];

        for(uint j = lcl_id; j < size1; j += lcl_sz)
            QL[i * strideQ + j] = QL[i * strideQ + j] - (2 * V[j] * sum_Qv);
    }
    // doing it in slightly different way to avoid cache misses
    for(uint i = glb_id + col_start; i < size2; i += glb_sz) {
        ss = 0.0f;
        for(uint j = row_start; j < size1; j++) ss = ss + (V[j] * A[j * stride + i]);

        for(uint j = row_start; j < size1; j++)
            A[j * stride + i] = A[j * stride + i] - (2 * V[j] * ss);
    }
}
