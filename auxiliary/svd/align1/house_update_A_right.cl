void col_reduce_lcl_array(__local float* sums, uint lcl_id, uint lcl_sz);

__kernel void house_update_A_right(
                        __global float* A,
                        __global float* V, // householder vector
                        uint row_start,
                        uint col_start,
                        uint size1,
                        uint size2,
                        uint stride,
                        __local float* sums
                        ) {

    uint glb_id = get_global_id(0);

    uint grp_id = get_group_id(0);
    uint grp_nm = get_num_groups(0);

    uint lcl_id = get_local_id(0);
    uint lcl_sz = get_local_size(0);

    float ss = 0;

    // update of A matrix
    for(uint i = grp_id + row_start; i < size1; i += grp_nm) {
        ss = 0;

        for(uint j = lcl_id; j < size2; j += lcl_sz) ss = ss + (V[j] * A[i * stride + j]);
        sums[lcl_id] = ss;

        barrier(CLK_LOCAL_MEM_FENCE);
        col_reduce_lcl_array(sums, lcl_id, lcl_sz);
        barrier(CLK_LOCAL_MEM_FENCE);

        float sum_Av = sums[0];

        for(uint j = lcl_id; j < size2; j += lcl_sz)
            A[i * stride + j] = A[i * stride + j] - (2 * V[j] * sum_Av);
    }
}

