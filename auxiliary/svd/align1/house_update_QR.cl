void col_reduce_lcl_array(__local float* sums, uint lcl_id, uint lcl_sz);

__kernel void house_update_QR(
                        __global float* QR,
                        __global float* V, // householder vector
                        uint size1,
                        uint size2,
                        uint strideQ,
                        __local float* sums
                        ) {

    uint glb_id = get_global_id(0);

    uint grp_id = get_group_id(0);
    uint grp_nm = get_num_groups(0);

    uint lcl_id = get_local_id(0);
    uint lcl_sz = get_local_size(0);

    float ss = 0;

    // update of QR matrix
    // Actually, we are calculating a transpose of right matrix. This allows to avoid cache
    // misses.
    for(uint i = grp_id; i < size2; i += grp_nm) {
        ss = 0;
        for(uint j = lcl_id; j < size2; j += lcl_sz) ss = ss + (V[j] * QR[i * strideQ + j]);
        sums[lcl_id] = ss;

        barrier(CLK_LOCAL_MEM_FENCE);
        col_reduce_lcl_array(sums, lcl_id, lcl_sz);
        barrier(CLK_LOCAL_MEM_FENCE);

        float sum_Qv = sums[0];
        for(uint j = lcl_id; j < size2; j += lcl_sz)
            QR[i * strideQ + j] = QR[i * strideQ + j] - (2 * V[j] * sum_Qv);
    }
}
