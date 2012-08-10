void col_reduce_lcl_array(__local float* sums, uint lcl_id, uint lcl_sz);

__kernel void house_update_A_left(
                        __global float* A,
                        __constant float* V, //householder vector
                        uint row_start,
                        uint col_start,
                        uint size1,
                        uint size2,
                        uint stride,
                        __local float* sums
                        ) {
    uint glb_id = get_global_id(0);
    uint glb_sz = get_global_size(0);

    uint grp_id = get_group_id(0);
    uint grp_nm = get_num_groups(0);

    uint lcl_id = get_local_id(0);
    uint lcl_sz = get_local_size(0);

    float ss = 0;

    // doing it in slightly different way to avoid cache misses
    for(uint i = glb_id + col_start; i < size2; i += glb_sz) {
        ss = 0;
        for(uint j = row_start; j < size1; j++) ss = ss + (V[j] * A[j * stride + i]);

        for(uint j = row_start; j < size1; j++)
            A[j * stride + i] = A[j * stride + i] - (2 * V[j] * ss);
    }
}
