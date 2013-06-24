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
