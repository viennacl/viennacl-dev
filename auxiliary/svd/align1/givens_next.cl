__kernel void givens_next(__global float* matr,
                            __global float* cs,
                            __global float* ss,
                            uint size,
                            uint stride,
                            uint start_i,
                            uint end_i
                            )
{
    uint glb_id = get_global_id(0);
    uint glb_sz = get_global_size(0);

    uint lcl_id = get_local_id(0);
    uint lcl_sz = get_local_size(0);

    uint j = glb_id;

    __local float cs_lcl[256];
    __local float ss_lcl[256];

    float x = (j < size) ? matr[(end_i + 1) * stride + j] : 0;

    uint elems_num = end_i - start_i + 1;
    uint block_num = (elems_num + lcl_sz - 1) / lcl_sz;

    for(uint block_id = 0; block_id < block_num; block_id++)
    {
        uint to = min(elems_num - block_id * lcl_sz, lcl_sz);

        if(lcl_id < to)
        {
            cs_lcl[lcl_id] = cs[end_i - (lcl_id + block_id * lcl_sz)];
            ss_lcl[lcl_id] = ss[end_i - (lcl_id + block_id * lcl_sz)];;
        }

        barrier(CLK_LOCAL_MEM_FENCE);

        if(j < size)
        {
            for(uint ind = 0; ind < to; ind++)
            {
                uint i = end_i - (ind + block_id * lcl_sz);

                float z = matr[i * stride + j];

                float cs_val = cs_lcl[ind];
                float ss_val = ss_lcl[ind];

                matr[(i + 1) * stride + j] = x * cs_val + z * ss_val;
                x = -x * ss_val + z * cs_val;
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if(j < size)
        matr[(start_i) * stride + j] = x;
}
