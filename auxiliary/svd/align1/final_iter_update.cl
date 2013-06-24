__kernel void final_iter_update(__global float* A,
                                uint stride,
                                uint n,
                                uint last_n,
                                float q,
                                float p
                                )
{
    uint glb_id = get_global_id(0);
    uint glb_sz = get_global_size(0);

    for (uint px = glb_id; px < last_n; px += glb_sz)
    {
        float v_in = A[n * stride + px];
        float z = A[(n - 1) * stride + px];
        A[(n - 1) * stride + px] = q * z + p * v_in;
        A[n * stride + px] = q * v_in - p * z;
    }
}
