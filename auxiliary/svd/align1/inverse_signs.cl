


__kernel void inverse_signs(__global float* v,
                            __global float* signs,
                            uint size,
                            uint stride
                            )
{
    uint glb_id_x = get_global_id(0);
    uint glb_id_y = get_global_id(1);

    if((glb_id_x < size) && (glb_id_y < size))
        v[glb_id_x * stride + glb_id_y] *= signs[glb_id_x];
}

