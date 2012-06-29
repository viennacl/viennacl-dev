// embedd a real-valued vector into a complex one
__kernel void real_to_complex(__global float* in,
                              __global float2* out,
                              unsigned int size) {
    for (unsigned int i = get_global_id(0); i < size; i += get_global_size(0)) {
        float2 val = 0;
        val.x = in[i];
        out[i] = val;
    }
}

