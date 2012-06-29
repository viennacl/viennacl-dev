__kernel void complex_to_real(__global float2* in,
                              __global float* out,
                              unsigned int size) {
    for (unsigned int i = get_global_id(0); i < size; i += get_global_size(0)) {
        out[i] = in[i].x;
    }
}

