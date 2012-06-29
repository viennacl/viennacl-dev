// divide a vector by a scalar (to be removed...)
__kernel void fft_div_vec_scalar(__global float2* input1, unsigned int size, float factor) {
    for (unsigned int i = get_global_id(0); i < size; i += get_global_size(0)) {
        input1[i] /= factor;
    }
}

