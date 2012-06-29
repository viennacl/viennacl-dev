// elementwise product of two complex vectors
__kernel void fft_mult_vec(__global const float2* input1,
                          __global const float2* input2,
                          __global float2* output,
                          unsigned int size) {
    for (unsigned int i = get_global_id(0); i < size; i += get_global_size(0)) {
        float2 in1 = input1[i];
        float2 in2 = input2[i];

        output[i] = (float2)(in1.x * in2.x - in1.y * in2.y, in1.x * in2.y + in1.y * in2.x);
    }
}

