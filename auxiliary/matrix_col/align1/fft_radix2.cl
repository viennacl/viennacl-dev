__kernel void fft_radix2(__global float2* input,
                         unsigned int s,
                         unsigned int bit_size,
                         unsigned int size,
                         unsigned int stride,
                         unsigned int batch_num,
                         float sign) {

    unsigned int ss = 1 << s;
    unsigned int half_size = size >> 1;

    float cs, sn;
    const float NUM_PI = 3.14159265358979323846;

    unsigned int glb_id = get_global_id(0);
    unsigned int glb_sz = get_global_size(0);

    for(unsigned int batch_id = 0; batch_id < batch_num; batch_id++) {
        for(unsigned int tid = glb_id; tid < half_size; tid += glb_sz) {
            unsigned int group = (tid & (ss - 1));
            unsigned int pos = ((tid >> s) << (s + 1)) + group;

            unsigned int offset = pos * stride + batch_id;
            float2 in1 = input[offset];//index
            float2 in2 = input[offset + ss * stride];//index

            float arg = group * sign * NUM_PI / ss;

            sn = sincos(arg, &cs);
            float2 ex = (float2)(cs, sn);

            float2 tmp = (float2)(in2.x * ex.x - in2.y * ex.y, in2.x * ex.y + in2.y * ex.x);

            input[offset + ss * stride] = in1 - tmp;//index
            input[offset] = in1 + tmp;//index
        }
    }
}

