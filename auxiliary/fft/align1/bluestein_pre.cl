// Preprocessing phase of Bluestein algorithm
__kernel void bluestein_pre(__global float2* input,
                            __global float2* A,
                            __global float2* B,
                            unsigned int size,
                            unsigned int ext_size
                           ) {
    unsigned int glb_id = get_global_id(0);
    unsigned int glb_sz = get_global_size(0);

    unsigned int double_size = size << 1;

    float sn_a, cs_a;
    const float NUM_PI = 3.14159265358979323846;

    for(unsigned int i = glb_id; i < size; i += glb_sz) {
        unsigned int rm = i * i % (double_size);
        float angle = (float)rm / size * NUM_PI;

        sn_a = sincos(-angle, &cs_a);

        float2 a_i = (float2)(cs_a, sn_a);
        float2 b_i = (float2)(cs_a, -sn_a);

        A[i] = (float2)(input[i].x * a_i.x - input[i].y * a_i.y, input[i].x * a_i.y + input[i].y * a_i.x);

        B[i] = b_i;

        // very bad instruction, to be fixed
        if(i)
          B[ext_size - i] = b_i;
    }
}

