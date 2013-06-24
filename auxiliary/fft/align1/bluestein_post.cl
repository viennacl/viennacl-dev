// Postprocessing phase of Bluestein algorithm
__kernel void bluestein_post(__global float2* Z,
                             __global float2* out,
                             unsigned int size)
{
    unsigned int glb_id = get_global_id(0);
    unsigned int glb_sz = get_global_size(0);

    unsigned int double_size = size << 1;
    float sn_a, cs_a;
    const float NUM_PI = 3.14159265358979323846;

    for(unsigned int i = glb_id; i < size; i += glb_sz) {
        unsigned int rm = i * i % (double_size);
        float angle = (float)rm / size * (-NUM_PI);

        sn_a = sincos(angle, &cs_a);

        float2 b_i = (float2)(cs_a, sn_a);
        out[i] = (float2)(Z[i].x * b_i.x - Z[i].y * b_i.y, Z[i].x * b_i.y + Z[i].y * b_i.x);
    }
}

