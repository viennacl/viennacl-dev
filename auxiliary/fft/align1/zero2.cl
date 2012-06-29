// Zero two complex vectors (to avoid kernel launch overhead)
__kernel void zero2(__global float2* input1,
                    __global float2* input2,
                    unsigned int size) {
    for (unsigned int i = get_global_id(0); i < size; i += get_global_size(0)) {
        input1[i] = 0;
        input2[i] = 0;
    }

}

