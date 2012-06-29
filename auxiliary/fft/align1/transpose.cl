// simplistic matrix transpose function
__kernel void transpose(__global float2* input,
                        __global float2* output,
                        unsigned int row_num,
                        unsigned int col_num) {
    unsigned int size = row_num * col_num;
    for(unsigned int i = get_global_id(0); i < size; i+= get_global_size(0)) {
        unsigned int row = i / col_num;
        unsigned int col = i - row*col_num;

        unsigned int new_pos = col * row_num + row;

        output[new_pos] = input[i];
    }
}

