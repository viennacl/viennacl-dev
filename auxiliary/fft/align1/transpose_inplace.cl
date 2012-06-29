// inplace-transpose of a matrix
__kernel void transpose_inplace(__global float2* input,
                        unsigned int row_num,
                        unsigned int col_num) {
    unsigned int size = row_num * col_num;
    for(unsigned int i = get_global_id(0); i < size; i+= get_global_size(0)) {
        unsigned int row = i / col_num;
        unsigned int col = i - row*col_num;

        unsigned int new_pos = col * row_num + row;

        //new_pos = col < row?0:1;
        //input[i] = new_pos;

        if(i < new_pos) {
            float2 val = input[i];
            input[i] = input[new_pos];
            input[new_pos] = val;
        }
    }
}

