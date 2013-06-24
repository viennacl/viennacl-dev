/*
* Performs reordering of input data in bit-reversal order
* Probably it's better to do in host side,
*/
unsigned int get_reorder_num_2(unsigned int v, unsigned int bit_size) {
    v = ((v >> 1) & 0x55555555) | ((v & 0x55555555) << 1);
    v = ((v >> 2) & 0x33333333) | ((v & 0x33333333) << 2);
    v = ((v >> 4) & 0x0F0F0F0F) | ((v & 0x0F0F0F0F) << 4);
    v = ((v >> 8) & 0x00FF00FF) | ((v & 0x00FF00FF) << 8);
    v = (v >> 16) | (v << 16);

    v = v >> (32 - bit_size);

    return v;
}

__kernel void fft_reorder(__global float2* input,
                          unsigned int bit_size,
                          unsigned int size,
                          unsigned int stride,
                          int batch_num) {
    //unsigned int base_offset = 0;

    unsigned int glb_id = get_global_id(0);
    unsigned int glb_sz = get_global_size(0);

    for(unsigned int batch_id = 0; batch_id < batch_num; batch_id++) {
        for(unsigned int i = glb_id; i < size; i += glb_sz) {
            unsigned int v = get_reorder_num_2(i, bit_size);

            if(i < v) {
                float2 tmp = input[batch_id * stride + i]; // index
                input[batch_id * stride + i] = input[batch_id * stride + v]; //index
                input[batch_id * stride + v] = tmp; //index
            }
        }

        //base_offset += stride;
    }
}


